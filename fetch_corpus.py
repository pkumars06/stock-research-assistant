"""Download the filings the capstone corpus is missing, from BSE.

Lists every announcement for the two stocks since START, keeps the document
types worth searching (results, transcripts, decks, annual reports, governance
filings), and downloads each PDF into corpus/<stock>/. Writes
corpus/downloaded.csv with one row per file.

Reads a local list of BSE scrip codes (the file named by BSE_SCRIP_MASTER); never writes there.
"""
import csv
import hashlib
import json
import os
import re
import time
from datetime import date, timedelta

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
SCRIP_MASTER = os.environ.get("BSE_SCRIP_MASTER", "bse_scrip_master.json")

STOCKS = {"Stylam": "INE239C01020", "Speciality": "INE247M01014"}
START = date(2025, 4, 1)

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json",
}
BASES = ["https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
         "https://www.bseindia.com/xml-data/corpfiling/AttachHis/"]

# (doc_type, pattern on the headline). First match wins; no match = skipped.
KEEP = [
    ("transcript", r"earnings call transcript"),
    ("annual_report", r"reg\. ?34 ?\(1\) annual report"),
    ("agm_notice", r"notice of .*annual general meeting"),
    ("investor_deck", r"investor presentation"),
    ("press_release", r"press release"),
    ("governance", r"change in management|change in directorate|amendments to memorandum"
                   r"|open offer|post-offer|letter of offer"),
    ("operations", r"commercial production|award_of_order|award of order"),
]


# Filings with generic headlines that the patterns above can't pick out, kept by
# attachment prefix. Stylam 26-Dec-2025: the Aica deal disclosures, filed with the open offer.
EXTRA = {"b53c8381": "governance", "4126d528": "governance", "aa274cda": "governance"}


def doc_type(headline, category, attach=""):
    if attach[:8] in EXTRA:
        return EXTRA[attach[:8]]
    h = headline.lower()
    if category == "Result":
        return "results"
    for kind, pat in KEEP:
        if re.search(pat, h):
            return kind
    return None


def list_filings(s, code, fr, to):
    u = ("https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
         f"?pageno=1&strCat=-1&strPrevDate={fr:%Y%m%d}&strScrip={code}"
         f"&strSearch=P&strToDate={to:%Y%m%d}&strType=C&subcategory=-1")
    return s.get(u, timeout=40).json().get("Table", []) or []


def download(s, attach, path):
    # BSE refuses some requests when they come quickly; the same URL works on a retry.
    for attempt in range(3):
        for base in BASES:
            try:
                r = s.get(base + attach, timeout=90)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(path, "wb") as f:
                    f.write(r.content)
                return base.rstrip("/").rsplit("/", 1)[1], len(r.content)
            time.sleep(0.5)
        time.sleep(5 * (attempt + 1))
    return None, 0


def main():
    master = json.load(open(SCRIP_MASTER, encoding="utf-8"))
    codes = {row["ISIN_NUMBER"].strip(): str(row["SCRIP_CD"]).strip()
             for row in master if row.get("ISIN_NUMBER")}
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://www.bseindia.com/", timeout=20)

    manifest = os.path.join(CORPUS, "downloaded.csv")
    known_dups = {}
    if os.path.exists(manifest):
        known_dups = {r["attachment"]: r["status"] for r in csv.DictReader(open(manifest, encoding="utf-8"))
                      if r["status"].startswith("duplicate")}

    rows = []
    for stock, isin in STOCKS.items():
        code = codes[isin]
        os.makedirs(os.path.join(CORPUS, stock), exist_ok=True)
        seen = {}
        fr = START
        while fr <= date.today():  # 60-day windows: the API returns one page only
            to = min(fr + timedelta(days=59), date.today())
            for r in list_filings(s, code, fr, to):
                seen[(r.get("NEWS_DT"), r.get("NEWSSUB"))] = r
            fr = to + timedelta(days=1)
            time.sleep(0.4)
        print(f"{stock} ({code}): {len(seen)} announcements since {START}")

        for r in sorted(seen.values(), key=lambda r: r.get("NEWS_DT") or ""):
            headline = (r.get("NEWSSUB") or r.get("HEADLINE") or "").strip()
            attach = (r.get("ATTACHMENTNAME") or "").strip()
            kind = doc_type(headline, r.get("CATEGORYNAME", ""), attach)
            if not kind or not attach:
                continue
            day = (r.get("NEWS_DT") or "")[:10]
            name = f"{day}_{kind}_{attach[:8]}.pdf"
            path = os.path.join(CORPUS, stock, name)
            if attach in known_dups:
                rows.append({"stock": stock, "date": day, "doc_type": kind, "file": "",
                             "status": known_dups[attach], "bytes": 0,
                             "headline": headline, "attachment": attach})
                continue
            if os.path.exists(path):
                where, size = "cached", os.path.getsize(path)
            else:
                where, size = download(s, attach, path)
                time.sleep(0.6)
            print(f"  {day} {kind:13} {size/1e6:6.2f} MB {where or 'FAILED'}  {headline[:70]}")
            rows.append({"stock": stock, "date": day, "doc_type": kind,
                         "file": f"{stock}/{name}" if where else "",
                         "status": where or "failed", "bytes": size,
                         "headline": headline, "attachment": attach})

    # BSE often attaches one PDF to several announcements. Keep the first copy only.
    first = {}
    for r in rows:
        if not r["file"]:
            continue
        path = os.path.join(CORPUS, r["file"])
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        if digest in first:
            os.remove(path)
            r["status"], r["file"] = "duplicate of " + first[digest], ""
        else:
            first[digest] = r["file"]

    with open(os.path.join(CORPUS, "downloaded.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = sum(bool(r["file"]) for r in rows)
    dup = sum(r["status"].startswith("duplicate") for r in rows)
    bad = sum(r["status"] == "failed" for r in rows)
    print(f"\n{ok} files kept, {dup} duplicates removed, {bad} failed; list in corpus/downloaded.csv")


if __name__ == "__main__":
    main()
