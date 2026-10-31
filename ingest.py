"""Step 1: turn the PDFs into pages and chunks.

Inputs: corpus/downloaded.csv (filings since Apr-2025, from fetch_corpus.py) plus the
FY22-FY24 annual reports in a local folder named by EQUITY_FOLDER (read only).

Outputs:
  .store/pages.jsonl   one row per page: cleaned text, section, OCR flag (the parents, later)
  .store/chunks.jsonl  one row per chunk: header + text + metadata
  docs.csv             one row per document: what was kept, dropped, OCR'd
  ingest_report.txt    the counts, and the chunk sizes in mpnet tokens

Choices (each one can be measured in step 3):
  - text from PyMuPDF with sort=True, so a table row keeps its label next to its numbers
  - pages with no text layer are OCR'd (RapidOCR); results cached in .store/ocr_cache.json
  - lines that repeat on more than 30% of a document's pages are dropped (running headers)
  - transcripts are chunked by speaker turn; everything else by line, never across a page
  - every chunk starts with a header naming company, document, period, section and page
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import date

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
STORE = os.path.join(HERE, ".store")
EQUITY = os.environ.get("EQUITY_FOLDER", "annual_reports")

COMPANY = {"Stylam": "Stylam Industries", "Speciality": "Speciality Restaurants"}
LOCAL_ARS = {"Stylam": "StylamIndustriesLtd", "Speciality": "SpecialityRestaurantsLtd"}
LOCAL_AR_YEARS = [2022, 2023, 2024]  # FY25 and FY26 come from the download

TARGET = 900          # characters per chunk, before the header
OVERLAP = 150         # characters of trailing lines carried into the next chunk
MIN_CHUNK = 80        # drop chunks shorter than this (page numbers, stray labels)
BOILERPLATE = 0.30    # a line on more than this share of pages is a running header
OCR_DPI = 150
HEADING_MAX = 90      # a section heading is a line at most this long
MAX_TOKENS = 360      # mpnet reads 384 tokens; keep a margin. Longer chunks are split in two.

DOC_LABEL = {"annual_report": "Annual report", "results": "Results", "transcript": "Earnings call",
             "investor_deck": "Investor deck", "press_release": "Press release",
             "agm_notice": "AGM notice", "governance": "Filing", "operations": "Filing"}

# First match wins. Checked against the top lines of each page.
SECTIONS = [
    ("Independent auditor's report", r"independent auditor"),
    ("Cash flow statement", r"cash flow statement|statement of cash flows?"),
    ("Balance sheet", r"balance sheet"),
    ("Profit and loss", r"statement of profit|profit and loss"),
    ("Changes in equity", r"(statement of )?changes in equity"),
    ("Notes to accounts", r"notes? (to|forming part of)|notes? \d+\b|notes?$"
                          r"|(summary of )?(significant|material) accounting polic"),
    ("Management discussion", r"management discussion|management's discussion"),
    ("Board's report", r"directors'? ?report|board'?s'? report|report of the (board|directors)"),
    ("Corporate governance", r"corporate governance"),
    ("BRSR", r"business responsibility"),
    ("AGM notice", r"notice of (the )?\d*\w* annual general meeting"),
]
HEAD_PREFIX = r"(the |standalone |consolidated |report on |audited )*"
SPEAKER = re.compile(r"^([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,4})\s?:\s*(.*)$")


def fiscal_period(doc_type, day, fy=None):
    """Indian fiscal year ends 31 March. A filing's date tells which period it reports."""
    if fy:
        return f"FY{fy % 100:02d}"
    d = date.fromisoformat(day)
    if doc_type in ("annual_report", "agm_notice"):  # filed Aug-Sep, for the year to March
        return f"FY{d.year % 100:02d}"
    if doc_type in ("results", "transcript", "investor_deck", "press_release"):
        if 4 <= d.month <= 6:
            return f"Q4 FY{d.year % 100:02d}"
        if 7 <= d.month <= 9:
            return f"Q1 FY{(d.year + 1) % 100:02d}"
        if 10 <= d.month <= 12:
            return f"Q2 FY{(d.year + 1) % 100:02d}"
        return f"Q3 FY{d.year % 100:02d}"
    return day


def list_documents():
    docs = []
    for stock, folder in LOCAL_ARS.items():
        for y in LOCAL_AR_YEARS:
            docs.append({"stock": stock, "doc_type": "annual_report", "date": "",
                         "path": os.path.join(EQUITY, folder, "AnnualReports", f"{y}.pdf"),
                         "period": fiscal_period("annual_report", "", fy=y), "url": "",
                         "source": "EquityInvestments"})
    for r in csv.DictReader(open(os.path.join(CORPUS, "downloaded.csv"), encoding="utf-8")):
        if not r["file"]:
            continue
        docs.append({"stock": r["stock"], "doc_type": r["doc_type"], "date": r["date"],
                     "path": os.path.join(CORPUS, r["file"]),
                     "period": fiscal_period(r["doc_type"], r["date"]),
                     "url": "https://www.bseindia.com/xml-data/corpfiling/AttachHis/" + r["attachment"],
                     "source": "BSE"})
    for i, d in enumerate(docs):
        d["doc_id"] = f"{d['stock']}-{i:03d}"
    return docs


_ocr = None


def ocr_page(page, cache, key):
    global _ocr
    if key in cache:
        return cache[key]
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    import numpy as np
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result, _ = _ocr(img)
    cache[key] = "\n".join(r[1] for r in result) if result else ""
    return cache[key]


def clean_line(line):
    line = line.replace("\u00a0", " ").replace("\ufffd", "-")
    line = re.sub(r"[\ue000-\uf8ff]", "-", line)  # bullet symbols from private-use font slots
    line = re.sub(r"\bH(?= ?(lakh|crore|cr\b|in lakh|in crore|million))", "₹", line, flags=re.I)
    if re.search(r"\d", line) and re.search(r"\S {3,}\S", line):
        line = re.sub(r" {3,}", " | ", line.strip())   # a table row: keep the columns apart
    return re.sub(r"\s+", " ", line).strip()


def read_pages(doc, cache):
    pdf = pymupdf.open(doc["path"])
    pages = []
    for n, page in enumerate(pdf, start=1):
        text = page.get_text(sort=True)
        ocr = False
        if len(text.strip()) < 50 and (page.get_images() or len(page.get_drawings()) > 50):
            text = ocr_page(page, cache, f"{doc['path']}#{n}")
            ocr = True
        lines = [clean_line(l) for l in text.splitlines()]
        pages.append({"page": n, "lines": [l for l in lines if l], "ocr": ocr})
    return pages


def drop_boilerplate(pages):
    if len(pages) < 4:
        return 0
    counts = Counter(l for p in pages for l in set(p["lines"]))
    limit = BOILERPLATE * len(pages)
    common = {l for l, c in counts.items() if c > limit and len(l) < 120}
    for p in pages:
        p["lines"] = [l for l in p["lines"] if l not in common]
    return len(common)


def label_sections(doc, pages):
    """Run before drop_boilerplate: a section heading often repeats on every page of its section."""
    current = DOC_LABEL[doc["doc_type"]]
    for p in pages:
        # A heading is a short line. A sentence that mentions "the Statement of Profit and Loss" is not.
        heads = [l.lower() for l in p["lines"][:4] if len(l) <= HEADING_MAX]
        top = " ".join(heads)
        for name, pat in SECTIONS:
            # ...and it starts or ends with the section's name. "Standalone for the year ended
            # Mar 31, 2026 Cash Flow Statement" does; "(a) Income tax expense recognised in the
            # Statement of profit & loss" does neither.
            if any(re.match(HEAD_PREFIX + "(" + pat + ")", h) or re.search("(" + pat + r")\W*$", h)
                   for h in heads):
                scope = (" (consolidated)" if "consolidated" in top else
                         " (standalone)" if "standalone" in top else "")
                current = name + (scope if name in ("Cash flow statement", "Balance sheet",
                                                    "Profit and loss", "Notes to accounts",
                                                    "Changes in equity", "Independent auditor's report") else "")
                break
        p["section"] = current


def split_long(text, limit):
    """A unit longer than the target is cut at sentence ends, then hard-cut if needed."""
    parts, cur = [], ""
    for s in re.split(r"(?<=[.?!])\s+", text):
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
        while len(cur) > limit:
            parts.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        parts.append(cur)
    return parts


def pack(units, overlap):
    """units: [(text, page)]. Greedy packing up to TARGET characters."""
    chunks, cur = [], []

    def flush():
        if cur:
            chunks.append(([u[0] for u in cur], sorted({u[1] for u in cur})))

    for text, page in units:
        for piece in (split_long(text, TARGET) if len(text) > TARGET else [text]):
            if cur and sum(len(u[0]) + 1 for u in cur) + len(piece) > TARGET:
                flush()
                keep, size = [], 0
                for u in reversed(cur):  # carry the last short lines forward
                    if size + len(u[0]) > overlap:
                        break
                    keep.insert(0, u)
                    size += len(u[0])
                cur = keep
            cur.append((piece, page))
    flush()
    return chunks


def fit_tokens(parts, pages, head, count):
    """Split a packed chunk in two, by lines, until each half fits mpnet's input."""
    if count(head + "\n" + "\n".join(parts)) <= MAX_TOKENS:
        return [(parts, pages)]
    if len(parts) == 1:  # one long line: cut it by characters instead
        text = parts[0]
        parts = [text[:len(text) // 2], text[len(text) // 2:]]
    mid = len(parts) // 2
    return fit_tokens(parts[:mid], pages, head, count) + fit_tokens(parts[mid:], pages, head, count)


def transcript_turns(pages):
    turns = []
    for p in pages:
        for line in p["lines"]:
            m = SPEAKER.match(line)
            if m and len(m.group(1)) <= 40:
                turns.append([f"{m.group(1)}: {m.group(2)}".strip(), p["page"]])
            elif turns:
                turns[-1][0] += " " + line
            else:
                turns.append([line, p["page"]])
    return [tuple(t) for t in turns]


def header(doc, section, pages):
    when = doc["period"] + (f", filed {doc['date']}" if doc["date"] else "")
    pg = f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
    return f"[{COMPANY[doc['stock']]} | {DOC_LABEL[doc['doc_type']]} {when} | {section} | {pg}]"


# Superseded documents filed far apart, which the 3-day rule below can't catch. By attachment prefix.
# Stylam: the draft letter of offer (10-Jan-2026) shares 83% of its words with the final one (10-Apr-2026).
SUPERSEDED = {"1ec161bd": "D88AD7AD"}


def near_duplicates(docs, texts):
    """Same stock and type, filed within 3 days, 90% of words shared: keep the longer one."""
    drop = {}
    for d in docs:
        for old, new in SUPERSEDED.items():
            if old in d["path"]:
                drop[d["doc_id"]] = next(e["doc_id"] for e in docs if new in e["path"])
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            if a["stock"] != b["stock"] or a["doc_type"] != b["doc_type"] or not a["date"] or not b["date"]:
                continue
            if abs((date.fromisoformat(a["date"]) - date.fromisoformat(b["date"])).days) > 3:
                continue
            wa, wb = set(texts[a["doc_id"]].split()), set(texts[b["doc_id"]].split())
            if wa and wb and len(wa & wb) / len(wa | wb) >= 0.9:
                short, long_ = (a, b) if len(texts[a["doc_id"]]) < len(texts[b["doc_id"]]) else (b, a)
                drop[short["doc_id"]] = long_["doc_id"]
    return drop


def main():
    os.makedirs(STORE, exist_ok=True)
    cache_path = os.path.join(STORE, "ocr_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    docs = list_documents()
    all_pages, texts = {}, {}
    for d in docs:
        pages = read_pages(d, cache)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        d["pages"] = len(pages)
        d["ocr_pages"] = sum(p["ocr"] for p in pages)
        label_sections(d, pages)
        d["boilerplate_lines"] = drop_boilerplate(pages)
        all_pages[d["doc_id"]] = pages
        texts[d["doc_id"]] = "\n".join(l for p in pages for l in p["lines"])
        d["chars"] = len(texts[d["doc_id"]])
        print(f"  {d['doc_id']} {d['doc_type']:13} {d['period']:8} {d['pages']:4}p "
              f"ocr={d['ocr_pages']:3} {d['chars']:>9,}c", flush=True)

    dups = near_duplicates(docs, texts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tok.model_max_length = 100_000  # we count long texts on purpose; silence the length warning

    def count_tokens(text):
        return len(tok(text)["input_ids"])

    page_rows, chunk_rows = [], []
    for d in docs:
        d["status"] = f"near-duplicate of {dups[d['doc_id']]}" if d["doc_id"] in dups else "kept"
        if d["doc_id"] in dups:
            d["chunks"] = 0
            continue
        pages = all_pages[d["doc_id"]]
        for p in pages:
            page_rows.append({"doc_id": d["doc_id"], "page": p["page"], "section": p["section"],
                              "ocr": p["ocr"], "text": "\n".join(p["lines"])})
        section_of = {p["page"]: p["section"] for p in pages}
        ocr_of = {p["page"]: p["ocr"] for p in pages}
        if d["doc_type"] == "transcript":
            packed = pack(transcript_turns(pages), overlap=0)
        else:
            packed = [c for p in pages for c in pack([(l, p["page"]) for l in p["lines"]], OVERLAP)]
        fitted = []
        for parts, pgs in packed:
            fitted += fit_tokens(parts, pgs, header(d, section_of[pgs[0]], pgs), count_tokens)
        n = 0
        for parts, pgs in fitted:
            body = "\n".join(parts)
            if len(body) < MIN_CHUNK:
                continue
            section = section_of[pgs[0]]
            chunk_rows.append({
                "chunk_id": f"{d['doc_id']}-{n:04d}", "doc_id": d["doc_id"], "stock": d["stock"],
                "doc_type": d["doc_type"], "period": d["period"], "date": d["date"],
                "section": section, "pages": pgs, "ocr": any(ocr_of[p] for p in pgs),
                "url": d["url"], "header": header(d, section, pgs), "text": body})
            n += 1
        d["chunks"] = n

    with open(os.path.join(STORE, "pages.jsonl"), "w", encoding="utf-8") as f:
        for r in page_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(STORE, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in chunk_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fields = ["doc_id", "stock", "doc_type", "period", "date", "pages", "ocr_pages",
              "boilerplate_lines", "chars", "chunks", "status", "source", "url", "path"]
    with open(os.path.join(HERE, "docs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(docs)

    report(docs, chunk_rows, count_tokens)


def report(docs, chunks, count_tokens):
    lengths = [count_tokens(c["header"] + "\n" + c["text"]) for c in chunks]
    out = []
    kept = [d for d in docs if d["status"] == "kept"]
    out.append(f"Ingest run {date.today()}: {len(docs)} documents, {len(kept)} kept, "
               f"{len(docs) - len(kept)} near-duplicates dropped")
    for stock in COMPANY:
        ds = [d for d in kept if d["stock"] == stock]
        cs = [c for c in chunks if c["stock"] == stock]
        out.append(f"  {stock}: {len(ds)} documents, {sum(d['pages'] for d in ds):,} pages "
                   f"({sum(d['ocr_pages'] for d in ds)} OCR'd), {len(cs):,} chunks")
    by_type = Counter(c["doc_type"] for c in chunks)
    out.append("Chunks by document type: " + ", ".join(f"{k} {v:,}" for k, v in by_type.most_common()))
    out.append(f"Chunks from OCR'd pages: {sum(c['ocr'] for c in chunks):,}")
    over = sum(l > 384 for l in lengths)
    q = statistics.quantiles(lengths, n=20)
    out.append(f"Chunk size in mpnet tokens (header included): median {statistics.median(lengths):.0f}, "
               f"95th percentile {q[18]:.0f}, max {max(lengths)}")
    out.append(f"Chunks over mpnet's 384-token limit (the end would be cut off silently): "
               f"{over:,} of {len(lengths):,} ({100 * over / len(lengths):.1f}%)")
    sec = Counter(c["section"] for c in chunks if c["doc_type"] == "annual_report")
    out.append("Annual-report chunks by section: " + ", ".join(f"{k} {v}" for k, v in sec.most_common()))
    text = "\n".join(out)
    print("\n" + text)
    open(os.path.join(HERE, "ingest_report.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
