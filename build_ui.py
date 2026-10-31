"""Step 6: everything the UI shows that does not depend on the user's question, worked out once at build time.

The app (app.py) only reads .store/ui/ui_data.json, so opening a tab is instant. Only the Ask tab runs at
question time, because only it depends on what the user types (build time against question time, guide 2 Part 2).

    overview   tab 1: the BRSR's Section A fields of the latest annual report, and one search for its own words
    growth     tab 1: revenue and profit for every year from lookup_financials, growth worked out here
    health     tab 2 and Compare: health_check() with the searcher, so check 7 runs too (one call_llm() each)
    events     tab 4: filings since 1 October 2025, newest first, the opening lines of each letter
    promises   tab 3: check 7's dated excerpts, oldest first, and the model's promise lines

No LLM writes the overview or the events: they are passages from the filings, cited. With --llm (meant for
when the OpenAI key arrives) call_llm() adds a plain-words line to each, kept beside the passage it came from.

python build_ui.py           about 5 minutes on this CPU with local Qwen (check 7, twice)
python build_ui.py --llm     also the plain-words lines
"""
import csv
import json
import os
import re
import sys
import time

from answer import cite
from financials import series
from healthcheck import health_check
from retrieve import Searcher

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, ".store", "ui", "ui_data.json")
STOCKS = ["Stylam", "Speciality"]
OVERVIEW_FY = 26
EVENTS_FROM = "2025-10-01"
EVENT_TYPES = ("governance", "operations", "press_release", "results")

# Tab 1 reads Section A of the BRSR (the business responsibility report inside every annual report): fixed,
# numbered fields that say what the company sells, where, and to whom. Search ranked the BRSR's stock wording
# above these fields (2026-10-23), so the fields are read by their own headings, like check 4 reads its note.
BRSR_FIELDS = [
    ("What it sells", r"Products/Services sold by the entity", r"III\.\s*Operations|18\.\s*Number of locations"),
    ("Where it makes or serves", r"Number of locations where plants", r"19\.\s*Markets served"),
    ("Where it sells", r"Markets served by the entity", r"c\.\s*A brief on types of customers"),
    ("Who buys", r"A brief on types of customers", r"IV\.\s*Employees"),
]
OWN_WORDS_QUERY = "company overview: who we are, what we make or serve, our brands, plants and outlets"
OWN_WORDS_SECTIONS = ("Annual report", "Management discussion")
OVERVIEW_DOC_TYPES = ("annual_report",)

# Years a growth rate should not be read on its own. Each note comes from a filing already checked
# (context/capstone-plan.md, 2026-10-20), not from memory.
DISTORTED = {
    "Speciality": ["FY22 starts from a COVID year (restaurants closed for part of FY21), so FY22's growth is a rebound.",
                   "FY23 profit includes a one-off tax credit of about ₹33.4 crore; profit has been ₹21-27 crore a year since."],
    "Stylam": [],
}

PLAIN_SYSTEM = """Rewrite the excerpt from a company filing in two short, plain sentences for someone who has never
read a filing. Use only what the excerpt says. Keep every number and date exactly as written. No advice, no opinion."""


def annual_report(stock, fy=OVERVIEW_FY):
    return next(r["doc_id"] for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
                if r["stock"] == stock and r["doc_type"] == "annual_report" and r["period"] == f"FY{fy}")


def page_source(searcher, doc_id, page):
    c = next((c for c in searcher.chunks if c["doc_id"] == doc_id and c["pages"][0] == page), None)
    return (cite(c, page), c["url"]) if c else (f"{doc_id}, PDF p{page}", "")


def brsr_fields(searcher, stock):
    """Each Section A field as printed: its text, with the PDF page where it starts."""
    doc_id = annual_report(stock)
    pages = [json.loads(l) for l in open(os.path.join(HERE, ".store", "pages.jsonl"), encoding="utf-8")]
    pages = [p for p in pages if p["doc_id"] == doc_id]
    start = next(p["page"] for p in pages if re.search(r"Details of business activities", p["text"], re.I))
    # Section A sits on two or three pages; join them, dropping the page number printed at each foot
    text, where = "", []
    for p in pages:
        if start <= p["page"] <= start + 2:
            body = re.sub(r"(?<=[^|\s])\s+\d{1,3}\s*$", "", " ".join(p["text"].split()))
            where.append((len(text), p["page"]))
            text += body + " "
    out = []
    for heading, begin, end in BRSR_FIELDS:
        m = re.search(f"({begin}.*?)(?:{end})", text, re.I | re.S)
        if not m:
            out.append({"heading": heading, "passages": []})
            continue
        page = max(pg for at, pg in where if at <= m.start())
        source, url = page_source(searcher, doc_id, page)
        out.append({"heading": heading, "passages": [{"text": m.group(1).strip(), "section": "BRSR, Section A",
                                                      "source": source, "url": url}]})
    return out


def own_words(searcher, stock):
    hits = [i for i in searcher.rerank(OWN_WORDS_QUERY, {"stocks": [stock], "fy": (OVERVIEW_FY, OVERVIEW_FY)}, 30)
            if searcher.chunks[i]["doc_type"] in OVERVIEW_DOC_TYPES
            and searcher.chunks[i]["section"] in OWN_WORDS_SECTIONS][:2]
    return {"heading": "In the company's own words", "passages": [
        {"text": searcher.chunks[i]["text"], "section": searcher.chunks[i]["section"],
         "source": cite(searcher.chunks[i], searcher.chunks[i]["pages"][0]), "url": searcher.chunks[i]["url"]}
        for i in hits]}


def overview(searcher, stock):
    return brsr_fields(searcher, stock) + [own_words(searcher, stock)]


def growth(stock):
    rows = {}
    for metric in ("revenue", "pat"):
        prev = None
        for r in series(stock, metric):
            if "error" in r:
                continue
            v = r["value_crore"]
            g = (v / prev - 1) if prev not in (None, 0) and v is not None else None
            rows.setdefault(r["fiscal_year"], {})[metric] = {"value": v, "growth": g, "source": r["source"]}
            prev = v
    return {"years": [{"fiscal_year": fy, **m} for fy, m in sorted(rows.items())], "notes": DISTORTED.get(stock, [])}


def letter_body(text):
    """The opening of a filing's letter: what follows "Dear Sir/Madam", or the subject line, or the start."""
    flat = " ".join(text.split())
    m = re.search(r"Dear\s+Sir(?:\s*/\s*|\s+or\s+)?(?:Madam|Ma'am)?[,.:]?\s*(.+)", flat, re.I)
    if m:
        return m.group(1)[:600]
    m = re.search(r"(?:Sub(?:ject)?\s*[:\-.])\s*(.+)", flat, re.I)
    return (m.group(1) if m else flat)[:600]


def events(stock):
    docs = [r for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
            if r["stock"] == stock and r["doc_type"] in EVENT_TYPES and r["date"] >= EVENTS_FROM and r["status"] == "kept"]
    first_pages = {}
    for line in open(os.path.join(HERE, ".store", "pages.jsonl"), encoding="utf-8"):
        p = json.loads(line)
        if p["page"] <= 2 and p["doc_id"] in {d["doc_id"] for d in docs}:
            first_pages[p["doc_id"]] = first_pages.get(p["doc_id"], "") + "\n" + p["text"]
    return [{"date": d["date"], "doc_type": d["doc_type"].replace("_", " "), "period": d["period"],
             "pages": int(d["pages"]), "url": d["url"], "doc_id": d["doc_id"],
             "excerpt": letter_body(first_pages.get(d["doc_id"], ""))}
            for d in sorted(docs, key=lambda d: d["date"], reverse=True)]


def plain_lines(data):
    from llm import call_llm
    for stock in STOCKS:
        for e in data[stock]["events"]:
            e["plain"] = call_llm([{"role": "system", "content": PLAIN_SYSTEM},
                                   {"role": "user", "content": e["excerpt"]}], max_tokens=120, purpose="ui event")["text"]
        for part in data[stock]["overview"]:
            for p in part["passages"]:
                p["plain"] = call_llm([{"role": "system", "content": PLAIN_SYSTEM},
                                       {"role": "user", "content": p["text"]}], max_tokens=120, purpose="ui overview")["text"]


def main():
    t = time.time()
    searcher = Searcher("v1")
    data = {"built": time.strftime("%Y-%m-%d %H:%M"), "stocks": STOCKS}
    for stock in STOCKS:
        print(f"{stock}: overview, growth, events ...", flush=True)
        data[stock] = {"overview": overview(searcher, stock), "growth": growth(stock), "events": events(stock)}
        print(f"{stock}: health check with check 7 ({time.time() - t:.0f} s so far) ...", flush=True)
        data[stock]["health"] = health_check(stock, searcher=searcher)
    if "--llm" in sys.argv:
        plain_lines(data)
    data["llm_model"] = next((c["llm"].get("model") for s in STOCKS for c in data[s]["health"]["checks"]
                              if c.get("llm")), None)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved {OUT} in {time.time() - t:.0f} s")
    for stock in STOCKS:
        d = data[stock]
        print(f"  {stock}: {sum(len(p['passages']) for p in d['overview'])} overview passages, "
              f"{len(d['growth']['years'])} years, {len(d['events'])} filings, "
              f"labels {[c['label'] for c in d['health']['checks']]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
