"""Check 6: money moving to the owners' side, read row by row from the related-party note.

A related party is someone close to the company: its directors and top managers (key managerial personnel,
KMP), their relatives, companies they control, and the company's own subsidiaries. Dealings with them are
legal and common; the reader should see how big they are.

1. Find the note: its heading inside the standalone statements, to the start of the next note (Stylam's runs
   over 4 pages, Speciality's over 3).
2. Read it: a page printed sideways (Stylam) goes through table_ocr.py, the others use the text layer.
3. Parse each row into {party, kind, item, section, FY26, FY25}. kind is "owner" (a company the owners
   control), "kmp" (a key person or relative), "subsidiary" or "director" (sitting fees).
4. Add up, per year: pay to key people; other dealings with owners' companies and key people (rent, purchases,
   sales, fees, reimbursements; not dividends, which every shareholder gets); the company's money held by them
   (deposits, receivables, loans) apart from what the company owes them (payables, mostly unpaid pay).
5. Cross-check: where the note prints a total for pay (Stylam's summary table), the rows must add up to it.

python related_parties.py   print the rows and the totals for both stocks, FY26
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
NOTE_HEAD = re.compile(r"^\s*(?:note\s*[-–]?\s*)?(\d+)\.?\s+related part\w*\s+d\w*sclosure", re.I)
MAX_PAGES = 6
FIG = r"\(?-?[\d,]+\.\d\d\)?|-"
TEXT_ROW = re.compile(rf"^(.*?)\s*\|\s*({FIG})\s*\|\s*({FIG})\s*$")
OCR_FIG = re.compile(r"^FY(\d\d): (\(?-?[\d,]+\.\d\d\)?)$")
FURNITURE = {"statutory reports", "corporateoverview", "corporate overview", "financialstatements",
             "financial statements", "annual report 2025-26"}
NOT_DEALINGS = ("remuneration", "dividend", "sitting fees", "commission")


def value(s):
    if s in ("-", ""):
        return 0.0
    neg = s.startswith("(")
    v = float(s.strip("()").replace(",", ""))
    return -v if neg else v


def squash(s):
    return re.sub(r"[^a-z]", "", s.lower())


def note_pages(stock, fy):
    """(doc row, [(pdf page, text)]) for the related-party note of that annual report, cut at the next note."""
    from build_financials import find_statements
    d = next(r for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
             if r["stock"] == stock and r["doc_type"] == "annual_report" and r["period"] == f"FY{fy}")
    pages = [json.loads(l) for l in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8")
             if l.startswith('{"doc_id": "' + d["doc_id"] + '"')]
    bs = find_statements(pages)["bs"]
    start, end = bs["standalone"]["page"], bs.get("consolidated", {"page": 10_000})["page"]
    for p in pages:
        if start < p["page"] < end:
            m = next((NOTE_HEAD.match(l) for l in p["text"].split("\n") if NOTE_HEAD.match(l)), None)
            if m:
                head, number = p["page"], int(m.group(1))
                break
    else:
        return d, []
    nxt = re.compile(rf"^\s*(?:note\s*[-–]?\s*)?{number + 1}\b", re.I)
    out = []
    for p in pages:
        if head <= p["page"] < head + MAX_PAGES:
            lines = p["text"].split("\n")
            cut = next((i for i, l in enumerate(lines) if nxt.match(l)), None)
            if cut is None or cut > 3:          # a page that only carries the header before the next note is left out
                out.append((p["page"], "\n".join(lines[:cut])))
            if cut is not None:
                break
    return d, out


def sideways(doc_path, page_no):
    import pymupdf
    from table_ocr import turns
    return turns(pymupdf.open(doc_path)[page_no - 1]) != 0


def parse_text(page_rows, fy):
    """Speciality's layout: group lines ("Entities in which KMP ...", "Subsidiaries") above "name | this | last"."""
    rows, section, item, kind = [], None, None, None
    for page, line in page_rows:
        low = line.lower().strip()
        if low.startswith("b) details of transactions"):
            section = "transaction"
            continue
        if low.startswith("c) balances outstanding"):
            section = "balance"
            continue
        if section is None or not low or low.startswith("particulars") or low == "(restated)" or low.isdigit():
            continue
        m = TEXT_ROW.match(line)
        if m:
            rows.append({"party": m.group(1).strip(), "kind": kind, "item": item, "section": section, "page": page,
                         f"FY{fy}": value(m.group(2)), f"FY{fy - 1}": value(m.group(3))})
        elif low.startswith("entities in which kmp"):
            kind = "owner"
        elif low.startswith("key managerial personnel"):
            kind = "kmp"
        elif low.startswith("subsidiaries"):
            kind = "subsidiary"
        elif low.startswith("notes:"):
            section = None
        else:   # an item heading; some name their kind too
            item = re.sub(r"\s*\(refer.*$", "", line.strip(), flags=re.I)
            kind = ("kmp" if "kmp" in low and "remuneration" in low else
                    "director" if "non-executive" in low or "sitting fees" in low else None)
    return rows


def owner_names(text):
    """Stylam's names list: the companies after 'Companies/ Concerns controlled by KMP', before 'Key Managerial'.
    A match can carry the words before the name on its line ("...their Gourmet House Private Limited"), so the
    caller matches a party by how a name ends."""
    block = text[text.find("Concerns controlled"):text.find("Key Managerial")]
    return {squash(n) for n in re.findall(r"([A-Z][A-Za-z ]+? Private Limited)", block)}


def parse_ocr(page_rows, fy, owners):
    """Stylam's layout, after OCR: a party line ("Mr. ... (Managing Director)", "Gourmet House Private Limited")
    above "item | FY26: x | FY25: y". The category summary table before the per-party table gives the totals."""
    rows, summary, party, kind, per_party = [], {}, None, None, False
    for page, line in page_rows:
        cells = [c.strip() for c in line.split("|") if c.strip().lower() not in FURNITURE]
        if not cells:
            continue
        if squash(cells[0]).startswith("transactionsbetweenthecompany"):
            per_party = True
            continue
        figs = [(int(m.group(1)), value(m.group(2))) for c in cells for m in [OCR_FIG.match(c)] if m]
        first_fig = next((i for i, c in enumerate(cells) if OCR_FIG.match(c)), len(cells))
        words = cells[:first_fig]
        if re.match(r"^(mr|ms|mrs)\.?", squash(words[0]) if words else "") or re.search(r"limited|bhd", words[0].lower() if words else ""):
            party = words[0]
            kind = ("kmp" if re.match(r"^(mr|ms|mrs)", squash(party)) else
                    "owner" if any(o.endswith(squash(party)) for o in owners) else "subsidiary")
            words = words[1:]
        if not figs:
            continue
        item = words[-1] if words else ""
        by_year = {}
        for y, v in figs:                      # a summary row can carry several columns for one year: keep them all
            by_year.setdefault(f"FY{y}", []).append(v)
        if not per_party:
            summary[item.lower()] = by_year
            continue
        rows.append({"party": party, "kind": kind, "item": item, "section": "transaction", "page": page,
                     **{k: v[0] for k, v in by_year.items()}})
    return rows, summary


def related_parties(stock, fy):
    from table_ocr import page_rows
    d, pages = note_pages(stock, fy)
    if not pages:
        return {"doc": d, "pages": [], "rows": [], "summary": {}, "read": "note not found"}
    turned = [p for p, _ in pages if sideways(d["path"], p)]
    if turned:
        all_rows = [(p, l) for p, _ in pages for l in (page_rows(d["doc_id"], p) if p in turned else [])]
        rows, summary = parse_ocr(all_rows, fy, owner_names(pages[0][1]))
        how = f"OCR of {len(turned)} sideways pages"
    else:
        rows = parse_text([(p, l) for p, t in pages for l in t.split("\n")], fy)
        summary, how = {}, "text layer"
    return {"doc": d, "pages": [p for p, _ in pages], "rows": rows, "summary": summary, "read": how}


def totals(rp, fy):
    """Per year: pay to key people, other dealings with the owners' side, money held by them and owed to them
    (in the note's unit, as printed)."""
    out = {}
    for y in (f"FY{fy}", f"FY{fy - 1}"):
        pay = deal = held = owed = 0.0
        for r in rp["rows"]:
            v = r.get(y, 0.0)
            item = (r["item"] or "").lower()
            if r["kind"] == "kmp" and r["section"] == "transaction" and "remuneration" in item:
                pay += v
            elif r["kind"] in ("owner", "kmp") and r["section"] == "transaction" and not any(w in item for w in NOT_DEALINGS):
                deal += v
            elif r["kind"] in ("owner", "kmp") and r["section"] == "balance":
                if "payable" in item:
                    owed += v
                else:
                    held += v
        out[y] = {"pay": round(pay, 2), "dealings": round(deal, 2), "held_by_them": round(held, 2),
                  "owed_to_them": round(owed, 2)}
        printed = rp["summary"].get("remuneration", {}).get(y)
        if printed:
            out[y]["pay_printed_total"] = max(printed)
    return out


if __name__ == "__main__":
    for stock in ("Speciality", "Stylam"):
        rp = related_parties(stock, 26)
        print(f"===== {stock}: note on PDF pages {rp['pages']}, read by {rp['read']}, {len(rp['rows'])} rows")
        for r in rp["rows"]:
            print(f"  p{r['page']} {r['kind'] or '?':10} {r['section']:11} {(r['item'] or '')[:38]:38} "
                  f"{r['party'][:45]:45} FY26 {r.get('FY26', 0):>9,.2f}  FY25 {r.get('FY25', 0):>9,.2f}")
        print("  totals:", totals(rp, 26), "\n")
