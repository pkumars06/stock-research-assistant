"""Step 4, first part: the lookup_financials table, built once from the annual reports' own statements.

Why (step 3, 2026-10-20): 11 of the 19 questions search still missed were health-check numbers, and making
table chunks cleaner (version 2) did not fix it: a column of numbers does not embed close to a question in
words. So the numbers the health check needs are read here, at build time, each with its page, and the app
looks them up instead of searching for them.

How:
  1. In each annual report, find the three statement pages by rows only they have: the profit and loss
     (revenue, profit before tax and basic EPS on one page), the balance sheet ("total equity and
     liabilities"), the cash flow (net cash from operating activities). The first of each is standalone;
     one more than 10 pages later is consolidated.
  2. On that page, read each metric's row and take its last two numbers: this year, last year.
  3. The unit comes from the page (₹ lakh or ₹ million); every value is also given in ₹ crore.
  4. Cross-check: last year's column in report N should equal this year's column in report N-1.
     A difference is a restatement (or a misread), and is printed.

Writes financials.csv (one row per stock, year, basis, metric) and financials_report.txt.
"""
import csv
import json
import os
import re
from collections import defaultdict

from add_fields import page_numbers
from rechunk import unit_of

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
TO_CRORE = {"₹ lakh": 0.01, "₹ million": 0.1, "₹ crore": 1.0}
FIGURE = re.compile(r"^\(?-?\d[\d,]*\.\d+\)?$|^[-–]$")   # 1,12,929.39, (75.92), -1,010.24; a dash is nil

# metric -> (statement, row pattern, how). "first": the first matching row; "sum": all matching rows added up
# (borrowings appear once under non-current and once under current liabilities). A tuple of patterns is
# tried in order. "nil": no such row means nil (a company with no borrowings may print no borrowings row).
ROWS = {
    "revenue":          ("pl", r"revenue from operations", "first"),
    "other_income":     ("pl", r"other income", "first"),
    "pbt_before_exceptional": ("pl", r"before exceptional", "first"),
    "pbt":              ("pl", r"profit\s*/?\s*(?:\(loss\)\s*)?before tax", "first"),
    "pat":              ("pl", r"profit\s*/?\s*(?:\(loss\)\s*)?(?:for the (?:year|period)|from continued operations|after tax)", "first"),
    "eps_basic":        ("pl", r"^(?:\(\d\)\s*)?basic", "first"),
    "cash":             ("bs", r"cash and cash equivalents", "first"),
    "other_bank":       ("bs", r"other bank balances|bank balances? other than", "sum nil"),
    "borrowings":       ("bs", r"borrowings", "sum nil"),
    "lease_liabilities": ("bs", r"lease lia", "sum nil"),   # "Lease liablities" in Speciality's FY22 report
    # Stylam's FY22-FY24 reports print no equity total, only its two parts
    "equity":           ("bs", (r"^(?:sub-total \(equity\)|total equity)\s*\|",
                                r"^(?:[a-z]\.\s*\|\s*)?(?:equity share capital|other equity)\s*\|"), "first, then sum"),
    "cwip":             ("bs", r"capital.work.in.progress", "first nil"),
    # mutual funds held as current assets: Speciality keeps most of its spare money there, not in the bank
    "current_investments": ("bs", r"^(?:[a-z]+\.\s*\|\s*)?(?:\([ivx]+\)\s*)?investments\s*\|", "first current nil"),
    "cfo":              ("cf", r"net cash.*operating activities", "first"),
    # rent on leased premises: under Ind AS 116 it is a financing outflow, so CFO looks larger than the cash a
    # restaurant chain really keeps after paying rent. Printed on the cash flow's second page ("next").
    "lease_payments":   ("cf", r"payment of lease liabilit", "first next nil"),
}
PAGE_TESTS = {
    "pl": [r"revenue from operations", r"before tax", r"^(?:\(\d\)\s*)?basic"],
    "bs": [r"total equity and liabilities"],
    "cf": [r"net cash.*operating activities"],
}


def number(cell):
    cell = cell.strip()
    if cell in ("-", "–"):
        return 0.0
    neg = cell.startswith("(") or cell.startswith("-")
    value = float(cell.strip("()-").replace(",", ""))
    return -value if neg else value


def row_values(line):
    """The last two figures on a statement row: (this year, last year), or None.
    A row with one figure gives (that figure, None): Stylam's FY23 report prints "Capital work in progress |
    1,283.47" with last year's cell empty. That is a guess about the column; the cross-check tests it."""
    figs = [t for t in re.split(r"[|\s]+", line) if FIGURE.match(t)]
    if not figs:
        return None
    if len(figs) == 1:
        return (number(figs[0]), None) if figs[0] not in ("-", "–") else None
    return number(figs[-2]), number(figs[-1])


def find_statements(pages):
    """statement -> {"standalone": page, "consolidated": page} for one report."""
    out = {}
    for st, tests in PAGE_TESTS.items():
        hits = [p for p in pages if all(re.search(t, p["text"], re.I | re.M) for t in tests)
                and any(v and v[1] is not None for l in p["text"].split("\n") if re.search(tests[-1], l, re.I)
                        for v in [row_values(l)])]
        found = {}
        if hits:
            found["standalone"] = hits[0]
            later = [p for p in hits if p["page"] > hits[0]["page"] + 10]
            if later:
                found["consolidated"] = later[0]
        out[st] = found
    return out


def read_metric(page, pattern, add_up):
    vals, lines = [], []
    for l in page["text"].split("\n"):
        if re.search(pattern, l, re.I):
            v = row_values(l)
            if v:
                vals.append(v)
                lines.append(l)
                if not add_up:
                    break
    if not vals:
        return None
    last = None if any(v[1] is None for v in vals) else sum(v[1] for v in vals)
    return (sum(v[0] for v in vals), last), " + ".join(l.strip()[:80] for l in lines)


def read_row(page, patterns, how):
    """Read one metric from a statement page: ((this year, last year), the row text), or None."""
    if "current" in how:   # only the rows between "Current assets" and its total
        lines = page["text"].split("\n")
        start = next((i for i, l in enumerate(lines) if re.match(r"^current assets\s*$", l.strip(), re.I)), None)
        if start is None:
            return None
        end = next((i for i in range(start, len(lines)) if re.match(r"^(sub-)?total", lines[i].strip(), re.I)), len(lines))
        page = dict(page, text="\n".join(lines[start:end]))
    if isinstance(patterns, str):
        return read_metric(page, patterns, how.startswith("sum"))
    return read_metric(page, patterns[0], False) or next(
        (got for p in patterns[1:] for got in [read_metric(page, p, True)] if got), None)


# Contingent liabilities are in the notes, not the statements, and printed two ways: Speciality gives a note
# with a "Total" row; Stylam lists the claims with no total, after its commitments ("Note - 38 Commitment and
# Contingencies ... (b) Contigent liabilities" - the report's own spelling), and before guarantees or hedging.
# Only claims against the company are counted, which is what the golden set's answers count (S05, B04, B05).
CONT_HEAD = re.compile(r"contin?gent liabilit|commitments? and contingenc", re.I)
CONT_START = re.compile(r"claims? against|disputed claims", re.I)
CONT_STOP = re.compile(r"^\([b-z]\)|guarantee|hedging|commitments?\b|^(?:note\s*-?\s*)?\d+\.?\s+[A-Z]", re.I)


def contingent(page):
    """((this year, last year), rows) from a contingent-liability note on this page, or None."""
    lines = page["text"].split("\n")
    for h, head in enumerate(lines):
        if len(head) > 90 or not CONT_HEAD.search(head) or re.search(r"policy|provisions", head, re.I):
            continue
        start = next((i for i in range(h, min(h + 12, len(lines))) if CONT_START.search(lines[i])), h)
        # Stylam's FY25-FY26 note opens with commitments; its claims come after "(b) Contigent liabilities"
        if re.search(r"contingenc", head, re.I) and start == h:
            continue
        vals, rows = [], []
        for l in lines[start + 1:start + 20]:
            if re.match(r"^total\b", l, re.I):
                v = row_values(l)
                if v and v[1] is not None:   # the printed total; the rows before it say what the claims are
                    return v, " + ".join(rows + [l.strip()[:90]])
                break
            if CONT_STOP.search(l):
                break
            v = row_values(l)
            if v and v[1] is not None:
                vals.append(v)
                rows.append(l.strip()[:90])
        if vals:
            return (round(sum(v[0] for v in vals), 2), round(sum(v[1] for v in vals), 2)), " + ".join(rows)
    return None


def find_contingent(pages, stmts):
    """The standalone and consolidated contingent notes: after each balance sheet, in the notes that follow."""
    out = {}
    bs = stmts["bs"]
    for basis, page in bs.items():
        end = bs["consolidated"]["page"] if basis == "standalone" and "consolidated" in bs else 10_000
        for p in pages:
            if page["page"] < p["page"] < end:
                got = contingent(p)
                if got:
                    out[basis] = (p, got)
                    break
    return out


def main():
    docs = [r for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
            if r["status"] == "kept" and r["doc_type"] == "annual_report"]
    want = {d["doc_id"] for d in docs}
    pages = defaultdict(list)
    for line in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        if rec["doc_id"] in want:
            pages[rec["doc_id"]].append(rec)
    printed, _ = page_numbers()

    # each report's unit, from its statement pages; Stylam's FY22 report never states one, so it takes the
    # unit of the next year's report, and the cross-check below shows whether that was right
    stmts_of, unit_of_doc, unit_note = {}, {}, {}
    for d in docs:
        stmts_of[d["doc_id"]] = find_statements(pages[d["doc_id"]])
        units = [unit_of(p["text"].split("\n")) for found in stmts_of[d["doc_id"]].values() for p in found.values()]
        units = [u for u in units if u]
        unit_of_doc[d["doc_id"]] = max(set(units), key=units.count) if units else None
    for d in docs:
        if unit_of_doc[d["doc_id"]] is None:
            nxt = next(e for e in docs if e["stock"] == d["stock"] and e["period"] == f"FY{int(d['period'][2:]) + 1}")
            unit_of_doc[d["doc_id"]] = unit_of_doc[nxt["doc_id"]]
            unit_note[d["doc_id"]] = f"unit not printed; taken from the {nxt['period']} report"

    rows, report = [], []
    for d in sorted(docs, key=lambda d: (d["stock"], d["period"])):
        fy = int(d["period"][2:])
        stmts = stmts_of[d["doc_id"]]
        unit = unit_of_doc[d["doc_id"]]
        report.append(f"{d['doc_id']} {d['period']}: " + ", ".join(
            f"{st} " + "/".join(f"{b[:4]} p{p['page']}" for b, p in found.items()) for st, found in stmts.items())
            + f"; {unit}" + (f" ({unit_note[d['doc_id']]})" if d["doc_id"] in unit_note else ""))
        notes = find_contingent(pages[d["doc_id"]], stmts)
        for basis in stmts["bs"]:
            if basis not in notes:
                report.append(f"   MISSING {d['doc_id']} {basis} contingent_liabilities")
        for metric, (st, patterns, how) in [*ROWS.items(), ("contingent_liabilities", ("notes", None, "note"))]:
            for basis, page in (stmts[st].items() if st != "notes" else ((b, n[0]) for b, n in notes.items())):
                if "next" in how:   # the statement continues on the next page
                    nxt = next((p for p in pages[d["doc_id"]] if p["page"] == page["page"] + 1), None)
                    page = dict(page, text=page["text"] + "\n" + (nxt["text"] if nxt else ""))
                got = read_row(page, patterns, how) if st != "notes" else notes[basis][1]
                if got is None:
                    if how.endswith("nil"):
                        got = ((0.0, 0.0), "(no such row on the statement: nil)")
                    else:
                        if metric != "pbt_before_exceptional":   # absent when there were no exceptional items
                            report.append(f"   MISSING {d['doc_id']} {basis} {metric} on p{page['page']}")
                        continue
                (this, last), line = got
                if "next" in how and nxt and line.split(" + ")[0][:40] in nxt["text"]:
                    page = nxt   # cite the page the row is on
                label = printed[(d["doc_id"], page["page"])][0] or ""
                for year, value, column in ((fy, this, "this year"), (fy - 1, last, "last year")):
                    if value is None:
                        continue
                    per_share = metric == "eps_basic"
                    rows.append({
                        "stock": d["stock"], "fy": year, "fiscal_year": f"FY{year:02}", "basis": basis,
                        "metric": metric, "value": value, "unit": "₹ per share" if per_share else unit,
                        "value_crore": value if per_share else round(value * TO_CRORE[unit], 2),
                        "doc_id": d["doc_id"], "report": d["period"], "column": column,
                        "pdf_page": page["page"], "printed_page": label, "url": d["url"], "row": line})

    # one value per stock, year, basis, metric: the LATEST report that gives it. A later report may restate
    # last year (Stylam's FY22 CFO is 12.32 crore in the FY22 report and 5.87 in the FY23 one); the latest
    # figure is the one comparable with the year after it. The first-reported figure is kept beside it.
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["stock"], r["fy"], r["basis"], r["metric"])].append(r)
    best, checks = {}, []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r["report"])
        first, latest = rs[0], rs[-1]
        best[key] = dict(latest, first_reported=first["value_crore"], first_report=first["report"],
                         first_pdf_page=first["pdf_page"])
        if abs(first["value_crore"] - latest["value_crore"]) > max(0.005 * abs(latest["value_crore"]), 0.02):
            checks.append(f"   {key[0]} FY{key[1]:02} {key[2][:4]} {key[3]}: {first['report']} report "
                          f"{first['value_crore']} cr (p{first['pdf_page']}), {latest['report']} report "
                          f"{latest['value_crore']} cr (p{latest['pdf_page']})")
    # derived, so the sign convention of each report does not matter: some print an exceptional "loss/(gain)",
    # others a "gain/(loss)"; the difference between the two profit lines is the same either way
    for (stock, fy, basis, metric), r in list(best.items()):
        if metric != "pbt":
            continue
        before = best.get((stock, fy, basis, "pbt_before_exceptional"))
        pat = best.get((stock, fy, basis, "pat"))
        derived = {"exceptional": (r["value"] - before["value"] if before else 0.0,
                                   "derived: profit before tax - profit before exceptional items"),
                   "tax": (r["value"] - pat["value"] if pat else None, "derived: profit before tax - profit for the year")}
        for name, (value, how) in derived.items():
            if value is not None:
                best[(stock, fy, basis, name)] = dict(r, metric=name, value=round(value, 2),
                                                      value_crore=round(value * TO_CRORE[r["unit"]], 2), row=how)


    out = sorted(best.values(), key=lambda r: (r["stock"], r["basis"], r["metric"], r["fy"]))
    with open(os.path.join(HERE, "financials.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)

    lines = ["Statement pages found (PDF page numbers):", *report, "",
             f"{len(out)} values in financials.csv", "",
             "Figures that differ between two reports, over 0.5% (first reported, then latest; the latest is kept):", *(checks or ["   none"]), ""]
    for stock in ("Stylam", "Speciality"):
        lines.append(f"{stock}, standalone, ₹ crore (EPS in ₹)")
        years = sorted({r["fy"] for r in out if r["stock"] == stock})
        lines.append(f"{'metric':18}" + "".join(f"{'FY%02d' % y:>10}" for y in years))
        for metric in [*ROWS, "contingent_liabilities", "exceptional", "tax"]:
            cells = []
            for y in years:
                r = best.get((stock, y, "standalone", metric))
                cells.append(f"{r['value_crore']:>10.2f}" if r else f"{'':>10}")
            lines.append(f"{metric:18}" + "".join(cells))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "financials_report.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
