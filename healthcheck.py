"""Step 4: the health check, a workflow (fixed steps in plain code), not an agent.

The seven checks agreed on 2026-10-20 for people who invest through apps but never read filings. Each check
gives a finding with numbers, a label ("Looks fine" / "Worth a closer look") with the threshold shown, what it
means, why it matters, and the pages it came from. No advice: no buy / sell / hold, no score, no target price.

Checks 1-4 are arithmetic on the lookup_financials table. Check 5 reads the auditor's report with fixed rules.
Check 6 adds up the related-party note's rows (related_parties.py). Check 7 is the only one that needs an LLM:
fixed searches gather what management said, call_llm() lists each promise in a fixed format, and a rule sets
the label (any date moved back -> worth a closer look).

The thresholds are rules of thumb chosen for this app and shown to the reader, not laws or official limits.
"""
import json
import os
import re
import sys
from collections import defaultdict

from build_financials import TO_CRORE
from financials import lookup_financials

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
FINE, LOOK = "Looks fine", "Worth a closer look"
YEARS = 3   # checks 2 looks at three years, so one odd year does not decide the label

# thresholds, shown to the reader with every label
ONE_OFF_SHARE = 0.10        # exceptional items above 10% of profit before them
TAX_LOW, TAX_HIGH = 0.15, 0.35   # India's usual company tax rate is about 25%
OTHER_INCOME_SHARE = 0.30   # other income above 30% of profit before tax
CASH_RATIO = 1.0            # cash from operations / profit after tax, over three years
CONTINGENT_SHARE = 0.10     # contingent liabilities above 10% of net worth


def get(stock, metric, fy):
    return lookup_financials(stock, metric, fy)


def cr(x):
    return f"₹{x:,.2f} crore"


def pct(x):
    return f"{x * 100:.2f}%" if abs(x) < 0.01 else f"{x * 100:.1f}%"   # 0.03%, not "0.0%"


def evidence(*items):
    # a nil that comes from a row being absent is not printed on any page, so it gets no citation
    return [{"what": r["label"], "year": r["fiscal_year"], "value": r["as_printed"], "source": r["source"],
             "url": r["url"]} for r in items if "error" not in r and not r["nil_by_absence"]]


def claim_items(r):
    """The claims behind a contingent-liability figure, from the note's rows, in ₹ crore."""
    from build_financials import row_values
    unit = TO_CRORE[r["as_printed"].split(" ", 1)[1]]
    col = 0 if r["column"] == "this year" else 1
    out = []
    for part in r["row"].split(" + "):
        v = row_values(part)
        # "a) | Legal cases against the Company | 166.16 | 166.16" -> "Legal cases against the Company"
        cells = [c.strip() for c in part.split("|") if not re.fullmatch(r"\s*[a-z][.)]\s*", c, re.I)]
        name = re.sub(r"^[a-z][.)]\s*", "", cells[0], flags=re.I)
        if not v or not v[col] or name.lower().startswith("total"):
            continue
        out.append(f"{name} {cr(v[col] * unit)}")
    return out


def latest_year(stock):
    return int(get(stock, "pat", None)["fiscal_year"][2:])


# --- check 1 -------------------------------------------------------------------------------------------
def profit_real(stock, fy):
    pbt, pat, tax, exc, oth = (get(stock, m, fy) for m in ("pbt", "pat", "tax", "exceptional", "other_income"))
    before = pbt["value_crore"] - exc["value_crore"]
    one_off = abs(exc["value_crore"]) / abs(before) if before else 0
    rate = tax["value_crore"] / pbt["value_crore"] if pbt["value_crore"] > 0 else None
    other = oth["value_crore"] / pbt["value_crore"] if pbt["value_crore"] > 0 else None
    flags, lines = [], [f"Profit before tax was {cr(pbt['value_crore'])} and profit after tax {cr(pat['value_crore'])} "
                        f"in FY{fy}."]
    if exc["value_crore"]:
        way = "cut" if exc["value_crore"] < 0 else "added to"
        lines.append(f"One-off (exceptional) items {way} profit by {cr(abs(exc['value_crore']))}, "
                     f"{pct(one_off)} of the profit before them; without them, profit before tax would have "
                     f"been {cr(before)}.")
        if one_off > ONE_OFF_SHARE:
            flags.append("one-off items")
    else:
        lines.append("There were no one-off (exceptional) items.")
    if rate is not None:
        lines.append(f"Tax took {pct(rate)} of profit before tax.")
        if not TAX_LOW <= rate <= TAX_HIGH:
            flags.append("tax rate")
    # in a loss year a share of profit means nothing, so it is not printed (pct(None) crashed; the review of
    # 2026-10-23 found it, no year checked so far is a loss year)
    lines.append(f"Other income (interest, dividends, gains on investments; not the main business) was "
                 f"{cr(oth['value_crore'])}" + (f", equal to {pct(other)} of profit before tax." if other is not None
                                                else ", in a year with no profit before tax."))
    if other is not None and other > OTHER_INCOME_SHARE:
        flags.append("other income")
    if flags:
        lines.append("Outside the usual range: " + ", ".join(flags) + ".")
    return {
        "id": 1, "title": "Is the profit real?",
        "label": LOOK if flags else FINE,
        "threshold": (f"Looks fine when one-off items are under {pct(ONE_OFF_SHARE)} of profit, tax is between "
                      f"{pct(TAX_LOW)} and {pct(TAX_HIGH)} of profit (the usual company rate in India is about 25%), "
                      f"and other income is under {pct(OTHER_INCOME_SHARE)} of profit before tax."),
        "finding": " ".join(lines),
        "meaning": ("Profit can be lifted by things that will not happen again: a one-off gain, an unusually "
                    "low tax bill, or income from investments rather than from selling the company's products."),
        "why": "The profit that repeats next year is the part that comes from the main business at a normal tax rate.",
        "evidence": evidence(pbt, pat, exc, oth),
    }


# --- check 2 -------------------------------------------------------------------------------------------
def cash_backed(stock, fy):
    years = list(range(fy - YEARS + 1, fy + 1))
    cfo = [get(stock, "cfo", y) for y in years]
    pat = [get(stock, "pat", y) for y in years]
    rent = [get(stock, "lease_payments", y) for y in years]
    sum_cfo = sum(r["value_crore"] for r in cfo)
    sum_pat = sum(r["value_crore"] for r in pat)
    sum_rent = -sum(r["value_crore"] for r in rent)   # printed as a minus (cash out)
    span = f"FY{years[0]}-FY{years[-1]}"
    if sum_pat <= 0:
        # cash ÷ profit has no meaning when the profit is a loss; say so instead of dividing (the review of 2026-10-23)
        return {"id": 2, "title": "Is the profit backed by cash?", "label": LOOK,
                "threshold": "The cash-to-profit test needs a profit to compare with.",
                "finding": (f"Over {span}, the company reported a combined loss of {cr(-sum_pat)} after tax, and "
                            f"{cr(sum_cfo)} of cash from its operations. With no profit, cash ÷ profit cannot be worked out."),
                "meaning": ("Profit is an accounting figure; cash is money actually received. In loss years, look at "
                            "whether the business still brings in cash."),
                "why": "A company pays salaries, suppliers and dividends with cash, not with profit.",
                "evidence": evidence(*cfo, *pat)}
    ratio = sum_cfo / sum_pat
    lines = [f"Over {span}, the company reported {cr(sum_pat)} of profit after tax and brought in "
             f"{cr(sum_cfo)} of cash from its operations: {ratio:.2f} times the profit."]
    used = ratio
    if sum_rent > 0:
        used = (sum_cfo - sum_rent) / sum_pat
        lines.append(f"It also paid {cr(sum_rent)} in rent on leased premises. The accounts count that rent as "
                     f"financing, not operations, so after paying it the cash is {cr(sum_cfo - sum_rent)}: "
                     f"{used:.2f} times the profit. The label uses this figure.")
    per_year = ", ".join(f"FY{y} {c['value_crore'] / p['value_crore']:.2f}" for y, c, p in zip(years, cfo, pat)
                         if p["value_crore"] > 0)
    lines.append(f"Year by year, cash from operations ÷ profit: {per_year}.")
    restated = [c for c in cfo if "restated" in c]
    for c in restated:
        lines.append(f"FY{c['fiscal_year'][2:]} cash from operations was {c['restated']}.")
    return {
        "id": 2, "title": "Is the profit backed by cash?",
        "label": FINE if used >= CASH_RATIO else LOOK,
        "threshold": (f"Looks fine when cash from operations is at least {CASH_RATIO:.1f} times profit after tax, "
                      f"added up over {YEARS} years (after rent, for a company that leases its premises)."),
        "finding": " ".join(lines),
        "meaning": ("Profit is an accounting figure; cash is money actually received. A sale on credit counts as "
                    "profit before the customer pays. If cash keeps falling short of profit, money is getting "
                    "stuck in unpaid bills or stock."),
        "why": "A company pays salaries, suppliers and dividends with cash, not with profit.",
        "evidence": evidence(*cfo, *pat, *[r for r in rent if r["value_crore"]]),
    }


# --- check 3 -------------------------------------------------------------------------------------------
def owes(stock, fy):
    items = {m: get(stock, m, fy) for m in ("borrowings", "lease_liabilities", "cash", "other_bank",
                                           "current_investments")}
    v = {m: r["value_crore"] for m, r in items.items()}
    owed = v["borrowings"] + v["lease_liabilities"]
    liquid = v["cash"] + v["other_bank"] + v["current_investments"]
    lines = [f"At 31 March 20{fy}: borrowings {cr(v['borrowings'])}"]
    if v["lease_liabilities"]:
        lines[0] += (f", plus {cr(v['lease_liabilities'])} of lease liabilities (rent it has agreed to pay "
                     f"landlords in future years, which the accounts treat like a loan)")
    lines[0] += "."
    parts = [f"cash {cr(v['cash'])}"]
    if v["other_bank"]:
        parts.append(f"bank deposits {cr(v['other_bank'])}")
    if v["current_investments"]:
        parts.append(f"current investments such as mutual funds {cr(v['current_investments'])}")
    lines.append(f"Against that it held {', '.join(parts)}: {cr(liquid)} in all.")
    gap = liquid - owed
    lines.append(f"So it holds {cr(gap)} more than it owes." if gap >= 0 else
                 f"So it owes {cr(-gap)} more than it holds in cash and investments.")
    if v["borrowings"] and not v["lease_liabilities"] and v["borrowings"] < 0.5 * liquid:
        lines.append("Its borrowings are small next to its cash, but they are not zero.")
    return {
        "id": 3, "title": "How much does it owe?",
        "label": FINE if gap >= 0 else LOOK,
        "threshold": "Looks fine when cash, bank deposits and current investments cover borrowings plus lease liabilities.",
        "finding": " ".join(lines),
        "meaning": ("Borrowings are loans from banks or lenders. Lease liabilities are rent promised to landlords. "
                    "Both have to be paid whether business is good or bad."),
        "why": "A company with more cash than debt can get through a bad year; one with more debt than cash depends on its lenders.",
        "evidence": evidence(*items.values()),
    }


# --- check 4 -------------------------------------------------------------------------------------------
def hidden_claims(stock, fy):
    cont, eq = get(stock, "contingent_liabilities", fy), get(stock, "equity", fy)
    prev = get(stock, "contingent_liabilities", fy - 1)
    share = cont["value_crore"] / eq["value_crore"]
    lines = [f"At 31 March 20{fy}, claims against the company that it disputes came to "
             f"{cr(cont['value_crore'])}, which is {pct(share)} of its net worth of {cr(eq['value_crore'])}."]
    if "error" not in prev:
        change = "down" if prev["value_crore"] > cont["value_crore"] else "up"
        lines.append(f"A year earlier they were {cr(prev['value_crore'])} ({change}).")
    items = claim_items(cont)
    if items:
        lines.append("The claims: " + ", ".join(items) + ".")
    return {
        "id": 4, "title": "Hidden claims (contingent liabilities)",
        "label": FINE if share <= CONTINGENT_SHARE else LOOK,
        "threshold": f"Looks fine when disputed claims are at most {pct(CONTINGENT_SHARE)} of net worth.",
        "finding": " ".join(lines),
        "meaning": ("A contingent liability is a claim the company disputes, such as a tax demand it is fighting. "
                    "It is not in the balance sheet's debts; it is paid only if the company loses."),
        "why": "If the company loses, the money comes out of the owners' share of the company (its net worth).",
        "evidence": evidence(cont, eq),
    }


# --- check 5 -------------------------------------------------------------------------------------------
MODIFIED = re.compile(r"qualified opinion|adverse opinion|disclaimer of opinion|except for the (?:effects?|possible)", re.I)
EMPHASIS = re.compile(r"emphasis of matters?|material uncertainty related to going concern", re.I)
BOOKS = re.compile(r"proper books of account[^.]{0,200}?except for", re.I | re.S)


def auditor_pages(doc_id):
    """The standalone auditor's report: the run of auditor pages before the standalone balance sheet."""
    pages = [json.loads(l) for l in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8")
             if l.startswith('{"doc_id": "' + doc_id + '"')]
    aud = [p for p in pages if p["section"].lower().startswith("independent auditor")
           and "consolidated" not in p["section"].lower()]
    if not aud:
        return []
    run = [aud[0]]   # the first contiguous run is the standalone report and its annexures
    for p in aud[1:]:
        if p["page"] - run[-1]["page"] > 3:
            break
        run.append(p)
    return run


def auditor(stock, fy):
    import csv
    from add_fields import page_numbers
    printed, _ = page_numbers()
    docs = sorted((r for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
                   if r["stock"] == stock and r["doc_type"] == "annual_report" and r["status"] == "kept"),
                  key=lambda r: r["period"])
    history, latest = [], None
    for d in docs:
        pages = auditor_pages(d["doc_id"])
        text = {p["page"]: re.sub(r"\s+", " ", p["text"]) for p in pages}
        found = {"modified": [n for n, t in text.items() if MODIFIED.search(t)],
                 "emphasis": [n for n, t in text.items() if EMPHASIS.search(t)],
                 "books": [n for n, t in text.items() if BOOKS.search(t)]}
        # what the emphasis paragraph points to: "We draw attention to Note 35 ..., which describes ..."
        # (a sentence ends at a full stop followed by a capital, so "Rs 333.47" does not end it)
        found["emphasis_text"] = ""
        for n in found["emphasis"]:
            after = text[n][EMPHASIS.search(text[n]).end():]
            m = (re.search(r"we draw attention to.{20,300}?\.(?=\s+[A-Z(]|$)", after, re.I)
                 or re.search(r"note \d+.{20,300}?\.(?=\s+[A-Z(]|$)", after, re.I))
            if m:
                found["emphasis_text"] = re.sub(r"\s+\d+\.$", "", m.group(0))
                break
        history.append((d, pages, found))
        if d["period"] == f"FY{fy}":
            latest = (d, pages, found)
    d, pages, found = latest

    def cite(doc, n):
        label = printed[(doc["doc_id"], n)][0]
        return f"{stock} annual report {doc['period']}, PDF p{n}" + (f" (printed {label})" if label else "")

    lines = []
    if not pages:
        lines.append(f"The auditor's report was not found in the FY{fy} annual report.")
    elif found["modified"]:
        lines.append(f"The auditor's opinion for FY{fy} is modified (qualified, adverse or a disclaimer).")
    else:
        lines.append(f"The auditor's opinion for FY{fy} is unmodified: the accounts 'give a true and fair view'.")
    if found["emphasis"]:
        lines.append("The auditor added an emphasis-of-matter or going-concern paragraph, pointing readers to something specific.")
    if found["books"]:
        lines.append("The auditor noted one exception about the books of account: under Rule 11(g) the accounting "
                     "software must keep an unchangeable log of every edit (an audit trail), and this was not fully "
                     "met. This is a record-keeping point, not a finding that the figures are wrong.")
    past = [f"{h[0]['period']}: " + ("modified" if h[2]["modified"] else "unmodified")
            + (f", emphasis of matter (\"{h[2]['emphasis_text']}\")" if h[2]["emphasis"] else "")
            + (", books exception" if h[2]["books"] else "") for h in history if h[1]]
    lines.append("By year: " + "; ".join(past))
    pages_cited = sorted(set(found["modified"] + found["emphasis"] + found["books"])) or [p["page"] for p in pages[:1]]
    return {
        "id": 5, "title": "Did the auditor raise concerns?",
        "label": LOOK if (not pages or found["modified"] or found["emphasis"]) else FINE,
        "threshold": ("Looks fine when the opinion is unmodified ('true and fair view') with no emphasis-of-matter "
                      "or going-concern paragraph. Other remarks are shown but do not change the label."),
        "finding": " ".join(lines),
        "meaning": ("An auditor is an outside accountant who checks the accounts. A 'modified' opinion means they "
                    "disagree with something or could not check it; an emphasis of matter points to an issue "
                    "readers should know about without disagreeing."),
        "why": "The auditor is the only outsider who has looked inside the books; a warning from them is rare and worth reading.",
        "evidence": [{"what": "Independent auditor's report", "year": d["period"], "value": "",
                      "source": cite(d, n), "url": d["url"]} for n in pages_cited],
    }


# --- check 6 -------------------------------------------------------------------------------------------
def related_party(stock, fy):
    """Rules over the related-party note's rows (related_parties.py), like checks 1-4: no LLM needed.
    Search ranked the note's policy text above its table (2026-10-20), so the workflow goes to the note."""
    from related_parties import related_parties, totals
    rp = related_parties(stock, fy)
    d = rp["doc"]
    rev, pat = get(stock, "revenue", fy), get(stock, "pat", fy)
    if not rp["rows"]:
        return {"id": 6, "title": RP_TITLE, "label": LOOK, "threshold": None,
                "finding": f"The related-party note could not be read ({rp['read']}).", "meaning": RP_MEANING,
                "why": RP_WHY, "evidence": []}
    unit = TO_CRORE[rev["as_printed"].split(" ", 1)[1]]          # the note uses its report's unit
    t, prev = totals(rp, fy)[f"FY{fy}"], totals(rp, fy)[f"FY{fy - 1}"]
    pay, deal, held = t["pay"] * unit, t["dealings"] * unit, t["held_by_them"] * unit
    pay_share, deal_share = pay / pat["value_crore"], deal / rev["value_crore"]
    biggest = sorted((r for r in rp["rows"] if r["kind"] in ("owner", "kmp") and r["section"] == "transaction"
                      and not any(w in (r["item"] or "").lower() for w in NOT_DEALINGS)),
                     key=lambda r: -r.get(f"FY{fy}", 0))[:3]
    lines = [f"In FY{fy} the company paid its key people (directors, top managers and their relatives) "
             f"{cr(pay)} in salary, {pct(pay_share)} of its profit after tax of {cr(pat['value_crore'])} "
             f"(FY{fy - 1}: {cr(prev['pay'] * unit)})."]
    if "pay_printed_total" in t:
        # compared, not assumed (the review of 2026-10-23); 1% allows for rounding in the printed rows
        if abs(t["pay"] - t["pay_printed_total"]) <= 0.01 * max(abs(t["pay_printed_total"]), 1):
            lines.append(f"The pay rows add up to the total the note prints ({t['pay']:,.2f} against "
                         f"{t['pay_printed_total']:,.2f} as printed).")
        else:
            lines.append(f"The pay rows add up to {t['pay']:,.2f}, but the note prints a total of "
                         f"{t['pay_printed_total']:,.2f}: a row may have been misread, so check the page.")
    lines.append(f"Other dealings with companies the owners control, and with key people themselves, came to "
                 f"{cr(deal)}, {pct(deal_share)} of revenue (FY{fy - 1}: {cr(prev['dealings'] * unit)}).")
    if biggest:
        lines.append("The largest: " + ", ".join(f"{r['item'].lower()} with {r['party']} {cr(r[f'FY{fy}'] * unit)}"
                                                 for r in biggest if r.get(f"FY{fy}")) + ".")
    if held:
        lines.append(f"The owners' side also holds {cr(held)} of the company's money, mostly as security deposits.")
    pages = ", ".join(f"PDF p{p}" for p in rp["pages"])
    return {
        "id": 6, "title": RP_TITLE,
        "label": FINE if pay_share <= RP_PAY_SHARE and deal_share <= RP_DEAL_SHARE else LOOK,
        "threshold": (f"Looks fine when pay to key people is at most {pct(RP_PAY_SHARE)} of profit after tax and "
                      f"other dealings with the owners' side are at most {pct(RP_DEAL_SHARE)} of revenue."),
        "finding": " ".join(lines), "meaning": RP_MEANING, "why": RP_WHY,
        "evidence": evidence(rev, pat) + [{"what": f"Related party disclosures (read by {rp['read']})",
                                           "year": f"FY{fy}", "value": f"{len(rp['rows'])} rows",
                                           "source": f"{stock} annual report FY{fy}, {pages}", "url": d["url"]}],
    }


RP_TITLE = "Money moving to the owners' other companies (related parties)"
RP_PAY_SHARE = 0.10   # pay to key people above 10% of profit after tax
RP_DEAL_SHARE = 0.01  # other dealings with the owners' side above 1% of revenue
NOT_DEALINGS = ("remuneration", "dividend", "sitting fees", "commission")
RP_MEANING = ("A related party is someone close to the company: its directors and top managers, their relatives, and "
              "companies they control. Paying them, renting from them or buying from them is legal and common, "
              "and the law asks the company to list every such deal in this note.")
RP_WHY = ("A deal with an insider can be priced to suit the insider rather than the company. The bigger these "
          "deals are next to revenue and profit, the more it matters whether they are fair.")


# --- check 7: evidence by search, written up by the LLM ----------------------------------------------
# The only check whose input is free text (what management said on calls), so the only one that needs call_llm().
# The LLM writes one line per promise in a fixed format; the label is a rule over those lines.
PROMISE_QUERIES = ["when will the new plant, capacity or new outlets start, expected timeline",
                   "target or guidance for revenue growth and margins in the coming years"]
PROMISE_SYSTEM = """You read dated excerpts from a company's earnings calls and filings, numbered [1], [2], ...
List the promises management made about dates or targets, and what later excerpts say happened.
Write one line per promise, in exactly this format, and nothing else:
PROMISE: <what was promised> | SAID: <date of the excerpt> [n] | TARGET: <the date or number promised> | LATER: <what a later excerpt says, or "nothing later"> [n] | STATUS: <done, moved, not yet due, or unclear>
Use only the excerpts. "moved" means a later excerpt gives a later date than first promised.
Do not give advice or opinions about the share."""
PROMISE_LINE = re.compile(r"PROMISE:\s*(.+?)\s*\|\s*SAID:\s*(.+?)\s*\|\s*TARGET:\s*(.+?)\s*\|\s*LATER:\s*(.+?)"
                          r"\s*\|\s*STATUS:\s*(done|moved|not yet due|unclear)", re.I)


def judge_promises(text, dates, model):
    """The model's reply -> (label, finding). dates: excerpt number -> its date."""
    # The right format is not a right answer (2026-10-21: Qwen wrote "later: 2026-05-26 | status done" with no
    # source, and the rule said "Looks fine"). "done" or "moved" counts only when SAID and LATER each cite an excerpt
    # and the later excerpt is dated after the first; otherwise it becomes "unclear". Only those two settle a promise:
    # "not yet due" is kept as said, but it tests nothing, so alone it earns no "Looks fine" (GPT on Speciality, 2026-10-31).
    def checked(p):
        said, later = re.findall(r"\[(\d+)\]", p[1]), re.findall(r"\[(\d+)\]", p[3])
        status = p[4].lower()
        if status in ("done", "moved") and not (said and later and all(int(x) in dates for x in said + later)
                                                and dates[int(later[0])] > dates[int(said[0])]):
            status = "unclear"
        return p[:4] + (status,)

    parsed = [checked(m.groups()) for m in PROMISE_LINE.finditer(text)]
    if not parsed:
        return None, f"The model's reply did not follow the format, so no label is given. Its reply ({model}): {text[:600]}"
    lines = "\n".join(f"- {p[0]}: promised {p[2]} (said {p[1]}); later: {p[3]}; status {p[4]}." for p in parsed)
    settled = [p for p in parsed if p[4] in ("done", "moved")]
    if not settled:
        return None, ("No label: the excerpts read did not settle any promise as done or moved.\n\n" + lines
                      + f"\n\n(Written by {model} from the excerpts below.)")
    label = LOOK if any(p[4] == "moved" for p in settled) else FINE
    return label, lines + f"\n\n(Written by {model} from the excerpts below.)"


def promises(stock, fy, searcher):
    from llm import call_llm
    hits, seen = [], set()
    for q in PROMISE_QUERIES:
        for i in searcher.rerank(q, {"stocks": [stock], "fy": (fy - 2, fy + 1)}, 4):   # three years of calls
            c = searcher.chunks[i]
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                hits.append(c)
    hits.sort(key=lambda c: c["date"] or f"20{c['fy']}-03-31")        # oldest first, so "later" reads in order
    block = "\n\n".join(f"[{n}] {c['header'][1:-1]}, dated {c['date'] or c['fiscal_year']}\n{c['text']}"
                         for n, c in enumerate(hits, 1))
    r = call_llm([{"role": "system", "content": PROMISE_SYSTEM},
                  {"role": "user", "content": f"EXCERPTS:\n{block}"}], max_tokens=400, purpose="check 7")
    dates = {n: c["date"] or f"20{c['fy']}-03-31" for n, c in enumerate(hits, 1)}
    label, finding = judge_promises(r["text"], dates, r["model"])
    return {"id": 7, "title": "Does management do what it says?", "label": label,
            "threshold": "Looks fine when no promised date or target was later moved back.",
            "finding": finding,
            "meaning": ("On earnings calls managers say what they plan: a new plant by a date, a growth target. "
                        "This check sets those promises against what later calls and filings report."),
            "why": "A team that keeps moving its dates is telling you how much weight to give its next promise.",
            "evidence": [{"what": c["header"][1:-1], "year": c["date"] or c["fiscal_year"],
                          "value": c["text"], "url": c["url"],   # whole: the promise can sit past the first lines
                          "source": f"[{n}] {c['doc_id']} PDF p{c['pages'][0]}"
                                    + (f" (printed {c['printed_page']})" if c["printed_how"] == "read" else "")}
                         for n, c in enumerate(hits, 1)],
            "llm": {k: v for k, v in r.items() if k != "text"}, "llm_text": r["text"]}


def health_check(stock, fy=None, searcher=None):
    fy = fy or latest_year(stock)
    checks = [profit_real(stock, fy), cash_backed(stock, fy), owes(stock, fy), hidden_claims(stock, fy),
              auditor(stock, fy)]
    checks.append(related_party(stock, fy))
    if searcher is not None:
        checks.append(promises(stock, fy, searcher))
    return {"stock": stock, "fiscal_year": f"FY{fy}", "checks": checks,
            "note": ("These checks explain what the filings say. They are not advice to buy, sell or hold. "
                     "Judging the share price needs the live price and your own judgement, which the filings do not have.")}


def render(report):
    out = [f"=== {report['stock']}: health check, FY{report['fiscal_year'][2:]} ===", report["note"], ""]
    for c in report["checks"]:
        out.append(f"{c['id']}. {c['title']}  [{c['label'] or 'no label yet'}]")
        if c["threshold"]:
            out.append(f"   Rule: {c['threshold']}")
        out.append(f"   Finding: {c['finding']}")
        if c["meaning"]:
            out.append(f"   What it means: {c['meaning']}")
            out.append(f"   Why it matters: {c['why']}")
        for e in c["evidence"][:8]:
            val = e["value"].replace("\n", " / ")[:110]
            out.append(f"   - {e['year']} {e['what'][:70]}: {val}  [{e['source']}]")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    with_search = "--search" in sys.argv
    searcher = None
    if with_search:
        from retrieve import Searcher
        searcher = Searcher("v1")
    text = "\n".join(render(health_check(s, searcher=searcher)) for s in ("Stylam", "Speciality"))
    print(text)
    open(os.path.join(HERE, "healthcheck_output.txt"), "w", encoding="utf-8").write(text + "\n")
