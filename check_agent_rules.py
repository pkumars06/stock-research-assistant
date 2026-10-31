"""Tests for the plain-code rules around the agent (no LLM): metric names, stock names, missing rows, the number
and percentage checks. Each case is one that went wrong once (2026-10-22 and 2026-10-23, and the review that day).

python check_agent_rules.py     prints each failure, then the count
"""
import sys

import json

import healthcheck
import price
from agent import (PLAIN, UNITS, compare_years, lookup_financials_tool, metrics_looked_up, metrics_named, percents,
                   resolve_metric, stock_name, tool_percents, unmatched, wrong_year)
from retrieve import named_years, parse_query

METRICS = [
    # names the model asked for in past runs
    ("net_cash_from_operations", "cash_from_operations"), ("net_cash_flow_from_operating_activities", "cash_from_operations"),
    ("operating_cash_flow", "cash_from_operations"), ("cash", "cash_balance"), ("eps_basic", "eps"),
    ("rent_paid_on_leased_premises", "rent_paid"), ("debt", "borrowings"), ("sales", "revenue"),
    ("exceptional", "exceptional_items"), ("bank balances other than cash and cash equivalents", "bank_deposits"),
    ("dine_in_sales", None), ("capex", None), ("exports", None), ("new_outlets", None), ("pay_to_companies", None),
    # a different figure is refused, not matched to a near one (the review of 2026-10-23)
    ("other_comprehensive_income", None), ("rental_income", None), ("interest_on_lease_liabilities", None),
    ("equity_share_capital", None), ("loans_and_advances", None), ("security_deposits", None),
    ("debt_to_equity", None), ("net_profit_margin", None), ("short_term_borrowings", None),
    ("deferred_tax", None), ("revenue_growth", None),
    # an abbreviation is a whole word, not the start of one (the fourth review)
    ("patents", None), ("epsilon", None), ("pat", "profit_after_tax"),
    # the real names still work
    *[(k, k) for k in PLAIN],
]

NUMBERS = [   # (answer numbers, tool numbers, flagged)
    ({"149"}, {"149.47"}, []),                 # rounded
    ({"2026"}, set(), []),                     # a year
    ({"104.2"}, {"10.2"}, ["104.2"]),
    ({"10"}, {"10.2"}, []),                    # "10%" for 10.2%
    ({"80000000"}, {"149.47"}, ["80000000"]),
]

# The Sonnet run of 2026-10-27: 9 of 40 answers flagged, all right. (answer numbers, tool numbers, every number in
# the answer, flagged), checked with the units the app uses. Each pass has a made-up number beside it that must fail.
NUMBERS_SONNET = [
    ({"28.10"}, {"28.1"}, set(), []),                                       # P15: a trailing zero
    ({"3.34"}, {"334.38"}, set(), []),                                      # P01: lakh -> crore
    ({"3.5"}, {"334.38"}, set(), ["3.5"]),                                  #   not 334.38 lakh in any unit
    ({"14.5"}, {"145009700"}, set(), []),                                   # P02: rupees -> crore
    ({"4.49"}, {"44.94"}, set(), []),                                       # P10: ₹ million -> crore
    ({"1064.5", "651"}, {"106451.25", "65109.42"}, set(), []),              # S07: lakh -> crore
    ({"2.56"}, {"72.55", "69.99"}, {"72.55", "69.99", "2.56"}, []),         # P07: a difference the answer shows
    ({"2.57"}, {"72.55", "69.99"}, {"72.55", "69.99", "2.57"}, ["2.57"]),   #   a wrong difference
    ({"2.55"}, {"72.55"}, {"72.55", "70.00", "2.55"}, ["2.55"]),            #   70.00 came from no tool
    ({"2.56"}, {"72.55", "69.99"}, {"2.56"}, ["2.56"]),                     #   the working is not in the answer
    ({"2378.77"}, {"6504.04", "4125.27"}, {"6504.04", "4125.27", "2378.77"}, []),   # S12
    ({"33"}, {"8", "15", "10"}, {"8", "15", "10", "33"}, []),               # P09: a sum of three
    ({"0.94"}, {"65109.42", "696"}, {"651", "696", "0.94"}, []),            # S07: a ratio, 651 from lakh
    ({"16"}, {"106451.25", "803"}, {"1064.5", "803", "1.33", "16"}, ["16"]),    # S07: "16 months" (x12) still flagged
    ({"16"}, {"160.4", "803"}, {"803", "16"}, ["16"]),                      #   not 160.4 / 10: too loose for "16"
    ({"0.09"}, {"9.24"}, set(), []),                                        # P06: 9.24 lakh = 0.09 crore
]


def main():
    failures = 0
    for name, want in METRICS:
        got = resolve_metric(name)
        if got != want:
            failures += 1
            print(f"  metric {name!r}: got {got}, want {want}")
    for name, want in [("Stylam Industries", "Stylam"), ("stylam", "Stylam"), ("Speciality Restaurants", "Speciality")]:
        if stock_name(name) != want:
            failures += 1
            print(f"  stock {name!r}: got {stock_name(name)}")
    # B03: Speciality has no borrowings; the FY26 balance sheet (PDF p139) prints no such line. The tool must give
    # 0 with that page and say why, not refuse (the first review's fix did, the second review caught it)
    r = lookup_financials_tool.invoke({"stock": "Speciality", "metric": "borrowings", "fiscal_year": "FY26"})
    if r.get("value_crore") != 0 or "p139" not in r.get("source", "") or "nil" not in r.get("note", ""):
        failures += 1
        print(f"  B03 Speciality borrowings FY26: got {r}")
    # growth from a nil year has no percent, and says so
    c = compare_years.invoke({"stock": "Stylam", "metric": "cwip", "from_year": "FY22", "to_year": "FY23"})
    if c.get("change_percent") is not None or "note" not in c.get("from", {}):
        failures += 1
        print(f"  Stylam CWIP FY22 -> FY23: got {c}")
    # a figure from a year the question does not name (P12: FY25 asked, FY26 given)
    def looked_up(fy):
        return json.dumps(lookup_financials_tool.invoke({"stock": "Stylam", "metric": "borrowings", "fiscal_year": fy}))
    for question, years, want in [("What were Stylam's borrowings in FY25?", ["FY26"], [26]),
                                  ("What were Stylam's borrowings in FY25?", ["FY25"], []),
                                  ("What are Stylam's borrowings?", ["FY26"], []),
                                  ("How has Stylam's debt moved since FY22?", ["FY22", "FY24", "FY26"], []),
                                  ("How has Stylam's debt moved since FY22?", ["FY25", "FY26"], [25, 26])]:
        got = wrong_year(question, [looked_up(fy) for fy in years])
        if got != want:
            failures += 1
            print(f"  year check {question!r} with {years}: got {got}, want {want}")
    for found, given, want in NUMBERS:
        if unmatched(found, given) != want:
            failures += 1
            print(f"  number check {found} against {given}: got {unmatched(found, given)}, want {want}")
    for found, given, shown, want in NUMBERS_SONNET:
        got = unmatched(found, given, UNITS, shown)
        if got != want:
            failures += 1
            print(f"  number check {found} against {given}, answer {shown}: got {got}, want {want}")
    if percents("fell by 6.4 per cent, then rose 10.2 percent and 3%") != {"6.4", "10.2", "3"}:
        failures += 1
        print("  percents() missed a form")
    # B08: a calculator result that ends "* 100" is a percent; one that does not is not
    calc = [json.dumps({"expression": "53.42 / 202.89 * 100", "result": 26.3295}),
            json.dumps({"expression": "1129.29 - 1025.09", "result": 104.2})]
    for said, want in [("26.3", []), ("104.2", ["104.2"]), ("25.2", ["25.2"])]:
        got = unmatched({said}, tool_percents(calc))
        if got != want:
            failures += 1
            print(f"  percent check {said}% against calculator results: got {got}, want {want}")
    # a bare number is a year only in the filings' span and not before a quantity word; the US spelling counts
    for question, want in [("Did Stylam add 2000 tonnes of capacity?", {"stocks": ["Stylam"], "fy": None}),
                           ("What did Specialty earn in FY26?", {"stocks": ["Speciality"], "fy": (25, 27)}),
                           ("What was revenue in 2025?", {"stocks": [], "fy": (24, 26)}),
                           ("Stylam profit in March 2026", {"stocks": ["Stylam"], "fy": (25, 27)})]:
        if parse_query(question) != want:
            failures += 1
            print(f"  parse_query {question!r}: got {parse_query(question)}, want {want}")
    # the Indian two-year form is the later year (2023-24 = FY24), "FY2023" is FY23 (the second review, 2026-10-23)
    years = [("FY 2023-24 profit", [24]), ("profit in FY23-24", [24]), ("2023-24 revenue", [24]),
             ("FY2025-26", [26]), ("Speciality revenue FY2023", [23]), ("FY 2025 and FY 2026", [25, 26]),
             ("FY22-FY26", [22, 26]), ("2022-26 trend", [22, 26]), ("Q3 FY26", [26]), ("pages 20-21", []),
             ("2099-00", []), ("2000 tonnes", []), ("FY 2019-2020", [20])]
    for question, want in years:
        if sorted(named_years(question)) != want:
            failures += 1
            print(f"  named_years {question!r}: got {sorted(named_years(question))}, want {want}")
    # a right FY24 lookup for "2023-24" must not be held back as another year's figure
    if wrong_year("Stylam revenue in 2023-24?", [json.dumps(lookup_financials_tool.invoke(
            {"stock": "Stylam", "metric": "revenue", "fiscal_year": "FY24"}))]):
        failures += 1
        print("  2023-24 with an FY24 lookup was held back")
    # the metrics a question names, so a skipped one is caught (suggested question 5: lease liabilities said to be
    # nil without a lookup; they are ₹13,994.45 lakh, B03); a quarter is skipped (the table holds years only)
    named = [("How much does Speciality owe in borrowings and lease liabilities?", {"borrowings", "lease_liabilities"}),
             ("How much did Stylam's profit before tax grow in Q1 FY27 compared with Q1 FY26?", set()),
             ("Is it safe to buy equity shares?", set()),
             ("What share of Speciality's dine-in sales came from metros?", set()),
             ("What was Stylam's ratio of operating cash flow to profit after tax in FY26?",
              {"cash_from_operations", "profit_after_tax"}),
             ("Is Speciality debt-free?", {"borrowings"}),
             # everyday words that loose keywords read as figures (the fourth review, 2026-10-23)
             ("What did management say on the earnings call about market share?", set()),
             ("Does Stylam hold any patents?", set()),
             ("What did Stylam say about the pattern of demand?", set()),
             ("What do current investors in Speciality own?", set()),
             ("How much income came from other restaurants?", set()),
             ("Was FY26 growth exceptional?", set()),
             # words beginners type (the fifth review)
             ("What was Stylam's net profit in FY26?", {"profit_after_tax"}),
             ("How much cash did Stylam generate from operations?", {"cash_from_operations"}),
             ("Does Stylam have any loans?", {"borrowings"}),
             ("What are Stylam's loans and advances?", set()),
             ("Did Stylam give loans to employees?", set()),
             ("Has Speciality given loans to its subsidiary, and does it have debt?", {"borrowings"}),
             ("How much cash came from the sale of the plant, not from operations?", set())]
    for question, want in named:
        if metrics_named(question) != want:
            failures += 1
            print(f"  metrics_named {question!r}: got {metrics_named(question)}, want {want}")
    looked = [{"kind": "model", "tool_calls": [{"name": "lookup_financials", "args": {"metric": "borrowings"}}]}]
    if metrics_named(named[0][0]) - metrics_looked_up(looked) != {"lease_liabilities"}:
        failures += 1
        print("  suggested question 5: the skipped lease liabilities were not found")
    r = lookup_financials_tool.invoke({"stock": "Speciality", "metric": "lease_liabilities", "fiscal_year": "FY26"})
    if r.get("value_crore") != 139.94:
        failures += 1
        print(f"  B03 Speciality lease liabilities FY26: got {r}, want 139.94 crore (13,994.45 lakh)")
    # The share price: a failed source moves to the next, and with none left the answer is an error, never a number
    bse = {"Header": {"LTP": "2,747.80", "Ason": "30 Oct 26 | 16:00"}}
    nse = {"chart": {"result": [{"meta": {"regularMarketPrice": 2742.8, "regularMarketTime": 1793354340}}]}}

    def replies(bse_reply, nse_reply):
        def fake(url, headers=None):
            reply = bse_reply if "bseindia" in url else nse_reply
            if isinstance(reply, Exception):
                raise reply
            return reply
        return fake

    down = OSError("blocked")
    garbled = {"Header": {"LTP": "-", "Ason": ""}}
    zero = {"Header": {"LTP": "0.00", "Ason": "30 Oct 26 | 16:00"}}
    prices = [(bse, nse, ("BSE", 2747.8)), (down, nse, ("NSE", 2742.8)), (garbled, nse, ("NSE", 2742.8)),
              ({"Error": "no data"}, nse, ("NSE", 2742.8)), (zero, nse, ("NSE", 2742.8)), (down, down, None),
              (garbled, {"chart": {"result": None}}, None)]
    real_get = price._get_json
    for bse_reply, nse_reply, want in prices:
        price._kept.clear()
        price._get_json = replies(bse_reply, nse_reply)
        r = price.share_price("Stylam")
        got = (r["exchange"], r["price_rupees"]) if "price_rupees" in r else None
        if got != want or (want is None and "error" not in r):
            failures += 1
            print(f"  share price with {bse_reply} and {nse_reply}: got {r}, want {want}")
    price._get_json, _ = real_get, price._kept.clear()
    if "error" not in price.share_price("Tata"):
        failures += 1
        print("  share price of a company the app does not cover did not give an error")
    # check 7: what earns a label (Qwen on 2026-10-21, GPT on 2026-10-31)
    line = "PROMISE: plant | SAID: 2025-05-30 {} | TARGET: Sep 2025 | LATER: {} | STATUS: {}"
    dates = {1: "2025-05-30", 2: "2026-01-23"}
    promise_cases = [
        (line.format("[1]", "March 2026 [2]", "moved"), healthcheck.LOOK, "moved, both cited"),
        (line.format("[1]", "started [2]", "done"), healthcheck.FINE, "done, both cited"),
        (line.format("[1]", "March 2026", "moved"), None, "moved, later not cited"),
        (line.format("[2]", "March 2026 [1]", "done"), None, "later excerpt is older"),
        (line.format("[1]", "started [9]", "done"), None, "cites an excerpt that does not exist"),
        (line.format("[1]", "nothing later", "not yet due"), None, "not yet due alone tests nothing"),
        (line.format("[1]", "a count not given [2]", "unclear"), None, "unclear alone"),
        ("Management promised a plant.", None, "not in the format"),
        (line.format("[1]", "nothing later", "not yet due") + "\n" + line.format("[1]", "March 2026 [2]", "moved"),
         healthcheck.LOOK, "one moved among others"),
    ]
    for text, want, why in promise_cases:
        got = healthcheck.judge_promises(text, dates, "m")[0]
        if got != want:
            failures += 1
            print(f"  check 7, {why}: {got!r}, wanted {want!r}")
    if "status not yet due" not in healthcheck.judge_promises(line.format("[1]", "nothing later", "not yet due"), dates, "m")[1]:
        failures += 1
        print("  check 7 changed the model's 'not yet due'")
    total = (len(METRICS) + 3 + 2 + 5 + len(NUMBERS) + len(NUMBERS_SONNET) + 1 + 3 + 4 + len(years) + 1 + len(named) + 2
             + len(prices) + 1 + len(promise_cases) + 1)
    print(f"{total - failures}/{total} passed")
    return failures


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(1 if main() else 0)
