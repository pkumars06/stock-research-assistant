"""lookup_financials: the key figures from the annual reports, each with the page it came from.

The table is built once by build_financials.py (financials.csv). This file only reads it, so the agent's tool
and the health check get a number and its citation without a search. Amounts are in ₹ crore; EPS is ₹ a share.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = {
    "revenue": "Revenue from operations",
    "other_income": "Other income",
    "pbt": "Profit before tax",
    "exceptional": "Exceptional items (effect on profit; a minus means they cut profit)",
    "tax": "Tax expense (profit before tax minus profit for the year)",
    "pat": "Profit for the year (after tax)",
    "eps_basic": "Basic earnings per share",
    "cfo": "Net cash from operating activities (CFO)",
    "lease_payments": "Rent paid on leased premises (shown under financing; a minus is cash out)",
    "cash": "Cash and cash equivalents",
    "other_bank": "Bank balances other than cash and cash equivalents (deposits)",
    "current_investments": "Current investments (mostly mutual funds)",
    "borrowings": "Borrowings (non-current plus current)",
    "lease_liabilities": "Lease liabilities (rent owed on leased premises)",
    "equity": "Total equity (net worth)",
    "cwip": "Capital work in progress (assets being built)",
    "contingent_liabilities": "Contingent liabilities (claims against the company it disputes)",
}
STOCK_NAMES = {"stylam": "Stylam", "speciality": "Speciality", "specialty": "Speciality"}   # also the US spelling

_table = None


def table():
    global _table
    if _table is None:
        _table = {}
        for r in csv.DictReader(open(os.path.join(HERE, "financials.csv"), encoding="utf-8-sig")):
            for k in ("value", "value_crore", "first_reported"):
                r[k] = float(r[k])
            r["fy"] = int(r["fy"])
            _table[(r["stock"], r["fy"], r["basis"], r["metric"])] = r
    return _table


def citation(r):
    printed = f" (printed {r['printed_page']})" if r["printed_page"] else ""
    return f"{r['stock']} annual report {r['report']}, PDF p{r['pdf_page']}{printed}"


def lookup_financials(stock, metric, fiscal_year=None, basis="standalone"):
    """One figure and where it came from.

    stock: "Stylam" or "Speciality". metric: a key of METRICS. fiscal_year: "FY26", 26 or None for the latest.
    basis: "standalone" (the company alone, the default) or "consolidated" (with its subsidiaries).
    Returns a dict, or {"error": ...} that says what is available.
    """
    stock = STOCK_NAMES.get(str(stock).strip().lower(), stock)
    if metric not in METRICS:
        return {"error": f"unknown metric {metric!r}; available: {', '.join(METRICS)}"}
    years = sorted(fy for (s, fy, b, m) in table() if s == stock and b == basis and m == metric)
    if not years:
        return {"error": f"no {metric} for {stock!r} ({basis}); stocks: Stylam, Speciality"}
    if fiscal_year is None:
        fy = years[-1]
    else:
        fy = int(str(fiscal_year).upper().replace("FY", "").strip()) % 100
    r = table().get((stock, fy, basis, metric))
    if r is None:
        return {"error": f"no {metric} for {stock} FY{fy:02}; years available: "
                         + ", ".join(f"FY{y:02}" for y in years)}
    per_share = metric == "eps_basic"
    out = {"stock": stock, "fiscal_year": f"FY{fy:02}", "basis": basis, "metric": metric,
           "label": METRICS[metric], "value_crore": None if per_share else r["value_crore"],
           "value_per_share": r["value"] if per_share else None,
           "as_printed": f"{r['value']:,.2f} {r['unit']}", "source": citation(r), "url": r["url"],
           "row": r["row"], "column": r["column"], "nil_by_absence": r["row"].startswith("(no such row")}
    if abs(r["first_reported"] - r["value_crore"]) > max(0.005 * abs(r["value_crore"]), 0.02):
        out["restated"] = (f"first reported as {r['first_reported']} in the {r['first_report']} report "
                           f"(PDF p{r['first_pdf_page']}); the later report's figure is used")
    return out


def series(stock, metric, basis="standalone"):
    """All years of one metric, oldest first."""
    stock = STOCK_NAMES.get(str(stock).strip().lower(), stock)
    years = sorted(fy for (s, fy, b, m) in table() if s == stock and b == basis and m == metric)
    return [lookup_financials(stock, metric, fy, basis) for fy in years]


if __name__ == "__main__":
    for args in [("Stylam", "cfo", "FY26"), ("speciality", "pbt", 2025), ("Stylam", "cfo", "FY22"),
                 ("Speciality", "contingent_liabilities", None), ("Stylam", "eps_basic", 26), ("Stylam", "ebitda", 26)]:
        print(args, "->", lookup_financials(*args))
