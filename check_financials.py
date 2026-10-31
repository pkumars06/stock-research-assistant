"""Check the lookup_financials table against figures checked by hand in the golden set (2026-10-20).

Each line: stock, year, basis, metric, the figure as printed in the report (in the report's unit), the question.
A figure that fails means the extractor misread a row, not that the report is wrong.
"""
import sys

from financials import lookup_financials, table

EXPECTED = [
    ("Stylam", 26, "standalone", "cfo", 17708.35, "S02, S03"),
    ("Stylam", 25, "standalone", "cfo", 10746.12, "S02"),
    ("Stylam", 26, "standalone", "pat", 14947.35, "S03"),
    ("Stylam", 26, "standalone", "revenue", 112929.39, "S04"),
    ("Stylam", 25, "standalone", "revenue", 102508.95, "S04"),
    ("Stylam", 26, "standalone", "contingent_liabilities", 1561.72, "S05, B04"),
    ("Stylam", 25, "standalone", "contingent_liabilities", 3592.96, "S05"),
    ("Stylam", 26, "standalone", "eps_basic", 88.02, "S11"),
    ("Stylam", 25, "standalone", "eps_basic", 71.43, "S11"),
    ("Stylam", 26, "standalone", "borrowings", 2908.90, "B02"),
    ("Stylam", 26, "standalone", "cash", 6521.04, "B02"),
    ("Stylam", 26, "standalone", "equity", 80679.77, "B04"),
    ("Stylam", 26, "standalone", "pbt", 20289.38, "B08"),
    ("Stylam", 26, "standalone", "cwip", 22144.51, "S10 (₹221 crore)"),
    ("Stylam", 26, "consolidated", "pbt", 20330.53, "page check, Stylam-036 p196"),
    ("Speciality", 26, "standalone", "cfo", 8432.57, "P04, P05, X01"),
    ("Speciality", 26, "standalone", "pat", 2295.00, "P05"),
    ("Speciality", 25, "standalone", "pbt", 2810.16, "P12, P15"),
    ("Speciality", 26, "standalone", "pbt", 2945.23, "P15"),
    ("Speciality", 26, "standalone", "contingent_liabilities", 2899.00, "P06, B05"),
    ("Speciality", 25, "standalone", "contingent_liabilities", 3001.81, "memo error: 30.02 crore, not 15.8"),
    ("Speciality", 26, "standalone", "equity", 34700.44, "B05"),
    ("Speciality", 26, "standalone", "exceptional", -334.38, "P01 (labour code)"),
    ("Speciality", 23, "standalone", "pbt", 622.31, "memo error: 62.2 crore"),
    ("Speciality", 23, "standalone", "pat", 955.78, "memo error: 95.6 crore; P11"),
    ("Speciality", 26, "consolidated", "revenue", 47647.09, "page check, Speciality-066 p192"),
]


def main():
    ok = 0
    for stock, fy, basis, metric, want, why in EXPECTED:
        r = table()[(stock, fy, basis, metric)]
        got = r["value"]
        good = abs(got - want) < 0.006
        ok += good
        print(f"{'ok  ' if good else 'FAIL'} {stock:10} FY{fy} {basis[:4]} {metric:22} want {want:>11,.2f} "
              f"got {got:>11,.2f} {r['unit']:9} p{r['pdf_page']:<4} {why}")
    print(f"\n{ok} of {len(EXPECTED)} match")
    print("\nThe tool's answer for one of them:")
    print(lookup_financials("Speciality", "contingent_liabilities", "FY25"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # the rupee sign does not print in the Windows code page
    main()
