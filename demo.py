"""See the capstone's working pieces so far, in the order a user would meet them.

1. lookup_financials: a figure and its page, no search.
2. The health check for one stock (checks 1-5 labelled; 6-7 only gather evidence).
3. A search over the filings: the chunks the Ask tab would hand to the LLM.

python demo.py                         all three, Speciality, a sample question
python demo.py Stylam "your question"  another stock or question
No LLM is called yet: everything here is extraction, rules and search.
"""
import sys
import time

from financials import lookup_financials
from healthcheck import health_check, render

stock = sys.argv[1] if len(sys.argv) > 1 else "Speciality"
question = sys.argv[2] if len(sys.argv) > 2 else f"Why did {stock} report an exceptional item in FY26?"

print("=" * 80, "\n1. lookup_financials: one figure and where it came from\n")
for metric, fy in [("pat", "FY26"), ("cfo", "FY26"), ("cfo", "FY22"), ("ebitda", "FY26")]:
    r = lookup_financials(stock, metric, fy)
    if "error" in r:
        print(f"{metric:8} {fy}: error -> {r['error']}\n")
        continue
    value = r["value_crore"] if r["value_crore"] is not None else r["value_per_share"]
    print(f"{r['label']}, {r['fiscal_year']}: {value:,.2f} crore  ({r['as_printed']})")
    print(f"   source: {r['source']}")
    if "restated" in r:
        print(f"   restated: {r['restated']}")
    print()

print("=" * 80, f"\n2. The health check for {stock} (rules only, no LLM)\n")
print(render(health_check(stock)))

print("=" * 80, f"\n3. Search the filings: {question!r}\n")
from retrieve import Searcher, parse_query

t = time.time()
s = Searcher("v1")
print(f"filters read from the question: {parse_query(question, default_recent=True)}")
for rank, i in enumerate(s.search(question, n=5), 1):
    c = s.chunks[i]
    text = " ".join(c["text"].split())[:220]
    print(f"\n#{rank} {c['header']}\n   {text}...")
print(f"\n(search took {time.time() - t:.0f} s, including loading three models)")
