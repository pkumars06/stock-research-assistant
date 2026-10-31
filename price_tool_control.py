"""Did adding get_share_price make these answers worse, or is it run-to-run variation? The 40-question rerun of
2026-10-31 lost S06, B01 and B07 on the figure check and flagged a number on B08, and never called the new tool. The
same questions, several times each, with and without the tool, on the same code otherwise.

LLM_BACKEND=openai OPENAI_MODEL=gpt-6.1-terra python price_tool_control.py
"""
import csv
import json
import os
import sys

import agent as a
from answer import numbers
from eval_agent import expected_numbers, has_figure

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = ["S06", "B01", "B07", "B08"]
REPEATS = 3
PRICE_LINES = ("- The latest share price: call get_share_price. Give the price, exchange, time and source, and say it "
               "is not from the\n  filings. Only report it: never call it high or low, and never compare it with "
               "earnings or any other figure.\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    golden = {g["id"]: g for g in csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig"))}
    with_tool = (a.TOOLS, a.SYSTEM)
    assert PRICE_LINES in a.SYSTEM
    without = ([t for t in a.TOOLS if t.name != "get_share_price"], a.SYSTEM.replace(PRICE_LINES, ""))
    rows = []
    for label, (tools, system) in (("without", without), ("with", with_tool)):
        a.TOOLS, a.SYSTEM = tools, system
        agent = a.build_agent()
        for qid in IDS:
            g = golden[qid]
            for k in range(REPEATS):
                r = a.run(g["question"], agent, tags=("golden", qid, "price-control"))
                row = {"config": label, "id": qid, "try": k,
                       "figure": has_figure(expected_numbers(g["expected_answer"]), numbers(r["answer"])),
                       "flagged": r["checks"]["numbers_not_from_a_tool"], "tools": r["tool_calls"],
                       "tokens_in": r["prompt_tokens"], "answer": r["answer"]}
                rows.append(row)
                print(label, qid, k, "figure", row["figure"], "flagged", row["flagged"], len(row["tools"]), "tools",
                      flush=True)
    print()
    for label in ("without", "with"):
        for qid in IDS:
            s = [x for x in rows if x["config"] == label and x["id"] == qid]
            print(label, qid, "figure", sum(bool(x["figure"]) for x in s), "of", len(s), "flagged",
                  sum(bool(x["flagged"]) for x in s))
    print("tokens in", sum(x["tokens_in"] for x in rows))
    json.dump(rows, open(os.path.join(HERE, "price_tool_control.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=str)


if __name__ == "__main__":
    main()
