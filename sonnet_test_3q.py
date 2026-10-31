"""Three golden questions through the agent on Claude, to measure real tokens and cost before the full eval.

ANTHROPIC_MODEL=claude-sonnet-5-5 python sonnet_test_3q.py [ids...]     default S01 (fact), S02 (table), S04 (calc)
Writes sonnet_test_3q.json; does not touch agent_results.csv (Qwen's "before the key" table).
Cost uses Sonnet 5.5's list price, $2 in / $10 out per million tokens (thinking is billed as output).
"""
import csv
import json
import os
import sys

from agent import build_agent, run
from answer import numbers
from eval_agent import FIRST_TOOL, expected_numbers, has_figure

HERE = os.path.dirname(os.path.abspath(__file__))
PRICE_IN, PRICE_OUT = 2.0, 10.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = sys.argv[1:] or ["S01", "S02", "S04"]
    golden = {g["id"]: g for g in csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig"))}
    agent = build_agent()
    rows = []
    for qid in ids:
        g = golden[qid]
        r = run(g["question"], agent, tags=("golden", qid, "sonnet-test"))
        n_out = sum(s.get("output_tokens") or 0 for s in r.get("steps", []) if s.get("kind") == "model")
        first = r["tool_calls"][0] if r["tool_calls"] else ""
        exp = expected_numbers(g["expected_answer"])
        row = {"id": qid, "type": g["type"], "model": r["model"], "tools": r["tool_calls"],
               "first_tool_fits": first in FIRST_TOOL.get(g["type"], {first}),
               "has_expected_figure": has_figure(exp, numbers(r["answer"])) if exp else "",
               "model_calls": r["model_calls"], "tokens_in": r["prompt_tokens"], "tokens_out": n_out,
               "cost": round(r["prompt_tokens"] / 1e6 * PRICE_IN + n_out / 1e6 * PRICE_OUT, 4),
               "seconds": r["seconds"], "stopped_by": r["stopped_by"], "checks": r["checks"],
               "answer": r["answer"], "expected": g["expected_answer"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=1, default=str), "\n" + "-" * 60, flush=True)
    cost = sum(x["cost"] for x in rows)
    print(f"total tokens in {sum(x['tokens_in'] for x in rows)}, out {sum(x['tokens_out'] for x in rows)}; "
          f"cost ${cost:.3f}, per question ${cost / len(rows):.3f}, 40 questions about ${cost / len(rows) * 40:.2f}")
    json.dump(rows, open(os.path.join(HERE, "sonnet_test_3q.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=str)


if __name__ == "__main__":
    main()
