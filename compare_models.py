"""The questions gpt-4o-mini got wrong on 2026-10-31, through the agent on another OpenAI model.

OPENAI_MODEL=gpt-4.1 LLM_BACKEND=openai python compare_models.py [ids...]
Writes model_test_<model>.json: the full answers, to be read against the answer key. Tokens are measured; no price here.
"""
import csv
import json
import os
import sys

from agent import build_agent, run
from llm import model_name

HERE = os.path.dirname(os.path.abspath(__file__))
WRONG_ON_MINI = ["S01", "S06", "S12", "S14", "S15", "P01", "P10", "P14", "B07"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = sys.argv[1:] or WRONG_ON_MINI
    golden = {g["id"]: g for g in csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig"))}
    agent = build_agent()
    rows = []
    for qid in ids:
        g = golden[qid]
        r = run(g["question"], agent, tags=("golden", qid, "model-test"))
        n_out = sum(s.get("output_tokens") or 0 for s in r["steps"] if s["kind"] == "model")
        rows.append({"id": qid, "question": g["question"], "model": model_name(), "tools": r["tool_calls"],
                     "tokens_in": r["prompt_tokens"], "tokens_out": n_out, "seconds": r["seconds"],
                     "stopped_by": r["stopped_by"], "checks": r["checks"], "answer": r["answer"],
                     "expected": g["expected_answer"]})
        print(f"\n== {qid} {len(r['tool_calls'])} tools, {r['prompt_tokens']} in, {n_out} out, {r['seconds']} s",
              flush=True)
        print(r["answer"][:420].replace("\n", " "), flush=True)
    print(f"\nTOTAL {model_name()}: in {sum(x['tokens_in'] for x in rows)}, out {sum(x['tokens_out'] for x in rows)}")
    json.dump(rows, open(os.path.join(HERE, f"model_test_{model_name()}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
