"""Run the agent on every golden question and count what it did. The "before the key" picture.

Per question: the tools asked for, whether the first tool fits the question type, the stop reason, model calls,
prompt tokens, seconds, the two checks, and a crude answer score: does the answer contain a figure from the
expected answer? (Numbers of 3+ digits, commas removed; rounding and lakh against crore count, see same_figure.)
The crude score is a first look only. Answers are marked by hand, or by a judge checked against my marks, in step 5.

Expected first tool, by type: table and calculation -> lookup_financials; fact and multi-hop -> search_filings;
not_in_filings -> either is fine, and the answer must say "Insufficient evidence".

python eval_agent.py [type]     all questions, or one type ("table") (local Qwen: about 40-60 s each)
Writes agent_results.csv and agent_summary.txt; each run's trace is in .store/agent_runs/.
"""
import csv
import os
import sys
from collections import defaultdict

from agent import MODEL_LIMIT, TOOL_LIMIT, build_agent, run
from answer import numbers
from llm import BACKEND, model_name

HERE = os.path.dirname(os.path.abspath(__file__))
# compare_years (added 2026-10-23, after the Qwen run) does the lookups and the growth sum in one call, so it fits
# a calculation question too
FIRST_TOOL = {"table": {"lookup_financials"}, "calculation": {"lookup_financials", "compare_years"},
              "fact": {"search_filings"}, "multi-hop": {"search_filings"}}


def expected_numbers(text):
    return {x for x in numbers(text) if len(x.replace(".", "")) >= 3}


def same_figure(a, b):
    """Two written numbers name the same figure: equal, or one is the other rounded, also across lakh and crore
    (100 lakh = 1 crore). Exact string matching scored "₹221.45 crores" against the key's "221" as a miss
    (S10; the review of 2026-10-23)."""
    def dec(x):
        return len(x.split(".")[1]) if "." in x else 0
    x, y = float(a), float(b)
    if abs(x - y) <= 0.5 * 10 ** -min(dec(a), dec(b)):      # 221 and 221.45: the less precise one's last digit
        return True
    return any(abs(x - y * scale) <= 0.005 * abs(x) for scale in (100, 0.01))   # lakh against crore, within 0.5%


def has_figure(expected, got):
    return any(same_figure(e, g) for e in expected for g in got)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    golden = list(csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig")))
    only = sys.argv[1] if len(sys.argv) > 1 else ""          # a question type, for example "table"
    golden = [g for g in golden if not only or g["type"] == only]
    suffix = f"_{only}" if only else ""
    if BACKEND != "qwen":                                      # keep Qwen's "before the key" files as they are
        suffix += "_" + model_name()
    agent = build_agent()
    rows = []
    for g in golden:
        r = run(g["question"], agent, tags=("golden", g["id"]))
        first = r["tool_calls"][0] if r["tool_calls"] else ""
        want = FIRST_TOOL.get(g["type"])
        exp = expected_numbers(g["expected_answer"])
        got = numbers(r["answer"])
        row = {"id": g["id"], "type": g["type"], "tools": " ".join(r["tool_calls"]), "first_tool": first,
               "first_tool_fits": (first in want) if want else bool(first),
               "stopped_by": r["stopped_by"], "model_calls": r["model_calls"], "prompt_tokens": r["prompt_tokens"],
               "output_tokens": sum(s.get("output_tokens") or 0 for s in r["steps"] if s["kind"] == "model"),
               "seconds": r["seconds"], "has_expected_figure": has_figure(exp, got) if exp else "",
               "says_insufficient": "insufficient evidence" in r["answer"].lower(),
               "numbers_not_from_a_tool": " ".join(r["checks"]["numbers_not_from_a_tool"]),
               "advice_ok": r["checks"]["advice"]["ok"], "run_id": r["run_id"], "answer": r["answer"][:500]}
        rows.append(row)
        print(f"{g['id']} {g['type']:<14} tools={row['tools'] or '-':<45} fits={row['first_tool_fits']!s:<5} "
              f"figure={row['has_expected_figure']!s:<5} {r['seconds']} s", flush=True)
    with open(os.path.join(HERE, f"agent_results{suffix}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"Agent on {len(rows)} golden questions, model {r['model']}, limits: "
             f"{MODEL_LIMIT} model calls, {TOOL_LIMIT} tool calls"]
    by = defaultdict(list)
    for row in rows:
        by[row["type"]].append(row)
    lines.append(f"{'type':<15}{'n':>3}{'any tool':>10}{'first fits':>12}{'figure':>9}{'calls':>7}{'tokens':>8}{'s':>7}")
    for t, rs in sorted(by.items()) + [("all", rows)]:
        n = len(rs)
        scored = [x for x in rs if x["has_expected_figure"] != ""]
        lines.append(f"{t:<15}{n:>3}{sum(bool(x['tools']) for x in rs):>10}"
                     f"{sum(x['first_tool_fits'] is True for x in rs):>12}"
                     f"{(str(sum(x['has_expected_figure'] is True for x in scored)) + '/' + str(len(scored))):>9}"
                     f"{sum(x['model_calls'] for x in rs) / n:>7.1f}{sum(x['prompt_tokens'] for x in rs) / n:>8.0f}"
                     f"{sum(x['seconds'] for x in rs) / n:>7.0f}")
    stops = defaultdict(int)
    for row in rows:
        stops[row["stopped_by"]] += 1
    lines.append("stopped by: " + ", ".join(f"{k} {v}" for k, v in stops.items()))
    lines.append(f"numbers not from a tool: {sum(bool(x['numbers_not_from_a_tool']) for x in rows)} answers; "
                 f"advice check failed: {sum(not x['advice_ok'] for x in rows)}")
    open(os.path.join(HERE, f"agent_summary{suffix}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
