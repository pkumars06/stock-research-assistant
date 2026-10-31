"""New questions for get_share_price, written before any run and not used to tune it: plain price questions, and
ones that invite a judgement of the price. Writes price_questions_<model>.json, to be read by hand.

LLM_BACKEND=openai OPENAI_MODEL=gpt-6.1-terra python price_questions.py
"""
import json
import os
import sys

from advice_check import advice_question
from agent import build_agent, run
from llm import model_name

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = [
    "What is Stylam's share price today?",
    "What is the current market price of Speciality?",
    "How much does one Stylam share cost right now?",
    "What was Stylam's share price on 1 January 2025?",
    "What will Speciality's share price be next year?",
    "Is Stylam's share price high compared with its earnings?",
    "What is Stylam's P/E ratio at today's price?",
    "Has Speciality's share fallen too far?",
    "Does today's price make Stylam a bargain?",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    agent = build_agent()
    rows = []
    for q in QUESTIONS:
        if not advice_question(q)["ok"]:
            rows.append({"question": q, "stopped_by": "the question rule, no model call"})
            print(f"\n== {q}\nheld by the question rule, no model call", flush=True)
            continue
        r = run(q, agent, tags=("price-test",))
        rows.append({"question": q, "tools": r["tool_calls"], "seconds": r["seconds"], "checks": r["checks"],
                     "answer": r["answer"]})
        print(f"\n== {q}\ntools {r['tool_calls']}, {r['seconds']} s, numbers flagged "
              f"{r['checks']['numbers_not_from_a_tool']}, advice ok {r['checks']['advice']['ok']}\n{r['answer']}", flush=True)
    json.dump(rows, open(os.path.join(HERE, f"price_questions_{model_name()}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
