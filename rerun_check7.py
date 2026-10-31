"""Rerun only check 7 with the demo's model and keep the result beside the UI data, for reading before it is used.

Check 7 is the only stored text a model wrote: checks 1-6, the overview, growth and events are plain code and stay
as they were. A full build_ui.py would rebuild those too, for no gain, one day before the demo.

OPENAI_MODEL=gpt-6.1-terra python rerun_check7.py          writes .store/ui/check7_new.json
python rerun_check7.py --rejudge                           the current rule on those stored replies, no model call
python rerun_check7.py --apply                             puts that file into ui_data.json (keeps a backup)
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, ".store", "ui", "ui_data.json")
NEW = os.path.join(HERE, ".store", "ui", "check7_new.json")
BACKUP = os.path.join(HERE, ".store", "ui", "ui_data_before_check7.json")
STOCKS = ["Stylam", "Speciality"]


def run():
    from healthcheck import latest_year, promises
    from llm import BACKEND, OPENAI_MODEL
    from retrieve import Searcher
    if BACKEND != "openai" or not OPENAI_MODEL.startswith("gpt-6"):
        sys.exit(f"stopped: backend {BACKEND}, model {OPENAI_MODEL}; this rerun is only for the demo's GPT model")
    searcher = Searcher("v1")
    out = {"built": time.strftime("%Y-%m-%d %H:%M")}
    for stock in STOCKS:
        out[stock] = promises(stock, latest_year(stock), searcher)
        c = out[stock]
        print(f"\n=== {stock}: label {c['label']}; {c['llm']['model']}, "
              f"{c['llm']['prompt_tokens']} in, {c['llm']['output_tokens']} out, {c['llm']['seconds']} s")
        print(c["llm_text"])
        print("--- finding as shown:\n" + c["finding"])
    json.dump(out, open(NEW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved", NEW)


def rejudge():
    """Apply the current rule to the stored replies, with no model call, so replies already read stay the same."""
    from healthcheck import judge_promises
    new = json.load(open(NEW, encoding="utf-8"))
    for stock in STOCKS:
        c = new[stock]
        dates = {int(e["source"][1:e["source"].index("]")]): e["year"] for e in c["evidence"]}
        c["label"], c["finding"] = judge_promises(c["llm_text"], dates, c["llm"]["model"])
        print(f"\n=== {stock}: label {c['label']}\n{c['finding']}")
    json.dump(new, open(NEW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def apply():
    new = json.load(open(NEW, encoding="utf-8"))
    data = json.load(open(UI, encoding="utf-8"))
    shutil.copy(UI, BACKUP)
    for stock in STOCKS:
        checks = data[stock]["health"]["checks"]
        checks[[c["id"] for c in checks].index(7)] = new[stock]
    data["built"] = f"{data['built']}; check 7 on {new['built']}"
    data["llm_model"] = new[STOCKS[0]]["llm"]["model"]
    json.dump(data, open(UI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("updated", UI, "backup", BACKUP)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    apply() if "--apply" in sys.argv else rejudge() if "--rejudge" in sys.argv else run()
