"""The no-advice check: does an answer tell the reader what to do with the share? Rules first, no LLM.

The line (capstone-plan.md, "Who it is for"): explain, don't advise. Advice to buy or sell needs SEBI
registration, so an answer must never say buy / sell / hold, give a target price, or call the share
cheap or expensive. Facts about the company are fine, even when they use the same words:
"Stylam sells laminates", "promoters hold 46%", "the board recommended a dividend".

So the rules look for phrases, not single words. Rules miss advice worded in a new way ("looks cheap at
these levels"), which is the case for an LLM judge later. advice_cases.csv holds labelled answers, and
`python advice_check.py` measures the rules on them: TPR (advice caught) and TNR (fine answers passed),
the same two numbers as for any judge (guide 2 Part 5).
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ACTION = r"(?:buy|sell|hold|accumulate|exit|avoid|add|book profits?|invest)(?:ing)?"
RULES = [
    ("action advice", rf"\b(?:you|investors?|one)\s+(?:should|could|may want to|might want to|can|must|need to)\s+(?:consider\s+)?{ACTION}\b"),
    ("action advice", rf"\b(?:i|we)\s+(?:would|recommend|suggest)\s+(?:you\s+)?(?:to\s+)?{ACTION}"),
    ("rating", r"\b(?:strong\s+)?(?:buy|sell|hold|accumulate)\s+(?:rating|recommendation|call|signal)\b"),
    ("rating", r"\b(?:is|as)\s+an?\s+(?:strong\s+)?(?:buy|sell|hold)\b"),
    ("rating", r"\bstrong\s+(?:buy|sell)\b"),
    ("recommend buying", r"\brecommend(?:ed|s)?\s+(?:buying|selling|holding|investing|the\s+(?:stock|share))"),
    ("target price", r"\b(?:target|fair)\s+(?:price|value)\b|\bprice\s+target\b"),
    ("valuation call", r"\b(?:under|over)[-\s]?valued\b|\bfairly\s+valued\b|\bmultibagger\b"),
    ("valuation call", r"\b(?:stock|share|shares)\s+(?:is|are|looks?|seems?)\s+(?:cheap|expensive|attractive|a bargain)\b"),
    ("good investment", r"\b(?:good|great|safe|bad|poor)\s+(?:investment|stock\s+to|buy|bet)\b|\bworth\s+(?:buying|investing|holding)\b"),
    # "at current prices" alone is also a fact ("costs stayed high at current prices", the review of 2026-10-23),
    # so it counts only after an action or a valuation word in the same sentence
    ("add on dips", rf"\b(?:on|at)\s+dips\b|\b(?:{ACTION}|cheap|expensive|attractive|bargain)\b[^.]{{0,40}}?"
                    r"\bat\s+(?:these|current)\s+(?:levels|prices)\b"),
]


def advice_check(text):
    """{"ok": True} or {"ok": False, "found": [(rule, matched text), ...]}."""
    found = []
    for name, pattern in RULES:
        for m in re.finditer(pattern, text or "", re.I):
            found.append((name, m.group()))
    return {"ok": not found, "found": found}


# The same line on the way in (guardrails "in", guide 2 Part 10): a question that asks what to do with the
# share gets a fixed reply from code and never reaches the model, so the safe reply does not depend on luck.
# Facts in the same words stay questions: "did promoters sell shares?", "what target did management give?".
SHARE = r"(?:the\s+)?(?:stock|shares?|stylam|speciality|it|this\s+company)"
QUESTION_RULES = [
    ("what to do", rf"\b(?:should|shall|can|must)\s+(?:i|we|one|investors?)\s+(?:still\s+|now\s+)?(?:{ACTION}|get\s+out|enter|stay)\b"),
    ("what to do", rf"\b(?:is\s+it|is\s+now)\s+(?:a\s+)?(?:good|right|bad|safe)\s+(?:time\s+)?(?:to\s+)?(?:{ACTION}|enter|investment)"),
    ("what to do", r"\b(?:buy|sell|hold)\s+or\s+(?:sell|hold|buy|not)\b"),
    ("worth it", rf"\b(?:is|are)\s+{SHARE}\s+(?:a\s+)?(?:good|safe|great|bad)\s+(?:investment|buy|stock|bet)\b"),
    ("worth it", r"\bworth\s+(?:buying|investing|holding)\b|\bwhich\s+(?:one\s+|stock\s+|share\s+)?(?:is\s+better|should\s+i)\b"),
    ("price", r"\b(?:target|fair)\s+(?:share\s+)?price\b|\bprice\s+target\b"),
    ("price", r"\b(?:under|over)[-\s]?valued\b|\b(?:cheap|expensive)\s+(?:now|at)\b|\bmultibagger\b"),
    ("price", rf"\bwill\s+{SHARE}\s+(?:share\s+)?(?:price\s+)?(?:go\s+up|go\s+down|rise|fall|double|crash)\b|\bshare\s+price\s+(?:go|rise|fall)\b"),
]
ADVICE_REPLY = ("I can't tell you whether to buy, sell or hold a share, or what its price should be. That is advice, "
                "and in India it needs SEBI registration. What I can do is show what the filings say: try the "
                "**Health check** tab, or ask about a figure, an event or a promise, for example \"How much cash did "
                "Stylam bring in from operations in FY26?\"")


def advice_question(text):
    """{"ok": True} or {"ok": False, "found": [(rule, matched text), ...]} for a user's question."""
    found = [(name, m.group()) for name, pattern in QUESTION_RULES for m in re.finditer(pattern, text or "", re.I)]
    return {"ok": not found, "found": found}


def measure_questions(path=os.path.join(HERE, "advice_questions.csv")):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tp = fn = tn = fp = 0
    for r in rows:
        flagged = not advice_question(r["question"])["ok"]
        advice = r["label"] == "advice"
        tp += advice and flagged
        fn += advice and not flagged
        tn += (not advice) and (not flagged)
        fp += (not advice) and flagged
        if flagged != advice:
            print(f"  WRONG {r['id']} ({r['label']}, flagged={flagged}): {r['question']}")
    print(f"{len(rows)} questions: {tp + fn} asking for advice, {tn + fp} fine")
    print(f"TPR (advice questions caught) {tp}/{tp + fn} = {tp / (tp + fn):.0%}")
    print(f"TNR (fine questions passed)   {tn}/{tn + fp} = {tn / (tn + fp):.0%}")


def measure(path=os.path.join(HERE, "advice_cases.csv")):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tp = fn = tn = fp = 0
    for r in rows:
        flagged = not advice_check(r["answer"])["ok"]
        advice = r["label"] == "advice"
        tp += advice and flagged
        fn += advice and not flagged
        tn += (not advice) and (not flagged)
        fp += (not advice) and flagged
        if flagged != advice:
            print(f"  WRONG {r['id']} ({r['label']}, flagged={flagged}): {r['answer']}")
    print(f"{len(rows)} answers: {tp + fn} advice, {tn + fp} fine")
    print(f"TPR (advice caught)      {tp}/{tp + fn} = {tp / (tp + fn):.0%}")
    print(f"TNR (fine answers passed) {tn}/{tn + fp} = {tn / (tn + fp):.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(advice_check(" ".join(sys.argv[1:])))
    else:
        measure()
        print()
        measure_questions()
