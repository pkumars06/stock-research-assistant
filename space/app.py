"""Step 6: the app. Five tabs plus Compare and a glossary, for people who invest through apps but never read filings.

    1 What the company does   passages from the latest annual report, and revenue and profit for every year
    2 Health check            the seven checks: finding, label with its rule, what it means, why it matters, pages
    3 Promises vs delivery    check 7: dated excerpts from calls and filings, oldest first, and the model's lines
    4 Recent events           filings since October 2025, newest first, the opening of each letter
    5 Ask a question          the agent (agent.py), its steps, and the number and advice checks on its answer
    Compare                   the two companies on the same checks
    Words                     the glossary (glossary.py)

Tabs 1-4 and Compare read .store/ui/ui_data.json, made at build time by build_ui.py. Only Ask runs at question
time. The line is "explain, don't advise": an answer that fails the advice check is held back, not shown.

python app.py     local, http://127.0.0.1:7860 ; APP_PASSWORD set -> login "demo" / that password
"""
import json
import os
import sys
import threading
import time

import gradio as gr

from advice_check import ADVICE_REPLY, advice_question
from glossary import GLOSSARY
from price import share_price
from retrieve import named_years, parse_query

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, ".store", "ui", "ui_data.json"), encoding="utf-8"))
STOCKS = DATA["stocks"]
FULL_NAME = {"Stylam": "Stylam Industries", "Speciality": "Speciality Restaurants"}
QUESTIONS_PER_SESSION = 10   # a password decides who; this decides how much (the key pays per call)

NOT_ADVICE = ("This app explains what the companies' public filings say. It is not advice to buy, sell or hold. "
              "Judging the share price needs the live price and your own judgement, which the filings do not have.")

SUGGESTED = [
    "What was Stylam's net cash flow from operating activities in FY26?",
    "How much did Speciality's revenue grow in FY26 compared with FY25?",
    "Why did Speciality report an exceptional item in FY26?",
    "When did Stylam's new plant start commercial production?",
    "How much does Speciality owe in borrowings and lease liabilities?",
    "What is Stylam's latest share price?",
    "Should I buy Stylam shares now?",
]


def cell(text):
    return " ".join(str(text).split()).replace("|", "/")


def link(source, url):
    return f"[{source}]({url})" if url else source


def label_text(label):
    return f"**{label}**" if label else "**No label** (see the finding)"


# --- tab 1 ------------------------------------------------------------------------------------------------
def price_md(stock):
    """Fetched on page load and on a change of company, so it is never the number from when the app started."""
    p = share_price(stock)
    if "error" in p:
        return "**Latest share price:** unavailable just now. Check it on your broker's app or bseindia.com."
    return (f"**Latest share price:** ₹{p['price_rupees']:,.2f} on {p['exchange']}, as of {p['as_of']} "
            f"(source: {p['source']}). This one figure is not from a filing. It may be delayed, and showing it "
            "is not advice.")


def overview_md(stock):
    d = DATA[stock]
    out = [f"## {FULL_NAME[stock]}: what the company does",
           "From the company's own FY26 annual report. The first four come from Section A of its business "
           "responsibility report (BRSR), a form every listed company fills in with the same numbered questions. "
           "They are the company's words, not a summary.", ""]
    for part in d["overview"]:
        out.append(f"### {part['heading']}")
        if not part["passages"]:
            out.append("*Nothing found in the annual report for this.*")
        for p in part["passages"]:
            if p.get("plain"):
                out.append(f"{p['plain']}")
            out.append("> " + " ".join(p["text"].split())[:700].replace(" | ", " · ").replace("|", "·"))
            out.append(f"> — {link(p['source'], p['url'])}, {p['section']}\n")
    out += ["", "### Revenue and profit, every year",
            "From the annual reports' profit and loss statements (standalone), in ₹ crore. "
            "Growth is against the year before.", "",
            "| Year | Revenue | Growth | Profit after tax | Growth | Source |", "|---|---:|---:|---:|---:|---|"]
    for y in d["growth"]["years"]:
        rev, pat = y.get("revenue", {}), y.get("pat", {})
        fmt = lambda m: f"{m['value']:,.2f}" if m.get("value") is not None else "–"
        gr_ = lambda m: f"{m['growth'] * 100:+.1f}%" if m.get("growth") is not None else "–"
        out.append(f"| {y['fiscal_year']} | {fmt(rev)} | {gr_(rev)} | {fmt(pat)} | {gr_(pat)} | "
                   f"{cell(rev.get('source') or pat.get('source', ''))} |")
    if d["growth"]["notes"]:
        out += ["", "**Years to read with care**"] + [f"- {n}" for n in d["growth"]["notes"]]
        out.append("- This is why the app shows every year instead of one average growth rate.")
    return "\n".join(out)


# --- tab 2 ------------------------------------------------------------------------------------------------
def check_md(c, limit=8):
    out = [f"### {c['id']}. {c['title']}", f"{label_text(c['label'])}"]
    if c["threshold"]:
        out.append(f"*Rule: {c['threshold']}*")
    out += ["", c["finding"]]
    if c["meaning"]:
        out += ["", f"**What it means.** {c['meaning']}", "", f"**Why it matters.** {c['why']}"]
    if c["evidence"]:
        out += ["", "| Year | What | As printed | Page |", "|---|---|---|---|"]
        for e in c["evidence"][:limit]:
            out.append(f"| {cell(e['year'])} | {cell(e['what'])[:80]} | {cell(e['value'])[:120]} | "
                       f"{link(cell(e['source']), e.get('url'))} |")
    return "\n".join(out)


def health_md(stock):
    h = DATA[stock]["health"]
    out = [f"## {FULL_NAME[stock]}: health check, {h['fiscal_year']}", f"*{h['note']}*", "",
           "Each label comes with the rule that set it. The rules are rules of thumb chosen for this app, "
           "not official limits.", ""]
    for c in h["checks"]:
        if c["id"] == 7:
            out.append(f"### 7. {c['title']}\n{label_text(c['label'])}\n\nSee the **Promises vs delivery** tab.")
        else:
            out.append(check_md(c))
        out.append("\n---")
    return "\n".join(out)


# --- tab 3 ------------------------------------------------------------------------------------------------
def promises_md(stock):
    c = next((c for c in DATA[stock]["health"]["checks"] if c["id"] == 7), None)
    if c is None:
        return "Check 7 was not run in this build."
    out = [f"## {FULL_NAME[stock]}: promises against delivery", label_text(c["label"]),
           f"*Rule: {c['threshold']}*", "", f"**What it means.** {c['meaning']}", "",
           f"**Why it matters.** {c['why']}", "", "### What the model found", c["finding"], "",
           "### The dated excerpts it read, oldest first",
           "Read these yourself: they are what the company said, with the date it said it."]
    for e in c["evidence"]:
        out += ["", f"**{cell(e['year'])}**, {link(cell(e['source']), e.get('url'))}",
                "> " + " ".join(e["value"].split())]
    return "\n".join(out)


# --- tab 4 ------------------------------------------------------------------------------------------------
def events_md(stock):
    ev = DATA[stock]["events"]
    out = [f"## {FULL_NAME[stock]}: recent filings", f"{len(ev)} filings since 1 October 2025, newest first. "
           "Each shows the opening of the company's letter to the stock exchange; open the link for the whole filing.", ""]
    for e in ev:
        out.append(f"**{e['date']}**, {e['doc_type']}" + (f" ({e['period']})" if e["period"] and e["period"] != e["date"] else "")
                   + f", {e['pages']} pages" + (f" · [on BSE]({e['url']})" if e["url"] else ""))
        if e.get("plain"):
            out.append(e["plain"])
        out.append("> " + (e["excerpt"][:450] + "…" if len(e["excerpt"]) > 450 else e["excerpt"]) + "\n")
    return "\n".join(out)


# --- compare ----------------------------------------------------------------------------------------------
def compare_md():
    checks = {s: {c["id"]: c for c in DATA[s]["health"]["checks"]} for s in STOCKS}
    ids = sorted(set().union(*[set(v) for v in checks.values()]))
    out = ["## The two companies on the same checks", f"*{NOT_ADVICE}*", "",
           "| Check | " + " | ".join(FULL_NAME[s] for s in STOCKS) + " |", "|---|" + "---|" * len(STOCKS)]
    for i in ids:
        title = next(checks[s][i]["title"] for s in STOCKS if i in checks[s])
        out.append(f"| {i}. {title} | " + " | ".join(
            (checks[s][i]["label"] or "No label") if i in checks[s] else "–" for s in STOCKS) + " |")
    out += ["", "A label is not a score. Read each finding in the **Health check** tab: the same label can hide "
            "very different numbers, and one company's restaurants and the other's factories run on different money."]
    return "\n".join(out)


def glossary_md():
    return "## Words used in this app\n\n" + "\n\n".join(f"**{t}.** {d}" for t, d in GLOSSARY)


# --- tab 5: the only part that runs at question time -------------------------------------------------------
_agent = None

# The key pays per call, so the app as a whole gets a daily allowance, kept on disk: the per-session count lives
# in the page and a reload resets it (the review of 2026-10-23).
# 30 from 2026-10-27, for the demo: Sonnet measured about $0.03 a question (40 golden questions, $1.11). The Space's
# disk does not survive a restart, so usage.json starts again from zero after one.
DAILY_LIMIT = 30
USAGE = os.path.join(HERE, ".store", "ui", "usage.json")
_usage_lock = threading.Lock()


def take_one_from_today():
    """Count one question against today's allowance; False when it is used up."""
    today = time.strftime("%Y-%m-%d")
    with _usage_lock:
        try:
            usage = json.load(open(USAGE, encoding="utf-8"))
        except (OSError, ValueError):
            usage = {}
        if usage.get(today, 0) >= DAILY_LIMIT:
            return False
        usage[today] = usage.get(today, 0) + 1
        json.dump(usage, open(USAGE, "w", encoding="utf-8"))
        return True


def code_lookups(question, metrics):
    """The app's own lookups for metrics the question names and the agent skipped: each company the question
    names, each year it names (or the latest), standalone."""
    import agent
    stocks = parse_query(question)["stocks"] or STOCKS
    years = [f"FY{y}" for y in sorted(set(named_years(question)))] or [""]
    lines = []
    for stock in stocks:
        for metric in metrics:
            for fy in years:
                out = agent.lookup_financials_tool.invoke({"stock": stock, "metric": metric, "fiscal_year": fy})
                if "error" in out:
                    lines.append(f"- {stock}, {metric.replace('_', ' ')}: {out['error']}")
                else:
                    value = (f"₹{out['value_crore']:,.2f} crore" if "value_crore" in out
                             else f"₹{out['value_per_share']:,.2f} per share")
                    lines.append(f"- {out['stock']}, {out['label']}, {out['fiscal_year']} ({out['basis']}): "
                                 f"{value} [{out['source']}]" + (f" ({out['note']})" if out.get("note") else ""))
    return "\n".join(lines)


def tool_findings(r):
    """What the tools returned, as plain lines with their pages: shown in place of an answer that is held back."""
    lines = []
    # the full tool output is in the saved messages; the steps keep only the first 2,000 characters, which cuts
    # a search result in the middle of its JSON
    for m in r.get("messages", []):
        if m.get("type") != "tool":
            continue
        try:
            out = json.loads(m["data"]["content"])
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(out, dict) and "from" in out and "to" in out:
            lines.append(f"- {out['stock']}, {out['metric']} ({out['basis']}): {out['from']['fiscal_year']} "
                         f"₹{out['from']['value']:,.2f} crore [{out['from']['source']}], {out['to']['fiscal_year']} "
                         f"₹{out['to']['value']:,.2f} crore [{out['to']['source']}]"
                         + (f", a change of {out['change_percent']:+.1f}%" if out.get("change_percent") is not None else ""))
        elif isinstance(out, dict) and "value_crore" in out:
            lines.append(f"- {out['stock']}, {out['label']}, {out['fiscal_year']} ({out['basis']}): "
                         f"₹{out['value_crore']:,.2f} crore [{out['source']}]"
                         + (" (nil: the statement has no line for it)" if out.get("note", "").startswith("nil") else ""))
        elif isinstance(out, dict) and "value_per_share" in out:
            lines.append(f"- {out['stock']}, {out['label']}, {out['fiscal_year']}: ₹{out['value_per_share']:,.2f} "
                         f"per share [{out['source']}]")
        elif isinstance(out, dict) and "price_rupees" in out:
            lines.append(f"- {out['stock']}, latest share price: ₹{out['price_rupees']:,.2f} on {out['exchange']}, as of "
                         f"{out['as_of']} [{out['source']}; not from a filing]")
        elif isinstance(out, list):
            lines.append("- Passages found on: " + "; ".join(h["source"] for h in out if isinstance(h, dict)))
    return "\n".join(lines) or "*No tool returned a figure.*"


def agent_answer(question, used, stock=None):
    global _agent
    question = (question or "").strip()
    if not question:
        yield "Type a question or pick one below.", "", used
        return
    asked = advice_question(question)
    if not asked["ok"]:   # guardrail on the way in: code replies, the model is never called
        yield (ADVICE_REPLY, "No model call: the question matched the no-advice rule "
               + ", ".join(f"{rule} (“{text}”)" for rule, text in asked["found"]) + ".", used)
        return
    if used >= QUESTIONS_PER_SESSION:
        yield f"This session has used its {QUESTIONS_PER_SESSION} questions.", "", used
        return
    if not take_one_from_today():
        yield f"The app has answered its {DAILY_LIMIT} questions for today. Please come back tomorrow.", "", used
        return
    used += 1   # counted before the run, so a run that fails still counts
    # A question that names no company is about the one selected above (2026-10-23: "What is 2026 profit?"
    # with Stylam selected went to Speciality, because the selection never reached the agent).
    if stock and not parse_query(question)["stocks"]:
        question = f"About {stock}: {question}"
    yield "Working… the first question loads the models; with the local model each answer takes 1-2 minutes.", "", used
    import agent
    if _agent is None:
        _agent = agent.build_agent()
    r = agent.run(question, _agent, tags=("ui",))
    checks = r["checks"]
    named = parse_query(question)["stocks"]
    other = sorted({agent.stock_name(c["args"]["stock"]) for s in r["steps"] if s["kind"] == "model"
                    for c in s["tool_calls"] if c["args"].get("stock")} - set(named))
    missing = checks.get("metrics_not_looked_up", [])
    searched = "search_filings" in r["tool_calls"]
    claims_missing = [m for m in missing if agent.mentions(r["answer"], m)]
    if not checks["advice"]["ok"]:
        answer = ("**Answer held back.** The model's answer used advice wording "
                  f"({', '.join(f'{rule}: “{text}”' for rule, text in checks['advice']['found'])}). "
                  f"This app explains the filings and does not advise. {NOT_ADVICE}")
    elif checks.get("figure_years_not_asked"):
        years = ", ".join(f"FY{y}" for y in checks["figure_years_not_asked"])
        answer = (f"**Answer held back.** The question asks about {', '.join(f'FY{y}' for y in named_years(question))}, "
                  f"but every figure the agent looked up is for {years}, so the answer would give another year's "
                  "number. Here is what the tools found:\n\n" + tool_findings(r))
    elif claims_missing and not searched:
        # The answer talks about a figure the question names that no tool looked up, and nothing was searched
        # either, so what it says about it came from nowhere (suggested question 5: "lease liabilities nil")
        answer = ("**Answer held back.** It says something about "
                  + ", ".join(m.replace("_", " ") for m in claims_missing)
                  + " without looking it up. Here is what the tools found, with the missing figures looked up "
                  "by the app:\n\n" + tool_findings(r) + "\n" + code_lookups(question, missing))
    elif checks["numbers_not_from_a_tool"] or checks.get("percents_not_from_a_percent"):
        # A reader who never opens a filing cannot judge a warning, so the model's text is not shown at all
        # (2026-10-23: Qwen wrote the right profit, then a table of figures it made up).
        bad = checks["numbers_not_from_a_tool"] + [f"{p}%" for p in checks.get("percents_not_from_a_percent", [])]
        answer = ("**Answer held back.** The model's answer had figures that no tool returned ("
                  + ", ".join(bad) + "), so it may have made them up. Here is what the tools found, "
                  "straight from the filings:\n\n" + tool_findings(r))
    elif not r["answer"] and r["tool_calls"]:
        # It looked things up but ran out of model calls before writing the answer (B08, 2026-10-27)
        answer = ("**The agent ran out of steps before writing an answer.** Here is what the tools found, "
                  "straight from the filings:\n\n" + tool_findings(r))
    elif checks.get("still_a_plan") or not r["tool_calls"]:
        # The model described a step instead of taking it, even after the nudge, or used no tool at all: the
        # app runs the search itself (a fixed step, so plain code) and shows the passages, not the model's text.
        stock_arg = named[0] if len(named) == 1 else ""
        hits = agent.search_filings.invoke({"query": question, "stock": stock_arg})[:3]
        answer = ("**The model did not look anything up**, so its reply is not shown. Here are the passages a "
                  "search of the filings found for your question; the answer may be in them:\n\n"
                  + "\n\n".join(f"**{h['source']}**\n> " + " ".join(h["text"].split())[:500] for h in hits))
    else:
        answer = r["answer"] or "*The agent stopped without an answer.*"
        answer += "\n\n*Number check: every number in the answer came from a tool.*"
        if missing:
            answer += ("\n\n**Also from the annual reports** (the question names these, and the agent did not "
                       "look them up):\n" + code_lookups(question, missing))
        if other:
            answer += (f"\n\n**Check the company:** the question was about {' and '.join(named)}, but the agent "
                       f"asked the tools about {', '.join(repr(o) for o in other)}.")
        answer += ("\n\n*A number check confirms a number exists in what the tools returned, not that it answers "
                   "the question. Look at the steps.*")
    steps = [f"Question sent to the agent: *{question}*", "",
             f"**{r['model']}**, {r['seconds']} s, {r['model_calls']} model calls, {r['prompt_tokens']:,} prompt "
             f"tokens in all, stopped by: {r['stopped_by']}. Run id `{r['run_id']}`.", ""]
    n = 0
    for s in r["steps"]:
        if s["kind"] == "model":
            n += 1
            calls = "; ".join(f"`{c['name']}({json.dumps(c['args'], ensure_ascii=False)})`" for c in s["tool_calls"])
            steps.append(f"{len(steps) - 3}. Model call {n}: " + (f"asks for {calls}" if calls else "writes the answer"))
        elif s["kind"] == "harness":
            steps.append(f"{len(steps) - 3}. The code sends back: *{s['text']}* (the reply described a step "
                         "instead of taking it)")
        else:
            steps.append(f"{len(steps) - 3}. Tool `{s['name']}` returns: {cell(s['output'])[:300]}")
    yield answer, "\n".join(steps), used


with gr.Blocks(title="Equity Filings Assistant") as demo:
    gr.Markdown(f"# Equity Filings Assistant\n{FULL_NAME['Stylam']} and {FULL_NAME['Speciality']}, "
                f"from their public filings on BSE, with the page for every fact.\n\n*{NOT_ADVICE}*")
    stock = gr.Radio(STOCKS, value=STOCKS[0], label="Company (all tabs; a question that names no company is about this one)")
    with gr.Tab("1. What the company does"):
        t0 = gr.Markdown()
        t1 = gr.Markdown(overview_md(STOCKS[0]))
    with gr.Tab("2. Health check"):
        t2 = gr.Markdown(health_md(STOCKS[0]))
    with gr.Tab("3. Promises vs delivery"):
        t3 = gr.Markdown(promises_md(STOCKS[0]))
    with gr.Tab("4. Recent events"):
        t4 = gr.Markdown(events_md(STOCKS[0]))
    with gr.Tab("5. Ask a question"):
        gr.Markdown("Ask about either company. The agent looks figures up in a table built from the annual "
                    "reports, searches the filings for anything in words, and uses a calculator for sums. "
                    f"Up to {QUESTIONS_PER_SESSION} questions per session.")
        q = gr.Textbox(label="Your question", lines=2)
        ask = gr.Button("Ask", variant="primary")
        gr.Examples(SUGGESTED, inputs=q, label="Suggested questions")
        a = gr.Markdown()
        with gr.Accordion("How the agent got there (its steps)", open=False):
            trace = gr.Markdown()
        used = gr.State(0)
        ask.click(agent_answer, [q, used, stock], [a, trace, used])
        q.submit(agent_answer, [q, used, stock], [a, trace, used])
    with gr.Tab("Compare"):
        gr.Markdown(compare_md())
    with gr.Tab("Words"):
        gr.Markdown(glossary_md())
    gr.Markdown(f"*Built {DATA['built']}. Check 7 written by {DATA.get('llm_model') or 'no model'}.*")
    stock.change(lambda s: (overview_md(s), health_md(s), promises_md(s), events_md(s)), stock, [t1, t2, t3, t4])
    stock.change(price_md, stock, t0)
    demo.load(price_md, stock, t0)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from llm import BACKEND, model_name
    print(f"LLM backend {BACKEND}, model {model_name()}", flush=True)
    for name in STOCKS:     # a cloud server can be blocked where a laptop is not, so the log shows what works here
        print(f"Share price {name}: {share_price(name)}", flush=True)
    password = os.environ.get("APP_PASSWORD")
    demo.launch(auth=("demo", password) if password else None)
