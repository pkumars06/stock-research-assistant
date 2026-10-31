"""Step 4: the question-answering agent. LangChain's create_agent, our three tools, two limits, two checks.

Why an agent here and a workflow for the health check (guide 3 Part 3, "can I write the steps down first?"):
the health check runs the same seven steps for every company, so it is plain code. A question can need
one lookup, a search, two lookups and a sum, or a search per part of a multi-hop question, and which one
is only known from the question. So the model chooses the next step.

    search_filings      our hybrid search + reranker (retrieve.py), each hit widened to part of its page
    lookup_financials   one figure from the build-time table (financials.py), with its page
    calculator          arithmetic, so no number in an answer is the model's own sum
    get_share_price     the latest price, fetched online (price.py): the one figure not from a filing

After the run, two checks in plain code (no LLM):
    number check   every number in the answer must appear in something a tool returned
    advice check   no buy / sell / hold, target price or "undervalued" (advice_check.py)

Every run is saved to .store/agent_runs/<run id>.json: the whole chat, each model call's tokens and time,
each tool call, the answer and the checks. With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, LangChain
also sends the same run to LangSmith; nothing in this file changes for that.

python agent.py "question"     one question (local Qwen unless a key is set; see llm.py)
"""
import ast
import itertools
import json
import operator
import re
import os
import sys
import time
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware, wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.tools import tool

import financials
from advice_check import advice_check
from answer import NUMBER, cite, numbers, pages, window
from llm import BACKEND, chat_model, model_name
from price import share_price
from retrieve import STORE, YEAR_WINDOW, Searcher, named_years, parse_query

RUNS = os.path.join(STORE, "agent_runs")
# 6 and 6 until 2026-10-27: Sonnet on B08 (one-offs and tax rate) made 4 lookups and 2 sums and had no call left
# to write the answer
MODEL_LIMIT = 8        # model calls per question
TOOL_LIMIT = 8         # tool calls per question
LIMIT_TEXT = "Model call limits exceeded"   # the reply ModelCallLimitMiddleware writes when it ends the run
# B08 again with 8: four right lookups, two right sums, then FY25 for comparison, and the limit ended the run with
# nothing written. A higher limit only bought more calls, so the harness gives one last turn with no tools.
FINAL = ("You have used all your steps. Do not look anything else up. Answer the question now from what the tools "
         "returned above, and say what you could not check.")
SEARCH_HITS = 5
HIT_CHARS = 700        # each hit widened to this much of its page; 5 x 700 keeps one search near 1,000 tokens

SYSTEM = """You answer questions about two Indian listed companies, Stylam and Speciality, from their filings.
Use the tools. Do not answer from memory.
- A figure such as revenue, profit, cash flow or borrowings: call lookup_financials.
- A reason, an event, a promise or anything in words: call search_filings.
- How a figure changed between two years (growth, rise, fall): call compare_years. It looks up both years.
- Other arithmetic on figures you already have: call calculator with numbers only. Never do sums yourself.
- The latest share price: call get_share_price. Give the price, exchange, time and source, and say it is not from the
  filings. Only report it: never call it high or low, and never compare it with earnings or any other figure.
- Name the source of every fact, as the tool gave it.
- If the tools do not give the answer, say: Insufficient evidence in the filings.
- Explain, do not advise: never say buy, sell, hold, undervalued or overvalued, and never give a target price.
Answer in short, plain sentences."""

_searcher = None


def searcher():
    global _searcher
    if _searcher is None:
        _searcher = Searcher("v1")
    return _searcher


def _fy_number(fiscal_year):
    return int(str(fiscal_year).upper().replace("FY", "").strip()) % 100


# --- the three tools. The docstring is what the model reads (guide 3 Part 1: a tool is described in text) ---

@tool
def search_filings(query: str, stock: str = "", fiscal_year: str = "") -> list:
    """Search the annual reports, earnings calls and filings. Returns passages with their source page.
    query: what to look for, in words. stock: "Stylam" or "Speciality" (optional).
    fiscal_year: like "FY26" (optional; the Indian financial year, April to March)."""
    f = parse_query(query, default_recent=True)
    if stock:
        f["stocks"] = [stock_name(stock)]
    if fiscal_year:
        fy = _fy_number(fiscal_year)
        f["fy"] = (fy - YEAR_WINDOW, fy + YEAR_WINDOW)
    s = searcher()
    hits = []
    for i in s.rerank(query, f, SEARCH_HITS):
        c = s.chunks[i]
        page = c["pages"][0]
        text = window(pages().get((c["doc_id"], page), c["text"]), c["text"], HIT_CHARS)
        hits.append({"source": cite(c, page), "text": text})
    return hits


# The metric names are listed with their meanings in the tool's description; since 2026-10-23 the argument is
# free text, matched by resolve_metric() below (an enum refused near-misses). The first run (2026-10-22) had
# no meanings and no list: the model invented
# "net_cash_flow_from_operating_activities", got an error back, and told the user the figure was not there.
# Plain names for the model, mapped to the table's keys. Run 2 (40 questions, 2026-10-22): with the key "cfo"
# the model asked for "cash" (the cash balance) on both cash-from-operations questions.
PLAIN = {"revenue": "revenue", "other_income": "other_income", "profit_before_tax": "pbt",
         "exceptional_items": "exceptional", "tax": "tax", "profit_after_tax": "pat", "eps": "eps_basic",
         "cash_from_operations": "cfo", "rent_paid": "lease_payments", "cash_balance": "cash",
         "bank_deposits": "other_bank", "current_investments": "current_investments", "borrowings": "borrowings",
         "lease_liabilities": "lease_liabilities", "net_worth": "equity", "capital_work_in_progress": "cwip",
         "contingent_liabilities": "contingent_liabilities"}
METRIC_MEANINGS = "; ".join(f"{k} = {financials.METRICS[v]}" for k, v in PLAIN.items())

# A name close to a real one is taken, not refused (2026-10-23, the UI: the model asked for
# "net_cash_from_operations", the schema refused it, and the model asked the user instead of retrying; it never
# retries). Keywords in order, most specific first: "cash" with "operat" is cash from operations, and only
# "cash" alone is the cash balance, the mix-up seen in run 2.
ALIASES = [
    (("cash", "operat"), "cash_from_operations"), (("cfo",), "cash_from_operations"),
    (("operating", "cash"), "cash_from_operations"),
    (("profit", "before"), "profit_before_tax"), (("pbt",), "profit_before_tax"),
    (("profit", "after"), "profit_after_tax"), (("pat",), "profit_after_tax"), (("net", "profit"), "profit_after_tax"),
    (("earning", "share"), "eps"), (("eps",), "eps"),
    (("exceptional",), "exceptional_items"), (("one", "off"), "exceptional_items"),
    (("other", "income"), "other_income"),
    (("contingent",), "contingent_liabilities"),
    (("work", "progress"), "capital_work_in_progress"), (("cwip",), "capital_work_in_progress"),
    (("lease", "liabilit"), "lease_liabilities"), (("rent",), "rent_paid"), (("lease", "pay"), "rent_paid"),
    (("borrow",), "borrowings"), (("debt",), "borrowings"), (("loan",), "borrowings"),
    (("net", "worth"), "net_worth"), (("equity",), "net_worth"),
    (("deposit",), "bank_deposits"), (("bank", "balance"), "bank_deposits"),
    (("current", "invest"), "current_investments"),
    (("revenue",), "revenue"), (("sales",), "revenue"), (("turnover",), "revenue"),
    (("tax",), "tax"), (("cash",), "cash_balance"),
]
# A broad word counts only when the name holds nothing else: "dine_in_sales" is part of revenue, not all of it,
# and a confident wrong figure is worse than an error the model can read.
BROAD = {"sales", "revenue", "turnover", "tax", "cash"}
ONLY_WITH_BROAD = {"net", "total", "from", "operations", "and", "equivalents", "balance", "expense", "the"}
# Words that make it a different figure, so the name is refused even when a keyword matches (the review of
# 2026-10-23 measured all nine): rental_income is not rent paid, loans_and_advances is money owed to the
# company, short_term_borrowings is a part of borrowings, debt_to_equity and net_profit_margin are ratios.
DIFFERENT = [("comprehensive",), ("rental",), ("interest",), ("share", "capital"), ("advance",), ("security",),
             ("ratio",), ("to",), ("margin",), ("short",), ("long",), ("deferred",), ("growth",), ("per", "cent")]


WHOLE_WORD = {"to", "pat", "pbt", "eps", "cfo", "cwip", "per"}   # "pat" is not the start of "patents"


def _has(words, needle):
    # a short word or an abbreviation must be a whole word; a longer one may start a word ("advance" -> "advances")
    return any(w == needle if needle in WHOLE_WORD or len(needle) <= 2 else w.startswith(needle) for w in words)


def resolve_metric(name):
    """The plain metric name the model meant, or None."""
    key = re.sub(r"[^a-z]+", "_", str(name).lower()).strip("_")
    if key in PLAIN:
        return key
    words = key.split("_")
    if any(all(_has(words, n) for n in needles) for needles in DIFFERENT):
        return None
    for needles, metric in ALIASES:
        if all(_has(words, n) for n in needles):
            if len(needles) == 1 and needles[0] in BROAD and not set(words) <= ONLY_WITH_BROAD | {needles[0]}:
                continue
            return metric
    return None


# The metrics a question names, so the code can check the agent looked each one up (the plan's "question words
# against the metric asked for" rule; the third review, 2026-10-23). Suggested question 5 asks for borrowings
# and lease liabilities; Qwen looked up borrowings only and said lease liabilities were nil (they are
# ₹13,994.45 lakh, golden B03) - a false claim with no number in it. Only terms that name one figure: everyday
# words (cash, sales, tax, equity, rent) appear in questions without naming a figure ("equity shares").
# Whole phrases, in order, and abbreviations as whole words: a set of loose keywords read "market share" on an
# "earnings call" as EPS, "patents" as PAT and "current investors" as current investments (the fourth review).
NAMED = {
    # up to three words between: "how much cash did Stylam generate from operations" (the fifth review)
    "cash_from_operations": r"\bcash\b(?:\s+\w+){0,3}?\s+from\s+(?:its\s+)?operat\w*"
                            r"|\boperating\s+cash\s*flows?\b|\bcfo\b",
    "profit_before_tax": r"\bprofits?\s+before\s+tax\w*|\bpbt\b",
    "profit_after_tax": r"\bprofits?\s+after\s+tax\w*|\bnet\s+profits?\b|\bpat\b",
    "eps": r"\bearnings\s+per\s+share\b|\beps\b",
    "exceptional_items": r"\bexceptional\s+items?\b",
    "other_income": r"\bother\s+income\b",
    "contingent_liabilities": r"\bcontingent\s+liabilit\w*",
    "capital_work_in_progress": r"\bcapital\s+work[-\s]+in[-\s]+progress\b|\bcwip\b",
    "lease_liabilities": r"\blease\s+liabilit\w*",
    # "debt-free" and "zero-debt" count, by design; "loans and advances" is money owed to the company, so not
    "borrowings": r"\bborrowings?\b|\bdebt\b|\bloans?\b(?!\s+(?:and\s+|&\s*)?advances)",
    "net_worth": r"\bnet\s+worth\b",
    "current_investments": r"\bcurrent\s+investments?\b",
}


def metrics_named(question):
    """The plain metric names a question asks for. None for a quarter: the table holds years only, so
    "PBT growth in Q1 FY27" must come from a search, and an annual figure would answer the wrong period."""
    if re.search(r"\bq[1-4]\b|\bquarter|\bh[12]\b|\bhalf[- ]year", question, re.I):
        return set()
    found = {metric for metric, pattern in NAMED.items() if re.search(pattern, question, re.I)}
    # loans the company gave are money owed to it, not borrowings ("did Stylam give loans to employees?",
    # the sixth review); borrowings still count when the question also says borrowings or debt
    if ("borrowings" in found and re.search(r"\b(?:give|gave|given|giving|lent|lend|extended)\b[^.?]{0,30}\bloans?\b"
                                            r"|\bloans?\s+(?:given\s+)?to\b", question, re.I)
            and not re.search(r"\bborrowings?\b|\bdebt\b", question, re.I)):
        found.discard("borrowings")
    return found


def mentions(text, metric):
    """The text talks about this metric (by the same phrases)."""
    return bool(re.search(NAMED[metric], text or "", re.I))


def metrics_looked_up(steps):
    """The plain metric names the agent asked lookup_financials or compare_years for."""
    return {resolve_metric(c["args"].get("metric", "")) for s in steps if s["kind"] == "model"
            for c in s["tool_calls"] if c["name"] in ("lookup_financials", "compare_years")} - {None}


def stock_name(stock):
    """ "Stylam Industries", "stylam" -> "Stylam" (the model sometimes writes the full name)."""
    s = str(stock).strip().lower()
    return next((name for key, name in financials.STOCK_NAMES.items() if key in s), str(stock).strip())


# A row the statement does not print, on a statement page that was found, means the company has none: Speciality
# has no borrowings (golden set B03). The first fix for the review of 2026-10-23 refused these figures as "not
# printed", which broke B03; the second review caught it. All 54 such rows have their statement page, so the
# tools give 0 and say why. (The 2026-10-20 lesson, "a missing row is not a zero", was about rows the
# extractor failed to find on a page that does print them; reading each year in two reports caught those.)
NIL_NOTE = "nil: the {year} statement ({source}) has no line for this, so the company had none"


def nil_note(r):
    return NIL_NOTE.format(year=r["fiscal_year"], source=r["source"])


@tool("lookup_financials")
def lookup_financials_tool(stock: str, metric: str, fiscal_year: str = "", basis: str = "standalone") -> dict:
    """One figure from the annual reports, in rupees crore, with the page it came from.
    stock: "Stylam" or "Speciality". fiscal_year: like "FY26" (empty for the latest).
    basis: "standalone" (default) or "consolidated". metric, with its meaning: {meanings}."""
    plain = resolve_metric(metric)
    if plain is None:
        return {"error": f"no metric called {metric!r}. Use one of: {', '.join(PLAIN)}"}
    r = financials.lookup_financials(stock_name(stock), PLAIN[plain], fiscal_year or None, basis)
    if r.get("nil_by_absence"):
        r["note"] = nil_note(r)
    keep = ("stock", "fiscal_year", "basis", "label", "value_crore", "value_per_share", "source", "restated",
            "note", "error")
    # "is not": a figure of 0.0 equals False, and "not in (None, False)" dropped it
    out = {k: r[k] for k in keep if k in r and r[k] is not None and r[k] is not False}
    if plain != metric:
        out["metric_used"] = f"{plain} (asked for {metric!r})"
    return out


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** ( ) are allowed")


@tool
def calculator(expression: str) -> dict:
    """Arithmetic on numbers, for example "(1129.4 - 1025.2) / 1025.2 * 100" for a growth rate in percent.
    Only numbers and + - * / ** ( ) are allowed."""
    try:
        value = _eval(ast.parse(expression.replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, NameError):
        # 2026-10-23: the model sent "(Revenue from operations for FY26 - ...)" before looking anything up
        return {"error": "the calculator takes numbers only, such as (1129.4 - 1025.2) / 1025.2 * 100. "
                         "Get each figure first with lookup_financials, or use compare_years for a change "
                         "between two years."}
    return {"expression": expression, "result": round(value, 4)}


# Growth between two years is always the same three steps (look up, look up, divide), so code does them in one
# call. On 2026-10-22 and 2026-10-23 the 0.5B model skipped the lookups and sent words to the calculator.
@tool
def compare_years(stock: str, metric: str, from_year: str, to_year: str, basis: str = "standalone") -> dict:
    """How one figure changed between two fiscal years: both figures with their pages, the change in rupees
    crore and in percent. stock: "Stylam" or "Speciality". from_year and to_year: like "FY25" and "FY26".
    metric: the same names as lookup_financials, for example revenue, profit_after_tax, cash_from_operations."""
    plain = resolve_metric(metric)
    if plain is None:
        return {"error": f"no metric called {metric!r}. Use one of: {', '.join(PLAIN)}"}
    a = financials.lookup_financials(stock_name(stock), PLAIN[plain], from_year, basis)
    b = financials.lookup_financials(stock_name(stock), PLAIN[plain], to_year, basis)
    for r in (a, b):
        if "error" in r:
            return {"error": r["error"]}
    unit = "value_crore" if a.get("value_crore") is not None else "value_per_share"
    x, y = a[unit], b[unit]
    out = {"stock": a["stock"], "metric": a["label"], "basis": basis,
           "from": {"fiscal_year": a["fiscal_year"], "value": x, "source": a["source"]},
           "to": {"fiscal_year": b["fiscal_year"], "value": y, "source": b["source"]},
           "unit": "rupees crore" if unit == "value_crore" else "rupees per share",
           "change_crore" if unit == "value_crore" else "change_per_share": round(y - x, 2),
           "change_percent": round((y - x) / abs(x) * 100, 1) if x else None}
    for side, r in (("from", a), ("to", b)):
        if r.get("nil_by_absence"):
            out[side]["note"] = nil_note(r)
    if not x:
        out["note"] = "no percent change: the starting figure is zero"
    if plain != metric:
        out["metric_used"] = f"{plain} (asked for {metric!r})"
    return out


@tool
def get_share_price(stock: str) -> dict:
    """The latest share price of one company in rupees, fetched online now, with its exchange and time. It is not
    from a filing. Only the latest price: no past prices, targets or forecasts. stock: "Stylam" or "Speciality"."""
    return share_price(stock_name(stock))


lookup_financials_tool.description = lookup_financials_tool.description.replace("{meanings}", METRIC_MEANINGS)
TOOLS = [search_filings, lookup_financials_tool, compare_years, calculator, get_share_price]


@wrap_tool_call
def tool_errors_to_text(request, handler):
    """A tool that fails sends its error back to the model as the observation, instead of stopping the run."""
    try:
        return handler(request)
    except Exception as e:      # a small model sends bad arguments often; the error is its feedback
        return ToolMessage(content=json.dumps({"error": f"{type(e).__name__}: {e}"}),
                           tool_call_id=request.tool_call["id"], name=request.tool_call["name"], status="error")


# A number and its label are checked together for percentages (2026-10-23): Qwen wrote "grew by 104.2%" from
# compare_years' change of 104.2 crore, and the number check passed it, because 104.2 was in the tool's output.
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b|per\s+cent\b)", re.I)


def percents(text):
    return {m.group(1).replace(",", "").rstrip(".") for m in PERCENT.finditer(text or "")}


def tool_percents(tool_text):
    """Every percentage a tool returned: a number written with %, or the value of a field named *percent*."""
    found = set()
    for t in tool_text:
        found |= percents(t)
        try:
            stack = [json.loads(t)]
        except ValueError:
            continue
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                for k, v in d.items():
                    if "percent" in k and isinstance(v, (int, float)):
                        found |= {str(v), f"{v:g}", f"{abs(v):g}"}
                    stack.append(v)
                # a calculator sum that ends "* 100" is a percent (B08, 2026-10-27: "53.42 / 202.89 * 100" = 26.3295
                # gave "26.3%", and the check held it back)
                v = d.get("result")
                if isinstance(v, (int, float)) and re.search(r"\*\s*100\s*\)?\s*$", str(d.get("expression", ""))):
                    found |= {str(v), f"{v:g}", f"{abs(v):g}"}
            elif isinstance(d, list):
                stack.extend(d)
    return found


YEAR = re.compile(r"(?:19|20)\d\d")

# 2026-10-23: "I'll use the `search_filings` function ... Let me check that now." came back as the answer, with
# no tool call (the Week 5 cleaning agent wrote plans as text the same way). One nudge, then the app falls back.
PLAN_AS_TEXT = re.compile(r"\b(?:search_filings|lookup_financials|compare_years|calculator|get_share_price)\b"
                          r"|\blet me (?:check|search|look|find|use|call)\b"
                          r"|\bI(?:'ll| will) (?:use|call|search|check|look)\b", re.I)
NUDGE = "Do not describe the step. Make the tool call now, then answer from what it returns."


def _decimals(x):
    return len(x.split(".")[1]) if "." in x else 0


# A figure written in another unit: crore from lakh (/100), from ₹ million (/10), from rupees (/1e7); lakh from
# rupees (/1e5); lakh from crore (x100). The Sonnet run of 2026-10-27 flagged 9 of 40 answers, all right, mostly
# for these (P01 "₹334.38 lakh (₹3.34 crore)", P02 "₹14,50,09,700 (about ₹14.5 crore)", P10 44.94 million).
UNITS = (100, 10, 1e7, 1e5, 0.01)


def all_numbers(text):
    """Every number in the text, one digit too, commas removed (numbers() keeps two digits or more)."""
    return {m.group().replace(",", "").rstrip(".") for m in NUMBER.finditer(re.sub(r"\[\d+\]", "", text))}


def _same(n, g, scales=()):
    """n names g: the same value (28.10 and 28.1), g rounded (149 for 149.47), or g in another unit, rounded."""
    x, v, d = float(n), float(g), _decimals(n)
    if x == v or (_decimals(g) > d and round(v, d) == x):
        return True
    # Another unit only for a number precise enough to pin the figure down: a whole "16" matches anything from 155
    # to 165 divided by 10, and S07's "16 months" (1.33 x 12) passed that way
    if d == 0 and len(n.lstrip("0")) < 3:
        return False
    return any(round(v / s, d) == x for s in scales)


def _worked_out(n, backed):
    """n is a sum, difference or ratio of two numbers the answer states (or the sum of three), rounded as
    written: P07 "72.55% ... 69.99% ... fell 2.56 points", S12 "6,504.04 - 4,125.27 = 2,378.77", P09 "8 + 15 +
    10 = 33", S07 "651 / 696 = 0.94". Only numbers the answer shows and a tool backs, so the reader can check."""
    x, d = float(n), _decimals(n)
    vals = sorted({float(b) for b in backed if float(b) != x})
    for a, b in itertools.permutations(vals, 2):
        if round(a + b, d) == x or round(a - b, d) == x or (b and round(a / b, d) == x):
            return True
    return any(round(sum(t), d) == x for t in itertools.combinations(vals, 3))


def unmatched(found, given, scales=(), answer_numbers=()):
    """The numbers in `found` that are in `given` neither as written nor rounded from one, years left out.
    The review of 2026-10-23: "2026" and "149" (149.47 rounded) were flagged, and noise teaches people to
    ignore the warning. With `scales`, a figure in another unit counts; with `answer_numbers` (every number in
    the answer), a sum, difference or ratio of the answer's own tool-backed numbers counts."""
    numeric = [g for g in given if re.fullmatch(r"-?\d+(?:\.\d+)?", g)]

    def backed(n):
        return n in given or any(_same(n, g, scales) for g in numeric)

    shown = [a for a in answer_numbers if backed(a) and not YEAR.fullmatch(a)]
    out = []
    for n in found:
        if YEAR.fullmatch(n) or backed(n) or (shown and _worked_out(n, shown)):
            continue
        out.append(n)
    return sorted(out)


def figure_years(tool_text):
    """The fiscal years of the figures the tools returned (lookup_financials and compare_years), as numbers."""
    found = set()
    for t in tool_text:
        try:
            stack = [json.loads(t)]
        except ValueError:
            continue
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                if re.fullmatch(r"FY\d\d", str(d.get("fiscal_year", ""))) and ("value_crore" in d or "value" in d
                                                                            or "value_per_share" in d):
                    found.add(int(d["fiscal_year"][2:]))
                stack.extend(d.values())
            elif isinstance(d, list):
                stack.extend(d)
    return found


def wrong_year(question, tool_text):
    """The question names a year, the tools returned figures, and none of them is from a year it names: the
    answer's figure is another year's (P12, the review of 2026-10-23: FY25 asked, FY26's 29.45 crore given,
    and the number check passed it). Other years are fine as long as one named year was looked up: "since FY22"
    may look up FY23-FY26, but not without FY22, its starting point."""
    asked, got = set(named_years(question)), figure_years(tool_text)
    return sorted(got) if asked and got and not asked & got else []


def build_agent(model=None):
    return create_agent(
        model=model or chat_model(), tools=TOOLS, system_prompt=SYSTEM,
        middleware=[ModelCallLimitMiddleware(run_limit=MODEL_LIMIT, exit_behavior="end"),
                    ToolCallLimitMiddleware(run_limit=TOOL_LIMIT, exit_behavior="continue"),
                    tool_errors_to_text])


def run(question, agent=None, tags=()):
    """Ask once; return the run record (also saved to .store/agent_runs/)."""
    agent = agent or build_agent()
    run_id = uuid.uuid4().hex[:12]
    t = time.time()
    config = {"run_name": "filings-agent", "tags": ["capstone", BACKEND, *tags],
              "metadata": {"run_id": run_id, "question": question}}
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    msgs = state["messages"]
    nudged = False
    last = msgs[-1]
    # .text, not .content: Claude's replies come back as a list of blocks (thinking, text), not one string
    if last.type == "ai" and not last.tool_calls and PLAN_AS_TEXT.search(last.text or ""):
        # The loop stops when the model writes no tool call, and nothing checks the work is done (guide 3 Part 2).
        # So the harness checks one sign of it: a reply that says what it will do instead of doing it.
        nudged = True
        state = agent.invoke({"messages": msgs + [{"role": "user", "content": NUDGE}]}, config=config)
        msgs = state["messages"]
    final_turn = False
    if msgs[-1].type == "ai" and (msgs[-1].text or "").startswith(LIMIT_TEXT):
        # One more model call, tools defined but none allowed (Claude needs the tools defined once the history
        # holds tool calls). The limit's own message is dropped, not sent back as if the model had said it.
        final_turn = True
        choice = {"anthropic": {"type": "none"}, "openai": "none"}.get(BACKEND)
        last_word = chat_model().bind_tools(TOOLS, **({"tool_choice": choice} if choice else {}))
        ask = HumanMessage(content=FINAL)
        msgs = msgs[:-1] + [ask, last_word.invoke(msgs[:-1] + [ask], config=config)]
    seconds = round(time.time() - t, 1)

    steps, tool_text = [], []
    for m in msgs:
        if m.type == "ai":
            u = m.usage_metadata or {}
            steps.append({"kind": "model", "prompt_tokens": u.get("input_tokens"),
                          "output_tokens": u.get("output_tokens"), "text": m.text,
                          "tool_calls": [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]})
        elif m.type == "tool":
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            tool_text.append(content)
            steps.append({"kind": "tool", "name": m.name, "status": getattr(m, "status", None),
                          "output": content[:2000]})
        elif m.type == "human" and m.content in (NUDGE, FINAL):
            steps.append({"kind": "harness", "text": m.content})
    last = msgs[-1]
    final = last.text if last.type == "ai" and not last.tool_calls else ""
    hit_limit = final.startswith(LIMIT_TEXT)
    if hit_limit:
        final = ""      # the limit's own message is not an answer (B08 showed it as one, 2026-10-27)
    still_a_plan = bool(final and PLAN_AS_TEXT.search(final))
    model_calls = sum(1 for s in steps if s["kind"] == "model")
    stopped = ("model call limit" if hit_limit or (model_calls >= MODEL_LIMIT and not final) else
               "model call limit, then a final answer without tools" if final_turn else
               "no final answer" if not final else "the model wrote no tool call")
    tools_seen = all_numbers(question)   # repeating a number from the question is not inventing one
    for t_ in tool_text:
        tools_seen |= all_numbers(t_)
    record = {
        "run_id": run_id, "question": question, "backend": BACKEND,
        "model": model_name(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds, "stopped_by": stopped,
        "model_calls": model_calls, "tool_calls": [c["name"] for s in steps if s["kind"] == "model" for c in s["tool_calls"]],
        "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in steps if s["kind"] == "model"),
        "answer": final,
        "checks": {"numbers_not_from_a_tool": unmatched(numbers(final), tools_seen, UNITS, all_numbers(final)),
                   "percents_not_from_a_percent": unmatched(percents(final), tool_percents(tool_text)),
                   "advice": advice_check(final),
                   "nudged": nudged, "still_a_plan": still_a_plan,
                   "figure_years_not_asked": wrong_year(question, tool_text),
                   "metrics_not_looked_up": sorted(metrics_named(question) - metrics_looked_up(steps))},
        "steps": steps, "messages": messages_to_dict(msgs),
    }
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return record


def show(r):
    print(f"Q: {r['question']}\nrun {r['run_id']}, {r['model']}, {r['seconds']} s, {r['model_calls']} model calls, "
          f"{r['prompt_tokens']} prompt tokens in all, stopped by: {r['stopped_by']}\n")
    n = 0
    for s in r["steps"]:
        if s["kind"] == "model":
            n += 1
            calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in s["tool_calls"])
            print(f"  model call {n} ({s['prompt_tokens']} tokens in): " + (f"asks for {calls}" if calls else "no tool call"))
            if s["text"]:
                print("    text: " + s["text"][:300].replace("\n", " "))
        else:
            print(f"  tool {s['name']} -> {s['output'][:200]}")
    print(f"\nA: {r['answer'] or '(none)'}\n")
    c = r["checks"]
    print("Number check: " + (f"NOT from any tool: {', '.join(c['numbers_not_from_a_tool'])}"
                              if c["numbers_not_from_a_tool"] else "every number came from a tool"))
    print("Advice check: " + (f"FAIL, {c['advice']['found']}" if not c["advice"]["ok"] else "pass"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # the rupee sign does not print in the Windows code page
    q = sys.argv[1] if len(sys.argv) > 1 else "What was Stylam's net cash flow from operating activities in FY26?"
    show(run(q))
