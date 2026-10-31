"""Step 4: a cited answer from the filings. Search small, send big (parent-child, guide 2 Part 7).

1. Search (rerank, filters from the question) finds the best small chunks: the children.
2. Each child is swapped for the page it sits on: the parent. Two children on one page send that page once.
   A page longer than PAGE_CHARS is cut to a window around the child (a few pages are 29,000 characters).
3. The pages go to call_llm() as numbered sources, best first, until the budget is spent.
4. The answer must cite [n]; with no answer in the sources it says "Insufficient evidence".
5. A check after the answer: every number in it must appear in a source it was given (guardrail, guide 2 Part 10).
   A number that appears nowhere is reported, not hidden.

python answer.py "question"          one question (local Qwen unless a key is set; see llm.py)
python answer.py --show "question"   print the sources only, no LLM call
"""
import json
import os
import re
import sys

from llm import BACKEND, call_llm
from retrieve import STORE, Searcher, parse_query

# measured 2026-10-21 (eval_context.py, context_summary.txt): of the settings tried, 10 children each widened
# to 1,600 characters of their page delivered the most facts (recall 0.57; 5 whole pages 0.43, 10 children 0.56)
N_CHILDREN = 10
PAGE_CHARS = 1600
BUDGET_CHARS = 16000   # about 5,000 tokens of sources

SYSTEM = """You answer questions about Indian listed companies using ONLY the numbered sources given.
Rules:
- Every sentence with a fact or a number ends with the source it came from, like [2].
- Copy numbers exactly as the source prints them, with their unit (lakh, crore, million). Do not convert or round.
- Say whether a figure is standalone or consolidated when the source says so.
- If the sources do not contain the answer, reply exactly: Insufficient evidence in the filings provided.
- Explain, do not advise: never say buy, sell, hold, undervalued, overvalued or give a target price.
- Answer in plain, short sentences."""

_pages = None


def pages():
    """(doc_id, pdf page) -> page text, read once."""
    global _pages
    if _pages is None:
        _pages = {}
        for line in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            _pages[(r["doc_id"], r["page"])] = r["text"]
    return _pages


def window(page_text, child_text, size=PAGE_CHARS):
    """The whole page if it fits, otherwise `size` characters centred on where the child sits."""
    if len(page_text) <= size:
        return page_text
    at = page_text.find(child_text[:80])
    centre = at + len(child_text) // 2 if at >= 0 else 0
    start = max(0, min(centre - size // 2, len(page_text) - size))
    return page_text[start:start + size]


def cite(c, page):
    name = " ".join(c["header"][1:-1].split(" | ")[:2])           # "Stylam Industries Annual report FY26"
    label = f"{name}, PDF p{page}"
    if c.get("printed_page") and page == c["pages"][0] and c.get("printed_how") == "read":
        label += f" (printed {c['printed_page']})"
    return label


def parents(searcher, question, n=N_CHILDREN, budget=BUDGET_CHARS, page_chars=PAGE_CHARS, children=None):
    """The children found, and the parent pages to send: [{"n", "source", "url", "text", "children"}]."""
    children = children[:n] if children is not None else searcher.search(question, n=n)
    sources, by_page, used = [], {}, 0
    for rank, i in enumerate(children, 1):
        c = searcher.chunks[i]
        key = (c["doc_id"], c["pages"][0])
        if key in by_page:
            by_page[key]["children"].append(rank)
            continue
        text = window(pages().get(key, c["text"]), c["text"], page_chars)
        if used + len(text) > budget and sources:
            continue                                   # over budget: keep the better-ranked pages already in
        used += len(text)
        src = {"n": len(sources) + 1, "doc_id": c["doc_id"], "page": key[1], "source": cite(c, key[1]), "url": c["url"], "text": text,
               "children": [rank], "child_chars": len(c["text"])}
        by_page[key] = src
        sources.append(src)
    return children, sources


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text):
    """Numbers with two or more digits, commas removed: '1,129.4' -> '1129.4'. Citation marks [3] are skipped."""
    text = re.sub(r"\[\d+\]", "", text)
    found = {m.group().replace(",", "").rstrip(".") for m in NUMBER.finditer(text)}
    return {x for x in found if len(x.replace(".", "")) >= 2}


def unsupported_numbers(answer, sources):
    given = set()
    for s in sources:
        given |= numbers(s["text"])
    return sorted(numbers(answer) - given)


def answer(question, searcher):
    _, sources = parents(searcher, question)
    block = "\n\n".join(f"[{s['n']}] {s['source']}\n{s['text']}" for s in sources)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"SOURCES:\n{block}\n\nQUESTION: {question}"}]
    r = call_llm(messages, max_tokens=250, purpose="answer")
    return {"question": question, "filters": parse_query(question, default_recent=True), "answer": r["text"],
            "sources": sources, "unsupported_numbers": unsupported_numbers(r["text"], sources),
            "llm": {k: v for k, v in r.items() if k != "text"}}


def show(result):
    print(f"Q: {result['question']}\nfilters: {result['filters']}\n")
    print(f"A ({result['llm']['model']}, {result['llm']['prompt_tokens']} tokens in, "
          f"{result['llm']['seconds']} s):\n{result['answer']}\n")
    bad = result["unsupported_numbers"]
    print("Number check: " + (f"NOT in any source: {', '.join(bad)}" if bad else "every number appears in a source"))
    print("\nSources sent:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['source']}  (children ranked {s['children']}, {len(s['text'])} characters)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    q = args[0] if args else "Why did Speciality report an exceptional item in FY26?"
    s = Searcher("v1")
    if "--show" in sys.argv:
        children, sources = parents(s, q)
        sent_children = sum(len(s.chunks[i]["text"]) for i in children)
        print(f"{len(children)} children, {sent_children} characters; "
              f"{len(sources)} parent pages, {sum(len(x['text']) for x in sources)} characters\n")
        for x in sources:
            print(f"[{x['n']}] {x['source']}  children {x['children']}\n{x['text'][:600]}\n...\n")
    else:
        show(answer(q, s))
