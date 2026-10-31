"""Step 4: does the fact reach the model? Children only against parent pages, on the golden set.

eval_retrieval.py counts a fact as found when a returned chunk sits on a page that carries it. That is
generous: the model is sent text, not pages. Here a fact counts as delivered only when the text actually
sent contains all of the fact's strings (same GROUPS). Search is the same for every row (rerank, +recent).

    children k      the k best chunks, as plain RAG sends them
    parents n/B/P   the pages of the n best chunks, each cut to P characters around its chunk, added
                    best first until B characters are used (answer.parents)

Writes context_summary.txt and context_results.csv.
"""
import csv
import os
import statistics
import time

from answer import parents
from eval_retrieval import GROUPS, HERE, squash
from retrieve import Searcher

SETTINGS = [("children", 5, None, None), ("children", 10, None, None),
            ("parents", 5, 8000, 4000), ("parents", 5, 8000, 1600), ("parents", 5, 10**9, 4000),
            ("parents", 10, 16000, 4000), ("parents", 10, 16000, 1600)]


def delivered(facts, sent):
    """facts: GROUPS[qid]; sent: [(doc_id, squashed text)]. One True or False per fact."""
    return [any(d.startswith(prefix) and all(t in text for t in terms)
                for prefix, terms in alternatives for d, text in sent) for alternatives in facts]


def main():
    questions = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig"))}
    s = Searcher("v1")
    t = time.time()
    hits = {qid: s.search(questions[qid]["question"], n=10) for qid in GROUPS}
    print(f"search: {time.time() - t:.0f} s for {len(hits)} questions", flush=True)

    rows = []
    for kind, n, budget, page in SETTINGS:
        name = f"children {n}" if kind == "children" else f"parents {n}/{budget if budget < 10**9 else 'all'}/{page}"
        for qid, facts in GROUPS.items():
            if kind == "children":
                sent = [(s.chunks[i]["doc_id"], s.chunks[i]["text"]) for i in hits[qid][:n]]
            else:
                _, sources = parents(s, questions[qid]["question"], n, budget, page, children=hits[qid])
                sent = [(src["doc_id"], src["text"]) for src in sources]
            got = delivered(facts, [(d, squash(x)) for d, x in sent])
            rows.append({"setting": name, "id": qid, "type": questions[qid]["type"], "facts": len(facts),
                         "delivered": sum(got), "any": int(any(got)), "pieces": len(sent),
                         "chars": sum(len(x) for _, x in sent)})

    with open(os.path.join(HERE, "context_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    types = sorted({r["type"] for r in rows})
    lines = [f"What reaches the model: {len(GROUPS)} golden-set questions, search = rerank +recent (top 10). "
             f"Written {time.strftime('%Y-%m-%d %H:%M')}.",
             "A fact counts only when the text sent contains all its strings.", "",
             f"{'setting':24} {'any fact':>8} {'recall':>6} {'pieces':>6} {'chars (median)':>14} {'max':>6}   "
             + " ".join(f"{t[:11]:>11}" for t in types)]
    for name in dict.fromkeys(r["setting"] for r in rows):
        rs = [r for r in rows if r["setting"] == name]
        recall = [r["delivered"] / r["facts"] for r in rs]
        by_type = [statistics.mean(r["delivered"] / r["facts"] for r in rs if r["type"] == t) for t in types]
        lines.append(f"{name:24} {statistics.mean(r['any'] for r in rs):8.2f} {statistics.mean(recall):6.2f} "
                     f"{statistics.mean(r['pieces'] for r in rs):6.1f} {statistics.median(r['chars'] for r in rs):14,.0f} "
                     f"{max(r['chars'] for r in rs):6,}   " + " ".join(f"{x:11.2f}" for x in by_type))
    lines.append("\nrecall by type: " + ", ".join(f"{t} ({sum(1 for q in GROUPS if questions[q]['type'] == t)})" for t in types))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "context_summary.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
