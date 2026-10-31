"""Step 3: score every search method on the golden set.

What counts as found: each question needs one or more facts (GROUPS below). A fact is found when a
returned chunk sits on a page that carries it: a page of the right document that contains all of the
fact's strings (the answer's own figures, or a phrase from the passage). So a figure that appears on the
statement, a note and the ten-year table can be found on any of them.

    hit@k     at least one fact found in the top k chunks
    recall@k  share of the question's facts found in the top k (below 1 for a half-found multi-hop)
    MRR       1 / rank of the first chunk that carries any fact, counted to rank 10 (0 if none)

The 4 not-in-filings questions have no page and are scored at the answer stage (step 5).
Writes retrieval_results.csv (one row per question, method and filter) and retrieval_summary.txt.
`python eval_retrieval.py v2` scores the table-aware chunks and writes the same files with "_v2".
"""
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

from retrieve import STORE, Searcher, parse_query

HERE = os.path.dirname(os.path.abspath(__file__))

# question -> facts; a fact is a list of alternatives (doc_id prefix, [strings all on one page])
GROUPS = {
    "S01": [[("Stylam-036", ["mittal goel", "independent auditor"]), ("Stylam-036", ["mittal goel", "17,708.35"])]],
    "S02": [[("Stylam-036", ["17,708.35", "10,746.12"])]],
    "S03": [[("Stylam-036", ["17,708.35"])], [("Stylam-036", ["14,947.35"])]],
    "S04": [[("Stylam-036", ["1,12,929.39", "1,02,508.95"]), ("Stylam-036", ["revenue", "| 1,025 | 1,129"])]],
    "S05": [[("Stylam-036", ["1,544.67", "17.05"])]],
    # Added 2026-10-20 after reading the misses: the Q4 FY26 deck's sales chart (Stylam-028 p19) gives
    # export 831 / domestic 298 crore for FY26 and 731 / 294 for FY25, so it answers S06 and S14 too.
    "S06": [[("Stylam-036", ["83,077.45", "1,12,929.39"]), ("Stylam-028", ["831", "298", "domestic", "exports"])]],
    "S07": [[("Stylam-036", ["1,06,451.25"])], [("Stylam-036", ["83,077.45"])]],
    "S08": [[("Stylam", ["2,250", "44,06,496"])]],
    "S09": [[("Stylam-008", ["full production in new plant"])],
            [("Stylam-023", ["september 2025 and now we are saying march 2026"])],
            [("Stylam-029", ["maximum middle of july"])],
            [("Stylam-037", ["commercial production", "september 22, 2026"])]],
    "S10": [[("Stylam-036", ["cwip", "| 83 | 221"])]],
    "S11": [[("Stylam-036", ["88.02", "71.43"]), ("Stylam-036", ["eps |", "| 71 | 88"])]],
    "S12": [[("Stylam-032", ["6,504"])]],
    "S14": [[("Stylam-036", ["29,851.95", "83,077.45"]), ("Stylam-028", ["831", "731", "298", "294", "domestic"])]],
    "P01": [[("Speciality-054", ["334.38"]), ("Speciality", ["labour code", "exceptional"])]],
    "P02": [[("Speciality-068", ["14,50,09,700"])]],
    "P03": [[("Speciality", ["25,03,60,440"])], [("Speciality-068", ["14,50,09,700"])]],
    "P04": [[("Speciality-066", ["8,432.57"])]],
    "P05": [[("Speciality-066", ["8,432.57"])], [("Speciality-066", ["2,295.00"])]],
    "P06": [[("Speciality-066", ["1,306.90", "787.54"])]],
    "P07": [[("Speciality-058", ["69.99%"])]],
    "P08": [[("Speciality", ["11.35%"])]],
    "P09": [[("Speciality-059", ["15 new walters", "5 crores"])]],
    "P10": [[("Speciality-046", ["44.94", "situations advertising"])]],
    "P11": [[("Speciality-004", ["334.11", "955.78"]), ("Speciality-005", ["334.11", "955.78"])]],
    "P12": [[("Speciality", ["2,810.16"])]],
    "P13": [[("Speciality-059", ["15%", "600 crores"])], [("Speciality-064", ["16.8%"])]],
    "P15": [[("Speciality", ["2,945.23", "2,810.16"])]],
    "X01": [[("Stylam-036", ["17,708.35"])], [("Speciality-066", ["8,432.57"])]],
    "B01": [[("Stylam-036", ["countries served globally"]), ("Stylam", ["decorative laminates", "80 countries"])],
            [("Stylam-036", ["83,077.45"]), ("Stylam-028", ["831", "298", "domestic", "exports"])]],
    "B02": [[("Stylam-036", ["zero", "debt company"])], [("Stylam-036", ["2,908.90"])], [("Stylam-036", ["6,521.04"])]],
    "B03": [[("Speciality-066", ["10,450.37", "3,544.08"])]],
    "B04": [[("Stylam-036", ["1,544.67"])], [("Stylam-036", ["80,679.77"])]],
    "B05": [[("Speciality-066", ["2,899.00"])], [("Speciality-066", ["34,700.44"])]],
    "B06": [[("Speciality-066", ["true and fair view", "key audit matter"])]],
    "B07": [[("Speciality-066", ["495.53", "situations advertising"])]],
    "B08": [[("Stylam-036", ["20,289.38", "5,228.55"])], [("Stylam-036", ["313.36"])]],
}

METHODS = ["dense", "keyword", "hybrid", "rerank"]
FILTERS = {"none": lambda q: None,
           "company": lambda q: {"stocks": parse_query(q)["stocks"], "fy": None},
           "company+year": parse_query,
           "+recent": lambda q: parse_query(q, default_recent=True)}
KS = (5, 10)


def squash(text):
    return re.sub(r"\s+", " ", text).lower()


def fact_pages():
    """question -> list of facts, each a set of (doc_id, page) that carry it."""
    pages = [json.loads(l) for l in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8")]
    pages = [(p["doc_id"], p["page"], squash(p["text"])) for p in pages]
    out = {}
    for qid, facts in GROUPS.items():
        out[qid] = []
        for alternatives in facts:
            found = {(d, n) for prefix, terms in alternatives for d, n, text in pages
                     if d.startswith(prefix) and all(t in text for t in terms)}
            assert found, f"{qid}: no page carries {alternatives}"
            out[qid].append(found)
    return out


def score(ranked_pages, facts):
    """ranked_pages: one set of (doc_id, page) per returned chunk, best first."""
    row = {}
    for k in KS:
        got = [any(f & pages for pages in ranked_pages[:k]) for f in facts]
        row[f"hit@{k}"] = int(any(got))
        row[f"recall@{k}"] = sum(got) / len(facts)
    first = next((r for r, pages in enumerate(ranked_pages[:10], 1) if any(f & pages for f in facts)), None)
    row["mrr"] = 1 / first if first else 0.0
    row["first_rank"] = first or ""
    return row


def main():
    questions = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "golden_set.csv"), encoding="utf-8-sig"))}
    facts = fact_pages()
    for qid, f in facts.items():
        print(qid, [len(x) for x in f], "pages per fact")
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    out = "" if version == "v1" else "_" + version
    s = Searcher(version)
    rows = []
    for method in METHODS:
        for fname, make in FILTERS.items():
            t = time.time()
            for qid in GROUPS:
                q = questions[qid]
                ranked = s.__getattribute__(method)(q["question"], make(q["question"]), 10)
                ranked_pages = [{(s.chunks[i]["doc_id"], p) for p in s.chunks[i]["pages"]} for i in ranked]
                row = {"id": qid, "type": q["type"], "method": method, "filter": fname,
                       "facts": len(facts[qid]), **score(ranked_pages, facts[qid]),
                       "top3": " ; ".join(f"{s.chunks[i]['doc_id']} p{s.chunks[i]['pages'][0]}" for i in ranked[:3])}
                rows.append(row)
            print(f"{method:8} {fname:13} {time.time() - t:5.0f} s", flush=True)

    with open(os.path.join(HERE, f"retrieval_results{out}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    lines = [f"Retrieval on {len(GROUPS)} golden-set questions (not-in-filings excluded). "
             f"Chunks {version}: {len(s.chunks)}, mpnet with header. Written {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    head = f"{'method':8} {'filter':13} {'hit@5':>6} {'recall@5':>8} {'hit@10':>6} {'recall@10':>9} {'MRR':>5}"
    lines += ["All questions", head]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["filter"])].append(r)
    for (m, fn), rs in groups.items():
        n = len(rs)
        lines.append(f"{m:8} {fn:13} {sum(r['hit@5'] for r in rs) / n:6.2f} {sum(r['recall@5'] for r in rs) / n:8.2f} "
                     f"{sum(r['hit@10'] for r in rs) / n:6.2f} {sum(r['recall@10'] for r in rs) / n:9.2f} "
                     f"{sum(r['mrr'] for r in rs) / n:5.2f}")
    types = sorted({r["type"] for r in rows})
    lines += ["", "recall@5 by question type (count in brackets)",
              f"{'method':8} {'filter':13} " + " ".join(f"{t[:11]:>11}({sum(1 for q in GROUPS if questions[q]['type'] == t)})" for t in types)]
    for (m, fn), rs in groups.items():
        cells = []
        for t in types:
            sub = [r for r in rs if r["type"] == t]
            cells.append(f"{sum(r['recall@5'] for r in sub) / len(sub):14.2f}")
        lines.append(f"{m:8} {fn:13} " + " ".join(cells))
    text = "\n".join(lines)
    print("\n" + text)
    open(os.path.join(HERE, f"retrieval_summary{out}.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
