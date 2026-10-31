"""Step 3, first part: add fiscal_year and printed_page to every chunk, without re-embedding.

fiscal_year  the Indian financial year a chunk belongs to, as "FY26" plus an int fy = 26 for range filters.
             From the period ("FY26", "Q1 FY27"); a filing with only a date (governance, orders) gets the
             year its date falls in: April 2025 to March 2026 is FY26.
printed_page the page number printed on the paper page, which is what a reader finds (found 2026-10-19:
             Stylam FY26 AR PDF p151 is printed 168). Read from the first two and last three lines of the
             page. A number is kept only if it fits the document's usual offset (printed minus PDF page),
             so a stray "58" at the end of a table is not taken for a page number. Pages with no number
             in a document with a steady offset get PDF page + offset, marked inferred.

Rewrites .store/chunks.jsonl in place and updates the metadata in the ChromaDB collection "filings".
"""
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
NUMBER = re.compile(r"^\d{1,3}$")
SPREAD = re.compile(r"^(\d{1,3}) \| (\d{1,3})$")   # a two-page spread, "02 | 03"
# a number beside a short label: "Directors' Report | 057", "58 | Stylam Industries Limited"
LABELLED = re.compile(r"^(?:[^|\d][^|]{0,59} \| (\d{1,3})|(\d{1,3}) \| [^|\d][^|]{0,59})$")
STEADY = 0.6      # an offset is the document's own if it explains this share of the numbers found
MIN_FOUND = 10    # and at least this many pages had a number


def fiscal_year(period, date):
    m = re.search(r"FY(\d\d)", period or "")
    if m:
        return int(m.group(1))
    if date:
        y, mo = int(date[:4]), int(date[5:7])
        return (y + 1 if mo >= 4 else y) % 100
    return None


def printed_numbers(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    found = []
    for l in lines[:2] + lines[-3:]:
        m = SPREAD.match(l)
        if m:
            found.append((int(m.group(1)), m.group(0)))
            continue
        m = NUMBER.match(l) or LABELLED.match(l)
        if m:
            n = next((g for g in m.groups() if g), m.group(0))
            found.append((int(n), n.lstrip("0") or "0"))
    return found


def page_numbers():
    """(doc_id, pdf page) -> (printed label or None, how it was found)."""
    by_doc = defaultdict(list)
    for line in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        by_doc[rec["doc_id"]].append((rec["page"], printed_numbers(rec["text"])))

    result, report = {}, []
    for doc, pages in by_doc.items():
        offsets = Counter(n - p for p, found in pages for n, _ in found)
        total = sum(offsets.values())
        offset, share = (offsets.most_common(1)[0] if offsets else (None, 0))
        steady = total >= MIN_FOUND and share / total >= STEADY
        counts = Counter()
        read = {}   # pdf page -> first printed number on it, for pages whose number we trust
        for p, found in pages:
            spreads = [(n, label) for n, label in found if "|" in label]
            fits = spreads or [(n, label) for n, label in found if steady and n - p == offset]
            if fits:
                read[p] = fits[0][0]
                result[(doc, p)] = (fits[0][1].replace(" | ", "-"), "read")
        for p, _ in pages:
            if (doc, p) in result:
                pass
            elif read:
                # the nearest page with a trusted number gives the local offset (spreads shift it)
                near = min(read, key=lambda q: (abs(q - p), q > p))
                result[(doc, p)] = (str(read[near] + p - near), "inferred")
            else:
                result[(doc, p)] = (None, "none")
            counts[result[(doc, p)][1]] += 1
        report.append((doc, len(pages), offset if steady else None, dict(counts)))
    return result, report


def main():
    numbers, report = page_numbers()
    path = os.path.join(STORE, "chunks.jsonl")
    chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
    for c in chunks:
        fy = fiscal_year(c["period"], c["date"])
        c["fy"] = fy
        c["fiscal_year"] = f"FY{fy:02}" if fy is not None else ""
        label, how = numbers[(c["doc_id"], c["pages"][0])]
        c["printed_page"] = label or ""
        c["printed_how"] = how
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    import chromadb
    col = chromadb.PersistentClient(path=os.path.join(STORE, "chroma")).get_collection("filings")
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = list(by_id)
    for i in range(0, len(ids), 2000):
        batch = ids[i:i + 2000]
        metas = col.get(ids=batch, include=["metadatas"])
        new = []
        for cid, m in zip(metas["ids"], metas["metadatas"]):
            c = by_id[cid]
            m = dict(m, fy=c["fy"] if c["fy"] is not None else -1, fiscal_year=c["fiscal_year"],
                     printed_page=c["printed_page"])
            new.append((cid, m))
        col.update(ids=[x[0] for x in new], metadatas=[x[1] for x in new])

    print("fiscal_year:", sorted(Counter(c["fiscal_year"] for c in chunks).items()))
    print("printed_page:", dict(Counter(c["printed_how"] for c in chunks)))
    print("\ndoc, pages, offset (printed - PDF), how each page got its number")
    for row in report:
        print(*row, sep="  ")


if __name__ == "__main__":
    main()
