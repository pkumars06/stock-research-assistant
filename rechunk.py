"""Step 3: table-aware chunks (version 2), built from .store/pages.jsonl. The first chunks stay as they are.

Why (found 2026-10-20, Stylam FY26 AR PDF p174): version 1 cut pages by size, so the contingent-liability
table shared a chunk with provident-fund figures (one mixed vector), the next chunk repeated its rows
without the heading or the year columns, and the unit "₹ Lakh's" sat only in the page's first chunk.

Changes, for everything except call transcripts (still by speaker turn):
  - a note heading ("Note - 38 Commitment and Contingencies", "41 Contingent Liabilities") starts a new chunk
  - a note too long for one chunk is split at its sub-headings ("(b) Contingent liabilities"), then by line
  - every piece of a table carries the note's heading and its column line ("Particulars | 31-Mar-26 | 31-Mar-25")
    in the header, and a note that runs onto the next page carries them there too
  - the header names the unit (₹ lakh, ₹ crore, ₹ million), from the page, or for financial statements
    from the document's usual unit when the page's unit line was dropped as a running header
  - table blocks get no overlap (the carried heading and columns do that job); prose keeps the 150 characters

Writes .store/chunks_v2.jsonl with the same fields as version 1, including fiscal_year and printed_page.
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

from add_fields import fiscal_year, page_numbers
from ingest import (COMPANY, DOC_LABEL, MIN_CHUNK, OVERLAP, TARGET, fit_tokens, pack, split_long,
                    transcript_turns)

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")

NOTE = re.compile(r"^(note\s*[-–:.]?\s*\d+[a-z]?\b.*|\d{1,2}(\.\d{1,2})?\s+[A-Z][A-Za-z ,&()'’/-]{7,80})$", re.I)
SUBHEAD = re.compile(r"^\((?:[ivx]{1,4}|[a-h])\)\s+[A-Za-z][^|]{3,80}$")
COLUMNS = re.compile(r"\|.*(20\d\d|\d\d-[a-z]{3}-\d\d|fy\s?\d\d|march|mar\b|q[1-4])", re.I)
UNIT = re.compile(r"(?:₹|`|\brs\.?|\binr\b|amounts?|figures?|in)\W[^\n]{0,30}?\b(lakhs?|lakh[’']s|crores?|cr\b|millions?|mn\b)",
                  re.I)
UNIT_NAME = {"lakh": "₹ lakh", "crore": "₹ crore", "cr": "₹ crore", "million": "₹ million", "mn": "₹ million"}
MIN_BLOCK = 200
FINANCIAL = ("Balance sheet", "Profit and loss", "Cash flow statement", "Notes to accounts", "Changes in equity")


def unit_of(lines):
    for l in lines:
        if len(l) < 160:
            m = UNIT.search(l)
            if m:
                word = re.sub(r"[’']s$|s$", "", m.group(1).lower())
                return UNIT_NAME.get(word)
    return None


def is_financial(section):
    return section.startswith(FINANCIAL)


def blocks_of(lines):
    """Split a page at note headings. The first block may continue the previous page's note."""
    blocks, cur = [], []
    for l in lines:
        # a block under MIN_BLOCK characters is not cut off: quarterly results number their rows
        # ("45 Expenses"), and a run of tiny blocks would be dropped as too short to keep
        if NOTE.match(l) and "|" not in l and sum(len(x) for x in cur) >= MIN_BLOCK:
            blocks.append(cur)
            cur = []
        cur.append(l)
    if cur:
        blocks.append(cur)
    return blocks


def tableish(lines):
    return sum("|" in l for l in lines) >= 0.3 * len(lines)


def pieces_of(block):
    """Pack a table block by sub-heading, then by line, without overlap. Returns lists of lines."""
    subs, cur = [], []
    for l in block:
        if SUBHEAD.match(l) and cur:
            subs.append(cur)
            cur = []
        cur.append(l)
    if cur:
        subs.append(cur)
    pieces, cur = [], []
    for sub in subs:
        lines = [p for l in sub for p in (split_long(l, TARGET) if len(l) > TARGET else [l])]
        if cur and sum(len(l) + 1 for l in cur + lines) > TARGET:
            pieces.append(cur)
            cur = []
        for l in lines:
            if cur and sum(len(x) + 1 for x in cur) + len(l) > TARGET:
                pieces.append(cur)
                cur = []
            cur.append(l)
    if cur:
        pieces.append(cur)
    return pieces


def header(doc, section, pages, unit, note, columns):
    when = doc["period"] + (f", filed {doc['date']}" if doc["date"] else "")
    pg = f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
    parts = [COMPANY[doc["stock"]], f"{DOC_LABEL[doc['doc_type']]} {when}", section, pg]
    if unit:
        parts.append(f"amounts in {unit}")
    if note:
        parts.append(note[:90])
    if columns:
        parts.append("columns: " + columns[:90])
    return "[" + " | ".join(parts) + "]"


def main():
    docs = {r["doc_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
            if r["status"] == "kept"}
    pages = defaultdict(list)
    for line in open(os.path.join(STORE, "pages.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        rec["lines"] = rec["text"].split("\n")
        pages[rec["doc_id"]].append(rec)
    printed, _ = page_numbers()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tok.model_max_length = 100_000

    def count(text):
        return len(tok(text)["input_ids"])

    rows, stats = [], Counter()
    for doc_id, doc in docs.items():
        doc_pages = pages[doc_id]
        # the document's usual unit in its financial statements, for pages whose unit line was dropped
        usual = Counter(unit_of(p["lines"]) for p in doc_pages if is_financial(p["section"]))
        usual.pop(None, None)
        usual = usual.most_common(1)[0][0] if usual else None

        made = []   # (lines, pages, section, unit, note, columns)
        if doc["doc_type"] == "transcript":
            sec = {p["page"]: p["section"] for p in doc_pages}
            for parts, pgs in pack(transcript_turns(doc_pages), overlap=0):
                made.append((parts, pgs, sec[pgs[0]], None, None, None))
        else:
            note = columns = None
            last_section = None
            for p in doc_pages:
                section = p["section"]
                if section != last_section:
                    note = columns = None
                last_section = section
                unit = unit_of(p["lines"]) or (usual if is_financial(section) else None)
                stats["unit " + ("on page" if unit_of(p["lines"]) else "from document" if unit else "none")] += 1
                for i, block in enumerate(blocks_of(p["lines"])):
                    if NOTE.match(block[0]) and "|" not in block[0]:
                        note, columns = block[0], None
                    elif i == 0 and not is_financial(section):
                        note = columns = None
                    col = next((l for l in block if COLUMNS.search(l)), None)
                    columns = col or columns
                    if tableish(block):
                        stats["table blocks"] += 1
                        for piece in pieces_of(block):
                            made.append((piece, [p["page"]], section, unit, note, columns))
                    else:
                        for parts, pgs in pack([(l, p["page"]) for l in block], OVERLAP):
                            made.append((parts, pgs, section, unit, note if is_financial(section) else None, None))

        # a piece too short to keep joins the piece before it on the same page, rather than being dropped
        merged = []
        for m in made:
            if merged and sum(len(l) for l in m[0]) < MIN_CHUNK and merged[-1][1][-1] == m[1][0]:
                prev = merged[-1]
                merged[-1] = (prev[0] + m[0], sorted(set(prev[1]) | set(m[1])), *prev[2:])
            else:
                merged.append(m)
        made = merged

        n = 0
        for parts, pgs, section, unit, note, columns in made:
            # the heading and columns are in the header; drop them from the body only if repeated
            head = header(doc, section, pgs, unit, note, columns)
            for fparts, fpgs in fit_tokens(parts, pgs, head, count):
                body = "\n".join(fparts)
                if len(body) < MIN_CHUNK:
                    continue
                fy = fiscal_year(doc["period"], doc["date"])
                label, how = printed[(doc_id, fpgs[0])]
                rows.append({"chunk_id": f"{doc_id}-v2-{n:04d}", "doc_id": doc_id, "stock": doc["stock"],
                             "doc_type": doc["doc_type"], "period": doc["period"], "date": doc["date"],
                             "section": section, "pages": fpgs, "ocr": False, "url": doc["url"],
                             "header": header(doc, section, fpgs, unit, note, columns), "text": body,
                             "fy": fy, "fiscal_year": f"FY{fy:02}" if fy is not None else "",
                             "printed_page": label or "", "printed_how": how})
                n += 1

    ocr_pages = {(p["doc_id"], p["page"]) for ps in pages.values() for p in ps if p["ocr"]}
    for r in rows:
        r["ocr"] = any((r["doc_id"], pg) in ocr_pages for pg in r["pages"])
    with open(os.path.join(STORE, "chunks_v2.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # nothing lost: every line of version 1 should still be in some version 2 chunk
    v1 = [json.loads(l) for l in open(os.path.join(STORE, "chunks.jsonl"), encoding="utf-8")]
    lines_v1 = {(c["doc_id"], l) for c in v1 for l in c["text"].split("\n")}
    lines_v2 = {(r["doc_id"], l) for r in rows for l in r["text"].split("\n")}
    lost = lines_v1 - lines_v2
    print(f"lines in version 1 missing from version 2: {len(lost):,} of {len(lines_v1):,}")
    for doc_id, l in list(lost)[:8]:
        print("   ", doc_id, l[:100])

    lengths = [count(r["header"] + "\n" + r["text"]) for r in rows]
    lengths.sort()
    print(f"version 2: {len(rows):,} chunks (version 1: 12,485)")
    print(f"tokens with header: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, over 384: {sum(l > 384 for l in lengths)}")
    print(dict(stats))
    print("chunks with a unit in the header:", sum("amounts in" in r["header"] for r in rows),
          "| with a note heading:", sum("columns:" in r["header"] or "Note" in r["header"] for r in rows))


if __name__ == "__main__":
    main()
