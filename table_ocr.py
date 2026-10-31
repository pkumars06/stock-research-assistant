"""Read a sideways table page by OCR, row by row (step 4, check 6).

Stylam's related-party note (FY26 AR, PDF p180-181) is a landscape table printed on portrait pages.
Its text layer reads column by column and fuses figures ("13.280.05" is 13.28 and 0.05). So:
1. Find which way the text runs (PyMuPDF gives each line a direction); turn the image upright.
2. OCR it (RapidOCR, as in ingest.py), at 200 dpi.
3. Rebuild rows: boxes whose vertical centres are within half a line height are one row, read left to right,
   cells joined with " | ".
Results are cached in .store/table_ocr_cache.json.

python table_ocr.py Stylam-036 180 181   print the rebuilt rows
"""
import csv
import json
import os
import re
import sys

import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".store", "table_ocr_cache.json")
DPI = 200
_ocr = None


def turns(page):
    """Quarter-turns (np.rot90 k) that make the page's text upright, from the direction most lines run in."""
    dirs = {}
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            d = tuple(round(x) for x in line["dir"])
            dirs[d] = dirs.get(d, 0) + 1
    main = max(dirs, key=dirs.get) if dirs else (1, 0)
    return {(1, 0): 0, (0, -1): 3, (-1, 0): 2, (0, 1): 1}.get(main, 0)   # (0, -1): runs upwards


YEAR_HEAD = re.compile(r"march\s*31\s*,?\s*20(\d\d)", re.I)
FIGURE = re.compile(r"^\(?-?[\d,]+\.\d\d\)?$")


def rows(boxes):
    """OCR boxes [(corners, text, confidence)] -> lines of text, top to bottom, cells left to right.
    A figure is tagged with the year of the nearest "March 31, 20yy" heading above it whose left edge is
    left of the figure (the two year blocks sit side by side), because empty cells ("-") are not read."""
    items = []
    for corners, text, _ in boxes:
        ys = [p[1] for p in corners]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(p[0] for p in corners), text))
    items.sort()
    grouped, current, centre, height = [], [], None, None
    for y, h, x, text in items:
        if current and abs(y - centre) > height / 2:
            grouped.append(current)
            current = []
        if not current:
            centre, height = y, h
        current.append((x, text))
    if current:
        grouped.append(current)
    out, heads = [], []          # heads: [(left x, "FYyy")] from the latest heading row
    for r in grouped:
        r.sort()
        found = [(x, "FY" + m.group(1)) for x, t in r for m in [YEAR_HEAD.search(t)] if m]
        if len(found) >= 2:
            heads = found
        cells = []
        for x, t in r:
            year = next((fy for hx, fy in sorted(heads, reverse=True) if x >= hx - 20), None) if heads else None
            cells.append(f"{year}: {t}" if year and FIGURE.match(t) else t)
        out.append(" | ".join(cells))
    return out


def page_rows(doc_id, page_no):
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    key = f"{doc_id}:{page_no}"
    if key not in cache:
        global _ocr
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        path = next(r["path"] for r in csv.DictReader(open(os.path.join(HERE, "docs.csv"), encoding="utf-8"))
                    if r["doc_id"] == doc_id)
        page = pymupdf.open(path)[page_no - 1]
        pix = page.get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        boxes, _ = _ocr(np.ascontiguousarray(np.rot90(img, turns(page))))
        cache[key] = rows(boxes or [])
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return cache[key]


if __name__ == "__main__":
    doc_id, nums = sys.argv[1], [int(x) for x in sys.argv[2:]]
    for n in nums:
        print(f"=== {doc_id} PDF p{n}")
        print("\n".join(page_rows(doc_id, n)))
