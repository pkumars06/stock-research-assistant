"""Find the pages that contain every given word, to locate the source page of a golden-set answer.

Usage:
    python find_pages.py "contingent liabilities" "15.62" --doc Stylam-036
    python find_pages.py "cash generated from operations" --doc Speciality-066 --show 600

Matching is case-insensitive substring matching on the page text, after squashing runs of spaces.
"""
import argparse
import json
import re
from pathlib import Path

PAGES = Path(__file__).parent / ".store" / "pages.jsonl"


def squash(text):
    return re.sub(r"\s+", " ", text).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--doc", help="doc_id prefix, for example Stylam-036 or Speciality")
    ap.add_argument("--show", type=int, default=0, help="characters of context to print around the first term")
    ap.add_argument("--max", type=int, default=15)
    args = ap.parse_args()

    terms = [t.lower() for t in args.terms]
    hits = 0
    with PAGES.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.doc and not rec["doc_id"].startswith(args.doc):
                continue
            text = squash(rec["text"])
            if all(t in text for t in terms):
                hits += 1
                print(f"{rec['doc_id']} p{rec['page']} [{rec['section']}]")
                if args.show:
                    i = text.find(terms[0])
                    print("   ...", text[max(0, i - args.show // 3): i + args.show], "...\n")
                if hits >= args.max:
                    break
    print(f"{hits} page(s)")


if __name__ == "__main__":
    main()
