"""Step 3: the search methods, measured by eval_retrieval.py and reused by the app in step 4.

    dense    mpnet vectors, cosine similarity (exact, with numpy; the app's ChromaDB uses HNSW)
    bm25     keyword scores (rank-bm25) over header + text
    hybrid   dense top 50 and BM25 top 50 merged with reciprocal rank fusion (RRF, constant 60)
    rerank   hybrid top 30 re-scored by the cross-encoder ms-marco-MiniLM-L-6-v2

Filters come from the question (parse_query): the company named, and a window of financial years
around any year named. The window is one year either side, because the answer to "FY25" is often in
the FY26 report (last year's column) and a guide for FY27 is given on a call in FY26.
A bare year means the Indian financial year: "2026" = FY26 = April 2025 to March 2026.
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
FUSE_DEPTH = 50     # how deep each list goes into the RRF merge
RERANK_DEPTH = 30   # how many hybrid results the cross-encoder re-scores
YEAR_WINDOW = 1

STOCKS = {"stylam": "Stylam", "speciality": "Speciality", "specialty": "Speciality"}   # also the US spelling
QUANTITY = (r"tonnes?|tons?|units?|sheets?|sq|square|crores?|lakhs?|shares?|outlets?|restaurants?|employees?|"
            r"people|customers?|stores?|plants?|machines?|presses|%|percent")
MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
STOP = set("a an the of in on and or to for by with what was is are did does do how much many which "
           "its it from at as be this that their there than more any".split())


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


LATEST_FY = 27
RECENT = (LATEST_FY - 2, LATEST_FY)   # with no year named, a beginner means now: FY25 to FY27


def parse_query(question, default_recent=False):
    """The filters a question implies: {"stocks": [...], "fy": (low, high) or None}.
    default_recent: a question that names no year searches only RECENT, not the FY22-FY24 reports."""
    q = question.lower()
    stocks = sorted({name for key, name in STOCKS.items() if key in q})
    years = named_years(question)
    fy = (min(years) - YEAR_WINDOW, max(years) + YEAR_WINDOW) if years else (RECENT if default_recent else None)
    return {"stocks": stocks, "fy": fy}


def named_years(question):
    """The fiscal years a question names, as numbers (FY26 -> 26). Also used to check that an answer's figure
    is from the year asked (agent.py)."""
    q = question.lower()
    years = []
    # The Indian two-year form: "2023-24", "FY 2023-24", "FY23-24" all mean FY24 (April 2023 to March 2024).
    # Read first and blanked out, so "2023" is not read again as a bare year (it gave FY23; the review of
    # 2026-10-23). Consecutive years are one fiscal year; "2022-26" is a range and gives both ends, like
    # "FY22-FY26". Only years in the filings' span (FY15-FY30) count.
    def two_year(m):
        first, second = int(m.group(1)) % 100, int(m.group(2))
        plain_pair = len(m.group(1)) == 2 and not m.group(0).startswith("fy")    # "pages 20-21" is not a year
        if plain_pair or not (15 <= first <= 30 and 15 <= second <= 30 and second > first):
            return m.group(0)
        years.extend([second] if second == first + 1 else [first, second])
        return " "
    q = re.sub(r"\b(?:fy\s?)?((?:20)?\d\d)\s*[-–/]\s*(?:20)?(\d\d)\b", two_year, q)   # also "FY 2019-2020"
    # "FY2023" is FY23
    years += [int(y) for y in re.findall(r"\bfy\s?20(\d\d)\b", q)]
    q = re.sub(r"\bfy\s?20\d\d\b", " ", q)
    years += [int(y) for y in re.findall(r"fy\s?(\d\d)\b", q)]
    for month, year in re.findall(r"\b(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", q):
        y = int(year) % 100
        years.append(y + 1 if MONTHS[month] >= 4 else y)
    # a bare year ("in 2026") is the financial year ending in it; skip years already read with a month
    q_no_months = re.sub(r"\b(" + "|".join(MONTHS) + r")\s+20\d\d\b", "", q)
    # a bare 20xx is a year only within the filings' span and not before a quantity word ("did Stylam add 2000
    # tonnes" filtered to FY-1..FY1 and found nothing; the review of 2026-10-23)
    years += [int(y) % 100 for y in re.findall(r"\b(20[1-3]\d)\b(?!\s*(?:" + QUANTITY + r"))", q_no_months)
              if 2015 <= int(y) <= 2030]
    return years


class Searcher:
    def __init__(self, version="v1"):
        """version "v1": the first chunks (cut by size); "v2": table-aware chunks from rechunk.py."""
        suffix = "" if version == "v1" else "_" + version
        self.version = version
        self.chunks = [json.loads(l) for l in open(os.path.join(STORE, f"chunks{suffix}.jsonl"), encoding="utf-8")]
        self.texts = [c["header"] + "\n" + c["text"] for c in self.chunks]
        self.vecs = np.load(os.path.join(STORE, f"chunks{suffix}_mpnet.npy"))
        self.stock = np.array([c["stock"] for c in self.chunks])
        self.fy = np.array([c["fy"] if c["fy"] is not None else -1 for c in self.chunks])
        self._embedder = self._bm25 = self._reranker = None

    # --- the three models, loaded on first use -------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        return self._embedder

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokens(t) for t in self.texts])
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        return self._reranker

    # --- filters ----------------------------------------------------------------------------------
    def mask(self, filters):
        keep = np.ones(len(self.chunks), dtype=bool)
        if filters and filters.get("stocks"):
            keep &= np.isin(self.stock, filters["stocks"])
        if filters and filters.get("fy"):
            low, high = filters["fy"]
            keep &= (self.fy >= low) & (self.fy <= high)
        return keep

    @staticmethod
    def top(scores, keep, n):
        scores = np.where(keep, scores, -np.inf)
        idx = np.argsort(-scores)[:n]
        return [int(i) for i in idx if np.isfinite(scores[i])]

    # --- the methods; each returns chunk indices, best first ---------------------------------------
    def dense(self, question, filters=None, n=10):
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        return self.top(self.vecs @ q, self.mask(filters), n)

    def keyword(self, question, filters=None, n=10):
        return self.top(self.bm25.get_scores(tokens(question)), self.mask(filters), n)

    def hybrid(self, question, filters=None, n=10):
        fused = {}
        for ranked in (self.dense(question, filters, FUSE_DEPTH), self.keyword(question, filters, FUSE_DEPTH)):
            for rank, i in enumerate(ranked, 1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused, key=fused.get, reverse=True)[:n]

    def rerank(self, question, filters=None, n=10):
        pool = self.hybrid(question, filters, RERANK_DEPTH)
        scores = self.reranker.predict([(question, self.texts[i]) for i in pool], batch_size=16)
        return [pool[j] for j in np.argsort(-scores)[:n]]

    def search(self, question, method="rerank", use_filters=True, n=10):
        # recent-by-default measured best on 2026-10-20 (rerank hit@5 0.58 -> 0.64; retrieval_summary.txt)
        filters = parse_query(question, default_recent=True) if use_filters else None
        return getattr(self, method)(question, filters, n)
