"""Step 1, last part: embed every chunk with mpnet and load it into ChromaDB.

Reads   .store/chunks.jsonl (written by ingest.py)
Writes  .store/emb_parts/part_NNN.npy   1,000 chunks each, with part_NNN.sha256 (a hash of the texts embedded);
                                        a re-run skips a part only when its texts still hash the same
        .store/chunks_mpnet.npy         all vectors, in chunks.jsonl order (normalised, 768 numbers)
        .store/chroma/                  collection "filings", cosine space, with the metadata for filters

The embedded text is header + "\n" + text, the same string ingest.py counted to 360 tokens.
Step 3 measures header on against header off; that run embeds text only into its own file.
The whole run took 1 h 50 min on this CPU (2026-10-18; embed_run.log), not the 47 minutes first estimated
from the Week 2 rate, which was measured on shorter passages.

`python embed.py v2` does the same for the table-aware chunks from rechunk.py: .store/chunks_v2.jsonl in,
.store/emb_parts_v2/, .store/chunks_v2_mpnet.npy and the collection "filings_v2" out. Version 1 is untouched.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, ".store")
SUFFIX = "_" + sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "v1" else ""
PARTS = os.path.join(STORE, "emb_parts" + SUFFIX)
MODEL = "sentence-transformers/all-mpnet-base-v2"
PART = 1000


def load_chunks():
    return [json.loads(l) for l in open(os.path.join(STORE, f"chunks{SUFFIX}.jsonl"), encoding="utf-8")]


def done_already(path, digest):
    """The part file exists and was made from exactly these texts."""
    try:
        return os.path.exists(path) and open(path + ".sha256").read() == digest
    except OSError:
        return False


def embed_all(chunks):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    os.makedirs(PARTS, exist_ok=True)
    texts = [c["header"] + "\n" + c["text"] for c in chunks]
    n_parts = (len(texts) + PART - 1) // PART
    start = time.time()
    for i in range(n_parts):
        path = os.path.join(PARTS, f"part_{i:03}.npy")
        part = texts[i * PART:(i + 1) * PART]
        digest = hashlib.sha256("\x00".join(part).encode("utf-8")).hexdigest()
        # Skipping on "the file exists" reused old vectors after chunks.jsonl changed, and the count assert
        # below still passed (the review of 2026-10-23). A part with no hash file is embedded again.
        if done_already(path, digest):
            continue
        t = time.time()
        vecs = model.encode(part, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        np.save(path, vecs.astype(np.float32))
        open(path + ".sha256", "w").write(digest)
        print(f"part {i + 1}/{n_parts}: {time.time() - t:.0f} s (total {time.time() - start:.0f} s)", flush=True)
    vecs = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(PARTS, "part_*.npy")))])
    assert len(vecs) == len(chunks), (len(vecs), len(chunks))
    np.save(os.path.join(STORE, f"chunks{SUFFIX}_mpnet.npy"), vecs)
    return vecs


def build_chroma(chunks, vecs):
    import chromadb
    client = chromadb.PersistentClient(path=os.path.join(STORE, "chroma"))
    try:
        client.delete_collection("filings" + SUFFIX)
    except Exception:
        pass
    col = client.create_collection("filings" + SUFFIX, metadata={"hnsw:space": "cosine"})
    # Chroma metadata must be str, int, float or bool, so the page list becomes first page + a string.
    metas = [{"doc_id": c["doc_id"], "stock": c["stock"], "doc_type": c["doc_type"],
              "period": c["period"], "date": c["date"], "section": c["section"],
              "page": c["pages"][0], "pages": ",".join(map(str, c["pages"])),
              "ocr": c["ocr"], "url": c["url"], "header": c["header"],
              # added by add_fields.py (version 1) or rechunk.py (version 2)
              "fy": c["fy"] if c.get("fy") is not None else -1, "fiscal_year": c.get("fiscal_year", ""),
              "printed_page": c.get("printed_page", "")} for c in chunks]
    for i in range(0, len(chunks), 2000):
        col.add(ids=[c["chunk_id"] for c in chunks[i:i + 2000]],
                embeddings=vecs[i:i + 2000].tolist(),
                documents=[c["text"] for c in chunks[i:i + 2000]],
                metadatas=metas[i:i + 2000])
    print(f"ChromaDB: {col.count()} chunks in collection 'filings{SUFFIX}'")


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"{len(chunks)} chunks to embed with {MODEL}", flush=True)
    vecs = embed_all(chunks)
    build_chroma(chunks, vecs)
