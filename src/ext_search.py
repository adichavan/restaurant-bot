# src/ext_search.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
EXT_INDEX_PATH = "faiss_ext_index.bin"
EXT_META_PATH  = "faiss_ext_metadata.pkl"

def embed_query(q: str):
    model = SentenceTransformer(EMBED_MODEL)
    v = model.encode([q]).astype("float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def search_external(query: str, k: int = 5):
    index = faiss.read_index(EXT_INDEX_PATH)
    with open(EXT_META_PATH, "rb") as f:
        metas = pickle.load(f)
    qv = embed_query(query)
    D, I = index.search(qv, k)
    return [{"score": float(s), **metas[idx]} for s, idx in zip(D[0], I[0])]

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "dessert trends San Francisco"
    res = search_external(q, k=5)
    for i, r in enumerate(res, 1):
        print(f"#{i} {r['source']} | {r.get('title')} | {r.get('published')} | score={r['score']:.3f}")
        print(f"   {r.get('url')}")
