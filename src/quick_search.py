# src/quick_search.py

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"

def embed_query(q: str):
    """Why: embed the user query with the same model & normalization as docs."""
    model = SentenceTransformer(EMBED_MODEL)
    vec = model.encode([q]).astype("float32")
    # faiss.normalize_L2(vec)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    vec = vec / norm
    return vec

def main():
    # Load index + metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metas = pickle.load(f)

    # Ask for a test query
    q = input("Query (e.g., 'Impossible Meat tacos in Los Angeles'): ")
    qv = embed_query(q)

    # Top-5 nearest neighbors
    D, I = index.search(qv, 5)

    # Show results with context you can later use in the bot
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        m = metas[idx]
        print(f"\n#{rank} score={score:.3f}")
        print(f"- restaurant: {m.get('restaurant_name')}")
        print(f"- city/state: {m.get('city')}, {m.get('state')}")
        print(f"- categories: {m.get('categories')}")
        print(f"- price/rating: {m.get('price')} / {m.get('rating')}")
        print(f"- item_id: {m.get('item_id')}  confidence: {m.get('confidence')}")

if __name__ == "__main__":
    main()
