# src/ingest_embeddings.py
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---- Config ----
CSV_PATH = "data/restaurants.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free, solid semantic model
EMBED_DIM = 384
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
# ---------------

def load_data(path: str) -> pd.DataFrame:
    """Ensure required text columns exist and have no NaNs."""
    df = pd.read_csv(path)
    for col in ["menu_item", "menu_description", "ingredient_name"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df

def build_text_and_meta(df: pd.DataFrame):
    """
    texts: semantic content to embed (menu item + description + ingredients)
    metas: structured fields we keep for filtering/display
    """
    texts, metas = [], []
    for _, r in df.iterrows():
        text = f"{r['menu_item']}: {r['menu_description']}. Ingredients: {r['ingredient_name']}."
        texts.append(text)
        metas.append({
            "restaurant_name": r.get("restaurant_name"),
            "categories": r.get("categories"),
            "city": r.get("city"),
            "state": r.get("state"),
            "zip_code": r.get("zip_code"),
            "rating": r.get("rating"),
            "price": r.get("price"),
            "review_count": r.get("review_count"),
            "item_id": r.get("item_id"),
            "confidence": r.get("confidence"),
        })
    return texts, metas

def embed_texts(texts):
    """Encode with SentenceTransformers and L2-normalize via NumPy (cosine-ready)."""
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(texts, batch_size=128, show_progress_bar=True).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs

def build_faiss(embs: np.ndarray) -> faiss.Index:
    """Create an inner-product index (works as cosine since vectors are normalized)."""
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embs)
    return index

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Place your CSV at {CSV_PATH}")

    print("Loading CSV…")
    df = load_data(CSV_PATH)
    print(f"Rows: {len(df)}")

    print("Preparing texts + metadata…")
    texts, metas = build_text_and_meta(df)

    print("Embedding texts…")
    embs = embed_texts(texts)

    print("Building FAISS index…")
    index = build_faiss(embs)

    print("Saving index + metadata…")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metas, f)

    print("✅ Done. Files written:", FAISS_INDEX_PATH, METADATA_PATH)

if __name__ == "__main__":
    main()
