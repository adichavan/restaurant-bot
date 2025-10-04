# src/upgrade_metadata.py
import pickle, pandas as pd, os

CSV_PATH = "data/restaurants.csv"
METADATA_PATH = "faiss_metadata.pkl"
OUT_PATH = METADATA_PATH  # in-place update

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH} (build your index first)")

    df = pd.read_csv(CSV_PATH)
    # ensure expected columns
    for col in ["menu_item", "menu_description", "ingredient_name"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    # load current metas
    with open(METADATA_PATH, "rb") as f:
        metas = pickle.load(f)

    n = min(len(df), len(metas))
    if len(df) != len(metas):
        print(f"⚠️ Row count mismatch: CSV={len(df)} vs metas={len(metas)}. Updating first {n} rows safely.")

    # attach text snippet + a stable id
    for i in range(n):
        r = df.iloc[i]
        text = f"{r['menu_item']}: {r['menu_description']}. Ingredients: {r['ingredient_name']}."
        metas[i]["text"] = text
        metas[i]["source"] = "internal"
        # prefer existing item_id if present, else fallback to row index
        metas[i]["source_id"] = metas[i].get("item_id", i)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(metas, f)

    print(f"✅ Updated {OUT_PATH} with 'text', 'source', 'source_id' for {n} items.")

if __name__ == "__main__":
    main()
