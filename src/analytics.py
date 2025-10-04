# src/analytics.py
import argparse, pandas as pd
from typing import List

CSV_PATH = "data/restaurants.csv"

def _norm(s): return "" if s is None else str(s).strip()

def _price_to_num(p):
    """Map '$' scales to 1-4; pass through numeric if available."""
    if p is None or p == "": return None
    try:
        return float(p)
    except Exception:
        if isinstance(p, str):
            s = p.strip()
            if set(s) <= {"$"}:   # "$", "$$", "$$$", "$$$$"
                return float(len(s))
    return None

def _matches_category(row_cats: str, want: List[str]) -> bool:
    rc = _norm(row_cats).lower()
    return any(w.lower() in rc for w in want)

def avg_price_for_category(df: pd.DataFrame, city: str, category_terms: List[str]) -> float:
    mask_city = df["city"].fillna("").str.contains(city, case=False, na=False)
    mask_cat = df["categories"].fillna("").apply(lambda x: _matches_category(x, category_terms))
    sub = df[mask_city & mask_cat]
    if sub.empty: return float("nan")
    prices = sub["price"].apply(_price_to_num).dropna()
    return float(prices.mean()) if not prices.empty else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="San Francisco")
    ap.add_argument("--a", nargs="+", required=True, help="Category terms for group A, e.g. vegan")
    ap.add_argument("--b", nargs="+", required=True, help="Category terms for group B, e.g. mexican")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    for col in ["categories","city","price"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    avg_a = avg_price_for_category(df, args.city, args.a)
    avg_b = avg_price_for_category(df, args.city, args.b)

    def fmt(label, val):
        return f"{label}: {val:.2f}" if val == val else f"{label}: N/A"

    print(f"City: {args.city}")
    print(fmt(f"A ({' '.join(args.a)})", avg_a))
    print(fmt(f"B ({' '.join(args.b)})", avg_b))

if __name__ == "__main__":
    main()
