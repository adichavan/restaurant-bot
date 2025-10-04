# src/cli.py
import argparse, pickle, os
from typing import List
from .retriever import find_restaurants, DEFAULT_CITY
from .rag_answer import answer_query as rag_answer
from .analytics import avg_price_for_category, CSV_PATH
from .trend_external import monthly_trend, EXT_META_PATH

def _print_rows(rows, limit=10):
    for i, r in enumerate(rows[:limit], 1):
        name = r.get("restaurant_name")
        city = r.get("city")
        state = r.get("state")
        cats = r.get("categories")
        score = r.get("score")
        print(f"#{i} {name} — {city}, {state} | {cats} | score={score:.3f}")

def cmd_search(args):
    filters = {}
    if args.city:
        filters["city"] = args.city
    if args.categories:
        filters["categories_any"] = args.categories
    res = find_restaurants(
        query=args.q,
        k=args.k,
        filters=filters,
        limit_per_restaurant=1,
        default_city=args.city or DEFAULT_CITY,
        auto_city=True,
    )
    if not res:
        print("No results.")
        return
    _print_rows(res, limit=args.k)

def cmd_rag(args):
    # answer prints itself (real LLM or mock fallback)
    rag_answer(args.q, city=args.city)

def cmd_compare(args):
    import pandas as pd
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).fillna("")
    a = avg_price_for_category(df, args.city, args.a)
    b = avg_price_for_category(df, args.city, args.b)
    def fmt(label, v): return f"{label}: {v:.2f}" if v == v else f"{label}: N/A"
    print(f"City: {args.city}")
    print(fmt(f"A ({' '.join(args.a)})", a))
    print(fmt(f"B ({' '.join(args.b)})", b))

def cmd_trend(args):
    if not os.path.exists(EXT_META_PATH):
        raise FileNotFoundError(f"Missing {EXT_META_PATH}. Run ext_ingest first.")
    with open(EXT_META_PATH, "rb") as f:
        meta = pickle.load(f)
    must = args.must_include.strip() or None
    trend = monthly_trend(meta, args.terms, must_include=must, months=args.months, mode=args.mode)
    if not trend:
        print("No matches with the current feeds/filters.")
        return
    print(f"Trend for terms={args.terms} months={args.months} must_include={must} mode={args.mode}")
    for ym, count, samples in trend:
        print(f"{ym}: {count}")
        for s in samples:
            print(f"   - {s['title']}  ({s['url']})")

def main():
    ap = argparse.ArgumentParser(prog="restaurant-bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # search
    ap_search = sub.add_parser("search", help="Internal semantic search (+ filters)")
    ap_search.add_argument("--q", required=True, help="Query text")
    ap_search.add_argument("--city", default=None, help="City (defaults to San Francisco if omitted or 'near me')")
    ap_search.add_argument("--categories", nargs="+", default=None, help="Category terms to filter (any match)")
    ap_search.add_argument("--k", type=int, default=5)
    ap_search.set_defaults(func=cmd_search)

    # rag
    ap_rag = sub.add_parser("rag", help="RAG answer (internal + external with citations)")
    ap_rag.add_argument("--q", required=True)
    ap_rag.add_argument("--city", default=None)
    ap_rag.set_defaults(func=cmd_rag)

    # compare
    ap_cmp = sub.add_parser("compare", help="Average price comparison by categories")
    ap_cmp.add_argument("--city", default="San Francisco")
    ap_cmp.add_argument("--a", nargs="+", required=True, help="Category terms for group A")
    ap_cmp.add_argument("--b", nargs="+", required=True, help="Category terms for group B")
    ap_cmp.set_defaults(func=cmd_compare)

    # trend (external)
    ap_trend = sub.add_parser("trend", help="Monthly trend from external RSS/Wiki")
    ap_trend.add_argument("--months", type=int, default=12)
    ap_trend.add_argument("--terms", nargs="+", required=True)
    ap_trend.add_argument("--must_include", default="", help="Optional location keyword")
    ap_trend.add_argument("--mode", choices=["all","any"], default="all")
    ap_trend.set_defaults(func=cmd_trend)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
