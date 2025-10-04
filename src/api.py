# src/api.py
from typing import List, Optional
from fastapi import FastAPI, Query

app = FastAPI(title="Restaurant Bot API", version="0.1.0")
def _to_jsonable(obj):
    """Recursively convert non-serializable objects (NumPy, NaN, sets, datetimes) to plain JSON-safe Python."""
    import math
    try:
        import numpy as np
    except Exception:
        np = None

    # simple types
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        # replace NaN/inf with None
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # numpy scalars / arrays
    if np is not None:
        if isinstance(obj, np.generic):
            val = obj.item()
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
        if isinstance(obj, np.ndarray):
            return [_to_jsonable(v) for v in obj.tolist()]

    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # datetimes or similar objects with isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # last resort
    return str(obj)



@app.get("/health")
def health():
    return {"ok": True}

def _err_payload(e: Exception):
    return {"error": f"{type(e).__name__}: {e}"}

@app.get("/search")
def search(
    q: str = Query(...),
    city: Optional[str] = None,
    categories: Optional[List[str]] = Query(None),
    k: int = 5,
):
    """
    Internal semantic search (+ simple filters).
    Lazy-imports to avoid crashing the whole app if a module has issues.
    """
    try:
        from .retriever import find_restaurants, DEFAULT_CITY
        filters = {}
        if city:
            filters["city"] = city
        if categories:
            filters["categories_any"] = categories
        res = find_restaurants(
            query=q,
            k=k,
            filters=filters,
            limit_per_restaurant=1,
            default_city=city or DEFAULT_CITY,
            auto_city=True,
        )
        return {"count": len(res), "results": _to_jsonable(res[:k])}
    except Exception as e:
        return _err_payload(e)

@app.get("/rag")
def rag(
    q: str = Query(...),
    city: Optional[str] = None,
    k_internal: int = 5,
    k_external: int = 5,
):
    """
    Return the retrieved contexts + citations (LLM call is handled in CLI;
    API shows the evidence clearly for demo).
    """
    try:
        from .dual_retriever import dual_retrieve
        bundle = dual_retrieve(query=q, city=city, k_internal=k_internal, k_external=k_external)

        citations = []
        for i, m in enumerate(bundle.get("internal", []), start=1):
            citations.append({
                "tag": f"IN-{i}",
                "source": "internal",
                "restaurant_name": m.get("restaurant_name"),
                "city": m.get("city"),
                "state": m.get("state"),
                "item_id": m.get("source_id"),
            })
        for i, m in enumerate(bundle.get("external", []), start=1):
            citations.append({
                "tag": f"EX-{i}",
                "source": m.get("source", "external"),
                "title": m.get("title"),
                "url": m.get("url"),
                "published": m.get("published"),
            })

        return {
    "query": q,
    "contexts": _to_jsonable(bundle),
    "citations": _to_jsonable(citations),
}
    except Exception as e:
        return _err_payload(e)

@app.post("/compare")
def compare(
    city: str = "San Francisco",
    a: List[str] = Query(..., description="Category terms for group A"),
    b: List[str] = Query(..., description="Category terms for group B"),
):
    """
    Average price comparison using your internal CSV.
    """
    try:
        import os, pandas as pd
        from .analytics import avg_price_for_category, CSV_PATH
        if not os.path.exists(CSV_PATH):
            return {"error": f"Missing {CSV_PATH}"}
        df = pd.read_csv(CSV_PATH).fillna("")
        avg_a = avg_price_for_category(df, city, a)
        avg_b = avg_price_for_category(df, city, b)
        return {
            "city": city,
            "a": {"terms": a, "avg_price": None if str(avg_a) == "nan" else avg_a},
            "b": {"terms": b, "avg_price": None if str(avg_b) == "nan" else avg_b},
        }
    except Exception as e:
        return _err_payload(e)

@app.get("/trend")
def trend(
    terms: List[str] = Query(...),
    months: int = 12,
    must_include: str = "",
    mode: str = "all",
):
    """
    Monthly trend from external feeds (recency-aware).
    """
    try:
        import os, pickle
        from .trend_external import monthly_trend, EXT_META_PATH
        if not os.path.exists(EXT_META_PATH):
            return {"error": f"Missing {EXT_META_PATH}. Run ext_ingest first."}
        with open(EXT_META_PATH, "rb") as f:
            meta = pickle.load(f)
        must = must_include.strip() or None
        rows = monthly_trend(meta, terms, must_include=must, months=months, mode=mode)
        return {"terms": terms, "months": months, "must_include": must, "mode": mode,
                "buckets": [{"month": ym, "count": c, "samples": s} for ym, c, s in rows]}
    except Exception as e:
        return _err_payload(e)
