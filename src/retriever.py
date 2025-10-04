# src/retriever.py
from typing import Dict, Any, Optional, List
from . import vector_store as vs
# Default location used when user doesn't specify a city (your dataset is SF-heavy)
DEFAULT_CITY = "San Francisco"


def _eq_ci(a, b):
    return (str(a).strip().lower() == str(b).strip().lower()) if (a is not None and b is not None) else False

def _passes_filters(meta: Dict[str, Any], f: Optional[Dict[str, Any]]) -> bool:
    if not f:
        return True
    # City / State exact (case-insensitive)
    if "city" in f and not _eq_ci(meta.get("city"), f["city"]):
        return False
    if "state" in f and not _eq_ci(meta.get("state"), f["state"]):
        return False
    # Categories contains any of these substrings
    if "categories_any" in f:
        cats = (meta.get("categories") or "").lower()
        wanted = [c.lower() for c in f["categories_any"]]
        if not any(w in cats for w in wanted):
            return False
    # Min rating
    if "min_rating" in f:
        try:
            if meta.get("rating") is not None and float(meta["rating"]) < float(f["min_rating"]):
                return False
        except Exception:
            pass
    # Max price (supports "$", "$$", "$$$" by length or numeric)
    if "max_price" in f:
        price = meta.get("price")
        maxp = f["max_price"]
        if isinstance(price, str) and isinstance(maxp, int):
            if len(price.strip()) > maxp:
                return False
        elif isinstance(price, (int, float)) and float(price) > float(maxp):
            return False
    # Ingredient confidence threshold (optional)
    if "confidence_min" in f:
        try:
            conf = meta.get("confidence")
            if conf is not None and float(conf) < float(f["confidence_min"]):
                return False
        except Exception:
            pass
    return True

import re
from typing import Dict, Any, Optional, List

def _norm(s):
    return "" if s is None else str(s).strip()

def _merge_filters_with_default_city(filters: Optional[Dict[str, Any]], default_city: str) -> Dict[str, Any]:
    """If caller didn't provide a city, inject the default city so we get local results."""
    f = dict(filters) if filters else {}
    if not _norm(f.get("city")):
        f["city"] = default_city
    return f

_NEAR_ME_RE = re.compile(r"\bnear me\b", flags=re.IGNORECASE)

def _strip_near_me(query: str) -> str:
    """Remove 'near me' from the text (case-insensitive) but keep the rest of the query."""
    return _NEAR_ME_RE.sub("", query).strip()

def find_restaurants(
    query: str,
    k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    limit_per_restaurant: int = 1,
    default_city: str = DEFAULT_CITY,
    auto_city: bool = True,
):
    """
    Wrapper on top of semantic_search that:
      - detects 'near me' and injects the default city,
      - or injects default city whenever no city is provided (if auto_city=True).
    """
    q_clean = _strip_near_me(query)
    f = dict(filters) if filters else {}

    needs_default = bool(_NEAR_ME_RE.search(query)) or not _norm(f.get("city"))
    if auto_city and needs_default:
        f = _merge_filters_with_default_city(f, default_city)

    # Call the original semantic search you already have
    return semantic_search(q_clean, k=k, filters=f, limit_per_restaurant=limit_per_restaurant)


def semantic_search(
    query: str,
    k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    limit_per_restaurant: int = 1,
) -> List[Dict[str, Any]]:
    """Semantic search with optional structured filters and de-dup by restaurant."""
    # Over-retrieve, then filter & dedupe
    D, I, metas = vs.search(query, k=max(k * 3, k))
    out: List[Dict[str, Any]] = []
    seen = set()
    for score, idx in zip(D, I):
        if idx == -1:
            continue
        m = metas[idx]
        if not _passes_filters(m, filters):
            continue
        key = (m.get("restaurant_name") or "").strip().lower()
        if limit_per_restaurant and key in seen:
            continue
        if limit_per_restaurant:
            seen.add(key)
        out.append({"score": float(score), **m})
        if len(out) >= k:
            break
    return out
