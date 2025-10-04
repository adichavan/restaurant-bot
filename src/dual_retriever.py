# src/dual_retriever.py
from typing import List, Dict, Any, Optional
from .retriever import find_restaurants, DEFAULT_CITY
from .ext_search import search_external

def dual_retrieve(
    query: str,
    city: Optional[str] = None,
    k_internal: int = 5,
    k_external: int = 5,
    limit_per_restaurant: int = 1,
    auto_city: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    # internal: use our helper (defaults to SF or provided city)
    filters = {"city": city} if city else {}
    internal = find_restaurants(
        query=query,
        k=k_internal,
        filters=filters,
        limit_per_restaurant=limit_per_restaurant,
        default_city=city or DEFAULT_CITY,
        auto_city=auto_city,
    )
    # ensure each has text (from upgrade script)
    for m in internal:
        m.setdefault("text", "")
        m.setdefault("source", "internal")
        m.setdefault("source_id", m.get("item_id"))

    # external: top-k chunks with titles/urls
    external = search_external(query, k=k_external)
    for e in external:
        e.setdefault("text", "")
        e.setdefault("source", e.get("source", "external"))

    return {"internal": internal, "external": external}
