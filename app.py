# app.py — Streamlit UI for your Restaurant Bot (free local)
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Restaurant Bot", layout="wide")
st.title("🍽️ Restaurant Bot")

# ---- API base URL (points to your running FastAPI) ----
with st.sidebar:
    st.header("Backend")
    base_url = st.text_input("FastAPI base URL", "http://127.0.0.1:8000")
    if st.button("Health check"):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

# -------------------------
# Ask (freeform) — rules-based router (no extra cost)
# -------------------------
ask_tab = st.tabs(["Ask (freeform)"])[0]

def _extract_city(text: str):
    import re
    m = re.search(r"\b(in|near|around)\s+([A-Za-z][A-Za-z\s]+)$", text.strip(), re.IGNORECASE)
    return m.group(2).strip() if m else ""

def _looks_compare(q: str) -> bool:
    ql = q.lower()
    return " vs " in ql or "compare" in ql or "versus" in ql

def _looks_trend(q: str) -> bool:
    ql = q.lower()
    return "trend" in ql or "over the last" in ql or "last year" in ql or "over time" in ql

def _looks_explain(q: str) -> bool:
    ql = q.lower()
    return any(tok in ql for tok in ["what is", "why is", "explain", "history of", "background of"])

def _terms_from_compare(q: str):
    import re
    ql = q.lower()
    m = re.search(r"([a-zA-Z\s]+?)\s+(?:vs|versus)\s+([a-zA-Z\s]+)", ql)
    if m:
        a = m.group(1).strip().split()
        b = m.group(2).strip().split()
        return [a[-1]] if a else [], [b[-1]] if b else []
    m2 = re.search(r"compare\s+([a-zA-Z]+)\s+(?:and|&)\s+([a-zA-Z]+)", ql)
    if m2:
        return [m2.group(1)], [m2.group(2)]
    return [], []

with ask_tab:
    st.subheader("Ask anything")
    user_q = st.text_input("Type your question", "Which restaurants near me serve gluten-free pizza?")
    mode = st.selectbox("Mode (let me choose or auto-detect)", ["Auto", "Search", "RAG", "Compare", "Trend"])
    if st.button("Ask"):
        try:
            chosen = mode
            city = _extract_city(user_q)

            if chosen == "Auto":
                if _looks_compare(user_q):
                    chosen = "Compare"
                elif _looks_trend(user_q):
                    chosen = "Trend"
                elif _looks_explain(user_q):
                    chosen = "RAG"
                else:
                    chosen = "Search"

            st.write(f"**Router chose:** {chosen}")

            if chosen == "Search":
                params = {"q": user_q, "k": 5}
                if city:
                    params["city"] = city
                r = requests.get(f"{base_url}/search", params=params, timeout=20)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    results = data.get("results", [])
                    if results:
                        df = pd.DataFrame(results)
                        cols = [c for c in ["restaurant_name","city","state","menu_item","categories","score"] if c in df.columns]
                        st.dataframe(df[cols] if cols else df, use_container_width=True)
                    else:
                        st.info("No results.")

            elif chosen == "RAG":
                r = requests.get(f"{base_url}/rag", params={"q": user_q, "k_internal": 5, "k_external": 5}, timeout=30)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    ctx = data.get("contexts", {})
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Internal (restaurants)**")
                        for i, m in enumerate(ctx.get("internal", [])[:10], 1):
                            st.write(f"**#{i}** {m.get('restaurant_name')} — {m.get('city')}, {m.get('state')}")
                            if m.get("menu_item"):
                                st.caption(m.get("menu_item"))
                    with c2:
                        st.markdown("**External (articles/wiki)**")
                        for i, m in enumerate(ctx.get("external", [])[:10], 1):
                            title = m.get("title") or "(untitled)"
                            url = m.get("url")
                            if url:
                                st.markdown(f"**#{i}** [{title}]({url})")
                            else:
                                st.write(f"**#{i}** {title}")
                            if m.get("published"):
                                st.caption(str(m.get("published")))
                    st.markdown("---")
                    st.markdown("**Citations**")
                    for c in data.get("citations", [])[:12]:
                        st.code(c, language="json")

            elif chosen == "Compare":
                a, b = _terms_from_compare(user_q)
                if not a or not b:
                    st.warning("Couldn’t detect 'X vs Y'—please add 'vs' or switch mode to Compare and enter terms.")
                else:
                    city2 = city or "San Francisco"
                    r = requests.post(f"{base_url}/compare", params={"city": city2, "a": a, "b": b}, timeout=20)
                    data = r.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.json(data)

            elif chosen == "Trend":
                months = 12
                must = city or ""
                core = [w for w in user_q.strip().split() if w.isalpha()]
                terms = [core[-1]] if core else ["dessert"]
                r = requests.get(f"{base_url}/trend",
                                 params={"terms": terms, "months": months, "mode": "any", "must_include": must},
                                 timeout=30)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    buckets = data.get("buckets", [])
                    if buckets:
                        df = pd.DataFrame([{"month": b["month"], "count": b["count"]} for b in buckets])
                        df = df.sort_values("month")
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index("month"))
                    else:
                        st.info("No matches with the current feeds/filters.")
        except Exception as e:
            st.error(f"Ask failed: {e}")

# -------------------------
# Classic tabs (Search / RAG / Compare / Trend)
# -------------------------
tabs = st.tabs(["Search", "RAG", "Compare", "Trend"])

# Search
with tabs[0]:
    st.subheader("Ingredient / Dish Discovery")
    q = st.text_input("Query", "gluten-free pizza near me", key="s_q")
    city = st.text_input("City (optional)", "", key="s_city")
    k = st.number_input("Top K", 1, 20, 5, key="s_k")
    if st.button("Search", key="btn_search"):
        try:
            params = {"q": q, "k": k}
            if city.strip():
                params["city"] = city.strip()
            r = requests.get(f"{base_url}/search", params=params, timeout=20)
            data = r.json()
            if "error" in data:
                st.error(data["error"])
            else:
                results = data.get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    cols = [c for c in ["restaurant_name","city","state","menu_item","categories","score"] if c in df.columns]
                    st.dataframe(df[cols] if cols else df, use_container_width=True)
                else:
                    st.info("No results.")
        except Exception as e:
            st.error(f"Search failed: {e}")

# RAG
with tabs[1]:
    st.subheader("Context + Citations (Internal + External)")
    rq = st.text_input("Question", "What is the history of sushi, and which restaurants near me are known for it?", key="r_q")
    k_int = st.number_input("Top K (internal)", 1, 20, 5, key="r_ki")
    k_ext = st.number_input("Top K (external)", 1, 20, 5, key="r_ke")
    if st.button("Retrieve Context", key="btn_rag"):
        try:
            r = requests.get(f"{base_url}/rag", params={"q": rq, "k_internal": k_int, "k_external": k_ext}, timeout=30)
            data = r.json()
            if "error" in data:
                st.error(data["error"])
            else:
                ctx = data.get("contexts", {})
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Internal (restaurants)**")
                    for i, m in enumerate(ctx.get("internal", [])[:10], 1):
                        st.write(f"**#{i}** {m.get('restaurant_name')} — {m.get('city')}, {m.get('state')}")
                        if m.get("menu_item"):
                            st.caption(m.get("menu_item"))
                with c2:
                    st.markdown("**External (articles/wiki)**")
                    for i, m in enumerate(ctx.get("external", [])[:10], 1):
                        title = m.get("title") or "(untitled)"
                        url = m.get("url")
                        if url:
                            st.markdown(f"**#{i}** [{title}]({url})")
                        else:
                            st.write(f"**#{i}** {title}")
                        if m.get("published"):
                            st.caption(str(m.get("published")))
                st.markdown("---")
                st.markdown("**Citations**")
                for c in data.get("citations", [])[:12]:
                    st.code(c, language="json")
        except Exception as e:
            st.error(f"RAG failed: {e}")

# Compare
with tabs[2]:
    st.subheader("Average Menu Price by Category")
    city_cmp = st.text_input("City", "San Francisco", key="cmp_city")
    a_terms = st.text_input("Group A terms (space-separated)", "vegan", key="cmp_a").split()
    b_terms = st.text_input("Group B terms (space-separated)", "mexican", key="cmp_b").split()
    if st.button("Compare", key="btn_compare"):
        try:
            r = requests.post(f"{base_url}/compare", params={"city": city_cmp, "a": a_terms, "b": b_terms}, timeout=20)
            data = r.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.json(data)
        except Exception as e:
            st.error(f"Compare failed: {e}")

# Trend
with tabs[3]:
    st.subheader("External Trend (Monthly)")
    terms_raw = st.text_input("Terms (comma-separated)", "dessert", key="t_terms")
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()]
    months = st.slider("Months window", 3, 24, 12, key="t_months")
    mode = st.selectbox("Match mode", ["any","all"], index=0, key="t_mode")
    must_include = st.text_input("Must include (optional, e.g., 'San Francisco')", "", key="t_must")
    if st.button("Show Trend", key="btn_trend"):
        try:
            r = requests.get(f"{base_url}/trend",
                             params={"terms": terms, "months": months, "mode": mode, "must_include": must_include},
                             timeout=30)
            data = r.json()
            if "error" in data:
                st.error(data["error"])
            else:
                buckets = data.get("buckets", [])
                if buckets:
                    df = pd.DataFrame([{"month": b["month"], "count": b["count"]} for b in buckets])
                    df = df.sort_values("month")
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("month"))
                else:
                    st.info("No matches with the current feeds/filters.")
        except Exception as e:
            st.error(f"Trend failed: {e}")
