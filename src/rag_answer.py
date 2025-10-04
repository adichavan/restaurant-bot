# src/rag_answer.py
import os, textwrap
from typing import Dict, List, Any

# Silence tokenizers warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional OpenAI import; we handle absence or errors gracefully
try:
    import openai  # for exception classes
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from .dual_retriever import dual_retrieve

MAX_CHARS = 1600  # truncate long chunks to keep prompts small

def _make_citations(internal: List[Dict[str,Any]], external: List[Dict[str,Any]]):
    lines = []
    citations = []
    i_num, e_num = 0, 0
    for m in internal:
        i_num += 1
        tag = f"[IN-{i_num}]"
        txt = (m.get("text") or "")[:MAX_CHARS]
        src = f"{m.get('restaurant_name','')} — {m.get('city','')}, {m.get('state','')}"
        lines.append(f"{tag} {src}\n{txt}\n")
        citations.append({"tag": tag, "type": "internal", "restaurant_name": m.get("restaurant_name"), "city": m.get("city"), "state": m.get("state"), "item_id": m.get("source_id")})
    for m in external:
        e_num += 1
        tag = f"[EX-{e_num}]"
        txt = (m.get("text") or "")[:MAX_CHARS]
        src = f"{m.get('source','')} | {m.get('title','')} | {m.get('url','')}"
        lines.append(f"{tag} {src}\n{txt}\n")
        citations.append({"tag": tag, "type": m.get("source","external"), "title": m.get("title"), "url": m.get("url"), "published": m.get("published")})
    return "\n".join(lines), citations

_SYSTEM = """You are a helpful restaurant and food knowledge assistant.
Use the provided CONTEXT only. If something is unclear, say so and ask for specifics.
Always cite with the tags [IN-*] for internal items and [EX-*] for external sources.
Keep answers concise and factual. Provide short bullet points when helpful."""

def _print_mock(query: str, ctx_text: str, cites: List[Dict[str,Any]]):
    print("🔎 MOCK ANSWER (LLM unavailable or quota exceeded)\n")
    print(f"Q: {query}\n")
    print("Suggested answer outline:")
    print("- Key points from internal results (see [IN-*])")
    print("- Supporting history/trends from external sources (see [EX-*])")
    print("- Finish with 2–3 suggested restaurants (with city) + citations\n")
    print("Context used:\n" + "-"*40)
    print(ctx_text[:4000])
    print("\nCitations:")
    for c in cites[:10]:
        print(c)

def answer_query(query: str, city: str = None):
    bundle = dual_retrieve(query=query, city=city, k_internal=5, k_external=5)
    ctx_text, cites = _make_citations(bundle["internal"], bundle["external"])

    # If OpenAI isn't installed or no key, go mock immediately
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        _print_mock(query, ctx_text, cites)
        return

    # Try real call, but fall back on *any* OpenAI exception (401/429/network/etc.)
    try:
        client = OpenAI()
        user_prompt = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{ctx_text}\n\n"
            "INSTRUCTIONS:\n"
            "- Cite sources inline using [IN-*] and [EX-*].\n"
            "- Be concise. If data is missing or conflicting, say so.\n"
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        print(resp.choices[0].message.content)
        print("\n—\nSources:", ", ".join([c["tag"] for c in cites[:8]]))
    except Exception as e:
        # Specific OpenAI exceptions (auth/quota/rate) collapse to mock view
        # Examples: openai.AuthenticationError, openai.RateLimitError, openai.APIError, etc.
        print(f"\n⚠️ OpenAI call failed: {type(e).__name__}: {e}\nFalling back to mock output.\n")
        _print_mock(query, ctx_text, cites)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the history of sushi, and which restaurants near me are known for it?"
    city = None
    answer_query(q, city=city)
