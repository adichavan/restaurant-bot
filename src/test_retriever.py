# src/test_retriever.py
import sys
print(">>> START test_retriever.py")
print("python:", sys.executable)

try:
    from src.retriever import semantic_search
    print(">>> imported semantic_search OK")
except Exception as e:
    print(">>> import error:", repr(e))
    raise

res = semantic_search("Impossible Meat", k=3, filters={"city": "Los Angeles"}, limit_per_restaurant=1)
print(">>> results:", len(res))
for i, r in enumerate(res, 1):
    print(f"#{i}", r.get("restaurant_name"), "-", r.get("city"), r.get("state"), "| score:", r.get("score"))

print(">>> END test_retriever.py")

