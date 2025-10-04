# src/trend_external.py
import argparse, pickle
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dateutil.tz import tzutc


EXT_META_PATH = "faiss_ext_metadata.pkl"

def _norm(s): return "" if s is None else str(s).strip()
def _contains_ci(text: str, needle: str) -> bool:
    return needle.lower() in _norm(text).lower()

def _parse_dt(dt_str):
    if not dt_str:
        return None
    try:
        d = dateparser.parse(dt_str)
        if not d:
            return None
        # Normalize to UTC-aware
        if d.tzinfo is None:
            d = d.replace(tzinfo=tzutc())
        else:
            d = d.astimezone(tzutc())
        return d
    except Exception:
        return None

def monthly_trend(meta, terms, must_include=None, months=12, mode="all"):
    """
    Count items per month where text OR title matches terms.
    mode='all' => all terms must appear (AND)
    mode='any' => any one term is enough (OR)
    """
    now = datetime.now(tzutc())
    start = now - timedelta(days=months * 31)
    buckets = Counter()
    samples = defaultdict(list)

    for m in meta:
        pub = _parse_dt(m.get("published"))
        if not pub or pub < start or pub > now:
            continue

        text = _norm(m.get("text")) + " " + _norm(m.get("title"))

        if must_include and not _contains_ci(text, must_include):
            continue

        if mode == "all":
            ok = all(_contains_ci(text, t) for t in terms)
        else:
            ok = any(_contains_ci(text, t) for t in terms)
        if not ok:
            continue

        ym = pub.strftime("%Y-%m")
        buckets[ym] += 1
        if len(samples[ym]) < 3:
            samples[ym].append({"title": m.get("title"), "url": m.get("url")})

    out = []
    for ym in sorted(buckets.keys()):
        out.append((ym, buckets[ym], samples[ym]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--terms", nargs="+", default=["saffron", "dessert"], help="Terms to match")
    ap.add_argument("--must_include", type=str, default="", help="Optional location keyword ('' to disable)")
    ap.add_argument("--mode", choices=["all", "any"], default="all", help="Require all terms or any term")
    args = ap.parse_args()

    with open(EXT_META_PATH, "rb") as f:
        meta = pickle.load(f)

    must_include = args.must_include.strip() or None
    trend = monthly_trend(meta, args.terms, must_include=must_include, months=args.months, mode=args.mode)

    if not trend:
        print("No matches with the current feeds/filters. Try broader feeds, --mode any, or remove --must_include.")
        return

    print(f"Trend for terms={args.terms} months={args.months} must_include={must_include} mode={args.mode}")
    for ym, count, samples in trend:
        print(f"{ym}: {count}")
        for s in samples:
            print(f"   - {s['title']}  ({s['url']})")

if __name__ == "__main__":
    main()
