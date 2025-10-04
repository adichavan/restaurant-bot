# src/ext_ingest.py
import argparse, pickle, re
from typing import List, Tuple, Dict, Optional
import wikipedia
import feedparser
from dateutil import parser as dateparser
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
EXT_INDEX_PATH = "faiss_ext_index.bin"
EXT_META_PATH  = "faiss_ext_metadata.pkl"

def clean_text(t: Optional[str]) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(t: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    t = clean_text(t)
    if not t:
        return []
    chunks, start = [], 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        if end == len(t): break
        start = end - overlap
    return chunks

def fetch_wikipedia_pages(titles: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            out.append((page.title, page.url, page.content))
        except Exception:
            try:
                hits = wikipedia.search(title)
                if hits:
                    page = wikipedia.page(hits[0])
                    out.append((page.title, page.url, page.content))
            except Exception:
                pass
    return out

def fetch_rss_articles(feeds: List[str], max_items_per_feed: int = 10) -> List[Tuple[str, str, str, str]]:
    import time
    out = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items_per_feed]:
                title = clean_text(getattr(entry, "title", ""))
                link = getattr(entry, "link", "")

                # Prefer content if present; fall back to summary/description
                summary = ""
                parts = getattr(entry, "content", None)
                if parts:
                    if isinstance(parts, list):
                        vals = []
                        for p in parts:
                            try:
                                if isinstance(p, dict):
                                    vals.append(clean_text(p.get("value", "")))
                                else:
                                    vals.append(clean_text(getattr(p, "value", "")))
                            except Exception:
                                pass
                        summary = " ".join(v for v in vals if v)
                    else:
                        # Unknown structure; fall back to summary/description
                        summary = clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
                else:
                    summary = clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))

                # Try multiple date fields; fall back to *_parsed tuples
                published_iso = ""
                candidates = [
                    getattr(entry, "published", None),
                    getattr(entry, "updated", None),
                    getattr(entry, "created", None),
                    getattr(entry, "pubDate", None),
                    getattr(entry, "dc:date", None),
                ]
                for cand in candidates:
                    if cand:
                        try:
                            d = dateparser.parse(cand)
                            if d:
                                published_iso = d.isoformat()
                                break
                        except Exception:
                            pass
                if not published_iso:
                    for t in [
                        getattr(entry, "published_parsed", None),
                        getattr(entry, "updated_parsed", None),
                        getattr(entry, "created_parsed", None),
                    ]:
                        if t:
                            try:
                                published_iso = time.strftime("%Y-%m-%dT%H:%M:%S", t)
                                break
                            except Exception:
                                pass

                out.append((title, link, summary, published_iso))
        except Exception:
            # Skip any feed that fails to parse
            pass
    return out

def embed_texts(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms[norms==0] = 1.0
    return vecs / norms

def build_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia", nargs="*", default=[], help="Wikipedia page titles")
    parser.add_argument("--rss", nargs="*", default=[], help="RSS feed URLs")
    parser.add_argument("--rss_file", type=str, default=None, help="Path to a txt file of RSS feed URLs (one per line, # for comments)")
    parser.add_argument("--max_rss_items", type=int, default=10)
    args = parser.parse_args()


    docs, texts = [], []
    
    # Read feeds from a file if given
    if args.rss_file:
        try:
            with open(args.rss_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    args.rss.append(line)
        except Exception as e:
            print(f"⚠️ Could not read rss_file: {e}")


    if args.wikipedia:
        wikipedia.set_lang("en")
        pages = fetch_wikipedia_pages(args.wikipedia)
        for title, url, content in pages:
            for ch in chunk_text(content):
                texts.append(ch)
                docs.append({"source": "wikipedia", "title": title, "url": url, "published": None, "text": ch})

    if args.rss:
        items = fetch_rss_articles(args.rss, max_items_per_feed=args.max_rss_items)
        for title, link, summary, published in items:
            for ch in chunk_text(summary):
                texts.append(ch)
                docs.append({"source": "rss", "title": title, "url": link, "published": published, "text": ch})

    if not texts:
        print("No external texts gathered. Provide --wikipedia and/or --rss.")
        return

    seen = set()
    dedup_docs, dedup_texts = [], []
    for d, t in zip(docs, texts):
        key = (d.get("title"), d.get("url"), t)
        if key in seen:
            continue
        seen.add(key)
        dedup_docs.append(d)
        dedup_texts.append(t)
    docs, texts = dedup_docs, dedup_texts
    print(f"Deduplicated to {len(texts)} chunks.")


    print(f"Embedding {len(texts)} chunks…")
    vecs = embed_texts(texts)
    print("Building FAISS index…")
    index = build_index(vecs)

    print("Saving index + metadata…")
    faiss.write_index(index, EXT_INDEX_PATH)
    with open(EXT_META_PATH,"wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Wrote {EXT_INDEX_PATH} and {EXT_META_PATH} with {len(texts)} chunks.")

if __name__ == "__main__":
    main()
