# src/vector_store.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"

_model = None
_index = None
_metas = None

def _load_all():
    """Lazy-load index, metadata, and model once."""
    global _model, _index, _metas
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _metas is None:
        with open(METADATA_PATH, "rb") as f:
            _metas = pickle.load(f)
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _index, _metas, _model

def embed(texts):
    """Encode a list of texts and L2-normalize (cosine-ready)."""
    _, _, model = _load_all()
    vecs = model.encode(texts, batch_size=64).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def search(query: str, k: int = 10):
    """Semantic search over FAISS; returns (scores, indices, metas)."""
    index, metas, _ = _load_all()
    qv = embed([query])
    D, I = index.search(qv, k)
    return D[0], I[0], metas
