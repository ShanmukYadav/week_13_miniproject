"""
retrieve.py — Retrieve relevant complaint chunks with similarity threshold and refusal logic.
"""
import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Optional

ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
DOC_MAP_PATH = ARTIFACTS_DIR / "doc_map.pkl"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"

SIMILARITY_THRESHOLD = 0.35  # Below this → refuse
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"

# Global cache (loaded once at startup)
_index: Optional[faiss.Index] = None
_doc_map: Optional[list] = None
_embed_model: Optional[SentenceTransformer] = None
_index_meta: dict = {}


def load_index():
    global _index, _doc_map, _embed_model, _index_meta
    if _index is not None:
        return  # already loaded

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run build_index.py first.")

    print("[retrieve] Loading FAISS index...")
    _index = faiss.read_index(str(INDEX_PATH))

    with open(DOC_MAP_PATH, "rb") as f:
        _doc_map = pickle.load(f)

    if INDEX_META_PATH.exists():
        with open(INDEX_META_PATH) as f:
            _index_meta = json.load(f)

    model_name = _index_meta.get("embedding_model", MODEL_NAME)
    print(f"[retrieve] Loading embedding model: {model_name}")
    _embed_model = SentenceTransformer(model_name)

    print(f"[retrieve] Index loaded: {_index.ntotal} chunks, {len(_doc_map)} doc entries")


def embed_query(query: str) -> np.ndarray:
    vec = _embed_model.encode([query], normalize_embeddings=True)
    return vec.astype(np.float32)


def retrieve(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    product_filter: Optional[str] = None,
    company_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """
    Retrieve top-k complaint chunks for a query.
    Returns dict with:
      - chunks: list of matching chunk dicts
      - max_score: best similarity score
      - refused: True if no chunk crossed the threshold
      - reason: why refused (if applicable)
    """
    if _index is None:
        load_index()

    query_vec = embed_query(query)

    # Retrieve more than top_k to allow post-filtering
    search_k = min(top_k * 10, _index.ntotal)
    scores, indices = _index.search(query_vec, search_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        if score < threshold:
            continue

        chunk = _doc_map[idx].copy()
        chunk["similarity_score"] = round(float(score), 4)

        # Apply metadata filters
        if product_filter and product_filter.lower() not in str(chunk.get("product", "")).lower():
            continue
        if company_filter and company_filter.lower() not in str(chunk.get("company", "")).lower():
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    max_score = float(scores[0]) if len(scores) > 0 else 0.0

    if not results:
        return {
            "chunks": [],
            "max_score": max_score,
            "refused": True,
            "reason": (
                f"No complaint chunk crossed the similarity threshold of {threshold}. "
                f"Best score was {max_score:.3f}. Cannot provide a grounded answer."
            ),
        }

    return {
        "chunks": results,
        "max_score": max_score,
        "refused": False,
        "reason": None,
    }


def get_index_version() -> str:
    return _index_meta.get("version", "unknown")


if __name__ == "__main__":
    load_index()
    # Quick test
    result = retrieve("mortgage payment issues and foreclosure problems")
    print(f"\nQuery: mortgage payment issues")
    print(f"Refused: {result['refused']}")
    print(f"Max score: {result['max_score']}")
    print(f"Results: {len(result['chunks'])}")
    for c in result["chunks"][:2]:
        print(f"  [{c['similarity_score']:.3f}] {c['complaint_id']}: {c['text'][:100]}...")

    # Test refusal
    result2 = retrieve("quantum physics neutron stars space telescope")
    print(f"\nOff-topic query refused: {result2['refused']} | reason: {result2['reason']}")
