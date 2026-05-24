"""
build_index.py — Chunk CFPB complaints, embed with sentence-transformers, persist FAISS index.
Usage: python src/rag/build_index.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
DOC_MAP_PATH = ARTIFACTS_DIR / "doc_map.pkl"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # words per chunk
CHUNK_OVERLAP = 50


def load_complaints(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    col_map = {
        "Consumer complaint narrative": "narrative",
        "complaint_what_happened": "narrative",
        "Product": "product",
        "Issue": "issue",
        "Company": "company",
        "Complaint ID": "complaint_id",
        "Date received": "date_received",
        "Company response to consumer": "company_response",
        "State": "state",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Keep only rows with narratives
    if "narrative" in df.columns:
        df = df[df["narrative"].notna()]
        df = df[df["narrative"].astype(str).str.len() > 50]
    else:
        print("[build_index] WARNING: No narrative column found")

    print(f"[build_index] Loaded {len(df)} complaints with narratives")
    return df.reset_index(drop=True)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping word-based chunks."""
    words = str(text).split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
        if end == len(words):
            break
    return chunks


def build_document_chunks(df: pd.DataFrame) -> tuple:
    """
    Returns:
      chunks: list of text strings
      doc_map: list of dicts with metadata for each chunk
    """
    chunks = []
    doc_map = []

    for idx, row in df.iterrows():
        narrative = str(row.get("narrative", ""))
        if not narrative or narrative == "nan":
            continue

        complaint_id = str(row.get("complaint_id", idx))
        product = str(row.get("product", ""))
        issue = str(row.get("issue", ""))
        company = str(row.get("company", ""))
        date_received = str(row.get("date_received", ""))

        text_chunks = chunk_text(narrative)
        for chunk_i, chunk_text_content in enumerate(text_chunks):
            chunk_id = f"{complaint_id}_c{chunk_i}"
            chunks.append(chunk_text_content)
            doc_map.append({
                "chunk_id": chunk_id,
                "complaint_id": complaint_id,
                "product": product,
                "issue": issue,
                "company": company,
                "date_received": date_received,
                "chunk_index": chunk_i,
                "text": chunk_text_content,
            })

    print(f"[build_index] Created {len(chunks)} chunks from {len(df)} complaints")
    return chunks, doc_map


def embed_chunks(chunks: list, model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed all chunks using sentence-transformers."""
    print(f"[build_index] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[build_index] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity via inner product
    )
    print(f"[build_index] Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS IndexFlatIP (cosine similarity with normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
    index.add(embeddings)
    print(f"[build_index] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_artifacts(index: faiss.Index, doc_map: list, n_complaints: int):
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOC_MAP_PATH, "wb") as f:
        pickle.dump(doc_map, f)

    meta = {
        "n_complaints": n_complaints,
        "n_chunks": len(doc_map),
        "embedding_model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "index_path": str(INDEX_PATH),
        "doc_map_path": str(DOC_MAP_PATH),
        "version": "v1",
    }
    with open(INDEX_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[build_index] ✓ FAISS index saved → {INDEX_PATH}")
    print(f"[build_index] ✓ Doc map saved     → {DOC_MAP_PATH}")
    print(f"[build_index] ✓ Meta saved        → {INDEX_META_PATH}")


def main():
    data_path = "data/raw/complaints.csv"
    if not Path(data_path).exists():
        data_path = "data/samples/complaints_sample.csv"
    if not Path(data_path).exists():
        print("[build_index] ERROR: Run ingest.py first.")
        sys.exit(1)

    df = load_complaints(data_path)
    chunks, doc_map = build_document_chunks(df)

    if not chunks:
        print("[build_index] ERROR: No chunks created. Check data.")
        sys.exit(1)

    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_artifacts(index, doc_map, len(df))
    print("\n[build_index] ✓ Index built successfully.")


if __name__ == "__main__":
    main()
