"""tests/test_retrieval.py — Tests for retrieval logic (with mocks for CI)."""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_chunk_text_basic():
    from src.rag.build_index import chunk_text
    text = " ".join([f"word{i}" for i in range(400)])
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 300 for c in chunks)


def test_chunk_text_short_text():
    from src.rag.build_index import chunk_text
    text = "This is a short complaint."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap():
    from src.rag.build_index import chunk_text
    words = [f"w{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Second chunk should start with words from end of first chunk
    first_words = set(chunks[0].split()[-20:])
    second_words = set(chunks[1].split()[:20])
    assert len(first_words & second_words) > 0


def test_retrieve_with_mocked_index():
    """Test retrieve logic without real FAISS index."""
    import src.rag.retrieve as retrieve_module

    mock_index = MagicMock()
    mock_index.ntotal = 100

    # Simulate: first result above threshold, second below
    mock_index.search.return_value = (
        np.array([[0.75, 0.60, 0.20]]),
        np.array([[0, 1, 2]]),
    )

    mock_doc_map = [
        {"chunk_id": "123_c0", "complaint_id": "123", "product": "Mortgage",
         "issue": "payment", "company": "TestBank", "date_received": "2023-01-01",
         "chunk_index": 0, "text": "I had a mortgage payment problem."},
        {"chunk_id": "456_c0", "complaint_id": "456", "product": "Credit card",
         "issue": "billing", "company": "TestBank", "date_received": "2023-02-01",
         "chunk_index": 0, "text": "My credit card billing was wrong."},
        {"chunk_id": "789_c0", "complaint_id": "789", "product": "Student loan",
         "issue": "other", "company": "TestBank", "date_received": "2023-03-01",
         "chunk_index": 0, "text": "Student loan problem here."},
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)

    # Patch globals
    retrieve_module._index = mock_index
    retrieve_module._doc_map = mock_doc_map
    retrieve_module._embed_model = mock_model
    retrieve_module._index_meta = {"version": "v_test"}

    from src.rag.retrieve import retrieve

    result = retrieve("mortgage payment issue", top_k=5, threshold=0.35)
    assert result["refused"] is False
    assert len(result["chunks"]) >= 1
    assert result["chunks"][0]["complaint_id"] == "123"
    assert result["max_score"] == 0.75


def test_retrieve_refusal_when_scores_too_low():
    """Test that retrieval refuses when all scores below threshold."""
    import src.rag.retrieve as retrieve_module

    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_index.search.return_value = (
        np.array([[0.10, 0.08, 0.05]]),
        np.array([[0, 1, 2]]),
    )

    retrieve_module._index = mock_index
    retrieve_module._doc_map = [
        {"chunk_id": "1_c0", "complaint_id": "1", "product": "Mortgage",
         "issue": "test", "company": "Bank", "date_received": "2023-01-01",
         "chunk_index": 0, "text": "unrelated text"},
    ] * 3
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    retrieve_module._embed_model = mock_model

    from src.rag.retrieve import retrieve

    result = retrieve("quantum physics neutron stars", top_k=5, threshold=0.35)
    assert result["refused"] is True
    assert result["chunks"] == []
    assert "threshold" in result["reason"].lower()
