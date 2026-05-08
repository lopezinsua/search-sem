"""
Tests for the search-sem API.
Uses a small in-memory FAISS index (no disk, no real corpus).
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

DUMMY_TEXTS = [
    "RAG combines retrieval with generation to improve LLM answers.",
    "FAISS is a library for efficient similarity search of dense vectors.",
    "Sentence transformers produce embeddings for semantic search tasks.",
    "FastAPI is a modern Python web framework built on Starlette.",
    "Vector databases store embeddings and support nearest-neighbor queries.",
    "Chunking splits documents into smaller pieces before embedding.",
    "Cosine similarity measures the angle between two vectors.",
    "LangChain orchestrates chains of LLM calls and tools.",
    "Retrieval-augmented generation reduces hallucinations in LLMs.",
    "Dense retrieval outperforms BM25 on many semantic benchmarks.",
]

DIM = 384  # all-MiniLM-L6-v2 dimension


def _make_dummy_index(tmp_dir: str) -> tuple[str, str]:
    """Build a tiny FAISS index from random normalized vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(DUMMY_TEXTS), DIM)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)

    index_path = str(Path(tmp_dir) / "index.faiss")
    chunks_path = str(Path(tmp_dir) / "chunks.json")
    faiss.write_index(index, index_path)

    chunks = [{"text": t, "source": f"dummy/{i}.txt"} for i, t in enumerate(DUMMY_TEXTS)]
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    return index_path, chunks_path


def _dummy_embed(text: str) -> list[float]:
    """Returns a deterministic normalized vector so tests are reproducible."""
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    vec = rng.standard_normal(DIM).astype("float32")
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmp:
        index_path, chunks_path = _make_dummy_index(tmp)
        with (
            patch("src.config.INDEX_PATH", index_path),
            patch("src.config.CHUNKS_PATH", chunks_path),
            patch("src.embeddings.embed_one", side_effect=_dummy_embed),
            patch("src.index.INDEX_PATH", index_path),
            patch("src.index.CHUNKS_PATH", chunks_path),
        ):
            from src.api import app

            with TestClient(app) as c:
                yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_search_returns_results(client):
    r = client.post("/search", json={"q": "how does RAG work"})
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(k in results[0] for k in ("text", "source", "score"))


def test_search_top_k(client):
    r = client.post("/search", json={"q": "vector similarity", "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_search_empty_query(client):
    r = client.post("/search", json={"q": "   "})
    assert r.status_code == 400


def test_search_results_ordered_by_score(client):
    r = client.post("/search", json={"q": "semantic search embeddings"})
    assert r.status_code == 200
    scores = [h["score"] for h in r.json()]
    assert scores == sorted(scores, reverse=True)
