import json
import numpy as np
import faiss
from src.config import INDEX_PATH, CHUNKS_PATH


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def search(index: faiss.Index, chunks: list[dict], query_vec: list[float], top_k: int) -> list[dict]:
    vec = np.array([query_vec], dtype="float32")
    distances, ids = index.search(vec, top_k)
    results = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(dist)})
    return results
