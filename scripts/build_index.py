"""
Build the FAISS index from a corpus directory.
Usage: python scripts/build_index.py --corpus data/corpus/
"""
import argparse
import json
import os
import re
import numpy as np
import faiss
from pathlib import Path

from src.config import INDEX_PATH, CHUNKS_PATH
from src.embeddings import embed

ABSTRACT_SEP = "\n\n---\n\n"
MIN_WORDS = 30


def clean(text: str) -> str:
    text = re.sub(r"@xmath\d+", "[math]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_abstracts(text: str, source: str) -> list[dict]:
    parts = text.split(ABSTRACT_SEP)
    chunks = []
    for part in parts:
        cleaned = clean(part)
        if len(cleaned.split()) >= MIN_WORDS:
            chunks.append({"text": cleaned, "source": source})
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus", help="Directory with .txt files")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    all_chunks: list[dict] = []

    for txt_file in sorted(corpus_path.glob("**/*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        abstracts = split_into_abstracts(text, txt_file.stem)
        all_chunks.extend(abstracts)
        print(f"  {txt_file.name}: {len(abstracts)} abstracts")

    print(f"\nTotal chunks: {len(all_chunks)}")

    batch_size = 128
    all_vectors = []
    for i in range(0, len(all_chunks), batch_size):
        batch = [c["text"] for c in all_chunks[i : i + batch_size]]
        vecs = embed(batch)
        all_vectors.extend(vecs)
        print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    dim = len(all_vectors[0])
    matrix = np.array(all_vectors, dtype="float32")

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"\nIndex saved: {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")


if __name__ == "__main__":
    main()
