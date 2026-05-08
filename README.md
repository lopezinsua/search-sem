---
title: search-sem
emoji: 🔍
colorFrom: green
colorTo: black
sdk: docker
pinned: false
---

# search-sem

Semantic search over AI/ML research paper abstracts from arXiv.

**Stack:** FastAPI · FAISS · sentence-transformers (all-MiniLM-L6-v2) · Python 3.12

## Endpoints

- `GET /` — terminal-style search UI
- `GET /health` — health check
- `POST /search` — semantic search

```json
POST /search
{ "q": "how does attention mechanism work", "top_k": 5 }
```

## Run locally

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Build index (needs corpus in data/corpus/)
python scripts/download_corpus.py --limit 5000
python scripts/build_index.py

# Start server
uvicorn src.api:app --reload
```
