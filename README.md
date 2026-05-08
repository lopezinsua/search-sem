# search-sem

Semantic search engine over AI/ML research paper abstracts from arXiv.

**Live demo:** [huggingface.co/spaces/lopezinsua/search-sem](https://huggingface.co/spaces/lopezinsua/search-sem)

---

## What it does

Send a natural language query, get back the most semantically similar AI/ML paper abstracts ranked by cosine similarity. No keyword matching — pure vector search.

```
POST /search
{ "q": "how does retrieval augmented generation work", "top_k": 5 }
```

Returns papers like:
```json
[
  {
    "text": "LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG...",
    "source": "arxiv_0000",
    "score": 0.69
  }
]
```

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI + uvicorn |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, free) |
| Vector search | FAISS `IndexFlatIP` (cosine similarity) |
| Corpus | arXiv API — cs.AI, cs.CL, cs.LG, cs.IR |
| Deploy | HuggingFace Spaces (Docker) |
| Runtime | Python 3.12 |

**No API keys required.** Embeddings run locally via sentence-transformers.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Terminal-style search UI |
| `GET` | `/health` | Health check |
| `POST` | `/search` | Semantic search |

## Run locally

```bash
# Python 3.12 required
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Download corpus and build index
python scripts/download_corpus.py --limit 2000
python scripts/build_index.py

# Start server
uvicorn src.api:app --reload
# → http://localhost:8000
```

## Project structure

```
search-sem/
├── src/
│   ├── api.py          # FastAPI: GET / · GET /health · POST /search
│   ├── embeddings.py   # sentence-transformers wrapper
│   ├── index.py        # FAISS load + search
│   └── config.py       # constants
├── scripts/
│   ├── download_corpus.py  # arXiv API → data/corpus/*.txt
│   └── build_index.py      # corpus → data/index.faiss + chunks.json
├── static/
│   └── index.html      # terminal UI (vanilla JS)
├── tests/
│   └── test_api.py     # pytest + httpx TestClient
└── Dockerfile          # HuggingFace Spaces deploy
```

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```
