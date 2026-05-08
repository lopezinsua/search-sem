from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.index import load_index, search
from src.embeddings import embed_one
from src.config import TOP_K

_index, _chunks = None, None

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _chunks
    _index, _chunks = load_index()
    yield


app = FastAPI(title="search-sem", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Query(BaseModel):
    q: str
    top_k: int = TOP_K


class Result(BaseModel):
    text: str
    source: str
    score: float


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=list[Result])
def search_endpoint(query: Query):
    if not query.q.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    vec = embed_one(query.q)
    hits = search(_index, _chunks, vec, query.top_k)
    return hits
