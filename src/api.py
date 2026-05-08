from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.config import TOP_K
from src.embeddings import embed_one, get_model
from src.index import load_index, search

_index, _chunks = None, None

STATIC_DIR = Path(__file__).parent.parent / "static"
MAX_QUERY_LENGTH = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _chunks
    get_model()  # pre-load at startup to prevent race conditions on first request
    _index, _chunks = load_index()
    yield


app = FastAPI(title="search-sem", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Query(BaseModel):
    q: str
    top_k: int = TOP_K

    @field_validator("q")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} characters")
        return v


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
