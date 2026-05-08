from sentence_transformers import SentenceTransformer
from src.config import SENTENCE_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return shared model instance, loaded once at startup."""
    global _model
    if _model is None:
        _model = SentenceTransformer(SENTENCE_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
