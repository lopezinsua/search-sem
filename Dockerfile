FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformers model so first request is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Download 300 arXiv AI/ML abstracts and build the FAISS index at build time
RUN PYTHONPATH=. python scripts/download_corpus.py --limit 300 --out data/corpus --no-delay \
    && PYTHONPATH=. python scripts/build_index.py --corpus data/corpus \
    && rm -rf data/corpus

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
