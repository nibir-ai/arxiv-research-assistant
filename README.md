# ArXiv Research Assistant

> RAG-powered semantic search over arXiv ML papers — built for ML engineers.

Ask natural language questions about ML research. Get cited, grounded answers
from 80,000+ arXiv papers with hybrid BM25 + vector retrieval and cross-encoder reranking.

## Architecture
User Query → Hybrid Retrieval (BM25 + Vector) → Cross-Encoder Reranking → LLM Generation → Cited Answer

## Stack

- **LlamaIndex** — RAG orchestration
- **Qdrant** — vector database
- **BM25 + Dense hybrid** — retrieval
- **cross-encoder/ms-marco** — reranking
- **RAGAS** — evaluation
- **FastAPI** — API server

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

## Run

```bash
# 1. Ingest papers (fetches from arXiv API — no GPU needed)
python scripts/ingest_papers.py

# 2. Embed + index (run on Google Colab for speed)
# See notebooks/01_data_exploration.ipynb

# 3. Start API
uvicorn backend.app.main:app --reload --port 8000
```

## Evaluation

RAGAS metrics: faithfulness · answer relevancy · context recall

## Status

🚧 In active development