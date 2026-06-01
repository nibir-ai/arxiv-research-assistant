import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.routes import search, ingest
from app.core.index_manager import build_retriever, set_retriever
from dotenv import load_dotenv

# Load .env from project root (two directories up from this file)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ArXiv Research Assistant...")
    logger.info(f"Qdrant: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.collection_name}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    retriever = build_retriever()
    if retriever:
        set_retriever(retriever)
        logger.info("Retriever loaded successfully — API ready.")
    else:
        logger.warning(
            "Retriever NOT loaded. "
            "Run scripts/embed_and_index.py first, then restart."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ArXiv Research Assistant",
    description="RAG-powered semantic search over arXiv ML papers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "collection": settings.collection_name,
        "qdrant": settings.qdrant_url,
    }