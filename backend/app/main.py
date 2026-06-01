import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from app.config import get_settings
from app.api.routes import search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global retriever — loaded once at startup
_retriever = None


def get_retriever():
    if _retriever is None:
        raise RuntimeError("Retriever not initialised. Ingest papers first.")
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ArXiv Research Assistant API...")
    logger.info(f"Qdrant: {settings.qdrant_url} | Collection: {settings.collection_name}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ArXiv Research Assistant",
    description="RAG-powered semantic search over arXiv ML papers for engineers.",
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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}