"""
Manages the global index singleton.
Index is built once via scripts/embed_and_index.py
then loaded on every API startup.
"""
import json
import logging
import os
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.llms.ollama import Ollama     # <-- Ollama instead of Gemini
from dotenv import load_dotenv

from app.config import get_settings

logger = logging.getLogger(__name__)

# Load .env from project root (so environment variables are set)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ── Global LLM: Ollama (local, free, no quotas) ───────────
Settings.llm = Ollama(model="gemma2:2b", request_timeout=120.0)

settings = get_settings()

# ── Globals ────────────────────────────────────────────
_retriever = None
_qdrant_client: Optional[QdrantClient] = None
_async_qdrant_client: Optional[AsyncQdrantClient] = None
# Resolve the absolute path to the project root directory (4 levels up from backend/app/core/index_manager.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NODES_PATH = os.path.join(BASE_DIR, "data", "nodes.json")


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        logger.info(f"Qdrant client connected: {settings.qdrant_url}")
    return _qdrant_client


def get_async_qdrant_client() -> AsyncQdrantClient:
    global _async_qdrant_client
    if _async_qdrant_client is None:
        _async_qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        logger.info(f"Async Qdrant client connected: {settings.qdrant_url}")
    return _async_qdrant_client


def load_nodes_from_disk() -> list[TextNode]:
    """Load saved nodes from JSON for BM25 retriever."""
    if not os.path.exists(NODES_PATH):
        logger.warning(f"No nodes file found at {NODES_PATH}. Run embed_and_index.py first.")
        return []

    with open(NODES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes = [
        TextNode(text=item["text"], metadata=item["metadata"])
        for item in raw
    ]
    logger.info(f"Loaded {len(nodes)} nodes from disk for BM25")
    return nodes


def build_retriever() -> Optional[QueryFusionRetriever]:
    """Build hybrid retriever from existing Qdrant index + BM25."""
    client = get_qdrant_client()
    aclient = get_async_qdrant_client()

    # Check collection exists
    collections = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in collections:
        logger.warning(
            f"Collection '{settings.collection_name}' not found in Qdrant. "
            "Run scripts/embed_and_index.py first."
        )
        return None

    # Set embedding model (this stays the same)
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    Settings.embed_model = embed_model

    # Vector retriever from existing Qdrant collection (with async client)
    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=settings.top_k_retrieval
    )

    # BM25 retriever from saved nodes
    nodes = load_nodes_from_disk()
    if not nodes:
        logger.warning("BM25 disabled — no nodes on disk. Falling back to vector-only.")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=settings.top_k_retrieval,
    )

    # Hybrid fusion
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.top_k_retrieval,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
    )

    logger.info("Hybrid retriever (BM25 + vector) ready.")
    return hybrid_retriever


def get_retriever():
    global _retriever
    if _retriever is None:
        raise RuntimeError(
            "Retriever not initialised. "
            "Run scripts/embed_and_index.py then restart the server."
        )
    return _retriever


def set_retriever(r):
    global _retriever
    _retriever = r