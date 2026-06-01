import logging
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import QueryBundle
from qdrant_client import QdrantClient
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def build_hybrid_retriever(nodes, qdrant_client: QdrantClient):
    """Build a hybrid BM25 + vector retriever."""

    # Vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # BM25 retriever (keyword)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=settings.top_k_retrieval,
    )

    # Vector retriever
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=settings.top_k_retrieval
    )

    # Fusion retriever — combines both
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.top_k_retrieval,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
    )

    logger.info("Hybrid retriever (BM25 + vector) built successfully")
    return hybrid_retriever


async def retrieve(retriever, query: str):
    """Run retrieval for a query."""
    query_bundle = QueryBundle(query_str=query)
    nodes = await retriever.aretrieve(query_bundle)
    logger.info(f"Retrieved {len(nodes)} nodes for query: '{query}'")
    return nodes