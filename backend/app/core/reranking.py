import logging
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Rerank retrieved nodes using a cross-encoder."""
    if not nodes:
        return nodes

    reranker = get_reranker()
    pairs = [(query, node.get_content()) for node in nodes]
    scores = reranker.predict(pairs)

    for node, score in zip(nodes, scores):
        node.score = float(score)

    reranked = sorted(nodes, key=lambda x: x.score, reverse=True)
    top = reranked[:settings.top_k_rerank]

    logger.info(f"Reranked {len(nodes)} → top {len(top)} nodes")
    return top