from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchRequest
from app.core.reranking import rerank
from app.core.generation import generate_response
from app.core.index_manager import get_retriever
from app.core.retrieval import retrieve
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/")
async def search_papers(request: SearchRequest):
    """
    Search arXiv papers using hybrid RAG.
    Returns an AI-generated answer with inline citations.
    """
    try:
        retriever = get_retriever()
        nodes = await retrieve(retriever, request.query)

        if not nodes:
            raise HTTPException(
                status_code=404,
                detail="No relevant papers found. Try a different query."
            )

        if request.use_reranking:
            nodes = rerank(request.query, nodes)

        result = await generate_response(request.query, nodes)

        return {
            "query":             request.query,
            "answer":            result["answer"],
            "citations":         result["citations"],
            "total_retrieved":   len(nodes),
            "reranking_applied": request.use_reranking,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))