from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import SearchRequest, SearchResponse, PaperResult
from app.core.reranking import rerank
from app.core.generation import generate_response
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=dict)
async def search_papers(request: SearchRequest):
    """Search arXiv papers and return an AI-generated answer with citations."""
    try:
        # Import here to avoid circular deps
        from app.main import get_retriever
        retriever = get_retriever()

        from app.core.retrieval import retrieve
        nodes = await retrieve(retriever, request.query)

        if not nodes:
            raise HTTPException(status_code=404, detail="No relevant papers found.")

        if request.use_reranking:
            nodes = rerank(request.query, nodes)

        result = await generate_response(request.query, nodes)

        return {
            "query": request.query,
            "answer": result["answer"],
            "citations": result["citations"],
            "total_retrieved": len(nodes),
            "reranking_applied": request.use_reranking,
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))