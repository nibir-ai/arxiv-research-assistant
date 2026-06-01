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
    """Search arXiv papers — returns AI answer with citations."""
    try:
        retriever = get_retriever()
        nodes = await retrieve(retriever, request.query)

        if not nodes:
            raise HTTPException(status_code=404, detail="No relevant papers found.")

        if request.use_reranking:
            nodes = rerank(request.query, nodes)

        # generate_response now returns a plain string (the answer)
        answer = await generate_response(request.query, nodes)

        # Build citations from node metadata
        citations = []
        for node in nodes[:request.top_k]:   # limit to requested number
            citations.append({
                "title": node.metadata.get("title", "Untitled"),
                "authors": node.metadata.get("authors", "Unknown"),
                "url": node.metadata.get("url", ""),
                "published": node.metadata.get("published", ""),
            })

        return {
            "query": request.query,
            "answer": answer,
            "citations": citations,
            "total_retrieved": len(nodes),
            "reranking_applied": request.use_reranking,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))