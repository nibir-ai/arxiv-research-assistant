from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import IngestRequest, IngestResponse
from app.core.ingestion import fetch_arxiv_papers, papers_to_documents, chunk_documents
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_ingest_status = {"status": "idle", "papers": 0, "chunks": 0}


def _run_ingestion(query: str, max_papers: int, categories):
    global _ingest_status
    _ingest_status["status"] = "running"
    try:
        papers = fetch_arxiv_papers(query, max_papers, categories)
        docs = papers_to_documents(papers)
        nodes = chunk_documents(docs)
        _ingest_status = {
            "status": "done",
            "papers": len(papers),
            "chunks": len(nodes),
        }
        logger.info(f"Ingestion complete: {len(papers)} papers, {len(nodes)} chunks")
    except Exception as e:
        _ingest_status["status"] = f"error: {str(e)}"
        logger.error(f"Ingestion failed: {e}")


@router.post("/", response_model=IngestResponse)
async def ingest_papers(request: IngestRequest, background_tasks: BackgroundTasks):
    """Fetch arXiv papers and queue them for indexing."""
    if _ingest_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    background_tasks.add_task(
        _run_ingestion,
        request.query,
        request.max_papers,
        request.categories,
    )

    return IngestResponse(
        papers_fetched=0,
        chunks_indexed=0,
        collection="queued",
        status="ingestion started in background",
    )


@router.get("/status")
async def ingestion_status():
    """Check current ingestion status."""
    return _ingest_status