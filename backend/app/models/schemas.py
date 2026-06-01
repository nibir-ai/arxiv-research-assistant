from pydantic import BaseModel
from typing import Optional


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True


class PaperResult(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    score: float
    chunk_text: str
    published: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[PaperResult]
    total_retrieved: int
    reranking_applied: bool


class IngestRequest(BaseModel):
    query: str           # arXiv search query e.g. "RAG retrieval augmented generation"
    max_papers: int = 50
    categories: Optional[list[str]] = None   # e.g. ["cs.AI", "cs.CL"]


class IngestResponse(BaseModel):
    papers_fetched: int
    chunks_indexed: int
    collection: str
    status: str