import arxiv
import logging
import time
from typing import Optional
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


def fetch_arxiv_papers(
        query: str,
        max_results: int = 50,
        categories: Optional[list[str]] = None,
        client: Optional[arxiv.Client] = None
) -> list[dict]:
    """Fetch papers from arXiv API with automatic retry and backoff on rate limits."""
    if categories:
        cat_filter = " OR ".join([f"cat:{c}" for c in categories])
        full_query = f"({query}) AND ({cat_filter})"
    else:
        full_query = query

    if client is None:
        # Dynamically set page_size to match max_results to optimize query performance
        # and minimize the data transferred per request.
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=5,
            num_retries=5
        )

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    max_retries = 3
    backoff = 10.0

    for attempt in range(max_retries):
        try:
            papers = []
            for result in client.results(search):
                papers.append({
                    "paper_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "url": result.entry_id,
                    "published": str(result.published.date()) if result.published else None,
                    "categories": result.categories,
                })
            logger.info(f"Fetched {len(papers)} papers for query: '{query}'")
            return papers
        except arxiv.HTTPError as e:
            if "429" in str(e) or "503" in str(e):
                if attempt < max_retries - 1:
                    print(f"  -> Got {e}. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            raise e


def papers_to_documents(papers: list[dict]) -> list[Document]:
    """Convert raw paper dicts to LlamaIndex Documents."""
    docs = []
    for p in papers:
        text = (
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Published: {p.get('published', 'N/A')}\n"
            f"Abstract: {p['abstract']}"
        )
        doc = Document(
            text=text,
            metadata={
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": p["authors"],
                "url": p["url"],
                "published": p.get("published", ""),
                "categories": p.get("categories", []),
            },
            excluded_embed_metadata_keys=["paper_id", "url", "categories"],
            excluded_llm_metadata_keys=["paper_id", "categories"],
        )
        docs.append(doc)
    return docs


def chunk_documents(
        documents: list[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into chunks for indexing."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")
    return nodes