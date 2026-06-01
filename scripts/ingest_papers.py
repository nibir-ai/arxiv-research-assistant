"""
Run this script to fetch arXiv papers and index them into Qdrant.
Usage: python scripts/ingest_papers.py
"""
import sys, os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.config import get_settings
from app.core.ingestion import fetch_arxiv_papers, papers_to_documents, chunk_documents

settings = get_settings()

QUERIES = [
    "retrieval augmented generation RAG",
    "large language model fine-tuning LoRA",
    "AI agents autonomous LLM planning",
    "vector database semantic search embeddings",
    "transformer attention mechanism NLP",
]

MAX_PAPERS_PER_QUERY = 40

if __name__ == "__main__":
    all_papers = []
    seen_ids = set()

    for idx, q in enumerate(QUERIES):
        print(f"[{idx+1}/{len(QUERIES)}] Fetching papers for query: '{q}'...")
        try:
            papers = fetch_arxiv_papers(q, max_results=MAX_PAPERS_PER_QUERY)
            new_papers = 0
            for p in papers:
                if p["paper_id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["paper_id"])
                    new_papers += 1
            print(f"  -> Fetched {len(papers)} papers ({new_papers} new unique papers).")
        except Exception as e:
            print(f"  -> ERROR: Failed to fetch papers for query '{q}': {e}")

        # Sleep to comply with arXiv's API usage policy (max 1 request per 3 seconds)
        if idx < len(QUERIES) - 1:
            print("  -> Sleeping 3.5 seconds before next query...")
            time.sleep(3.5)

    print(f"\nTotal unique papers fetched: {len(all_papers)}")

    if not all_papers:
        print("No papers fetched. Exiting.")
        sys.exit(0)

    docs = papers_to_documents(all_papers)
    nodes = chunk_documents(docs)

    print(f"Total chunks created: {len(nodes)}")
    print("\nNext step: run the embedding + indexing in Google Colab (GPU recommended).")
    print("See notebooks/01_data_exploration.ipynb to get started.")