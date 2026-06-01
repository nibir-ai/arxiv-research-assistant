"""
Run once to fetch papers, embed them, and index into Qdrant.
Usage:
    cd arxiv-research-assistant
    source venv/bin/activate
    python scripts/embed_and_index.py
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config import get_settings
from app.core.ingestion import fetch_arxiv_papers, papers_to_documents, chunk_documents
from app.core.index_manager import get_qdrant_client

settings = get_settings()

# ── Config ─────────────────────────────────────────────
QUERIES = [
    "retrieval augmented generation RAG",
    "large language model fine-tuning LoRA",
    "AI agents autonomous LLM planning tool use",
    "vector database semantic search embeddings",
    "transformer attention mechanism BERT GPT",
    "model context protocol MCP tool use",
    "LLM evaluation hallucination benchmark",
]
MAX_PER_QUERY   = 30
NODES_PATH      = "data/nodes.json"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64


def main():
    # ── 1. Fetch ────────────────────────────────────────
    print("\n[1/5] Fetching papers from arXiv...")
    all_papers, seen_ids = [], set()
    for idx, q in enumerate(QUERIES):
        print(f"[{idx+1}/{len(QUERIES)}] Fetching query: '{q}'...")
        try:
            papers = fetch_arxiv_papers(q, max_results=MAX_PER_QUERY)
            new_papers = 0
            for p in papers:
                if p["paper_id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["paper_id"])
                    new_papers += 1
            print(f"  -> Fetched {len(papers)} papers ({new_papers} new unique papers).")
        except Exception as e:
            print(f"  -> ERROR: Failed to fetch query '{q}': {e}")

        # Sleep to comply with arXiv's API usage policy (max 1 request per 3 seconds)
        if idx < len(QUERIES) - 1:
            print("  -> Sleeping 4.0 seconds before next query...")
            time.sleep(4.0)

    print(f"\n  Total unique papers: {len(all_papers)}")
    if not all_papers:
        print("No papers fetched. Exiting.")
        sys.exit(0)

    # ── 2. Chunk ────────────────────────────────────────
    print("\n[2/5] Converting and chunking...")
    docs  = papers_to_documents(all_papers)
    nodes = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  Total chunks: {len(nodes)}")

    # ── 3. Save nodes for BM25 ──────────────────────────
    print(f"\n[3/5] Saving nodes to {NODES_PATH}...")
    os.makedirs("data", exist_ok=True)
    node_data = [
        {"text": n.get_content(), "metadata": n.metadata}
        for n in nodes
    ]
    with open(NODES_PATH, "w", encoding="utf-8") as f:
        json.dump(node_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(node_data)} node records.")

    # ── 4. Embed ────────────────────────────────────────
    print(f"\n[4/5] Loading embedding model: {settings.embedding_model}")
    print("  (Downloading ~130MB on first run - this is normal)")
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    Settings.embed_model = embed_model
    print("  Embedding model ready.")

    # ── 5. Index into Qdrant ────────────────────────────
    print(f"\n[5/5] Indexing into Qdrant collection '{settings.collection_name}'...")
    print(f"  Endpoint: {settings.qdrant_url}")

    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\n[OK] Done!")
    print(f"   Papers indexed : {len(all_papers)}")
    print(f"   Chunks indexed : {len(nodes)}")
    print(f"   Collection     : {settings.collection_name}")
    print(f"\nNext: restart the API server -> uvicorn backend.app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()