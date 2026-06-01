"""
Generation module — builds a cited answer from retrieved context.
Uses the globally configured LLM via LlamaIndex Settings.llm.
"""
import logging
from typing import Optional

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    "You are an expert AI research assistant helping ML engineers.\n"
    "Use the following arXiv papers to answer the question.\n"
    "Cite papers using [1], [2], etc. matching their order below.\n"
    "If the context does not contain the answer, say so clearly.\n"
    "Be technical, precise, and concise.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def _build_context(nodes: list) -> str:
    """Build numbered context string from retrieved nodes."""
    sections = []
    for i, node in enumerate(nodes, 1):
        # Handle both NodeWithScore and TextNode
        if hasattr(node, "node"):
            text = node.node.get_content()
            meta = node.node.metadata
        else:
            text = node.get_content()
            meta = node.metadata

        title   = meta.get("title", "Unknown")
        authors = meta.get("authors", [])
        url     = meta.get("url", "")

        author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        sections.append(f"[{i}] {title}\nAuthors: {author_str}\nURL: {url}\n{text}")

    return "\n\n---\n\n".join(sections)


def _build_citations(nodes: list) -> list[dict]:
    """Build citation list from retrieved nodes."""
    citations = []
    for i, node in enumerate(nodes, 1):
        if hasattr(node, "node"):
            meta = node.node.metadata
        else:
            meta = node.metadata

        citations.append({
            "index":     i,
            "title":     meta.get("title", "Unknown"),
            "authors":   meta.get("authors", []),
            "url":       meta.get("url", ""),
            "published": meta.get("published", ""),
        })
    return citations


async def generate_response(
        query: str,
        nodes: Optional[list] = None,
) -> dict:
    """
    Generate a cited answer to `query` using the configured LLM.

    Returns:
        dict with keys: answer (str), citations (list[dict])
    """
    try:
        if not nodes:
            return {
                "answer":    "No relevant papers found. Try refining your query.",
                "citations": [],
            }

        context   = _build_context(nodes)
        citations = _build_citations(nodes)

        response = await Settings.llm.apredict(
            QA_PROMPT,
            context_str=context,
            query_str=query,
        )

        return {
            "answer":    str(response).strip(),
            "citations": citations,
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return {
            "answer":    f"Error generating answer: {e}",
            "citations": [],
        }