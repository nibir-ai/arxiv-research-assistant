"""
Generation module – builds an answer from retrieved context.
Uses the globally configured LLM (Gemini) via llama_index.
"""
import logging
from typing import Optional
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from app.core.index_manager import get_retriever

logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    "You are a helpful AI research assistant. "
    "Use the following pieces of context to answer the user's question. "
    "If the context doesn't contain the answer, say you don't know.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

async def generate_response(
        query: str,
        nodes: Optional[list[TextNode]] = None,
) -> str:
    """
    Generate an answer to `query` using the LLM.
    If `nodes` are provided, they are used as context directly.
    Otherwise, the retriever is used to fetch relevant documents.
    """
    try:
        # If nodes are passed (from search route), use them directly
        if nodes is None:
            retriever = get_retriever()
            nodes = await retriever.aretrieve(query)

        if not nodes:
            return "No relevant papers found. Try refining your query."

        # Build context from the nodes
        context = "\n\n".join(
            f"Title: {node.metadata.get('title', 'Untitled')}\n"
            f"Authors: {node.metadata.get('authors', 'Unknown')}\n"
            f"Abstract: {node.text}"
            for node in nodes
        )

        response = await Settings.llm.apredict(
            QA_PROMPT,
            context_str=context,
            query_str=query,
        )
        return str(response)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Sorry, an error occurred while generating the answer: {e}"