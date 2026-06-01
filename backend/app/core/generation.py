import logging
from openai import AsyncOpenAI
from llama_index.core.schema import NodeWithScore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


def build_context(nodes: list[NodeWithScore]) -> str:
    """Build context string from retrieved nodes."""
    sections = []
    for i, node in enumerate(nodes, 1):
        meta = node.metadata
        sections.append(
            f"[{i}] {meta.get('title', 'Unknown')}\n"
            f"URL: {meta.get('url', '')}\n"
            f"{node.get_content()}\n"
        )
    return "\n---\n".join(sections)


async def generate_response(query: str, nodes: list[NodeWithScore]) -> dict:
    """Generate a response with citations."""
    context = build_context(nodes)

    system_prompt = (
        "You are a research assistant for ML engineers. "
        "Answer questions based strictly on the provided arXiv papers. "
        "Cite papers using [1], [2] etc. corresponding to their order in context. "
        "Be concise, technical, and precise. "
        "If the context doesn't contain enough information, say so clearly."
    )

    user_prompt = (
        f"Context from arXiv papers:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a thorough answer with inline citations."
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    citations = [
        {
            "index": i + 1,
            "title": node.metadata.get("title", ""),
            "url": node.metadata.get("url", ""),
            "authors": node.metadata.get("authors", []),
            "published": node.metadata.get("published", ""),
        }
        for i, node in enumerate(nodes)
    ]

    return {"answer": answer, "citations": citations}