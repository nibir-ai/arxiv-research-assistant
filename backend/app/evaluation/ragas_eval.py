"""
Custom RAG evaluation — no OpenAI dependency.
Computes 4 meaningful metrics using local embeddings + Ollama.

Metrics:
    answer_relevancy  — cosine similarity between question and answer embeddings
    faithfulness      — Ollama judges if answer is grounded in retrieved context
    context_recall    — cosine similarity between retrieved context and ground truth
    context_precision — fraction of retrieved chunks relevant to the question
"""
import json
import logging
import os
import asyncio
from typing import Optional

from sentence_transformers import SentenceTransformer, util
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Reuse the embedding model already loaded by the pipeline
_embed_model: Optional[SentenceTransformer] = None

FAITHFULNESS_PROMPT = PromptTemplate(
    "You are an evaluation judge.\n\n"
    "Context (retrieved documents):\n{context}\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "How faithful is the answer to the context? "
    "A faithful answer only uses information present in the context.\n"
    "Respond with ONLY a single decimal number between 0.0 and 1.0.\n"
    "Examples: 0.9 (very faithful)  0.5 (partially faithful)  0.1 (not faithful)\n"
    "Score:"
)


def _get_embed_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading evaluation embedding model: {model_name}")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _cosine(text_a: str, text_b: str, model_name: str = "BAAI/bge-small-en-v1.5") -> float:
    """Compute cosine similarity between two texts."""
    model = _get_embed_model(model_name)
    emb_a = model.encode(text_a, convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(text_b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb_a, emb_b))


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    How relevant is the answer to the question?
    Cosine similarity between question and answer embeddings.
    Range: 0.0 – 1.0
    """
    if not answer or not question:
        return 0.0
    return round(max(0.0, _cosine(question, answer)), 4)


def compute_context_recall(
    context_texts: list[str],
    ground_truth: str,
    top_k: int = 3,
) -> float:
    """
    Does the retrieved context cover the ground truth?
    Cosine similarity between combined top-k context chunks and ground truth.
    Range: 0.0 – 1.0
    """
    if not context_texts or not ground_truth:
        return 0.0
    combined = " ".join(context_texts[:top_k])
    return round(max(0.0, _cosine(combined, ground_truth)), 4)


def compute_context_precision(
    question: str,
    context_texts: list[str],
    relevance_threshold: float = 0.45,
) -> float:
    """
    What fraction of retrieved chunks are relevant to the question?
    Uses cosine similarity with a relevance threshold.
    Range: 0.0 – 1.0
    """
    if not context_texts:
        return 0.0
    scores    = [_cosine(question, ctx) for ctx in context_texts]
    relevant  = sum(1 for s in scores if s >= relevance_threshold)
    return round(relevant / len(scores), 4)


async def compute_faithfulness(
    question: str,
    answer: str,
    context_texts: list[str],
    top_k: int = 3,
) -> float:
    """
    Is the answer grounded in the retrieved context?
    Asks the configured LLM to judge on a 0–1 scale.
    Falls back to 0.5 if LLM response cannot be parsed.
    """
    if not answer or not context_texts:
        return 0.0

    context_str = "\n\n---\n\n".join(context_texts[:top_k])[:2000]

    try:
        raw = await Settings.llm.apredict(
            FAITHFULNESS_PROMPT,
            context=context_str,
            question=question,
            answer=answer[:800],
        )
        score = float(raw.strip().split()[0])
        return round(min(max(score, 0.0), 1.0), 4)
    except Exception as e:
        logger.warning(f"Faithfulness scoring failed: {e}. Using 0.5 fallback.")
        return 0.5


async def evaluate_single(
    question: str,
    answer: str,
    context_texts: list[str],
    ground_truth: str,
) -> dict:
    """Evaluate a single QA pair across all 4 metrics."""
    faithfulness = await compute_faithfulness(question, answer, context_texts)
    return {
        "answer_relevancy":  compute_answer_relevancy(question, answer),
        "faithfulness":      faithfulness,
        "context_recall":    compute_context_recall(context_texts, ground_truth),
        "context_precision": compute_context_precision(question, context_texts),
    }


async def run_evaluation(
    questions:     list[str],
    answers:       list[str],
    contexts:      list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run evaluation across all QA pairs and return averaged scores.
    Note: This function is now async — await it from the eval script.
    """
    logger.info(f"Running custom evaluation on {len(questions)} samples...")

    all_scores = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        scores = await evaluate_single(q, a, ctx, gt)
        all_scores.append(scores)
        logger.debug(f"  Q: {q[:50]}... | {scores}")

    # Average across all samples
    averaged = {
        metric: round(
            sum(s[metric] for s in all_scores) / len(all_scores), 4
        )
        for metric in all_scores[0].keys()
    }

    logger.info(f"Averaged scores: {averaged}")
    return averaged


def save_results(scores: dict, path: str = "data/eval_scores.json"):
    """Persist scores to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Scores saved to {path}")


# ── Legacy shim so old imports don't break ────────────────────────────────────
def build_ragas_dataset(questions, answers, contexts, ground_truths):
    """Kept for backward compatibility. Returns a plain dict now."""
    return {
        "questions":     questions,
        "answers":       answers,
        "contexts":      contexts,
        "ground_truths": ground_truths,
    }