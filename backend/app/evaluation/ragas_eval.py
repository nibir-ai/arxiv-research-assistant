"""
RAGAS evaluation for the ArXiv RAG pipeline.
Run this after you have the retrieval pipeline working.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import json
import logging

logger = logging.getLogger(__name__)


# ── Sample test set ───────────────────────────────────
# Add your own ground-truth QA pairs here as you build them
SAMPLE_EVAL_SET = [
    {
        "question": "What is retrieval augmented generation?",
        "ground_truth": "RAG combines a retrieval system with a language model to generate answers grounded in retrieved documents.",
    },
    {
        "question": "How does LoRA reduce the number of trainable parameters?",
        "ground_truth": "LoRA decomposes weight update matrices into two low-rank matrices, drastically reducing trainable parameters.",
    },
    {
        "question": "What is the role of cross-encoder reranking in RAG?",
        "ground_truth": "Cross-encoders jointly encode the query and each document to produce more accurate relevance scores than bi-encoders.",
    },
]


def build_ragas_dataset(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
) -> Dataset:
    """Build a HuggingFace Dataset for RAGAS evaluation."""
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(dataset: Dataset) -> dict:
    """Run RAGAS metrics and return scores."""
    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    scores = {
        "faithfulness":        round(result["faithfulness"], 4),
        "answer_relevancy":    round(result["answer_relevancy"], 4),
        "context_recall":      round(result["context_recall"], 4),
        "context_precision":   round(result["context_precision"], 4),
    }
    logger.info(f"RAGAS scores: {scores}")
    return scores


def save_results(scores: dict, path: str = "eval_results.json"):
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Results saved to {path}")