"""
RAGAS evaluation script — run against the live pipeline.
Usage:
    cd E:\\arxiv-research-assistant
    venv\\Scripts\\activate
    python scripts/run_eval.py
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.index_manager import build_retriever, set_retriever, init_llm
from app.core.retrieval import retrieve
from app.core.reranking import rerank
from app.core.generation import generate_response
from app.evaluation.ragas_eval import build_ragas_dataset, run_evaluation, save_results

# ── Evaluation dataset ────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "question": "What is retrieval augmented generation and how does it work?",
        "ground_truth": (
            "RAG combines a retrieval system with a language model. "
            "It retrieves relevant documents from a knowledge base, "
            "then uses an LLM to generate an answer grounded in those documents."
        ),
    },
    {
        "question": "What is LoRA and how does it reduce training parameters?",
        "ground_truth": (
            "LoRA is a parameter-efficient fine-tuning method that decomposes "
            "weight update matrices into two low-rank matrices, drastically "
            "reducing trainable parameters while maintaining model quality."
        ),
    },
    {
        "question": "How do AI agents use tools for planning and reasoning?",
        "ground_truth": (
            "AI agents use reasoning loops such as ReAct to interleave thinking "
            "and tool use. They select tools based on the current task, call them, "
            "observe results, and continue reasoning until the task is complete."
        ),
    },
    {
        "question": "What are vector databases used for in AI applications?",
        "ground_truth": (
            "Vector databases store high-dimensional embeddings and support "
            "efficient approximate nearest neighbour search, enabling semantic "
            "similarity retrieval for RAG systems and recommendation engines."
        ),
    },
    {
        "question": "How is the RAGAS framework used to evaluate RAG systems?",
        "ground_truth": (
            "RAGAS evaluates RAG systems using metrics including faithfulness, "
            "answer relevancy, and context recall, measuring whether answers are "
            "grounded in retrieved context and relevant to the original question."
        ),
    },
    {
        "question": "What is the difference between fine-tuning and prompt engineering?",
        "ground_truth": (
            "Fine-tuning updates model weights on task-specific data, while "
            "prompt engineering crafts input instructions to guide the model "
            "without changing weights. Fine-tuning is more costly but produces "
            "specialised behaviour; prompt engineering is faster and cheaper."
        ),
    },
    {
        "question": "How does cross-encoder reranking improve retrieval quality?",
        "ground_truth": (
            "Cross-encoders jointly encode the query and each candidate document "
            "to produce more accurate relevance scores than bi-encoders. "
            "Reranking applies this after initial retrieval to promote the most "
            "relevant documents to the top."
        ),
    },
]


async def main():
    # Configure stdout to use UTF-8 on Windows to print nerd emojis without crashing
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "═" * 60)
    print("  ARXIV RAG — RAGAS EVALUATION")
    print("═" * 60)

    print("\n[Setup] Initialising pipeline...")
    init_llm()
    retriever = build_retriever()
    if retriever is None:
        print("ERROR: Retriever not built. Run embed_and_index.py first.")
        return
    set_retriever(retriever)
    print("[Setup] Pipeline ready.\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_SET, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(EVAL_SET)}] {q[:65]}...")

        nodes  = await retrieve(retriever, q)
        nodes  = rerank(q, nodes)
        result = await generate_response(q, nodes)

        answer = result["answer"] if isinstance(result, dict) else str(result)

        questions.append(q)
        answers.append(answer)
        contexts.append([n.get_content() if hasattr(n, "get_content") else n.node.get_content() for n in nodes])
        ground_truths.append(gt)

        print(f"   → {answer[:100]}...")
        print(f"   → {len(nodes)} nodes retrieved\n")

    print("Running RAGAS metrics (this may take 2–5 minutes)...")
    dataset = build_ragas_dataset(questions, answers, contexts, ground_truths)
    scores  = run_evaluation(dataset)

    print("\n" + "═" * 60)
    print("  RESULTS")
    print("═" * 60)
    for metric, score in scores.items():
        filled = int(score * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        grade  = "\uf058 [PASS]" if score >= 0.75 else "\uf071 [WARN]" if score >= 0.5 else "\uf057 [FAIL]"
        print(f"  {grade}  {metric:<28} {score:.4f}  {bar}")
    print("═" * 60)

    os.makedirs("data", exist_ok=True)
    save_results(scores, "data/ragas_scores.json")
    print(f"\n  Saved → data/ragas_scores.json")
    print("  These are your resume metrics.\n")


if __name__ == "__main__":
    asyncio.run(main())