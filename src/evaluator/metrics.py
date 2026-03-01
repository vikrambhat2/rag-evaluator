from src.evaluator.judge import (
    ANSWER_RELEVANCE_SYSTEM_PROMPT,
    CONTEXT_PRECISION_SYSTEM_PROMPT,
    CONTEXT_RECALL_SYSTEM_PROMPT,
    FAITHFULNESS_SYSTEM_PROMPT,
    OllamaJudge,
)

_judge = OllamaJudge()


def faithfulness_score(answer: str, chunks: list[str]) -> float:
    """Score how faithful the answer is to the retrieved chunks."""
    context = "\n\n---\n\n".join(chunks)
    user_prompt = (
        f"Answer:\n{answer}\n\n"
        f"Context Chunks:\n{context}\n\n"
        "Rate the faithfulness of the answer to the context."
    )
    return _judge.score(FAITHFULNESS_SYSTEM_PROMPT, user_prompt)


def answer_relevance_score(query: str, answer: str) -> float:
    """Score how relevant the answer is to the query."""
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        "Rate how well the answer addresses the question."
    )
    return _judge.score(ANSWER_RELEVANCE_SYSTEM_PROMPT, user_prompt)


def context_precision_score(query: str, chunks: list[str]) -> float:
    """Score what fraction of retrieved chunks are relevant to the query."""
    context = "\n\n---\n\n".join(
        f"Chunk {i + 1}:\n{c}" for i, c in enumerate(chunks)
    )
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Retrieved Chunks:\n{context}\n\n"
        "Rate what proportion of the retrieved chunks are relevant to the query."
    )
    return _judge.score(CONTEXT_PRECISION_SYSTEM_PROMPT, user_prompt)


def context_recall_score(
    query: str, chunks: list[str], ground_truth: str
) -> float:
    """Score how much of the ground truth is covered by the retrieved chunks."""
    context = "\n\n---\n\n".join(chunks)
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Ground Truth Answer:\n{ground_truth}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Rate how much of the ground truth information is present in the retrieved context."
    )
    return _judge.score(CONTEXT_RECALL_SYSTEM_PROMPT, user_prompt)
