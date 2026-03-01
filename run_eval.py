"""Entry point: run RAG evaluation on the test set."""

import json
import logging
import sys
from pathlib import Path

from rich.console import Console

from src.evaluator.metrics import (
    answer_relevance_score,
    context_precision_score,
    context_recall_score,
    faithfulness_score,
)
from src.evaluator.report import generate_report, print_report, save_report
from src.models.schemas import EvalResult, TestCase
from src.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_SET_PATH = PROJECT_ROOT / "data" / "test_set.json"


def load_test_set() -> list[TestCase]:
    """Load test cases from data/test_set.json."""
    if not TEST_SET_PATH.exists():
        print(f"Test set not found at {TEST_SET_PATH}. Run ingest.py first.")
        sys.exit(1)

    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]


def main() -> None:
    console = Console()

    console.print("\n[bold]RAG Evaluation Pipeline[/bold]")
    console.print("=" * 60)

    # Load test set
    test_cases = load_test_set()
    console.print(f"Loaded {len(test_cases)} test cases from {TEST_SET_PATH}\n")

    # Initialize RAG pipeline
    pipeline = RAGPipeline()

    results: list[EvalResult] = []

    for i, tc in enumerate(test_cases, 1):
        console.print(f"[bold cyan][{i}/{len(test_cases)}][/bold cyan] Evaluating: {tc.query[:60]}...")

        # Run RAG pipeline
        console.print("  -> Retrieving and generating answer...")
        answer, chunks = pipeline.query(tc.query)

        # Run all 4 metrics
        console.print("  -> Scoring faithfulness...")
        faith = faithfulness_score(answer, chunks)

        console.print("  -> Scoring answer relevance...")
        relevance = answer_relevance_score(tc.query, answer)

        console.print("  -> Scoring context precision...")
        precision = context_precision_score(tc.query, chunks)

        console.print("  -> Scoring context recall...")
        recall = context_recall_score(tc.query, chunks, tc.ground_truth)

        result = EvalResult(
            query=tc.query,
            answer=answer,
            chunks=chunks,
            ground_truth=tc.ground_truth,
            faithfulness=faith,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
        )
        results.append(result)

        console.print(
            f"  -> Scores: faith={faith:.2f} rel={relevance:.2f} "
            f"prec={precision:.2f} recall={recall:.2f}\n"
        )

    # Generate and display report
    report = generate_report(results)
    print_report(report)
    save_report(report)


if __name__ == "__main__":
    main()
