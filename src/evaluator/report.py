import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.models.schemas import EvalReport, EvalResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "eval_report.json"

WEAK_THRESHOLD = 0.70


def generate_report(results: list[EvalResult]) -> EvalReport:
    """Compute averages, identify weak spots, and build an EvalReport."""
    n = len(results)
    if n == 0:
        return EvalReport(results=[])

    avg_faith = sum(r.faithfulness for r in results) / n
    avg_relevance = sum(r.answer_relevance for r in results) / n
    avg_precision = sum(r.context_precision for r in results) / n
    avg_recall = sum(r.context_recall for r in results) / n

    weak_spots = []
    if avg_faith < WEAK_THRESHOLD:
        weak_spots.append(f"Faithfulness ({avg_faith:.2f}) below {WEAK_THRESHOLD}")
    if avg_relevance < WEAK_THRESHOLD:
        weak_spots.append(f"Answer Relevance ({avg_relevance:.2f}) below {WEAK_THRESHOLD}")
    if avg_precision < WEAK_THRESHOLD:
        weak_spots.append(f"Context Precision ({avg_precision:.2f}) below {WEAK_THRESHOLD}")
    if avg_recall < WEAK_THRESHOLD:
        weak_spots.append(f"Context Recall ({avg_recall:.2f}) below {WEAK_THRESHOLD}")

    return EvalReport(
        results=results,
        avg_faithfulness=round(avg_faith, 4),
        avg_answer_relevance=round(avg_relevance, 4),
        avg_context_precision=round(avg_precision, 4),
        avg_context_recall=round(avg_recall, 4),
        weak_spots=weak_spots,
    )


def print_report(report: EvalReport) -> None:
    """Print a rich table to the terminal with per-query scores and averages."""
    console = Console()

    table = Table(title="RAG Evaluation Report", show_lines=True)
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Faithful", justify="center")
    table.add_column("Relevance", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")

    for result in report.results:
        query_display = result.query[:40] + "..." if len(result.query) > 40 else result.query
        table.add_row(
            query_display,
            _format_score(result.faithfulness),
            _format_score(result.answer_relevance),
            _format_score(result.context_precision),
            _format_score(result.context_recall),
        )

    # Averages row
    table.add_row(
        "[bold]AVERAGE[/bold]",
        _format_score(report.avg_faithfulness, bold=True),
        _format_score(report.avg_answer_relevance, bold=True),
        _format_score(report.avg_context_precision, bold=True),
        _format_score(report.avg_context_recall, bold=True),
    )

    console.print()
    console.print(table)

    if report.weak_spots:
        console.print()
        console.print("[bold red]Weak Spots Identified:[/bold red]")
        for spot in report.weak_spots:
            console.print(f"  - {spot}", style="red")
    else:
        console.print()
        console.print("[bold green]All metrics above threshold ({:.2f}).[/bold green]".format(WEAK_THRESHOLD))

    console.print()


def _format_score(score: float, bold: bool = False) -> str:
    """Format a score with color coding based on threshold."""
    formatted = f"{score:.2f}"
    if score < WEAK_THRESHOLD:
        styled = f"[red]{formatted}[/red]"
    else:
        styled = f"[green]{formatted}[/green]"
    if bold:
        styled = f"[bold]{styled}[/bold]"
    return styled


def save_report(report: EvalReport) -> None:
    """Save the full report to eval_report.json."""
    data = report.model_dump()
    with open(REPORT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Full report saved to {REPORT_PATH}")
