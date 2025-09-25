"""Command-line interface for managing embedding cache."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kaggle_map.core.models import EvaluationRow
from kaggle_map.dataloader.dataset import extract_correct_answers, load_training_data
from kaggle_map.mlp.embedding_cache import (
    clear_cache,
    list_cached_embeddings,
    precompute_all_embeddings,
)

console = Console()


def cmd_list() -> None:
    """List all cached embeddings."""
    cached = list_cached_embeddings()

    if not cached:
        console.print("[yellow]No cached embeddings found[/yellow]")
        return

    table = Table(title="Cached Embeddings")
    table.add_column("File", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("Model", style="blue")
    table.add_column("Strategy", style="magenta")
    table.add_column("Samples", justify="right")
    table.add_column("Dims", justify="right")
    table.add_column("Size (MB)", justify="right")

    for entry in cached:
        table.add_row(
            entry["file"],
            Path(entry["dataset_path"]).name,
            entry["model"],
            entry["strategy"],
            str(entry["n_samples"]),
            str(entry["embedding_dim"]),
            f"{entry['size_mb']:.1f}",
        )

    console.print(table)


def cmd_clear() -> None:
    """Clear all cached embeddings."""
    console.print("[yellow]Clearing all cached embeddings...[/yellow]")
    clear_cache()
    console.print("[green]Cache cleared successfully[/green]")


def cmd_precompute(dataset_path: str | Path) -> None:
    """Precompute all embedding combinations for a dataset."""
    path = Path(dataset_path)

    if not path.exists():
        console.print(f"[red]Dataset not found: {path}[/red]")
        return

    console.print(f"[yellow]Loading dataset: {path}[/yellow]")
    training_data = load_training_data(path)
    correct_answers = extract_correct_answers(training_data)

    console.print(f"[yellow]Preparing {len(training_data)} rows for embedding...[/yellow]")

    # Create evaluation rows and metadata
    eval_rows = []
    metadata_tuples = []
    for row in training_data:
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
            correct_answer=correct_answers.get(row.question_id, ""),
        )
        eval_rows.append(eval_row)
        metadata_tuples.append((row.question_id, str(row.prediction), row.mc_answer))

    console.print("[yellow]Precomputing embeddings for all model/strategy combinations...[/yellow]")
    console.print("This will compute 4 combinations:")
    console.print("  • QWEN + GOAL_DRIVEN (8192 dims)")
    console.print("  • QWEN + DOUBLE_BLIND (16384 dims)")
    console.print("  • GEMMA + GOAL_DRIVEN (768 dims)")
    console.print("  • GEMMA + DOUBLE_BLIND (1536 dims)")

    precompute_all_embeddings(eval_rows, metadata_tuples, path)

    console.print("[green]✓ All embeddings precomputed successfully[/green]")


def main() -> None:
    """Main entry point for cache CLI."""
    parser = argparse.ArgumentParser(
        description="Manage embedding cache for MLP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all cached embeddings
  python -m kaggle_map.mlp.cache_cli list

  # Clear all cached embeddings
  python -m kaggle_map.mlp.cache_cli clear

  # Precompute all embeddings for a dataset
  python -m kaggle_map.mlp.cache_cli precompute datasets/train.csv
  python -m kaggle_map.mlp.cache_cli precompute datasets/synth_balanced_30000_total.csv
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List all cached embeddings")

    # Clear command
    subparsers.add_parser("clear", help="Clear all cached embeddings")

    # Precompute command
    precompute_parser = subparsers.add_parser("precompute", help="Precompute all embedding combinations for a dataset")
    precompute_parser.add_argument("dataset", type=str, help="Path to the dataset CSV file")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "clear":
        cmd_clear()
    elif args.command == "precompute":
        cmd_precompute(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
