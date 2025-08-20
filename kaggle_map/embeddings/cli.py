"""CLI for generating and saving embeddings from training data."""

from pathlib import Path

import click
import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.table import Table

from kaggle_map.dataset import parse_training_data
from kaggle_map.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.models import TrainingRow


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True, path_type=Path),
    default=Path("datasets/train_original.csv"),
    help="Path to input CSV file (default: datasets/train_original.csv)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets"),
    help="Output directory for embeddings file (default: datasets)",
)
@click.option(
    "--embedding-model",
    type=click.Choice(["mini-lm"], case_sensitive=False),
    default="mini-lm",
    help="Embedding model to use (default: mini-lm)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for processing embeddings (default: 100)",
)
def generate_embeddings(input_csv: Path, output_dir: Path, embedding_model: str, batch_size: int) -> None:
    """Generate embeddings for training data and save to numpy format.

    This tool reads the training CSV, generates embeddings for each row,
    and saves row_id, misconception, and embeddings to a .npz file for reuse.
    """
    console = Console()

    console.print(f"[bold blue]🚀 Generating Embeddings from {input_csv}[/bold blue]")

    # Display configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Input CSV", str(input_csv))
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Embedding Model", embedding_model)
    config_table.add_row("Batch Size", str(batch_size))
    console.print(config_table)

    # Load and parse training data
    with console.status("[bold green]Loading training data..."):
        training_data = parse_training_data(input_csv)
        logger.info(f"Loaded {len(training_data)} training rows")

    console.print(f"✅ [bold green]Loaded {len(training_data)} training rows[/bold green]")

    # Initialize embedding model
    with console.status("[bold green]Initializing embedding model..."):
        embedding_model_obj = EmbeddingModel.MINI_LM
        tokenizer = get_tokenizer(embedding_model_obj)
        logger.info(f"Initialized embedding model: {embedding_model_obj.model_id}")

    console.print(f"✅ [bold green]Initialized {embedding_model_obj.model_id}[/bold green]")

    # Generate embeddings
    console.print("[bold blue]Generating embeddings...[/bold blue]")
    row_ids, misconceptions, embeddings = _generate_embeddings_batch(training_data, tokenizer, batch_size, console)

    # Save to numpy format
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "train_embeddings.npz"

    with console.status(f"[bold green]Saving embeddings to {output_file}..."):
        np.savez_compressed(
            output_file,
            row_ids=np.array(row_ids),
            misconceptions=np.array(misconceptions),  # String array, no object dtype needed
            embeddings=np.array(embeddings),
        )
        logger.info(f"Saved embeddings to {output_file}")

    console.print(f"✅ [bold green]Embeddings saved to {output_file}[/bold green]")

    # Display summary
    _display_summary(console, output_file, len(training_data), embeddings[0].shape[0])


def _generate_embeddings_batch(
    training_data: list[TrainingRow],
    tokenizer: object,  # SentenceTransformer, but avoiding import
    batch_size: int,
    console: Console,
) -> tuple[list[int], list[str], list[np.ndarray]]:
    """Generate embeddings in batches with progress tracking."""
    row_ids = []
    misconceptions = []
    embeddings = []

    # Process in batches with progress bar
    for i in track(
        range(0, len(training_data), batch_size),
        description="Processing batches...",
        console=console,
    ):
        batch = training_data[i : i + batch_size]
        batch_texts = []
        batch_row_ids = []
        batch_misconceptions = []

        # Prepare batch
        for row in batch:
            text = repr(row)  # Uses TrainingRow.__repr__ format
            batch_texts.append(text)
            batch_row_ids.append(row.row_id)

            # Use "NA" for empty misconceptions as requested
            misconception_str = row.misconception if row.misconception is not None else "NA"
            batch_misconceptions.append(misconception_str)

        # Generate embeddings for batch
        batch_embeddings = tokenizer.encode(batch_texts)  # type: ignore[attr-defined]

        # Store results
        row_ids.extend(batch_row_ids)
        misconceptions.extend(batch_misconceptions)
        embeddings.extend(batch_embeddings)

    logger.info(f"Generated embeddings for {len(row_ids)} rows")
    return row_ids, misconceptions, embeddings


def _display_summary(console: Console, output_file: Path, num_rows: int, embedding_dim: int) -> None:
    """Display summary of embedding generation."""
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    summary_table = Table(title="Embedding Generation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Total Rows Processed", f"{num_rows:,}")
    summary_table.add_row("Embedding Dimensions", str(embedding_dim))
    summary_table.add_row("Output File", str(output_file))
    summary_table.add_row("File Size", f"{file_size_mb:.1f} MB")
    summary_table.add_row("Status", "[green]✅ Successfully generated[/green]")

    console.print(summary_table)

    console.print("\n[bold]Usage Instructions:[/bold]")
    console.print("Load the embeddings in your code with:")
    console.print("[dim]import numpy as np[/dim]")
    console.print(f"[dim]data = np.load('{output_file}')[/dim]")
    console.print("[dim]row_ids = data['row_ids'][/dim]")
    console.print("[dim]misconceptions = data['misconceptions'][/dim]")
    console.print("[dim]embeddings = data['embeddings'][/dim]")


if __name__ == "__main__":
    generate_embeddings()
