"""Embedding cache helpers for MLP training."""

from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger

from kaggle_map.core.models import (
    EmbeddingModel,
    EmbeddingStrategy,
    default_mlp_training_config,
)
from kaggle_map.dataloader.dataset import MAPDataset
from kaggle_map.embeddings import encode

DEFAULT_DATASET_PATH = Path("datasets/train.csv")
DEFAULT_CLI_MODEL = EmbeddingModel.QWEN
DEFAULT_CLI_STRATEGY = EmbeddingStrategy.DOUBLE_BLIND


def _get_cache_path(dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> Path:
    dataset_name = dataset_path.stem
    cache_key = f"{dataset_name}_{model.value}_{strategy.value}"

    cache_dir = Path(".cache/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.npz"


def get_or_compute_embeddings(
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
    dataset_path: Path,
) -> torch.Tensor:
    cache_path = _get_cache_path(dataset_path, model, strategy)
    if cache_path.exists():
        return torch.load(cache_path)

    dataset_path = DEFAULT_DATASET_PATH
    assert dataset_path.exists(), f"Training dataset missing: {dataset_path}"

    config = default_mlp_training_config()
    dataset = MAPDataset(csv_path=dataset_path, config=config)

    eval_rows = dataset.evaluation_rows()
    embeddings = encode(model, strategy, eval_rows)
    cache_path = _get_cache_path(dataset_path, model, strategy)
    torch.save(embeddings, cache_path)
    logger.info(f"Saved {embeddings.shape} embeddings to cache: {cache_path}")
    return embeddings


def preview_cached_embedding(model: EmbeddingModel, strategy: EmbeddingStrategy, row_id: int) -> None:
    """Print a cached embedding summary for ``row_id``."""
    dataset_path = DEFAULT_DATASET_PATH
    assert dataset_path.exists(), f"Training dataset missing: {dataset_path}"

    cache_path = _get_cache_path(dataset_path, model, strategy)
    embeddings = torch.load(cache_path)

    # To find the precomputed embedding for a given `row_id`, we need to
    # locate its index in the `Sequence[RowId]` and then use it to look up
    # the corresponding row in the `Tensor` of embeddings.
    config = default_mlp_training_config()
    dataset = MAPDataset(csv_path=dataset_path, config=config)
    eval_rows = dataset.evaluation_rows()
    assert embeddings.shape[0] == len(eval_rows), "Cached embeddings size mismatch; rebuild cache"
    row_index = next((idx for idx, row in enumerate(eval_rows) if row.row_id == row_id), -1)
    assert row_index >= 0, f"row_id {row_id} not found in training dataset"

    embedding_tensor = embeddings[row_index]
    assert isinstance(embedding_tensor, torch.Tensor) or torch.is_tensor(embedding_tensor)

    preview = embedding_tensor[: min(8, embedding_tensor.shape[0])].tolist()
    print(f"row_id={row_id}")
    print(f"embedding_dim={embedding_tensor.shape[0]} preview={preview}")


if __name__ == "__main__":
    import click

    MODEL_CHOICES = [model.value for model in EmbeddingModel]
    STRATEGY_CHOICES = [strategy.value for strategy in EmbeddingStrategy]

    def _cache_options(command: Callable[..., None]) -> Callable[..., None]:
        option_defs = [
            click.option(
                "--model",
                type=click.Choice(MODEL_CHOICES),
                default=DEFAULT_CLI_MODEL.value,
                show_default=True,
            ),
            click.option(
                "--strategy",
                type=click.Choice(STRATEGY_CHOICES),
                default=DEFAULT_CLI_STRATEGY.value,
                show_default=True,
            ),
        ]

        for option in reversed(option_defs):
            command = option(command)
        return command

    @click.group(help="Build or inspect the canonical embedding cache.")
    def cli() -> None:  # pragma: no cover - CLI utility
        logger.info("Embedding cache CLI ready")

    @cli.command()
    @_cache_options
    def build(model: str, strategy: str) -> None:  # pragma: no cover - CLI utility
        # Remove the old cache if it exists
        cache_path = _get_cache_path(
            dataset_path=DEFAULT_DATASET_PATH,
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
        )
        if cache_path.exists():  # pragma: no cover - CLI utility
            cache_path.unlink()
            logger.info(f"Removed old cache: {cache_path}")

        embeddings = get_or_compute_embeddings(
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
            dataset_path=DEFAULT_DATASET_PATH,
        )
        logger.info(f"Computed embeddings shape: {embeddings.shape}")
        click.echo(f"cache: {cache_path}")

    @cli.command()
    @_cache_options
    @click.option("--row-id", type=int, required=True)
    def preview(model: str, strategy: str, row_id: int) -> None:  # pragma: no cover - CLI utility
        preview_cached_embedding(
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
            row_id=row_id,
        )

    cli()
