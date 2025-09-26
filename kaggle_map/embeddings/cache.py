"""Embedding cache helpers for MLP training."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from kaggle_map.core.models import (
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    default_mlp_training_config,
)
from kaggle_map.dataloader.dataset import MAPDataset
from kaggle_map.embeddings import encode

DEFAULT_DATASET_PATH = Path("datasets/train.csv")
EMBEDDING_RANK = 2


@dataclass(frozen=True)
class CacheSaveRequest:
    embeddings: torch.Tensor
    question_ids: np.ndarray
    predictions: np.ndarray
    mc_answers: np.ndarray
    dataset_path: Path
    model: EmbeddingModel
    strategy: EmbeddingStrategy


def _get_cache_dir() -> Path:
    cache_dir = Path(".cache/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _generate_cache_key(dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> str:
    dataset_name = dataset_path.stem
    return f"{dataset_name}_{model.value}_{strategy.value}"


def _get_cache_path(dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> Path:
    cache_key = _generate_cache_key(dataset_path, model, strategy)
    return _get_cache_dir() / f"{cache_key}.npz"


def _compute_dataset_hash(dataset_path: Path) -> str:
    if not dataset_path.exists():
        return ""

    hasher = hashlib.md5()
    with dataset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_embeddings(cache_path: Path, request: CacheSaveRequest) -> None:
    embeddings_cpu = request.embeddings.detach().cpu().to(dtype=torch.float32)
    assert embeddings_cpu.ndim == EMBEDDING_RANK, "Embeddings tensor must be 2D"

    dataset_hash = _compute_dataset_hash(request.dataset_path)
    embedding_dim = int(embeddings_cpu.shape[1]) if embeddings_cpu.ndim == EMBEDDING_RANK else 0
    n_samples = int(embeddings_cpu.shape[0]) if embeddings_cpu.ndim >= 1 else 0

    payload = {
        "embeddings": embeddings_cpu,
        "question_ids": request.question_ids,
        "predictions": request.predictions,
        "mc_answers": request.mc_answers,
        "metadata": {
            "dataset_path": str(request.dataset_path),
            "dataset_hash": dataset_hash,
            "model": request.model.value,
            "strategy": request.strategy.value,
            "embedding_dim": embedding_dim,
            "n_samples": n_samples,
        },
    }

    torch.save(payload, cache_path)
    logger.info(
        "Cached embeddings to %s (%s samples, %s dims)",
        cache_path,
        n_samples,
        embedding_dim,
    )


def load_embeddings(cache_path: Path) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict]:
    assert cache_path.exists(), f"Cache file missing: {cache_path}"

    payload = torch.load(cache_path, map_location="cpu")
    assert "embeddings" in payload, f"Missing embeddings tensor in cache: {cache_path}"
    assert "metadata" in payload, f"Missing metadata in cache: {cache_path}"

    embeddings = payload["embeddings"]
    metadata = payload["metadata"]
    question_ids = payload["question_ids"]
    predictions = payload["predictions"]
    mc_answers = payload["mc_answers"]

    logger.info(
        "Loaded cached embeddings from %s (%s samples, %s dims)",
        cache_path,
        metadata["n_samples"],
        metadata["embedding_dim"],
    )
    return embeddings, question_ids, predictions, mc_answers, metadata


def get_or_compute_embeddings(
    eval_rows: Sequence[EvaluationRow],
    metadata_tuples: Sequence[tuple[int, str, str]],
    dataset_path: Path,
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _get_cache_path(dataset_path, model, strategy)

    if cache_path.exists():
        embeddings, question_ids, predictions, mc_answers, metadata = load_embeddings(cache_path)
        assert metadata["dataset_path"] == str(dataset_path), (
            f"Cache dataset path mismatch: {metadata['dataset_path']} vs {dataset_path}"
        )
        assert metadata["model"] == model.value, f"Cache model mismatch: {metadata['model']} vs {model.value}"
        assert metadata["strategy"] == strategy.value, (
            f"Cache strategy mismatch: {metadata['strategy']} vs {strategy.value}"
        )

        current_hash = _compute_dataset_hash(dataset_path)
        if metadata["dataset_hash"] == current_hash:
            return embeddings, question_ids, predictions, mc_answers

        logger.info(
            "Cache hash mismatch for %s (stored=%s, current=%s); recomputing",
            cache_path,
            metadata["dataset_hash"],
            current_hash,
        )

    logger.info(
        "Computing embeddings for %s rows with %s/%s",
        len(eval_rows),
        model.value,
        strategy.value,
    )
    embeddings_tensor = encode(eval_rows, strategy, model)
    embeddings_cpu = embeddings_tensor.detach().cpu()

    if not (metadata_unpacked := list(zip(*metadata_tuples, strict=True))):
        msg = "No metadata to process"
        raise ValueError(msg)

    question_ids, predictions, mc_answers = map(np.array, metadata_unpacked)

    save_embeddings(
        cache_path,
        CacheSaveRequest(
            embeddings=embeddings_cpu,
            question_ids=question_ids,
            predictions=predictions,
            mc_answers=mc_answers,
            dataset_path=dataset_path,
            model=model,
            strategy=strategy,
        ),
    )

    return embeddings_cpu, question_ids, predictions, mc_answers


def get_or_compute_embeddings_tensor(  # noqa: PLR0913 - keep explicit signature for clarity
    eval_rows: Sequence[EvaluationRow],
    metadata_tuples: Sequence[tuple[int, str, str]],
    dataset_path: Path,
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    embeddings_tensor, question_ids, predictions, mc_answers = get_or_compute_embeddings(
        eval_rows,
        metadata_tuples,
        dataset_path,
        model,
        strategy,
    )
    embeddings_tensor = embeddings_tensor.to(dtype=dtype)
    assert not embeddings_tensor.is_cuda, "Cached embeddings tensor must reside on CPU"
    return embeddings_tensor, question_ids, predictions, mc_answers


def precompute_all_embeddings(
    eval_rows: Sequence[EvaluationRow],
    metadata_tuples: Sequence[tuple[int, str, str]],
    dataset_path: Path,
) -> None:
    models = [EmbeddingModel.QWEN, EmbeddingModel.GEMMA]
    strategies = [EmbeddingStrategy.DOUBLE_BLIND, EmbeddingStrategy.GOAL_DRIVEN]

    for model in models:
        for strategy in strategies:
            cache_path = _get_cache_path(dataset_path, model, strategy)
            if cache_path.exists():
                logger.info("Skipping %s/%s - already cached", model.value, strategy.value)
                continue
            logger.info("Precomputing %s/%s", model.value, strategy.value)
            get_or_compute_embeddings(eval_rows, metadata_tuples, dataset_path, model, strategy)


def clear_cache() -> None:
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        logger.info("No cache to clear")
        return

    for cache_file in cache_dir.glob("*.npz"):
        cache_file.unlink()
        logger.info("Removed %s", cache_file)

    logger.info("Cache cleared")


def list_cached_embeddings() -> list[dict]:
    cache_dir = _get_cache_dir()
    cached: list[dict] = []

    for cache_file in cache_dir.glob("*.npz"):
        _, _, _, _, metadata = load_embeddings(cache_file)
        cached.append(
            {
                "file": cache_file.name,
                "size_mb": cache_file.stat().st_size / (1024 * 1024),
                **metadata,
            }
        )

    return cached


def build_embedding_cache(*, model: EmbeddingModel, strategy: EmbeddingStrategy) -> Path:
    """Compute and cache embeddings for the canonical training set."""
    dataset_path = DEFAULT_DATASET_PATH
    assert dataset_path.exists(), f"Training dataset missing: {dataset_path}"

    config = default_mlp_training_config()
    dataset = MAPDataset(csv_path=dataset_path, config=config)

    pairs = dataset.evaluation_pairs()
    eval_rows = [row for row, _ in pairs]
    assert eval_rows, "Training dataset must yield evaluation rows"

    metadata: list[tuple[int, str, str]] = [
        (row.question_id, str(prediction), row.mc_answer) for row, prediction in pairs
    ]

    get_or_compute_embeddings(eval_rows, metadata, dataset_path, model, strategy)

    cache_path = _get_cache_path(dataset_path, model, strategy)
    logger.info("Cache ready at %s", cache_path)
    return cache_path


def preview_cached_embedding(*, model: EmbeddingModel, strategy: EmbeddingStrategy, row_id: int) -> None:
    """Print a cached embedding summary for ``row_id``."""
    dataset_path = DEFAULT_DATASET_PATH
    assert dataset_path.exists(), f"Training dataset missing: {dataset_path}"

    cache_path = _get_cache_path(dataset_path, model, strategy)
    embeddings, question_ids, predictions, mc_answers, _ = load_embeddings(cache_path)

    config = default_mlp_training_config()
    dataset = MAPDataset(csv_path=dataset_path, config=config)
    pairs = dataset.evaluation_pairs()
    eval_rows = [row for row, _ in pairs]

    assert embeddings.shape[0] == len(eval_rows), "Cached embeddings size mismatch; rebuild cache"

    row_index = next((idx for idx, row in enumerate(eval_rows) if row.row_id == row_id), -1)
    assert row_index >= 0, f"row_id {row_id} not found in training dataset"

    embedding_tensor = embeddings[row_index]
    assert isinstance(embedding_tensor, torch.Tensor) or torch.is_tensor(embedding_tensor)

    question_id = int(question_ids[row_index])
    prediction = str(predictions[row_index])
    mc_answer = str(mc_answers[row_index])

    preview = embedding_tensor[: min(8, embedding_tensor.shape[0])].tolist()
    print(f"row_id={row_id} question_id={question_id} mc_answer={mc_answer} prediction={prediction}")
    print(f"embedding_dim={embedding_tensor.shape[0]} preview={preview}")


if __name__ == "__main__":
    import click

    MODEL_CHOICES = [model.value for model in EmbeddingModel]
    STRATEGY_CHOICES = [strategy.value for strategy in EmbeddingStrategy]

    @click.group(help="Build or inspect the canonical embedding cache.")
    def cli() -> None:  # pragma: no cover - CLI utility
        logger.info("Embedding cache CLI ready")

    @cli.command()
    @click.option(
        "--model",
        type=click.Choice(MODEL_CHOICES),
        default=EmbeddingModel.GEMMA.value,
        show_default=True,
    )
    @click.option(
        "--strategy",
        type=click.Choice(STRATEGY_CHOICES),
        default=EmbeddingStrategy.GOAL_DRIVEN.value,
        show_default=True,
    )
    def build(model: str, strategy: str) -> None:  # pragma: no cover - CLI utility
        cache_path = build_embedding_cache(
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
        )
        click.echo(f"cache: {cache_path}")

    @cli.command()
    @click.option(
        "--model",
        type=click.Choice(MODEL_CHOICES),
        default=EmbeddingModel.GEMMA.value,
        show_default=True,
    )
    @click.option(
        "--strategy",
        type=click.Choice(STRATEGY_CHOICES),
        default=EmbeddingStrategy.GOAL_DRIVEN.value,
        show_default=True,
    )
    @click.option("--row-id", type=int, required=True)
    def preview(model: str, strategy: str, row_id: int) -> None:  # pragma: no cover - CLI utility
        preview_cached_embedding(
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
            row_id=row_id,
        )

    cli()
