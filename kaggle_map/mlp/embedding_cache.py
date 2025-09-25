"""Caching system for embeddings to accelerate hyperparameter optimization.

Example::

    $ uv run -m kaggle_map.mlp.embedding_cache -h
    Usage: python -m kaggle_map.mlp.embedding_cache [OPTIONS]

      Demonstrate cache creation, inspection, and reuse timings.

    Options:
      --dataset-path PATH    Training CSV used to build embeddings. [default:
                             datasets/33474_tiny_train.csv]
      --model [qwen|gemma]   Embedding model to demonstrate. [default: gemma]
      --strategy [double_blind|goal_driven]
                             Embedding strategy to use. [default: goal_driven]
      --limit INTEGER RANGE  Number of rows to sample from the dataset for the
                             demo. [default: 32]
      --clear-cache / --keep-cache
                             Whether to remove any existing cache file before
                             running the demo. [default: clear-cache]
      --help                 Show this message and exit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from kaggle_map.core.models import EmbeddingModel, EmbeddingStrategy, EvaluationRow, TrainingRow
from kaggle_map.embeddings import encode


@dataclass(frozen=True)
class CacheSaveRequest:
    embeddings: np.ndarray
    question_ids: np.ndarray
    predictions: np.ndarray
    mc_answers: np.ndarray
    dataset_path: Path
    model: EmbeddingModel
    strategy: EmbeddingStrategy


def _get_cache_dir() -> Path:
    """Get the cache directory for embeddings."""
    cache_dir = Path(".cache/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _generate_cache_key(dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> str:
    """Generate a deterministic cache key based on dataset and embedding configuration."""
    dataset_name = dataset_path.stem
    return f"{dataset_name}_{model.value}_{strategy.value}"


def _get_cache_path(dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> Path:
    """Get the full path for a cached embedding file."""
    cache_key = _generate_cache_key(dataset_path, model, strategy)
    return _get_cache_dir() / f"{cache_key}.npz"


def _compute_dataset_hash(dataset_path: Path) -> str:
    """Compute a hash of the dataset file for validation."""
    if not dataset_path.exists():
        return ""

    hasher = hashlib.md5()
    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_embeddings(cache_path: Path, request: CacheSaveRequest) -> None:
    """Save embeddings and metadata to cache file."""
    dataset_hash = _compute_dataset_hash(request.dataset_path)
    embedding_dim = request.embeddings.shape[1]
    n_samples = request.embeddings.shape[0]

    np.savez_compressed(
        cache_path,
        embeddings=request.embeddings,
        question_ids=request.question_ids,
        predictions=request.predictions,
        mc_answers=request.mc_answers,
        dataset_path=np.array(str(request.dataset_path)),
        dataset_hash=np.array(dataset_hash),
        model=np.array(request.model.value),
        strategy=np.array(request.strategy.value),
        embedding_dim=np.array(embedding_dim, dtype=np.int32),
        n_samples=np.array(n_samples, dtype=np.int32),
    )

    logger.info(
        "Cached embeddings to %s (%s samples, %s dims)",
        cache_path,
        n_samples,
        embedding_dim,
    )


def load_embeddings(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict] | None:
    """Load embeddings from cache file if valid."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path)
        metadata = {
            "dataset_path": str(data["dataset_path"].item()),
            "dataset_hash": str(data["dataset_hash"].item()),
            "model": str(data["model"].item()),
            "strategy": str(data["strategy"].item()),
            "embedding_dim": int(data["embedding_dim"].item()),
            "n_samples": int(data["n_samples"].item()),
        }

        embeddings = data["embeddings"]
        question_ids = data["question_ids"]
        predictions = data["predictions"]
        mc_answers = data["mc_answers"]

        logger.info(
            "Loaded cached embeddings from %s (%s samples, %s dims)",
            cache_path,
            metadata["n_samples"],
            metadata["embedding_dim"],
        )
        return embeddings, question_ids, predictions, mc_answers, metadata

    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def validate_cache(metadata: dict, dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> bool:
    """Validate that cached data matches current configuration."""
    current_hash = _compute_dataset_hash(dataset_path)

    is_valid = (
        metadata["dataset_path"] == str(dataset_path)
        and metadata["dataset_hash"] == current_hash
        and metadata["model"] == model.value
        and metadata["strategy"] == strategy.value
    )

    if not is_valid:
        logger.debug(f"Cache validation failed - dataset_hash: {metadata['dataset_hash']} vs {current_hash}")

    return is_valid


def get_or_compute_embeddings(
    eval_rows: list[EvaluationRow],
    metadata_tuples: list[tuple],
    dataset_path: Path,
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get embeddings from cache or compute them if not cached."""
    cache_path = _get_cache_path(dataset_path, model, strategy)

    # Try to load from cache
    cached_result = load_embeddings(cache_path)
    if cached_result is not None:
        embeddings, question_ids, predictions, mc_answers, metadata = cached_result
        if validate_cache(metadata, dataset_path, model, strategy):
            return embeddings, question_ids, predictions, mc_answers
        logger.info("Cache validation failed, recomputing embeddings")

    # Compute embeddings
    logger.info(f"Computing embeddings for {len(eval_rows)} rows with {model.value}/{strategy.value}")
    embeddings_tensor = encode(eval_rows, strategy, model)
    embeddings = embeddings_tensor.cpu().numpy()

    # Unpack metadata
    if not (metadata_unpacked := list(zip(*metadata_tuples, strict=True))):
        msg = "No metadata to process"
        raise ValueError(msg)
    question_ids, predictions, mc_answers = map(np.array, metadata_unpacked)

    # Save to cache
    save_embeddings(
        cache_path,
        CacheSaveRequest(
            embeddings=embeddings,
            question_ids=question_ids,
            predictions=predictions,
            mc_answers=mc_answers,
            dataset_path=dataset_path,
            model=model,
            strategy=strategy,
        ),
    )

    return embeddings, question_ids, predictions, mc_answers


def precompute_all_embeddings(eval_rows: list[EvaluationRow], metadata_tuples: list[tuple], dataset_path: Path) -> None:
    """Precompute embeddings for all model/strategy combinations."""
    models = [EmbeddingModel.QWEN, EmbeddingModel.GEMMA]
    strategies = [EmbeddingStrategy.DOUBLE_BLIND, EmbeddingStrategy.GOAL_DRIVEN]

    for model in models:
        for strategy in strategies:
            cache_path = _get_cache_path(dataset_path, model, strategy)
            if cache_path.exists():
                logger.info(f"Skipping {model.value}/{strategy.value} - already cached")
                continue

            logger.info(f"Precomputing {model.value}/{strategy.value}")
            get_or_compute_embeddings(eval_rows, metadata_tuples, dataset_path, model, strategy)


def clear_cache() -> None:
    """Clear all cached embeddings."""
    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.npz"):
            cache_file.unlink()
            logger.info(f"Removed {cache_file}")
        logger.info("Cache cleared")
    else:
        logger.info("No cache to clear")


def list_cached_embeddings() -> list[dict]:
    """List all cached embeddings with their metadata."""
    cache_dir = _get_cache_dir()
    cached = []

    for cache_file in cache_dir.glob("*.npz"):
        result = load_embeddings(cache_file)
        if result is not None:
            _, _, _, _, metadata = result
            cached.append({"file": cache_file.name, "size_mb": cache_file.stat().st_size / (1024 * 1024), **metadata})

    return cached


if __name__ == "__main__":
    import time

    import click
    from rich.console import Console
    from rich.table import Table

    from kaggle_map.dataloader.dataset import extract_correct_answers, load_training_data
    from kaggle_map.utils.logger_config import configure_logger

    @dataclass(frozen=True)
    class DemoArguments:
        dataset_path: Path
        model: EmbeddingModel
        strategy: EmbeddingStrategy
        limit: int
        clear_cache: bool

    def _select_subset(training_rows: list[TrainingRow], limit: int) -> list[TrainingRow]:
        subset = training_rows[: min(len(training_rows), limit)]
        if not subset:
            msg = "Dataset slice for embedding demo cannot be empty"
            raise ValueError(msg)
        return subset

    def _derive_correct_answers(training_rows: list[TrainingRow], subset: list[TrainingRow]) -> dict[int, str]:
        try:
            return extract_correct_answers(training_rows)
        except AssertionError as exc:  # pragma: no cover - defensive branch
            logger.warning(
                "Falling back to student's multiple-choice answer as the correct answer"
                " because the dataset contains no True_Correct rows."
            )
            fallback = {row.question_id: row.mc_answer for row in subset}
            if not fallback:
                msg = "Failed to infer fallback correct answers from subset"
                raise RuntimeError(msg) from exc
            return fallback

    def _build_embedding_inputs(
        subset: list[TrainingRow],
        correct_answers: dict[int, str],
    ) -> tuple[list[EvaluationRow], list[tuple[int, str, str]]]:
        eval_rows: list[EvaluationRow] = []
        metadata: list[tuple[int, str, str]] = []
        for row in subset:
            eval_rows.append(
                EvaluationRow(
                    row_id=row.row_id,
                    question_id=row.question_id,
                    question_text=row.question_text,
                    mc_answer=row.mc_answer,
                    student_explanation=row.student_explanation,
                    correct_answer=correct_answers.get(row.question_id, ""),
                )
            )
            metadata.append((row.question_id, str(row.prediction), row.mc_answer))
        return eval_rows, metadata

    def _time_request(
        eval_rows: list[EvaluationRow],
        metadata_tuples: list[tuple[int, str, str]],
        args: DemoArguments,
    ) -> float:
        start = time.perf_counter()
        get_or_compute_embeddings(
            eval_rows,
            metadata_tuples,
            args.dataset_path,
            args.model,
            args.strategy,
        )
        return time.perf_counter() - start

    def _render_cache_table(console: Console, cache_path: Path) -> None:
        cache_exists = cache_path.exists()
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_exists else 0.0
        table = Table(title="Cache Status", show_lines=False)
        table.add_column("Cache File")
        table.add_column("Exists")
        table.add_column("Size (MB)", justify="right")
        table.add_row(cache_path.name, str(cache_exists), f"{cache_size_mb:.2f}")
        console.print(table)

    def _render_timing_table(
        console: Console,
        first_duration: float,
        second_duration: float,
        sample_count: int,
    ) -> None:
        assert sample_count > 0, "Sample count must be positive"
        per_text_initial = (first_duration / sample_count) * 1000
        per_text_cached = (second_duration / sample_count) * 1000

        table = Table(title="Embedding Cache Timings")
        table.add_column("Request")
        table.add_column("Total (s)", justify="right")
        table.add_column("Per Text (ms)", justify="right")
        table.add_row("Initial compute", f"{first_duration:.3f}", f"{per_text_initial:.1f}")
        table.add_row("Cached load", f"{second_duration:.3f}", f"{per_text_cached:.1f}")
        console.print(table)

    def _prepare_demo_arguments(
        dataset_path: Path,
        model: str,
        strategy: str,
        limit: int,
        *,
        clear_cache: bool,
    ) -> DemoArguments:
        return DemoArguments(
            dataset_path=dataset_path,
            model=EmbeddingModel(model),
            strategy=EmbeddingStrategy(strategy),
            limit=limit,
            clear_cache=clear_cache,
        )

    def _run_demo(args: DemoArguments) -> None:
        configure_logger(__name__, console_level="DEBUG")
        console = Console()

        training_rows = load_training_data(args.dataset_path)
        subset = _select_subset(training_rows, args.limit)
        correct_answers = _derive_correct_answers(training_rows, subset)
        eval_rows, metadata_tuples = _build_embedding_inputs(subset, correct_answers)

        cache_path = _get_cache_path(args.dataset_path, args.model, args.strategy)
        if args.clear_cache and cache_path.exists():
            cache_path.unlink()
            logger.info("Removed existing cache at {}", cache_path)

        first_duration = _time_request(eval_rows, metadata_tuples, args)
        _render_cache_table(console, cache_path)
        second_duration = _time_request(eval_rows, metadata_tuples, args)
        _render_timing_table(console, first_duration, second_duration, len(eval_rows))

    @click.command()
    @click.option(
        "--dataset-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("datasets/33474_tiny_train.csv"),
        show_default=True,
        help="Training CSV used to build embeddings.",
    )
    @click.option(
        "--model",
        type=click.Choice([model.value for model in EmbeddingModel]),
        default=EmbeddingModel.GEMMA.value,
        show_default=True,
        help="Embedding model to demonstrate.",
    )
    @click.option(
        "--strategy",
        type=click.Choice([strategy.value for strategy in EmbeddingStrategy]),
        default=EmbeddingStrategy.GOAL_DRIVEN.value,
        show_default=True,
        help="Embedding strategy to use.",
    )
    @click.option(
        "--limit",
        type=click.IntRange(1, None),
        default=32,
        show_default=True,
        help="Number of rows to sample from the dataset for the demo.",
    )
    @click.option(
        "--clear-cache/--keep-cache",
        default=True,
        show_default=True,
        help="Whether to remove any existing cache file before running the demo.",
    )
    def main(
        dataset_path: Path,
        model: str,
        strategy: str,
        limit: int,
        *,
        clear_cache: bool,
    ) -> None:
        """Demonstrate cache creation, inspection, and reuse timings."""

        args = _prepare_demo_arguments(
            dataset_path=dataset_path,
            model=model,
            strategy=strategy,
            limit=limit,
            clear_cache=clear_cache,
        )
        _run_demo(args)

    main()
