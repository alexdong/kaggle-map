"""Caching system for embeddings to accelerate hyperparameter optimization."""

import hashlib
from pathlib import Path

import numpy as np
from loguru import logger

from kaggle_map.core.models import EmbeddingModel, EmbeddingStrategy, EvaluationRow
from kaggle_map.embeddings import encode


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
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_embeddings(
    cache_path: Path,
    embeddings: np.ndarray,
    question_ids: np.ndarray,
    predictions: np.ndarray,
    mc_answers: np.ndarray,
    dataset_path: Path,
    model: EmbeddingModel,
    strategy: EmbeddingStrategy
) -> None:
    """Save embeddings and metadata to cache file."""
    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_hash": _compute_dataset_hash(dataset_path),
        "model": model.value,
        "strategy": strategy.value,
        "embedding_dim": embeddings.shape[1],
        "n_samples": embeddings.shape[0],
    }

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        question_ids=question_ids,
        predictions=predictions,
        mc_answers=mc_answers,
        **metadata
    )

    logger.info(f"Cached embeddings to {cache_path} ({embeddings.shape[0]} samples, {embeddings.shape[1]} dims)")


def load_embeddings(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict] | None:
    """Load embeddings from cache file if valid."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path)
        metadata = {
            "dataset_path": str(data["dataset_path"]),
            "dataset_hash": str(data["dataset_hash"]),
            "model": str(data["model"]),
            "strategy": str(data["strategy"]),
            "embedding_dim": int(data["embedding_dim"]),
            "n_samples": int(data["n_samples"]),
        }

        embeddings = data["embeddings"]
        question_ids = data["question_ids"]
        predictions = data["predictions"]
        mc_answers = data["mc_answers"]

        logger.info(f"Loaded cached embeddings from {cache_path} ({metadata['n_samples']} samples, {metadata['embedding_dim']} dims)")
        return embeddings, question_ids, predictions, mc_answers, metadata

    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def validate_cache(metadata: dict, dataset_path: Path, model: EmbeddingModel, strategy: EmbeddingStrategy) -> bool:
    """Validate that cached data matches current configuration."""
    current_hash = _compute_dataset_hash(dataset_path)

    is_valid = (
        metadata["dataset_path"] == str(dataset_path) and
        metadata["dataset_hash"] == current_hash and
        metadata["model"] == model.value and
        metadata["strategy"] == strategy.value
    )

    if not is_valid:
        logger.debug(f"Cache validation failed - dataset_hash: {metadata['dataset_hash']} vs {current_hash}")

    return is_valid


def get_or_compute_embeddings(
    eval_rows: list[EvaluationRow],
    metadata_tuples: list[tuple],
    dataset_path: Path,
    model: EmbeddingModel,
    strategy: EmbeddingStrategy
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
        embeddings,
        question_ids,
        predictions,
        mc_answers,
        dataset_path,
        model,
        strategy
    )

    return embeddings, question_ids, predictions, mc_answers


def precompute_all_embeddings(
    eval_rows: list[EvaluationRow],
    metadata_tuples: list[tuple],
    dataset_path: Path
) -> None:
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
            cached.append({
                "file": cache_file.name,
                "size_mb": cache_file.stat().st_size / (1024 * 1024),
                **metadata
            })

    return cached
