"""Tests for embedding cache functionality."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from kaggle_map.core.models import (
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    QuestionId,
)
from kaggle_map.mlp.embedding_cache import (
    _generate_cache_key,
    _get_cache_path,
    clear_cache,
    get_or_compute_embeddings,
    list_cached_embeddings,
    load_embeddings,
    precompute_all_embeddings,
    save_embeddings,
    validate_cache,
)


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Use a temporary directory for cache during tests."""
    cache_dir = tmp_path / ".cache" / "embeddings"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr("kaggle_map.mlp.embedding_cache._get_cache_dir", lambda: cache_dir)
    return cache_dir


@pytest.fixture
def tiny_eval_rows():
    """Create minimal evaluation rows for testing."""
    return [
        EvaluationRow(
            row_id=i,
            question_id=QuestionId(i // 2),
            question_text=f"Question {i // 2}",
            mc_answer=f"Answer {i % 3}",
            student_explanation=f"Explanation {i}",
            correct_answer=f"Correct {i // 2}",
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_encode():
    """Mock the encode function to return deterministic embeddings quickly."""
    def _mock_encode(rows, strategy, model):
        n_rows = len(rows) if isinstance(rows, list) else 1

        dim_map = {
            (EmbeddingModel.QWEN, EmbeddingStrategy.GOAL_DRIVEN): 8192,
            (EmbeddingModel.QWEN, EmbeddingStrategy.DOUBLE_BLIND): 16384,
            (EmbeddingModel.GEMMA, EmbeddingStrategy.GOAL_DRIVEN): 768,
            (EmbeddingModel.GEMMA, EmbeddingStrategy.DOUBLE_BLIND): 1536,
        }
        dim = dim_map[(model, strategy)]

        np.random.seed(42)
        return torch.tensor(np.random.randn(n_rows, dim).astype(np.float32))

    return _mock_encode


def test_cache_key_generation():
    """Test that cache keys are deterministic and unique."""
    key1 = _generate_cache_key(
        Path("datasets/train.csv"),
        EmbeddingModel.QWEN,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    key2 = _generate_cache_key(
        Path("datasets/train.csv"),
        EmbeddingModel.QWEN,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert key1 == key2
    assert key1 == "train_qwen_goal_driven"

    # Different dataset
    key3 = _generate_cache_key(
        Path("datasets/synth.csv"),
        EmbeddingModel.QWEN,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert key1 != key3

    # Different model
    key4 = _generate_cache_key(
        Path("datasets/train.csv"),
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert key1 != key4

    # Different strategy
    key5 = _generate_cache_key(
        Path("datasets/train.csv"),
        EmbeddingModel.QWEN,
        EmbeddingStrategy.DOUBLE_BLIND
    )
    assert key1 != key5


def test_cache_path_generation(temp_cache_dir):
    """Test that cache paths are correctly generated."""
    path = _get_cache_path(
        Path("datasets/train.csv"),
        EmbeddingModel.QWEN,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert path.parent == temp_cache_dir
    assert path.name == "train_qwen_goal_driven.npz"


def test_save_and_load_embeddings(temp_cache_dir, tmp_path):
    """Test saving and loading embeddings from cache."""
    # Create test data
    embeddings = np.random.randn(10, 768).astype(np.float32)
    question_ids = np.arange(10)
    predictions = np.arange(10)
    mc_answers = np.array([f"Answer{i}" for i in range(10)])

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")

    cache_path = temp_cache_dir / "test_cache.npz"

    # Save embeddings
    save_embeddings(
        cache_path,
        embeddings,
        question_ids,
        predictions,
        mc_answers,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )

    assert cache_path.exists()

    # Load embeddings
    result = load_embeddings(cache_path)
    assert result is not None

    loaded_embeddings, loaded_qids, loaded_preds, loaded_answers, metadata = result

    assert np.array_equal(loaded_embeddings, embeddings)
    assert np.array_equal(loaded_qids, question_ids)
    assert np.array_equal(loaded_preds, predictions)
    assert np.array_equal(loaded_answers, mc_answers)
    assert metadata["model"] == "gemma"
    assert metadata["strategy"] == "goal_driven"
    assert metadata["embedding_dim"] == 768
    assert metadata["n_samples"] == 10


def test_cache_validation(temp_cache_dir, tmp_path):
    """Test cache validation with matching and mismatching metadata."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")

    embeddings = np.random.randn(5, 768).astype(np.float32)
    cache_path = temp_cache_dir / "test_cache.npz"

    save_embeddings(
        cache_path,
        embeddings,
        np.arange(5),
        np.arange(5),
        np.array([f"A{i}" for i in range(5)]),
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )

    result = load_embeddings(cache_path)
    _, _, _, _, metadata = result

    # Valid cache
    assert validate_cache(metadata, dataset_path, EmbeddingModel.GEMMA, EmbeddingStrategy.GOAL_DRIVEN)

    # Invalid - different model
    assert not validate_cache(metadata, dataset_path, EmbeddingModel.QWEN, EmbeddingStrategy.GOAL_DRIVEN)

    # Invalid - different strategy
    assert not validate_cache(metadata, dataset_path, EmbeddingModel.GEMMA, EmbeddingStrategy.DOUBLE_BLIND)

    # Invalid - modified dataset
    dataset_path.write_text("modified,content\n3,4")
    assert not validate_cache(metadata, dataset_path, EmbeddingModel.GEMMA, EmbeddingStrategy.GOAL_DRIVEN)


@patch("kaggle_map.mlp.embedding_cache.encode")
def test_cache_miss_then_hit(mock_encode_func, temp_cache_dir, tiny_eval_rows, tmp_path):
    """Test that cache miss computes embeddings, cache hit loads them."""
    mock_encode_func.return_value = torch.randn(5, 768)

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")

    metadata_tuples = [(i, i % 2, f"A{i}") for i in range(5)]

    # First call - cache miss
    embeddings1, qids1, _preds1, _answers1 = get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert mock_encode_func.call_count == 1

    # Second call - cache hit
    embeddings2, qids2, _preds2, _answers2 = get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert mock_encode_func.call_count == 1  # Not called again
    assert np.array_equal(embeddings1, embeddings2)
    assert np.array_equal(qids1, qids2)


@patch("kaggle_map.mlp.embedding_cache.encode")
def test_different_configs_use_different_caches(mock_encode_func, temp_cache_dir, tiny_eval_rows, tmp_path):
    """Test that different model/strategy combinations use separate caches."""
    mock_encode_func.side_effect = lambda rows, strategy, model: torch.randn(
        len(rows),
        768 if model == EmbeddingModel.GEMMA else 8192
    )

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")
    metadata_tuples = [(i, i % 2, f"A{i}") for i in range(5)]

    # GEMMA + GOAL_DRIVEN
    get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert mock_encode_func.call_count == 1

    # QWEN + GOAL_DRIVEN - different cache
    get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.QWEN,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert mock_encode_func.call_count == 2

    # GEMMA + GOAL_DRIVEN again - should hit cache
    get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    assert mock_encode_func.call_count == 2  # No additional call


@patch("kaggle_map.mlp.embedding_cache.encode")
def test_precompute_all_embeddings(mock_encode_func, temp_cache_dir, tiny_eval_rows, tmp_path):
    """Test precomputing all model/strategy combinations."""
    call_count = 0
    def mock_encode_with_dims(rows, strategy, model):
        nonlocal call_count
        call_count += 1
        dim_map = {
            (EmbeddingModel.QWEN, EmbeddingStrategy.GOAL_DRIVEN): 8192,
            (EmbeddingModel.QWEN, EmbeddingStrategy.DOUBLE_BLIND): 16384,
            (EmbeddingModel.GEMMA, EmbeddingStrategy.GOAL_DRIVEN): 768,
            (EmbeddingModel.GEMMA, EmbeddingStrategy.DOUBLE_BLIND): 1536,
        }
        dim = dim_map[(model, strategy)]
        return torch.randn(len(rows), dim)

    mock_encode_func.side_effect = mock_encode_with_dims

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")
    metadata_tuples = [(i, i % 2, f"A{i}") for i in range(5)]

    # Precompute all combinations
    precompute_all_embeddings(tiny_eval_rows, metadata_tuples, dataset_path)

    # Should have computed 4 combinations
    assert call_count == 4

    # All cache files should exist
    cache_files = list(temp_cache_dir.glob("*.npz"))
    assert len(cache_files) == 4

    # Running precompute again should skip all (already cached)
    call_count = 0
    precompute_all_embeddings(tiny_eval_rows, metadata_tuples, dataset_path)
    assert call_count == 0


def test_list_cached_embeddings(temp_cache_dir, tmp_path):
    """Test listing all cached embeddings."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")

    # Create multiple cache files
    for model in [EmbeddingModel.QWEN, EmbeddingModel.GEMMA]:
        for strategy in [EmbeddingStrategy.GOAL_DRIVEN, EmbeddingStrategy.DOUBLE_BLIND]:
            dim = 768 if model == EmbeddingModel.GEMMA else 8192
            if strategy == EmbeddingStrategy.DOUBLE_BLIND:
                dim *= 2

            cache_path = _get_cache_path(dataset_path, model, strategy)
            save_embeddings(
                cache_path,
                np.random.randn(5, dim).astype(np.float32),
                np.arange(5),
                np.arange(5),
                np.array([f"A{i}" for i in range(5)]),
                dataset_path,
                model,
                strategy
            )

    cached = list_cached_embeddings()
    assert len(cached) == 4

    # Check that all combinations are present
    configs = [(c["model"], c["strategy"]) for c in cached]
    assert ("qwen", "goal_driven") in configs
    assert ("qwen", "double_blind") in configs
    assert ("gemma", "goal_driven") in configs
    assert ("gemma", "double_blind") in configs


def test_clear_cache(temp_cache_dir, tmp_path):
    """Test clearing all cached embeddings."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")

    # Create a cache file
    cache_path = temp_cache_dir / "test_cache.npz"
    save_embeddings(
        cache_path,
        np.random.randn(5, 768).astype(np.float32),
        np.arange(5),
        np.arange(5),
        np.array([f"A{i}" for i in range(5)]),
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )

    assert cache_path.exists()

    # Clear cache
    clear_cache()

    assert not cache_path.exists()
    assert len(list(temp_cache_dir.glob("*.npz"))) == 0


@patch("kaggle_map.mlp.embedding_cache.encode")
def test_cache_performance(mock_encode_func, temp_cache_dir, tiny_eval_rows, tmp_path):
    """Test that cache hit is significantly faster than cache miss."""
    # Add artificial delay to encode function
    def slow_encode(rows, strategy, model):
        time.sleep(0.05)  # 50ms delay
        return torch.randn(len(rows), 768)

    mock_encode_func.side_effect = slow_encode

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("dummy,data\n1,2")
    metadata_tuples = [(i, i % 2, f"A{i}") for i in range(5)]

    # Cache miss (with delay)
    start = time.time()
    get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    miss_time = time.time() - start

    # Cache hit (no delay)
    start = time.time()
    get_or_compute_embeddings(
        tiny_eval_rows,
        metadata_tuples,
        dataset_path,
        EmbeddingModel.GEMMA,
        EmbeddingStrategy.GOAL_DRIVEN
    )
    hit_time = time.time() - start

    # Cache hit should be at least 10x faster
    assert hit_time < miss_time / 10


def test_corrupt_cache_recovery(temp_cache_dir, tiny_eval_rows, tmp_path, mock_encode):
    """Test that corrupt cache files are handled gracefully."""
    with patch("kaggle_map.mlp.embedding_cache.encode", side_effect=mock_encode):
        dataset_path = tmp_path / "test.csv"
        dataset_path.write_text("dummy,data\n1,2")

        # Create corrupt cache file
        cache_path = _get_cache_path(
            dataset_path,
            EmbeddingModel.GEMMA,
            EmbeddingStrategy.GOAL_DRIVEN
        )
        cache_path.write_text("corrupt data")

        metadata_tuples = [(i, i % 2, f"A{i}") for i in range(5)]

        # Should handle corruption and compute fresh embeddings
        embeddings, _qids, _preds, _answers = get_or_compute_embeddings(
            tiny_eval_rows,
            metadata_tuples,
            dataset_path,
            EmbeddingModel.GEMMA,
            EmbeddingStrategy.GOAL_DRIVEN
        )

        assert embeddings is not None
        assert embeddings.shape == (5, 768)
