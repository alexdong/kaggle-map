"""Tests for embedding module batch encoding functionality."""

from unittest.mock import Mock, patch

import pytest
import torch

from kaggle_map.core.models import (
    Category,
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    Prediction,
    TrainingRow,
)
from kaggle_map.embeddings import encode


@pytest.fixture
def mock_embedder():
    """Mock embedder that handles both single and batch encoding."""
    mock = Mock()

    def encode_side_effect(text, batch_size=32) -> torch.Tensor:  # noqa: ARG001
        # Determine dimension based on model type flag
        dim = getattr(mock, "model_type_dim", 768)  # Default to Gemma (768)

        if isinstance(text, list):
            return torch.randn(len(text), dim)  # Batch: 2D tensor
        return torch.randn(dim)  # Single: 1D tensor

    mock.encode.side_effect = encode_side_effect
    mock.model_type_dim = 768  # Default dimension for test mock
    return mock


@pytest.fixture
def create_evaluation_row():
    """Factory fixture for creating test EvaluationRow instances."""

    def _create(row_id=1, question_id=100, correct_answer="4") -> EvaluationRow:
        return EvaluationRow(
            row_id=row_id,
            question_id=question_id,
            question_text="What is 2 + 2?",
            mc_answer="4",
            student_explanation="Because two plus two equals four",
            correct_answer=correct_answer,  # Required for GOAL_DRIVEN strategy
        )

    return _create


@pytest.fixture
def create_training_row():
    """Factory fixture for creating test TrainingRow instances."""

    def _create(row_id=1, question_id=100, category=Category.TRUE_CORRECT) -> TrainingRow:
        return TrainingRow(
            row_id=row_id,
            question_id=question_id,
            question_text="What is 2 + 2?",
            mc_answer="4",
            student_explanation="Because two plus two equals four",
            prediction=Prediction(category=category, misconception="NA"),
        )

    return _create


@pytest.fixture
def sample_training_data(create_training_row):
    """Fixture providing sample training data."""
    return [create_training_row(row_id=i, question_id=100 + i) for i in range(10)]


# =============================================================================
# Current Behavior Tests (Before Changes)
# =============================================================================


def test_encode_single_goal_driven_gemma_creates_unified_embedding(mock_embedder, create_evaluation_row):
    """Test GOAL_DRIVEN strategy creates single embedding from all text components."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        row = create_evaluation_row()
        result = encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        assert result.shape == (768,), f"Expected shape (768,), got {result.shape}"
        assert isinstance(result, torch.Tensor)
        # Should call encode once with unified text
        assert mock_embedder.encode.call_count == 1


def test_encode_single_goal_driven_gemma(mock_embedder, create_evaluation_row):
    """Test GOAL_DRIVEN strategy creates unified embedding."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        row = create_evaluation_row()
        result = encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        assert result.shape == (768,), f"Expected shape (768,), got {result.shape}"
        # Should call encode once with unified text
        assert mock_embedder.encode.call_count == 1


def test_encode_single_goal_driven_qwen_unified(mock_embedder, create_evaluation_row):
    """Test current single-row encode with GOAL_DRIVEN strategy for Qwen."""
    mock_embedder.model_type_dim = 8192
    with patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_embedder):
        row = create_evaluation_row()
        result = encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.QWEN)

        assert result.shape == (8192,), f"Expected shape (8192,), got {result.shape}"
        assert isinstance(result, torch.Tensor)
        assert mock_embedder.encode.call_count == 1


def test_encode_single_goal_driven_qwen(mock_embedder, create_evaluation_row):
    """Test current single-row encode with GOAL_DRIVEN strategy for Qwen."""
    mock_embedder.model_type_dim = 8192
    with patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_embedder):
        row = create_evaluation_row()
        result = encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.QWEN)

        assert result.shape == (8192,), f"Expected shape (8192,), got {result.shape}"
        assert mock_embedder.encode.call_count == 1


def test_encode_requires_correct_answer_for_goal_driven(create_evaluation_row):
    """Test that GOAL_DRIVEN strategy requires correct_answer."""
    row = create_evaluation_row(correct_answer=None)

    with pytest.raises(AssertionError, match="Correct answer is required"):
        encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)


# =============================================================================
# Future Behavior Tests (After Changes) - Currently Expected to Fail
# =============================================================================


def test_encode_batch_goal_driven_gemma_unified(mock_embedder, create_evaluation_row):
    """Test batch encoding with GOAL_DRIVEN strategy for Gemma."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        rows = [create_evaluation_row(row_id=i) for i in range(5)]

        # encode now accepts list of rows
        result = encode(rows, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        assert result.shape == (5, 768), f"Expected shape (5, 768), got {result.shape}"
        # Should make ONE batch call
        assert mock_embedder.encode.call_count == 1
        # Should pass a list of texts
        call_args = mock_embedder.encode.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 5


def test_encode_batch_goal_driven_gemma(mock_embedder, create_evaluation_row):
    """Test batch encoding with GOAL_DRIVEN strategy for Gemma."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        rows = [create_evaluation_row(row_id=i) for i in range(5)]

        result = encode(rows, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        assert result.shape == (5, 768), f"Expected shape (5, 768), got {result.shape}"
        # Should make ONE batch call
        assert mock_embedder.encode.call_count == 1


def test_encode_batch_empty_list(mock_embedder):
    """Test batch encoding with empty list."""
    mock_embedder.model_type_dim = 8192
    with patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_embedder):
        result = encode([], EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.QWEN)

        assert result.shape == (0, 8192), f"Expected shape (0, 8192), got {result.shape}"
        assert mock_embedder.encode.call_count == 0


def test_encode_batch_single_item(mock_embedder, create_evaluation_row):
    """Test batch encoding with single item returns 2D tensor."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        rows = [create_evaluation_row()]

        result = encode(rows, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        # Should return 2D tensor even for single item
        assert result.shape == (1, 768), f"Expected shape (1, 768), got {result.shape}"


def test_encode_backward_compatibility(mock_embedder, create_evaluation_row):
    """Test that single row still works after batch changes."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        row = create_evaluation_row()
        result = encode(row, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)

        # Should still return 1D tensor for single row
        assert result.shape == (768,), f"Expected shape (768,), got {result.shape}"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_encode_validates_strategy_parameter():
    """Test that invalid embedding strategy raises assertion error."""
    from kaggle_map.core.models import EvaluationRow

    row = EvaluationRow(
        row_id=1, question_id=100, question_text="What is 2+2?",
        mc_answer="4", student_explanation="Basic math", correct_answer="4"
    )

    with pytest.raises(AssertionError, match="Invalid embedding strategy"):
        encode(row, "INVALID_STRATEGY", EmbeddingModel.GEMMA)


def test_encode_validates_model_parameter():
    """Test that invalid embedding model raises assertion error."""
    from kaggle_map.core.models import EvaluationRow

    row = EvaluationRow(
        row_id=1, question_id=100, question_text="What is 2+2?",
        mc_answer="4", student_explanation="Basic math", correct_answer="4"
    )

    with pytest.raises(AssertionError, match="Invalid embedding model"):
        encode(row, EmbeddingStrategy.GOAL_DRIVEN, "INVALID_MODEL")


def test_encode_batch_requires_correct_answer_for_goal_driven(create_evaluation_row):
    """Test that batch GOAL_DRIVEN strategy validates all rows have correct_answer."""
    rows = [
        create_evaluation_row(row_id=1, correct_answer="4"),
        create_evaluation_row(row_id=2, correct_answer=None),  # Missing correct answer
    ]

    with pytest.raises(AssertionError, match="Correct answer is required"):
        encode(rows, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)


def test_encode_rejects_invalid_input_type():
    """Test that encode rejects invalid input types."""
    with pytest.raises(TypeError, match="Expected EvaluationRow or list"):
        encode("invalid_input", EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)


# =============================================================================
# Integration Tests for fit() Function
# =============================================================================


def test_fit_uses_batch_encoding(mock_embedder, sample_training_data):
    """Test that fit function uses batch encoding efficiently."""
    from kaggle_map.core.models import TrainingConfig
    from kaggle_map.mlp.main import fit

    # Create a mock encode function that returns the expected tensor shape
    def mock_encode(rows, strategy, model):
        return torch.randn(len(rows), 768)

    mock_encode_func = Mock(side_effect=mock_encode)

    with (
        patch("kaggle_map.mlp.main.load_training_data", return_value=sample_training_data),
        patch("kaggle_map.mlp.embedding_cache.encode", mock_encode_func),
        patch("kaggle_map.mlp.embedding_cache.load_embeddings", return_value=None),  # Force recomputation
        patch("kaggle_map.mlp.main.train_model"),
    ):
                config = TrainingConfig(epochs=1, batch_size=32)
                fit(config)

                # Should call encode with batch (list), not individual items
                # For GOAL_DRIVEN and SEMANTIC strategies, should make 1 batch call
                expected_calls = 1
                assert mock_encode_func.call_count == expected_calls

                # Verify it was called with a list
                call_args = mock_encode_func.call_args[0][0]
                assert isinstance(call_args, list)
                assert len(call_args) == len(sample_training_data)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
def test_batch_encoding_performance(mock_embedder, create_evaluation_row):
    """Test that batch encoding is significantly faster than sequential."""
    import time

    # Add realistic delay to mock
    def encode_with_delay(text, batch_size=32) -> torch.Tensor:  # noqa: ARG001
        # Batch is much faster per item
        delay = 0.001 * len(text) if isinstance(text, list) else 0.01
        time.sleep(delay)
        dim = 768
        if isinstance(text, list):
            return torch.randn(len(text), dim)
        return torch.randn(dim)

    mock_embedder.encode.side_effect = encode_with_delay

    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        rows = [create_evaluation_row(row_id=i) for i in range(10)]

        # Time batch encoding
        start = time.time()
        result = encode(rows, EmbeddingStrategy.GOAL_DRIVEN, EmbeddingModel.GEMMA)
        batch_time = time.time() - start

        # Batch should be < 0.05s (10 * 0.001 + overhead)
        assert batch_time < 0.05, f"Batch encoding took {batch_time:.3f}s, expected < 0.05s"
        assert result.shape == (10, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
