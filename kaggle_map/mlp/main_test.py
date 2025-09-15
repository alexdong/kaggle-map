"""Tests for MLP main module, focusing on dimension compatibility."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch
from loguru import logger

from kaggle_map.core.models import (
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    QuestionId,
    TrainingConfig,
)
from kaggle_map.mlp.main import evaluate, fit, load, predict, predict_batch, save
from kaggle_map.mlp.model import QuestionSpecificMLP


@pytest.fixture
def mock_embedder():
    """Mock embedder that handles both single and batch encoding."""
    mock = Mock()

    def encode_side_effect(text) -> torch.Tensor:
        # Handle both single text and list of texts
        if isinstance(text, list):
            # Return batch embeddings (2D)
            batch_size = len(text)
            return torch.randn(batch_size, 768)  # GEMMA dim
        # Single text (1D)
        return torch.randn(768)

    mock.encode.side_effect = encode_side_effect
    mock.model_type_dim = 768
    return mock


@pytest.fixture
def mock_qwen_embedder():
    """Mock QWEN embedder that handles both single and batch encoding."""
    mock = Mock()

    def encode_side_effect(text) -> torch.Tensor:
        # Handle both single text and list of texts
        if isinstance(text, list):
            # Return batch embeddings (2D)
            batch_size = len(text)
            return torch.randn(batch_size, 8192)  # QWEN dim
        # Single text (1D)
        return torch.randn(8192)

    mock.encode.side_effect = encode_side_effect
    mock.model_type_dim = 8192
    return mock


@pytest.fixture
def sample_training_csv():
    """Create a small CSV file with sample training data."""
    data = {
        "row_id": [1, 2, 3, 4, 5],
        "QuestionId": [1, 1, 1, 2, 2],
        "QuestionText": ["What is 2+2?"] * 3 + ["What is 3+3?"] * 2,
        "MC_Answer": ["4", "5", "3", "6", "7"],
        "StudentExplanation": [
            "Two plus two equals four",
            "I think it's five",
            "Maybe three",
            "Three plus three is six",
            "Could be seven",
        ],
        "Category": ["True_Correct", "False_Misconception", "False_Misconception", "True_Correct", "False_Misconception"],
        "Misconception": ["NA", "Addition error", "Addition error", "NA", "Addition error"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(data)
        df.to_csv(f, index=False)
        return Path(f.name)


@pytest.fixture
def sample_eval_row():
    """Create a sample evaluation row."""
    return EvaluationRow(
        row_id=100,
        question_id=QuestionId(1),
        question_text="What is 2+2?",
        mc_answer="4",
        student_explanation="Two plus two equals four",
        correct_answer="4",
    )


def test_fit_with_qwen_embeddings(mock_qwen_embedder, sample_training_csv):
    """Test that fit works with QWEN embeddings (8192-dim base)."""
    with patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_qwen_embedder):
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.QWEN,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _ = fit(config)

        assert isinstance(model, QuestionSpecificMLP)
        # QWEN goal-driven: 8192, plus 32 for correctness = 8224
        assert model.trunk[0].in_features == 8224


def test_fit_with_goal_driven_strategy_again(mock_embedder, sample_training_csv):
    """Test that fit works with goal-driven strategy (single embedding)."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _ = fit(config)

        assert isinstance(model, QuestionSpecificMLP)
        # GEMMA goal-driven: 768, plus 32 for correctness = 800
        assert model.trunk[0].in_features == 800


def test_save_and_load_preserves_architecture(mock_embedder, sample_training_csv):
    """Test that saving and loading a model preserves its architecture."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Train a model with specific dimensions
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        original_model, train_config = fit(config)

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model_path = Path(f.name)
    save(original_model, model_path, train_config)

    # Load the model
    loaded_model, _ = load(model_path)

    # Check that architectures match
    assert loaded_model.trunk[0].in_features == original_model.trunk[0].in_features
    assert len(loaded_model.trunk) == len(original_model.trunk)

    # Clean up
    model_path.unlink()


def test_predict_with_loaded_model(mock_embedder, sample_training_csv, sample_eval_row):
    """Test that predict works with a loaded model."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Train and save a model
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, train_config = fit(config)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model_path = Path(f.name)
    save(model, model_path, train_config)

    # Load and predict
    loaded_model, loaded_config = load(model_path)
    loaded_model = loaded_model.cpu()  # Force model to CPU
    # Mock predict's embedding call - no CUDA to avoid OOM
    with (
        patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")),
        patch("kaggle_map.mlp.main.encode", return_value=torch.randn(1, 1536)),  # 2D for batch processing
    ):
        result = predict(loaded_model, sample_eval_row, loaded_config)

    assert result.row_id == 100
    assert len(result.predicted_categories) == 3  # MAP@3 requires 3 predictions

    # Clean up
    model_path.unlink()


def test_evaluate_with_compatible_model(mock_embedder, sample_training_csv):
    """Test that evaluate works with a model trained on the same data."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Train a model
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _train_config = fit(config)

        # Load test data (using same file for simplicity)
        from kaggle_map.core.dataset import load_training_data
        test_data = load_training_data(sample_training_csv)[:2]  # Use just 2 samples

        # Evaluate with mocked embeddings
        with patch("kaggle_map.mlp.main.encode", return_value=torch.randn(2, 1536)):  # 2D for batch processing
            metrics = evaluate(model, test_data, config)

        assert "validation_map@3" in metrics
        assert "validation_samples" in metrics
        assert metrics["validation_samples"] == 2


@pytest.mark.parametrize(
    ("embedding_model", "embedding_strategy", "expected_input_dim"),
    [
        (EmbeddingModel.QWEN, EmbeddingStrategy.GOAL_DRIVEN, 8224),  # 8192 + 32
    ],
)
def test_model_adapts_to_embedding_dimensions(
    mock_qwen_embedder, sample_training_csv, embedding_model, embedding_strategy, expected_input_dim
):
    """Test that the model correctly adapts to different embedding dimensions."""
    with patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_qwen_embedder):
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=embedding_model,
            embedding_strategy=embedding_strategy,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _train_config = fit(config)

        # Check the first layer's input dimension matches expected
        assert model.trunk[0].in_features == expected_input_dim
        logger.info(
            f"Model with {embedding_model.value}/{embedding_strategy.value} "
            f"has input dim {model.trunk[0].in_features}"
        )


# ============================================================================
# Tests for New Batch Processing Functions
# ============================================================================


def test_predict_batch_with_empty_list():
    """Test predict_batch handles empty input gracefully."""

    # Arrange - no model needed for empty input
    evaluation_rows = []

    # Act
    result = predict_batch(None, evaluation_rows, None)

    # Assert
    assert result == [], "Empty input should return empty list"


def test_predict_batch_with_single_row(mock_embedder, sample_training_csv, sample_eval_row):
    """Test predict_batch works correctly with single evaluation row."""

    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange - train a model
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, train_config = fit(config)

        # Force CPU usage and mock batch encoding to return tensor with correct dimensions
        model = model.cpu()  # Move model to CPU
        with (
            patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")),
            patch("kaggle_map.mlp.main.encode", return_value=torch.randn(1, 1536)),  # Batch dimension
        ):
            # Act
            results = predict_batch(model, [sample_eval_row], train_config)

        # Assert
        assert len(results) == 1, "Single input should return single result"
        assert results[0].row_id == sample_eval_row.row_id
        assert len(results[0].predicted_categories) == 3, "Should return exactly 3 predictions per MAP@3"


def test_predict_batch_with_multiple_rows(mock_embedder, sample_training_csv):
    """Test predict_batch efficiently processes multiple evaluation rows."""

    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange - train a model
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, train_config = fit(config)

        # Create multiple evaluation rows
        eval_rows = [
            EvaluationRow(
                row_id=i,
                question_id=QuestionId(1),
                question_text="What is 2+2?",
                mc_answer="4",
                student_explanation=f"Test explanation {i}",
                correct_answer="4",
            )
            for i in range(5)
        ]

        # Force CPU usage and mock batch encoding to return tensor with correct batch dimensions
        model = model.cpu()  # Move model to CPU
        with (
            patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")),
            patch("kaggle_map.mlp.main.encode", return_value=torch.randn(5, 1536)),  # 5 rows, 1536 dim
        ):
            # Act
            results = predict_batch(model, eval_rows, train_config)

        # Assert
        assert len(results) == 5, "Should return one result per input row"
        for i, result in enumerate(results):
            assert result.row_id == i, f"Row {i} should have correct ID"
            assert len(result.predicted_categories) == 3, f"Row {i} should have 3 predictions"


def test_predict_batch_delegates_to_single_predict(mock_embedder, sample_training_csv, sample_eval_row):
    """Test that single predict() correctly delegates to predict_batch()."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange - train a model
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, train_config = fit(config)

        # Mock the batch prediction function to verify it's called
        with patch("kaggle_map.mlp.main.predict_batch") as mock_batch:
            from kaggle_map.core.models import Category, Prediction, SubmissionRow
            mock_batch.return_value = [SubmissionRow(
                row_id=sample_eval_row.row_id,
                predicted_categories=[
                    Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                    Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                    Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                ]
            )]

            # Act
            predict(model, sample_eval_row, train_config)

            # Assert
            mock_batch.assert_called_once_with(model, [sample_eval_row], train_config)


def test_predict_batch_respects_config_parameters(mock_embedder, sample_training_csv, sample_eval_row):
    """Test predict_batch uses config parameters for embedding strategy and model."""

    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, train_config = fit(config)

        # Force CPU usage and mock encode to verify it's called with correct parameters
        model = model.cpu()  # Move model to CPU
        with (
            patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")),
            patch("kaggle_map.mlp.main.encode") as mock_encode,
        ):
            mock_encode.return_value = torch.randn(1, 768)  # SEMANTIC dimension (before correctness embedding)

            # Act
            predict_batch(model, [sample_eval_row], train_config)

            # Assert - verify encode was called with config parameters
            mock_encode.assert_called_once()
            _, strategy_arg, model_arg = mock_encode.call_args[0]
            assert strategy_arg == EmbeddingStrategy.GOAL_DRIVEN
            assert model_arg == EmbeddingModel.QWEN


def test_predict_batch_uses_defaults_when_config_is_none(mock_embedder, sample_training_csv, sample_eval_row):
    """Test predict_batch falls back to defaults when no config provided."""

    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange - train model but don't pass config to predict_batch
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, _ = fit(config)

        # Force CPU usage and mock encode to verify default parameters are used
        model = model.cpu()  # Move model to CPU
        with (
            patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")),
            patch("kaggle_map.mlp.main.encode") as mock_encode,
        ):
            mock_encode.return_value = torch.randn(1, 8192)  # GOAL_DRIVEN dimension (before correctness embedding)

            # Act - pass None config
            predict_batch(model, [sample_eval_row], None)

            # Assert - verify encode was called with defaults
            mock_encode.assert_called_once()
            _, strategy_arg, model_arg = mock_encode.call_args[0]
            assert strategy_arg == EmbeddingStrategy.GOAL_DRIVEN
            assert model_arg == EmbeddingModel.QWEN


def test_evaluate_uses_batch_processing_for_efficiency(mock_embedder, sample_training_csv):
    """Test that evaluate() now uses batch processing instead of individual predictions."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        # Arrange
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
            architecture_size=ArchitectureSize.MEDIUM,
        )
        model, train_config = fit(config)

        # Create test data
        from kaggle_map.core.dataset import load_training_data
        test_data = load_training_data(sample_training_csv)[:3]  # Use 3 samples

        # Mock predict_batch to verify it's called instead of individual predict calls
        with patch("kaggle_map.mlp.main.predict_batch") as mock_batch:
            from kaggle_map.core.models import Category, Prediction, SubmissionRow
            # Mock return value with correct structure
            mock_batch.return_value = [
                SubmissionRow(
                    row_id=row.row_id,
                    predicted_categories=[
                        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                    ]
                )
                for row in test_data
            ]

            # Act
            evaluate(model, test_data, train_config)

            # Assert - verify batch processing was used
            mock_batch.assert_called_once()
            call_args = mock_batch.call_args[0]
            assert len(call_args[1]) == 3, "Should pass all 3 test samples to batch processor"


def test_evaluate_handles_empty_test_data():
    """Test evaluate returns appropriate metrics for empty test data."""
    # Arrange - create a mock model with eval method
    from unittest.mock import Mock
    mock_model = Mock()
    mock_model.eval = Mock()

    # Act
    metrics = evaluate(mock_model, [], None)

    # Assert
    assert metrics["validation_map@3"] == 0.0, "Empty test data should return 0.0 MAP@3"
    assert metrics["validation_samples"] == 0, "Empty test data should report 0 samples"
