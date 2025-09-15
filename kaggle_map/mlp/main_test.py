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
from kaggle_map.mlp.main import evaluate, fit, load, predict, save
from kaggle_map.mlp.model import QuestionSpecificMLP


@pytest.fixture
def mock_embedder():
    """Mock embedder that handles both single and batch encoding."""
    mock = Mock()

    def encode_side_effect(text):
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


def test_fit_with_gemma_embeddings(mock_embedder, sample_training_csv):
    """Test that fit works with GEMMA embeddings (768-dim base)."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.DOUBLE_BLIND,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _ = fit(config)

        assert isinstance(model, QuestionSpecificMLP)
        # GEMMA double-blind: 768 * 2 = 1536, plus 32 for correctness = 1568
        assert model.trunk[0].in_features == 1568


def test_fit_with_semantic_strategy(mock_embedder, sample_training_csv):
    """Test that fit works with semantic strategy (single embedding)."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        config = TrainingConfig(
            train_csv_path=sample_training_csv,
            epochs=1,
            batch_size=2,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.SEMANTIC,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _ = fit(config)

        assert isinstance(model, QuestionSpecificMLP)
        # GEMMA semantic: 768, plus 32 for correctness = 800
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
            embedding_strategy=EmbeddingStrategy.DOUBLE_BLIND,
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
            embedding_strategy=EmbeddingStrategy.DOUBLE_BLIND,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, train_config = fit(config)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model_path = Path(f.name)
    save(model, model_path, train_config)

    # Load and predict
    loaded_model, loaded_config = load(model_path)
    # Mock predict's embedding call - no CUDA to avoid OOM
    with patch("kaggle_map.mlp.main.get_device", return_value=torch.device("cpu")):
        with patch("kaggle_map.mlp.main.encode", return_value=torch.randn(1536)):
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
            embedding_strategy=EmbeddingStrategy.DOUBLE_BLIND,
            architecture_size=ArchitectureSize.MEDIUM,
        )

        model, _train_config = fit(config)

        # Load test data (using same file for simplicity)
        from kaggle_map.core.dataset import load_training_data
        test_data = load_training_data(sample_training_csv)[:2]  # Use just 2 samples

        # Evaluate with mocked embeddings
        with patch("kaggle_map.mlp.main.encode", return_value=torch.randn(1536)):
            metrics = evaluate(model, test_data, config)

        assert "validation_map@3" in metrics
        assert "validation_samples" in metrics
        assert metrics["validation_samples"] == 2


@pytest.mark.parametrize(
    ("embedding_model", "embedding_strategy", "expected_input_dim"),
    [
        (EmbeddingModel.GEMMA, EmbeddingStrategy.DOUBLE_BLIND, 1568),  # 768*2 + 32
        (EmbeddingModel.GEMMA, EmbeddingStrategy.SEMANTIC, 800),  # 768 + 32
    ],
)
def test_model_adapts_to_embedding_dimensions(
    mock_embedder, sample_training_csv, embedding_model, embedding_strategy, expected_input_dim
):
    """Test that the model correctly adapts to different embedding dimensions."""
    with patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
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
