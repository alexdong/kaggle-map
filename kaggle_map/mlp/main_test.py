"""Tests for MLP main module, focusing on dimension compatibility."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch
from loguru import logger

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    MLPTrainingConfig,
    OptimizerType,
    QuestionId,
    SchedulerType,
)
from kaggle_map.mlp.main import fit, load, predict_batch, save
from kaggle_map.mlp.model import QuestionSpecificMLP

CPU_DEVICE = torch.device("cpu")


def _build_mock_embedder(input_dim: int) -> Mock:
    """Return a mock embedder producing deterministic dimensionality."""
    embedder = Mock()

    def encode_side_effect(text: str | list[str]) -> torch.Tensor:
        if isinstance(text, list):
            return torch.randn(len(text), input_dim)
        return torch.randn(input_dim)

    embedder.encode.side_effect = encode_side_effect
    embedder.model_type_dim = input_dim
    return embedder


def _write_training_csv(tmp_path: Path) -> Path:
    """Persist a small training dataset to disk for integration tests."""
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
        "Category": [
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct",
            "False_Misconception",
        ],
        "Misconception": [
            "NA",
            "Addition error",
            "Addition error",
            "NA",
            "Addition error",
        ],
    }

    target_path = tmp_path / "training.csv"
    pd.DataFrame(data).to_csv(target_path, index=False)
    return target_path


def _make_training_config(
    return MLPTrainingConfig(
        train_csv_path=Path("datasets/train.csv"),
        epochs=1,
        batch_size=224,
        embedding_model=EMBEDDING_MODEL.GPT_OSS_20B,
        embedding_strategy=EmbeddingStrategy.DOUBLE_BLIND,
        architecture_size=ArchitectureSize.MEDIUM,
        activation=ActivationType.RELU,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.01,
        optimizer=OptimizerType.ADAM,
        scheduler=SchedulerType.COSINE,
        early_stopping_patience=5,
        train_split=0.8,
    )


@pytest.mark.slow
def test_fit_with_qwen_embeddings(tmp_path: Path) -> None:
    """Test that fit works with QWEN embeddings (8192-dim base)."""
    mock_qwen_embedder = _build_mock_embedder(8192)
    training_csv = _write_training_csv(tmp_path)

    with (
        patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_qwen_embedder),
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
    ):
        config = _make_training_config(
            training_csv,
            embedding_model=EmbeddingModel.QWEN,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
        )
        model, _ = fit(config)

    assert isinstance(model, QuestionSpecificMLP), "QWEN fit should return a QuestionSpecificMLP instance"
    assert model.trunk[0].in_features == 8224, "QWEN goal-driven should add 32-dim correctness head"


@pytest.mark.slow
def test_fit_with_gemma_goal_driven_strategy(tmp_path: Path) -> None:
    """Test that fit works with goal-driven strategy (single embedding)."""
    mock_embedder = _build_mock_embedder(768)
    training_csv = _write_training_csv(tmp_path)

    with (
        patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder),
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
    ):
        config = _make_training_config(
            training_csv,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
        )
        model, _ = fit(config)

    assert isinstance(model, QuestionSpecificMLP), "Gemma fit should return a QuestionSpecificMLP instance"
    assert model.trunk[0].in_features == 800, "Gemma goal-driven should add 32-dim correctness head"


def test_save_and_load_preserves_architecture(tmp_path: Path) -> None:
    """Test that saving and loading a model preserves its architecture."""
    mock_embedder = _build_mock_embedder(768)
    training_csv = _write_training_csv(tmp_path)

    with (
        patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder),
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
    ):
        config = _make_training_config(
            training_csv,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
        )
        original_model, train_config = fit(config)

    model_path = tmp_path / "model.pkl"
    save(original_model, model_path, train_config)
    loaded_model, _ = load(model_path)

    assert loaded_model.trunk[0].in_features == original_model.trunk[0].in_features
    assert len(loaded_model.trunk) == len(original_model.trunk)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("embedding_model", "embedding_strategy", "expected_input_dim"),
    [
        (EmbeddingModel.QWEN, EmbeddingStrategy.GOAL_DRIVEN, 8224),
        (EmbeddingModel.QWEN, EmbeddingStrategy.DOUBLE_BLIND, 16416),
    ],
)
def test_model_adapts_to_embedding_dimensions(
    tmp_path: Path, embedding_model: EmbeddingModel, embedding_strategy: EmbeddingStrategy, expected_input_dim: int
) -> None:
    """Test that the model correctly adapts to different embedding dimensions."""
    mock_qwen_embedder = _build_mock_embedder(8192)
    training_csv = _write_training_csv(tmp_path)

    with (
        patch("kaggle_map.embeddings.qwen.QwenEmbeddingModel.get_instance", return_value=mock_qwen_embedder),
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
    ):
        config = _make_training_config(
            training_csv,
            embedding_model=embedding_model,
            embedding_strategy=embedding_strategy,
        )
        model, _train_config = fit(config)

    assert model.trunk[0].in_features == expected_input_dim
    logger.info(
        f"Model with {embedding_model.value}/{embedding_strategy.value} has input dim {model.trunk[0].in_features}"
    )


@pytest.mark.slow
def test_predict_batch_with_multiple_rows(tmp_path: Path) -> None:
    """Test predict_batch efficiently processes multiple evaluation rows."""
    mock_embedder = _build_mock_embedder(768)
    training_csv = _write_training_csv(tmp_path)

    with (
        patch("kaggle_map.embeddings.gemma.GemmaEmbeddingModel.get_instance", return_value=mock_embedder),
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
    ):
        config = _make_training_config(
            training_csv,
            embedding_model=EmbeddingModel.GEMMA,
            embedding_strategy=EmbeddingStrategy.GOAL_DRIVEN,
        )
        model, train_config = fit(config)

    eval_rows = [
        EvaluationRow(
            row_id=index,
            question_id=QuestionId(1),
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation=f"Test explanation {index}",
            correct_answer="4",
        )
        for index in range(5)
    ]

    model = model.cpu()
    mock_embeddings = torch.randn(len(eval_rows), 768)
    correct_answers = {QuestionId(1): "4"}

    with (
        patch("kaggle_map.mlp.main.get_device", return_value=CPU_DEVICE),
        patch("kaggle_map.mlp.main.encode", return_value=mock_embeddings),
        patch("kaggle_map.mlp.main.load_training_data", return_value=[]),
        patch("kaggle_map.mlp.main.extract_correct_answers", return_value=correct_answers),
    ):
        results = predict_batch(model, eval_rows, train_config)

    assert len(results) == len(eval_rows), "predict_batch should emit one SubmissionRow per input"
    for index, result in enumerate(results):
        assert result.row_id == index, f"Row {index} should preserve its identifier"
        assert len(result.predicted_categories) == 3, f"Row {index} should have top-3 predictions"
