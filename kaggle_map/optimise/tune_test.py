"""Tests for optimisation fixed overrides and hyperparameter sampling."""

import pytest
from optuna.trial import FixedTrial

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    MLPTrainingConfig,
    OptimizerType,
    SchedulerType,
)
from kaggle_map.optimise.tune import parse_search_scope


def test_sample_from_trial_respects_search_scope() -> None:
    trial = FixedTrial(
        {
            "epochs": 72,
            "batch_size": 288,
            "dropout": 0.24,
            "activation": ActivationType.RELU.value,
            "learning_rate": 1.2e-4,
            "weight_decay": 0.0042,
            "optimizer": OptimizerType.SGD.value,
            "scheduler": SchedulerType.COSINE.value,
            "early_stopping_patience": 18,
            "train_split": 0.74,
            "train_csv_path": "datasets/train.csv",
            "architecture_size": ArchitectureSize.LARGE.value,
            "embedding_model": EmbeddingModel.GEMMA.value,
            "embedding_strategy": EmbeddingStrategy.DOUBLE_BLIND.value,
        }
    )

    scope = ("learning_rate", "optimizer")

    sampled = MLPTrainingConfig.sample_from_trial(trial, search_scope=scope)
    defaults = MLPTrainingConfig.model_validate({})

    assert sampled.learning_rate == pytest.approx(1.2e-4)
    assert sampled.optimizer is OptimizerType.SGD
    assert sampled.epochs == defaults.epochs
    assert sampled.activation is defaults.activation
    assert sampled.train_csv_path == defaults.train_csv_path
    assert sampled.embedding_model is defaults.embedding_model
    assert sampled.embedding_strategy is defaults.embedding_strategy


def test_parse_search_scope_accepts_comma_separated_values() -> None:
    scope = parse_search_scope(("learning_rate,dropout", "optimizer"))
    assert scope == ("learning_rate", "dropout", "optimizer")
