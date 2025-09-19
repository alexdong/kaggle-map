"""Tests for optimisation fixed overrides and hyperparameter sampling."""

import pytest
from optuna.trial import FixedTrial

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    OptimizerType,
    SchedulerType,
)
from kaggle_map.optimise.hyperparameters import TunableParameters, sample_hyperparameters
from kaggle_map.optimise.tune import parse_fixed_overrides


def test_sample_hyperparameters_respects_fixed_overrides() -> None:
    """Parameters specified in fixed overrides must not be resampled."""

    trial = FixedTrial(
        {
            "epochs": 72,
            "batch_size": 288,
            "dropout": 0.24,
            "activation": ActivationType.RELU.value,
            "learning_rate": 1.2e-4,
            "weight_decay": 0.0042,
            "optimizer": OptimizerType.ADAMW.value,
            "scheduler": SchedulerType.COSINE.value,
            "early_stopping_patience": 18,
            "train_split": 0.74,
            "train_csv_path": "datasets/train.csv",
            "architecture_size": ArchitectureSize.LARGE.value,
            "embedding_model": EmbeddingModel.GEMMA.value,
            "embedding_strategy": EmbeddingStrategy.DOUBLE_BLIND.value,
        }
    )

    fixed = {
        TunableParameters.LEARNING_RATE: 9.9e-5,
        TunableParameters.OPTIMIZER: OptimizerType.SGD,
    }

    sampled = sample_hyperparameters(trial, fixed)

    assert sampled["learning_rate"] == 9.9e-5
    assert sampled["optimizer"] is OptimizerType.SGD
    assert sampled["epochs"] == 72
    assert sampled["activation"] is ActivationType.RELU
    assert sampled["train_csv_path"].as_posix() == "datasets/train.csv"
    assert sampled["embedding_model"] is EmbeddingModel.GEMMA
    assert sampled["embedding_strategy"] is EmbeddingStrategy.DOUBLE_BLIND


def test_parse_fixed_overrides_casts_types() -> None:
    """Parsing CLI fixed overrides should coerce values into typed config fields."""

    overrides = parse_fixed_overrides(
        (
            "epochs=120",
            "dropout=0.19",
            "activation=gelu",
            "optimizer=adam",
            "train_csv_path=datasets/custom.csv",
        )
    )

    assert overrides[TunableParameters.EPOCHS] == 120
    assert overrides[TunableParameters.DROPOUT] == pytest.approx(0.19)
    assert overrides[TunableParameters.ACTIVATION] is ActivationType.GELU
    assert overrides[TunableParameters.OPTIMIZER] is OptimizerType.ADAM
    assert overrides[TunableParameters.TRAIN_CSV_PATH].as_posix() == "datasets/custom.csv"


def test_parse_fixed_overrides_rejects_unknown_keys() -> None:
    """Unknown parameters should be rejected early to avoid silent mistakes."""

    with pytest.raises(AssertionError):
        parse_fixed_overrides(("unknown=1",))
