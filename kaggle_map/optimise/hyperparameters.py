"""Hyperparameter sampling and search space definitions."""

from pathlib import Path
from typing import Literal

from optuna import Trial

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    OptimizerType,
    SchedulerType,
)

TunableParameters = Literal[
    "epochs",
    "batch_size",
    "dropout",
    "activation",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "early_stopping_patience",
    "train_split",
    "train_csv_path",
    "architecture_size",
    "embedding_model",
    "embedding_strategy",
]


def sample_hyperparameters(trial: Trial, search_scope: set[TunableParameters]) -> dict:
    """Sample hyperparameters from an Optuna trial for specified parameters."""

    suggestions = {
        "epochs": lambda t: t.suggest_int("epochs", 28, 180),
        "batch_size": lambda t: t.suggest_categorical("batch_size", [224, 256, 288, 320, 384, 448, 512]),
        "dropout": lambda t: t.suggest_float("dropout", 0.10, 0.42),
        "activation": lambda t: ActivationType(t.suggest_categorical("activation", [a.value for a in ActivationType])),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 8e-5, 3e-4, log=True),
        "weight_decay": lambda t: t.suggest_float("weight_decay", 3e-3, 1.5e-2, log=True),
        "optimizer": lambda t: OptimizerType(
            t.suggest_categorical("optimizer", [OptimizerType.ADAMW.value, OptimizerType.ADAM.value])
        ),
        "scheduler": lambda t: SchedulerType(
            t.suggest_categorical(
                "scheduler",
                [
                    SchedulerType.COSINE.value,
                    SchedulerType.COSINE.value,
                    SchedulerType.ONECYCLE.value,
                    SchedulerType.NONE.value,
                ],
            )
        ),
        "early_stopping_patience": lambda t: t.suggest_int("early_stopping_patience", 10, 22),
        "train_split": lambda t: t.suggest_float("train_split", 0.6, 0.85),
        "train_csv_path": lambda t: Path(
            t.suggest_categorical(
                "train_csv_path",
                [
                    "datasets/train.csv",
                    "datasets/synth_balanced_30000_total.csv",
                    "datasets/synth_original_366960_unbalanced.csv",
                    "datasets/synth_median_balanced_354210_total.csv",
                ],
            )
        ),
        "architecture_size": lambda t: ArchitectureSize(
            t.suggest_categorical(
                "architecture_size",
                [ArchitectureSize.XLARGE.value] * 17
                + [ArchitectureSize.LARGE.value] * 2
                + [ArchitectureSize.MEDIUM.value],
            )
        ),
        "embedding_model": lambda t: EmbeddingModel(
            t.suggest_categorical("embedding_model", [EmbeddingModel.QWEN.value, EmbeddingModel.GEMMA.value])
        ),
        "embedding_strategy": lambda t: EmbeddingStrategy(
            t.suggest_categorical(
                "embedding_strategy", [EmbeddingStrategy.DOUBLE_BLIND.value, EmbeddingStrategy.SEMANTIC.value]
            )
        ),
    }

    sampled = {}
    for field in search_scope:
        sampled[field] = suggestions[field](trial)

    return sampled
