"""Hyperparameter sampling and search space definitions."""

from enum import Enum
from pathlib import Path

from optuna import Trial

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    OptimizerType,
    SchedulerType,
)


class TunableParameters(str, Enum):
    """Parameters that can be tuned during hyperparameter optimization."""

    EPOCHS = "epochs"
    BATCH_SIZE = "batch_size"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    EARLY_STOPPING_PATIENCE = "early_stopping_patience"
    TRAIN_SPLIT = "train_split"
    TRAIN_CSV_PATH = "train_csv_path"
    ARCHITECTURE_SIZE = "architecture_size"
    EMBEDDING_MODEL = "embedding_model"
    EMBEDDING_STRATEGY = "embedding_strategy"


def sample_hyperparameters(trial: Trial, search_scope: set[TunableParameters]) -> dict:
    """Sample hyperparameters from an Optuna trial for specified parameters."""

    suggestions = {
        TunableParameters.EPOCHS: lambda t: t.suggest_int("epochs", 28, 180),
        TunableParameters.BATCH_SIZE: lambda t: t.suggest_categorical(
            "batch_size", [224, 256, 288, 320, 384, 448, 512]
        ),
        TunableParameters.DROPOUT: lambda t: t.suggest_float("dropout", 0.10, 0.42),
        TunableParameters.ACTIVATION: lambda t: ActivationType(
            t.suggest_categorical("activation", [a.value for a in ActivationType])
        ),
        TunableParameters.LEARNING_RATE: lambda t: t.suggest_float("learning_rate", 8e-5, 3e-4, log=True),
        TunableParameters.WEIGHT_DECAY: lambda t: t.suggest_float("weight_decay", 3e-3, 1.5e-2, log=True),
        TunableParameters.OPTIMIZER: lambda t: OptimizerType(
            t.suggest_categorical("optimizer", [OptimizerType.ADAMW.value, OptimizerType.ADAM.value])
        ),
        TunableParameters.SCHEDULER: lambda t: SchedulerType(
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
        TunableParameters.EARLY_STOPPING_PATIENCE: lambda t: t.suggest_int("early_stopping_patience", 10, 22),
        TunableParameters.TRAIN_SPLIT: lambda t: t.suggest_float("train_split", 0.6, 0.85),
        TunableParameters.TRAIN_CSV_PATH: lambda t: Path(
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
        TunableParameters.ARCHITECTURE_SIZE: lambda t: ArchitectureSize(
            t.suggest_categorical(
                "architecture_size",
                [ArchitectureSize.XLARGE.value] * 17
                + [ArchitectureSize.LARGE.value] * 2
                + [ArchitectureSize.MEDIUM.value],
            )
        ),
        TunableParameters.EMBEDDING_MODEL: lambda t: EmbeddingModel(
            t.suggest_categorical("embedding_model", [EmbeddingModel.QWEN.value, EmbeddingModel.GEMMA.value])
        ),
        TunableParameters.EMBEDDING_STRATEGY: lambda t: EmbeddingStrategy(
            t.suggest_categorical(
                "embedding_strategy", [EmbeddingStrategy.GOAL_DRIVEN.value]
            )
        ),
    }

    sampled = {}
    for field in search_scope:
        sampled[field.value] = suggestions[field](trial)

    return sampled
