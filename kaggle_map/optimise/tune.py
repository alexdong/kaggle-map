"""Optuna-driven hyperparameter search utilities and CLI.

Run ``uv run -m kaggle_map.optimise.tune --help`` or
``uv run kaggle_map/optimise/tune.py -h`` for full usage details.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import click
import optuna
import torch
from loguru import logger

from kaggle_map.core.dataset import load_training_data
from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    OptimizerType,
    SchedulerType,
    TrainingConfig,
)
from kaggle_map.mlp.main import _get_split_indices, evaluate, fit
from kaggle_map.optimise.hyperparameters import TunableParameters, sample_hyperparameters
from kaggle_map.optimise.studies import create as create_study
from kaggle_map.optimise.studies import save as save_study
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


_FIXED_CASTERS: dict[TunableParameters, Callable[[str], Any]] = {
    TunableParameters.EPOCHS: int,
    TunableParameters.BATCH_SIZE: int,
    TunableParameters.DROPOUT: float,
    TunableParameters.ACTIVATION: ActivationType,
    TunableParameters.LEARNING_RATE: float,
    TunableParameters.WEIGHT_DECAY: float,
    TunableParameters.OPTIMIZER: OptimizerType,
    TunableParameters.SCHEDULER: SchedulerType,
    TunableParameters.EARLY_STOPPING_PATIENCE: int,
    TunableParameters.TRAIN_SPLIT: float,
    TunableParameters.TRAIN_CSV_PATH: Path,
    TunableParameters.ARCHITECTURE_SIZE: ArchitectureSize,
    TunableParameters.EMBEDDING_MODEL: EmbeddingModel,
    TunableParameters.EMBEDDING_STRATEGY: EmbeddingStrategy,
}


def parse_fixed_overrides(pairs: tuple[str, ...]) -> dict[TunableParameters, Any]:
    """Convert ``KEY=VALUE`` CLI pairs into typed fixed overrides."""

    overrides: dict[TunableParameters, Any] = {}
    for raw in pairs:
        assert "=" in raw, f"Fixed override must use KEY=VALUE format: {raw}"
        key_str, value = raw.split("=", 1)
        key_str = key_str.strip()
        value = value.strip()
        assert key_str, "Fixed override key cannot be empty"

        try:
            parameter = TunableParameters(key_str)
        except ValueError as error:
            valid = ", ".join(param.value for param in TunableParameters)
            msg = f"Unknown fixed parameter: {key_str}. Valid options: {valid}"
            raise AssertionError(msg) from error

        caster = _FIXED_CASTERS[parameter]
        overrides[parameter] = caster(value)
    return overrides


def objective(trial: optuna.Trial, fixed: dict[TunableParameters, Any]) -> float:
    """Optuna objective that trains the MLP and returns validation MAP@3."""

    params = sample_hyperparameters(trial, fixed)
    config = TrainingConfig(**params)
    model, trained_config = fit(config)

    training_data = load_training_data(config.train_csv_path)
    n_samples = len(training_data)
    split = _get_split_indices(n_samples, config.train_split)
    validation_rows = [training_data[i] for i in split.val_indices]
    result = evaluate(model, validation_rows, trained_config)
    return result["validation_map@3"]


def run_search(
    study_name: str,
    n_trials: int,
    *,
    fixed: dict[TunableParameters, Any],
) -> optuna.Study:
    """Run an Optuna study with optional fixed hyperparameters."""

    assert n_trials > 0, "Number of trials must be positive"

    study = create_study(study_name)
    logger.info(
        "Starting optimisation",
        study=study.study_name,
        trials=n_trials,
        fixed_parameters=fixed,
    )
    study.optimize(
        partial(objective, fixed=fixed),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=[torch.cuda.OutOfMemoryError],
    )

    logger.info("Search complete", best_value=study.best_value, best_params=study.best_params)
    save_study(study)
    return study


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--study-name", default="mlp_default", show_default=True, help="Optuna study name")
@click.option("--trials", type=int, default=100, show_default=True, help="Number of Optuna trials to run")
@click.option(
    "--fixed",
    "-f",
    multiple=True,
    help=(
        "Hold a hyperparameter constant during search, "
        "e.g. --fixed learning_rate=1e-4. Repeat the flag to lock multiple values."
    ),
)
def main(study_name: str, trials: int, fixed: tuple[str, ...]) -> None:
    """CLI entrypoint for launching an MLP hyperparameter search.

    Examples:
        uv run -m kaggle_map.optimise.tune --trials 50
        uv run -m kaggle_map.optimise.tune --trials 150 --study-name mlp_experiment
        uv run -m kaggle_map.optimise.tune --fixed learning_rate=1e-4 --fixed optimizer=adamw
        uv run kaggle_map/optimise/tune.py --trials 500 -f scheduler=cosine -f batch_size=384
    """

    overrides = parse_fixed_overrides(fixed)
    run_search(study_name, trials, fixed=overrides)


if __name__ == "__main__":
    main()
