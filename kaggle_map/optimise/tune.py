"""Optuna-driven hyperparameter search utilities and CLI.

Run ``uv run -m kaggle_map.optimise.tune --help`` or
``uv run kaggle_map/optimise/tune.py -h`` for full usage details.
"""

from functools import partial
from typing import Any

import click
import optuna
import torch
from loguru import logger

from kaggle_map.core.dataset import load_training_data
from kaggle_map.core.models import (
    MLPTrainingConfig,
)
from kaggle_map.mlp.main import _get_split_indices, evaluate, fit
from kaggle_map.optimise.studies import create as create_study
from kaggle_map.optimise.studies import save as save_study
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


def objective(trial: optuna.Trial, search_scope: tuple[str, ...]) -> float:
    """Optuna objective that trains the MLP and returns validation MAP@3."""

    config = MLPTrainingConfig.sample_from_trial(trial, search_scope=search_scope)
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
    search_scope: tuple[str, ...],
) -> optuna.Study:
    """Run an Optuna study with a restricted search scope."""

    assert n_trials > 0, "Number of trials must be positive"

    study = create_study(study_name)
    logger.info(
        "Starting optimisation",
        study=study.study_name,
        trials=n_trials,
        search_scope=search_scope,
    )
    study.optimize(
        partial(objective, search_scope=search_scope),
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
    "--scope",
    "-s",
    multiple=True,
    help=("Restrict optimisation to specific hyperparameters, e.g. --scope learning_rate --scope dropout."),
)
def main(study_name: str, trials: int, search_scope: tuple[str, ...]) -> None:
    """CLI entrypoint for launching an MLP hyperparameter search.

    Examples:
        uv run -m kaggle_map.optimise.tune --trials 50
        uv run -m kaggle_map.optimise.tune --trials 150 --study-name mlp_experiment
        uv run -m kaggle_map.optimise.tune --scope learning_rate --scope optimizer
        uv run kaggle_map/optimise/tune.py --trials 500 -s scheduler,dropout
    """
    run_search(study_name, trials, search_scope=search_scope)


if __name__ == "__main__":
    main()
