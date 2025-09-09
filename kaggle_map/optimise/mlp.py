import click
import optuna
import torch
from loguru import logger

from kaggle_map.core.dataset import load_training_data
from kaggle_map.core.models import TrainingConfig
from kaggle_map.mlp.predictor import _get_split_indices, evaluate, fit
from kaggle_map.optimise.studies import (
    create as create_study,
)
from kaggle_map.optimise.studies import (
    save as save_study,
)


def objective(trial: optuna.Trial) -> float:
    config = TrainingConfig.get_sample_hyperparameters(trial)
    model = fit(config)

    training_data = load_training_data(config.train_csv_path)
    n_samples = len(training_data)
    split = _get_split_indices(n_samples, config.train_split, config.random_seed)
    test_data = [training_data[i] for i in split.val_indices]
    result = evaluate(model, test_data)
    return result["validation_map@3"]


def run_search(study_name: str, n_trials: int) -> optuna.Study:
    study = create_study(study_name)
    logger.info(f"Starting optimization with study: {study.study_name}")
    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=[torch.cuda.OutOfMemoryError],
    )
    logger.info(f"Search completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    save_study(study)
    return study
