from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import optuna
import torch
from loguru import logger

from kaggle_map.core.dataset import load_training_data
from kaggle_map.core.models import TrainingConfig
from kaggle_map.mlp.predictor import _get_split_indices, evaluate, fit

from .utils import (
    STORAGE_URL,
    clear_gpu_memory,
    create_study,
    save_best_config,
)


def objective(config: TrainingConfig) -> float:
    model = fit(config)

    training_data = load_training_data(config.train_csv_path)
    n_samples = len(training_data)
    split = _get_split_indices(n_samples, config.train_split, config.random_seed)
    test_data = [training_data[i] for i in split.val_indices]
    result = evaluate(model, test_data)
    map_score = result["validation_map@3"]
    logger.info(f"Evaluation completed: MAP@3={map_score:.4f}")

    clear_gpu_memory()
    return map_score


def run_search() -> optuna.Study:
    """Run hyperparameter search with the given configuration."""
    study = create_study(study_name)

    def objective_wrapper(trial: optuna.Trial) -> float:
        training_config = TrainingConfig.get_sample_hyperparameters(trial)

        score = objective(training_config)
        logger.info(f"Trial {trial.number} completed: MAP@3={score:.4f}")
        return score

    logger.info(f"Starting optimization with study: {study.study_name}")

    study.optimize(
        objective_wrapper,
        n_trials=config.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=[torch.cuda.OutOfMemoryError],
    )

    logger.info(f"Search completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    save_best_config(study, config)
    return study


@click.group()
def cli() -> None:
    pass


@click.command()
@click.option("--trials", default=100, help="Number of trials to run")
@click.option("--jobs", default=1, help="Number of parallel jobs")
@click.option("--train-data", default=None, help="Path to training data CSV")
def search(trials: int, jobs: int, timeout: int | None, train_data: str | None) -> None:
    config = SearchConfig(
        strategy_name="mlp",
        n_trials=trials,
        train_data_path=train_data,
        search_type="regular",
    )
    study = run_search(config)

    logger.info("Search completed!")
    logger.info(f"Study name: {study.study_name}")
    logger.info(f"Best MAP@3: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")


@click.command("search-embeddings")
@click.option("--trials", default=70, help="Number of trials to run (default: 70)")
@click.option("--jobs", default=1, help="Number of parallel jobs (default: 1)")
@click.option("--train-data", default=None, help="Path to training data CSV")
def search_embeddings(trials: int, jobs: int, train_data: str | None) -> None:
    logger.info("=" * 60)
    logger.info("EMBEDDING MODEL COMPARISON STUDY")
    logger.info("=" * 60)
    logger.info("Strategy: mlp")
    logger.info(f"Trials: {trials} (7 models x ~10 configs each)")
    logger.info(f"Parallel jobs: {jobs}")
    logger.info("")
    logger.info("Models to test:")
    logger.info("  - MINI_LM (384 dim) - baseline")
    logger.info("  - E5_BASE (768 dim)")
    logger.info("  - INSTRUCTOR_BASE (768 dim)")
    logger.info("  - BGE_BASE (768 dim)")
    logger.info("  - CONTRIEVER (768 dim)")
    logger.info("  - SENTENCE_T5_BASE (768 dim)")
    logger.info("  - MINI_LM_L12 (384 dim)")
    logger.info("=" * 60)

    config = SearchConfig(
        strategy_name="mlp",
        n_trials=trials,
        train_data_path=train_data,
        search_type="embedding",
    )
    study = run_search(config)

    logger.info("Embedding search completed!")
    logger.info(f"Study name: {study.study_name}")
    logger.info(f"Best MAP@3: {study.best_value:.4f}")
    logger.info(f"Best embedding: {study.best_params.get('embedding_model', 'N/A')}")


@click.command()
@click.argument("study")
def analyze(study: str) -> None:
    study_obj = optuna.load_study(study_name=study, storage=STORAGE_URL)

    if len(study_obj.trials) == 0:
        logger.info(f"Study {study} has no trials")
        return

    # Basic info
    logger.info(f"Study: {study}")
    logger.info(f"Trials: {len(study_obj.trials)}")
    logger.info(f"Best Value: {study_obj.best_value:.4f}")
    logger.info(f"Best Trial: #{study_obj.best_trial.number}")

    # Best parameters
    logger.info("Best Parameters:")
    for param, value in study_obj.best_params.items():
        logger.info(f"  {param}: {value}")

    # Detailed analysis available via optuna-dashboard:
    # Run 'make dashboard' to launch interactive visualizatio


# Add commands to CLI
cli.add_command(search)
cli.add_command(search_embeddings)
cli.add_command(analyze)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
