from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import optuna
import torch
from loguru import logger
from optuna import Trial

from kaggle_map.mlp import Predictor
from kaggle_map.mlp.trainer import TrainingConfig

from .utils import (
    STORAGE_URL,
    cleanup_after_trial,
    clear_gpu_memory,
    create_study,
    handle_oom_error,
    save_best_config,
    track_gpu_memory,
)


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""

    strategy_name: str
    n_trials: int
    n_jobs: int = 1
    timeout: int | None = None
    train_data_path: str | None = None
    search_type: str = "regular"


def get_hyperparameter_search_space(trial: Trial) -> dict[str, Any]:
    """Get hyperparameter search space for MLP."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 8e-5, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [224, 256, 288, 320, 384, 448, 512]),
        "dropout": trial.suggest_float("dropout", 0.10, 0.42),
        "architecture_size": trial.suggest_categorical(
            "architecture_size", ["xlarge"] * 17 + ["large"] * 2 + ["medium"]
        ),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
        "weight_decay": trial.suggest_float("weight_decay", 3e-3, 1.5e-2, log=True),
        "activation": trial.suggest_categorical("activation", ["gelu", "silu", "relu", "leaky_relu"]),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "cosine", "onecycle", "none"]),
        "early_stopping_patience": trial.suggest_int("early_stopping_patience", 10, 22),
        "epochs": trial.suggest_int("epochs", 28, 180),
        "embedding_strategy": trial.suggest_categorical("embedding_strategy", ["double_blind", "semantic"]),
    }


def objective_function(
    trial: optuna.Trial,
    train_data_path: str | None = None,
) -> float:
    # Get hyperparameters
    hyperparams = get_hyperparameter_search_space(trial)

    # Create training config
    config = TrainingConfig(
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        optimizer=hyperparams["optimizer"],
        scheduler=hyperparams["scheduler"],
        early_stopping_patience=hyperparams["early_stopping_patience"],
        architecture_size=hyperparams["architecture_size"],
        dropout=hyperparams["dropout"],
        activation=hyperparams["activation"],
    )

    if train_data_path:
        config.train_csv_path = Path(train_data_path)

    # Clear GPU memory before training
    clear_gpu_memory()

    logger.info(f"Starting trial {trial.number}")

    # Handle OOM gracefully but let other errors crash
    try:
        model = Predictor.fit(config, embedding_strategy=hyperparams["embedding_strategy"])
        result = model.evaluate()

        # Track GPU utilization
        track_gpu_memory(trial)

        map_score = result["validation_map@3"]
        logger.info(f"Trial {trial.number} completed: MAP@3={map_score:.4f}")

        return map_score

    except torch.cuda.OutOfMemoryError as e:
        return handle_oom_error(trial, e)

    except Exception as e:
        # Log any other exceptions and ensure cleanup
        logger.error(f"Trial {trial.number} failed with error: {e}")
        raise

    finally:
        cleanup_after_trial()

    # This should never be reached due to return/raise above, but pyrefly requires it
    return 0.0


def run_search(config: SearchConfig) -> optuna.Study:
    """Run hyperparameter search with the given configuration."""
    search_desc = "embedding model comparison" if config.search_type == "embedding" else "hyperparameter"
    logger.info(f"Starting {search_desc} search for {config.strategy_name}")
    logger.info(f"Trials: {config.n_trials}, Jobs: {config.n_jobs}, Timeout: {config.timeout}s")
    if config.train_data_path:
        logger.info(f"Using training data: {config.train_data_path}")

    # Create timestamped study name
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if config.search_type == "embedding":
        study_name = f"{config.strategy_name}_embedding_{timestamp}"
    else:
        study_name = f"{config.strategy_name}_{timestamp}"

    study = create_study(study_name)

    # Create objective with bound parameters
    def objective(trial: optuna.Trial) -> float:
        return objective_function(trial, config.train_data_path)

    logger.info(f"Starting optimization with study: {study.study_name}")

    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_jobs,
        timeout=config.timeout,
        show_progress_bar=True,
    )

    logger.info(f"Search completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best configuration
    save_best_config(study, config.strategy_name)

    # Summary can be viewed via optuna-dashboard

    return study


@click.group()
def cli() -> None:
    pass


@click.command()
@click.option("--trials", default=100, help="Number of trials to run")
@click.option("--jobs", default=1, help="Number of parallel jobs")
@click.option("--timeout", default=None, type=int, help="Timeout in seconds")
@click.option("--train-data", default=None, help="Path to training data CSV")
def search(trials: int, jobs: int, timeout: int | None, train_data: str | None) -> None:
    config = SearchConfig(
        strategy_name="mlp",
        n_trials=trials,
        n_jobs=jobs,
        timeout=timeout,
        train_data_path=train_data,
        search_type="regular",
    )
    study = run_search(config)

    print("\nSearch completed!")
    print(f"Study name: {study.study_name}")
    print(f"Best MAP@3: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


@click.command("search-embeddings")
@click.option("--trials", default=70, help="Number of trials to run (default: 70)")
@click.option("--jobs", default=1, help="Number of parallel jobs (default: 1)")
@click.option("--timeout", default=21600, type=int, help="Timeout in seconds (default: 6 hours)")
@click.option("--train-data", default=None, help="Path to training data CSV")
def search_embeddings(trials: int, jobs: int, timeout: int, train_data: str | None) -> None:
    logger.info("=" * 60)
    logger.info("EMBEDDING MODEL COMPARISON STUDY")
    logger.info("=" * 60)
    logger.info("Strategy: mlp")
    logger.info(f"Trials: {trials} (7 models x ~10 configs each)")
    logger.info(f"Parallel jobs: {jobs}")
    logger.info(f"Timeout: {timeout}s ({timeout / 3600:.1f} hours)")
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
        n_jobs=jobs,
        timeout=timeout,
        train_data_path=train_data,
        search_type="embedding",
    )
    study = run_search(config)

    print("\nEmbedding search completed!")
    print(f"Study name: {study.study_name}")
    print(f"Best MAP@3: {study.best_value:.4f}")
    print(f"Best embedding: {study.best_params.get('embedding_model', 'N/A')}")


@click.command()
@click.argument("study")
def analyze(study: str) -> None:
    study_obj = optuna.load_study(study_name=study, storage=STORAGE_URL)

    if len(study_obj.trials) == 0:
        print(f"Study {study} has no trials")
        return

    # Basic info
    print(f"\nStudy: {study}")
    print(f"Trials: {len(study_obj.trials)}")
    print(f"Best Value: {study_obj.best_value:.4f}")
    print(f"Best Trial: #{study_obj.best_trial.number}")

    # Best parameters
    print("\nBest Parameters:")
    for param, value in study_obj.best_params.items():
        print(f"  {param}: {value}")

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
