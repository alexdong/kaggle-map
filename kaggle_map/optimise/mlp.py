from datetime import datetime
from pathlib import Path
from typing import Any

import click
import optuna
import torch
from loguru import logger

from kaggle_map.strategies import get_strategy

from .utils import (
    STORAGE_URL,
    build_wandb_run_name,
    cleanup_after_trial,
    clear_gpu_memory,
    create_study,
    generate_study_summary,
    handle_oom_error,
    save_best_config,
    track_gpu_memory,
)


def objective_function(
    trial: optuna.Trial,
    strategy_class: Any,  # Use Any to avoid type checking issues
    train_data_path: str | None = None,
    search_type: str = "regular"
) -> float:

    # Get hyperparameters from strategy based on search type
    if search_type == "embedding":
        # Use embedding-specific search space
        if hasattr(strategy_class, "get_embedding_search_space"):
            hyperparams = strategy_class.get_embedding_search_space(trial)
        else:
            msg = f"Strategy {strategy_class.__name__} does not support embedding search"
            raise ValueError(msg)
    else:
        # Use regular hyperparameter search space
        hyperparams = strategy_class.get_hyperparameter_search_space(trial)

    # Add train_csv_path if provided
    if train_data_path:
        hyperparams["train_csv_path"] = Path(train_data_path)

    # Build wandb run name and metadata
    wandb_run_name = build_wandb_run_name(trial, hyperparams)
    hyperparams["wandb_run_name"] = wandb_run_name

    # Add metadata for wandb tracking
    hyperparams["wandb_tags"] = [
        f"study_{trial.study.study_name}",
        f"trial_{trial.number}",
        "hypersearch"
    ]
    hyperparams["study_id"] = trial.study.study_name
    hyperparams["trial_number"] = trial.number

    # Clear GPU memory before training
    clear_gpu_memory()

    logger.info(f"Starting trial {trial.number} with params: {hyperparams}")

    # Handle OOM gracefully but let other errors crash
    try:
        model = strategy_class.fit(**hyperparams)
        result = strategy_class.evaluate_on_split(model)

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


def run_search(
    strategy_name: str,
    n_trials: int,
    n_jobs: int,
    timeout: int | None = None,
    train_data_path: str | None = None,
    search_type: str = "regular",
) -> optuna.Study:

    search_desc = "embedding model comparison" if search_type == "embedding" else "hyperparameter"
    logger.info(f"Starting {search_desc} search for {strategy_name}")
    logger.info(f"Trials: {n_trials}, Jobs: {n_jobs}, Timeout: {timeout}s")
    if train_data_path:
        logger.info(f"Using training data: {train_data_path}")

    strategy_class = get_strategy(strategy_name)

    # Create timestamped study name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if search_type == "embedding":
        study_name = f"{strategy_name}_embedding_{timestamp}"
    else:
        study_name = f"{strategy_name}_{timestamp}"

    study = create_study(study_name)

    # Create objective with bound parameters
    def objective(trial: optuna.Trial) -> float:
        return objective_function(trial, strategy_class, train_data_path, search_type)

    logger.info(f"Starting optimization with study: {study.study_name}")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True,
    )

    logger.info(f"Search completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best configuration
    save_best_config(study, strategy_name)

    # Generate study summary
    generate_study_summary(study)

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
    study = run_search("mlp", trials, jobs, timeout, train_data, "regular")

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
    logger.info(f"Timeout: {timeout}s ({timeout/3600:.1f} hours)")
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

    study = run_search("mlp", trials, jobs, timeout, train_data, "embedding")

    print("\nEmbedding search completed!")
    print(f"Study name: {study.study_name}")
    print(f"Best MAP@3: {study.best_value:.4f}")
    print(f"Best embedding: {study.best_params.get('embedding_model', 'N/A')}")


@click.command()
@click.argument("study")
def analyze(study: str) -> None:
    try:
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

        # Generate and save summary
        summary_path = generate_study_summary(study_obj)
        if summary_path:
            print(f"\nDetailed summary saved to: {summary_path}")

    except Exception as e:
        logger.error(f"Failed to analyze study: {e}")
        print(f"Error: Failed to load study '{study}'")


# Add commands to CLI
cli.add_command(search)
cli.add_command(search_embeddings)
cli.add_command(analyze)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

