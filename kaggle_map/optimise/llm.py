"""LLM GGUF quantization comparison and optimization."""

import time
from pathlib import Path

import click
import optuna
import pandas as pd
from huggingface_hub import hf_hub_download
from loguru import logger

from .utils import STORAGE_URL

# Quantization options to test
QUANTIZATION_OPTIONS = [
    "IQ4_XS",   # 6.55 GB - Smallest 4-bit
    "IQ4_NL",   # 6.89 GB - Non-linear 4-bit
    "Q4_0",     # 6.91 GB - Original 4-bit
    "Q4_1",     # 7.56 GB - 4-bit with importance
    "Q4_K_S",   # 6.94 GB - K-quant small
    "Q4_K_M",   # 7.3 GB - K-quant medium (recommended)
    "Q4_K_XL",  # 7.43 GB - K-quant extra large
]

# Model base URL on HuggingFace
MODEL_BASE = "unsloth/gemma-3-12b-it-GGUF"


def get_model_size(quantization: str) -> float:
    """Get approximate model size in GB.

    Args:
        quantization: Quantization type

    Returns:
        Model size in GB
    """
    sizes = {
        "IQ4_XS": 6.55,
        "IQ4_NL": 6.89,
        "Q4_0": 6.91,
        "Q4_1": 7.56,
        "Q4_K_S": 6.94,
        "Q4_K_M": 7.30,
        "Q4_K_XL": 7.43,
    }
    return sizes.get(quantization, 0.0)


def download_model_if_needed(quantization: str) -> Path:
    """Download the GGUF model if not already cached.

    Args:
        quantization: Quantization type

    Returns:
        Path to downloaded model
    """
    model_name = f"gemma-3-12b-it-{quantization}.gguf"
    model_dir = Path("models/gguf")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    if model_path.exists():
        logger.info(f"Model {model_name} already cached")
        return model_path

    logger.info(f"Downloading {model_name} from HuggingFace...")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_BASE,
        filename=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )

    logger.info(f"Downloaded to {downloaded_path}")
    return Path(downloaded_path)


def evaluate_quantization(quantization: str, sample_size: int = 100) -> dict:
    """Evaluate a specific quantization option.

    Args:
        quantization: Quantization type to evaluate
        sample_size: Number of samples to evaluate

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating quantization: {quantization}")

    # Download model if needed
    model_path = download_model_if_needed(quantization)

    # Import here to avoid circular imports
    from kaggle_map.core.dataset import (
        extract_correct_answers,
        extract_misconceptions_by_popularity,
        parse_training_data,
    )
    from kaggle_map.strategies.llm import LLMStrategy
    from kaggle_map.strategies.utils import split_training_data

    # Create strategy with specific quantization
    strategy = LLMStrategy(model_path=str(model_path))

    # Load training data for fit
    training_data = parse_training_data(Path("datasets/train.csv"))

    # Fit the strategy (loads correct answers and misconceptions)
    train_data, _, _ = split_training_data(training_data, train_ratio=0.7, random_seed=42)

    strategy.correct_answers = extract_correct_answers(train_data)
    strategy.misconceptions_by_question = extract_misconceptions_by_popularity(train_data)

    # Measure evaluation time
    start_time = time.time()

    # Evaluate on validation split
    results = strategy.evaluate_on_split(
        model=strategy,
        train_split=0.7,
        random_seed=42,
        sample_size=sample_size
    )

    evaluation_time = time.time() - start_time

    # Clean up model from memory
    del strategy

    return {
        "quantization": quantization,
        "map_at_3": results["map_at_3"],
        "evaluation_time": evaluation_time,
        "samples_per_second": sample_size / evaluation_time,
        "model_size_gb": get_model_size(quantization),
    }


def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Optuna objective function for multi-objective optimization.

    Args:
        trial: Optuna trial

    Returns:
        Tuple of (negative MAP@3, evaluation time) for multi-objective optimization
    """
    # Select quantization
    quantization = trial.suggest_categorical("quantization", QUANTIZATION_OPTIONS)

    # Evaluate
    results = evaluate_quantization(quantization, sample_size=100)

    # Log results
    logger.info(
        f"Quantization: {quantization} | "
        f"MAP@3: {results['map_at_3']:.4f} | "
        f"Time: {results['evaluation_time']:.2f}s | "
        f"Speed: {results['samples_per_second']:.2f} samples/s"
    )

    # Store additional metrics
    trial.set_user_attr("evaluation_time", results["evaluation_time"])
    trial.set_user_attr("samples_per_second", results["samples_per_second"])
    trial.set_user_attr("model_size_gb", results["model_size_gb"])

    # Return both objectives: maximize MAP@3, minimize time
    # Optuna minimizes by default, so negate MAP@3
    return -results["map_at_3"], results["evaluation_time"]


def run_comparison(n_trials: int | None = None, sample_size: int = 100) -> optuna.Study:
    """Run the quantization comparison study.

    Args:
        n_trials: Number of trials (default: test all quantizations)
        sample_size: Number of samples to evaluate per trial

    Returns:
        Completed Optuna study
    """
    # If n_trials not specified, test all quantizations once
    if n_trials is None:
        n_trials = len(QUANTIZATION_OPTIONS)

    # Create multi-objective study
    study = optuna.create_study(
        study_name=f"llm_quantization_{time.strftime('%Y%m%d_%H%M%S')}",
        directions=["maximize", "minimize"],  # maximize MAP@3, minimize time
        storage=STORAGE_URL,
        load_if_exists=False,
    )

    logger.info(f"Starting quantization comparison with {n_trials} trials")

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Save results
    save_results(study)

    return study


def save_results(study: optuna.Study) -> Path:
    """Save study results to CSV.

    Args:
        study: Completed Optuna study

    Returns:
        Path to saved results file
    """
    # Collect all trials
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                "quantization": trial.params["quantization"],
                "model_size_gb": trial.user_attrs["model_size_gb"],
                "map_at_3": -trial.values[0],  # Negate back to positive
                "evaluation_time": trial.values[1],
                "samples_per_second": trial.user_attrs["samples_per_second"],
            })

    # Sort by MAP@3 descending
    results.sort(key=lambda x: x["map_at_3"], reverse=True)

    # Save to CSV
    df = pd.DataFrame(results)
    output_path = Path("results/llm_quantization_comparison.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n=== Quantization Comparison Results ===")
    print(df.to_string(index=False))

    # Find Pareto optimal solutions
    print("\n=== Pareto Optimal Solutions ===")
    pareto_front = study.best_trials
    for trial in pareto_front[:3]:  # Show top 3 Pareto optimal
        print(
            f"  • {trial.params['quantization']}: "
            f"MAP@3={-trial.values[0]:.4f}, "
            f"Time={trial.values[1]:.2f}s"
        )

    # Best trade-off recommendation
    if results:
        q4_k_m = next((r for r in results if r["quantization"] == "Q4_K_M"), None)
        if q4_k_m:
            print("\n✓ Recommended: Q4_K_M")
            print(f"  Good balance of quality ({q4_k_m['map_at_3']:.4f} MAP@3) "
                  f"and speed ({q4_k_m['samples_per_second']:.2f} samples/s)")

    return output_path


@click.group()
def cli() -> None:
    """LLM quantization optimization commands."""


@click.command()
@click.option("--trials", type=int, help="Number of trials (default: test all quantizations)")
@click.option("--sample-size", type=int, default=100, help="Number of samples to evaluate")
def compare(trials: int | None, sample_size: int) -> None:
    """Compare LLM quantization options."""
    study = run_comparison(n_trials=trials, sample_size=sample_size)
    logger.success(f"Quantization comparison complete! Study: {study.study_name}")


# Add commands to CLI
cli.add_command(compare)


def main() -> None:
    """Entry point for LLM optimization CLI."""
    cli()


if __name__ == "__main__":
    main()

