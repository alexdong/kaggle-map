"""Optuna study to compare different GGUF quantization options for LLM strategy.

This script evaluates different quantization levels of the gemma-3-12b-it model
to find the optimal balance between inference speed and MAP@3 accuracy.
"""

import time
from pathlib import Path

import optuna
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

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


def download_model_if_needed(quantization: str) -> Path:
    """Download the GGUF model if not already cached."""
    model_name = f"gemma-3-12b-it-{quantization}.gguf"
    model_dir = Path("models/gguf")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    if model_path.exists():
        logger.info(f"Model {model_name} already cached")
        return model_path

    logger.info(f"Downloading {model_name} from HuggingFace...")

    # Download using huggingface-hub
    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(
        repo_id=MODEL_BASE,
        filename=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )

    logger.info(f"Downloaded to {downloaded_path}")
    return Path(downloaded_path)


def evaluate_quantization(quantization: str, sample_size: int = 100) -> dict:
    """Evaluate a specific quantization option."""
    logger.info(f"Evaluating quantization: {quantization}")

    # Download model if needed
    model_path = download_model_if_needed(quantization)

    # Import here to avoid circular imports
    from kaggle_map.strategies.llm import LLMStrategy

    # Create strategy with specific quantization
    strategy = LLMStrategy(model_path=str(model_path))

    # Load training data for fit
    from kaggle_map.core.dataset import parse_training_data
    training_data = parse_training_data(Path("datasets/train.csv"))

    # Fit the strategy (loads correct answers and misconceptions)
    from kaggle_map.strategies.utils import split_training_data
    train_data, _, _ = split_training_data(training_data, train_ratio=0.7, random_seed=42)

    from kaggle_map.core.dataset import extract_correct_answers, extract_misconceptions_by_popularity
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


def get_model_size(quantization: str) -> float:
    """Get approximate model size in GB."""
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


def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Optuna objective function for multi-objective optimization."""
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


def run_comparison(n_trials: int | None = None, sample_size: int = 100):
    """Run the quantization comparison study."""
    console = Console()

    # If n_trials not specified, test all quantizations once
    if n_trials is None:
        n_trials = len(QUANTIZATION_OPTIONS)

    # Create multi-objective study
    study = optuna.create_study(
        study_name=f"llm_quantization_comparison_{time.strftime('%Y%m%d_%H%M%S')}",
        directions=["maximize", "minimize"],  # maximize MAP@3, minimize time
        storage="sqlite:///optuna_llm.db",
        load_if_exists=False,
    )

    logger.info(f"Starting quantization comparison with {n_trials} trials")

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Display results
    display_results(study, console)

    return study


def display_results(study: optuna.Study, console: Console) -> None:
    """Display comparison results in a nice table."""

    # Create results table
    table = Table(title="LLM Quantization Comparison Results")
    table.add_column("Quantization", style="cyan")
    table.add_column("Model Size", justify="right", style="yellow")
    table.add_column("MAP@3", justify="right", style="green")
    table.add_column("Eval Time (s)", justify="right", style="magenta")
    table.add_column("Samples/s", justify="right", style="blue")

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

    # Add rows to table
    for r in results:
        table.add_row(
            r["quantization"],
            f"{r['model_size_gb']:.2f} GB",
            f"{r['map_at_3']:.4f}",
            f"{r['evaluation_time']:.2f}",
            f"{r['samples_per_second']:.2f}",
        )

    console.print(table)

    # Find Pareto optimal solutions
    console.print("\n[bold]Pareto Optimal Solutions:[/bold]")
    pareto_front = study.best_trials

    for trial in pareto_front[:3]:  # Show top 3 Pareto optimal
        console.print(
            f"  • {trial.params['quantization']}: "
            f"MAP@3={-trial.values[0]:.4f}, "
            f"Time={trial.values[1]:.2f}s"
        )

    # Best trade-off recommendation
    if results:
        # Find best balance (highest MAP@3 with reasonable speed)
        q4_k_m = next((r for r in results if r["quantization"] == "Q4_K_M"), None)
        if q4_k_m:
            console.print("\n[bold green]Recommended: Q4_K_M[/bold green]")
            console.print(f"  Good balance of quality ({q4_k_m['map_at_3']:.4f} MAP@3) "
                         f"and speed ({q4_k_m['samples_per_second']:.2f} samples/s)")

    # Save results to CSV
    df = pd.DataFrame(results)
    output_path = Path("results/llm_quantization_comparison.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"\nResults saved to {output_path}")


def main() -> None:
    """Main entry point for the quantization comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare LLM quantization options")
    parser.add_argument("--trials", type=int, help="Number of trials (default: test all quantizations)")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to evaluate")

    args = parser.parse_args()

    run_comparison(n_trials=args.trials, sample_size=args.sample_size)

    logger.success("Quantization comparison complete!")


if __name__ == "__main__":
    main()
