import time
from pathlib import Path

import click
import optuna
import pandas as pd
from huggingface_hub import hf_hub_download
from loguru import logger

from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    parse_training_data,
)
from kaggle_map.optimise.utils import STORAGE_URL
from kaggle_map.strategies.llm import QUANTIZATION_OPTIONS, LLMStrategy, QuantizationType
from kaggle_map.strategies.utils import split_training_data

# Model base URL on HuggingFace
MODEL_BASE = "unsloth/gemma-3-12b-it-GGUF"


def download_model_if_needed(quantization: QuantizationType) -> Path:
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


def evaluate_quantization(quantization: QuantizationType, sample_size: int = 100) -> dict:
    logger.info(f"Evaluating quantization: {quantization}")

    # Download model if needed
    model_path = download_model_if_needed(quantization)


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
        "model_size_gb": QUANTIZATION_OPTIONS[quantization],
    }


def objective(trial: optuna.Trial) -> tuple[float, float]:
    quantization = trial.suggest_categorical("quantization", list(QUANTIZATION_OPTIONS.keys()))

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


def run_comparison(n_trials: int= len(QUANTIZATION_OPTIONS)) -> optuna.Study:
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
    # Collect all trials
    results = [
        {
            "quantization": trial.params["quantization"],
            "model_size_gb": trial.user_attrs["model_size_gb"],
            "map_at_3": -trial.values[0] if trial.values else 0.0,  # Negate back to positive
            "evaluation_time": trial.values[1] if trial.values else 0.0,
            "samples_per_second": trial.user_attrs["samples_per_second"],
        }
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
    ]

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
        # Check if trial has values (some trials might not have been completed)
        if trial.values is not None:
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
    pass


@click.command()
@click.option("--trials", type=int, help="Number of trials (default: test all quantizations)", default=len(QUANTIZATION_OPTIONS))
def compare(trials: int) -> None:
    study = run_comparison(n_trials=trials)
    logger.success(f"Quantization comparison complete! Study: {study.study_name}")


# Add commands to CLI
cli.add_command(compare)

if __name__ == "__main__":
    cli()

