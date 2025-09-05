# ruff: noqa: PD011  # Optuna Trial.values is not a pandas operation

import time
from pathlib import Path
from typing import cast

import click
import optuna
import pandas as pd
from loguru import logger

from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    parse_training_data,
)
from kaggle_map.core.models import GGUF_MODELS, MODEL_OPTIONS, LLMModelLoadConfig, ModelName, QuantizationLevel
from kaggle_map.optimise.utils import STORAGE_URL
from kaggle_map.strategies.llm import LLMStrategy
from kaggle_map.strategies.utils import split_training_data


def evaluate_quantization(model_name: ModelName, quantization: QuantizationLevel, sample_size: int = 100) -> dict:
    logger.info(f"Evaluating model: {model_name}, quantization: {quantization}")

    # Create strategy with specific model and quantization
    config = LLMModelLoadConfig(model_name=model_name, quantization=quantization)
    strategy = LLMStrategy(config=config)

    # Load training data for fit
    training_data = parse_training_data(Path("datasets/train.csv"))

    # Fit the strategy (loads correct answers and misconceptions)
    train_data, _, _ = split_training_data(training_data, train_ratio=0.7, random_seed=42)

    # Build problem contexts
    correct_answers = extract_correct_answers(train_data)
    misconceptions_by_question = extract_misconceptions_by_popularity(train_data)

    # Context is now stored directly as dict

    for question_id in correct_answers:
        strategy.question_contexts[question_id] = {
            "correct_answer": correct_answers[question_id],
            "known_misconceptions": misconceptions_by_question.get(question_id, []),
        }

    # Measure evaluation time
    start_time = time.time()

    # Evaluate on validation split
    results = strategy.evaluate_on_split(model=strategy, train_split=0.7, random_seed=42, sample_size=sample_size)

    evaluation_time = time.time() - start_time

    # Clean up model from memory
    del strategy

    return {
        "model_name": model_name,
        "quantization": quantization,
        "map_at_3": results["map_at_3"],
        "evaluation_time": evaluation_time,
        "samples_per_second": sample_size / evaluation_time,
    }


def objective(trial: optuna.Trial) -> tuple[float, float]:
    model_name: ModelName = cast("ModelName", trial.suggest_categorical("model_name", MODEL_OPTIONS))
    available_quantizations = GGUF_MODELS[model_name].available_quantizations
    quantization = cast("QuantizationLevel", trial.suggest_categorical("quantization", available_quantizations))

    results = evaluate_quantization(model_name, quantization, sample_size=100)
    logger.info(
        f"Model: {model_name} | "
        f"Quantization: {quantization} | "
        f"MAP@3: {results['map_at_3']:.4f} | "
        f"Time: {results['evaluation_time']:.2f}s | "
        f"Speed: {results['samples_per_second']:.2f} samples/s"
    )

    # Store additional metrics
    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("evaluation_time", results["evaluation_time"])
    trial.set_user_attr("samples_per_second", results["samples_per_second"])

    # Return both objectives: maximize MAP@3, minimize time
    # Optuna minimizes by default, so negate MAP@3
    return -results["map_at_3"], results["evaluation_time"]


def save_results(study: optuna.Study) -> Path:
    # Collect all trials
    results = [
        {
            "model_name": trial.params["model_name"],
            "quantization": trial.params["quantization"],
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
    results_df = pd.DataFrame(results)
    output_path = Path("results/llm_model_quantization_comparison.csv")
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n=== Quantization Comparison Results ===")
    print(results_df.to_string(index=False))

    # Find Pareto optimal solutions
    print("\n=== Pareto Optimal Solutions ===")
    pareto_front = study.best_trials
    for trial in pareto_front[:5]:  # Show top 5 Pareto optimal
        # Check if trial has values (some trials might not have been completed)
        if trial.values is not None:
            print(
                f"  • {trial.params['model_name']}-{trial.params['quantization']}: "
                f"MAP@3={-trial.values[0]:.4f}, Time={trial.values[1]:.2f}s"
            )

    return output_path


def run_comparison(n_trials: int) -> optuna.Study:
    # Calculate total combinations if n_trials not specified
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


@click.command()
@click.option(
    "--trials", type=int, help="Number of trials (default: test all model+quantization combinations)", default=100
)
def compare(trials: int) -> None:
    study = run_comparison(n_trials=trials)
    logger.success(f"Quantization comparison complete! Study: {study.study_name}")
