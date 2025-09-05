# ruff: noqa: PD011  # Optuna Trial.values is not a pandas operation

import time
from pathlib import Path
from typing import cast

import optuna
import pandas as pd
from loguru import logger

from kaggle_map.core.models import GGUF_MODELS, MODEL_OPTIONS, LLMModelLoadConfig, ModelName, QuantizationLevel
from kaggle_map.optimise.utils import STORAGE_URL
from kaggle_map.strategies.llm import LLMStrategy


def evaluate_gguf_models(model_name: ModelName, quantization: QuantizationLevel) -> dict:
    logger.info(f"Evaluating model: {model_name}, quantization: {quantization}")

    # Create strategy with specific model and quantization
    assert model_name in MODEL_OPTIONS, f"Invalid model name: {model_name}"
    assert quantization in GGUF_MODELS[model_name].available_quantizations, (
        f"Invalid quantization {quantization} for model {model_name}"
    )
    config = LLMModelLoadConfig(model_name=model_name, quantization=quantization)

    # NOTE: Unconventional use of train_split=1.0 for both fit and eval
    # This is intentional for LLM strategy optimization:
    # - fit() with train_split=1.0: Extracts correct answers and misconceptions from ALL data
    #   (LLM doesn't train, just builds a knowledge base from the training set)
    # - evaluate_on_split() with train_split=0.0: Tests on ALL data to maximize signal
    #   for comparing different quantization levels (we're not validating generalization)

    # Fit the strategy with the config (only to extract stats from training data)
    strategy = LLMStrategy.fit(
        train_split=1.0,
        config=config,
    )

    # Measure evaluation time using all training data
    start_time = time.time()

    # Use evaluate_on_split to evaluate on all data
    results = LLMStrategy.evaluate_on_split(
        model=strategy,
        train_split=0.0,  # Use all data for evaluation
    )
    evaluation_time = time.time() - start_time

    return {
        "model_name": model_name,
        "quantization": quantization,
        "map_at_3": results["map_at_3"],
        "evaluation_time": evaluation_time,
    }


def objective(trial: optuna.Trial) -> tuple[float, float]:
    model_name: ModelName = cast("ModelName", trial.suggest_categorical("model_name", MODEL_OPTIONS))
    available_quantizations = GGUF_MODELS[model_name].available_quantizations
    quantization = cast("QuantizationLevel", trial.suggest_categorical("quantization", available_quantizations))

    results = evaluate_gguf_models(model_name, quantization)
    logger.info(
        f"Model: {model_name} | "
        f"Quantization: {quantization} | "
        f"MAP@3: {results['map_at_3']:.4f} | "
        f"Time: {results['evaluation_time']:.2f}s"
    )

    # Store additional metrics
    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("evaluation_time", results["evaluation_time"])

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


if __name__ == "__main__":
    study = run_comparison(n_trials=100)
    logger.success(f"Quantization comparison complete! Study: {study.study_name}")
