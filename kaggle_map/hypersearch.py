"""Hyperparameter search system for strategy optimization."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import optuna
from loguru import logger
from optuna.trial import Trial
from rich.console import Console
from rich.table import Table

from .strategies import get_strategy
from .strategies.base import Strategy


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""

    strategy: str
    n_trials: int
    n_jobs: int
    study_name: str | None = None
    storage: str | None = None
    direction: str = "maximize"
    metric: str = "accuracy"
    timeout: int | None = None
    pruner: str = "median"
    sampler: str = "tpe"


class HyperparameterSearch:
    """Manages hyperparameter optimization for strategies."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.strategy_class = get_strategy(config.strategy)
        self.console = Console()

        # Create results directory
        self.results_dir = Path("hypersearch_results")
        self.results_dir.mkdir(exist_ok=True)

        # Study name with timestamp if not provided
        if not config.study_name:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            config.study_name = f"{config.strategy}_{timestamp}"

        self.study_path = self.results_dir / f"{config.study_name}.json"
        self.best_params_path = self.results_dir / f"{config.study_name}_best.json"

    def get_search_space(self, trial: Trial) -> dict[str, Any]:
        """Define search space based on strategy type."""
        # Use strategy's own search space definition
        search_space = self.strategy_class.get_hyperparameter_search_space(trial)

        # If strategy doesn't define search space, try some defaults
        if not search_space:
            logger.warning(f"Strategy {self.config.strategy} has no search space defined")
            # Try some generic defaults based on strategy name
            if self.config.strategy == "baseline":
                return {
                    "smoothing": trial.suggest_float("smoothing", 0.0, 1.0, step=0.1),
                    "min_count": trial.suggest_int("min_count", 1, 10),
                }
            return {
                "param1": trial.suggest_float("param1", 0.0, 1.0),
                "param2": trial.suggest_int("param2", 1, 100),
            }

        return search_space

    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        params = self.get_search_space(trial)

        logger.info(f"Trial {trial.number}: Testing params {params}")

        try:
            # Train model with suggested parameters
            model = self.train_with_params(params)

            # Evaluate model
            eval_results = self.strategy_class.evaluate_on_split(model)

            # Get metric value
            metric_value = eval_results.get(self.config.metric, 0.0)

            # Log result
            logger.info(f"Trial {trial.number}: {self.config.metric}={metric_value:.4f}")

            # Report intermediate value for pruning
            trial.report(metric_value, 0)

            # Save trial results
            self.save_trial_result(trial.number, params, eval_results)

            return metric_value

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst possible value on failure

    def train_with_params(self, params: dict[str, Any]) -> Strategy:
        """Train model with specific hyperparameters."""
        # Let strategy handle its own config creation
        config_or_params = self.strategy_class.create_config_from_hyperparams(
            params,
            epochs=50,  # Use fewer epochs for search
            early_stopping_patience=5,
        )

        # If the strategy returns a config object, pass it as config
        # Otherwise pass as kwargs
        if isinstance(config_or_params, dict):
            return self.strategy_class.fit(**config_or_params)
        return self.strategy_class.fit(config=config_or_params)

    def save_trial_result(self, trial_number: int, params: dict, results: dict) -> None:
        """Save individual trial results."""
        trial_file = self.results_dir / f"{self.config.study_name}_trial_{trial_number}.json"

        trial_data = {
            "trial_number": trial_number,
            "timestamp": datetime.now(UTC).isoformat(),
            "params": params,
            "results": results,
        }

        with open(trial_file, "w") as f:
            json.dump(trial_data, f, indent=2)

    def run(self) -> None:
        """Execute hyperparameter search."""
        # Configure pruner
        pruner = None
        if self.config.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        elif self.config.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()

        # Configure sampler
        sampler = None
        if self.config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif self.config.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.config.sampler == "grid":
            # Note: Grid search requires explicit search space
            sampler = optuna.samplers.GridSampler(self.get_grid_search_space())

        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction=self.config.direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )

        # Add callbacks
        def log_best(study, trial) -> None:
            logger.info(f"Best trial so far: {study.best_value:.4f}")

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            callbacks=[log_best],
            show_progress_bar=True,
        )

        # Save results
        self.save_results(study)

        # Display results
        self.display_results(study)

    def save_results(self, study: optuna.Study) -> None:
        """Save study results to file."""
        # Save best parameters
        best_params = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        with open(self.best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)

        # Save full study history
        history = []
        for trial in study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            })

        with open(self.study_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.success(f"Results saved to {self.results_dir}")

    def display_results(self, study: optuna.Study) -> None:
        """Display search results in a nice table."""
        table = Table(title=f"Hyperparameter Search Results: {self.config.strategy}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Best Value", f"{study.best_value:.4f}")
        table.add_row("Best Trial", str(study.best_trial.number))
        table.add_row("Total Trials", str(len(study.trials)))
        table.add_row("Completed Trials", str(len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])))

        self.console.print(table)

        # Display best parameters
        params_table = Table(title="Best Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="magenta")

        for param, value in study.best_params.items():
            params_table.add_row(param, str(value))

        self.console.print(params_table)

    def get_grid_search_space(self) -> dict[str, list[Any]]:
        """Define grid search space for systematic exploration."""
        # Check if strategy defines a grid search space
        if hasattr(self.strategy_class, "get_grid_search_space"):
            return self.strategy_class.get_grid_search_space()

        # Default grid spaces for known strategies
        if self.config.strategy == "mlp":
            return {
                "hidden_dim": [256, 512, 768],
                "n_layers": [3, 4, 5],
                "dropout": [0.2, 0.3, 0.4],
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "batch_size": [32, 64],
            }

        logger.warning(f"No grid search space defined for {self.config.strategy}")
        return {}


@click.command()
@click.argument("strategy", type=str)
@click.option("--n-trials", default=100, help="Number of trials to run")
@click.option("--n-jobs", default=1, help="Number of parallel jobs")
@click.option("--metric", default="accuracy", help="Metric to optimize")
@click.option("--direction", default="maximize", type=click.Choice(["maximize", "minimize"]))
@click.option("--timeout", type=int, help="Timeout in seconds")
@click.option("--study-name", type=str, help="Name for the study")
@click.option("--pruner", default="median", type=click.Choice(["median", "hyperband", "none"]))
@click.option("--sampler", default="tpe", type=click.Choice(["tpe", "random", "grid"]))
def search_cli(
    strategy: str,
    n_trials: int,
    n_jobs: int,
    metric: str,
    direction: str,
    timeout: int | None,
    study_name: str | None,
    pruner: str,
    sampler: str,
) -> None:
    """Run hyperparameter search for a strategy."""
    config = SearchConfig(
        strategy=strategy,
        n_trials=n_trials,
        n_jobs=n_jobs,
        metric=metric,
        direction=direction,
        timeout=timeout,
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
    )

    search = HyperparameterSearch(config)
    search.run()


if __name__ == "__main__":
    search_cli()
