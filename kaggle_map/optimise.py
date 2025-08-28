"""Hyperparameter optimization for kaggle-map prediction strategies."""

import json
from datetime import datetime
from pathlib import Path

import click
import optuna
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

from .strategies import get_strategy


class OptimiseManager:
    """Manager for hyperparameter optimization and analysis."""

    def __init__(self, storage_url: str = "sqlite:///optuna.db") -> None:
        self.storage = storage_url

    def create_study(self, strategy_name: str) -> optuna.Study:
        """Create or load an Optuna study for hyperparameter optimization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{strategy_name}_{timestamp}"

        logger.info(f"Creating study: {study_name}")

        return optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1,
            ),
        )


    def objective_function(self, trial: optuna.Trial, strategy_class: type) -> float:
        """Objective function for hyperparameter optimization."""

        # Get hyperparameters from strategy
        hyperparams = strategy_class.get_hyperparameter_search_space(trial)

        # Add trial information to wandb run name
        trial_info = f"trial_{trial.number}"
        key_params = []

        # Include key parameters in run name for easy identification
        if "learning_rate" in hyperparams:
            key_params.append(f"lr_{hyperparams['learning_rate']:.1e}")
        if "batch_size" in hyperparams:
            key_params.append(f"bs_{hyperparams['batch_size']}")
        if "dropout" in hyperparams:
            key_params.append(f"do_{hyperparams['dropout']:.2f}")
        if "architecture_size" in hyperparams:
            key_params.append(f"arch_{hyperparams['architecture_size']}")

        wandb_run_name = f"hypersearch_{trial_info}_{'_'.join(key_params)}"
        hyperparams["wandb_run_name"] = wandb_run_name

        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Starting trial {trial.number} with params: {hyperparams}")

        # Handle OOM gracefully but let other errors crash
        try:
            model = strategy_class.fit(**hyperparams)
            result = strategy_class.evaluate_on_split(model)

            # Track GPU utilization for analysis
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                trial.set_user_attr("peak_gpu_memory_gb", peak_memory)
                logger.info(f"Trial {trial.number} peak GPU memory: {peak_memory:.2f}GB")

            map_score = result["validation_map@3"]
            logger.info(f"Trial {trial.number} completed: MAP@3={map_score:.4f}")

            return map_score

        except torch.cuda.OutOfMemoryError as e:
            # Special handling for OOM - log details and return poor score
            if torch.cuda.is_available():
                logger.error(f"Trial {trial.number} OOM: {e}")
                logger.error(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                logger.error(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                logger.error(f"Trial params causing OOM: {hyperparams}")
                torch.cuda.empty_cache()

            trial.set_user_attr("oom_error", True)
            trial.set_user_attr("oom_details", str(e))

            # Return very poor score to avoid this configuration
            return 0.0

    def run_search(
        self,
        strategy_name: str,
        n_trials: int,
        n_jobs: int,
        timeout: int | None = None
    ) -> optuna.Study:
        """Run hyperparameter search for a strategy."""

        logger.info(f"Starting hyperparameter search for {strategy_name}")
        logger.info(f"Trials: {n_trials}, Jobs: {n_jobs}, Timeout: {timeout}s")

        strategy_class = get_strategy(strategy_name)
        study = self.create_study(strategy_name)

        # Create objective with bound strategy class
        def objective(trial: optuna.Trial) -> float:
            return self.objective_function(trial, strategy_class)

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
        best_config_path = Path(f"models/{strategy_name}_best_config.json")
        best_config_path.parent.mkdir(parents=True, exist_ok=True)

        with best_config_path.open("w") as f:
            json.dump({
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            }, f, indent=2)

        logger.info(f"Best configuration saved to {best_config_path}")

        return study

    def list_studies(self) -> None:
        """List all available optimization studies."""
        storage = optuna.storages.RDBStorage(url=self.storage)
        study_summaries = storage.get_all_studies()

        if not study_summaries:
            logger.info("No optimization studies found")
            return

        console = Console()
        table = Table(title="Optimization Studies")
        table.add_column("Study Name", style="cyan")
        table.add_column("Trials", style="yellow")
        table.add_column("Best Value", style="green")
        table.add_column("Status", style="magenta")

        for summary in sorted(study_summaries, key=lambda x: x.study_name):
            study = optuna.load_study(
                study_name=summary.study_name,
                storage=self.storage
            )

            completed_trials = [t for t in study.trials if t.value is not None]
            status = "COMPLETED" if completed_trials else "EMPTY"

            # Safely get best value
            best_value = f"{max(t.value for t in completed_trials):.4f}" if completed_trials else "N/A"

            table.add_row(
                summary.study_name,
                str(len(study.trials)),
                best_value,
                status
            )

        console.print(table)

    def compare_studies(self, study_names: list[str]) -> None:
        """Compare multiple optimization studies."""
        console = Console()

        if len(study_names) < 2:
            console.print("[red]Need at least 2 studies to compare[/red]")
            return

        table = Table(title="Study Comparison")
        table.add_column("Study", style="cyan")
        table.add_column("Best MAP@3", style="green")
        table.add_column("Trials", style="yellow")
        table.add_column("Best Params", style="magenta")

        studies = []
        for name in study_names:
            study = optuna.load_study(study_name=name, storage=self.storage)
            studies.append(study)

            # Format key parameters for display
            best_params = study.best_params
            key_param_str = ", ".join([
                f"lr={best_params.get('learning_rate', 'N/A'):.1e}",
                f"bs={best_params.get('batch_size', 'N/A')}",
                f"do={best_params.get('dropout', 'N/A'):.2f}",
                f"arch={best_params.get('architecture_size', 'N/A')}"
            ])

            table.add_row(
                name,
                f"{study.best_value:.4f}" if study.best_value else "N/A",
                str(len(study.trials)),
                key_param_str
            )

        console.print(table)

        # Show best overall
        best_study = max(studies, key=lambda s: s.best_value or 0)
        console.print(f"\n[green]Best Study: {best_study.study_name}[/green]")
        console.print(f"[green]Best MAP@3: {best_study.best_value:.4f}[/green]")

    def analyze_study(self, study_name: str) -> None:
        """Analyze a single optimization study in detail."""
        console = Console()

        study = optuna.load_study(study_name=study_name, storage=self.storage)

        if len(study.trials) == 0:
            console.print(f"[red]Study {study_name} has no trials[/red]")
            return

        # Study overview
        console.print(f"[bold]Study Analysis: {study_name}[/bold]")
        console.print(f"Total Trials: {len(study.trials)}")
        console.print(f"Best Value: {study.best_value:.4f}")
        console.print(f"Best Trial: #{study.best_trial.number}")

        # Best parameters
        console.print("\n[bold]Best Parameters:[/bold]")
        for param, value in study.best_params.items():
            console.print(f"  {param}: {value}")

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            console.print("\n[bold]Parameter Importance:[/bold]")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  {param}: {imp:.3f}")
        except Exception:
            console.print("\n[yellow]Parameter importance analysis unavailable[/yellow]")

        # Trial statistics
        completed_trials = [t for t in study.trials if t.value is not None]
        if completed_trials:
            values = [t.value for t in completed_trials]
            console.print("\n[bold]Trial Statistics:[/bold]")
            console.print(f"  Completed: {len(completed_trials)}/{len(study.trials)}")
            console.print(f"  Mean: {sum(values)/len(values):.4f}")
            console.print(f"  Std: {(sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5:.4f}")
            console.print(f"  Min: {min(values):.4f}")
            console.print(f"  Max: {max(values):.4f}")

        # OOM analysis
        oom_trials = [t for t in study.trials if t.user_attrs.get("oom_error", False)]
        if oom_trials:
            console.print(f"\n[red]OOM Trials: {len(oom_trials)}[/red]")
            for trial in oom_trials[:3]:  # Show first 3 OOM trials
                batch_size = trial.params.get("batch_size", "N/A")
                arch_size = trial.params.get("architecture_size", "N/A")
                console.print(f"  Trial #{trial.number}: batch_size={batch_size}, arch={arch_size}")


@click.group()
def cli() -> None:
    """Hyperparameter optimization for kaggle-map strategies."""


@click.command()
@click.argument("strategy")
@click.option("--trials", default=100, help="Number of trials to run")
@click.option("--jobs", default=3, help="Number of parallel jobs")
@click.option("--timeout", default=None, type=int, help="Timeout in seconds")
def search(strategy: str, trials: int, jobs: int, timeout: int | None) -> None:
    """Run hyperparameter search for a strategy."""
    manager = OptimiseManager()
    study = manager.run_search(strategy, trials, jobs, timeout)

    print("\nSearch completed!")
    print(f"Study name: {study.study_name}")
    print(f"Best MAP@3: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


@click.command("list-studies")
def list_studies_cmd() -> None:
    """List all optimization studies."""
    manager = OptimiseManager()
    manager.list_studies()


@click.command()
@click.argument("studies", nargs=-1, required=True)
def compare(studies: tuple[str, ...]) -> None:
    """Compare multiple optimization studies."""
    manager = OptimiseManager()
    manager.compare_studies(list(studies))


@click.command()
@click.argument("study")
def analyze(study: str) -> None:
    """Analyze a single optimization study."""
    manager = OptimiseManager()
    manager.analyze_study(study)


# Add commands to CLI
cli.add_command(search)
cli.add_command(list_studies_cmd)
cli.add_command(compare)
cli.add_command(analyze)


def main() -> None:
    """Entry point for the optimise CLI."""
    cli()


if __name__ == "__main__":
    main()
