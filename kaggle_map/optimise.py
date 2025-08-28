"""Hyperparameter optimization for kaggle-map prediction strategies."""

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
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


    def objective_function(self, trial: optuna.Trial, strategy_class: type, train_data_path: str | None = None) -> float:
        """Objective function for hyperparameter optimization."""

        # Get hyperparameters from strategy
        hyperparams = strategy_class.get_hyperparameter_search_space(trial)

        # Add train_csv_path if provided
        if train_data_path:
            from pathlib import Path
            hyperparams["train_csv_path"] = Path(train_data_path)

        # Add trial information to wandb run name
        trial_num = trial.number
        trial_info = f"trial_{trial_num}"
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

        # Add metadata for wandb tracking
        hyperparams["wandb_tags"] = [f"study_{trial.study.study_name}", f"trial_{trial_num}", "4hour_focused"]
        hyperparams["study_id"] = trial.study.study_name
        hyperparams["trial_number"] = trial_num

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

        except Exception as e:
            # Log any other exceptions and ensure cleanup
            logger.error(f"Trial {trial.number} failed with error: {e}")
            raise

        finally:
            # Always ensure wandb is properly closed to free file handles
            import wandb
            if wandb.run is not None:
                wandb.finish()

            # Clear GPU memory after each trial
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_search(
        self,
        strategy_name: str,
        n_trials: int,
        n_jobs: int,
        timeout: int | None = None,
        train_data_path: str | None = None
    ) -> optuna.Study:
        """Run hyperparameter search for a strategy."""

        logger.info(f"Starting hyperparameter search for {strategy_name}")
        logger.info(f"Trials: {n_trials}, Jobs: {n_jobs}, Timeout: {timeout}s")
        if train_data_path:
            logger.info(f"Using training data: {train_data_path}")

        strategy_class = get_strategy(strategy_name)
        study = self.create_study(strategy_name)

        # Create objective with bound strategy class and train data path
        def objective(trial: optuna.Trial) -> float:
            return self.objective_function(trial, strategy_class, train_data_path)

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
        table.add_column("Progress", style="blue")

        for summary in sorted(study_summaries, key=lambda x: x.study_name):
            study = optuna.load_study(
                study_name=summary.study_name,
                storage=self.storage
            )

            # Check trial states properly
            running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

            # Determine status
            if running_trials:
                status = "RUNNING"
                # Create progress bar for running studies
                total_trials = len(study.trials)
                completed_count = len(completed_trials)
                progress_bar = self._create_progress_bar(completed_count, total_trials)
            elif completed_trials:
                status = "COMPLETED"
                progress_bar = "✓ Done"
            else:
                status = "EMPTY"
                progress_bar = "—"

            # Safely get best value from completed trials
            best_value = f"{max(t.value for t in completed_trials):.4f}" if completed_trials else "N/A"

            table.add_row(
                summary.study_name,
                str(len(study.trials)),
                best_value,
                status,
                progress_bar
            )

        console.print(table)

    def _create_progress_bar(self, completed: int, total: int) -> str:
        """Create a text-based progress bar for running studies."""
        if total == 0:
            return "—"

        # Calculate percentage
        percentage = (completed / total) * 100

        # Create a simple text progress bar
        bar_width = 20
        filled = int((completed / total) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        return f"{bar} {completed}/{total} ({percentage:.1f}%)"

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

    def generate_study_summary(self, study_name: str) -> None:
        """Generate a markdown summary of the optimization study."""
        study = optuna.load_study(study_name=study_name, storage=self.storage)

        if len(study.trials) == 0:
            logger.warning(f"Study {study_name} has no trials, skipping summary")
            return

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Generate summary file
        summary_path = logs_dir / f"{study_name}.md"

        with summary_path.open("w") as f:
            f.write(f"# Optimization Study: {study_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## Study Overview\n\n")
            f.write(f"- **Total Trials**: {len(study.trials)}\n")
            f.write(f"- **Best MAP@3**: {study.best_value:.4f}\n")
            f.write(f"- **Best Trial**: #{study.best_trial.number}\n\n")

            # Best parameters
            f.write("## Best Parameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for param, value in sorted(study.best_params.items()):
                if isinstance(value, float):
                    if value < 0.01:
                        f.write(f"| {param} | {value:.2e} |\n")
                    else:
                        f.write(f"| {param} | {value:.4f} |\n")
                else:
                    f.write(f"| {param} | {value} |\n")

            # Parameter importance
            f.write("\n## Parameter Importance\n\n")
            try:
                importance = optuna.importance.get_param_importances(study)
                f.write("| Parameter | Importance |\n")
                f.write("|-----------|------------|\n")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {param} | {imp:.3f} |\n")
            except Exception:
                f.write("*Parameter importance analysis unavailable*\n")

            # Trial statistics
            completed_trials = [t for t in study.trials if t.value is not None]
            if completed_trials:
                values = [t.value for t in completed_trials]
                f.write("\n## Trial Statistics\n\n")
                f.write(f"- **Completed**: {len(completed_trials)}/{len(study.trials)}\n")
                f.write(f"- **Mean MAP@3**: {np.mean(values):.4f}\n")
                f.write(f"- **Std Dev**: {np.std(values):.4f}\n")
                f.write(f"- **Min**: {min(values):.4f}\n")
                f.write(f"- **Max**: {max(values):.4f}\n")

            # Performance analysis by key parameters
            f.write("\n## Performance Analysis\n\n")

            # Architecture analysis
            arch_performance = {}
            for trial in completed_trials:
                if "architecture_size" in trial.params:
                    arch = trial.params["architecture_size"]
                    if arch not in arch_performance:
                        arch_performance[arch] = []
                    arch_performance[arch].append(trial.value)

            if arch_performance:
                f.write("### By Architecture Size\n\n")
                f.write("| Architecture | Mean MAP@3 | Std Dev | Count | Max |\n")
                f.write("|--------------|------------|---------|-------|-----|\n")
                for arch, vals in sorted(arch_performance.items()):
                    vals_array = np.array(vals)
                    f.write(f"| {arch} | {vals_array.mean():.4f} | {vals_array.std():.4f} | {len(vals)} | {vals_array.max():.4f} |\n")

            # Learning rate analysis
            lr_ranges = {
                "very_low (<1e-4)": (0, 1e-4),
                "low (1e-4 to 1e-3)": (1e-4, 1e-3),
                "medium (1e-3 to 5e-3)": (1e-3, 5e-3),
                "high (>5e-3)": (5e-3, 1)
            }
            lr_performance = {k: [] for k in lr_ranges}

            for trial in completed_trials:
                if "learning_rate" in trial.params:
                    lr = trial.params["learning_rate"]
                    for range_name, (low, high) in lr_ranges.items():
                        if low <= lr < high:
                            lr_performance[range_name].append(trial.value)
                            break

            f.write("\n### By Learning Rate Range\n\n")
            f.write("| LR Range | Mean MAP@3 | Count | Max |\n")
            f.write("|----------|------------|-------|-----|\n")
            for range_name in ["very_low (<1e-4)", "low (1e-4 to 1e-3)", "medium (1e-3 to 5e-3)", "high (>5e-3)"]:
                vals = lr_performance[range_name]
                if vals:
                    vals_array = np.array(vals)
                    f.write(f"| {range_name} | {vals_array.mean():.4f} | {len(vals)} | {vals_array.max():.4f} |\n")
                else:
                    f.write(f"| {range_name} | - | 0 | - |\n")

            # Top trials
            f.write("\n## Top 5 Trials\n\n")
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]

            for _i, trial in enumerate(top_trials, 1):
                f.write(f"### Trial #{trial.number} (MAP@3: {trial.value:.4f})\n\n")
                f.write("```json\n")
                f.write(json.dumps(trial.params, indent=2))
                f.write("\n```\n\n")

            # Key insights
            f.write("## Key Insights\n\n")

            # Automatic insights based on data
            if arch_performance:
                best_arch = max(arch_performance.items(), key=lambda x: np.mean(x[1]))
                f.write(f"- **Best architecture**: {best_arch[0]} (mean MAP@3: {np.mean(best_arch[1]):.4f})\n")

            # Learning rate insight
            lr_with_data = [(k, v) for k, v in lr_performance.items() if v]
            if lr_with_data:
                best_lr_range = max(lr_with_data, key=lambda x: np.mean(x[1]))
                f.write(f"- **Optimal learning rate range**: {best_lr_range[0]} (mean MAP@3: {np.mean(best_lr_range[1]):.4f})\n")

            # Parameter correlation insights
            if study.best_params.get("dropout"):
                f.write(f"- **Best dropout**: {study.best_params['dropout']:.3f}\n")
            if study.best_params.get("batch_size"):
                f.write(f"- **Best batch size**: {study.best_params['batch_size']}\n")

            # OOM analysis
            oom_trials = [t for t in study.trials if t.user_attrs.get("oom_error", False)]
            if oom_trials:
                f.write("\n### Memory Issues\n")
                f.write(f"- **OOM Trials**: {len(oom_trials)} trials encountered out-of-memory errors\n")
                f.write("- **Common patterns**: ")
                batch_sizes = [t.params.get("batch_size", "N/A") for t in oom_trials[:3]]
                archs = [t.params.get("architecture_size", "N/A") for t in oom_trials[:3]]
                f.write(f"batch_sizes={batch_sizes}, architectures={archs}\n")

        logger.info(f"Generated study summary: {summary_path}")
        print(f"\n✅ Study summary saved to: {summary_path}")


@click.group()
def cli() -> None:
    """Hyperparameter optimization for kaggle-map strategies."""


@click.command()
@click.argument("strategy")
@click.option("--trials", default=100, help="Number of trials to run")
@click.option("--jobs", default=3, help="Number of parallel jobs")
@click.option("--timeout", default=None, type=int, help="Timeout in seconds")
@click.option("--train-data", default=None, help="Path to training data CSV")
def search(strategy: str, trials: int, jobs: int, timeout: int | None, train_data: str | None) -> None:
    """Run hyperparameter search for a strategy."""
    manager = OptimiseManager()
    study = manager.run_search(strategy, trials, jobs, timeout, train_data)

    print("\nSearch completed!")
    print(f"Study name: {study.study_name}")
    print(f"Best MAP@3: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Generate study summary
    manager.generate_study_summary(study.study_name)


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
