"""Compare results from different hyperparameter searches and models."""

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console
from rich.table import Table


class ModelComparison:
    """Compare models and hyperparameter search results."""

    def __init__(self) -> None:
        self.console = Console()
        self.results_dir = Path("hypersearch_results")
        self.models_dir = Path("models")

    def load_study_results(self, study_name: str) -> dict[str, Any]:
        """Load results from a hyperparameter search study."""
        best_file = self.results_dir / f"{study_name}_best.json"
        history_file = self.results_dir / f"{study_name}.json"

        if not best_file.exists():
            msg = f"Study {study_name} not found"
            raise FileNotFoundError(msg)

        with open(best_file) as f:
            best = json.load(f)

        history = []
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

        return {"best": best, "history": history}

    def compare_studies(self, study_names: list[str]) -> None:
        """Compare multiple hyperparameter search studies."""
        table = Table(title="Hyperparameter Search Comparison")

        # Add columns
        table.add_column("Study", style="cyan")
        table.add_column("Best Value", style="magenta")
        table.add_column("N Trials", style="green")
        table.add_column("Best Trial", style="yellow")

        # Collect data
        studies_data = []
        for name in study_names:
            try:
                data = self.load_study_results(name)
                studies_data.append({
                    "name": name,
                    "best_value": data["best"]["best_value"],
                    "n_trials": data["best"]["n_trials"],
                    "best_trial": data["best"]["best_trial"],
                    "params": data["best"]["best_params"],
                })
            except FileNotFoundError:
                self.console.print(f"[red]Study {name} not found[/red]")
                continue

        # Sort by best value
        studies_data.sort(key=lambda x: x["best_value"], reverse=True)

        # Add rows
        for study in studies_data:
            table.add_row(
                study["name"],
                f"{study['best_value']:.4f}",
                str(study["n_trials"]),
                str(study["best_trial"]),
            )

        self.console.print(table)

        # Show best parameters for top study
        if studies_data:
            best_study = studies_data[0]
            params_table = Table(title=f"Best Parameters from {best_study['name']}")
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="magenta")

            for param, value in best_study["params"].items():
                params_table.add_row(param, str(value))

            self.console.print(params_table)

    def analyze_parameter_importance(self, study_name: str) -> None:
        """Analyze which parameters have the most impact on performance."""
        data = self.load_study_results(study_name)
        history = data["history"]

        if not history:
            self.console.print("[red]No history data available[/red]")
            return

        # Convert to DataFrame for analysis
        df = pd.DataFrame(history)

        # Filter completed trials only
        df = df[df["state"] == "TrialState.COMPLETE"]

        if df.empty:
            self.console.print("[red]No completed trials found[/red]")
            return

        # Extract parameters into columns
        params_df = pd.json_normalize(df["params"])
        df = pd.concat([df[["number", "value"]], params_df], axis=1)

        # Calculate correlation with performance
        correlations = {}
        for col in params_df.columns:
            if df[col].dtype in ["float64", "int64"]:
                corr = df[col].corr(df["value"])
                correlations[col] = abs(corr)  # Use absolute correlation

        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        # Display results
        table = Table(title=f"Parameter Importance Analysis: {study_name}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Correlation", style="magenta")
        table.add_column("Importance", style="green")

        for param, corr in sorted_params:
            importance = "High" if corr > 0.5 else "Medium" if corr > 0.3 else "Low"
            table.add_row(param, f"{corr:.3f}", importance)

        self.console.print(table)

        # Show value distributions for top parameters
        if sorted_params:
            top_param = sorted_params[0][0]
            self.console.print(f"\n[bold]Top parameter: {top_param}[/bold]")

            # Group by parameter value and show average performance
            grouped = df.groupby(top_param)["value"].agg(["mean", "std", "count"])

            dist_table = Table(title=f"{top_param} Performance Distribution")
            dist_table.add_column("Value", style="cyan")
            dist_table.add_column("Mean Performance", style="magenta")
            dist_table.add_column("Std Dev", style="yellow")
            dist_table.add_column("N Trials", style="green")

            for idx, row in grouped.iterrows():
                dist_table.add_row(
                    str(idx),
                    f"{row['mean']:.4f}",
                    f"{row['std']:.4f}" if pd.notna(row["std"]) else "N/A",
                    str(int(row["count"])),
                )

            self.console.print(dist_table)

    def list_studies(self) -> None:
        """List all available hyperparameter search studies."""
        if not self.results_dir.exists():
            self.console.print("[red]No hypersearch results found[/red]")
            return

        # Find all best.json files
        best_files = list(self.results_dir.glob("*_best.json"))

        if not best_files:
            self.console.print("[red]No completed studies found[/red]")
            return

        table = Table(title="Available Studies")
        table.add_column("Study Name", style="cyan")
        table.add_column("Best Value", style="magenta")
        table.add_column("Timestamp", style="yellow")

        studies = []
        for file in best_files:
            study_name = file.stem.replace("_best", "")
            with open(file) as f:
                data = json.load(f)

            studies.append({
                "name": study_name,
                "value": data["best_value"],
                "timestamp": data.get("timestamp", "Unknown"),
            })

        # Sort by timestamp
        studies.sort(key=lambda x: x["timestamp"], reverse=True)

        for study in studies:
            table.add_row(
                study["name"],
                f"{study['value']:.4f}",
                study["timestamp"][:19] if study["timestamp"] != "Unknown" else "Unknown",
            )

        self.console.print(table)


@click.group()
def cli() -> None:
    """Compare and analyze hyperparameter search results."""


@cli.command()
def list_studies() -> None:
    """List all available studies."""
    comparison = ModelComparison()
    comparison.list_studies()


@cli.command()
@click.argument("study_names", nargs=-1, required=True)
def compare(study_names) -> None:
    """Compare multiple studies."""
    comparison = ModelComparison()
    comparison.compare_studies(list(study_names))


@cli.command()
@click.argument("study_name")
def analyze(study_name) -> None:
    """Analyze parameter importance for a study."""
    comparison = ModelComparison()
    comparison.analyze_parameter_importance(study_name)


if __name__ == "__main__":
    cli()
