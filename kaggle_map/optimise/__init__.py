import click
from loguru import logger

from .utils import STORAGE_URL, list_all_studies


@click.group()
def cli() -> None:
    pass


@click.command("list-studies")
def list_studies() -> None:
    studies = list_all_studies()

    if not studies:
        print("No optimization studies found")
        return

    print("Available optimization studies:")
    for study_name in studies:
        print(f"  - {study_name}")

    print(f"\nTotal: {len(studies)} studies")
    print(f"Database: {STORAGE_URL}")


@click.command()
def dashboard() -> None:
    import subprocess
    import sys

    print("Launching Optuna dashboard...")
    print(f"Database: {STORAGE_URL}")
    print("\nDashboard will open in your browser at http://127.0.0.1:8080")
    print("Press Ctrl+C to stop the dashboard\n")

    try:
        subprocess.run([sys.executable, "-m", "optuna", "dashboard", STORAGE_URL], check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch dashboard: {e}")
        print("\nError: Failed to launch dashboard")
        print("Make sure optuna-dashboard is installed: pip install optuna-dashboard")


# Import and register sub-commands
from . import llm, mlp


# Register MLP commands
@click.group("mlp")
def mlp_group() -> None:
    pass


mlp_group.add_command(mlp.search)
mlp_group.add_command(mlp.search_embeddings)
mlp_group.add_command(mlp.analyze)


# Register LLM commands
@click.group("llm")
def llm_group() -> None:
    pass


llm_group.add_command(llm.compare)

# Add sub-groups to main CLI
cli.add_command(mlp_group)
cli.add_command(llm_group)
cli.add_command(list_studies)
cli.add_command(dashboard)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
