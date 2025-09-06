import subprocess
import sys

import click
from loguru import logger

from . import mlp
from .utils import STORAGE_URL, list_all_studies


@click.group()
def cli() -> None:
    pass


@click.command("list-studies")
def list_studies() -> None:
    studies = list_all_studies()

    if not studies:
        logger.info("No optimization studies found")
        return

    logger.info("Available optimization studies:")
    for study_name in studies:
        logger.info(f"  - {study_name}")

    logger.info(f"Total: {len(studies)} studies")
    logger.info(f"Database: {STORAGE_URL}")


@click.command()
def dashboard() -> None:
    logger.info("Launching Optuna dashboard...")
    logger.info(f"Database: {STORAGE_URL}")
    logger.info("Dashboard will open in your browser at http://127.0.0.1:8080")
    logger.info("Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run([sys.executable, "-m", "optuna", "dashboard", STORAGE_URL], check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch dashboard: {e}")
        logger.error("Error: Failed to launch dashboard")
        logger.error("Make sure optuna-dashboard is installed: pip install optuna-dashboard")


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


# Add sub-groups to main CLI
cli.add_command(mlp_group)
cli.add_command(llm_group)
cli.add_command(list_studies)
cli.add_command(dashboard)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
