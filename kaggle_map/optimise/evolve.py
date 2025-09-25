"""OpenEvolve-driven prompt evolution for maximising MAP@3.

Example:
    uv run -m kaggle_map.optimise.evolve --help
"""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import click
from loguru import logger
from openevolve import OpenEvolve
from openevolve.database import Program
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaggle_map.core.random_seed import configure_random_seed, get_active_seed
from kaggle_map.llm.evaluator import EvaluationConfig, evaluate_with_llm
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

_DATASET_ENV = "KAGGLE_MAP_OPENEVO_DATASET"
_SAMPLE_RATIO_ENV = "KAGGLE_MAP_OPENEVO_SAMPLE_RATIO"
_ROW_IDS_ENV = "KAGGLE_MAP_OPENEVO_ROW_IDS"
_OUTPUT_DIR_ENV = "KAGGLE_MAP_OPENEVO_OUTPUT_DIR"
_RANDOM_SEED_ENV = "KAGGLE_MAP_RANDOM_SEED"


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
StrictProbability = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class PromptEvolutionConfig(BaseModel):
    """Configuration for launching OpenEvolve prompt optimisation."""

    dataset_path: Path = Field(..., description="CSV containing labelled rows for evaluation")
    template_path: Path = Field(..., description="Initial prompt template to evolve")
    output_dir: Path = Field(default=Path("logs/openevolve"), description="Directory for OpenEvolve artefacts")
    sample_ratio: Probability = 0.5
    row_ids: list[int] | None = None
    openevolve_config: Path | None = None
    max_iterations: PositiveInt = 100
    target_score: StrictProbability | None = None
    best_prompt_path: Path | None = Field(default=None, description="Where to persist the best evolved prompt")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_paths(self) -> "PromptEvolutionConfig":
        assert self.dataset_path.exists(), f"Dataset not found at {self.dataset_path}"
        assert self.dataset_path.is_file(), "Dataset path must point to a file"
        assert self.template_path.exists(), f"Template not found at {self.template_path}"
        assert self.template_path.is_file(), "Template path must point to a file"
        if self.row_ids is not None:
            assert self.row_ids, "Row IDs cannot be an empty list"
            assert len(set(self.row_ids)) == len(self.row_ids), "Row IDs must be unique"
            assert all(rid > 0 for rid in self.row_ids), "Row IDs must be positive integers"
        if self.openevolve_config is not None:
            assert self.openevolve_config.exists(), f"Config not found at {self.openevolve_config}"
            assert self.openevolve_config.is_file(), "OpenEvolve config must be a file"
        if self.best_prompt_path is not None:
            assert not self.best_prompt_path.is_dir(), "best_prompt_path cannot be a directory"
        return self

    def runtime_environment(self) -> dict[str, str]:
        """Environment variables consumed by the evaluation callback."""

        env: dict[str, str] = {
            _DATASET_ENV: str(self.dataset_path.resolve()),
            _SAMPLE_RATIO_ENV: str(self.sample_ratio),
            _OUTPUT_DIR_ENV: str(self.output_dir.resolve()),
        }
        if self.row_ids is not None:
            env[_ROW_IDS_ENV] = ",".join(str(rid) for rid in self.row_ids)
        return env

    def best_prompt_output_path(self) -> Path:
        suffix = self.template_path.suffix or ".j2"
        if self.best_prompt_path is not None:
            destination = self.best_prompt_path
        else:
            destination = self.template_path.parent / f"{self.template_path.stem}.best{suffix}"
        return destination.resolve()


class PromptEvolutionRuntimeSettings(BaseModel):
    """Runtime parameters exposed to the OpenEvolve evaluator."""

    dataset_path: Path
    sample_ratio: Probability = 1.0
    row_ids: list[int] | None = None
    output_dir: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_inputs(self) -> "PromptEvolutionRuntimeSettings":
        assert self.dataset_path.exists(), f"Dataset not found at {self.dataset_path}"
        assert self.dataset_path.is_file(), "Dataset path must be a file"
        assert self.output_dir.exists(), f"Output directory missing at {self.output_dir}"
        assert self.output_dir.is_dir(), "Output path must be a directory"
        if self.row_ids is not None:
            assert self.row_ids, "Row IDs cannot be empty"
            assert len(set(self.row_ids)) == len(self.row_ids), "Row IDs must be unique"
            assert all(rid > 0 for rid in self.row_ids), "Row IDs must be positive"
        return self

    @classmethod
    def from_environment(cls) -> "PromptEvolutionRuntimeSettings":
        dataset_value = os.environ.get(_DATASET_ENV)
        assert dataset_value, f"Environment variable {_DATASET_ENV} is required"
        sample_ratio_value = float(os.environ.get(_SAMPLE_RATIO_ENV, "1.0"))
        output_dir_value = os.environ.get(_OUTPUT_DIR_ENV)
        assert output_dir_value, f"Environment variable {_OUTPUT_DIR_ENV} is required"
        row_ids_value = os.environ.get(_ROW_IDS_ENV)
        row_ids = None
        if row_ids_value:
            row_ids = [int(raw) for raw in row_ids_value.split(",") if raw]
        settings = cls(
            dataset_path=Path(dataset_value),
            sample_ratio=sample_ratio_value,
            row_ids=row_ids,
            output_dir=Path(output_dir_value),
        )
        logger.debug(
            "Resolved runtime settings",
            dataset=settings.dataset_path,
            sample_ratio=settings.sample_ratio,
            row_ids=settings.row_ids,
            output_dir=settings.output_dir,
        )
        return settings


def prepare_environment(config: PromptEvolutionConfig) -> dict[str, str]:
    """Materialise runtime environment variables for OpenEvolve."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = config.runtime_environment()
    seed_value = str(get_active_seed())
    env[_RANDOM_SEED_ENV] = seed_value

    for key, value in env.items():
        os.environ[key] = value
    logger.info("Prepared OpenEvolve environment", environment=env)
    return env


def evaluate(candidate_prompt_path: str) -> dict[str, float]:
    """Evaluation entrypoint consumed by OpenEvolve.

    Args:
        candidate_prompt_path: Path to the prompt template proposed by OpenEvolve.

    Returns:
        Mapping of metric name to score. Must include ``combined_score``.
    """

    active_seed = configure_random_seed()
    logger.debug("Evaluating candidate prompt", candidate=candidate_prompt_path, seed=active_seed)
    runtime = PromptEvolutionRuntimeSettings.from_environment()
    prompt_path = Path(candidate_prompt_path)
    assert prompt_path.exists(), f"Candidate prompt not found at {prompt_path}"
    assert prompt_path.is_file(), "Candidate prompt must be a file"

    evaluation_config = EvaluationConfig(
        data_path=runtime.dataset_path,
        sample_ratio=runtime.sample_ratio,
        row_ids=runtime.row_ids,
        template_path=prompt_path,
        random_seed=active_seed,
    )

    score = evaluate_with_llm(evaluation_config)
    metrics = {"combined_score": score, "map@3": score}
    logger.info("Evaluation complete", metrics=metrics)
    return metrics


def _execute_evolution(evolver: OpenEvolve, config: PromptEvolutionConfig) -> Program | None:
    return asyncio.run(
        evolver.run(
            iterations=config.max_iterations,
            target_score=config.target_score,
        )
    )


def _persist_best_prompt(best_program: Program | str | None, config: PromptEvolutionConfig) -> Path | None:
    if best_program is None:
        logger.warning("Evolution produced no candidate prompt")
        return None

    prompt_text = getattr(best_program, "code", None)
    if prompt_text is None:
        if isinstance(best_program, str):
            prompt_text = best_program
        else:
            msg = "Best program does not contain prompt code"
            raise AssertionError(msg)

    destination = config.best_prompt_output_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(prompt_text, encoding="utf-8")
    logger.success("Saved best prompt", path=str(destination))
    return destination


def run_prompt_evolution(config: PromptEvolutionConfig) -> Program | None:
    """Run OpenEvolve using the provided configuration."""

    active_seed = configure_random_seed()
    logger.debug("Prompt evolution configured random seed: {}", active_seed)
    prepare_environment(config)
    logger.info("Starting OpenEvolve run", config=config.model_dump())
    evolver = OpenEvolve(
        initial_program_path=str(config.template_path.resolve()),
        evaluation_file=str(Path(__file__).resolve()),
        config_path=str(config.openevolve_config) if config.openevolve_config else None,
        output_dir=str(config.output_dir.resolve()),
    )
    result = _execute_evolution(evolver, config)
    _persist_best_prompt(result, config)
    logger.success("Prompt evolution completed")
    return result


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(path_type=Path, exists=True),
    default=Path("datasets/33474_focus_train.csv"),
    show_default=True,
    help="CSV file used to score prompts",
)
@click.option(
    "--template",
    "template_path",
    type=click.Path(path_type=Path, exists=True),
    default=Path("kaggle_map/llm/prompts/predict.j2"),
    show_default=True,
    help="Seed prompt template",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("logs/openevolve"),
    show_default=True,
    help="Directory for OpenEvolve artefacts",
)
@click.option(
    "--sample-ratio",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help="Subset of rows to evaluate per candidate",
)
@click.option(
    "--row-ids",
    type=str,
    default=None,
    help="Comma-separated row IDs to evaluate instead of random sampling",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=False,
    help="Override random seed propagated to all evaluations",
)
@click.option(
    "--openevolve-config",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional OpenEvolve YAML configuration overriding defaults",
)
@click.option(
    "--max-iterations",
    type=int,
    default=100,
    show_default=True,
    help="Maximum OpenEvolve iterations",
)
@click.option(
    "--target-score",
    type=float,
    default=None,
    help="Stop once combined_score reaches this threshold",
)
@click.option(
    "--save-best-to",
    "best_prompt_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional destination file for the best evolved prompt (defaults beside the seed template)",
)
def main(  # noqa: PLR0913
    dataset_path: Path,
    template_path: Path,
    output_dir: Path,
    sample_ratio: float,
    row_ids: str | None,
    seed: int | None,
    openevolve_config: Path | None,
    max_iterations: int,
    target_score: float | None,
    best_prompt_path: Path | None,
) -> None:
    """CLI for running OpenEvolve against the MAP@3 evaluator."""

    active_seed = configure_random_seed(override=seed)
    logger.debug("OpenEvolve CLI configured random seed: {}", active_seed)

    parsed_row_ids = None
    if row_ids:
        parsed_row_ids = [int(value.strip()) for value in row_ids.split(",") if value.strip()]
    config = PromptEvolutionConfig(
        dataset_path=dataset_path,
        template_path=template_path,
        output_dir=output_dir,
        sample_ratio=sample_ratio,
        row_ids=parsed_row_ids,
        openevolve_config=openevolve_config,
        max_iterations=max_iterations,
        target_score=target_score,
        best_prompt_path=best_prompt_path,
    )
    run_prompt_evolution(config)


if __name__ == "__main__":
    configure_logger(__name__, console_level="DEBUG")
    main()
