"""Checkpoint helpers for the MLP classifier.

Run ``uv run -m kaggle_map.mlp.checkpoint -h`` to inspect checkpoints from the CLI.

Examples:
    uv run -m kaggle_map.mlp.checkpoint models/mlp_latest.pt
    uv run -m kaggle_map.mlp.checkpoint models/mlp_latest.pt --show-predictions
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
import torch
from loguru import logger
from rich.console import Console

from kaggle_map.core.models import MLPTrainingConfig, QuestionId
from kaggle_map.embeddings import get_input_embeddings_dimension
from kaggle_map.mlp.model import CORRECTNESS_EMBEDDING_DIMENSIONS, QuestionSpecificMLP
from kaggle_map.utils.device import get_device
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


@dataclass(frozen=True)
class CheckpointBundle:
    """Structured view of the serialized checkpoint payload."""

    state_dict: dict[str, torch.Tensor]
    raw_config: dict[str, Any]
    question_predictions: dict[QuestionId, list[str]]


def _collect_question_predictions(model: QuestionSpecificMLP) -> dict[QuestionId, list[str]]:
    question_predictions: dict[QuestionId, list[str]] = {}

    for question_id, encoder in model.true_label_encoders.items():
        preds = question_predictions.setdefault(question_id, [])
        preds.extend(encoder.classes_.tolist())

    for question_id, encoder in model.false_label_encoders.items():
        preds = question_predictions.setdefault(question_id, [])
        preds.extend(encoder.classes_.tolist())

    return {qid: sorted(set(preds)) for qid, preds in question_predictions.items()}


def _prepare_checkpoint(payload: object) -> CheckpointBundle:
    assert isinstance(payload, dict), "Checkpoint payload must be a dict"
    assert "state_dict" in payload, "Checkpoint missing state_dict"
    assert "config" in payload, "Checkpoint missing config"
    assert "question_predictions" in payload, "Checkpoint missing question_predictions"

    state_dict = cast("dict[str, torch.Tensor]", payload["state_dict"])
    raw_config = dict(cast("dict[str, Any]", payload["config"]))
    question_predictions = cast("dict[QuestionId, list[str]]", payload["question_predictions"])

    assert state_dict, "state_dict cannot be empty"
    assert question_predictions, "question_predictions cannot be empty"

    return CheckpointBundle(state_dict=state_dict, raw_config=raw_config, question_predictions=question_predictions)


def save_checkpoint(model: QuestionSpecificMLP, filepath: Path, config: MLPTrainingConfig) -> None:
    """Persist model weights and minimal configuration metadata."""
    logger.info(f"Saving checkpoint to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    expected_dim = get_input_embeddings_dimension(config.embedding_strategy, config.embedding_model)
    model_dim = getattr(model, "embedding_dim", None)
    assert model_dim == expected_dim, (
        f"Model embedding dimension mismatch with config: model={model_dim}, expected={expected_dim}"
    )

    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "architecture_size": config.architecture_size.value,
            "dropout": float(config.dropout),
            "activation": config.activation.value,
            "embedding_model": config.embedding_model.value,
            "embedding_strategy": config.embedding_strategy.value,
            "train_csv_path": str(config.train_csv_path),
        },
        "question_predictions": _collect_question_predictions(model),
    }

    torch.save(payload, filepath)


def load_checkpoint(filepath: Path) -> tuple[QuestionSpecificMLP, MLPTrainingConfig]:
    """Load model and configuration from a checkpoint file."""
    logger.info(f"Loading checkpoint from {filepath}")
    assert filepath.exists(), f"Checkpoint not found: {filepath}"

    payload = torch.load(filepath, weights_only=False)
    bundle = _prepare_checkpoint(payload)

    config = MLPTrainingConfig.model_validate(bundle.raw_config)

    embedding_model = config.embedding_model
    embedding_strategy = config.embedding_strategy
    expected_dim = get_input_embeddings_dimension(embedding_strategy, embedding_model)

    model = QuestionSpecificMLP(
        bundle.question_predictions,
        embedding_model=embedding_model,
        embedding_strategy=embedding_strategy,
        architecture_size=config.architecture_size,
        dropout=float(config.dropout),
        activation=config.activation,
        correctness_embedding_dim=CORRECTNESS_EMBEDDING_DIMENSIONS,
    )
    model.load_state_dict(bundle.state_dict)

    model_dim = getattr(model, "embedding_dim", None)
    assert model_dim == expected_dim, (
        f"Loaded model embedding dimension mismatch: model={model_dim}, expected={expected_dim}"
    )

    device = get_device()
    model = model.to(device)

    return model, config


__all__ = ["load_checkpoint", "save_checkpoint"]


def _serialise_config(config: MLPTrainingConfig) -> dict[str, Any]:
    details = config.model_dump()
    details["embedding_model"] = config.embedding_model.value
    details["embedding_strategy"] = config.embedding_strategy.value
    details["activation"] = config.activation.value
    details["architecture_size"] = config.architecture_size.value
    details["train_csv_path"] = str(config.train_csv_path)
    details["optimizer"] = config.optimizer.value
    details["scheduler"] = config.scheduler.value
    details["expected_embedding_dim"] = get_input_embeddings_dimension(
        config.embedding_strategy,
        config.embedding_model,
    )
    return details


@click.command(help="Inspect an MLP checkpoint payload and configuration.")
@click.argument("checkpoint_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--show-predictions", is_flag=True, default=False, help="Print question-level label vocabulary")
def main(checkpoint_path: Path, *, show_predictions: bool) -> None:
    """Load the checkpoint and pretty-print its configuration."""
    logger.debug(f"Inspecting checkpoint: {checkpoint_path}")
    model, config = load_checkpoint(checkpoint_path)
    console = Console()

    config_payload = _serialise_config(config)
    config_payload["checkpoint_path"] = str(checkpoint_path)
    config_payload["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())

    console.print("[bold]Checkpoint configuration[/bold]")
    console.print_json(data=json.dumps(config_payload))

    if show_predictions:
        console.print("\n[bold]Question label vocabularies[/bold]")
        for question_id, labels in sorted(_collect_question_predictions(model).items()):
            console.print(f"[cyan]Q{question_id}[/cyan] -> {labels}")


if __name__ == "__main__":
    main()
