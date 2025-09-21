"""MLP module main entry point for misconception prediction.

This module provides both a Python API and command-line interface for training,
evaluating, and using Multi-Layer Perceptron models for student misconception prediction.
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from loguru import logger
from torch.nn import functional
from torch.utils.data import DataLoader

from kaggle_map.core.dataset import extract_correct_answers, load_training_data
from kaggle_map.core.models import (
    RANDOM_SEED,
    ActivationType,
    Answer,
    ArchitectureSize,
    Category,
    EmbeddingModel,
    EmbeddingStrategy,
    EvaluationRow,
    MLPTrainingConfig,
    Prediction,
    QuestionId,
    SubmissionRow,
    TrainingRow,
)
from kaggle_map.embeddings import encode, get_input_embeddings_dimension
from kaggle_map.mlp.dataset import DatasetArrays, MLPDataset
from kaggle_map.mlp.embedding_cache import get_or_compute_embeddings
from kaggle_map.mlp.label_encoder import LabelEncoders
from kaggle_map.mlp.loss import ListMLELoss
from kaggle_map.mlp.model import (
    CORRECTNESS_EMBEDDING_DIMENSIONS,
    EvaluationResult,
    QuestionSpecificMLP,
)
from kaggle_map.mlp.trainer import TrainingSetup, train_model
from kaggle_map.utils.device import get_device
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Callable

    SubparsersAction = argparse._SubParsersAction[ArgumentParser]
else:
    SubparsersAction = argparse._SubParsersAction

configure_logger(__name__)

# Maximum predictions per question as required by Kaggle MAP competition format
# MAP@3 evaluation metric requires exactly 3 predictions per question
MAX_PREDICTIONS = 3

# Default configuration instances to avoid None checks
_DEFAULT_TRAINING_CONFIG = MLPTrainingConfig()
EMBEDDING_TENSOR_DIMENSIONS = 2
LEGACY_DEFAULT_EMBEDDING_DIM = 8192


@dataclass(frozen=True)
class LoadedCheckpoint:
    state_dict: dict[str, torch.Tensor]
    raw_config: dict[str, Any]
    question_predictions: dict[QuestionId, list[str]]
    embedding_dim: int
    is_new_format: bool


def _extract_question_predictions(training_data: list[TrainingRow]) -> dict[QuestionId, list[str]]:
    question_predictions = defaultdict(list)
    for row in training_data:
        pred_str = str(row.prediction)
        question_predictions[row.question_id].append(pred_str)

    return {qid: list(set(preds)) for qid, preds in question_predictions.items()}


def _load_training_question_predictions() -> dict[QuestionId, list[str]]:
    training_rows = load_training_data(MLPTrainingConfig().train_csv_path)
    return _extract_question_predictions(training_rows)


def _infer_embedding_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    trunk_weight = state_dict.get("trunk.0.weight")
    if trunk_weight is None:
        return LEGACY_DEFAULT_EMBEDDING_DIM
    return trunk_weight.shape[1] - CORRECTNESS_EMBEDDING_DIMENSIONS


def _prepare_loaded_checkpoint(checkpoint: object) -> LoadedCheckpoint:
    is_new_format = isinstance(checkpoint, dict) and "state_dict" in checkpoint

    if is_new_format:
        mapping = cast("dict[str, Any]", checkpoint)
        state_dict = cast("dict[str, torch.Tensor]", mapping["state_dict"])
        raw_config = dict(cast("dict[str, Any]", mapping.get("config", {})))
        embedding_dim = cast("int | None", raw_config.get("embedding_dim"))
        if embedding_dim is None:
            embedding_dim = _infer_embedding_dim_from_state_dict(state_dict)
        question_predictions = cast("dict[QuestionId, list[str]] | None", mapping.get("question_predictions"))
        if question_predictions is None:
            question_predictions = _load_training_question_predictions()
        return LoadedCheckpoint(
            state_dict=state_dict,
            raw_config=raw_config,
            question_predictions=question_predictions,
            embedding_dim=embedding_dim,
            is_new_format=True,
        )

    legacy_state_dict = cast("dict[str, torch.Tensor]", checkpoint)
    embedding_dim = _infer_embedding_dim_from_state_dict(legacy_state_dict)
    question_predictions = _load_training_question_predictions()
    legacy_config = {
        "architecture_size": ArchitectureSize.XLARGE.value,
        "dropout": 0.3,
        "activation": ActivationType.GELU.value,
        "embedding_model": EmbeddingModel.QWEN.value,
        "embedding_strategy": EmbeddingStrategy.GOAL_DRIVEN.value,
    }

    return LoadedCheckpoint(
        state_dict=legacy_state_dict,
        raw_config=legacy_config,
        question_predictions=question_predictions,
        embedding_dim=embedding_dim,
        is_new_format=False,
    )


@dataclass(frozen=True)
class DataSplit:
    """Train/validation/test split indices for dataset partitioning."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size


def _get_split_indices(n_samples: int, train_ratio: float = 0.7) -> DataSplit:
    """Get train/val/test split indices."""
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * 0.15)

    return DataSplit(
        train_indices=indices[:train_size],
        val_indices=indices[train_size : train_size + val_size],
        test_indices=indices[train_size + val_size :],
    )


def _configure_fit_parser(
    subparsers: SubparsersAction,
) -> None:
    fit_parser = subparsers.add_parser("fit", help="Train the MLP model")
    fit_parser.add_argument(
        "--train-data",
        type=str,
        default="datasets/train.csv",
        help="Path to training data CSV (default: datasets/train.csv)",
    )
    fit_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    fit_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)",
    )
    fit_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    fit_parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    fit_parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp.pkl",
        help="Path to save the trained model (default: models/mlp.pkl)",
    )
    fit_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed training progress",
    )


def _configure_eval_parser(
    subparsers: SubparsersAction,
) -> None:
    eval_parser = subparsers.add_parser("eval", help="Evaluate the MLP model")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp.pkl",
        help="Path to the saved model (default: models/mlp.pkl)",
    )
    eval_parser.add_argument(
        "--train-data",
        type=str,
        default="datasets/train.csv",
        help="Path to training data CSV for evaluation (default: datasets/train.csv)",
    )
    eval_parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data used for training (default: 0.7)",
    )
    eval_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed evaluation metrics",
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLP model for student misconception prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model with default settings
  python -m kaggle_map.mlp fit

  # Train with custom parameters
  python -m kaggle_map.mlp fit --epochs 100 --learning-rate 0.001

  # Evaluate a trained model
  python -m kaggle_map.mlp eval --model-path models/mlp.pkl
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _configure_fit_parser(subparsers)
    _configure_eval_parser(subparsers)
    return parser


def _dispatch_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "fit": handle_fit,
        "eval": handle_eval,
    }

    handler = handlers.get(args.command)
    if handler is None:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    try:
        assert handler is not None
        handler(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Error during {args.command}: {exc}")
        if getattr(args, "verbose", False):
            logger.exception("Full traceback:")
        sys.exit(1)


def fit(config: MLPTrainingConfig = _DEFAULT_TRAINING_CONFIG) -> tuple[QuestionSpecificMLP, MLPTrainingConfig]:
    """Train an MLP model for misconception prediction.

    Args:
        config: Training configuration

    Returns:
        Trained QuestionSpecificMLP model
    """
    strategy = config.embedding_strategy if config else EmbeddingStrategy.GOAL_DRIVEN

    device = get_device()
    logger.info(f"Training on {device} with embedding strategy: {strategy.value}")

    training_data = load_training_data(config.train_csv_path)
    correct_answers = extract_correct_answers(training_data)
    question_predictions = _extract_question_predictions(training_data)

    logger.info("Preparing embeddings...")

    # First pass: Create all EvaluationRows and collect metadata
    eval_rows = []
    metadata_tuples = []
    for row in training_data:
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
            correct_answer=correct_answers.get(row.question_id, ""),
        )
        eval_rows.append(eval_row)
        metadata_tuples.append((row.question_id, str(row.prediction), row.mc_answer))

    # Get embeddings from cache or compute them
    embeddings, question_ids, predictions, mc_answers = get_or_compute_embeddings(
        eval_rows, metadata_tuples, config.train_csv_path, config.embedding_model, strategy
    )

    embedding_dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    expected_embedding_dim = get_input_embeddings_dimension(strategy, config.embedding_model)
    assert (
        embedding_dim == expected_embedding_dim
    ), (
        "Observed embedding dimension does not match expected dimension from embedding configuration: "
        f"observed={embedding_dim}, expected={expected_embedding_dim}, "
        f"model={config.embedding_model.value}, strategy={strategy.value}"
    )

    model = QuestionSpecificMLP(
        question_predictions,
        embedding_model=config.embedding_model,
        embedding_strategy=strategy,
        architecture_size=config.architecture_size,
        dropout=config.dropout,
        activation=config.activation,
        correctness_embedding_dim=CORRECTNESS_EMBEDDING_DIMENSIONS,
    )
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    n_samples = len(embeddings)
    split = _get_split_indices(n_samples, config.train_split)
    logger.info(f"Data split - Train: {split.train_size}, Val: {split.val_size}")

    train_arrays = DatasetArrays(
        embeddings=embeddings[split.train_indices],
        question_ids=question_ids[split.train_indices],
        predictions=predictions[split.train_indices],
        mc_answers=mc_answers[split.train_indices],
    )

    val_arrays = DatasetArrays(
        embeddings=embeddings[split.val_indices],
        question_ids=question_ids[split.val_indices],
        predictions=predictions[split.val_indices],
        mc_answers=mc_answers[split.val_indices],
    )

    label_encoders = LabelEncoders(
        true_label_encoders=model.true_label_encoders,
        false_label_encoders=model.false_label_encoders,
    )

    train_dataset = MLPDataset(train_arrays, correct_answers, label_encoders)
    val_dataset = MLPDataset(val_arrays, correct_answers, label_encoders)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)

    setup = TrainingSetup(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        loss_fn=ListMLELoss(),
    )
    result = train_model(setup)

    logger.info(f"Training complete. Best val loss: {result.history.get('best_val_loss', 'N/A')}")

    # Return both model and config for saving
    return result.model, config


def _process_single_prediction(
    model: QuestionSpecificMLP,
    eval_row: EvaluationRow,
    embedding_tensor: torch.Tensor,
    correct_answers: dict[QuestionId, Answer],
    device: torch.device,
) -> SubmissionRow:
    """Core prediction logic for a single evaluation row.

    Determines correctness, runs forward pass, and extracts top-k predictions
    from the appropriate encoder (correct/incorrect branch). Falls back to
    default categories when model has no predictions for the question.
    """
    # Preconditions - fail fast with clear messages
    expected_dims = 2
    assert embedding_tensor.dim() == expected_dims, (
        f"Expected {expected_dims}D embedding tensor, got {embedding_tensor.dim()}D"
    )
    assert embedding_tensor.size(0) == 1, f"Expected batch size 1, got {embedding_tensor.size(0)}"

    is_correct = eval_row.mc_answer == correct_answers.get(eval_row.question_id, "")
    correctness_idx = torch.tensor([1 if is_correct else 0], dtype=torch.long).to(device)

    # Forward pass
    question_tensor = torch.tensor([eval_row.question_id], dtype=torch.long, device=device)
    outputs: dict[EvaluationResult, torch.Tensor] = model(
        embedding_tensor,
        question_tensor,
        correctness_idx,
    )

    # Extract predictions
    predictions: list[Prediction] = []
    key = EvaluationResult(question_id=eval_row.question_id, is_correct=is_correct)

    if key in outputs:
        logits = outputs[key]
        assert logits is not None, f"Got None logits for key {key}"
        assert logits.dim() >= 1, f"Expected at least 1D logits, got {logits.dim()}D"

        probs = functional.softmax(logits, dim=-1)[0]
        top_k = min(MAX_PREDICTIONS, logits.size(-1))
        assert top_k > 0, f"No predictions available for question {eval_row.question_id}"

        top_indices = torch.topk(probs, k=top_k)[1]

        encoder = (
            model.true_label_encoders.get(eval_row.question_id)
            if is_correct
            else model.false_label_encoders.get(eval_row.question_id)
        )

        if encoder:
            indices_array = top_indices.cpu().numpy()
            assert len(indices_array) > 0, f"Empty indices for question {eval_row.question_id}"

            pred_strings = encoder.inverse_transform(indices_array)
            assert len(pred_strings) > 0, f"No predicted strings for question {eval_row.question_id}"

            predictions.extend(Prediction.from_string(pred_str) for pred_str in pred_strings)

    # Fill with default predictions if needed
    default = Prediction(
        category=Category.TRUE_NEITHER if is_correct else Category.FALSE_NEITHER,
        misconception="NA",
    )
    predictions.extend([default] * (MAX_PREDICTIONS - len(predictions)))

    # Postconditions - verify output contract
    assert len(predictions) >= MAX_PREDICTIONS, (
        f"Expected at least {MAX_PREDICTIONS} predictions, got {len(predictions)}"
    )
    assert all(pred is not None for pred in predictions), "Found None predictions in result"

    result = SubmissionRow(row_id=eval_row.row_id, predicted_categories=predictions[:MAX_PREDICTIONS])
    assert result.row_id == eval_row.row_id, "Row ID mismatch in result"
    assert len(result.predicted_categories) == MAX_PREDICTIONS, f"Expected {MAX_PREDICTIONS} categories in result"
    return result


def predict_batch(
    model: QuestionSpecificMLP, evaluation_rows: list[EvaluationRow], config: MLPTrainingConfig
) -> list[SubmissionRow]:
    assert evaluation_rows, "Evaluation rows list is empty"

    # Amortize expensive data loading across all predictions in batch
    logger.debug("Loading training data for batch prediction")
    training_data = load_training_data(MLPTrainingConfig().train_csv_path)
    correct_answers = extract_correct_answers(training_data)

    device = get_device()
    logger.debug(f"Device selection: using {device}")
    assert config is not None, "MLPTrainingConfig must be provided for prediction"
    embedding_strategy = config.embedding_strategy
    embedding_model = config.embedding_model
    logger.debug(f"Embedding configuration: strategy={embedding_strategy.value}, model={embedding_model.value}")

    # Add correct answers to evaluation rows for embedding
    eval_rows_with_answers: list[EvaluationRow] = [
        EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
            correct_answer=correct_answers.get(row.question_id, ""),
        )
        for row in evaluation_rows
    ]

    # Batch encode all rows to amortize model loading overhead
    logger.debug(f"Batch encoding {len(eval_rows_with_answers)} rows")
    embeddings = encode(eval_rows_with_answers, embedding_strategy, embedding_model)

    # Validate embeddings integrity
    assert embeddings is not None, "Encoding returned None embeddings"
    assert embeddings.size(0) == len(eval_rows_with_answers), (
        f"Embedding count mismatch: got {embeddings.size(0)}, expected {len(eval_rows_with_answers)}"
    )
    assert embeddings.dim() == EMBEDDING_TENSOR_DIMENSIONS, (
        f"Expected {EMBEDDING_TENSOR_DIMENSIONS}D embeddings, got {embeddings.dim()}D"
    )

    # Ensure embeddings are on CPU for numpy conversion if needed
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()

    # Process predictions
    model.eval()
    submission_rows: list[SubmissionRow] = []
    predict_start = time.time()

    with torch.no_grad():
        for i, eval_row in enumerate(evaluation_rows):
            embedding_tensor = embeddings[i].unsqueeze(0).to(device)
            submission_row = _process_single_prediction(model, eval_row, embedding_tensor, correct_answers, device)
            submission_rows.append(submission_row)

    predict_time = time.time() - predict_start
    predictions_per_sec = len(submission_rows) / predict_time
    logger.debug(
        f"Batch prediction complete for {len(submission_rows)} rows "
        f"in {predict_time:.2f}s ({predictions_per_sec:.1f} predictions/sec)"
    )
    return submission_rows


def evaluate(model: QuestionSpecificMLP, test_data: list[TrainingRow], config: MLPTrainingConfig) -> dict[str, float]:
    assert test_data, "Test data is empty, cannot evaluate model"
    eval_rows: list[EvaluationRow] = [
        EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )
        for row in test_data
    ]

    logger.info(f"Starting batch evaluation of {len(test_data)} samples")
    predictions = predict_batch(model, eval_rows, config)
    assert len(test_data) == len(predictions), f"Mismatch: {len(test_data)} test rows vs {len(predictions)} predictions"

    # Score each prediction against ground truth for overall metric
    map_scores = []
    for row, prediction in zip(test_data, predictions, strict=True):
        assert row.prediction is not None, f"Missing prediction for row {row.row_id}"
        assert prediction.predicted_categories is not None, f"Missing predicted categories for row {prediction.row_id}"
        score = calculate_map_at_3(row.prediction, prediction.predicted_categories)
        map_scores.append(score)

    assert len(map_scores) > 0, "No valid MAP scores calculated"
    avg_map = sum(map_scores) / len(map_scores)
    logger.info(f"Evaluation MAP@3: {avg_map:.4f} on {len(test_data)} samples")

    return {
        "validation_map@3": avg_map,
        "validation_samples": len(test_data),
    }


def save(model: QuestionSpecificMLP, filepath: Path, config: MLPTrainingConfig | None = None) -> None:
    """Save model to disk with configuration.

    Args:
        model: Model to save
        filepath: Path to save the model
        config: Optional training config to save with model
    """
    logger.info(f"Saving model to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save both model state and configuration
    # Extract architecture info from the model
    first_layer = model.trunk[0]
    assert hasattr(first_layer, "in_features"), "First trunk layer must expose in_features"
    total_input_dim = cast("int", first_layer.in_features)
    correctness_dim = getattr(model, "correctness_embedding_dim", CORRECTNESS_EMBEDDING_DIMENSIONS)
    embedding_dim = getattr(model, "embedding_dim", total_input_dim - correctness_dim)

    # Reconstruct question_predictions from the model's encoders
    question_predictions = {}
    for qid, encoder in model.true_label_encoders.items():
        if qid not in question_predictions:
            question_predictions[qid] = []
        question_predictions[qid].extend(encoder.classes_.tolist())
    for qid, encoder in model.false_label_encoders.items():
        if qid not in question_predictions:
            question_predictions[qid] = []
        question_predictions[qid].extend(encoder.classes_.tolist())

    save_dict = {
        "state_dict": model.state_dict(),
        "config": {
            "embedding_dim": embedding_dim,
            "architecture_size": config.architecture_size.value if config else ArchitectureSize.XLARGE.value,
            "dropout": config.dropout if config else 0.3,
            "activation": config.activation.value if config else ActivationType.GELU.value,
            "embedding_model": (
                config.embedding_model.value
                if config
                else cast("EmbeddingModel", getattr(model, "embedding_model", EmbeddingModel.QWEN)).value
            ),
            "embedding_strategy": (
                config.embedding_strategy.value
                if config
                else cast(
                    "EmbeddingStrategy",
                    getattr(model, "embedding_strategy", EmbeddingStrategy.GOAL_DRIVEN),
                ).value
            ),
        },
        "question_predictions": question_predictions,
    }
    torch.save(save_dict, filepath)


def load(filepath: Path) -> tuple[QuestionSpecificMLP, MLPTrainingConfig | None]:
    """Load model from disk with configuration.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {filepath}")
    assert filepath.exists(), f"Model file not found: {filepath}"

    checkpoint = torch.load(filepath, weights_only=False)
    loaded = _prepare_loaded_checkpoint(checkpoint)

    config_dict = loaded.raw_config
    architecture_size = ArchitectureSize(config_dict.get("architecture_size", "xlarge"))
    raw_dropout = config_dict.get("dropout", 0.3)
    dropout = float(raw_dropout) if isinstance(raw_dropout, str) else float(cast("float", raw_dropout))
    activation = ActivationType(config_dict.get("activation", "gelu"))

    embedding_model = EmbeddingModel(config_dict.get("embedding_model", EmbeddingModel.QWEN.value))
    raw_strategy_value = config_dict.get("embedding_strategy", EmbeddingStrategy.GOAL_DRIVEN.value)
    normalized_strategy_value = "goal_driven" if raw_strategy_value == "semantic" else raw_strategy_value
    embedding_strategy = EmbeddingStrategy(normalized_strategy_value)

    expected_embedding_dim = get_input_embeddings_dimension(embedding_strategy, embedding_model)
    assert (
        loaded.embedding_dim == expected_embedding_dim
    ), (
        "Loaded embedding dimension does not match expected dimension for embedding configuration: "
        f"loaded={loaded.embedding_dim}, expected={expected_embedding_dim}, "
        f"model={embedding_model.value}, strategy={embedding_strategy.value}"
    )

    model = QuestionSpecificMLP(
        loaded.question_predictions,
        embedding_model=embedding_model,
        embedding_strategy=embedding_strategy,
        architecture_size=architecture_size,
        dropout=dropout,
        activation=activation,
        correctness_embedding_dim=CORRECTNESS_EMBEDDING_DIMENSIONS,
    )
    model.load_state_dict(loaded.state_dict)

    device = get_device()
    model = model.to(device)
    logger.debug(f"Model loaded and moved to device: {device}")

    loaded_config: MLPTrainingConfig | None = None
    if loaded.is_new_format:
        loaded_config = MLPTrainingConfig(
            embedding_dim=loaded.embedding_dim,
            architecture_size=architecture_size,
            dropout=dropout,
            activation=activation,
            embedding_model=embedding_model,
            embedding_strategy=embedding_strategy,
        )

    return model, loaded_config


def handle_fit(args: argparse.Namespace) -> None:
    """Handle the fit command to train a model."""

    # Configure logging level
    if args.verbose:
        logger.info("Starting model training with parameters:")
        logger.info(f"  Training data: {args.train_data}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Train split: {args.train_split}")
        logger.info(f"  Model path: {args.model_path}")

    # Create MLPTrainingConfig from arguments
    config = MLPTrainingConfig(
        train_csv_path=Path(args.train_data),
        train_split=args.train_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Train the model
    logger.info("Loading training data...")
    model, train_config = fit(config)

    # Save the model with its config
    output_path = Path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save(model, output_path, train_config)
    logger.success(f"Model saved to {output_path}")


def handle_eval(args: argparse.Namespace) -> None:
    """Handle the eval command to evaluate a model."""

    # Configure logging level
    if args.verbose:
        logger.info("Starting model evaluation with parameters:")
        logger.info(f"  Model path: {args.model_path}")
        logger.info(f"  Training data: {args.train_data}")
        logger.info(f"  Train split: {args.train_split}")

    # Load the model
    model_path = Path(args.model_path)
    assert model_path.exists(), f"Model file not found: {model_path}"

    logger.info(f"Loading model from {model_path}...")
    model, config = load(model_path)
    assert config is not None, "Loaded model missing MLPTrainingConfig; retrain with the latest pipeline"
    train_data_path = Path(args.train_data)
    assert train_data_path.exists(), f"Training data not found: {train_data_path}"

    # Evaluate the model
    logger.info("Evaluating model performance...")
    training_data = load_training_data(train_data_path)
    metrics = evaluate(model, training_data, config)
    logger.success(f"MAP@3 Score: {metrics['validation_map@3']:.4f}")
    if args.verbose:
        logger.info(f"Evaluation complete. Model achieved MAP@3 score of {metrics['validation_map@3']:.4f}")


def main() -> None:
    """Main entry point for CLI execution."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) is None:
        parser.print_help()
        sys.exit(1)

    _dispatch_command(args, parser)


if __name__ == "__main__":
    main()
