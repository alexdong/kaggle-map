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

import numpy as np
import pandas as pd
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
    Prediction,
    QuestionId,
    SubmissionRow,
    TrainingConfig,
    TrainingRow,
)
from kaggle_map.embeddings import encode
from kaggle_map.mlp.dataset import DatasetArrays, MLPDataset
from kaggle_map.mlp.label_encoder import LabelEncoders
from kaggle_map.mlp.loss import ListMLELoss
from kaggle_map.mlp.model import EvaluationResult, QuestionSpecificMLP
from kaggle_map.mlp.trainer import TrainingSetup, train_model
from kaggle_map.utils.device import get_device
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3

configure_logger(__name__)

# Maximum predictions per question as required by Kaggle MAP competition format
# MAP@3 evaluation metric requires exactly 3 predictions per question
MAX_PREDICTIONS = 3

# Default configuration instances to avoid None checks
_DEFAULT_TRAINING_CONFIG = TrainingConfig()


def _extract_question_predictions(training_data: list[TrainingRow]) -> dict[QuestionId, list[str]]:
    """Extract unique prediction strings per question."""
    question_predictions = defaultdict(list)
    for row in training_data:
        pred_str = str(row.prediction)
        question_predictions[row.question_id].append(pred_str)

    return {qid: list(set(preds)) for qid, preds in question_predictions.items()}


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


def fit(config: TrainingConfig = _DEFAULT_TRAINING_CONFIG) -> tuple[QuestionSpecificMLP, TrainingConfig]:
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

    logger.info("Computing embeddings...")

    # First pass: Create all EvaluationRows and collect metadata
    eval_rows = []
    metadata = []
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
        metadata.append((row.question_id, str(row.prediction), row.mc_answer))

    # Second pass: Batch encode all rows at once
    logger.info(f"Batch encoding {len(eval_rows)} rows...")
    embeddings_tensor = encode(eval_rows, strategy, config.embedding_model)
    embeddings = embeddings_tensor.cpu().numpy()

    # Unpack metadata using modern unpacking with walrus operator
    if not (metadata_unpacked := list(zip(*metadata, strict=True))):
        msg = "No metadata to process"
        raise ValueError(msg)
    question_ids, predictions, mc_answers = map(np.array, metadata_unpacked)

    embedding_dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    model = QuestionSpecificMLP(
        question_predictions,
        embedding_dim=embedding_dim,
        architecture_size=config.architecture_size,
        dropout=config.dropout,
        activation=config.activation,
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


def predict(
    model: QuestionSpecificMLP, evaluation_row: EvaluationRow, config: TrainingConfig | None = None
) -> SubmissionRow:
    """Single-row prediction interface delegating to efficient batch processing.

    Delegates to predict_batch() to avoid code duplication and ensure consistent
    encoding behavior between single and batch predictions.

    Args:
        model: Trained MLP model
        evaluation_row: Input data for prediction
        config: Embedding configuration (defaults to model's training settings)

    Returns:
        Submission row with top 3 misconception predictions
    """
    # Single-item batch ensures consistent embedding behavior across prediction modes
    results = predict_batch(model, [evaluation_row], config)
    assert len(results) == 1, "Expected exactly one prediction result"
    return results[0]


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
    model: QuestionSpecificMLP, evaluation_rows: list[EvaluationRow], config: TrainingConfig | None = None
) -> list[SubmissionRow]:
    """Batch prediction optimizing expensive embedding computation.

    Key optimization: computes embeddings for all rows in a single batch call
    rather than individual encoding. This dramatically reduces overhead from
    model loading and GPU memory transfers for large prediction sets.

    Args:
        model: Trained MLP model
        evaluation_rows: Input data for predictions
        config: Embedding configuration (defaults to GOAL_DRIVEN/GEMMA)

    Returns:
        Submission rows with top 3 misconception predictions per input
    """
    # Preconditions - enforce function contract

    if not evaluation_rows:
        return []

    # Amortize expensive data loading across all predictions in batch
    logger.debug("Loading training data for batch prediction")
    training_data = load_training_data(TrainingConfig().train_csv_path)
    correct_answers = extract_correct_answers(training_data)

    device = get_device()
    logger.debug(f"Device selection: using {device}")

    # Determine embedding configuration
    embedding_strategy = config.embedding_strategy if config else EmbeddingStrategy.GOAL_DRIVEN
    embedding_model = config.embedding_model if config else EmbeddingModel.QWEN
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
    assert embeddings.dim() == 2, f"Expected 2D embeddings, got {embeddings.dim()}D"

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


def evaluate(
    model: QuestionSpecificMLP, test_data: list[TrainingRow], config: TrainingConfig | None = None
) -> dict[str, float]:
    """Calculate MAP@3 score by comparing predictions to ground truth.

    Converts training data to evaluation format, generates predictions via
    batch processing, then computes Mean Average Precision at 3 for each
    question's misconception ranking.

    Args:
        model: Trained MLP model
        test_data: Training rows with ground truth predictions
        config: Embedding configuration for prediction generation

    Returns:
        Dictionary containing 'validation_map@3' score and sample count
    """
    # Preconditions - fail fast on invalid inputs

    if not test_data:
        return {
            "validation_map@3": 0.0,
            "validation_samples": 0,
        }

    # Convert training rows to evaluation rows using comprehension
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

    # Generate predictions for MAP@3 calculation
    logger.info(f"Starting batch evaluation of {len(test_data)} samples")
    predictions = predict_batch(model, eval_rows, config)

    # Validate prediction integrity before scoring
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


def save(model: QuestionSpecificMLP, filepath: Path, config: TrainingConfig | None = None) -> None:
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
    total_input_dim = first_layer.in_features
    embedding_dim = total_input_dim - 32  # Subtract correctness embedding dimension

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
            "embedding_model": config.embedding_model.value if config else EmbeddingModel.QWEN.value,
            "embedding_strategy": config.embedding_strategy.value if config and hasattr(config, 'embedding_strategy') else "goal_driven",
        },
        "question_predictions": question_predictions,
    }
    torch.save(save_dict, filepath)


def load(filepath: Path) -> tuple[QuestionSpecificMLP, TrainingConfig | None]:
    """Load model from disk with configuration.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {filepath}")
    assert filepath.exists(), f"Model file not found: {filepath}"

    checkpoint = torch.load(filepath, weights_only=False)

    # Handle both old (state_dict only) and new (with config) formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # New format with configuration
        state_dict = checkpoint["state_dict"]
        config = checkpoint.get("config", {})
        embedding_dim = config.get("embedding_dim")
        question_predictions = checkpoint.get("question_predictions")

        # If question_predictions not in checkpoint, load from training data
        if question_predictions is None:
            training_data = load_training_data(TrainingConfig().train_csv_path)
            question_predictions = _extract_question_predictions(training_data)
    else:
        # Old format - just state_dict
        state_dict = checkpoint

        # Try to infer embedding dimension from the first layer
        # trunk.0.weight shape is [hidden_dim, input_dim]
        if "trunk.0.weight" in state_dict:
            total_input_dim = state_dict["trunk.0.weight"].shape[1]
            embedding_dim = total_input_dim - 32  # Subtract correctness embedding
        else:
            # Fallback to old hardcoded logic
            config = TrainingConfig()
            embedding_dim = 4096  # GOAL_DRIVEN and SEMANTIC both use single embedding

        # Load question predictions from training data
        training_data = load_training_data(TrainingConfig().train_csv_path)
        question_predictions = _extract_question_predictions(training_data)

    # Create model with detected/loaded dimensions and config
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        architecture_size = ArchitectureSize(config.get("architecture_size", "xlarge"))
        dropout = config.get("dropout", 0.3)
        activation = ActivationType(config.get("activation", "gelu"))
    else:
        # Use defaults for old format
        architecture_size = ArchitectureSize.XLARGE
        dropout = 0.3
        activation = ActivationType.GELU

    model = QuestionSpecificMLP(
        question_predictions,
        embedding_dim=embedding_dim,
        architecture_size=architecture_size,
        dropout=dropout,
        activation=activation,
    )

    model.load_state_dict(state_dict)

    # Move model to appropriate device
    device = get_device()
    model = model.to(device)
    logger.debug(f"Model loaded and moved to device: {device}")

    # Reconstruct config if available
    loaded_config = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config_dict = checkpoint["config"]
        loaded_config = TrainingConfig(
            embedding_dim=embedding_dim,
            architecture_size=ArchitectureSize(config_dict.get("architecture_size", "xlarge")),
            dropout=config_dict.get("dropout", 0.3),
            activation=ActivationType(config_dict.get("activation", "gelu")),
            embedding_model=EmbeddingModel(config_dict.get("embedding_model", "qwen")),
            # Handle backward compatibility: semantic -> goal_driven
            embedding_strategy=EmbeddingStrategy(
                "goal_driven"
                if config_dict.get("embedding_strategy") == "semantic"
                else config_dict.get("embedding_strategy", "double_blind")
            ),
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

    # Create TrainingConfig from arguments
    config = TrainingConfig(
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
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    model, config = load(model_path)

    # Load training data for evaluation
    train_data_path = Path(args.train_data)
    if not train_data_path.exists():
        logger.error(f"Training data not found: {train_data_path}")
        sys.exit(1)

    # Evaluate the model
    logger.info("Evaluating model performance...")
    training_data = load_training_data(train_data_path)
    metrics = evaluate(model, training_data, config)

    logger.success(f"MAP@3 Score: {metrics['validation_map@3']:.4f}")
    if args.verbose:
        logger.info(f"Evaluation complete. Model achieved MAP@3 score of {metrics['validation_map@3']:.4f}")


def handle_predict(args: argparse.Namespace) -> None:
    """Handle the predict command to generate predictions."""

    # Configure logging level
    if args.verbose:
        logger.info("Starting prediction generation with parameters:")
        logger.info(f"  Model path: {args.model_path}")
        logger.info(f"  Input file: {args.input_file}")
        logger.info(f"  Output file: {args.output_file}")

    # Load the model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    model, config = load(model_path)

    # Load input data
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading input data from {input_path}...")
    # Read the CSV to get evaluation rows
    df = pd.read_csv(input_path)

    # Convert to EvaluationRow objects using comprehension
    # Handle both test.csv format (row_id) and expected format (id)
    eval_rows = [
        EvaluationRow(
            row_id=row.get("row_id", row.get("id")),
            question_id=QuestionId(row["QuestionId"]),
            question_text=row.get("QuestionText", ""),
            mc_answer=row.get("MC_Answer", row.get("MCAnswer", "")),
            student_explanation=row.get("StudentExplanation", ""),
            # These fields may not exist in test.csv, use empty strings as defaults
            construct_name=row.get("ConstructName", ""),
            subject_name=row.get("SubjectName", ""),
            correct_answer=row.get("CorrectAnswer", row.get("MC_Answer", "")),  # Use MC_Answer as correct answer if CorrectAnswer not present
            wrong_answer=row.get("WrongAnswer", ""),
        )
        for _, row in df.iterrows()
    ]

    # Generate predictions using batch processing for efficiency
    logger.info(f"Generating predictions for {len(eval_rows)} rows...")
    predictions = predict_batch(model, eval_rows, config)

    # Convert predictions to submission format using comprehension
    submission_rows = [
        {
            "row_id": pred.row_id,  # Use row_id to match test.csv format
            "prediction": " ".join(str(p) for p in pred.predicted_categories[:MAX_PREDICTIONS]),
        }
        for pred in predictions
    ]

    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}")

    if args.verbose:
        logger.info(f"Generated predictions for {len(submission_rows)} samples")


def main() -> None:
    """Main entry point for CLI execution."""
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

  # Generate predictions for submission
  python -m kaggle_map.mlp predict --input-file test.csv --output-file submission.csv
        """,
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fit command
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

    # Eval command
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

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp.pkl",
        help="Path to the saved model (default: models/mlp.pkl)",
    )
    predict_parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file with test data",
    )
    predict_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output CSV file for predictions",
    )
    predict_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show prediction progress",
    )

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate handler
    try:
        if args.command == "fit":
            handle_fit(args)
        elif args.command == "eval":
            handle_eval(args)
        elif args.command == "predict":
            handle_predict(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during {args.command}: {e}")
        if args.verbose if hasattr(args, "verbose") else False:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
