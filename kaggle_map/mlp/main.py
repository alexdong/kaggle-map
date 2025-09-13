"""MLP module main entry point for misconception prediction.

This module provides both a Python API and command-line interface for training,
evaluating, and using Multi-Layer Perceptron models for student misconception prediction.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.nn import functional
from torch.utils.data import DataLoader

from kaggle_map.core.dataset import extract_correct_answers, load_training_data
from kaggle_map.core.models import (
    RANDOM_SEED,
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


def fit(config: TrainingConfig = _DEFAULT_TRAINING_CONFIG) -> QuestionSpecificMLP:
    """Train an MLP model for misconception prediction.

    Args:
        config: Training configuration

    Returns:
        Trained QuestionSpecificMLP model
    """
    strategy = config.embedding_strategy

    device = get_device()
    logger.info(f"Training on {device} with embedding strategy: {strategy.value}")

    training_data = load_training_data(config.train_csv_path)
    correct_answers = extract_correct_answers(training_data)
    question_predictions = _extract_question_predictions(training_data)

    logger.info("Computing embeddings...")

    # Process all rows to extract embeddings and metadata
    processed_rows = []
    for row in training_data:
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
            correct_answer=correct_answers.get(row.question_id, ""),
        )
        embedding = encode(eval_row, strategy, config.embedding_model)
        processed_rows.append((embedding.numpy(), row.question_id, str(row.prediction), row.mc_answer))

    # Unpack into arrays using zip
    embeddings_list, question_ids_list, predictions_list, mc_answers_list = map(
        list, zip(*processed_rows, strict=False)
    )
    embeddings = np.array(embeddings_list)
    question_ids = np.array(question_ids_list)
    predictions = np.array(predictions_list)
    mc_answers = np.array(mc_answers_list)

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

    return result.model


def predict(model: QuestionSpecificMLP, evaluation_row: EvaluationRow) -> SubmissionRow:
    """Predict misconceptions for a single evaluation row.

    Args:
        model: Trained MLP model
        evaluation_row: Input data for prediction

    Returns:
        Submission row with top 3 predictions
    """
    # Load training data to get correct answers
    training_data = load_training_data(TrainingConfig().train_csv_path)
    correct_answers = extract_correct_answers(training_data)
    device = get_device()
    embedding_strategy = EmbeddingStrategy.DOUBLE_BLIND  # Default, could be made configurable

    eval_row_with_answer = EvaluationRow(
        row_id=evaluation_row.row_id,
        question_id=evaluation_row.question_id,
        question_text=evaluation_row.question_text,
        mc_answer=evaluation_row.mc_answer,
        student_explanation=evaluation_row.student_explanation,
        correct_answer=correct_answers.get(evaluation_row.question_id, ""),
    )

    # Use default embedding model (QWEN) - could be made configurable
    embedding_model = EmbeddingModel.QWEN
    embedding = encode(eval_row_with_answer, embedding_strategy, embedding_model)
    embedding = embedding.numpy()
    logger.debug(f"Computing embedding for question {evaluation_row.question_id}, embedding_dim={len(embedding)}")
    embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)

    is_correct = evaluation_row.mc_answer == correct_answers.get(evaluation_row.question_id, "")
    logger.debug(
        f"Question {evaluation_row.question_id}: is_correct={is_correct}, mc_answer='{evaluation_row.mc_answer}'"
    )
    correctness_idx = torch.tensor([1 if is_correct else 0], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            embedding_tensor,
            torch.LongTensor([evaluation_row.question_id]).to(device),
            correctness_idx,
        )

    predictions = []
    key = EvaluationResult(question_id=evaluation_row.question_id, is_correct=is_correct)

    if key in outputs:
        logits = outputs[key]
        probs = functional.softmax(logits, dim=-1)[0]
        top_k = min(MAX_PREDICTIONS, logits.size(-1))
        top_indices = torch.topk(probs, k=top_k)[1]
        logger.debug(f"Model outputs for {key}: top_{top_k}_probs={probs[top_indices].tolist()}")

        encoder = (
            model.true_label_encoders.get(evaluation_row.question_id)
            if is_correct
            else model.false_label_encoders.get(evaluation_row.question_id)
        )

        if encoder:
            predictions.extend(
                Prediction.from_string(pred_str) for pred_str in encoder.inverse_transform(top_indices.cpu().numpy())
            )

    default = Prediction(
        category=Category.TRUE_NEITHER if is_correct else Category.FALSE_NEITHER,
        misconception="NA",
    )
    while len(predictions) < MAX_PREDICTIONS:
        predictions.append(default)

    prediction_strs = [str(p) for p in predictions[:MAX_PREDICTIONS]]
    logger.debug(f"Final predictions for row {evaluation_row.row_id}: {prediction_strs}")
    return SubmissionRow(row_id=evaluation_row.row_id, predicted_categories=predictions[:MAX_PREDICTIONS])


def evaluate(model: QuestionSpecificMLP, test_data: list[TrainingRow]) -> dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Trained MLP model
        test_data: Test data rows

    Returns:
        Dictionary with evaluation metrics
    """
    map_scores = []
    for row in test_data:
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )
        prediction = predict(model, eval_row)
        score = calculate_map_at_3(row.prediction, prediction.predicted_categories)
        map_scores.append(score)

    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    logger.info(f"Evaluation MAP@3: {avg_map:.4f} on {len(test_data)} samples")

    return {
        "validation_map@3": avg_map,
        "validation_samples": len(test_data),
    }


def save(model: QuestionSpecificMLP, filepath: Path) -> None:
    """Save model to disk.

    Args:
        model: Model to save
        filepath: Path to save the model
    """
    logger.info(f"Saving model to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load(filepath: Path) -> QuestionSpecificMLP:
    """Load model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {filepath}")
    assert filepath.exists(), f"Model file not found: {filepath}"

    # Need to reconstruct the model architecture
    # This requires knowing the question predictions, which we get from training data
    training_data = load_training_data(TrainingConfig().train_csv_path)
    question_predictions = _extract_question_predictions(training_data)

    # Use default config for architecture (could be saved with model for better reconstruction)
    config = TrainingConfig()
    embedding_dim = 8192 if config.embedding_strategy == EmbeddingStrategy.DOUBLE_BLIND else 4096

    model = QuestionSpecificMLP(
        question_predictions,
        embedding_dim=embedding_dim,
        architecture_size=config.architecture_size,
        dropout=config.dropout,
        activation=config.activation,
    )

    model.load_state_dict(torch.load(filepath))
    return model


def handle_fit(args: "argparse.Namespace") -> None:
    """Handle the fit command to train a model."""
    from pathlib import Path

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
    model = fit(config)

    # Save the model
    output_path = Path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save(model, output_path)
    logger.success(f"Model saved to {output_path}")


def handle_eval(args: "argparse.Namespace") -> None:
    """Handle the eval command to evaluate a model."""
    import sys
    from pathlib import Path

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
    model = load(model_path)

    # Load training data for evaluation
    train_data_path = Path(args.train_data)
    if not train_data_path.exists():
        logger.error(f"Training data not found: {train_data_path}")
        sys.exit(1)

    # Evaluate the model
    logger.info("Evaluating model performance...")
    training_data = load_training_data(train_data_path)
    map_score = evaluate(model, training_data, train_split=args.train_split)

    logger.success(f"MAP@3 Score: {map_score:.4f}")
    if args.verbose:
        logger.info(f"Evaluation complete. Model achieved MAP@3 score of {map_score:.4f}")


def handle_predict(args: "argparse.Namespace") -> None:
    """Handle the predict command to generate predictions."""
    import sys
    from pathlib import Path

    import pandas as pd

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
    model = load(model_path)

    # Load input data
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading input data from {input_path}...")
    # Read the CSV to get evaluation rows
    df = pd.read_csv(input_path)

    # Convert to EvaluationRow objects
    eval_rows = []
    for _, row in df.iterrows():
        eval_row = EvaluationRow(
            row_id=row["id"],
            question_id=QuestionId(row["QuestionId"]),
            construct_name=row["ConstructName"],
            subject_name=row["SubjectName"],
            correct_answer=row["CorrectAnswer"],
            wrong_answer=row["WrongAnswer"],
            mc_answer=row.get("MCAnswer", None),
        )
        eval_rows.append(eval_row)

    # Generate predictions
    logger.info(f"Generating predictions for {len(eval_rows)} rows...")
    predictions = predict(model, eval_rows)

    # Convert predictions to submission format
    submission_rows = []
    for pred in predictions:
        submission_rows.append({
            "id": pred.row_id,
            "prediction": " ".join(pred.predictions[:3]),  # Take top 3 predictions
        })

    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}")

    if args.verbose:
        logger.info(f"Generated predictions for {len(submission_rows)} samples")


if __name__ == "__main__":
    import argparse
    import sys

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
        "-v", "--verbose",
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
        "-v", "--verbose",
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
        "-v", "--verbose",
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
