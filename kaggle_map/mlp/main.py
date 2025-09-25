"""MLP module entry point for misconception prediction.

This module provides both a Python API and Click-powered command-line interface
for training, evaluating, and using Multi-Layer Perceptron models for student
misconception prediction. Run ``uv run -m kaggle_map.mlp.main --help`` for
usage, and ``uv run -m kaggle_map.mlp.main fit`` to train with defaults.
"""

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import torch
from loguru import logger
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset

from kaggle_map.core.models import (
    Answer,
    Category,
    EvaluationRow,
    MLPTrainingConfig,
    Prediction,
    QuestionId,
    SubmissionRow,
    TrainingRow,
    default_mlp_training_config,
)
from kaggle_map.core.random_seed import configure_random_seed
from kaggle_map.dataloader import MAPDataset
from kaggle_map.dataloader.dataset import extract_correct_answers, load_training_data
from kaggle_map.embeddings import encode
from kaggle_map.mlp.checkpoint import load_checkpoint, save_checkpoint
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

configure_logger(__name__)

# Maximum predictions per question as required by Kaggle MAP competition format
# MAP@3 evaluation metric requires exactly 3 predictions per question
MAX_PREDICTIONS = 3

DEFAULT_MODEL_PATH = Path("models/mlp.pkl")


class _EmbeddingBatchDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset wrapping embedding tensors for training."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        question_ids: torch.Tensor,
        labels: torch.Tensor,
        is_correct: torch.Tensor,
    ) -> None:
        assert embeddings.size(0) == question_ids.size(0) == labels.size(0) == is_correct.size(0), (
            "All tensors must share the same leading dimension"
        )
        self._embeddings = embeddings
        self._question_ids = question_ids
        self._labels = labels
        self._is_correct = is_correct

    def __len__(self) -> int:
        return int(self._embeddings.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": self._embeddings[index],
            "question_id": self._question_ids[index],
            "label": self._labels[index],
            "is_correct": self._is_correct[index],
        }


def _build_embedding_inputs(dataset: MAPDataset) -> tuple[list[EvaluationRow], list[tuple[int, str, str]]]:
    correct_answers = dataset.correct_answers
    eval_rows: list[EvaluationRow] = []
    metadata: list[tuple[int, str, str]] = []
    for row in dataset:
        eval_rows.append(
            EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
                correct_answer=correct_answers.get(row.question_id, ""),
            )
        )
        metadata.append((row.question_id, str(row.prediction), row.mc_answer))
    return eval_rows, metadata


def _materialize_training_tensors(
    dataset: MAPDataset,
    config: MLPTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eval_rows, metadata = _build_embedding_inputs(dataset)
    embeddings_np, question_ids_np, _predictions, _mc_answers = get_or_compute_embeddings(
        eval_rows,
        metadata,
        dataset.csv_path,
        config.embedding_model,
        config.embedding_strategy,
    )

    embeddings = torch.from_numpy(embeddings_np).float()
    question_ids = torch.as_tensor(question_ids_np, dtype=torch.long)
    is_correct = torch.as_tensor(
        [1 if row.category.is_correct_answer else 0 for row in dataset],
        dtype=torch.long,
    )
    return embeddings, question_ids, is_correct


def _encode_labels(dataset: MAPDataset, label_encoders: LabelEncoders) -> torch.Tensor:
    encoded = [
        label_encoders.encode(
            row.question_id,
            str(row.prediction),
            is_correct=row.category.is_correct_answer,
        )
        for row in dataset
    ]
    return torch.as_tensor(encoded, dtype=torch.long)


def _build_training_loaders(
    dataset: MAPDataset,
    label_encoders: LabelEncoders,
    config: MLPTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    embeddings, question_ids, is_correct = _materialize_training_tensors(dataset, config)
    labels = _encode_labels(dataset, label_encoders)

    base_dataset = _EmbeddingBatchDataset(embeddings, question_ids, labels, is_correct)

    split_indices = dataset.split_indices
    train_subset = Subset(base_dataset, split_indices["train"])
    val_subset = Subset(base_dataset, split_indices["val"])

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def fit(config: MLPTrainingConfig) -> QuestionSpecificMLP:
    configure_random_seed()
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)
    question_predictions = dataset.question_predictions

    device = get_device()
    logger.info(f"Training on {device} with embedding strategy: {config.embedding_strategy.value}")

    model = QuestionSpecificMLP(
        question_predictions,
        embedding_model=config.embedding_model,
        embedding_strategy=config.embedding_strategy,
        architecture_size=config.architecture_size,
        dropout=config.dropout,
        activation=config.activation,
        correctness_embedding_dim=CORRECTNESS_EMBEDDING_DIMENSIONS,
    )
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    label_encoders = LabelEncoders(
        true_label_encoders=model.true_label_encoders,
        false_label_encoders=model.false_label_encoders,
    )

    split_counts = dataset.split_counts
    logger.info(
        "Data split - Train: %d, Val: %d, Test: %d",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )

    train_loader, val_loader = _build_training_loaders(dataset, label_encoders, config)

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


def _process_single_prediction(
    model: QuestionSpecificMLP,
    eval_row: EvaluationRow,
    embedding_tensor: torch.Tensor,
    correct_answers: Mapping[QuestionId, Answer],
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
    model: QuestionSpecificMLP,
    evaluation_rows: Sequence[EvaluationRow],
    config: MLPTrainingConfig,
) -> list[SubmissionRow]:
    assert evaluation_rows, "Evaluation rows list is empty"

    # Amortize expensive data loading across all predictions in batch
    logger.debug("Loading training data for batch prediction")
    training_data = load_training_data(config.train_csv_path)
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
    model: QuestionSpecificMLP,
    test_data: Sequence[TrainingRow],
    config: MLPTrainingConfig,
) -> dict[str, float]:
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


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--seed", type=int, default=None, help="Override random seed for this run", show_default=False)
def cli(seed: int | None) -> None:
    """Train and evaluate MLP models for misconception prediction."""

    active_seed = configure_random_seed(override=seed)
    logger.debug("CLI configured random seed: {}", active_seed)


@cli.command(name="fit")
def fit_command() -> None:
    """Train an MLP model and persist it to ``models/mlp.pkl``."""

    config = default_mlp_training_config()
    assert config.learning_rate > 0.0, "Learning rate must be positive"

    model_path = DEFAULT_MODEL_PATH
    train_data = config.train_csv_path

    logger.info("Starting model training with parameters:")
    logger.info(f"  Training data: {train_data}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Train split: {config.train_split}")
    logger.info(f"  Random seed: {config.random_seed}")
    logger.info(f"  Model path: {model_path}")

    assert train_data.exists(), f"Training data not found: {train_data}"

    try:
        logger.info("Loading training data...")
        model = fit(config)
    except KeyboardInterrupt as exc:
        logger.info("Operation cancelled by user")
        raise click.Abort() from exc

    output_path = model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_path, config)
    logger.success(f"Model saved to {output_path}")


@cli.command(name="eval")
def eval_command() -> None:
    """Evaluate the default MLP checkpoint against the training data."""

    model_path = DEFAULT_MODEL_PATH
    config_defaults = default_mlp_training_config()
    train_data = config_defaults.train_csv_path

    logger.info("Starting model evaluation with parameters:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Training data: {train_data}")
    logger.info(f"  Train split: {config_defaults.train_split}")

    logger.info(f"Loading model from {model_path}...")

    try:
        model, config = load_checkpoint(model_path)
    except KeyboardInterrupt as exc:
        logger.info("Operation cancelled by user")
        raise click.Abort() from exc

    logger.info("Evaluating model performance...")
    assert train_data.exists(), f"Training data not found: {train_data}"
    training_data = load_training_data(train_data)
    metrics = evaluate(model, training_data, config)
    logger.success(f"MAP@3 Score: {metrics['validation_map@3']:.4f}")
    logger.info(f"Evaluation complete. Model achieved MAP@3 score of {metrics['validation_map@3']:.4f}")


def main() -> None:
    """Module entrypoint for ``python -m kaggle_map.mlp``."""

    cli()


if __name__ == "__main__":
    main()
