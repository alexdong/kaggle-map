"""Training logic and data preparation for MLP strategy.

Handles all aspects of MLP model training including data parsing, preprocessing,
batch processing, and training loop execution. Uses structured return types
and proper device handling.
"""

import random
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader

from kaggle_map.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.models import Answer, Category, Misconception, QuestionId, TrainingRow
from kaggle_map.strategies.mlp.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    LOSS_WEIGHTS,
)
from kaggle_map.strategies.mlp.dataset import DatasetItem, MLPDataset
from kaggle_map.strategies.mlp.model import MLPNet
from kaggle_map.strategies.utils import get_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class TrainingData(NamedTuple):
    """Structured training data replacing complex tuple returns."""

    embeddings: np.ndarray  # Shape: (n_samples, embedding_dim)
    correctness: np.ndarray  # Shape: (n_samples,) - binary correctness labels
    misconception_labels: dict[QuestionId, np.ndarray]  # Question-specific labels
    question_ids: np.ndarray  # Shape: (n_samples,) - question ID for each sample


class ProcessedRows(NamedTuple):
    """Structured data from row processing."""

    embeddings: list[np.ndarray]  # List of embedding arrays
    correctness: list[float]  # List of correctness values
    question_ids: list[QuestionId]  # List of question IDs
    misconception_labels: dict[QuestionId, list[np.ndarray]]  # Per-question labels


class TrainingSetup(NamedTuple):
    """Structured training setup components."""

    model: MLPNet  # The neural network model
    criterions: dict[str, nn.Module]  # Loss functions for each head
    optimizer: optim.Adam  # Optimizer for training


class BatchData(NamedTuple):
    """Structured batch data from collate function."""

    features: torch.Tensor  # Shape: (batch_size, embedding_dim)
    multi_labels: dict[str, torch.Tensor]  # Multi-head labels
    question_ids: torch.Tensor  # Question IDs for batch
    indices: torch.Tensor  # Sample indices


def collate_multihead_batch(batch: list[DatasetItem], device: torch.device | None = None) -> BatchData:
    """Custom collate function for multi-head training with variable tensor shapes.

    Handles padding of misconception and category labels to max batch size,
    while keeping correctness labels consistent. Moves tensors to target device.
    Essential for DataLoader when samples have different numbers of misconceptions/categories.

    Args:
        batch: List of DatasetItem objects from dataset
        device: Target device for tensors (gets from get_device if None)

    Returns:
        BatchData with padded and device-moved tensors
    """
    if device is None:
        device = get_device()

    features = torch.stack([item.features for item in batch]).to(device)
    question_ids = torch.tensor([item.question_id for item in batch], device=device)
    indices = torch.tensor([item.sample_index for item in batch], device=device)

    # Handle multi-labels with different shapes per sample
    multi_labels = {}

    # Correctness labels (consistent shape)
    multi_labels["correctness"] = torch.stack([item.labels["correctness"] for item in batch]).to(device)

    # Misconception labels (pad to max size in batch)
    misc_labels = [item.labels["misconceptions"] for item in batch]
    if misc_labels:
        max_misc_size = max(label.size(0) for label in misc_labels)
        padded_misc = []
        for label in misc_labels:
            if label.size(0) < max_misc_size:
                padded = torch.zeros(max_misc_size, device=device)
                padded[: label.size(0)] = label.to(device)
                padded_misc.append(padded)
            else:
                padded_misc.append(label.to(device))
        multi_labels["misconceptions"] = torch.stack(padded_misc)

    logger.debug(
        "Batch collated",
        batch_size=len(batch),
        features_shape=list(features.shape),
        device=str(device),
        label_keys=list(multi_labels.keys()),
    )

    return BatchData(
        features=features,
        multi_labels=multi_labels,
        question_ids=question_ids,
        indices=indices,
    )


def set_random_seeds(seed: int) -> None:
    """Set random seeds for deterministic training."""
    logger.debug("Setting random seeds for deterministic training", seed=seed)

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        # MPS doesn't have separate seed setting, but torch.manual_seed covers it
        pass

    # Make torch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.debug(
        "Random seeds configured",
        seed=seed,
        torch_seed=torch.initial_seed(),
        numpy_deterministic=True,
    )



def extract_correct_answers(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Answer]:
    """Extract the correct answer for each question."""
    logger.debug("Starting correct answer extraction", total_rows=len(training_data))
    assert training_data, "Training data cannot be empty"

    extract_start = time.time()
    correct_answers = {}
    true_correct_count = 0
    conflicts_checked = 0

    for row in training_data:
        if row.category == Category.TRUE_CORRECT:
            true_correct_count += 1
            if row.question_id in correct_answers:
                conflicts_checked += 1
                assert correct_answers[row.question_id] == row.mc_answer, (
                    f"Conflicting correct answers for question {row.question_id}"
                )
            else:
                correct_answers[row.question_id] = row.mc_answer

    extract_duration = time.time() - extract_start
    logger.debug(
        "Correct answer extraction completed",
        questions_with_correct_answers=len(correct_answers),
        true_correct_rows=true_correct_count,
        conflict_checks=conflicts_checked,
        extraction_time_seconds=f"{extract_duration:.3f}",
    )
    assert correct_answers, "Must find at least one correct answer"
    return correct_answers


def extract_question_misconceptions(
    training_data: list[TrainingRow],
) -> dict[QuestionId, list[Misconception]]:
    """Extract unique misconceptions per question, adding NA class."""
    logger.debug("Starting misconception extraction", total_rows=len(training_data))
    assert training_data, "Training data cannot be empty"

    extract_start = time.time()
    question_misconceptions_set = defaultdict(set)
    total_misconceptions_found = 0

    for row in training_data:
        if row.misconception is not None:
            question_misconceptions_set[row.question_id].add(row.misconception)
            total_misconceptions_found += 1

    # Convert to lists and add NA class
    question_misconceptions = {}
    total_unique_misconceptions = 0

    for question_id, misconceptions in question_misconceptions_set.items():
        misconception_list = sorted(misconceptions)  # Sort for consistency
        total_unique_misconceptions += len(misconception_list)
        misconception_list.append("NA")  # Add NA class
        question_misconceptions[question_id] = misconception_list

    extract_duration = time.time() - extract_start
    avg_misconceptions_per_question = (
        total_unique_misconceptions / len(question_misconceptions) if question_misconceptions else 0
    )

    logger.debug(
        "Misconception extraction completed",
        questions_with_misconceptions=len(question_misconceptions),
        total_misconception_instances=total_misconceptions_found,
        total_unique_misconceptions=total_unique_misconceptions,
        avg_misconceptions_per_question=f"{avg_misconceptions_per_question:.2f}",
        extraction_time_seconds=f"{extract_duration:.3f}",
    )
    return question_misconceptions


def prepare_training_data(
    training_data: list[TrainingRow],
    correct_answers: dict[QuestionId, Answer],
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
) -> TrainingData:
    """Prepare embeddings and labels for training with structured return."""
    logger.info(
        "Starting training data preparation",
        training_samples=len(training_data),
        embedding_model=embedding_model.model_id,
        questions_with_misconceptions=len(question_misconceptions),
    )

    tokenizer_start = time.time()
    tokenizer = get_tokenizer(embedding_model)
    tokenizer_load_time = time.time() - tokenizer_start

    logger.debug(
        "Tokenizer loaded",
        model_id=embedding_model.model_id,
        load_time_seconds=f"{tokenizer_load_time:.3f}",
    )

    processing_start = time.time()
    processed_rows = process_training_rows(
        training_data,
        correct_answers,
        question_misconceptions,
        tokenizer,
    )
    processing_time = time.time() - processing_start

    logger.debug(
        "Training row processing completed",
        embeddings_count=len(processed_rows.embeddings),
        processing_time_seconds=f"{processing_time:.3f}",
    )

    conversion_start = time.time()
    result = convert_to_arrays(
        processed_rows,
        question_misconceptions,
    )
    conversion_time = time.time() - conversion_start

    logger.debug(
        "Array conversion completed",
        conversion_time_seconds=f"{conversion_time:.3f}",
    )

    return result


def process_training_rows(
    training_data: list[TrainingRow],
    correct_answers: dict[QuestionId, Answer],
    question_misconceptions: dict[QuestionId, list[Misconception]],
    tokenizer: "SentenceTransformer",
) -> ProcessedRows:
    """Process training rows for training with structured return."""
    logger.debug(
        "Starting training row processing",
        total_rows=len(training_data),
        tokenizer_type=type(tokenizer).__name__,
    )

    process_start = time.time()

    embeddings = []
    correctness = []
    question_ids = []
    misconception_labels = {qid: [] for qid in question_misconceptions}

    processed_embeddings = 0

    for idx, row in enumerate(training_data):
        # Generate embedding (question/answer/explanation only)
        text = repr(row)
        embedding_start = time.time()
        embedding = tokenizer.encode(text)
        embedding_time = time.time() - embedding_start

        embeddings.append(embedding)
        processed_embeddings += 1

        # Determine correctness
        is_correct = row.question_id in correct_answers and row.mc_answer == correct_answers[row.question_id]
        correctness.append(float(is_correct))
        question_ids.append(row.question_id)

        # Log progress for large datasets
        if idx > 0 and idx % 500 == 0:
            logger.debug(
                "Row processing progress",
                processed_rows=idx + 1,
                embeddings_generated=processed_embeddings,
                avg_embedding_time_ms=f"{embedding_time * 1000:.2f}",
            )

        # Create misconception label (when applicable)
        if row.question_id in question_misconceptions:
            misconception_label = create_misconception_label(row, question_misconceptions)
            misconception_labels[row.question_id].append(misconception_label)

    process_duration = time.time() - process_start
    avg_embedding_time = process_duration / max(processed_embeddings, 1)

    logger.debug(
        "Training row processing completed",
        total_processed=processed_embeddings,
        total_time_seconds=f"{process_duration:.3f}",
        avg_embedding_time_ms=f"{avg_embedding_time * 1000:.2f}",
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
    )

    return ProcessedRows(
        embeddings=embeddings,
        correctness=correctness,
        question_ids=question_ids,
        misconception_labels=misconception_labels,
    )


def create_misconception_label(
    row: TrainingRow, question_misconceptions: dict[QuestionId, list[Misconception]]
) -> np.ndarray:
    """Create multi-hot encoding for misconception label."""
    misconceptions = question_misconceptions[row.question_id]
    label = np.zeros(len(misconceptions))

    if row.misconception is not None and row.misconception in misconceptions:
        label[misconceptions.index(row.misconception)] = 1.0
    else:
        label[-1] = 1.0  # NA is always last

    return label


def convert_to_arrays(
    processed_rows: ProcessedRows,
    question_misconceptions: dict[QuestionId, list[Misconception]],
) -> TrainingData:
    """Convert processed rows to numpy arrays for training."""
    embeddings_array = np.stack(processed_rows.embeddings)
    correctness_array = np.array(processed_rows.correctness)
    question_ids_array = np.array(processed_rows.question_ids)

    # Convert misconception labels to arrays per question with padding
    max_misconception_size = max(
        (len(misconceptions) for misconceptions in question_misconceptions.values()),
        default=1,
    )

    misconception_labels = {}
    for qid, labels in processed_rows.misconception_labels.items():
        if labels:
            # Pad all labels to max_misconception_size
            padded_labels = []
            for label in labels:
                padded_label = np.zeros(max_misconception_size)
                padded_label[: len(label)] = label
                padded_labels.append(padded_label)
            misconception_labels[qid] = np.stack(padded_labels)
        else:
            misconception_labels[qid] = np.empty((0, max_misconception_size))

    logger.info(f"Generated {len(embeddings_array)} embeddings for training")

    return TrainingData(
        embeddings=embeddings_array,
        correctness=correctness_array,
        misconception_labels=misconception_labels,
        question_ids=question_ids_array,
    )


def setup_training(
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
    device: torch.device,
) -> TrainingSetup:
    """Setup multi-head model, criterions, and optimizer with structured return."""
    logger.debug(
        "Setting up training components",
        questions_with_misconceptions=len(question_misconceptions),
        target_device=str(device),
        embedding_model=embedding_model.model_id,
    )

    model_create_start = time.time()
    model = MLPNet(question_misconceptions, embedding_model).to(device)
    model_create_time = time.time() - model_create_start

    # Verify model is on correct device
    model_device = next(model.parameters()).device
    logger.debug(
        "Model created and moved to device",
        model_device=str(model_device),
        device_matches=model_device.type == device.type,
        model_parameters=sum(p.numel() for p in model.parameters()),
        creation_time_seconds=f"{model_create_time:.3f}",
    )

    # Multiple loss functions for different heads
    criterions = {
        "correctness": nn.BCEWithLogitsLoss(),  # Binary classification for correctness
        "categories": nn.CrossEntropyLoss(),  # Multi-class for category selection
        "misconceptions": nn.BCEWithLogitsLoss(),  # Multi-label for misconceptions
    }

    # Move criterions to device
    for name, criterion in criterions.items():
        criterions[name] = criterion.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

    logger.debug(
        "Training setup completed",
        criterions_count=len(criterions),
        learning_rate=DEFAULT_LEARNING_RATE,
        optimizer_params=len(list(model.parameters())),
        device=str(device),
    )

    return TrainingSetup(
        model=model,
        criterions=criterions,
        optimizer=optimizer,
    )


def train_model(
    training_data: TrainingData,
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
    device: torch.device,
) -> MLPNet:
    """Train the MLP model with structured data."""
    logger.info(
        "Starting MLP model training",
        embeddings_shape=training_data.embeddings.shape,
        correctness_shape=training_data.correctness.shape,
        question_ids_shape=training_data.question_ids.shape,
        device=str(device),
        unique_questions=len(set(training_data.question_ids)),
    )

    setup_start = time.time()
    training_setup = setup_training(question_misconceptions, embedding_model, device)
    setup_time = time.time() - setup_start

    logger.debug(
        "Model setup completed",
        model_parameters=sum(p.numel() for p in training_setup.model.parameters()),
        criterion_types=[type(c).__name__ for c in training_setup.criterions.values()],
        optimizer_type=type(training_setup.optimizer).__name__,
        setup_time_seconds=f"{setup_time:.3f}",
    )

    dataset_start = time.time()
    dataset = MLPDataset(
        training_data.embeddings,
        training_data.correctness,
        training_data.misconception_labels,
        training_data.question_ids,
    )
    dataset_time = time.time() - dataset_start

    logger.debug(
        "Dataset creation completed",
        dataset_size=len(dataset),
        dataset_time_seconds=f"{dataset_time:.3f}",
    )

    # Create device-aware collate function
    device_collate_fn = partial(collate_multihead_batch, device=device)

    dataloader = DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        collate_fn=device_collate_fn,
    )

    logger.info(
        "Starting training epochs",
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_NUM_EPOCHS,
        total_batches_per_epoch=len(dataloader),
    )

    training_start = time.time()
    train_multihead_epochs(
        training_setup.model,
        training_setup.criterions,
        training_setup.optimizer,
        dataloader,
        num_epochs=DEFAULT_NUM_EPOCHS,
    )
    training_duration = time.time() - training_start

    logger.info(
        "Multi-head MLP training completed",
        training_time_seconds=f"{training_duration:.3f}",
        epochs_completed=DEFAULT_NUM_EPOCHS,
        final_model_parameters=sum(p.numel() for p in training_setup.model.parameters()),
    )
    return training_setup.model


def train_multihead_epochs(
    model: MLPNet,
    criterions: dict[str, nn.Module],
    optimizer: optim.Adam,
    dataloader: DataLoader,
    num_epochs: int,
) -> None:
    """Train multi-head model for specified number of epochs."""
    model.train()
    for epoch in range(num_epochs):
        total_losses = train_multihead_single_epoch(model, criterions, optimizer, dataloader)

        if epoch % 20 == 0:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in total_losses.items()])
            logger.debug(f"Epoch {epoch}, Losses: {loss_str}")


def train_multihead_single_epoch(
    model: MLPNet,
    criterions: dict[str, nn.Module],
    optimizer: optim.Adam,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Train multi-head model for a single epoch."""
    total_losses = {
        "correctness": 0.0,
        "categories": 0.0,
        "misconceptions": 0.0,
        "combined": 0.0,
    }
    num_batches = 0

    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_losses = process_multihead_batch(model, criterions, batch_data)

        if any(loss > 0 for loss in batch_losses.values()):
            # Combined loss with weights
            combined_loss = (
                batch_losses["correctness"] * LOSS_WEIGHTS["correctness"]
                + batch_losses["categories"] * LOSS_WEIGHTS["categories"]
                + batch_losses["misconceptions"] * LOSS_WEIGHTS["misconceptions"]
            )

            combined_loss.backward()
            optimizer.step()

            # Track losses
            for key, loss in batch_losses.items():
                total_losses[key] += loss.item()
            total_losses["combined"] += combined_loss.item()
            num_batches += 1

    # Return average losses
    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches

    return total_losses


def process_multihead_batch(
    model: MLPNet,
    criterions: dict[str, nn.Module],
    batch_data: BatchData,
) -> dict[str, torch.Tensor]:
    """Process a single batch for multi-head training."""
    device = batch_data.features.device
    model_device = next(model.parameters()).device
    assert model_device.type == device.type, (
        f"Model device type mismatch: expected {device.type}, got {model_device.type}"
    )

    # Initialize losses
    batch_losses = {
        "correctness": torch.tensor(0.0, device=device),
        "categories": torch.tensor(0.0, device=device),
        "misconceptions": torch.tensor(0.0, device=device),
    }

    valid_samples = 0

    for i in range(len(batch_data.features)):
        qid = batch_data.question_ids[i].item()
        feature = batch_data.features[i].unsqueeze(0)  # Shape: [1, embedding_dim]

        # Get multi-head model outputs
        outputs = model(feature, qid)

        # Correctness loss (always present)
        if "correctness" in outputs and "correctness" in batch_data.multi_labels:
            correctness_target = batch_data.multi_labels["correctness"][i].unsqueeze(0)
            correctness_loss = criterions["correctness"](outputs["correctness"], correctness_target)
            batch_losses["correctness"] += correctness_loss

        # Misconception loss (when applicable)
        if "misconceptions" in outputs and "misconceptions" in batch_data.multi_labels:
            misconception_target = batch_data.multi_labels["misconceptions"][i].unsqueeze(0)
            if misconception_target.sum() > 0:  # Only if there are actual misconceptions
                misconception_loss = criterions["misconceptions"](outputs["misconceptions"], misconception_target)
                batch_losses["misconceptions"] += misconception_loss

        valid_samples += 1

    # Average over batch
    if valid_samples > 0:
        for key in batch_losses:  # noqa: PLC0206
            batch_losses[key] = batch_losses[key] / valid_samples

    return batch_losses
