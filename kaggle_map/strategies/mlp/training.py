"""Training logic and data preparation for MLP strategy.

Handles all aspects of MLP model training including data parsing, preprocessing,
batch processing, and training loop execution. Uses structured return types
and proper device handling.
"""

import random
from functools import partial
from typing import NamedTuple

import numpy as np
import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader

from kaggle_map.dataset import extract_misconceptions_by_popularity
from kaggle_map.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.models import Answer, Misconception, QuestionId, TrainingRow
from kaggle_map.strategies.mlp.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    LOSS_WEIGHTS,
)
from kaggle_map.strategies.mlp.dataset import MLPDataset
from kaggle_map.strategies.mlp.model import MLPNet
from kaggle_map.strategies.utils import BatchData, collate_multihead_batch


class TrainingData(NamedTuple):
    """Structured training data replacing complex tuple returns."""

    embeddings: np.ndarray  # Shape: (n_samples, embedding_dim)
    correctness: np.ndarray  # Shape: (n_samples,) - binary correctness labels
    misconception_labels: dict[QuestionId, np.ndarray]  # Question-specific labels
    question_ids: np.ndarray  # Shape: (n_samples,) - question ID for each sample




class TrainingSetup(NamedTuple):
    """Structured training setup components."""

    model: MLPNet  # The neural network model
    criterions: dict[str, nn.Module]  # Loss functions for each head
    optimizer: optim.Adam  # Optimizer for training






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




def extract_question_misconceptions(
    training_data: list[TrainingRow],
) -> dict[QuestionId, list[Misconception]]:
    """Extract unique misconceptions per question, adding NA class for MLP training."""
    assert training_data, "Training data cannot be empty"

    # Use the dataset function and add NA class for MLP compatibility
    misconceptions_by_popularity = extract_misconceptions_by_popularity(training_data)

    # Convert to alphabetically sorted lists with NA class (needed for consistent MLP training)
    question_misconceptions = {}
    for question_id, misconceptions in misconceptions_by_popularity.items():
        if misconceptions:
            # Sort alphabetically for consistency, then add NA
            misconception_list = sorted(misconceptions)
            misconception_list.append("NA")
            question_misconceptions[question_id] = misconception_list
        else:
            question_misconceptions[question_id] = ["NA"]

    logger.debug(f"Extracted misconceptions for {len(question_misconceptions)} questions")
    return question_misconceptions


def prepare_training_data(
    training_data: list[TrainingRow],
    correct_answers: dict[QuestionId, Answer],
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
) -> TrainingData:
    """Prepare embeddings and labels for training."""
    logger.info(f"Preparing training data with {len(training_data)} samples")

    tokenizer = get_tokenizer(embedding_model)

    # Process all rows
    embeddings = []
    correctness = []
    question_ids = []
    misconception_labels = {qid: [] for qid in question_misconceptions}

    for row in training_data:
        # Generate embedding
        text = repr(row)
        embedding = tokenizer.encode(text)
        embeddings.append(embedding)

        # Determine correctness
        is_correct = row.question_id in correct_answers and row.mc_answer == correct_answers[row.question_id]
        correctness.append(float(is_correct))
        question_ids.append(row.question_id)

        # Create misconception label (when applicable)
        if row.question_id in question_misconceptions:
            misconceptions = question_misconceptions[row.question_id]
            label = np.zeros(len(misconceptions))

            if row.misconception is not None and row.misconception in misconceptions:
                label[misconceptions.index(row.misconception)] = 1.0
            else:
                label[-1] = 1.0  # NA is always last

            misconception_labels[row.question_id].append(label)

    # Convert to arrays
    embeddings_array = np.stack(embeddings)
    correctness_array = np.array(correctness)
    question_ids_array = np.array(question_ids)

    # Convert misconception labels to padded arrays
    max_misconception_size = max(
        (len(misconceptions) for misconceptions in question_misconceptions.values()), 
        default=1
    )

    padded_misconception_labels = {}
    for qid, labels in misconception_labels.items():
        if labels:
            padded_labels = []
            for label in labels:
                padded_label = np.zeros(max_misconception_size)
                padded_label[: len(label)] = label
                padded_labels.append(padded_label)
            padded_misconception_labels[qid] = np.stack(padded_labels)
        else:
            padded_misconception_labels[qid] = np.empty((0, max_misconception_size))

    logger.debug(f"Processed {len(embeddings_array)} samples")

    return TrainingData(
        embeddings=embeddings_array,
        correctness=correctness_array,
        misconception_labels=padded_misconception_labels,
        question_ids=question_ids_array,
    )


def setup_training(
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
    device: torch.device,
) -> TrainingSetup:
    """Setup multi-head model, criterions, and optimizer."""
    model = MLPNet(question_misconceptions, embedding_model).to(device)

    criterions = {
        "correctness": nn.BCEWithLogitsLoss().to(device),
        "categories": nn.CrossEntropyLoss().to(device),
        "misconceptions": nn.BCEWithLogitsLoss().to(device),
    }

    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

    logger.debug(f"Setup training on {device} with {sum(p.numel() for p in model.parameters())} parameters")

    return TrainingSetup(model=model, criterions=criterions, optimizer=optimizer)


def train_model(
    training_data: TrainingData,
    question_misconceptions: dict[QuestionId, list[Misconception]],
    embedding_model: EmbeddingModel,
    device: torch.device,
) -> MLPNet:
    """Train the MLP model."""
    logger.info(f"Starting MLP training with {training_data.embeddings.shape[0]} samples")

    training_setup = setup_training(question_misconceptions, embedding_model, device)

    dataset = MLPDataset(
        training_data.embeddings,
        training_data.correctness,
        training_data.misconception_labels,
        training_data.question_ids,
    )

    device_collate_fn = partial(collate_multihead_batch, device=device)
    dataloader = DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, collate_fn=device_collate_fn)

    train_multihead_epochs(
        training_setup.model,
        training_setup.criterions,
        training_setup.optimizer,
        dataloader,
        num_epochs=DEFAULT_NUM_EPOCHS,
    )

    logger.info(f"MLP training completed after {DEFAULT_NUM_EPOCHS} epochs")
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
