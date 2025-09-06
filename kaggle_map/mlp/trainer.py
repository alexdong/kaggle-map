"""Training utilities for MLP model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from kaggle_map.mlp.model import QuestionSpecificMLP


@dataclass
class TrainingConfig:
    """Configuration for MLP training."""

    # Training parameters
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 15

    # Data splits
    train_split: float = 0.7
    random_seed: int = 42

    # Paths
    train_csv_path: Path = Path("datasets/train.csv")
    checkpoint_dir: Path = Path("checkpoints")

    # Architecture
    architecture_size: str = "xlarge"
    dropout: float = 0.3
    activation: str = "gelu"


def process_batch(
    model: QuestionSpecificMLP,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    training: bool = True,
) -> tuple[torch.Tensor | None, int]:
    """Process a single batch through the model.

    Args:
        model: The MLP model
        batch: Tuple of (embeddings, question_ids, labels, is_correct)
        criterion: Loss function
        device: Device to run on
        training: Whether in training mode (affects gradient tracking)

    Returns:
        Tuple of (loss, batch_size) or (None, 0) if no valid samples
    """
    embeddings, question_ids, labels, is_correct = batch
    embeddings = embeddings.to(device)
    question_ids = question_ids.to(device)
    labels = labels.to(device)
    is_correct = is_correct.to(device)

    # Forward pass
    outputs = model(embeddings, question_ids, is_correct)

    # Calculate loss across all question heads
    total_loss = 0.0
    total_samples = 0

    for (qid, correct), logits in outputs.items():
        # Find matching samples in batch
        correctness_mask = is_correct > 0 if correct else is_correct == 0
        question_mask = question_ids == qid
        combined_mask = question_mask & correctness_mask

        if combined_mask.any():
            question_labels = labels[combined_mask]
            if logits.size(0) == question_labels.size(0):
                loss = criterion(logits, question_labels)
                total_loss += loss * logits.size(0)
                total_samples += logits.size(0)

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        # Detach for validation to save memory
        if not training:
            avg_loss = avg_loss.detach()
        return avg_loss, embeddings.size(0)

    return None, 0


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    logger.warning(f"Unknown optimizer {config.optimizer}, using AdamW")
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create learning rate scheduler based on configuration."""
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10, epochs=config.epochs, steps_per_epoch=steps_per_epoch
        )
    logger.warning(f"Unknown scheduler {config.scheduler}, not using any")
    return None


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")

    def __call__(self, current_value: float) -> bool:
        """Check if should stop. Returns True if should stop."""
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        return False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, n_samples = process_batch(model, batch, criterion, device, training=True)

        if loss is not None:
            total_loss += loss.item() * n_samples
            total_samples += n_samples

            loss.backward()
            optimizer.step()

            # OneCycle scheduler steps per batch
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            loss, n_samples = process_batch(model, batch, criterion, device, training=False)

            if loss is not None:
                total_loss += loss.item() * n_samples
                total_samples += n_samples

    return total_loss / total_samples if total_samples > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train the model and return trained model with history.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        criterion: Loss function

    Returns:
        Tuple of (trained_model, training_history)
    """
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    history = {"train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float("inf")
    best_model_state = None

    logger.info(f"Starting training for {config.epochs} epochs")

    for epoch in range(1, config.epochs + 1):
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Log progress
        logger.info(f"Epoch {epoch}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            logger.debug(f"New best model at epoch {epoch}")

        # Step scheduler (except OneCycle which steps per batch)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Check early stopping
        if early_stopping(val_loss):
            history["early_stopped"] = epoch
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with val loss {best_val_loss:.4f}")
        history["best_val_loss"] = best_val_loss

    return model, history
