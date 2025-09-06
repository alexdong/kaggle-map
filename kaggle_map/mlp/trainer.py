"""Training utilities for MLP model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from kaggle_map.mlp.model import ActivationType, ArchitectureSize, EvaluationResult, QuestionSpecificMLP


@dataclass(frozen=True)
class BatchResult:
    """Result from processing a single batch."""

    loss: torch.Tensor | None
    sample_count: int

    @property
    def is_valid(self) -> bool:
        return self.loss is not None and self.sample_count > 0

    @property
    def avg_loss(self) -> torch.Tensor | None:
        """Get average loss per sample."""
        if self.loss is None or self.sample_count == 0:
            return None
        return self.loss


@dataclass
class TrainingContext:
    """Context for training operations."""

    model: QuestionSpecificMLP
    criterion: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


@dataclass
class TrainingConfig:
    """Configuration for MLP training."""

    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 15

    train_split: float = 0.7
    random_seed: int = 42

    train_csv_path: Path = Path("datasets/train.csv")
    checkpoint_dir: Path = Path("checkpoints")

    architecture_size: ArchitectureSize = "xlarge"
    dropout: float = 0.3
    activation: ActivationType = "gelu"


@dataclass
class TrainingResult:
    """Result from training a model."""

    model: QuestionSpecificMLP
    history: dict[str, Any]


@dataclass
class TrainingSetup:
    """Setup for training a model."""

    model: QuestionSpecificMLP
    train_loader: DataLoader
    val_loader: DataLoader
    config: TrainingConfig
    device: torch.device
    criterion: nn.Module


def process_batch(  # noqa: C901
    model: QuestionSpecificMLP,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    *,
    training: bool = True,
) -> BatchResult:
    """Process a single batch through the model.

    Args:
        model: The MLP model
        batch: Tuple of (embeddings, question_ids, labels, is_correct)
        criterion: Loss function
        device: Device to run on
        training: Whether in training mode (affects gradient tracking)

    Returns:
        BatchResult containing loss and sample count
    """
    embeddings, question_ids, labels, is_correct = batch
    embeddings = embeddings.to(device)
    question_ids = question_ids.to(device)
    labels = labels.to(device)
    is_correct = is_correct.to(device)

    outputs: dict[EvaluationResult, torch.Tensor] = model(embeddings, question_ids, is_correct)

    total_loss = 0.0
    total_samples = 0

    for eval_result, logits in outputs.items():
        correctness_mask = is_correct > 0 if eval_result.is_correct else is_correct == 0
        question_mask = question_ids == eval_result.question_id
        combined_mask = question_mask & correctness_mask

        if combined_mask.any():
            question_labels = labels[combined_mask]
            if logits.size(0) == question_labels.size(0):
                loss = criterion(logits, question_labels)
                total_loss += loss * logits.size(0)
                total_samples += logits.size(0)

    if total_samples == 0:
        return BatchResult(loss=None, sample_count=0)

    avg_loss = total_loss / total_samples
    # Detach for validation to save memory
    if not training and isinstance(avg_loss, torch.Tensor):
        avg_loss = avg_loss.detach()
    # Ensure avg_loss is a torch.Tensor
    if not isinstance(avg_loss, torch.Tensor):
        avg_loss = torch.tensor(avg_loss)
    return BatchResult(loss=avg_loss, sample_count=int(embeddings.size(0)))


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
    ctx: TrainingContext,
    train_loader: DataLoader,
) -> float:
    """Train for one epoch.

    Args:
        ctx: Training context with model, optimizer, criterion, device
        train_loader: DataLoader for training data

    Returns:
        Average training loss for the epoch
    """
    assert ctx.optimizer is not None, "Optimizer required for training"
    optimizer = ctx.optimizer

    ctx.model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        result = process_batch(ctx.model, batch, ctx.criterion, ctx.device, training=True)

        if result.is_valid:
            assert result.loss is not None  # Type guard for type checker
            total_loss += result.loss.item() * result.sample_count
            total_samples += result.sample_count

            result.loss.backward()
            optimizer.step()

            if ctx.scheduler and isinstance(ctx.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                ctx.scheduler.step()

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    ctx: TrainingContext,
    val_loader: DataLoader,
) -> float:
    """Validate for one epoch.

    Args:
        ctx: Training context with model, criterion, device
        val_loader: DataLoader for validation data

    Returns:
        Average validation loss for the epoch
    """
    ctx.model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            result = process_batch(ctx.model, batch, ctx.criterion, ctx.device, training=False)

            if result.is_valid:
                assert result.loss is not None  # Type guard for type checker
                total_loss += result.loss.item() * result.sample_count
                total_samples += result.sample_count

    return total_loss / total_samples if total_samples > 0 else 0.0


def _run_training_loop(
    ctx: TrainingContext,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run the main training loop."""
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(ctx, train_loader)
        val_loss = validate_epoch(ctx, val_loader)

        logger.info(f"Epoch {epoch}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = ctx.model.state_dict().copy()
            logger.debug(f"New best model at epoch {epoch}")

        assert ctx.scheduler is not None, "Scheduler should be set"
        ctx.scheduler.step()

        if early_stopping(val_loss):
            history["early_stopped"] = epoch
            break

    if best_model_state is not None:
        ctx.model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with val loss {best_val_loss:.4f}")
        history["best_val_loss"] = best_val_loss

    return history


def train_model(setup: TrainingSetup) -> TrainingResult:
    """Train the model and return trained model with history.

    Args:
        setup: Training setup with all required components

    Returns:
        TrainingResult with trained model and history
    """
    optimizer = create_optimizer(setup.model, setup.config)
    scheduler = create_scheduler(optimizer, setup.config, len(setup.train_loader))

    ctx = TrainingContext(
        model=setup.model,
        criterion=setup.criterion,
        device=setup.device,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    logger.info(f"Starting training for {setup.config.epochs} epochs")
    history = _run_training_loop(ctx, setup.train_loader, setup.val_loader, setup.config)

    return TrainingResult(model=setup.model, history=history)
