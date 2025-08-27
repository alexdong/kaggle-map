"""Shared utilities for PyTorch-based strategies.

This module provides reusable components for neural network strategies:
- Base configuration classes
- Training loops with early stopping
- Checkpoint management
- Wandb integration
- Device management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from loguru import logger
from torch import nn

import wandb

from .utils import TRAIN_RATIO

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass
class TorchConfig:
    """Base configuration for PyTorch-based strategies."""

    # Data configuration
    train_split: float = TRAIN_RATIO
    random_seed: int = 42
    train_csv_path: Path = field(default_factory=lambda: Path("datasets/train.csv"))
    embeddings_path: Path | None = field(default_factory=lambda: Path("datasets/train_embeddings.npz"))

    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 5

    # Architecture hyperparameters
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"  # relu, gelu, leaky_relu

    # Optimizer and scheduler
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "onecycle"  # none, cosine, step, onecycle

    # Infrastructure
    device: str = "auto"  # auto, cpu, cuda, mps
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Wandb configuration
    wandb_project: str = "kaggle-map"
    wandb_run_name: str | None = None


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch with metrics."""
        ...

    def on_training_end(self) -> None:
        """Called when training completes."""
        ...


class WandbCallback:
    """Wandb logging callback for training."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize wandb run."""
        self.run = wandb.init(
            project=config.get("wandb_project", "kaggle-map"),
            name=config.get("wandb_run_name"),
            config=config,
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Log epoch start."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics to wandb."""
        wandb.log({"epoch": epoch, **metrics})

    def on_training_end(self) -> None:
        """Finish wandb run."""
        wandb.finish()


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(self, checkpoint_dir: Path, model_name: str = "model") -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.best_metric = float("inf")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        config: TorchConfig | None = None,
    ) -> Path:
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metrics,
        }

        if config is not None:
            checkpoint["config"] = config

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def save_best_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        metrics: dict[str, float],
        config: TorchConfig | None = None,
        metric_name: str = "val_loss",
    ) -> bool:
        """Save model if it's the best so far. Returns True if saved."""
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                f"best_{metric_name}": metric_value,
                **metrics,
            }

            if config is not None:
                checkpoint["config"] = config

            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path} ({metric_name}={metric_value:.4f})")
            return True
        return False

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load a training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # weights_only=False needed for loading custom classes
        return torch.load(checkpoint_path, weights_only=False)

    def load_best_model(self) -> dict[str, Any] | None:
        """Load the best saved model."""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if best_path.exists():
            return self.load_checkpoint(best_path)
        return None

    def find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_checkpoint_*.pt"))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None


class EarlyStopping:
    """Early stopping handler for training."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float("inf")
        self.should_stop = False


def create_optimizer(
    model: nn.Module,
    config: TorchConfig,
) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    logger.warning(f"Unknown optimizer {config.optimizer}, using Adam")
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TorchConfig,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create learning rate scheduler based on configuration."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )
    if config.scheduler == "onecycle":
        if steps_per_epoch is None:
            logger.warning("OneCycle scheduler requires steps_per_epoch, skipping")
            return None
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 10,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
        )
    if config.scheduler == "none":
        return None
    logger.warning(f"Unknown scheduler {config.scheduler}, not using any")
    return None


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
    }
    return activation_map.get(activation, nn.ReLU())


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batch_callback: callable | None = None,
) -> float:
    """Generic training epoch for PyTorch models.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to run on
        scheduler: Optional learning rate scheduler
        batch_callback: Optional callback for custom batch processing

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Allow custom batch processing via callback
        if batch_callback:
            loss, n_samples = batch_callback(model, batch, criterion, device)
        else:
            # Default processing for standard supervised learning
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            n_samples = inputs.size(0)

        if loss is not None and loss.requires_grad:
            loss.backward()
            optimizer.step()

            # Step scheduler if it's OneCycle (per-batch stepping)
            if scheduler and hasattr(scheduler, "__class__") and scheduler.__class__.__name__ == "OneCycleLR":
                scheduler.step()

            total_loss += loss.item() * n_samples
            total_samples += n_samples

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    batch_callback: callable | None = None,
) -> float:
    """Generic validation epoch for PyTorch models.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        batch_callback: Optional callback for custom batch processing

    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # Allow custom batch processing via callback
            if batch_callback:
                loss, n_samples = batch_callback(model, batch, criterion, device)
            else:
                # Default processing for standard supervised learning
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                n_samples = inputs.size(0)

            if loss is not None:
                total_loss += loss.item() * n_samples
                total_samples += n_samples

    return total_loss / total_samples if total_samples > 0 else 0.0


def log_model_info(model: nn.Module) -> dict[str, int]:
    """Log model parameter information.

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters")

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
