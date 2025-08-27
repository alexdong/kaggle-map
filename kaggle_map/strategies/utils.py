"""Utilities for strategy implementations."""

import json
import pickle
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel

from kaggle_map.core.models import TrainingRow

# Data split constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug(
            "Device selection: using MPS (Apple Metal)",
            device_type="mps",
            available_backends=["mps", "cuda", "cpu"],
        )
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_count = torch.cuda.device_count()
        logger.debug(
            "Device selection: using CUDA",
            device_type="cuda",
            cuda_devices=cuda_count,
            available_backends=["cuda", "cpu"],
        )
        return device
    device = torch.device("cpu")
    logger.debug(
        "Device selection: fallback to CPU",
        device_type="cpu",
        available_backends=["cpu"],
    )
    return device


class ModelParameters(BaseModel):
    """Parameters for model training and evaluation."""

    train_split: float
    random_seed: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    timestamp: str
    total_samples: int

    @classmethod
    def create(
        cls,
        train_split: float,
        random_seed: int,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        total_samples: int,
    ) -> "ModelParameters":
        """Create ModelParameters with current timestamp."""
        return cls(
            train_split=train_split,
            random_seed=random_seed,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            timestamp=datetime.now(UTC).isoformat(),
            total_samples=total_samples,
        )


def split_training_data(
    training_rows: list[TrainingRow],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = 42,
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Split training data into train/validation/test sets.

    Args:
        training_rows: List of training data rows
        train_ratio: Fraction of data for training (default: 0.70)
        val_ratio: Fraction of data for validation (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_rows, val_rows, test_rows)
    """
    assert training_rows, "Training data cannot be empty"
    assert train_ratio + val_ratio <= 1.0, "train_ratio + val_ratio must be <= 1.0"

    n_samples = len(training_rows)
    indices = np.arange(n_samples)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate split points
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create splits
    train_rows = [training_rows[i] for i in train_indices]
    val_rows = [training_rows[i] for i in val_indices]
    test_rows = [training_rows[i] for i in test_indices]

    logger.debug(f"Split {n_samples} samples into train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    assert train_rows, "Training split cannot be empty"
    assert val_rows, "Validation split cannot be empty"

    return train_rows, val_rows, test_rows


def get_split_indices(
    n_samples: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Get indices for train/validation/test splits.

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction of data for training (default: 0.70)
        val_ratio: Fraction of data for validation (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(n_samples)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate split points
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Split indices
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size : train_size + val_size].tolist()
    test_indices = indices[train_size + val_size :].tolist()

    return train_indices, val_indices, test_indices


# PyTorch utilities for neural network strategies
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from torch import nn
from torch.utils.data import DataLoader

import wandb


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
    batch_callback: Any | None = None,
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

        if loss is not None:
            # Accumulate loss for reporting
            total_loss += loss.item() * n_samples
            total_samples += n_samples

            # Backward pass only if gradient required
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

                # Step scheduler if it's OneCycle (per-batch stepping)
                if scheduler and hasattr(scheduler, "__class__") and scheduler.__class__.__name__ == "OneCycleLR":
                    scheduler.step()

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    batch_callback: Any | None = None,
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


def init_wandb(config: TorchConfig, extra_config: dict[str, Any] | None = None) -> None:
    """Initialize wandb with configuration.

    Args:
        config: Training configuration
        extra_config: Additional configuration to log
    """
    # Auto-generate descriptive run name if not provided
    if not config.wandb_run_name:
        config.wandb_run_name = (
            f"{config.wandb_project.split('-')[-1]}-e{config.epochs}-"
            f"bs{config.batch_size}-"
            f"lr{config.learning_rate:.0e}-"
            f"split{int(config.train_split * 100)}-"
            f"seed{config.random_seed}"
        )

    wandb_config = {
        "train_split": config.train_split,
        "random_seed": config.random_seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "early_stopping_patience": config.early_stopping_patience,
        "optimizer": config.optimizer,
        "scheduler": config.scheduler,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
    }

    if extra_config:
        wandb_config.update(extra_config)

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=wandb_config,
    )


def extract_question_predictions(training_data: list) -> dict[int, list[str]]:
    """Extract unique prediction strings per question.

    Args:
        training_data: List of training data rows

    Returns:
        Dictionary mapping question ID to list of unique prediction strings
    """
    from collections import defaultdict

    question_predictions = defaultdict(list)
    for row in training_data:
        pred_str = str(row.prediction)
        question_predictions[row.question_id].append(pred_str)

    # Return unique predictions per question
    return {qid: list(set(preds)) for qid, preds in question_predictions.items()}


def load_embeddings(
    embeddings_path: Path | None,
    training_data: list,
    train_df: Any | None = None,
    compute_fn: Callable[[list], tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load or compute embeddings for training data.

    Args:
        embeddings_path: Path to precomputed embeddings
        training_data: List of training rows
        train_df: DataFrame with training data (for precomputed embeddings)
        compute_fn: Function to compute embeddings if not precomputed

    Returns:
        Tuple of (embeddings, question_ids, extra_data)
    """
    if embeddings_path and embeddings_path.exists():
        logger.info(f"Loading precomputed embeddings from {embeddings_path}")
        data = np.load(embeddings_path)
        embeddings = data["embeddings"]
        row_ids = data["row_ids"]

        if train_df is not None:
            # Get question IDs from DataFrame
            question_ids = train_df.set_index("row_id").loc[row_ids]["QuestionId"].to_numpy()
            mc_answers = train_df.set_index("row_id").loc[row_ids]["MC_Answer"].to_numpy()

            # Create prediction mapping from training data
            row_id_to_prediction = {row.row_id: str(row.prediction) for row in training_data}
            predictions = np.array([row_id_to_prediction[rid] for rid in row_ids])

            extra_data = {
                "mc_answers": mc_answers,
                "predictions": predictions,
                "row_ids": row_ids,
            }
        else:
            question_ids = data.get("question_ids", np.array([]))
            extra_data = {}
    else:
        logger.info("Computing embeddings from scratch")
        if compute_fn is None:
            msg = "No compute function provided for embeddings"
            raise ValueError(msg)

        embeddings, question_ids, extra_data = compute_fn(training_data)
        embeddings = np.array(embeddings)
        question_ids = np.array(question_ids)

    return embeddings, question_ids, extra_data


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TorchConfig,
    device: torch.device,
    criterion: nn.Module | None = None,
    train_batch_fn: Callable | None = None,
    val_batch_fn: Callable | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Generic training loop for PyTorch models.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        criterion: Loss function (default: CrossEntropyLoss)
        train_batch_fn: Custom training batch processing function
        val_batch_fn: Custom validation batch processing function

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Default criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Setup optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    # Initialize checkpoint manager and early stopping
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir, model_name=config.wandb_run_name or "model"
    )
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
    }

    # Training loop
    try:
        for epoch in range(config.epochs):
            # Training
            if train_batch_fn:
                avg_train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device, scheduler, batch_callback=train_batch_fn
                )
            else:
                avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)

            # Validation
            if val_batch_fn:
                avg_val_loss = validate_epoch(model, val_loader, criterion, device, batch_callback=val_batch_fn)
            else:
                avg_val_loss = validate_epoch(model, val_loader, criterion, device)

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Update history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["epochs"].append(epoch + 1)

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            # Step scheduler if not OneCycle (OneCycle steps per batch)
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Save best model and check early stopping
            metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
            if (
                checkpoint_manager.save_best_model(model, optimizer, epoch + 1, avg_val_loss, metrics, config)
                and wandb.run is not None
            ):
                wandb.log({"best_val_loss": avg_val_loss})

            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                if wandb.run is not None:
                    wandb.log({"early_stopped_epoch": epoch + 1})
                history["early_stopped"] = epoch + 1
                break

            # Stop after first epoch if requested (for testing)
            if config.epochs == 1:
                logger.info("Stopping after first epoch as requested")
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user!")
        if wandb.run is not None:
            wandb.log({"interrupted": True})
        history["interrupted"] = True

    # Load best checkpoint
    best_checkpoint = checkpoint_manager.load_best_model()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(
            f"Loaded best model from epoch {best_checkpoint['epoch']} with val loss {best_checkpoint['val_loss']:.4f}"
        )
        history["best_epoch"] = best_checkpoint["epoch"]
        history["best_val_loss"] = best_checkpoint["val_loss"]

    return model, history


def save_torch_strategy(
    strategy: Any,
    filepath: Path,
    *,
    save_params: bool = True,
) -> None:
    """Save a PyTorch-based strategy to disk.

    Args:
        strategy: The strategy object to save
        filepath: Path to save the model
        save_params: Whether to save parameters separately
    """
    logger.info(f"Saving model to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save entire strategy object with pickle
    with filepath.open("wb") as f:
        pickle.dump(strategy, f)

    # Save parameters if available
    if save_params and hasattr(strategy, "parameters") and strategy.parameters:
        params_path = filepath.with_suffix(".params.json")
        logger.info(f"Saving model parameters to {params_path}")
        with params_path.open("w") as f:
            json.dump(strategy.parameters.model_dump(), f, indent=2)


def load_torch_strategy(
    cls: type,
    filepath: Path,
    *,
    load_params: bool = True,
) -> Any:
    """Load a PyTorch-based strategy from disk.

    Args:
        cls: The strategy class
        filepath: Path to the saved model
        load_params: Whether to load parameters

    Returns:
        Loaded strategy instance
    """
    logger.info(f"Loading model from {filepath}")
    assert filepath.exists(), f"Model file not found: {filepath}"

    with filepath.open("rb") as f:
        loaded_model = pickle.load(f)

    # Try to load parameters if they exist
    if load_params:
        params_path = filepath.with_suffix(".params.json")
        if params_path.exists():
            logger.info(f"Loading model parameters from {params_path}")
            with params_path.open("r") as f:
                params_data = json.load(f)
                parameters = ModelParameters.model_validate(params_data)

            # Update loaded model's parameters if different
            if hasattr(loaded_model, "parameters") and parameters != loaded_model.parameters:
                # Create new instance with updated parameters
                return cls(
                    model=loaded_model.model,
                    correct_answers=loaded_model.correct_answers,
                    tokenizer=loaded_model.tokenizer,
                    device=loaded_model.device,
                    parameters=parameters,
                )
        else:
            logger.warning(f"No parameters file found at {params_path}")

    return loaded_model
