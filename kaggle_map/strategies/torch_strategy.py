"""Base class for PyTorch-based strategies with integrated training and search."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kaggle_map.core.dataset import extract_correct_answers, parse_training_data

from .base import Strategy
from .torch_utils import (
    CheckpointManager,
    EarlyStopping,
    TorchConfig,
    WandbCallback,
    create_optimizer,
    create_scheduler,
    get_device,
    log_model_info,
    train_epoch,
    validate_epoch,
)
from .utils import ModelParameters, get_split_indices

if TYPE_CHECKING:
    from pathlib import Path

    from optuna.trial import Trial

    from kaggle_map.core.models import Answer, QuestionId


@dataclass(frozen=True)
class TorchStrategy(Strategy):
    """Base class for PyTorch-based prediction strategies.

    Provides common functionality for neural network models including:
    - Training loops with early stopping
    - Checkpoint management
    - Wandb integration
    - Hyperparameter search support
    """

    model: nn.Module
    correct_answers: dict[QuestionId, Answer]
    device: torch.device
    parameters: ModelParameters | None = None

    @abstractmethod
    def create_model(self, config: TorchConfig, **kwargs) -> nn.Module:
        """Create the neural network model.

        Args:
            config: Configuration for the model
            **kwargs: Additional strategy-specific parameters

        Returns:
            The initialized neural network model
        """
        ...

    @abstractmethod
    def create_dataset(
        self,
        embeddings: np.ndarray,
        question_ids: np.ndarray,
        predictions: np.ndarray,
        correct_answers: dict[QuestionId, Answer],
        **kwargs,
    ) -> Dataset:
        """Create a PyTorch dataset for training.

        Args:
            embeddings: Feature embeddings
            question_ids: Question IDs for each sample
            predictions: Prediction labels for each sample
            correct_answers: Mapping of question IDs to correct answers
            **kwargs: Additional strategy-specific data

        Returns:
            PyTorch Dataset instance
        """
        ...

    @abstractmethod
    def process_batch(
        self,
        model: nn.Module,
        batch: Any,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int]:
        """Process a single batch during training or validation.

        This method allows strategies to customize how batches are processed,
        which is especially useful for complex architectures.

        Args:
            model: The neural network model
            batch: Batch data from DataLoader
            criterion: Loss function
            device: Device to run computations on

        Returns:
            Tuple of (loss, number_of_samples)
            Loss can be None if batch should be skipped
        """
        ...

    @classmethod
    def get_hyperparameter_search_space(cls, trial: Trial) -> dict[str, Any]:
        """Define the hyperparameter search space for Optuna.

        Override this method to define strategy-specific hyperparameters.

        Args:
            trial: Optuna trial object for suggesting parameters

        Returns:
            Dictionary of hyperparameter names to values
        """
        # Default search space - override in subclasses
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    @classmethod
    def create_config_from_hyperparams(
        cls,
        hyperparams: dict[str, Any],
        base_config: TorchConfig | None = None,
    ) -> TorchConfig:
        """Create a configuration object from hyperparameters.

        Args:
            hyperparams: Dictionary of hyperparameters from search
            base_config: Optional base configuration to update

        Returns:
            Configuration object for training
        """
        if base_config is None:
            base_config = TorchConfig()

        # Update config with hyperparameters
        for key, value in hyperparams.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config

    @classmethod
    def load_and_prepare_data(
        cls,
        config: TorchConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[QuestionId, Answer]]:
        """Load training data and prepare for model training.

        Args:
            config: Training configuration

        Returns:
            Tuple of (embeddings, question_ids, predictions, correct_answers)
        """
        logger.info(f"Loading training data from {config.train_csv_path}")

        # Load training data
        training_data = parse_training_data(config.train_csv_path)
        correct_answers = extract_correct_answers(training_data)

        # Load or compute embeddings
        if config.embeddings_path and config.embeddings_path.exists():
            logger.info(f"Loading precomputed embeddings from {config.embeddings_path}")
            data = np.load(config.embeddings_path)
            embeddings = data["embeddings"]
            row_ids = data["row_ids"]

            # Extract corresponding metadata
            import pandas as pd
            train_df = pd.read_csv(config.train_csv_path)
            question_ids = train_df.set_index("row_id").loc[row_ids]["QuestionId"].to_numpy()

            # Create prediction strings
            row_id_to_prediction = {row.row_id: str(row.prediction) for row in training_data}
            predictions = np.array([row_id_to_prediction[rid] for rid in row_ids])
        else:
            logger.warning("Embeddings not found, computing from scratch")
            from kaggle_map.core.embeddings.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()

            embeddings = []
            question_ids = []
            predictions = []

            for row in training_data:
                text = repr(row)
                embedding = tokenizer.encode(text)
                embeddings.append(embedding)
                question_ids.append(row.question_id)
                predictions.append(str(row.prediction))

            embeddings = np.array(embeddings)
            question_ids = np.array(question_ids)
            predictions = np.array(predictions)

        return embeddings, question_ids, predictions, correct_answers

    @classmethod
    def fit(
        cls,
        *,
        config: TorchConfig | None = None,
        resume_from_checkpoint: Path | None = None,
        **kwargs,
    ) -> TorchStrategy:
        """Train the PyTorch model.

        Args:
            config: Training configuration
            resume_from_checkpoint: Optional checkpoint to resume from
            **kwargs: Additional training parameters

        Returns:
            Trained strategy instance
        """
        if config is None:
            config = TorchConfig(**kwargs)

        # Set random seed
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Get device
        device = get_device() if config.device == "auto" else torch.device(config.device)
        logger.info(f"Using device: {device}")

        # Initialize wandb if configured
        callbacks = []
        if config.wandb_project:
            if not config.wandb_run_name:
                config.wandb_run_name = f"{cls.__name__}-e{config.epochs}-bs{config.batch_size}-lr{config.learning_rate:.0e}"

            wandb_callback = WandbCallback({
                "strategy": cls.__name__,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "device": str(device),
                **kwargs,
            })
            callbacks.append(wandb_callback)

        # Load and prepare data
        embeddings, question_ids, predictions, correct_answers = cls.load_and_prepare_data(config)

        # Split data
        n_samples = len(embeddings)
        train_indices, val_indices, test_indices = get_split_indices(
            n_samples,
            train_ratio=config.train_split,
            random_seed=config.random_seed,
        )

        logger.info(f"Data split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # Create model
        model_instance = cls(
            model=None,  # Will be set later
            correct_answers=correct_answers,
            device=device,
            parameters=None,
        )

        model = model_instance.create_model(
            config,
            question_ids=question_ids,
            predictions=predictions,
        )
        model = model.to(device)

        # Log model info
        model_info = log_model_info(model)
        for callback in callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(0, {"model_info": model_info})

        # Create datasets
        train_dataset = model_instance.create_dataset(
            embeddings[train_indices],
            question_ids[train_indices],
            predictions[train_indices],
            correct_answers,
        )

        val_dataset = model_instance.create_dataset(
            embeddings[val_indices],
            question_ids[val_indices],
            predictions[val_indices],
            correct_answers,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Setup training
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
        criterion = nn.CrossEntropyLoss()

        # Setup checkpoint manager and early stopping
        checkpoint_manager = CheckpointManager(config.checkpoint_dir, model_name=cls.__name__.lower())
        early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            checkpoint = checkpoint_manager.load_checkpoint(resume_from_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            logger.info(f"Resumed from epoch {start_epoch}")

        # Training loop
        float("inf")

        try:
            for epoch in range(start_epoch, config.epochs):
                # Train
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    scheduler,
                    lambda m, b, c, d: model_instance.process_batch(m, b, c, d),
                )

                # Validate
                val_loss = validate_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    lambda m, b, c, d: model_instance.process_batch(m, b, c, d),
                )

                logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Log metrics
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                for callback in callbacks:
                    if hasattr(callback, "on_epoch_end"):
                        callback.on_epoch_end(epoch + 1, metrics)

                # Step scheduler (except OneCycle which steps per batch)
                if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                # Save best model and check early stopping
                if checkpoint_manager.save_best_model(
                    model,
                    optimizer,
                    epoch + 1,
                    val_loss,
                    metrics,
                    config,
                    "val_loss",
                ):
                    pass

                if early_stopping(val_loss):
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        # Load best model
        best_checkpoint = checkpoint_manager.load_best_model()
        if best_checkpoint:
            model.load_state_dict(best_checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']}")

        # Cleanup callbacks
        for callback in callbacks:
            if hasattr(callback, "on_training_end"):
                callback.on_training_end()

        # Create final model parameters
        parameters = ModelParameters.create(
            train_split=config.train_split,
            random_seed=config.random_seed,
            train_indices=list(train_indices),
            val_indices=list(val_indices),
            test_indices=list(test_indices),
            total_samples=n_samples,
        )

        # Return trained strategy
        return cls(
            model=model,
            correct_answers=correct_answers,
            device=device,
            parameters=parameters,
        )
