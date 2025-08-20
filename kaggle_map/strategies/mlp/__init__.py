"""MLP strategy package for student misconception prediction.

This package provides a comprehensive neural network-based strategy for
predicting student misconceptions using multi-head MLPs with dynamic
embedding dimensions and structured data handling.

The package is organized into focused modules:
- strategy: Main MLPStrategy class implementing the Strategy interface
- model: MLPNet neural network architecture with dynamic embedding support
- training: Training pipeline with structured data types and batch processing
- prediction: Inference utilities with optimized sigmoid and misconception selection
- dataset: PyTorch dataset with device-agnostic design
- persistence: Model save/load functionality with validation
- evaluation: Display, statistics, and evaluation metrics
- config: Configuration constants and hyperparameters

Example usage:
    >>> from kaggle_map.strategies.mlp import MLPStrategy
    >>> strategy = MLPStrategy.fit(Path("datasets/train.csv"))
    >>> predictions = [strategy.predict(row) for row in test_data]
"""

# Import main strategy class and key components
# Import configuration constants for external use
from kaggle_map.strategies.mlp.config import (
    CORRECTNESS_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DROPOUT_RATE,
    HIDDEN_DIMS,
    LOSS_WEIGHTS,
    MAX_MISCONCEPTIONS_PER_QUESTION,
    MAX_PREDICTIONS,
    MISCONCEPTION_CONFIDENCE_THRESHOLD,
)
from kaggle_map.strategies.mlp.dataset import DatasetItem, MLPDataset
from kaggle_map.strategies.mlp.evaluation import MLPEvaluator
from kaggle_map.strategies.mlp.model import MLPNet
from kaggle_map.strategies.mlp.persistence import MLPPersistence

# Import utility functions for external use
from kaggle_map.strategies.mlp.prediction import (
    MLPPredictor,
    get_best_misconception,
    sigmoid,
)
from kaggle_map.strategies.mlp.strategy import MLPStrategy

# Import structured data types for external use
from kaggle_map.strategies.mlp.training import (
    ProcessedRows,
    TrainingData,
    TrainingSetup,
)

# Define public API
__all__ = [
    # Configuration constants
    "CORRECTNESS_THRESHOLD",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_EPOCHS",
    "DROPOUT_RATE",
    "HIDDEN_DIMS",
    "LOSS_WEIGHTS",
    "MAX_MISCONCEPTIONS_PER_QUESTION",
    "MAX_PREDICTIONS",
    "MISCONCEPTION_CONFIDENCE_THRESHOLD",
    "DatasetItem",
    "MLPDataset",
    "MLPEvaluator",
    "MLPNet",
    "MLPPersistence",
    "MLPPredictor",
    # Main classes
    "MLPStrategy",
    "ProcessedRows",
    # Structured data types
    "TrainingData",
    "TrainingSetup",
    "get_best_misconception",
    # Utility functions
    "sigmoid",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__description__ = "Multi-head MLP strategy for student misconception prediction"
