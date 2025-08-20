"""Configuration constants for MLP strategy.

Constants and hyperparameters used throughout the MLP strategy implementation.
Embedding dimensions are read dynamically from EmbeddingModel instead of hardcoded.
"""

# Prediction thresholds
CORRECTNESS_THRESHOLD: float = 0.5
MISCONCEPTION_CONFIDENCE_THRESHOLD: float = 0.3  # Minimum confidence for misconception
MAX_PREDICTIONS: int = 3  # Maximum number of predictions to return per observation

# Training hyperparameters
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_EPOCHS: int = 100
DEFAULT_LEARNING_RATE: float = 5e-4

# Model architecture
HIDDEN_DIMS: tuple[int, ...] = (512, 256, 128)
DROPOUT_RATE: float = 0.2
MAX_MISCONCEPTIONS_PER_QUESTION: int = 8

# Loss function weights for multi-head training
LOSS_WEIGHTS: dict[str, float] = {
    "correctness": 2.0,  # Correctness is most important
    "categories": 1.5,  # Categories are important
    "misconceptions": 1.0,  # Misconceptions are helpful
}
