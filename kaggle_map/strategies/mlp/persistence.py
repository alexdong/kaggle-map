"""Model save/load functionality for MLP strategy.

Handles serialization and deserialization of trained MLP models including
all metadata required for proper reconstruction. Uses structured approaches
for maintaining data integrity across save/load cycles.
"""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from kaggle_map.core.embeddings.embedding_models import EmbeddingModel

if TYPE_CHECKING:
    from kaggle_map.strategies.mlp.strategy import MLPStrategy


class MLPPersistence:
    """Handles saving and loading MLP models with proper type safety."""

    @staticmethod
    def save_model(strategy: "MLPStrategy", filepath: Path) -> None:
        """Save MLP strategy to disk with all metadata.

        Args:
            strategy: Trained MLPStrategy instance to save
            filepath: Path where to save the model
        """
        logger.info(f"Saving MLP model to {filepath}")

        # Prepare save data with proper types
        save_data = {
            "model_state_dict": strategy.model.state_dict(),
            "correct_answers": strategy.correct_answers,
            "question_misconceptions": strategy.question_misconceptions,
            "embedding_model": strategy.embedding_model.model_id,
            "embedding_dim": strategy.embedding_model.dim,  # Store for validation
            "eval_data": getattr(strategy.__class__, "_eval_data", None),  # Save eval data if available
            # Metadata for validation
            "model_metadata": {
                "total_parameters": sum(p.numel() for p in strategy.model.parameters()),
                "device": str(next(strategy.model.parameters()).device),
                "misconception_heads": len(strategy.model.misconception_heads),
            },
        }

        logger.debug(
            "Prepared save data",
            model_parameters=save_data["model_metadata"]["total_parameters"],
            embedding_model=save_data["embedding_model"],
            embedding_dim=save_data["embedding_dim"],
            questions_with_correct_answers=len(save_data["correct_answers"]),
            questions_with_misconceptions=len(save_data["question_misconceptions"]),
            misconception_heads=save_data["model_metadata"]["misconception_heads"],
        )

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save with error handling
        try:
            with filepath.open("wb") as f:
                pickle.dump(save_data, f)

            # Verify file was written
            file_size = filepath.stat().st_size
            logger.info(
                "MLP model saved successfully",
                filepath=str(filepath),
                file_size_mb=f"{file_size / 1024 / 1024:.2f}",
            )
        except Exception as e:
            logger.error(f"Failed to save MLP model: {e}")
            # Clean up partial file if it exists
            if filepath.exists():
                filepath.unlink()
            raise

    @staticmethod
    def load_model(filepath: Path) -> "MLPStrategy":
        """Load MLP strategy from disk with validation.

        Args:
            filepath: Path to saved model file

        Returns:
            Reconstructed MLPStrategy instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If embedding model is unknown or data is corrupted
        """
        logger.info(f"Loading MLP model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        file_size = filepath.stat().st_size
        logger.debug(
            "Model file found",
            filepath=str(filepath),
            file_size_mb=f"{file_size / 1024 / 1024:.2f}",
        )

        # Load save data with validation
        assert filepath.stat().st_size > 0, f"Model file is empty: {filepath}"

        with filepath.open("rb") as f:
            save_data = pickle.load(f)

        assert save_data is not None, f"Failed to load data from model file: {filepath}"
        assert isinstance(save_data, dict), f"Model file contains invalid data format: {filepath}"

        # Validate save data structure
        required_keys = {
            "model_state_dict",
            "correct_answers",
            "question_misconceptions",
            "embedding_model",
        }
        missing_keys = required_keys - set(save_data.keys())
        if missing_keys:
            msg = f"Model file missing required keys: {missing_keys}"
            raise ValueError(msg)

        logger.debug(
            "Save data loaded and validated",
            available_keys=list(save_data.keys()),
            embedding_model=save_data["embedding_model"],
            questions_with_correct_answers=len(save_data["correct_answers"]),
            questions_with_misconceptions=len(save_data["question_misconceptions"]),
        )

        # Find embedding model
        embedding_model = MLPPersistence._find_embedding_model(save_data["embedding_model"])

        # Validate embedding dimension if available
        if "embedding_dim" in save_data:
            expected_dim = save_data["embedding_dim"]
            actual_dim = embedding_model.dim
            if expected_dim != actual_dim:
                logger.warning(
                    "Embedding dimension mismatch",
                    expected=expected_dim,
                    actual=actual_dim,
                    embedding_model=embedding_model.model_id,
                )

        # Reconstruct strategy with the loaded data
        return MLPPersistence._reconstruct_strategy(save_data, embedding_model)

    @staticmethod
    def _reconstruct_strategy(save_data: dict, embedding_model: EmbeddingModel) -> "MLPStrategy":
        """Reconstruct MLPStrategy from save data."""
        # Import here to avoid circular imports
        from kaggle_map.strategies.mlp.model import MLPNet
        from kaggle_map.strategies.mlp.strategy import MLPStrategy

        # Reconstruct model with proper types
        question_misconceptions = save_data["question_misconceptions"]
        model = MLPNet(question_misconceptions, embedding_model)

        # Validate model state before loading
        model_state = save_data["model_state_dict"]
        assert model_state is not None, "Model state dictionary cannot be None"
        assert isinstance(model_state, dict), "Model state must be a dictionary"

        model.load_state_dict(model_state)

        # Restore eval data if available (using setattr to avoid private access warning)
        if "eval_data" in save_data and save_data["eval_data"] is not None:
            MLPStrategy._eval_data = save_data["eval_data"]  # noqa: SLF001
            logger.debug(f"Restored {len(save_data['eval_data'])} eval data rows")

        # Validate model metadata if available
        MLPPersistence._validate_model_metadata(save_data, model)

        logger.info(
            "MLP model loaded successfully",
            embedding_model=embedding_model.model_id,
            embedding_dim=embedding_model.dim,
            model_parameters=sum(p.numel() for p in model.parameters()),
            questions_with_correct_answers=len(save_data["correct_answers"]),
            questions_with_misconceptions=len(question_misconceptions),
        )

        return MLPStrategy(
            model=model,
            correct_answers=save_data["correct_answers"],
            question_misconceptions=question_misconceptions,
            embedding_model=embedding_model,
        )

    @staticmethod
    def _validate_model_metadata(save_data: dict, model: Any) -> None:
        """Validate model metadata against loaded model."""
        if "model_metadata" in save_data:
            metadata = save_data["model_metadata"]
            actual_params = sum(p.numel() for p in model.parameters())
            expected_params = metadata.get("total_parameters", actual_params)

            if actual_params != expected_params:
                logger.warning(
                    "Model parameter count mismatch",
                    expected=expected_params,
                    actual=actual_params,
                )

    @staticmethod
    def _find_embedding_model(model_id: str) -> EmbeddingModel:
        """Find embedding model by ID with validation.

        Args:
            model_id: Embedding model identifier

        Returns:
            EmbeddingModel instance

        Raises:
            ValueError: If model ID is unknown
        """
        logger.debug("Finding embedding model", model_id=model_id)

        # Search through all available embedding models
        for embedding_model in EmbeddingModel.all():
            if embedding_model.model_id == model_id:
                logger.debug(
                    "Embedding model found",
                    model_id=model_id,
                    dimension=embedding_model.dim,
                )
                return embedding_model

        # If not found, raise error with available options
        available_models = [em.model_id for em in EmbeddingModel.all()]
        logger.error(
            "Unknown embedding model",
            requested=model_id,
            available=available_models,
        )
        msg = f"Unknown embedding model: {model_id}. Available models: {available_models}"
        raise ValueError(msg)
