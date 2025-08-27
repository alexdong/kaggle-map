"""Main MLP strategy implementation and orchestration.

Coordinates all MLP components using absolute imports and clean interfaces.
Serves as the main entry point for the MLP strategy, implementing the
Strategy interface while delegating to specialized modules.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from platformdirs import PlatformDirs
from sklearn.model_selection import train_test_split

from kaggle_map.core.dataset import parse_training_data
from kaggle_map.core.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.core.models import (
    Answer,
    EvaluationRow,
    Misconception,
    QuestionId,
    SubmissionRow,
)
from kaggle_map.strategies.base import Strategy
from kaggle_map.strategies.mlp.config import MAX_PREDICTIONS
from kaggle_map.strategies.mlp.model import MLPNet
from kaggle_map.strategies.mlp.prediction import MLPPredictor
from kaggle_map.strategies.mlp.training import (
    extract_correct_answers,
    extract_question_misconceptions,
    prepare_training_data,
    set_random_seeds,
    train_model,
)
from kaggle_map.strategies.utils import get_device


@dataclass(frozen=True)
class MLPStrategy(Strategy):
    """MLP strategy for predicting student misconceptions.

    Uses a neural network with question-specific heads to predict misconceptions,
    then reconstructs categories deterministically based on answer correctness.
    Coordinates specialized modules for training, prediction, and evaluation.
    """

    model: MLPNet
    correct_answers: dict[QuestionId, Answer]
    question_misconceptions: dict[QuestionId, list[Misconception]]
    embedding_model: EmbeddingModel

    @property
    def name(self) -> str:
        """Strategy name."""
        return "mlp"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Question-specific MLP for misconception detection and reasoning quality"

    @classmethod
    def fit(
        cls,
        train_csv_path: Path = Path("datasets/train.csv"),
        train_split: float = 1.0,
        random_seed: int = 42,
        embeddings_path: Path | None = None,
    ) -> "MLPStrategy":
        """Train the MLP strategy on training data.

        Args:
            train_csv_path: Path to training CSV file
            train_split: Fraction of data to use for training (default: 1.0 = all data)
            random_seed: Random seed for reproducible results
            embeddings_path: Path to pre-computed embeddings .npz file (optional)

        Returns:
            Trained MLPStrategy instance
        """
        fit_start_time = time.time()

        # Set up logging for the entire fit process
        log_dir = PlatformDirs().user_log_dir
        logger.bind(
            operation="mlp_fit",
            train_path=str(train_csv_path),
            train_split=train_split,
            random_seed=random_seed,
            embeddings_path=str(embeddings_path) if embeddings_path else None,
            log_dir=str(log_dir),
        ).info("Starting MLP strategy fit")

        # Set random seeds for deterministic training
        logger.debug("Setting random seeds for deterministic training", seed=random_seed)
        set_random_seeds(random_seed)

        # Parse training data
        logger.debug("Starting training data parsing", file_path=str(train_csv_path))
        parse_start = time.time()
        all_training_data = parse_training_data(train_csv_path)
        parse_duration = time.time() - parse_start
        logger.debug(
            "Training data parsing completed",
            total_rows=len(all_training_data),
            parse_time_seconds=f"{parse_duration:.3f}",
        )

        # Split data if train_split < 1.0
        if train_split < 1.0:
            logger.debug(
                "Splitting data for train/eval",
                train_split=train_split,
                total_samples=len(all_training_data),
                stratify_by="question_id",
            )
            split_start = time.time()
            training_data, eval_data = train_test_split(
                all_training_data,
                train_size=train_split,
                random_state=random_seed,
                stratify=[row.question_id for row in all_training_data],
            )
            split_duration = time.time() - split_start
            logger.debug(
                "Data split completed",
                training_samples=len(training_data),
                eval_samples=len(eval_data),
                split_time_seconds=f"{split_duration:.3f}",
            )
            # Store eval data for later use
            cls._eval_data = eval_data
        else:
            logger.debug("Using all data for training", train_split=1.0)
            training_data = all_training_data
            cls._eval_data = None

        # Extract metadata for architecture
        logger.debug("Starting metadata extraction")
        metadata_start = time.time()

        correct_answers = extract_correct_answers(training_data)
        question_misconceptions = extract_question_misconceptions(training_data)

        metadata_duration = time.time() - metadata_start
        logger.debug(
            "Metadata extraction completed",
            questions_with_correct_answers=len(correct_answers),
            questions_with_misconceptions=len(question_misconceptions),
            extraction_time_seconds=f"{metadata_duration:.3f}",
        )

        # Generate embeddings and labels for training
        embedding_model = EmbeddingModel.MINI_LM
        embeddings_start = time.time()

        if embeddings_path is not None and embeddings_path.exists():
            logger.info(
                "Loading pre-computed embeddings",
                embeddings_path=str(embeddings_path),
                file_exists=True,
                embedding_model=embedding_model.model_id,
            )
            # For now, fall back to generating embeddings from scratch
            logger.warning("Pre-computed embeddings not yet supported, generating from scratch")
            training_data_prepared = prepare_training_data(
                training_data,
                correct_answers,
                question_misconceptions,
                embedding_model,
            )
        else:
            logger.info(
                "Generating embeddings and labels for training",
                embedding_model=embedding_model.model_id,
                training_samples=len(training_data),
            )
            training_data_prepared = prepare_training_data(
                training_data,
                correct_answers,
                question_misconceptions,
                embedding_model,
            )

        embeddings_duration = time.time() - embeddings_start
        logger.debug(
            "Embeddings preparation completed",
            embeddings_shape=training_data_prepared.embeddings.shape,
            correctness_shape=training_data_prepared.correctness.shape,
            question_ids_shape=training_data_prepared.question_ids.shape,
            preparation_time_seconds=f"{embeddings_duration:.3f}",
        )

        # Get device and train model
        device = get_device()
        training_start = time.time()
        logger.info(
            "Starting MLP training",
            device=str(device),
            device_type=device.type,
            embeddings_shape=training_data_prepared.embeddings.shape,
            training_samples=len(training_data),
        )

        model = train_model(
            training_data_prepared,
            question_misconceptions,
            embedding_model,
            device,
        )

        training_duration = time.time() - training_start
        total_duration = time.time() - fit_start_time

        logger.info(
            "MLP strategy fit completed",
            training_time_seconds=f"{training_duration:.3f}",
            total_fit_time_seconds=f"{total_duration:.3f}",
            model_parameters=sum(p.numel() for p in model.parameters()),
            device=str(device),
        )

        return cls(
            model=model,
            correct_answers=correct_answers,
            question_misconceptions=question_misconceptions,
            embedding_model=embedding_model,
        )

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make predictions for a single evaluation row.

        Args:
            evaluation_row: Single evaluation row to predict on

        Returns:
            Submission row with prediction
        """
        logger.debug(f"Making MLP prediction for row {evaluation_row.row_id}")

        # Setup model and device
        device = get_device()
        self.model.to(device)
        self.model.eval()

        # Create predictor
        predictor = MLPPredictor(
            self.correct_answers,
            self.question_misconceptions,
        )

        # Generate embedding
        tokenizer = get_tokenizer(self.embedding_model)
        text = repr(evaluation_row)
        embedding = tokenizer.encode(text)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            if evaluation_row.question_id not in self.question_misconceptions:
                logger.warning(f"Unknown question {evaluation_row.question_id} - using default prediction")
                predicted_categories = predictor.create_default_prediction(evaluation_row)
            else:
                # Model inference
                outputs = self.model(embedding_tensor, evaluation_row.question_id)
                predicted_categories = predictor.get_predictions_from_outputs(
                    outputs,
                    question_id=evaluation_row.question_id,
                    row=evaluation_row,
                )

        # Return final prediction (max 3 categories)
        final_predictions = predicted_categories[:MAX_PREDICTIONS]
        return SubmissionRow(
            row_id=evaluation_row.row_id,
            predicted_categories=final_predictions,
        )
