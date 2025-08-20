"""Main MLP strategy implementation and orchestration.

Coordinates all MLP components using absolute imports and clean interfaces.
Serves as the main entry point for the MLP strategy, implementing the
Strategy interface while delegating to specialized modules.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from platformdirs import PlatformDirs
from sklearn.model_selection import train_test_split

from kaggle_map.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.models import (
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
    parse_training_data,
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
        logger.debug(
            "Setting random seeds for deterministic training", seed=random_seed
        )
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
            logger.warning(
                "Pre-computed embeddings not yet supported, generating from scratch"
            )
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

    def predict(self, test_data: list[EvaluationRow]) -> list[SubmissionRow]:
        """Make predictions for test data.

        Args:
            test_data: List of test rows

        Returns:
            List of submission rows with predictions
        """
        prediction_start_time = time.time()

        # Input validation with logging
        assert test_data is not None, "Test data cannot be None"
        logger.bind(operation="mlp_predict").info(
            "Starting MLP prediction process",
            test_samples=len(test_data),
            max_predictions_per_sample=MAX_PREDICTIONS,
        )

        if not test_data:
            logger.debug("Empty test data provided, returning empty predictions")
            return []

        # Device setup and model preparation
        device_start = time.time()
        device = get_device()
        logger.debug(
            "Device selection completed",
            selected_device=str(device),
            device_type=device.type,
            device_selection_time_ms=f"{(time.time() - device_start) * 1000:.2f}",
        )

        model_setup_start = time.time()
        self.model.to(device)
        self.model.eval()

        # Verify model device consistency
        model_device = next(self.model.parameters()).device
        assert model_device == device, (
            f"Model device mismatch: expected {device}, got {model_device}"
        )
        model_setup_time = time.time() - model_setup_start

        logger.debug(
            "Model setup completed",
            model_device=str(model_device),
            device_matches=True,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            model_setup_time_ms=f"{model_setup_time * 1000:.2f}",
        )

        # Create predictor
        predictor = MLPPredictor(
            self.correct_answers,
            self.question_misconceptions,
        )

        # Embedding generation phase
        embedding_start_time = time.time()
        logger.debug(
            "Starting embedding generation",
            embedding_model=self.embedding_model.model_id,
            test_samples=len(test_data),
        )

        tokenizer_start = time.time()
        tokenizer = get_tokenizer(self.embedding_model)
        tokenizer_load_time = time.time() - tokenizer_start

        logger.debug(
            "Tokenizer loaded",
            tokenizer_type=type(tokenizer).__name__,
            load_time_ms=f"{tokenizer_load_time * 1000:.2f}",
        )

        test_embeddings = []
        text_lengths = []
        embedding_times = []

        for i, row in enumerate(test_data):
            # Generate embedding with timing
            single_embedding_start = time.time()
            text = repr(row)  # Uses EvaluationRow.__repr__ format
            text_lengths.append(len(text))

            embedding = tokenizer.encode(text)
            test_embeddings.append(embedding)

            single_embedding_time = time.time() - single_embedding_start
            embedding_times.append(single_embedding_time)

            # Log progress for large datasets
            if i > 0 and i % 100 == 0:
                avg_embedding_time = sum(embedding_times[-100:]) / min(
                    100, len(embedding_times)
                )
                logger.debug(
                    "Embedding generation progress",
                    processed_samples=i + 1,
                    avg_embedding_time_ms=f"{avg_embedding_time * 1000:.2f}",
                    avg_text_length=sum(text_lengths[-100:])
                    // min(100, len(text_lengths)),
                )

        # Tensor creation and device movement
        tensor_start = time.time()
        embeddings = torch.FloatTensor(np.stack(test_embeddings)).to(device)
        tensor_creation_time = time.time() - tensor_start

        # Verify tensor device consistency
        assert embeddings.device.type == device.type, (
            f"Embeddings device type mismatch: expected {device.type}, got {embeddings.device.type}"
        )

        total_embedding_time = time.time() - embedding_start_time
        logger.debug(
            "Embedding generation completed",
            total_embeddings=len(test_embeddings),
            embeddings_shape=list(embeddings.shape),
            embedding_dimension=embeddings.shape[1],
            total_embedding_time_seconds=f"{total_embedding_time:.3f}",
            tensor_creation_time_ms=f"{tensor_creation_time * 1000:.2f}",
            avg_embedding_time_ms=f"{sum(embedding_times) / len(embedding_times) * 1000:.2f}",
            avg_text_length=sum(text_lengths) // len(text_lengths),
        )

        # Prediction generation phase
        prediction_loop_start = time.time()
        predictions = []
        unknown_questions = 0
        known_questions = 0
        inference_times = []

        logger.debug(
            "Starting prediction loop",
            total_samples=len(test_data),
            model_in_eval_mode=not self.model.training,
            available_questions=len(self.question_misconceptions),
        )

        with torch.no_grad():
            for i, row in enumerate(test_data):
                sample_start_time = time.time()

                # Context binding for this sample
                sample_logger = logger.bind(
                    row_id=row.row_id, question_id=row.question_id, sample_index=i
                )

                embedding_features = embeddings[i].unsqueeze(
                    0
                )  # Shape: [1, embedding_dim]
                sample_logger.debug(
                    "Processing sample",
                    embedding_shape=list(embedding_features.shape),
                    question_text_length=len(row.question_text),
                    answer_length=len(row.mc_answer),
                    explanation_length=len(row.student_explanation),
                )

                # Check if question was in training data
                if row.question_id not in self.question_misconceptions:
                    unknown_questions += 1
                    sample_logger.warning(
                        "Unknown question - using default prediction",
                        available_questions=len(self.question_misconceptions),
                        prediction_strategy="default",
                    )
                    predicted_categories = predictor.create_default_prediction(row)
                else:
                    known_questions += 1

                    # Model inference with detailed logging
                    inference_start = time.time()
                    outputs = self.model(embedding_features, row.question_id)
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)

                    sample_logger.debug(
                        "Model inference completed",
                        inference_time_ms=f"{inference_time * 1000:.2f}",
                        output_heads=list(outputs.keys()),
                        output_shapes={k: list(v.shape) for k, v in outputs.items()},
                    )

                    # Get predictions from model outputs
                    predicted_categories = predictor.get_predictions_from_outputs(
                        outputs,
                        question_id=row.question_id,
                        row=row,
                    )

                    sample_logger.debug(
                        "Category prediction completed",
                        predicted_categories_count=len(predicted_categories),
                        categories=[
                            str(cat.category) for cat in predicted_categories[:3]
                        ],
                        misconceptions=[
                            str(cat.misconception)
                            for cat in predicted_categories
                            if cat.misconception
                        ][:3],
                    )

                # Final prediction assembly
                final_predictions = predicted_categories[:MAX_PREDICTIONS]
                sample_time = time.time() - sample_start_time

                sample_logger.debug(
                    "Sample processing completed",
                    final_prediction_count=len(final_predictions),
                    truncated=len(predicted_categories) > MAX_PREDICTIONS,
                    sample_processing_time_ms=f"{sample_time * 1000:.2f}",
                )

                predictions.append(
                    SubmissionRow(
                        row_id=row.row_id,
                        predicted_categories=final_predictions,
                    )
                )

                # Log progress for large datasets
                if i > 0 and i % 50 == 0:
                    avg_inference_time = (
                        sum(inference_times[-50:]) / min(50, len(inference_times))
                        if inference_times
                        else 0
                    )
                    logger.debug(
                        "Prediction progress",
                        processed_samples=i + 1,
                        known_questions_so_far=known_questions,
                        unknown_questions_so_far=unknown_questions,
                        avg_inference_time_ms=f"{avg_inference_time * 1000:.2f}",
                    )

        # Final performance summary
        prediction_loop_time = time.time() - prediction_loop_start
        total_prediction_time = time.time() - prediction_start_time

        # Calculate statistics
        avg_inference_time = (
            sum(inference_times) / len(inference_times) if inference_times else 0
        )

        logger.bind(operation="mlp_predict").info(
            "MLP prediction process completed",
            total_predictions=len(predictions),
            known_questions=known_questions,
            unknown_questions=unknown_questions,
            question_coverage_pct=f"{(known_questions / len(test_data)) * 100:.1f}"
            if test_data
            else "0.0",
            avg_inference_time_ms=f"{avg_inference_time * 1000:.2f}",
            prediction_loop_time_seconds=f"{prediction_loop_time:.3f}",
            total_time_seconds=f"{total_prediction_time:.3f}",
            throughput_samples_per_second=f"{len(test_data) / total_prediction_time:.1f}"
            if total_prediction_time > 0
            else "inf",
        )

        return predictions
