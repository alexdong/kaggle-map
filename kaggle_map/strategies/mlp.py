"""MLP neural network strategy for student misconception prediction."""

import pickle
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger
from platformdirs import PlatformDirs
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from kaggle_map.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.eval import evaluate
from kaggle_map.models import (
    Answer,
    Category,
    EvaluationRow,
    Prediction,
    QuestionId,
    SubmissionRow,
    TrainingRow,
)

from .base import Strategy

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Constants for multi-head MLP predictions
CORRECTNESS_THRESHOLD = 0.5
CATEGORY_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to include a category
MISCONCEPTION_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for misconception
FALLBACK_THRESHOLD = 0.3  # Add fallback if max prediction confidence below this
MAX_PREDICTIONS = 3  # Maximum number of predictions to return per observation


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Device selection: using MPS (Apple Metal)",
                    device_type="mps", available_backends=["mps", "cuda", "cpu"])
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_count = torch.cuda.device_count()
        logger.debug("Device selection: using CUDA",
                    device_type="cuda", cuda_devices=cuda_count, available_backends=["cuda", "cpu"])
        return device
    device = torch.device("cpu")
    logger.debug("Device selection: fallback to CPU",
                device_type="cpu", available_backends=["cpu"])
    return device


class MLPDataset(Dataset):
    """PyTorch dataset for multi-head MLP training.

    Provides:
    - Embeddings (question/answer/explanation)
    - Correctness labels (binary)
    - Category labels (per question, per correctness state)
    - Misconception labels (when applicable)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        correctness: np.ndarray,
        category_labels: dict[
            QuestionId, dict[str, np.ndarray]
        ],  # 'correct'/'incorrect' -> one-hot vectors
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.embeddings = torch.FloatTensor(embeddings).to(self.device)
        self.correctness = torch.FloatTensor(correctness).unsqueeze(1).to(self.device)
        self.question_ids = question_ids

        # Build label mappings for multi-head training
        self.category_labels = {}
        self.misconception_labels = []
        self._build_label_mappings(category_labels, misconception_labels, question_ids)

    def _build_label_mappings(
        self,
        category_labels: dict[QuestionId, dict[str, np.ndarray]],
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
    ) -> None:
        """Build mappings for multi-head training labels."""
        # Create local index trackers for each question
        question_local_indices = {}
        for qid in set(question_ids):
            question_local_indices[qid] = 0

        # Initialize storage
        for qid in set(question_ids):
            self.category_labels[qid] = {"correct": [], "incorrect": []}

        # Build labels in same order as global dataset
        for _global_idx, qid in enumerate(question_ids):
            local_idx = question_local_indices[qid]

            # Category labels for this sample
            if qid in category_labels:
                # Correct state categories
                if "correct" in category_labels[qid] and local_idx < len(
                    category_labels[qid]["correct"]
                ):
                    correct_label = category_labels[qid]["correct"][local_idx]
                    self.category_labels[qid]["correct"].append(
                        torch.FloatTensor(correct_label).to(self.device)
                    )
                else:
                    # Default: no category (zero vector)
                    self.category_labels[qid]["correct"].append(
                        torch.zeros(1).to(
                            self.device
                        )  # Will be resized based on actual question
                    )

                # Incorrect state categories
                if "incorrect" in category_labels[qid] and local_idx < len(
                    category_labels[qid]["incorrect"]
                ):
                    incorrect_label = category_labels[qid]["incorrect"][local_idx]
                    self.category_labels[qid]["incorrect"].append(
                        torch.FloatTensor(incorrect_label).to(self.device)
                    )
                else:
                    self.category_labels[qid]["incorrect"].append(
                        torch.zeros(1).to(self.device)
                    )

            # Misconception labels
            if qid in misconception_labels and local_idx < len(
                misconception_labels[qid]
            ):
                misc_label = misconception_labels[qid][local_idx]
                self.misconception_labels.append(
                    torch.FloatTensor(misc_label).to(self.device)
                )
            else:
                # Default: no misconception (zero vector)
                self.misconception_labels.append(
                    torch.zeros(1).to(
                        self.device
                    )  # Will be resized based on actual question
                )

            question_local_indices[qid] += 1

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, int]:
        """Get training sample with multi-head labels.

        Returns:
            features: Embedding tensor (384-dim)
            labels: Dictionary with correctness, category, and misconception labels
            question_id: Question ID
            idx: Sample index
        """
        assert 0 <= idx < len(self.embeddings), (
            f"Index {idx} out of range [0, {len(self.embeddings)})"
        )
        question_id = self.question_ids[idx]

        # Features: embedding only (no ground truth correctness)
        features = self.embeddings[idx]
        # Note: On MPS, device names can vary (mps vs mps:0), so check device type instead
        assert features.device.type == self.device.type, (
            f"Features device type mismatch: expected {self.device.type}, got {features.device.type}"
        )

        # Multi-head labels
        labels = {
            "correctness": self.correctness[
                idx
            ],  # Ground truth correctness for training
            "misconceptions": self.misconception_labels[idx],
        }

        # Add question-specific category labels
        # Find which local index this global index corresponds to for this question
        question_samples = [
            i for i, qid in enumerate(self.question_ids) if qid == question_id
        ]
        local_idx = question_samples.index(idx)

        if question_id in self.category_labels:
            if local_idx < len(self.category_labels[question_id]["correct"]):
                labels["correct_categories"] = self.category_labels[question_id][
                    "correct"
                ][local_idx]
            if local_idx < len(self.category_labels[question_id]["incorrect"]):
                labels["incorrect_categories"] = self.category_labels[question_id][
                    "incorrect"
                ][local_idx]

        # Ensure all labels are on correct device (check device type for MPS compatibility)
        for key, tensor in labels.items():
            assert tensor.device.type == self.device.type, (
                f"{key} label device type mismatch: expected {self.device.type}, got {tensor.device.type}"
            )

        return features, labels, question_id, idx


class MLPNet(nn.Module):
    """Multi-head MLP for direct category prediction aligned with competition format.

    Architecture:
    1. Shared embedding trunk: processes question/answer/explanation
    2. Correctness head: predicts if student answer is correct
    3. Category heads: predict category distributions for correct/incorrect states

    This directly mirrors the baseline's two-step process:
    - Determine correctness (neural vs rule-based)
    - Select category within correctness state (neural vs frequency-based)
    """

    def __init__(
        self,
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
    ) -> None:
        super().__init__()

        # Store metadata for prediction reconstruction
        self.question_categories = question_categories
        self.question_misconceptions = question_misconceptions

        # Shared trunk: embedding(384) only, no correctness input
        # Let the model learn correctness from content, not ground truth
        self.shared_trunk = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Correctness prediction head (binary classification)
        self.correctness_head = nn.Linear(128, 1)

        # Category prediction heads - separate for correct/incorrect states
        self.max_categories = 6  # Maximum possible categories (True_/False_ x Correct/Neither/Misconception)

        # Question-specific category heads
        self.correct_category_heads = nn.ModuleDict()
        self.incorrect_category_heads = nn.ModuleDict()

        for question_id, category_map in question_categories.items():
            qid_str = str(question_id)

            # Head for correct answer categories - use unique categories with consistent ordering
            if True in category_map:
                unique_correct_cats = sorted(set(category_map[True]), key=str)
                self.correct_category_heads[qid_str] = nn.Linear(
                    128, len(unique_correct_cats)
                )
                # Store category ordering for consistent label creation
                if not hasattr(self, "question_category_orders"):
                    self.question_category_orders = {}
                if question_id not in self.question_category_orders:
                    self.question_category_orders[question_id] = {}
                self.question_category_orders[question_id][True] = unique_correct_cats

            # Head for incorrect answer categories - use unique categories with consistent ordering
            if False in category_map:
                unique_incorrect_cats = sorted(set(category_map[False]), key=str)
                self.incorrect_category_heads[qid_str] = nn.Linear(
                    128, len(unique_incorrect_cats)
                )
                # Store category ordering for consistent label creation
                if not hasattr(self, "question_category_orders"):
                    self.question_category_orders = {}
                if question_id not in self.question_category_orders:
                    self.question_category_orders[question_id] = {}
                self.question_category_orders[question_id][False] = unique_incorrect_cats

        # Misconception prediction heads (for when category is *_Misconception)
        self.misconception_heads = nn.ModuleDict()
        for question_id, misconceptions in question_misconceptions.items():
            if misconceptions:  # Only create head if question has misconceptions
                self.misconception_heads[str(question_id)] = nn.Linear(
                    128, len(misconceptions)
                )

    def forward(self, x: torch.Tensor, question_id: int) -> dict[str, torch.Tensor]:
        """Multi-head forward pass returning all prediction components.

        Args:
            x: Input tensor (embedding only, shape: [batch_size, 384])
            question_id: Question ID for question-specific heads

        Returns:
            Dictionary with keys:
            - 'correctness': Correctness prediction logits [batch_size, 1]
            - 'correct_categories': Category logits for correct state (if applicable)
            - 'incorrect_categories': Category logits for incorrect state (if applicable)
            - 'misconceptions': Misconception logits (if question has misconceptions)
        """
        assert isinstance(x, torch.Tensor), f"Input must be a tensor, got {type(x)}"
        qid_str = str(question_id)

        # Ensure input tensor is on the same device as model (check device type for MPS)
        model_device = next(self.parameters()).device
        assert x.device.type == model_device.type, (
            f"Input device type mismatch: model on {model_device.type}, input on {x.device.type}"
        )

        # Shared feature extraction
        shared_features = self.shared_trunk(x)

        # Multi-head outputs
        outputs = {}

        # Correctness prediction (always present)
        outputs["correctness"] = self.correctness_head(shared_features)

        # Category predictions (question-specific)
        if qid_str in self.correct_category_heads:
            outputs["correct_categories"] = self.correct_category_heads[qid_str](
                shared_features
            )

        if qid_str in self.incorrect_category_heads:
            outputs["incorrect_categories"] = self.incorrect_category_heads[qid_str](
                shared_features
            )

        # Misconception predictions (if question has misconceptions)
        if qid_str in self.misconception_heads:
            outputs["misconceptions"] = self.misconception_heads[qid_str](
                shared_features
            )

        # Ensure all outputs are on correct device (check device type for MPS)
        for key, tensor in outputs.items():
            assert tensor.device.type == model_device.type, (
                f"{key} device type mismatch: expected {model_device.type}, got {tensor.device.type}"
            )

        return outputs


@dataclass(frozen=True)
class MLPStrategy(Strategy):
    """MLP strategy for predicting student misconceptions.

    Uses a neural network with question-specific heads to predict misconceptions,
    then reconstructs categories deterministically based on answer correctness.
    """

    model: MLPNet
    correct_answers: dict[QuestionId, Answer]
    question_misconceptions: dict[QuestionId, list[str]]
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
            log_dir=str(log_dir)
        ).info("Starting MLP strategy fit")

        # Set random seeds for deterministic training
        logger.debug("Setting random seeds for deterministic training", seed=random_seed)
        cls._set_random_seeds(random_seed)
        logger.debug("Random seeds configured",
                    seed=random_seed,
                    torch_seed=torch.initial_seed(),
                    numpy_deterministic=True)

        # Parse training data
        logger.debug("Starting training data parsing", file_path=str(train_csv_path))
        parse_start = time.time()
        all_training_data = cls._parse_training_data(train_csv_path)
        parse_duration = time.time() - parse_start
        logger.debug("Training data parsing completed",
                    total_rows=len(all_training_data),
                    parse_time_seconds=f"{parse_duration:.3f}")

        # Split data if train_split < 1.0
        if train_split < 1.0:
            logger.debug("Splitting data for train/eval",
                        train_split=train_split,
                        total_samples=len(all_training_data),
                        stratify_by="question_id")
            split_start = time.time()
            training_data, eval_data = train_test_split(
                all_training_data,
                train_size=train_split,
                random_state=random_seed,
                stratify=[
                    row.question_id for row in all_training_data
                ],  # Stratify by question
            )
            split_duration = time.time() - split_start
            logger.debug("Data split completed",
                        training_samples=len(training_data),
                        eval_samples=len(eval_data),
                        split_time_seconds=f"{split_duration:.3f}")
            # Store eval data for later use
            cls._eval_data = eval_data
        else:
            logger.debug("Using all data for training", train_split=1.0)
            training_data = all_training_data
            cls._eval_data = None

        # Extract metadata for new multi-head architecture
        logger.debug("Starting metadata extraction for multi-head architecture")
        metadata_start = time.time()
        
        correct_answers = cls._extract_correct_answers(training_data)
        question_categories = cls._extract_question_categories(
            training_data, correct_answers
        )
        question_misconceptions = cls._extract_question_misconceptions(training_data)
        
        metadata_duration = time.time() - metadata_start
        logger.debug("Metadata extraction completed",
                    questions_with_correct_answers=len(correct_answers),
                    questions_with_categories=len(question_categories),
                    questions_with_misconceptions=len(question_misconceptions),
                    extraction_time_seconds=f"{metadata_duration:.3f}")

        # Generate or load embeddings and labels for multi-head training
        embedding_model = EmbeddingModel.MINI_LM
        embeddings_start = time.time()
        
        if embeddings_path is not None and embeddings_path.exists():
            logger.info("Loading pre-computed embeddings",
                       embeddings_path=str(embeddings_path),
                       file_exists=True,
                       embedding_model=embedding_model.model_id)
            (
                embeddings,
                correctness,
                category_labels,
                misconception_labels,
                question_ids,
            ) = cls._load_precomputed_embeddings_multihead(
                training_data,
                correct_answers,
                question_categories,
                question_misconceptions,
                embeddings_path,
            )
        else:
            logger.info("Generating embeddings and labels for multi-head training",
                       embedding_model=embedding_model.model_id,
                       training_samples=len(training_data))
            (
                embeddings,
                correctness,
                category_labels,
                misconception_labels,
                question_ids,
            ) = cls._prepare_training_data_multihead(
                training_data,
                correct_answers,
                question_categories,
                question_misconceptions,
                embedding_model,
            )
            
        embeddings_duration = time.time() - embeddings_start
        logger.debug("Embeddings preparation completed",
                    embeddings_shape=embeddings.shape,
                    correctness_shape=correctness.shape,
                    question_ids_shape=question_ids.shape,
                    preparation_time_seconds=f"{embeddings_duration:.3f}")

        # Get device and train multi-head model
        device = get_device()
        training_start = time.time()
        logger.info("Starting multi-head MLP training",
                   device=str(device),
                   device_type=device.type,
                   embeddings_shape=embeddings.shape,
                   training_samples=len(training_data))
        
        model = cls._train_multihead_model(
            embeddings,
            correctness,
            category_labels,
            misconception_labels,
            question_ids,
            question_categories,
            question_misconceptions,
            device,
        )
        
        training_duration = time.time() - training_start
        total_duration = time.time() - fit_start_time
        
        logger.info("MLP strategy fit completed",
                   training_time_seconds=f"{training_duration:.3f}",
                   total_fit_time_seconds=f"{total_duration:.3f}",
                   model_parameters=sum(p.numel() for p in model.parameters()),
                   device=str(device))

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
            correctness_threshold=CORRECTNESS_THRESHOLD,
            category_confidence_threshold=CATEGORY_CONFIDENCE_THRESHOLD,
            misconception_confidence_threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD
        )

        if not test_data:
            logger.debug("Empty test data provided, returning empty predictions")
            return []

        # Device setup and model preparation
        device_start = time.time()
        device = get_device()
        logger.debug("Device selection completed",
                    selected_device=str(device),
                    device_type=device.type,
                    device_selection_time_ms=f"{(time.time() - device_start) * 1000:.2f}")
        
        model_setup_start = time.time()
        self.model.to(device)
        self.model.eval()  # Set evaluation mode early
        
        # Verify model device consistency
        model_device = next(self.model.parameters()).device
        assert model_device == device, (
            f"Model device mismatch: expected {device}, got {model_device}"
        )
        model_setup_time = time.time() - model_setup_start
        
        logger.debug("Model setup completed",
                    model_device=str(model_device),
                    device_matches=True,
                    model_parameters=sum(p.numel() for p in self.model.parameters()),
                    model_setup_time_ms=f"{model_setup_time * 1000:.2f}")

        # Embedding generation phase
        embedding_start_time = time.time()
        logger.debug("Starting embedding generation",
                    embedding_model=self.embedding_model.model_id,
                    test_samples=len(test_data))
        
        tokenizer_start = time.time()
        tokenizer = get_tokenizer(self.embedding_model)
        tokenizer_load_time = time.time() - tokenizer_start
        
        logger.debug("Tokenizer loaded",
                    tokenizer_type=type(tokenizer).__name__,
                    load_time_ms=f"{tokenizer_load_time * 1000:.2f}")
        
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
                avg_embedding_time = sum(embedding_times[-100:]) / min(100, len(embedding_times))
                logger.debug("Embedding generation progress",
                           processed_samples=i + 1,
                           avg_embedding_time_ms=f"{avg_embedding_time * 1000:.2f}",
                           avg_text_length=sum(text_lengths[-100:]) // min(100, len(text_lengths)))
        
        # Tensor creation and device movement
        tensor_start = time.time()
        embeddings = torch.FloatTensor(np.stack(test_embeddings)).to(device)
        tensor_creation_time = time.time() - tensor_start
        
        # Verify tensor device consistency
        assert embeddings.device.type == device.type, (
            f"Embeddings device type mismatch: expected {device.type}, got {embeddings.device.type}"
        )
        
        total_embedding_time = time.time() - embedding_start_time
        logger.debug("Embedding generation completed",
                    total_embeddings=len(test_embeddings),
                    embeddings_shape=list(embeddings.shape),
                    embedding_dimension=embeddings.shape[1],
                    total_embedding_time_seconds=f"{total_embedding_time:.3f}",
                    tensor_creation_time_ms=f"{tensor_creation_time * 1000:.2f}",
                    avg_embedding_time_ms=f"{sum(embedding_times) / len(embedding_times) * 1000:.2f}",
                    avg_text_length=sum(text_lengths) // len(text_lengths))

        # Prediction generation phase
        prediction_loop_start = time.time()
        predictions = []
        unknown_questions = 0
        known_questions = 0
        correctness_predictions = []
        category_prediction_counts = {"correct": 0, "incorrect": 0}
        inference_times = []

        logger.debug("Starting prediction loop",
                    total_samples=len(test_data),
                    model_in_eval_mode=not self.model.training,
                    available_questions=len(getattr(self.model, "question_categories", {})))

        with torch.no_grad():
            for i, row in enumerate(test_data):
                sample_start_time = time.time()
                
                # Context binding for this sample
                sample_logger = logger.bind(
                    row_id=row.row_id,
                    question_id=row.question_id,
                    sample_index=i
                )
                
                embedding_features = embeddings[i].unsqueeze(0)  # Shape: [1, 384]
                sample_logger.debug("Processing sample",
                                   embedding_shape=list(embedding_features.shape),
                                   question_text_length=len(row.question_text),
                                   answer_length=len(row.mc_answer),
                                   explanation_length=len(row.student_explanation))

                # Check if question was in training data
                if (
                    not hasattr(self.model, "question_categories")
                    or row.question_id not in self.model.question_categories
                ):
                    unknown_questions += 1
                    sample_logger.warning("Unknown question - using default prediction",
                                         available_questions=len(getattr(self.model, "question_categories", {})),
                                         prediction_strategy="default")
                    predicted_categories = self._create_default_prediction(row)
                else:
                    known_questions += 1
                    
                    # Model inference with detailed logging
                    inference_start = time.time()
                    outputs = self.model(embedding_features, row.question_id)
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    sample_logger.debug("Model inference completed",
                                       inference_time_ms=f"{inference_time * 1000:.2f}",
                                       output_heads=list(outputs.keys()),
                                       output_shapes={k: list(v.shape) for k, v in outputs.items()})

                    # Correctness prediction with detailed state
                    correctness_logit = outputs["correctness"].squeeze().cpu().numpy()
                    predicted_correctness = float(correctness_logit > 0)
                    correctness_sigmoid = 1 / (1 + np.exp(-correctness_logit))  # Convert to probability
                    correctness_predictions.append(predicted_correctness)
                    
                    sample_logger.debug("Correctness prediction",
                                       correctness_logit=float(correctness_logit),
                                       correctness_probability=float(correctness_sigmoid),
                                       predicted_correct=bool(predicted_correctness),
                                       threshold=CORRECTNESS_THRESHOLD)
                    
                    # Track correctness distribution
                    if predicted_correctness > CORRECTNESS_THRESHOLD:
                        category_prediction_counts["correct"] += 1
                    else:
                        category_prediction_counts["incorrect"] += 1

                    # Category reconstruction with detailed logging
                    predicted_categories = self._reconstruct_multihead_predictions(
                        outputs,
                        predicted_correctness=predicted_correctness > CORRECTNESS_THRESHOLD,
                        question_id=row.question_id,
                    )
                    
                    sample_logger.debug("Category prediction completed",
                                       predicted_categories_count=len(predicted_categories),
                                       categories=[str(cat.category) for cat in predicted_categories[:3]],
                                       misconceptions=[cat.misconception for cat in predicted_categories if cat.misconception][:3])
                
                # Final prediction assembly
                final_predictions = predicted_categories[:MAX_PREDICTIONS]
                sample_time = time.time() - sample_start_time
                
                sample_logger.debug("Sample processing completed",
                                   final_prediction_count=len(final_predictions),
                                   truncated=len(predicted_categories) > MAX_PREDICTIONS,
                                   sample_processing_time_ms=f"{sample_time * 1000:.2f}")
                
                predictions.append(
                    SubmissionRow(
                        row_id=row.row_id,
                        predicted_categories=final_predictions,
                    )
                )
                
                # Log progress for large datasets
                if i > 0 and i % 50 == 0:
                    avg_inference_time = sum(inference_times[-50:]) / min(50, len(inference_times)) if inference_times else 0
                    logger.debug("Prediction progress",
                               processed_samples=i + 1,
                               known_questions_so_far=known_questions,
                               unknown_questions_so_far=unknown_questions,
                               avg_inference_time_ms=f"{avg_inference_time * 1000:.2f}")
        
        # Final performance summary
        prediction_loop_time = time.time() - prediction_loop_start
        total_prediction_time = time.time() - prediction_start_time
        
        # Calculate statistics
        avg_correctness = sum(correctness_predictions) / len(correctness_predictions) if correctness_predictions else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        logger.bind(operation="mlp_predict").info(
            "MLP prediction process completed",
            total_predictions=len(predictions),
            known_questions=known_questions,
            unknown_questions=unknown_questions,
            question_coverage_pct=f"{(known_questions / len(test_data)) * 100:.1f}" if test_data else "0.0",
            predicted_correct_pct=f"{(category_prediction_counts['correct'] / known_questions) * 100:.1f}" if known_questions > 0 else "0.0",
            predicted_incorrect_pct=f"{(category_prediction_counts['incorrect'] / known_questions) * 100:.1f}" if known_questions > 0 else "0.0",
            avg_correctness_confidence=f"{avg_correctness:.3f}",
            avg_inference_time_ms=f"{avg_inference_time * 1000:.2f}",
            prediction_loop_time_seconds=f"{prediction_loop_time:.3f}",
            total_time_seconds=f"{total_prediction_time:.3f}",
            throughput_samples_per_second=f"{len(test_data) / total_prediction_time:.1f}" if total_prediction_time > 0 else "inf"
        )

        return predictions

    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        logger.info(f"Saving MLP model to {filepath}")

        # Save multi-head model state and metadata
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "correct_answers": self.correct_answers,
            "question_categories": getattr(self.model, "question_categories", {}),
            "question_misconceptions": self.question_misconceptions,
            "embedding_model": self.embedding_model.model_id,
            "eval_data": getattr(
                self.__class__, "_eval_data", None
            ),  # Save eval data if available
        }

        with filepath.open("wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath: Path) -> "MLPStrategy":
        """Load model from disk."""
        logger.info(f"Loading MLP model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        with filepath.open("rb") as f:
            save_data = pickle.load(f)

        # Reconstruct multi-head model
        question_categories = save_data.get("question_categories", {})
        question_misconceptions = save_data["question_misconceptions"]
        model = MLPNet(question_categories, question_misconceptions)
        model.load_state_dict(save_data["model_state_dict"])

        # Find embedding model
        embedding_model = None
        for em in EmbeddingModel.all():
            if em.model_id == save_data["embedding_model"]:
                embedding_model = em
                break

        if embedding_model is None:
            raise ValueError(f"Unknown embedding model: {save_data['embedding_model']}")

        # Restore eval data if available
        if "eval_data" in save_data and save_data["eval_data"] is not None:
            cls._eval_data = save_data["eval_data"]
            logger.debug(f"Restored {len(save_data['eval_data'])} eval data rows")

        return cls(
            model=model,
            correct_answers=save_data["correct_answers"],
            question_misconceptions=question_misconceptions,
            embedding_model=embedding_model,
        )

    def display_stats(self, console: Console) -> None:
        """Display model statistics."""
        stats_table = Table(title="MLP Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="magenta")

        stats_table.add_row(
            "Questions with correct answers", str(len(self.correct_answers))
        )
        stats_table.add_row(
            "Questions with misconceptions", str(len(self.question_misconceptions))
        )

        # Get model device
        model_device = next(self.model.parameters()).device
        stats_table.add_row("Model device", str(model_device))

        total_params = sum(p.numel() for p in self.model.parameters())
        stats_table.add_row("Total model parameters", f"{total_params:,}")

        console.print(stats_table)

    def display_detailed_info(self, console: Console) -> None:
        """Display detailed model info for verbose mode."""
        console.print("\n[bold]Detailed MLP Model Contents[/bold]")
        self._display_architecture(console)
        self._display_question_misconceptions(console)

    def _display_architecture(self, console: Console) -> None:
        console.print("\n[cyan]Model Architecture:[/cyan]")
        console.print(f"  Embedding model: {self.embedding_model.model_id}")
        console.print("  Input dimensions: 385 (384 embedding + 1 correctness)")
        console.print("  Shared trunk: 385 → 512 → 256 → 128")
        console.print(f"  Question-specific heads: {len(self.question_misconceptions)}")

    def _display_question_misconceptions(self, console: Console) -> None:
        console.print("\n[cyan]Question-specific misconceptions:[/cyan]")
        for qid, misconceptions in sorted(self.question_misconceptions.items()):
            misconception_str = ", ".join(misconceptions)
            console.print(f"  Q{qid}: [{misconception_str}]")

    def demonstrate_predictions(self, console: Console) -> None:
        """Show sample predictions to validate the model works."""
        # Test prediction format with a sample
        sample_test_row = EvaluationRow(
            row_id=99999,
            question_id=next(
                iter(self.question_misconceptions.keys())
            ),  # Use first available question
            question_text="Sample question",
            mc_answer="Sample answer",
            student_explanation="Sample explanation",
        )

        sample_predictions = self.predict([sample_test_row])

        console.print("\n[bold]Sample MLP Prediction Test[/bold]")
        console.print(f"Row ID: {sample_predictions[0].row_id}")
        console.print(
            f"Predictions: {[str(pred) for pred in sample_predictions[0].predicted_categories]}"
        )

    # Implementation helper methods

    @staticmethod
    def _set_random_seeds(seed: int) -> None:
        """Set random seeds for deterministic training."""
        random.seed(seed)
        # Use simple seed setting (legacy but functional)
        np.random.seed(seed)  # noqa: NPY002
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            # MPS doesn't have separate seed setting, but torch.manual_seed covers it
            pass
        # Make torch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def evaluate_on_split(
        cls, model: "MLPStrategy", eval_data: list[TrainingRow] | None = None
    ) -> dict[str, float]:
        """Evaluate model performance on eval split.

        Args:
            model: Trained MLPStrategy model
            eval_data: Evaluation data (uses stored split if None)

        Returns:
            Dictionary with evaluation metrics
        """
        if eval_data is None:
            eval_data = getattr(cls, "_eval_data", None)

        if eval_data is None:
            msg = "No evaluation data available. Train with train_split < 1.0 first."
            raise ValueError(msg)

        logger.info(f"Evaluating model on {len(eval_data)} validation samples")

        # Convert training rows to evaluation rows for prediction
        eval_rows = [
            EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )
            for row in eval_data
        ]

        # Make predictions
        predictions = model.predict(eval_rows)

        # Use existing evaluation pipeline with temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create ground truth CSV
            ground_truth_data = [
                {
                    "row_id": row.row_id,
                    "Category": row.category.value,
                    "Misconception": row.misconception
                    if row.misconception is not None
                    else "NA",
                }
                for row in eval_data
            ]

            ground_truth_df = pd.DataFrame(ground_truth_data)
            ground_truth_path = tmp_path / "ground_truth.csv"
            ground_truth_df.to_csv(ground_truth_path, index=False)

            # Create submission CSV
            submission_data = []
            for pred in predictions:
                prediction_strs = [str(p) for p in pred.predicted_categories]
                submission_data.append(
                    {"row_id": pred.row_id, "predictions": " ".join(prediction_strs)}
                )

            submission_df = pd.DataFrame(submission_data)
            submission_path = tmp_path / "submission.csv"
            submission_df.to_csv(submission_path, index=False)

            # Use existing evaluate function
            eval_result = evaluate(ground_truth_path, submission_path)

            return {
                "map_score": eval_result.map_score,
                "total_observations": eval_result.total_observations,
                "perfect_predictions": eval_result.perfect_predictions,
                "accuracy": eval_result.perfect_predictions
                / eval_result.total_observations
                if eval_result.total_observations > 0
                else 0.0,
            }

    def _is_answer_correct(
        self, question_id: QuestionId, student_answer: Answer
    ) -> bool:
        """Check if student answer matches the correct answer."""
        correct_answer = self.correct_answers.get(question_id, "")
        return student_answer == correct_answer

    def _reconstruct_multihead_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        predicted_correctness: bool,
        question_id: QuestionId,
    ) -> list[Prediction]:
        """Reconstruct predictions from multi-head model outputs.

        This method mirrors the baseline's approach:
        1. Determine correctness (neural vs rule-based)
        2. Select best categories within that correctness state
        3. Add misconceptions when appropriate
        """
        reconstruction_start = time.time()
        prefix = "True_" if predicted_correctness else "False_"
        predictions = []
        
        # Context binding for reconstruction logging
        recon_logger = logger.bind(
            question_id=question_id,
            predicted_correctness=predicted_correctness,
            prefix=prefix
        )
        
        recon_logger.debug("Starting prediction reconstruction",
                          output_heads=list(outputs.keys()),
                          output_tensor_shapes={k: list(v.shape) for k, v in outputs.items()})

        # Get category probabilities for the predicted correctness state
        category_key = (
            "correct_categories" if predicted_correctness else "incorrect_categories"
        )
        
        recon_logger.debug("Category head selection",
                          category_key=category_key,
                          category_head_available=category_key in outputs)

        if category_key in outputs:
            # Process category predictions with detailed logging
            category_processing_start = time.time()
            category_logits = outputs[category_key].squeeze().cpu().numpy()
            category_probs = self._softmax(category_logits)
            
            recon_logger.debug("Category logits processed",
                              category_logits_shape=category_logits.shape,
                              category_logits_range=[float(category_logits.min()), float(category_logits.max())],
                              category_probs_sum=float(category_probs.sum()))

            # Get available categories for this correctness state with consistent ordering
            # Must match the ordering used in model head creation and label generation
            available_categories = sorted(
                set(
                    self.model.question_categories[question_id].get(
                        predicted_correctness, []
                    )
                ),
                key=str
            )
            
            recon_logger.debug("Available categories loaded",
                              available_categories_count=len(available_categories),
                              categories=[str(cat) for cat in available_categories[:5]],
                              prob_category_alignment_check=len(category_probs) == len(available_categories))
            
            # Validate dimension alignment
            if len(category_probs) != len(available_categories):
                recon_logger.error("Category dimension mismatch",
                                 category_probs_length=len(category_probs),
                                 available_categories_length=len(available_categories),
                                 error_type="dimension_mismatch")

            # Add category predictions in order of confidence
            category_candidates = []
            misconception_processing_count = 0
            
            for i, prob in enumerate(category_probs):
                if (
                    i < len(available_categories)
                    and prob > CATEGORY_CONFIDENCE_THRESHOLD
                ):
                    category = available_categories[i]
                    
                    recon_logger.debug("Category candidate found",
                                      category_index=i,
                                      category=str(category),
                                      probability=float(prob),
                                      confidence_threshold=CATEGORY_CONFIDENCE_THRESHOLD,
                                      is_misconception_category=category.is_misconception)

                    # Check if this is a misconception category
                    if category.is_misconception and "misconceptions" in outputs:
                        misconception_processing_count += 1
                        
                        # Get misconception predictions with detailed logging
                        misconception_start = time.time()
                        misconception_logits = (
                            outputs["misconceptions"].squeeze().cpu().numpy()
                        )
                        misconception_probs = self._sigmoid(misconception_logits)
                        misconception_processing_time = time.time() - misconception_start
                        
                        recon_logger.debug("Misconception processing",
                                          misconception_logits_shape=misconception_logits.shape,
                                          misconception_probs_range=[float(misconception_probs.min()), float(misconception_probs.max())],
                                          processing_time_ms=f"{misconception_processing_time * 1000:.2f}")

                        # Find best misconception with logging
                        best_misconception = self._get_best_misconception(
                            misconception_probs, question_id
                        )
                        
                        recon_logger.debug("Best misconception selected",
                                          best_misconception=best_misconception,
                                          misconception_found=best_misconception is not None)

                        prediction_tuple = (
                            Prediction(
                                category=category, misconception=best_misconception
                            ),
                            prob,
                        )
                        predictions.append(prediction_tuple)
                        category_candidates.append((str(category), best_misconception, float(prob)))
                    else:
                        prediction_tuple = (Prediction(category=category), prob)
                        predictions.append(prediction_tuple)
                        category_candidates.append((str(category), None, float(prob)))
            
            category_processing_time = time.time() - category_processing_start
            recon_logger.debug("Category processing completed",
                              total_candidates=len(category_candidates),
                              misconception_categories_processed=misconception_processing_count,
                              processing_time_ms=f"{category_processing_time * 1000:.2f}",
                              candidate_details=category_candidates[:3])  # Show top 3 for brevity
        else:
            recon_logger.debug("No category head available for correctness state",
                              available_heads=list(outputs.keys()),
                              required_head=category_key)

        # Fallback handling with detailed logging
        max_confidence = max(pred[1] for pred in predictions) if predictions else 0.0
        needs_fallback = not predictions or max_confidence < FALLBACK_THRESHOLD
        
        recon_logger.debug("Fallback evaluation",
                          predictions_count=len(predictions),
                          max_confidence=float(max_confidence),
                          fallback_threshold=FALLBACK_THRESHOLD,
                          needs_fallback=needs_fallback)
        
        if needs_fallback:
            fallback_category = Category(f"{prefix}Neither")
            fallback_prediction = (Prediction(category=fallback_category), 0.4)
            predictions.append(fallback_prediction)
            
            recon_logger.debug("Fallback prediction added",
                              fallback_category=str(fallback_category),
                              fallback_confidence=0.4,
                              reason="low_confidence" if predictions else "no_predictions")

        # Final selection and sorting with performance logging
        selection_start = time.time()
        final_predictions = self._select_top_predictions(predictions)
        selection_time = time.time() - selection_start
        
        total_reconstruction_time = time.time() - reconstruction_start
        
        recon_logger.debug("Prediction reconstruction completed",
                          raw_predictions_count=len(predictions),
                          final_predictions_count=len(final_predictions),
                          final_categories=[str(pred.category) for pred in final_predictions],
                          final_misconceptions=[pred.misconception for pred in final_predictions if pred.misconception],
                          selection_time_ms=f"{selection_time * 1000:.2f}",
                          total_reconstruction_time_ms=f"{total_reconstruction_time * 1000:.2f}")

        return final_predictions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        logger.debug("Computing softmax",
                    input_shape=x.shape,
                    input_range=[float(x.min()), float(x.max())],
                    input_mean=float(x.mean()))
        
        # Numerical stability: subtract max to prevent overflow
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        softmax_probs = exp_x / np.sum(exp_x)
        
        logger.debug("Softmax computed",
                    output_sum=float(softmax_probs.sum()),
                    output_max=float(softmax_probs.max()),
                    output_min=float(softmax_probs.min()),
                    numerical_stability_offset=float(max_x))
        
        return softmax_probs

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid probabilities."""
        logger.debug("Computing sigmoid",
                    input_shape=x.shape,
                    input_range=[float(x.min()), float(x.max())],
                    input_mean=float(x.mean()))
        
        # Numerical stability for sigmoid computation
        sigmoid_probs = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
        logger.debug("Sigmoid computed",
                    output_range=[float(sigmoid_probs.min()), float(sigmoid_probs.max())],
                    output_mean=float(sigmoid_probs.mean()),
                    values_above_05=(sigmoid_probs > 0.5).sum())
        
        return sigmoid_probs

    def _get_best_misconception(
        self, misconception_probs: np.ndarray, question_id: QuestionId
    ) -> str | None:
        """Get the most likely misconception for a question."""
        logger.debug("Finding best misconception",
                    question_id=question_id,
                    misconception_probs_shape=misconception_probs.shape,
                    confidence_threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD)
        
        if question_id not in self.question_misconceptions:
            logger.debug("No misconceptions available for question",
                        question_id=question_id,
                        available_questions=len(self.question_misconceptions))
            return None

        misconceptions = self.question_misconceptions[question_id]
        logger.debug("Misconceptions loaded",
                    question_id=question_id,
                    total_misconceptions=len(misconceptions),
                    misconceptions=misconceptions[:3])  # Show first 3 for brevity

        # Exclude NA (last element) from consideration
        if len(misconceptions) > 1:
            # Get probabilities excluding the NA class
            valid_probs = misconception_probs[:-1]
            best_idx = np.argmax(valid_probs)
            best_prob = misconception_probs[best_idx]
            best_misconception = misconceptions[best_idx]
            
            logger.debug("Best misconception analysis",
                        best_index=int(best_idx),
                        best_misconception=best_misconception,
                        best_probability=float(best_prob),
                        threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD,
                        above_threshold=best_prob > MISCONCEPTION_CONFIDENCE_THRESHOLD,
                        na_probability=float(misconception_probs[-1]))
            
            if best_prob > MISCONCEPTION_CONFIDENCE_THRESHOLD:
                logger.debug("Misconception selected",
                            selected_misconception=best_misconception,
                            confidence=float(best_prob))
                return best_misconception
            logger.debug("No misconception above threshold",
                        best_probability=float(best_prob),
                        threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD)
        else:
            logger.debug("Insufficient misconceptions for selection",
                        misconceptions_count=len(misconceptions))

        return None

    # Removed old single-head prediction methods - replaced with multi-head approach

    def _select_top_predictions(self, predictions: list) -> list[Prediction]:
        """Select top unique predictions by confidence."""
        logger.debug("Selecting top predictions",
                    total_predictions=len(predictions),
                    max_predictions=MAX_PREDICTIONS)
        
        # Sort by confidence (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Log sorted predictions for debugging
        if predictions:
            sorted_details = [(str(pred.category), pred.misconception, float(score))
                             for pred, score in predictions[:5]]  # Show top 5
            logger.debug("Predictions sorted by confidence",
                        top_predictions=sorted_details,
                        highest_confidence=float(predictions[0][1]) if predictions else 0.0)

        unique_predictions = []
        seen_categories = set()
        duplicates_filtered = 0
        
        for pred, score in predictions:
            if pred.category not in seen_categories:
                unique_predictions.append(pred)
                seen_categories.add(pred.category)
                logger.debug("Prediction selected",
                           category=str(pred.category),
                           misconception=pred.misconception,
                           confidence=float(score),
                           position=len(unique_predictions))
            else:
                duplicates_filtered += 1
                logger.debug("Duplicate category filtered",
                           category=str(pred.category),
                           confidence=float(score))
            
            if len(unique_predictions) >= MAX_PREDICTIONS:
                break

        logger.debug("Top prediction selection completed",
                    unique_predictions_count=len(unique_predictions),
                    duplicates_filtered=duplicates_filtered,
                    final_categories=[str(pred.category) for pred in unique_predictions])

        return unique_predictions

    def _create_default_prediction(self, row: EvaluationRow) -> list[Prediction]:
        """Create default prediction for unknown questions."""
        logger.debug("Creating default prediction for unknown question",
                    question_id=row.question_id,
                    row_id=row.row_id)
        
        # Check correctness using available correct answers
        correctness_check_start = time.time()
        is_correct = self._is_answer_correct(row.question_id, row.mc_answer)
        correctness_check_time = time.time() - correctness_check_start
        
        prefix = "True_" if is_correct else "False_"
        default_category = Category(f"{prefix}Neither")
        default_prediction = Prediction(category=default_category)
        
        logger.debug("Default prediction created",
                    question_id=row.question_id,
                    predicted_correctness=is_correct,
                    correctness_source="rule_based",
                    default_category=str(default_category),
                    correctness_check_time_ms=f"{correctness_check_time * 1000:.2f}",
                    available_correct_answers=len(self.correct_answers))
        
        return [default_prediction]

    @staticmethod
    def _parse_training_data(csv_path: Path) -> list[TrainingRow]:
        """Parse CSV into strongly-typed training rows."""
        logger.debug("Starting CSV parsing", file_path=str(csv_path))
        assert csv_path.exists(), f"Training file not found: {csv_path}"
        
        file_size = csv_path.stat().st_size
        logger.debug("File validation passed",
                    file_exists=True,
                    file_size_bytes=file_size,
                    file_size_mb=f"{file_size / 1024 / 1024:.2f}")

        csv_start = time.time()
        training_df = pd.read_csv(csv_path)
        csv_load_time = time.time() - csv_start
        
        logger.debug("CSV loaded successfully",
                    columns=list(training_df.columns),
                    rows=len(training_df),
                    load_time_seconds=f"{csv_load_time:.3f}")
        assert not training_df.empty, "Training CSV cannot be empty"

        parse_start = time.time()
        training_rows = []
        misconceptions_found = 0
        categories_found = set()
        
        for idx, row in training_df.iterrows():
            misconception = (
                row["Misconception"] if pd.notna(row["Misconception"]) else None
            )
            if misconception is not None:
                misconceptions_found += 1
                
            category = Category(row["Category"])
            categories_found.add(category)

            training_rows.append(
                TrainingRow(
                    row_id=int(row["row_id"]),
                    question_id=int(row["QuestionId"]),
                    question_text=str(row["QuestionText"]),
                    mc_answer=str(row["MC_Answer"]),
                    student_explanation=str(row["StudentExplanation"]),
                    category=category,
                    misconception=misconception,
                )
            )
            
            if idx > 0 and idx % 1000 == 0:
                logger.debug("Row parsing progress", processed_rows=idx + 1)

        parse_duration = time.time() - parse_start
        unique_questions = len({row.question_id for row in training_rows})
        
        logger.debug("Training data parsing completed",
                    total_rows=len(training_rows),
                    unique_questions=unique_questions,
                    rows_with_misconceptions=misconceptions_found,
                    unique_categories=len(categories_found),
                    categories=sorted([cat.value for cat in categories_found]),
                    parse_time_seconds=f"{parse_duration:.3f}")
        assert training_rows, "Must parse at least one training row"
        return training_rows

    @staticmethod
    def _extract_correct_answers(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, Answer]:
        """Extract the correct answer for each question."""
        logger.debug("Starting correct answer extraction", total_rows=len(training_data))
        assert training_data, "Training data cannot be empty"

        extract_start = time.time()
        correct_answers = {}
        true_correct_count = 0
        conflicts_checked = 0
        
        for row in training_data:
            if row.category == Category.TRUE_CORRECT:
                true_correct_count += 1
                if row.question_id in correct_answers:
                    conflicts_checked += 1
                    assert correct_answers[row.question_id] == row.mc_answer, (
                        f"Conflicting correct answers for question {row.question_id}"
                    )
                else:
                    correct_answers[row.question_id] = row.mc_answer

        extract_duration = time.time() - extract_start
        logger.debug("Correct answer extraction completed",
                    questions_with_correct_answers=len(correct_answers),
                    true_correct_rows=true_correct_count,
                    conflict_checks=conflicts_checked,
                    extraction_time_seconds=f"{extract_duration:.3f}")
        assert correct_answers, "Must find at least one correct answer"
        return correct_answers

    @staticmethod
    def _extract_question_categories(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
    ) -> dict[QuestionId, dict[bool, list[Category]]]:
        """Extract category patterns per question by correctness state.

        This mirrors the baseline's category frequency approach but stores
        the raw category lists for neural network training.
        """
        logger.debug("Starting question category extraction",
                    total_rows=len(training_data),
                    questions_with_correct_answers=len(correct_answers))
        assert training_data, "Training data cannot be empty"
        assert correct_answers, "Correct answers cannot be empty"

        # Group categories by question and correctness
        extract_start = time.time()
        question_correctness_categories = defaultdict(lambda: defaultdict(list))
        correct_matches = 0
        incorrect_matches = 0

        for row in training_data:
            is_correct = (
                row.question_id in correct_answers
                and row.mc_answer == correct_answers[row.question_id]
            )
            if is_correct:
                correct_matches += 1
            else:
                incorrect_matches += 1
                
            question_correctness_categories[row.question_id][is_correct].append(
                row.category
            )

        # Convert to final format (keep raw lists, don't count frequencies yet)
        result = {}
        questions_with_correct_categories = 0
        questions_with_incorrect_categories = 0
        
        for question_id, correctness_map in question_correctness_categories.items():
            result[question_id] = {}
            for is_correct, categories in correctness_map.items():
                result[question_id][is_correct] = categories
                if is_correct and categories:
                    questions_with_correct_categories += 1
                elif not is_correct and categories:
                    questions_with_incorrect_categories += 1

        extract_duration = time.time() - extract_start
        logger.debug("Question category extraction completed",
                    questions_processed=len(result),
                    correct_answer_matches=correct_matches,
                    incorrect_answer_matches=incorrect_matches,
                    questions_with_correct_categories=questions_with_correct_categories,
                    questions_with_incorrect_categories=questions_with_incorrect_categories,
                    extraction_time_seconds=f"{extract_duration:.3f}")
        return result

    @staticmethod
    def _extract_question_misconceptions(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, list[str]]:
        """Extract unique misconceptions per question, adding NA class."""
        logger.debug("Starting misconception extraction", total_rows=len(training_data))
        assert training_data, "Training data cannot be empty"

        extract_start = time.time()
        question_misconceptions_set = defaultdict(set)
        total_misconceptions_found = 0

        for row in training_data:
            if row.misconception is not None:
                question_misconceptions_set[row.question_id].add(row.misconception)
                total_misconceptions_found += 1

        # Convert to lists and add NA class
        question_misconceptions = {}
        total_unique_misconceptions = 0
        
        for question_id, misconceptions in question_misconceptions_set.items():
            misconception_list = sorted(misconceptions)  # Sort for consistency
            total_unique_misconceptions += len(misconception_list)
            misconception_list.append("NA")  # Add NA class
            question_misconceptions[question_id] = misconception_list

        extract_duration = time.time() - extract_start
        avg_misconceptions_per_question = (total_unique_misconceptions / len(question_misconceptions)
                                         if question_misconceptions else 0)
        
        logger.debug("Misconception extraction completed",
                    questions_with_misconceptions=len(question_misconceptions),
                    total_misconception_instances=total_misconceptions_found,
                    total_unique_misconceptions=total_unique_misconceptions,
                    avg_misconceptions_per_question=f"{avg_misconceptions_per_question:.2f}",
                    extraction_time_seconds=f"{extract_duration:.3f}")
        return question_misconceptions

    @staticmethod
    def _load_precomputed_embeddings_multihead(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
        embeddings_path: Path,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[QuestionId, dict[str, np.ndarray]],
        dict[QuestionId, np.ndarray],
        np.ndarray,
    ]:
        """Load pre-computed embeddings and prepare for multi-head training."""
        # For now, fall back to generating embeddings from scratch for multi-head
        # This could be optimized later to handle pre-computed embeddings
        logger.warning(
            "Pre-computed embeddings not yet supported for multi-head model, generating from scratch"
        )

        embedding_model = EmbeddingModel.MINI_LM
        return MLPStrategy._prepare_training_data_multihead(
            training_data,
            correct_answers,
            question_categories,
            question_misconceptions,
            embedding_model,
        )

    @staticmethod
    def _prepare_training_data_multihead(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
        embedding_model: EmbeddingModel,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[QuestionId, dict[str, np.ndarray]],
        dict[QuestionId, np.ndarray],
        np.ndarray,
    ]:
        """Prepare embeddings and multi-head labels for training."""
        logger.info("Starting multi-head training data preparation",
                   training_samples=len(training_data),
                   embedding_model=embedding_model.model_id,
                   questions_with_categories=len(question_categories),
                   questions_with_misconceptions=len(question_misconceptions))

        tokenizer_start = time.time()
        tokenizer = get_tokenizer(embedding_model)
        tokenizer_load_time = time.time() - tokenizer_start
        
        logger.debug("Tokenizer loaded",
                    model_id=embedding_model.model_id,
                    load_time_seconds=f"{tokenizer_load_time:.3f}")
        
        processing_start = time.time()
        embeddings, correctness, question_ids, category_labels, misconception_labels = (
            MLPStrategy._process_training_rows_multihead(
                training_data,
                correct_answers,
                question_categories,
                question_misconceptions,
                tokenizer,
            )
        )
        processing_time = time.time() - processing_start
        
        logger.debug("Training row processing completed",
                    embeddings_count=len(embeddings),
                    processing_time_seconds=f"{processing_time:.3f}")

        conversion_start = time.time()
        result = MLPStrategy._convert_to_arrays_multihead(
            embeddings,
            correctness,
            question_ids,
            category_labels,
            misconception_labels,
            question_categories,
            question_misconceptions,
        )
        conversion_time = time.time() - conversion_start
        
        logger.debug("Array conversion completed",
                    conversion_time_seconds=f"{conversion_time:.3f}")
        
        return result

    @staticmethod
    def _process_training_rows_multihead(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
        tokenizer: "SentenceTransformer",
    ) -> tuple[list, list, list, dict, dict]:
        """Process training rows for multi-head training."""
        logger.debug("Starting training row processing for multi-head model",
                    total_rows=len(training_data),
                    tokenizer_type=type(tokenizer).__name__)
        
        process_start = time.time()
        embeddings = []
        correctness = []
        question_ids = []
        category_labels = {
            qid: {"correct": [], "incorrect": []} for qid in question_categories
        }
        misconception_labels = {qid: [] for qid in question_misconceptions}
        
        skipped_rows = 0
        processed_embeddings = 0

        for idx, row in enumerate(training_data):
            if row.question_id not in question_categories:
                skipped_rows += 1
                continue

            # Generate embedding (question/answer/explanation only)
            text = repr(row)
            embedding_start = time.time()
            embedding = tokenizer.encode(text)
            embedding_time = time.time() - embedding_start
            
            embeddings.append(embedding)
            processed_embeddings += 1

            # Determine correctness
            is_correct = (
                row.question_id in correct_answers
                and row.mc_answer == correct_answers[row.question_id]
            )
            correctness.append(float(is_correct))
            question_ids.append(row.question_id)
            
            # Log progress for large datasets
            if idx > 0 and idx % 500 == 0:
                logger.debug("Row processing progress",
                           processed_rows=idx + 1,
                           embeddings_generated=processed_embeddings,
                           avg_embedding_time_ms=f"{embedding_time * 1000:.2f}")

            # Create category labels (one-hot for the actual category within correctness state)
            category_label_correct, category_label_incorrect = (
                MLPStrategy._create_category_labels(
                    row, question_categories, is_correct
                )
            )
            category_labels[row.question_id]["correct"].append(category_label_correct)
            category_labels[row.question_id]["incorrect"].append(
                category_label_incorrect
            )

            # Create misconception label (when applicable)
            if row.question_id in question_misconceptions:
                misconception_label = MLPStrategy._create_misconception_label(
                    row, question_misconceptions
                )
                misconception_labels[row.question_id].append(misconception_label)
            else:
                # No misconceptions for this question
                misconception_labels[row.question_id] = misconception_labels.get(
                    row.question_id, []
                )

        process_duration = time.time() - process_start
        avg_embedding_time = process_duration / max(processed_embeddings, 1)
        
        logger.debug("Training row processing completed",
                    total_processed=processed_embeddings,
                    skipped_rows=skipped_rows,
                    total_time_seconds=f"{process_duration:.3f}",
                    avg_embedding_time_ms=f"{avg_embedding_time * 1000:.2f}",
                    embedding_dimension=len(embeddings[0]) if embeddings else 0)
        
        return (
            embeddings,
            correctness,
            question_ids,
            category_labels,
            misconception_labels,
        )

    @staticmethod
    def _create_category_labels(
        row: TrainingRow,
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        is_correct: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create one-hot category labels for correct and incorrect states.

        Returns:
            (correct_label, incorrect_label): One-hot vectors for each correctness state
        """
        qid = row.question_id

        # Get unique categories for each correctness state with consistent ordering
        # CRITICAL: Must use same sorting as model head creation to ensure dimension alignment
        correct_categories = sorted(set(question_categories[qid].get(True, [])), key=str)
        incorrect_categories = sorted(set(question_categories[qid].get(False, [])), key=str)

        # Add assertions to catch dimension mismatches early
        assert correct_categories or incorrect_categories, (
            f"Question {qid} has no categories for either correct or incorrect states"
        )

        # Create one-hot labels
        correct_label = np.zeros(max(1, len(correct_categories)))
        incorrect_label = np.zeros(max(1, len(incorrect_categories)))

        # Set the appropriate label based on actual category and correctness
        if is_correct and correct_categories and row.category in correct_categories:
            category_index = correct_categories.index(row.category)
            assert 0 <= category_index < len(correct_label), (
                f"Category index {category_index} out of range for correct_label size {len(correct_label)}"
            )
            correct_label[category_index] = 1.0
        elif not is_correct and incorrect_categories and row.category in incorrect_categories:
            category_index = incorrect_categories.index(row.category)
            assert 0 <= category_index < len(incorrect_label), (
                f"Category index {category_index} out of range for incorrect_label size {len(incorrect_label)}"
            )
            incorrect_label[category_index] = 1.0

        # Debug logging for dimension verification
        logger.debug("Category labels created",
                    question_id=qid,
                    is_correct=is_correct,
                    row_category=str(row.category),
                    correct_categories_count=len(correct_categories),
                    incorrect_categories_count=len(incorrect_categories),
                    correct_label_shape=correct_label.shape,
                    incorrect_label_shape=incorrect_label.shape)

        return correct_label, incorrect_label

    @staticmethod
    def _create_misconception_label(
        row: TrainingRow, question_misconceptions: dict[QuestionId, list[str]]
    ) -> np.ndarray:
        """Create multi-hot encoding for misconception label."""
        misconceptions = question_misconceptions[row.question_id]
        label = np.zeros(len(misconceptions))

        if row.misconception is not None and row.misconception in misconceptions:
            label[misconceptions.index(row.misconception)] = 1.0
        else:
            label[-1] = 1.0  # NA is always last

        return label

    @staticmethod
    def _convert_to_arrays_multihead(
        embeddings: list,
        correctness: list,
        question_ids: list,
        category_labels: dict,
        misconception_labels: dict,
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[QuestionId, dict[str, np.ndarray]],
        dict[QuestionId, np.ndarray],
        np.ndarray,
    ]:
        """Convert lists to numpy arrays for multi-head training."""
        embeddings_array = np.stack(embeddings)
        correctness_array = np.array(correctness)
        question_ids_array = np.array(question_ids)

        # Convert category labels to arrays per question with consistent shapes
        # Find max category sizes for padding - use same logic as model head creation
        max_correct_categories = max(
            (
                len(sorted(set(category_map.get(True, [])), key=str))
                for category_map in question_categories.values()
            ),
            default=1,
        )
        max_incorrect_categories = max(
            (
                len(sorted(set(category_map.get(False, [])), key=str))
                for category_map in question_categories.values()
            ),
            default=1,
        )
        
        logger.debug("Array conversion dimensions calculated",
                    max_correct_categories=max_correct_categories,
                    max_incorrect_categories=max_incorrect_categories,
                    total_questions=len(question_categories))

        for qid, labels in category_labels.items():
            for correctness_state, max_size in [
                ("correct", max_correct_categories),
                ("incorrect", max_incorrect_categories),
            ]:
                if labels[correctness_state]:
                    # Pad all label vectors to max_size with dimension validation
                    padded_labels = []
                    for idx, label in enumerate(labels[correctness_state]):
                        # Add assertion to catch dimension mismatches early
                        assert len(label) <= max_size, (
                            f"Question {qid}, {correctness_state} state, sample {idx}: "
                            f"Label size {len(label)} exceeds max_size {max_size}. "
                            f"This indicates inconsistent category counting between model head creation and label creation."
                        )
                        
                        if len(label) < max_size:
                            padded = np.zeros(max_size)
                            padded[: len(label)] = label
                            padded_labels.append(padded)
                        else:
                            padded_labels.append(label)  # Already correct size
                            
                    assert padded_labels, f"No padded labels created for question {qid}, {correctness_state} state"
                    category_labels[qid][correctness_state] = np.stack(padded_labels)
                    
                    logger.debug("Category labels padded",
                                question_id=qid,
                                correctness_state=correctness_state,
                                original_samples=len(labels[correctness_state]),
                                padded_shape=labels[correctness_state].shape,
                                target_max_size=max_size)
                else:
                    # No labels for this state - create empty array with correct shape
                    category_labels[qid][correctness_state] = np.empty((0, max_size))
                    logger.debug("Empty category labels created",
                                question_id=qid,
                                correctness_state=correctness_state,
                                empty_shape=(0, max_size))

        # Convert misconception labels to arrays per question with padding
        max_misconception_size = max(
            (
                len(misconceptions)
                for misconceptions in question_misconceptions.values()
            ),
            default=1,
        )

        for qid, labels in misconception_labels.items():
            if labels:
                # Pad all labels to max_misconception_size
                padded_labels = []
                for label in labels:
                    padded_label = np.zeros(max_misconception_size)
                    padded_label[: len(label)] = label
                    padded_labels.append(padded_label)
                misconception_labels[qid] = np.stack(padded_labels)
            else:
                misconception_labels[qid] = np.empty((0, max_misconception_size))

        logger.info(
            f"Generated {len(embeddings_array)} embeddings for multi-head training"
        )
        return (
            embeddings_array,
            correctness_array,
            category_labels,
            misconception_labels,
            question_ids_array,
        )

    @staticmethod
    def _train_multihead_model(
        embeddings: np.ndarray,
        correctness: np.ndarray,
        category_labels: dict[QuestionId, dict[str, np.ndarray]],
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
        device: torch.device,
    ) -> MLPNet:
        """Train the multi-head MLP model."""
        logger.info("Starting multi-head MLP model training",
                   embeddings_shape=embeddings.shape,
                   correctness_shape=correctness.shape,
                   question_ids_shape=question_ids.shape,
                   device=str(device),
                   unique_questions=len(set(question_ids)))

        setup_start = time.time()
        model, criterions, optimizer = MLPStrategy._setup_multihead_training(
            question_categories, question_misconceptions, device
        )
        setup_time = time.time() - setup_start
        
        logger.debug("Model setup completed",
                    model_parameters=sum(p.numel() for p in model.parameters()),
                    criterion_types=[type(c).__name__ for c in criterions.values()],
                    optimizer_type=type(optimizer).__name__,
                    setup_time_seconds=f"{setup_time:.3f}")
        
        dataset_start = time.time()
        dataset = MLPDataset(
            embeddings,
            correctness,
            category_labels,
            misconception_labels,
            question_ids,
            device,
        )
        dataset_time = time.time() - dataset_start
        
        logger.debug("Dataset creation completed",
                    dataset_size=len(dataset),
                    dataset_time_seconds=f"{dataset_time:.3f}")

        # Custom collate function to handle variable tensor shapes
        def collate_multihead_batch(batch):
            features = torch.stack([item[0] for item in batch])
            question_ids = torch.tensor([item[2] for item in batch])
            indices = torch.tensor([item[3] for item in batch])

            # Handle multi-labels with different shapes per sample
            multi_labels = {}

            # Correctness labels (consistent shape)
            multi_labels["correctness"] = torch.stack(
                [item[1]["correctness"] for item in batch]
            )

            # Misconception labels (pad to max size in batch)
            misc_labels = [item[1]["misconceptions"] for item in batch]
            if misc_labels:
                max_misc_size = max(label.size(0) for label in misc_labels)
                padded_misc = []
                for label in misc_labels:
                    if label.size(0) < max_misc_size:
                        padded = torch.zeros(max_misc_size, device=label.device)
                        padded[: label.size(0)] = label
                        padded_misc.append(padded)
                    else:
                        padded_misc.append(label)
                multi_labels["misconceptions"] = torch.stack(padded_misc)

            # Category labels (pad to max size in batch for each type)
            correct_cat_labels = [
                item[1].get("correct_categories")
                for item in batch
                if "correct_categories" in item[1]
            ]
            if correct_cat_labels:
                max_correct_size = max(label.size(0) for label in correct_cat_labels)
                padded_correct = []
                for item in batch:
                    if "correct_categories" in item[1]:
                        label = item[1]["correct_categories"]
                        if label.size(0) < max_correct_size:
                            padded = torch.zeros(max_correct_size, device=label.device)
                            padded[: label.size(0)] = label
                            padded_correct.append(padded)
                        else:
                            padded_correct.append(label)
                    else:
                        padded_correct.append(
                            torch.zeros(max_correct_size, device=features.device)
                        )
                multi_labels["correct_categories"] = torch.stack(padded_correct)

            incorrect_cat_labels = [
                item[1].get("incorrect_categories")
                for item in batch
                if "incorrect_categories" in item[1]
            ]
            if incorrect_cat_labels:
                max_incorrect_size = max(
                    label.size(0) for label in incorrect_cat_labels
                )
                padded_incorrect = []
                for item in batch:
                    if "incorrect_categories" in item[1]:
                        label = item[1]["incorrect_categories"]
                        if label.size(0) < max_incorrect_size:
                            padded = torch.zeros(
                                max_incorrect_size, device=label.device
                            )
                            padded[: label.size(0)] = label
                            padded_incorrect.append(padded)
                        else:
                            padded_incorrect.append(label)
                    else:
                        padded_incorrect.append(
                            torch.zeros(max_incorrect_size, device=features.device)
                        )
                multi_labels["incorrect_categories"] = torch.stack(padded_incorrect)

            return features, multi_labels, question_ids, indices

        batch_size = 32
        num_epochs = 100
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multihead_batch
        )
        
        logger.info("Starting training epochs",
                   batch_size=batch_size,
                   num_epochs=num_epochs,
                   total_batches_per_epoch=len(dataloader))
        
        training_start = time.time()
        MLPStrategy._train_multihead_epochs(
            model,
            criterions,
            optimizer,
            dataloader,
            num_epochs=num_epochs,
        )
        training_duration = time.time() - training_start

        logger.info("Multi-head MLP training completed",
                   training_time_seconds=f"{training_duration:.3f}",
                   epochs_completed=num_epochs,
                   final_model_parameters=sum(p.numel() for p in model.parameters()))
        return model

    @staticmethod
    def _setup_multihead_training(
        question_categories: dict[QuestionId, dict[bool, list[Category]]],
        question_misconceptions: dict[QuestionId, list[str]],
        device: torch.device,
    ) -> tuple[MLPNet, dict[str, nn.Module], optim.Adam]:
        """Setup multi-head model, criterions, and optimizer."""
        logger.debug("Setting up multi-head training components",
                    questions_with_categories=len(question_categories),
                    questions_with_misconceptions=len(question_misconceptions),
                    target_device=str(device))
        
        model_create_start = time.time()
        model = MLPNet(question_categories, question_misconceptions).to(device)
        model_create_time = time.time() - model_create_start
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        logger.debug("Model created and moved to device",
                    model_device=str(model_device),
                    device_matches=model_device.type == device.type,
                    model_parameters=sum(p.numel() for p in model.parameters()),
                    creation_time_seconds=f"{model_create_time:.3f}")

        # Multiple loss functions for different heads
        criterions = {
            "correctness": nn.BCEWithLogitsLoss(),  # Binary classification for correctness
            "categories": nn.CrossEntropyLoss(),  # Multi-class for category selection
            "misconceptions": nn.BCEWithLogitsLoss(),  # Multi-label for misconceptions
        }
        
        # Move criterions to device
        for name, criterion in criterions.items():
            criterions[name] = criterion.to(device)

        # Lower learning rate for more complex multi-head model
        learning_rate = 5e-4
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        logger.debug("Training setup completed",
                    criterions_count=len(criterions),
                    learning_rate=learning_rate,
                    optimizer_params=len(list(model.parameters())),
                    device=str(device))
        return model, criterions, optimizer

    @staticmethod
    def _train_multihead_epochs(
        model: MLPNet,
        criterions: dict[str, nn.Module],
        optimizer: optim.Adam,
        dataloader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Train multi-head model for specified number of epochs."""
        model.train()
        for epoch in range(num_epochs):
            total_losses = MLPStrategy._train_multihead_single_epoch(
                model, criterions, optimizer, dataloader
            )

            if epoch % 20 == 0:
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in total_losses.items()])
                logger.debug(f"Epoch {epoch}, Losses: {loss_str}")

    @staticmethod
    def _train_multihead_single_epoch(
        model: MLPNet,
        criterions: dict[str, nn.Module],
        optimizer: optim.Adam,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train multi-head model for a single epoch."""
        total_losses = {
            "correctness": 0.0,
            "categories": 0.0,
            "misconceptions": 0.0,
            "combined": 0.0,
        }
        num_batches = 0

        for features, multi_labels, question_id_batch, _indices in dataloader:
            optimizer.zero_grad()
            batch_losses = MLPStrategy._process_multihead_batch(
                model, criterions, features, multi_labels, question_id_batch
            )

            if any(loss > 0 for loss in batch_losses.values()):
                # Combined loss with weights
                combined_loss = (
                    batch_losses["correctness"] * 2.0  # Correctness is most important
                    + batch_losses["categories"] * 1.5  # Categories are important
                    + batch_losses["misconceptions"] * 1.0  # Misconceptions are helpful
                )

                combined_loss.backward()
                optimizer.step()

                # Track losses
                for key, loss in batch_losses.items():
                    total_losses[key] += loss.item()
                total_losses["combined"] += combined_loss.item()
                num_batches += 1

        # Return average losses
        if num_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_batches

        return total_losses

    @staticmethod
    def _process_multihead_batch(
        model: MLPNet,
        criterions: dict[str, nn.Module],
        features: torch.Tensor,
        multi_labels: dict[str, torch.Tensor],
        question_id_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process a single batch for multi-head training."""
        device = features.device
        model_device = next(model.parameters()).device
        assert model_device.type == device.type, (
            f"Model device type mismatch: expected {device.type}, got {model_device.type}"
        )

        # Initialize losses
        batch_losses = {
            "correctness": torch.tensor(0.0, device=device),
            "categories": torch.tensor(0.0, device=device),
            "misconceptions": torch.tensor(0.0, device=device),
        }

        valid_samples = 0

        for i in range(len(features)):
            qid = question_id_batch[i].item()
            feature = features[i].unsqueeze(0)  # Shape: [1, 384]

            # Get multi-head model outputs
            outputs = model(feature, qid)

            # Correctness loss (always present)
            if "correctness" in outputs and "correctness" in multi_labels:
                correctness_target = multi_labels["correctness"][i].unsqueeze(0)
                correctness_loss = criterions["correctness"](
                    outputs["correctness"], correctness_target
                )
                batch_losses["correctness"] += correctness_loss

            # Category losses (question-specific)
            # Determine which category head to use based on ground truth correctness
            is_correct = multi_labels["correctness"][i].item() > CORRECTNESS_THRESHOLD
            category_key = (
                "correct_categories" if is_correct else "incorrect_categories"
            )
            label_key = "correct_categories" if is_correct else "incorrect_categories"

            if category_key in outputs and label_key in multi_labels:
                category_target = multi_labels[label_key][i]
                if (
                    category_target.sum() > 0
                ):  # Only compute loss if we have a valid target
                    # Add assertion to catch dimension mismatches early
                    model_output = outputs[category_key]
                    expected_classes = model_output.size(-1)  # Number of classes from model head
                    target_size = category_target.size(-1)    # Size of target tensor
                    
                    assert target_size == expected_classes, (
                        f"Dimension mismatch for question {qid}, category_key={category_key}: "
                        f"Model output size: {model_output.size()} (expects {expected_classes} classes), "
                        f"Target tensor size: {category_target.size()} ({target_size} classes). "
                        f"This indicates inconsistent category counting between model head creation and label preparation. "
                        f"Model head was created with {expected_classes} unique categories, "
                        f"but label tensor has {target_size} categories."
                    )
                    
                    # Convert one-hot to class indices for CrossEntropyLoss
                    target_class = torch.argmax(category_target).unsqueeze(0)
                    category_loss = criterions["categories"](
                        outputs[category_key], target_class
                    )
                    batch_losses["categories"] += category_loss

            # Misconception loss (when applicable)
            if "misconceptions" in outputs and "misconceptions" in multi_labels:
                misconception_target = multi_labels["misconceptions"][i].unsqueeze(0)
                if (
                    misconception_target.sum() > 0
                ):  # Only if there are actual misconceptions
                    misconception_loss = criterions["misconceptions"](
                        outputs["misconceptions"], misconception_target
                    )
                    batch_losses["misconceptions"] += misconception_loss

            valid_samples += 1

        # Average over batch
        if valid_samples > 0:
            for key in batch_losses:
                batch_losses[key] = batch_losses[key] / valid_samples

        return batch_losses
