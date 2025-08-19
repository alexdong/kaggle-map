"""MLP neural network strategy for student misconception prediction."""

import pickle
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger
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

# Constants
MISCONCEPTION_THRESHOLD = 0.5
CORRECTNESS_THRESHOLD = 0.5


class MLPDataset(Dataset):
    """PyTorch dataset for MLP training."""

    def __init__(
        self,
        embeddings: np.ndarray,
        correctness: np.ndarray,
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
    ) -> None:
        self.embeddings = torch.FloatTensor(embeddings)
        self.correctness = torch.FloatTensor(correctness).unsqueeze(1)
        self.question_ids = question_ids

        # Flatten misconception labels to match global indexing
        self.labels = []
        self._build_label_mapping(misconception_labels, question_ids)

    def _build_label_mapping(
        self,
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
    ) -> None:
        """Build mapping from global index to misconception labels."""
        # Find the maximum label size across all questions for padding
        max_label_size = max(
            labels.shape[1] if len(labels) > 0 else 2
            for labels in misconception_labels.values()
        )

        # Create a mapping from question_id to local indices for that question
        question_local_indices = {}
        for qid in set(question_ids):
            question_local_indices[qid] = 0

        # Build labels list in the same order as the global dataset
        for _global_idx, qid in enumerate(question_ids):
            local_idx = question_local_indices[qid]
            if local_idx < len(misconception_labels[qid]):
                label = misconception_labels[qid][local_idx]
                # Pad to max_label_size
                padded_label = np.zeros(max_label_size)
                padded_label[: len(label)] = label
                self.labels.append(torch.FloatTensor(padded_label))
            else:
                # Fallback: create zero label with max_label_size
                self.labels.append(torch.zeros(max_label_size))
            question_local_indices[qid] += 1

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        question_id = self.question_ids[idx]
        features = torch.cat([self.embeddings[idx], self.correctness[idx]])
        misconception_label = self.labels[idx]
        return features, misconception_label, question_id, idx


class MLPNet(nn.Module):
    """Multi-layer perceptron with question-specific misconception heads."""

    def __init__(self, question_misconceptions: dict[QuestionId, list[str]]) -> None:
        super().__init__()

        # Shared trunk: embedding(384) + correctness(1) = 385 -> 128
        self.shared_trunk = nn.Sequential(
            nn.Linear(385, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        # Find max output size for consistent tensor shapes
        self.max_output_size = max(
            len(misconceptions) for misconceptions in question_misconceptions.values()
        )

        # Question-specific misconception heads - all output max_output_size
        self.question_heads = nn.ModuleDict()
        for question_id in question_misconceptions:
            self.question_heads[str(question_id)] = nn.Linear(128, self.max_output_size)

        self.question_misconceptions = question_misconceptions

    def forward(self, x: torch.Tensor, question_id: int) -> torch.Tensor:
        shared_features = self.shared_trunk(x)
        return self.question_heads[str(question_id)](shared_features)


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
        logger.info(f"Fitting MLP strategy from {train_csv_path}")

        # Set random seeds for deterministic training
        cls._set_random_seeds(random_seed)
        logger.debug(f"Set random seed to {random_seed} for deterministic training")

        # Parse training data
        all_training_data = cls._parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(all_training_data)} total training rows")

        # Split data if train_split < 1.0
        if train_split < 1.0:
            training_data, eval_data = train_test_split(
                all_training_data,
                train_size=train_split,
                random_state=random_seed,
                stratify=[
                    row.question_id for row in all_training_data
                ],  # Stratify by question
            )
            logger.debug(
                f"Split data: {len(training_data)} training, {len(eval_data)} eval"
            )
            # Store eval data for later use
            cls._eval_data = eval_data
        else:
            training_data = all_training_data
            cls._eval_data = None

        # Extract metadata
        correct_answers = cls._extract_correct_answers(training_data)
        question_misconceptions = cls._extract_question_misconceptions(training_data)

        logger.debug(f"Found {len(correct_answers)} questions")
        logger.debug(
            f"Found misconceptions for {len(question_misconceptions)} questions"
        )

        # Generate or load embeddings and labels
        embedding_model = EmbeddingModel.MINI_LM
        if embeddings_path is not None and embeddings_path.exists():
            logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
            embeddings, correctness, misconception_labels, question_ids = (
                cls._load_precomputed_embeddings(
                    training_data,
                    correct_answers,
                    question_misconceptions,
                    embeddings_path,
                )
            )
        else:
            logger.info("Generating embeddings from training data...")
            embeddings, correctness, misconception_labels, question_ids = (
                cls._prepare_training_data(
                    training_data,
                    correct_answers,
                    question_misconceptions,
                    embedding_model,
                )
            )

        # Train model
        model = cls._train_model(
            embeddings,
            correctness,
            misconception_labels,
            question_ids,
            question_misconceptions,
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
        logger.info(f"Making MLP predictions for {len(test_data)} test rows")

        if not test_data:
            return []

        # Generate embeddings for test data
        tokenizer = get_tokenizer(self.embedding_model)
        test_embeddings = []
        test_correctness = []

        for row in test_data:
            # Generate embedding
            text = repr(row)  # Uses EvaluationRow.__repr__ format
            embedding = tokenizer.encode(text)
            test_embeddings.append(embedding)

            # Determine correctness
            is_correct = self._is_answer_correct(row.question_id, row.mc_answer)
            test_correctness.append(float(is_correct))

        embeddings = torch.FloatTensor(np.stack(test_embeddings))
        correctness = torch.FloatTensor(test_correctness).unsqueeze(1)

        # Make predictions
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for i, row in enumerate(test_data):
                features = torch.cat([embeddings[i], correctness[i]])

                # Get misconception probabilities
                if row.question_id not in self.question_misconceptions:
                    logger.warning(f"Question {row.question_id} not in training data")
                    # Default to no misconception
                    predicted_categories = self._create_default_prediction(row)
                else:
                    logits = self.model(features.unsqueeze(0), row.question_id)
                    probs = torch.sigmoid(logits).squeeze().numpy()

                    # Only use the relevant part of the output for this question
                    num_misconceptions = len(
                        self.question_misconceptions[row.question_id]
                    )
                    relevant_probs = probs[:num_misconceptions]

                    predicted_categories = self._reconstruct_predictions(
                        relevant_probs,
                        is_correct=correctness[i].item() > CORRECTNESS_THRESHOLD,
                        question_id=row.question_id,
                    )

                predictions.append(
                    SubmissionRow(
                        row_id=row.row_id,
                        predicted_categories=predicted_categories[
                            :3
                        ],  # Max 3 predictions
                    )
                )

        return predictions

    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        logger.info(f"Saving MLP model to {filepath}")

        # Save model state and metadata
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "correct_answers": self.correct_answers,
            "question_misconceptions": self.question_misconceptions,
            "embedding_model": self.embedding_model.model_id,
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

        # Reconstruct model
        question_misconceptions = save_data["question_misconceptions"]
        model = MLPNet(question_misconceptions)
        model.load_state_dict(save_data["model_state_dict"])

        # Find embedding model
        embedding_model = None
        for em in EmbeddingModel.all():
            if em.model_id == save_data["embedding_model"]:
                embedding_model = em
                break

        if embedding_model is None:
            raise ValueError(f"Unknown embedding model: {save_data['embedding_model']}")

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

    def _reconstruct_predictions(
        self,
        misconception_probs: np.ndarray,
        *,
        is_correct: bool,
        question_id: QuestionId,
    ) -> list[Prediction]:
        """Reconstruct category predictions from misconception probabilities."""
        prefix = "True_" if is_correct else "False_"
        misconceptions = self.question_misconceptions[question_id]
        predictions = []

        # Check for misconceptions (exclude NA which is last)
        for i, prob in enumerate(misconception_probs[:-1]):
            if prob > MISCONCEPTION_THRESHOLD:
                misconception = misconceptions[i]
                pred = Prediction(
                    category=Category(f"{prefix}Misconception"),
                    misconception=misconception,
                )
                predictions.append((pred, prob))

        # If no misconceptions detected, use NA probability for Neither
        if not predictions:
            na_prob = misconception_probs[-1]
            pred = Prediction(category=Category(f"{prefix}Neither"))
            predictions.append((pred, na_prob))

        # Sort by confidence and return top predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [pred for pred, _score in predictions]

    def _create_default_prediction(self, row: EvaluationRow) -> list[Prediction]:
        """Create default prediction for unknown questions."""
        is_correct = self._is_answer_correct(row.question_id, row.mc_answer)
        prefix = "True_" if is_correct else "False_"
        return [Prediction(category=Category(f"{prefix}Neither"))]

    @staticmethod
    def _parse_training_data(csv_path: Path) -> list[TrainingRow]:
        """Parse CSV into strongly-typed training rows."""
        assert csv_path.exists(), f"Training file not found: {csv_path}"

        training_df = pd.read_csv(csv_path)
        logger.debug(f"Loaded CSV with columns: {list(training_df.columns)}")
        assert not training_df.empty, "Training CSV cannot be empty"

        training_rows = []
        for _, row in training_df.iterrows():
            misconception = (
                row["Misconception"] if pd.notna(row["Misconception"]) else None
            )

            training_rows.append(
                TrainingRow(
                    row_id=int(row["row_id"]),
                    question_id=int(row["QuestionId"]),
                    question_text=str(row["QuestionText"]),
                    mc_answer=str(row["MC_Answer"]),
                    student_explanation=str(row["StudentExplanation"]),
                    category=Category(row["Category"]),
                    misconception=misconception,
                )
            )

        logger.debug(f"Parsed {len(training_rows)} training rows")
        assert training_rows, "Must parse at least one training row"
        return training_rows

    @staticmethod
    def _extract_correct_answers(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, Answer]:
        """Extract the correct answer for each question."""
        assert training_data, "Training data cannot be empty"

        correct_answers = {}
        for row in training_data:
            if row.category == Category.TRUE_CORRECT:
                if row.question_id in correct_answers:
                    assert correct_answers[row.question_id] == row.mc_answer, (
                        f"Conflicting correct answers for question {row.question_id}"
                    )
                else:
                    correct_answers[row.question_id] = row.mc_answer

        logger.debug(f"Extracted correct answers for {len(correct_answers)} questions")
        assert correct_answers, "Must find at least one correct answer"
        return correct_answers

    @staticmethod
    def _extract_question_misconceptions(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, list[str]]:
        """Extract unique misconceptions per question, adding NA class."""
        assert training_data, "Training data cannot be empty"

        question_misconceptions_set = defaultdict(set)

        for row in training_data:
            if row.misconception is not None:
                question_misconceptions_set[row.question_id].add(row.misconception)

        # Convert to lists and add NA class
        question_misconceptions = {}
        for question_id, misconceptions in question_misconceptions_set.items():
            misconception_list = sorted(misconceptions)  # Sort for consistency
            misconception_list.append("NA")  # Add NA class
            question_misconceptions[question_id] = misconception_list

        logger.debug(
            f"Extracted misconceptions for {len(question_misconceptions)} questions"
        )
        return question_misconceptions

    @staticmethod
    def _load_precomputed_embeddings(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_misconceptions: dict[QuestionId, list[str]],
        embeddings_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, dict[QuestionId, np.ndarray], np.ndarray]:
        """Load pre-computed embeddings from npz file and align with training data."""
        # Load the embeddings file
        embeddings_data = np.load(embeddings_path)
        stored_row_ids = embeddings_data["row_ids"]
        stored_embeddings = embeddings_data["embeddings"]

        logger.debug(f"Loaded {len(stored_row_ids)} embeddings from {embeddings_path}")

        # Create mapping from row_id to embedding
        row_id_to_embedding = {}
        for i, row_id in enumerate(stored_row_ids):
            row_id_to_embedding[row_id] = stored_embeddings[i]

        # Process training data and align with embeddings
        embeddings = []
        correctness = []
        question_ids = []
        misconception_labels = {qid: [] for qid in question_misconceptions}

        matched_count = 0
        for row in training_data:
            if row.question_id not in question_misconceptions:
                continue

            # Check if we have embedding for this row
            if row.row_id not in row_id_to_embedding:
                logger.warning(f"No embedding found for row_id {row.row_id}, skipping")
                continue

            matched_count += 1
            embeddings.append(row_id_to_embedding[row.row_id])

            # Determine correctness
            is_correct = (
                row.question_id in correct_answers
                and row.mc_answer == correct_answers[row.question_id]
            )
            correctness.append(float(is_correct))
            question_ids.append(row.question_id)

            # Create misconception label
            label = MLPStrategy._create_misconception_label(
                row, question_misconceptions
            )
            misconception_labels[row.question_id].append(label)

        logger.info(f"Matched {matched_count} rows with pre-computed embeddings")

        return MLPStrategy._convert_to_arrays(
            embeddings,
            correctness,
            question_ids,
            misconception_labels,
            question_misconceptions,
        )

    @staticmethod
    def _prepare_training_data(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_misconceptions: dict[QuestionId, list[str]],
        embedding_model: EmbeddingModel,
    ) -> tuple[np.ndarray, np.ndarray, dict[QuestionId, np.ndarray], np.ndarray]:
        """Prepare embeddings and labels for training."""
        logger.info("Generating embeddings for training data...")

        tokenizer = get_tokenizer(embedding_model)
        embeddings, correctness, question_ids, misconception_labels = (
            MLPStrategy._process_training_rows(
                training_data, correct_answers, question_misconceptions, tokenizer
            )
        )

        return MLPStrategy._convert_to_arrays(
            embeddings,
            correctness,
            question_ids,
            misconception_labels,
            question_misconceptions,
        )

    @staticmethod
    def _process_training_rows(
        training_data: list[TrainingRow],
        correct_answers: dict[QuestionId, Answer],
        question_misconceptions: dict[QuestionId, list[str]],
        tokenizer: "SentenceTransformer",
    ) -> tuple[list, list, list, dict]:
        """Process training rows to extract features and labels."""
        embeddings = []
        correctness = []
        question_ids = []
        misconception_labels = {qid: [] for qid in question_misconceptions}

        for row in training_data:
            if row.question_id not in question_misconceptions:
                continue

            # Generate embedding
            text = repr(row)
            embedding = tokenizer.encode(text)
            embeddings.append(embedding)

            # Determine correctness
            is_correct = (
                row.question_id in correct_answers
                and row.mc_answer == correct_answers[row.question_id]
            )
            correctness.append(float(is_correct))
            question_ids.append(row.question_id)

            # Create misconception label
            label = MLPStrategy._create_misconception_label(
                row, question_misconceptions
            )
            misconception_labels[row.question_id].append(label)

        return embeddings, correctness, question_ids, misconception_labels

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
    def _convert_to_arrays(
        embeddings: list,
        correctness: list,
        question_ids: list,
        misconception_labels: dict,
        question_misconceptions: dict[QuestionId, list[str]],
    ) -> tuple[np.ndarray, np.ndarray, dict[QuestionId, np.ndarray], np.ndarray]:
        """Convert lists to numpy arrays."""
        embeddings_array = np.stack(embeddings)
        correctness_array = np.array(correctness)
        question_ids_array = np.array(question_ids)

        # Find maximum label size across all questions for consistent tensor shapes
        max_label_size = max(
            len(misconceptions) for misconceptions in question_misconceptions.values()
        )

        # Convert misconception labels to arrays per question with padding
        for qid, labels in misconception_labels.items():
            if labels:
                # Pad all labels to max_label_size
                padded_labels = []
                for label in labels:
                    padded_label = np.zeros(max_label_size)
                    padded_label[: len(label)] = label
                    padded_labels.append(padded_label)
                misconception_labels[qid] = np.stack(padded_labels)
            else:
                misconception_labels[qid] = np.empty((0, max_label_size))

        logger.info(f"Generated {len(embeddings_array)} embeddings")
        return (
            embeddings_array,
            correctness_array,
            misconception_labels,
            question_ids_array,
        )

    @staticmethod
    def _train_model(
        embeddings: np.ndarray,
        correctness: np.ndarray,
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
        question_misconceptions: dict[QuestionId, list[str]],
    ) -> MLPNet:
        """Train the MLP model."""
        logger.info("Training MLP model...")

        model, criterion, optimizer = MLPStrategy._setup_training(
            question_misconceptions
        )
        dataset = MLPDataset(
            embeddings, correctness, misconception_labels, question_ids
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        MLPStrategy._train_epochs(
            model, criterion, optimizer, dataloader, num_epochs=50
        )

        logger.info("MLP training completed")
        return model

    @staticmethod
    def _setup_training(
        question_misconceptions: dict[QuestionId, list[str]],
    ) -> tuple[MLPNet, nn.BCEWithLogitsLoss, optim.Adam]:
        """Setup model, criterion, and optimizer for training."""
        model = MLPNet(question_misconceptions)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model, criterion, optimizer

    @staticmethod
    def _train_epochs(
        model: MLPNet,
        criterion: nn.BCEWithLogitsLoss,
        optimizer: optim.Adam,
        dataloader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Train model for specified number of epochs."""
        model.train()
        for epoch in range(num_epochs):
            total_loss = MLPStrategy._train_single_epoch(
                model, criterion, optimizer, dataloader
            )

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Average Loss: {total_loss:.4f}")

    @staticmethod
    def _train_single_epoch(
        model: MLPNet,
        criterion: nn.BCEWithLogitsLoss,
        optimizer: optim.Adam,
        dataloader: DataLoader,
    ) -> float:
        """Train model for a single epoch."""
        total_loss = 0.0
        num_batches = 0

        for features, labels, question_id_batch, _indices in dataloader:
            optimizer.zero_grad()
            batch_loss = MLPStrategy._process_batch(
                model, criterion, features, labels, question_id_batch
            )

            if batch_loss > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @staticmethod
    def _process_batch(
        model: MLPNet,
        criterion: nn.BCEWithLogitsLoss,
        features: torch.Tensor,
        labels: torch.Tensor,
        question_id_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single batch for training."""
        batch_loss = torch.tensor(0.0)
        for i in range(len(features)):
            qid = question_id_batch[i].item()
            feature = features[i].unsqueeze(0)
            label = labels[i].unsqueeze(0)

            output = model(feature, qid)
            loss = criterion(output, label)
            batch_loss += loss

        return batch_loss
