"""MLP predictor for misconception prediction."""

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.nn import functional
from torch.utils.data import DataLoader

from kaggle_map.core.dataset import extract_correct_answers, load_training_data
from kaggle_map.core.models import (
    Answer,
    Category,
    EmbeddingStrategy,
    EvaluationRow,
    Prediction,
    QuestionId,
    SubmissionRow,
    TrainingRow,
)
from kaggle_map.mlp.dataset import DatasetArrays, DatasetEncoders, MLPDataset
from kaggle_map.mlp.loss import ListMLELoss
from kaggle_map.mlp.model import QuestionSpecificMLP
from kaggle_map.mlp.trainer import TrainingConfig, TrainingSetup, train_model
from kaggle_map.utils.device import get_device
from kaggle_map.utils.metrics import calculate_map_at_3

MAX_PREDICTIONS = 3


def _extract_question_predictions(training_data: list[TrainingRow]) -> dict[QuestionId, list[str]]:
    """Extract unique prediction strings per question."""
    question_predictions = defaultdict(list)
    for row in training_data:
        pred_str = str(row.prediction)
        question_predictions[row.question_id].append(pred_str)

    return {qid: list(set(preds)) for qid, preds in question_predictions.items()}


def _get_split_indices(
    n_samples: int, train_ratio: float = 0.7, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get train/val/test split indices."""
    rng = np.random.Generator(np.random.PCG64(random_seed))
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * 0.15)

    return (
        indices[:train_size],
        indices[train_size : train_size + val_size],
        indices[train_size + val_size :],
    )


@dataclass(frozen=True)
class Predictor:
    """MLP predictor for misconception classification."""

    model: QuestionSpecificMLP
    correct_answers: dict[QuestionId, Answer]
    device: torch.device
    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.DOUBLE_BLIND

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def description(self) -> str:
        return "MLP with question-specific heads for misconception prediction"

    @classmethod
    def fit(
        cls,
        config: TrainingConfig | None = None,
        embedding_strategy: str | None = None,
    ) -> "Predictor":
        """Train a new predictor.

        Args:
            config: Training configuration
            embedding_strategy: "double_blind" or "semantic"

        Returns:
            Trained Predictor instance
        """
        if config is None:
            config = TrainingConfig()

        strategy = EmbeddingStrategy.from_string(embedding_strategy)

        device = get_device()
        logger.info(f"Training on {device} with embedding strategy: {strategy.value}")

        training_data = load_training_data(config.train_csv_path)
        correct_answers = extract_correct_answers(training_data)
        question_predictions = _extract_question_predictions(training_data)

        logger.info("Computing embeddings...")
        embeddings_list: list[np.ndarray] = []
        question_ids_list: list[int] = []
        predictions_list: list[str] = []
        mc_answers_list: list[str] = []

        for row in training_data:
            eval_row = EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
                correct_answer=correct_answers.get(row.question_id, ""),
            )
            embedding = strategy.fn(eval_row)
            embeddings_list.append(embedding)
            question_ids_list.append(row.question_id)
            predictions_list.append(str(row.prediction))
            mc_answers_list.append(row.mc_answer)

        embeddings = np.array(embeddings_list)
        question_ids = np.array(question_ids_list)
        predictions = np.array(predictions_list)
        mc_answers = np.array(mc_answers_list)

        embedding_dim = embeddings.shape[1]
        logger.info(f"Embedding dimension: {embedding_dim}")

        model = QuestionSpecificMLP(
            question_predictions,
            embedding_dim=embedding_dim,
            architecture_size=config.architecture_size,
            dropout=config.dropout,
            activation=config.activation,
        )
        model = model.to(device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        n_samples = len(embeddings)
        train_idx, val_idx, _ = _get_split_indices(n_samples, config.train_split, config.random_seed)
        logger.info(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_arrays = DatasetArrays(
            embeddings=embeddings[train_idx],
            question_ids=question_ids[train_idx],
            predictions=predictions[train_idx],
            mc_answers=mc_answers[train_idx],
        )

        val_arrays = DatasetArrays(
            embeddings=embeddings[val_idx],
            question_ids=question_ids[val_idx],
            predictions=predictions[val_idx],
            mc_answers=mc_answers[val_idx],
        )

        encoders = DatasetEncoders(
            correct_answers=correct_answers,
            true_label_encoders=model.true_label_encoders,
            false_label_encoders=model.false_label_encoders,
        )

        train_dataset = MLPDataset(train_arrays, encoders)
        val_dataset = MLPDataset(val_arrays, encoders)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)

        setup = TrainingSetup(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            criterion=ListMLELoss(),
        )
        result = train_model(setup)

        logger.info(f"Training complete. Best val loss: {result.history.get('best_val_loss', 'N/A')}")

        return cls(model=result.model, correct_answers=correct_answers, device=device, embedding_strategy=strategy)

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make predictions for a single evaluation row.

        Args:
            evaluation_row: Row to predict

        Returns:
            SubmissionRow with top 3 predictions
        """
        eval_row_with_answer = EvaluationRow(
            row_id=evaluation_row.row_id,
            question_id=evaluation_row.question_id,
            question_text=evaluation_row.question_text,
            mc_answer=evaluation_row.mc_answer,
            student_explanation=evaluation_row.student_explanation,
            correct_answer=self.correct_answers.get(evaluation_row.question_id, ""),
        )

        embedding = self.embedding_strategy.fn(eval_row_with_answer)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)

        is_correct = evaluation_row.mc_answer == self.correct_answers.get(evaluation_row.question_id, "")
        correctness_idx = torch.tensor([1 if is_correct else 0], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                embedding_tensor,
                torch.LongTensor([evaluation_row.question_id]).to(self.device),
                correctness_idx,
            )

        predictions = []
        key = (evaluation_row.question_id, is_correct)

        if key in outputs:
            logits = outputs[key]
            probs = functional.softmax(logits, dim=-1)[0]
            top_k = min(MAX_PREDICTIONS, logits.size(-1))
            top_indices = torch.topk(probs, k=top_k)[1]

            encoder = (
                self.model.true_label_encoders.get(evaluation_row.question_id)
                if is_correct
                else self.model.false_label_encoders.get(evaluation_row.question_id)
            )

            if encoder:
                predictions.extend(
                    Prediction.from_string(pred_str)
                    for pred_str in encoder.inverse_transform(top_indices.cpu().numpy())
                )

        default = Prediction(
            category=Category.TRUE_NEITHER if is_correct else Category.FALSE_NEITHER,
            misconception="NA",
        )
        while len(predictions) < MAX_PREDICTIONS:
            predictions.append(default)

        return SubmissionRow(row_id=evaluation_row.row_id, predicted_categories=predictions[:MAX_PREDICTIONS])

    def evaluate(
        self,
        test_data: list[TrainingRow] | None = None,
        train_csv_path: Path = Path("datasets/train.csv"),
    ) -> dict[str, float]:
        """Evaluate the predictor on test data.

        Args:
            test_data: Optional test data. If None, uses validation split from train.csv
            train_csv_path: Path to training data

        Returns:
            Dictionary with evaluation metrics
        """
        if test_data is None:
            training_data = load_training_data(train_csv_path)
            n_samples = len(training_data)
            _, val_idx, _ = _get_split_indices(n_samples, 0.7, 42)
            test_data = [training_data[i] for i in val_idx]

        map_scores = []
        for row in test_data:
            eval_row = EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )
            prediction = self.predict(eval_row)
            score = calculate_map_at_3(row.prediction, prediction.predicted_categories)
            map_scores.append(score)

        avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
        logger.info(f"Evaluation MAP@3: {avg_map:.4f} on {len(test_data)} samples")

        return {
            "validation_map@3": avg_map,
            "validation_samples": len(test_data),
        }

    def save(self, filepath: Path) -> None:
        """Save predictor to disk."""
        logger.info(f"Saving predictor to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "Predictor":
        """Load predictor from disk."""
        logger.info(f"Loading predictor from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"
        with filepath.open("rb") as f:
            return pickle.load(f)
