"""MLP neural network strategy for student misconception prediction.

Architecture Design:
-------------------
Shared Trunk:
  - Input: embedding(384) from sentence-transformers/all-MiniLM-L6-v2 (answer + explanation only)
  - Hidden 1: 768 units, ReLU, 0.3 dropout
  - Hidden 2: 384 units, ReLU, 0.3 dropout
  - Shared: 192 units, ReLU, 0.3 dropout

Question-Specific Prediction Heads (nn.ModuleDict):
  - One head per question ID (e.g., Q31772, Q31774, etc.)
  - Each head outputs N classes where N = unique Category:Misconception pairs for that question
  - Examples:
    - Q31772: 8 outputs → [True_Correct:NA, False_Misconception:Incomplete, False_Misconception:WNB, ...]
    - Q31774: 10 outputs → [True_Correct:NA, False_Misconception:Mult, True_Misconception:SwapDividend, ...]

Key Design Decisions:
--------------------
1. Full Prediction Labels: Predicts complete "Category:Misconception" pairs as atomic classes
2. Loss Function: CrossEntropyLoss for multi-class classification per question head
3. Loss Aggregation: Average loss across all question heads during training
4. Optimizer: Adam with learning rate 1e-3
5. Early Stopping: Based on validation loss with patience of 10 epochs
6. Data Split: 70% train, 15% validation, 15% test (configurable)
7. Embeddings: 384-dim embeddings from MiniLM using only answer and explanation text

Training Process:
----------------
1. Load precomputed embeddings or compute on-the-fly
2. Create question-specific label encoders for full prediction mapping
3. Train with mini-batches, grouping by question ID for head-specific loss
4. Track progress with Weights & Biases (wandb) integration
5. Apply early stopping based on validation loss
6. Save model with pickle and parameters as JSON
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

import wandb
from kaggle_map.core.dataset import (
    extract_correct_answers,
    parse_training_data,
)
from kaggle_map.core.embeddings.tokenizer import get_tokenizer
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    Answer,
    Category,
    EvaluationRow,
    Prediction,
    QuestionId,
    SubmissionRow,
)

from .base import Strategy
from .utils import (
    TRAIN_RATIO,
    ModelParameters,
    TorchConfig,
    extract_question_predictions,
    get_activation,
    get_device,
    get_split_indices,
    init_wandb,
    load_embeddings,
    load_torch_strategy,
    save_torch_strategy,
    split_training_data,
    train_torch_model,
)


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific misconception heads."""

    def __init__(
        self,
        question_predictions: dict[QuestionId, list[str]],
        config: TorchConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = TorchConfig()

        activation = get_activation(config.activation)
        dims = [385, 512, 256, 192]

        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), activation, nn.Dropout(config.dropout)])

        self.trunk = nn.Sequential(*layers)
        self.output_dim = dims[-1]

        self.true_heads = nn.ModuleDict()
        self.false_heads = nn.ModuleDict()
        self.true_label_encoders = {}
        self.false_label_encoders = {}

        for question_id, predictions in question_predictions.items():
            true_preds = [p for p in predictions if p.startswith("True_")]
            false_preds = [p for p in predictions if p.startswith("False_")]

            if true_preds:
                self.true_heads[str(question_id)] = nn.Linear(self.output_dim, len(true_preds))
                true_encoder = LabelEncoder()
                true_encoder.fit(sorted(true_preds))
                self.true_label_encoders[question_id] = true_encoder

            if false_preds:
                self.false_heads[str(question_id)] = nn.Linear(self.output_dim, len(false_preds))
                false_encoder = LabelEncoder()
                false_encoder.fit(sorted(false_preds))
                self.false_label_encoders[question_id] = false_encoder

    def forward(self, x: torch.Tensor, question_ids: torch.Tensor, is_correct: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass returning logits per question, split by correctness."""
        shared_features = self.trunk(x)

        outputs = {}
        unique_questions = torch.unique(question_ids)

        for qid in unique_questions:
            qid_int = int(qid.item())
            mask = question_ids == qid

            if mask.any():
                question_features = shared_features[mask]
                question_correctness = is_correct[mask]

                # Use different heads based on correctness
                # We'll return a tuple (qid, is_correct) as key
                if question_correctness[0].item() > 0.5:  # Answer is correct
                    if str(qid_int) in self.true_heads:
                        outputs[(qid_int, True)] = self.true_heads[str(qid_int)](question_features)
                elif str(qid_int) in self.false_heads:
                    outputs[(qid_int, False)] = self.false_heads[str(qid_int)](question_features)

        return outputs


class MLPDataset(Dataset):
    """Dataset for MLP training."""

    def __init__(
        self,
        embeddings: np.ndarray,
        question_ids: np.ndarray,
        predictions: np.ndarray,  # Full prediction strings
        correct_answers: dict[QuestionId, Answer],
        mc_answers: np.ndarray,  # Student answers
        true_label_encoders: dict[QuestionId, LabelEncoder],
        false_label_encoders: dict[QuestionId, LabelEncoder],
    ) -> None:
        self.embeddings = torch.FloatTensor(embeddings)
        self.question_ids = torch.LongTensor(question_ids)
        self.predictions = predictions
        self.correct_answers = correct_answers
        self.mc_answers = mc_answers
        self.true_label_encoders = true_label_encoders
        self.false_label_encoders = false_label_encoders

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qid = int(self.question_ids[idx].item())
        prediction = self.predictions[idx]
        mc_answer = self.mc_answers[idx]

        is_correct = mc_answer == self.correct_answers.get(qid, "")
        is_correct_tensor = torch.tensor([1.0 if is_correct else 0.0], dtype=torch.float32)

        enhanced_embedding = torch.cat([self.embeddings[idx], is_correct_tensor], dim=0)

        label_encoder = self.true_label_encoders.get(qid) if is_correct else self.false_label_encoders.get(qid)

        if label_encoder and prediction in label_encoder.classes_:
            label = label_encoder.transform([prediction])[0]
        else:
            label = 0

        return enhanced_embedding, self.question_ids[idx], torch.tensor(label, dtype=torch.long), is_correct_tensor


@dataclass(frozen=True)
class MLPStrategy(Strategy):
    """MLP neural network strategy for misconception prediction.

    Uses a shared trunk with question-specific heads for multi-class
    classification of misconceptions per question.
    """

    model: QuestionSpecificMLP
    correct_answers: dict[QuestionId, Answer]
    tokenizer: object  # SentenceTransformer instance
    device: torch.device
    parameters: ModelParameters | None = None

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def description(self) -> str:
        return "MLP with shared trunk and question-specific heads for full prediction (Category:Misconception)"

    @staticmethod
    def process_mlp_batch(
        model: QuestionSpecificMLP,
        batch: tuple,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int]:
        """Process a batch for MLP's multi-head architecture.

        Args:
            model: The MLP model with question-specific heads
            batch: Tuple of (embeddings, question_ids, labels, is_correct)
            criterion: Loss function (typically CrossEntropyLoss)
            device: Device to run computations on

        Returns:
            Tuple of (loss, batch_size) or (None, 0) if no valid samples
        """
        batch_embeds, batch_quests, batch_lbls, batch_correct = batch
        batch_embeddings = batch_embeds.to(device)
        batch_questions = batch_quests.to(device)
        batch_labels = batch_lbls.to(device)
        batch_correctness = batch_correct.to(device)

        outputs = model(batch_embeddings, batch_questions, batch_correctness)

        total_loss = 0.0
        total_samples = 0
        for (qid, is_correct), logits in outputs.items():
            correctness_mask = batch_correctness.squeeze() > 0.5 if is_correct else batch_correctness.squeeze() <= 0.5
            question_mask = batch_questions == qid
            combined_mask = question_mask & correctness_mask

            if combined_mask.any():
                question_labels = batch_labels[combined_mask]
                if logits.size(0) == question_labels.size(0):
                    total_loss += criterion(logits, question_labels) * logits.size(0)
                    total_samples += logits.size(0)

        if total_samples > 0:
            loss = torch.tensor(total_loss / total_samples)
            return loss, int(batch_embeddings.size(0))
        return None, 0

    @classmethod
    def fit(cls, **kwargs) -> "MLPStrategy":
        """Fit the MLP strategy on training data."""
        config = TorchConfig(
            wandb_project="kaggle-map-mlp", **{k: v for k, v in kwargs.items() if hasattr(TorchConfig, k)}
        )
        logger.info(f"Fitting MLP strategy from {config.train_csv_path}")

        extra_config = {
            "architecture": "shared_trunk_question_heads",
            "trunk_layers": [384, 768, 384, 192],
            "dropout": 0.3,
        }
        init_wandb(config, extra_config)

        torch.manual_seed(config.random_seed)

        device = get_device()
        logger.info(f"Using device: {device}")
        wandb.config.update({"device": str(device)})

        training_data = parse_training_data(config.train_csv_path)
        train_df = pd.read_csv(config.train_csv_path)

        correct_answers = extract_correct_answers(training_data)
        question_predictions = extract_question_predictions(training_data)

        def compute_embeddings(data):
            logger.info("Computing embeddings (no precomputed file found)")
            tokenizer = get_tokenizer()

            embeddings_list = []
            question_ids_list = []
            predictions_list = []
            mc_answers_list = []

            for row in data:
                text = f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}"
                embedding = tokenizer.encode(text)

                embeddings_list.append(embedding)
                question_ids_list.append(row.question_id)
                predictions_list.append(str(row.prediction))
                mc_answers_list.append(row.mc_answer)

            return (
                np.array(embeddings_list),
                np.array(question_ids_list),
                {"predictions": np.array(predictions_list), "mc_answers": np.array(mc_answers_list)},
            )

        embeddings, question_ids, extra_data = load_embeddings(
            config.embeddings_path, training_data, train_df, compute_embeddings
        )
        predictions = extra_data.get("predictions", np.array([]))
        mc_answers = extra_data.get("mc_answers", np.array([]))

        model = QuestionSpecificMLP(question_predictions, config=config)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "num_questions": len(question_predictions),
                "total_samples": len(embeddings),
            }
        )

        n_samples = len(embeddings)
        train_indices, val_indices, test_indices = get_split_indices(
            n_samples, train_ratio=config.train_split, random_seed=config.random_seed
        )

        wandb.config.update(
            {
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
            }
        )

        # Create datasets
        train_dataset = MLPDataset(
            embeddings[train_indices],
            question_ids[train_indices],
            predictions[train_indices],
            correct_answers,
            mc_answers[train_indices],
            model.true_label_encoders,
            model.false_label_encoders,
        )

        val_dataset = MLPDataset(
            embeddings[val_indices],
            question_ids[val_indices],
            predictions[val_indices],
            correct_answers,
            mc_answers[val_indices],
            model.true_label_encoders,
            model.false_label_encoders,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Train the model using the generic training function
        model, history = train_torch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            criterion=nn.CrossEntropyLoss(),
            train_batch_fn=cls.process_mlp_batch,
            val_batch_fn=cls.process_mlp_batch,  # Same logic for validation
        )

        # Get tokenizer for predictions
        tokenizer = get_tokenizer()

        # Create model parameters for tracking
        parameters = ModelParameters.create(
            train_split=config.train_split,
            random_seed=config.random_seed,
            train_indices=list(train_indices),
            val_indices=list(val_indices),
            test_indices=list(test_indices),
            total_samples=n_samples,
        )

        # Finish wandb run
        wandb.finish()

        return cls(
            model=model,
            correct_answers=correct_answers,
            tokenizer=tokenizer,
            device=device,
            parameters=parameters,
        )

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make predictions on a single evaluation row."""
        # Prepare input
        is_correct = evaluation_row.mc_answer == self.correct_answers.get(evaluation_row.question_id, "")
        text = f"Answer: {evaluation_row.mc_answer}; Explanation: {evaluation_row.student_explanation}"
        embedding = torch.FloatTensor(self.tokenizer.encode(text))
        enhanced = torch.cat([embedding, torch.tensor([float(is_correct)])]).unsqueeze(0).to(self.device)

        # Get predictions from model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                enhanced,
                torch.LongTensor([evaluation_row.question_id]).to(self.device),
                torch.tensor([[float(is_correct)]]).to(self.device),
            )

        # Extract predictions
        predictions = []
        key = (evaluation_row.question_id, is_correct)

        if key in outputs:
            logits = outputs[key]
            top_indices = torch.topk(functional.softmax(logits, dim=-1)[0], k=min(3, logits.size(-1)))[1]
            encoder = (
                self.model.true_label_encoders.get(evaluation_row.question_id)
                if is_correct
                else self.model.false_label_encoders.get(evaluation_row.question_id)
            )

            if encoder:
                for pred_str in encoder.inverse_transform(top_indices.cpu().numpy()):
                    predictions.append(Prediction.from_string(pred_str))

        # Ensure 3 predictions with appropriate defaults
        default = Prediction(
            category=Category.TRUE_NEITHER if is_correct else Category.FALSE_NEITHER, misconception="NA"
        )
        while len(predictions) < 3:
            predictions.append(default)

        return SubmissionRow(row_id=evaluation_row.row_id, predicted_categories=predictions[:3])

    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        save_torch_strategy(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "MLPStrategy":
        """Load model from disk."""
        return load_torch_strategy(cls, filepath)

    @classmethod
    def evaluate_on_split(
        cls,
        model: "MLPStrategy | None" = None,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        checkpoint_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate model on validation split using MAP@3 metric."""
        if model is None:
            if not checkpoint_path:
                checkpoints = list(Path("checkpoints").glob("mlp_best_*.pt"))
                if not checkpoints:
                    msg = "No model provided and no checkpoints found!"
                    raise ValueError(msg)
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            config = checkpoint["config"]

            training_data = parse_training_data(train_csv_path)
            mlp_model = QuestionSpecificMLP(extract_question_predictions(training_data), config=config)
            mlp_model.load_state_dict(checkpoint["model_state_dict"])

            model = cls(
                model=mlp_model.to(get_device()),
                correct_answers=extract_correct_answers(training_data),
                tokenizer=get_tokenizer(),
                device=get_device(),
                parameters=None,
            )
            train_split = config.train_split
            random_seed = config.random_seed

        training_data = parse_training_data(train_csv_path)
        _, val_data, _ = split_training_data(training_data, train_ratio=train_split, random_seed=random_seed)

        map_scores = [
            calculate_map_at_3(
                row.prediction,
                model.predict(
                    EvaluationRow(
                        row_id=row.row_id,
                        question_id=row.question_id,
                        question_text=row.question_text,
                        mc_answer=row.mc_answer,
                        student_explanation=row.student_explanation,
                    )
                ).predicted_categories,
            )
            for row in val_data
        ]

        avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
        logger.info(f"Validation MAP@3: {avg_map:.4f} on {len(val_data)} samples")

        return {
            "validation_map@3": avg_map,
            "validation_samples": len(val_data),
        }


if __name__ == "__main__":
    import sys

    logger.info("Analyzing MLP model performance")
    model_path = Path("models/mlp.pkl")
    train_csv_path = Path("datasets/train.csv")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Analyze predictions
    logger.info(f"Loading model from {model_path}")
    model = MLPStrategy.load(model_path)

    logger.info(f"Loading training data from {train_csv_path}")
    training_data = parse_training_data(train_csv_path)

    # Split data to get validation set
    from .utils import split_training_data
    _, val_data, _ = split_training_data(training_data, train_ratio=0.7, random_seed=42)

    # Sample validation data for analysis
    n_samples = 500
    sampled_data = val_data[:n_samples] if len(val_data) > n_samples else val_data

    # Analyze predictions
    results = []
    correctness_errors = 0

    for row in sampled_data:
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )

        # Get predictions
        submission = model.predict(eval_row)
        predictions = submission.predicted_categories

        # Check for correctness consistency
        is_correct = row.mc_answer == model.correct_answers.get(row.question_id, "")
        expected_prefix = "True_" if is_correct else "False_"

        # Check if predictions have correct prefix
        for i, pred in enumerate(predictions):
            pred_str = f"{pred.category.value}:{pred.misconception}"
            if not pred_str.startswith(expected_prefix):
                correctness_errors += 1
                break

        # Calculate MAP@3
        score = calculate_map_at_3(row.prediction, predictions)

        results.append({
            "row_id": row.row_id,
            "question_id": row.question_id,
            "is_correct": is_correct,
            "ground_truth": f"{row.prediction.category.value}:{row.misconception}",
            "prediction_1": f"{predictions[0].category.value}:{predictions[0].misconception}",
            "map_score": score
        })

    # Calculate statistics
    map_scores = [r["map_score"] for r in results]
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    perfect = sum(1 for s in map_scores if s == 1.0)
    partial = sum(1 for s in map_scores if 0 < s < 1)
    misses = sum(1 for s in map_scores if s == 0)

    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE ANALYSIS RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples analyzed: {len(results)}")
    logger.info(f"Average MAP@3: {avg_map:.4f}")
    logger.info(f"Perfect predictions (MAP=1.0): {perfect} ({perfect/len(results)*100:.1f}%)")
    logger.info(f"Partial hits (0<MAP<1): {partial} ({partial/len(results)*100:.1f}%)")
    logger.info(f"Complete misses (MAP=0): {misses} ({misses/len(results)*100:.1f}%)")
    logger.info(f"Correctness prefix errors: {correctness_errors} ({correctness_errors/len(results)*100:.1f}%)")

    # Analyze model weights
    logger.info(f"\n{'='*60}")
    logger.info("MODEL WEIGHT ANALYSIS")
    logger.info(f"{'='*60}")

    mlp_model = model.model
    for i, layer in enumerate(mlp_model.trunk):
        if isinstance(layer, nn.Linear):
            weight_mean = layer.weight.data.mean().item()
            weight_std = layer.weight.data.std().item()
            logger.info(f"Trunk Layer {i}: weight mean={weight_mean:.4f}, std={weight_std:.4f}")

    logger.info("\nTrue heads (for correct answers):")
    for qid, head in list(mlp_model.true_heads.items())[:5]:
        weight_norm = torch.norm(head.weight.data).item()
        logger.info(f"  Q{qid}: weight norm={weight_norm:.4f}, outputs={head.out_features}")

    logger.info("\nFalse heads (for incorrect answers):")
    for qid, head in list(mlp_model.false_heads.items())[:5]:
        weight_norm = torch.norm(head.weight.data).item()
        logger.info(f"  Q{qid}: weight norm={weight_norm:.4f}, outputs={head.out_features}")