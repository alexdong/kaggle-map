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

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Disable tokenizer parallelism to avoid fork warnings with DataLoader multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import optuna
import pandas as pd
import torch
import wandb
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

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


class ListMLELoss(nn.Module):
    """ListMLE loss for learning to rank, optimized for MAP@3."""

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Compute ListMLE loss for ranking.

        Args:
            scores: [batch_size, n_classes] - predicted scores for each class
            labels: [batch_size] - ground truth class indices
            k: Top-k positions to optimize for (default 3 for MAP@3)
        """
        batch_size, n_classes = scores.shape

        # Create one-hot encoding of labels
        labels_one_hot = torch.zeros_like(scores)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Sort scores in descending order
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True)

        # Reorder labels according to sorted scores
        sorted_labels = labels_one_hot.gather(1, indices)

        # Compute ListMLE loss focusing on top-k positions
        top_k_scores = sorted_scores[:, :k]
        top_k_labels = sorted_labels[:, :k]

        # Compute probability for correct items being ranked high
        exp_scores = torch.exp(top_k_scores)
        cumsum_exp_scores = torch.cumsum(exp_scores, dim=1)

        # Avoid log(0)
        epsilon = 1e-10
        log_probs = top_k_scores - torch.log(cumsum_exp_scores + epsilon)

        # Weight by ground truth labels
        loss = -torch.sum(top_k_labels * log_probs, dim=1)

        return loss.mean()


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific misconception heads."""

    def __init__(
        self,
        question_predictions: dict[QuestionId, list[str]],
        embedding_dim: int = 768,  # Add parameter for actual embedding dimension
        config: TorchConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = TorchConfig()

        # Add learnable correctness embedding
        correctness_embedding_dim = 32
        self.correctness_embedding = nn.Embedding(2, correctness_embedding_dim)

        activation = get_activation(config.activation)
        # Dynamic input dimension: embedding_dim + correctness_embedding_dim
        input_dim = embedding_dim + correctness_embedding_dim

        # Use configurable trunk_layers if provided, otherwise use default scaling
        if hasattr(config, "trunk_layers") and config.trunk_layers:
            # Hyperparameter search case - use provided architecture
            dims = config.trunk_layers
            assert dims[0] == input_dim, f"First layer must match input_dim {input_dim}, got {dims[0]}"
        # Default case - scale based on embedding dimension
        elif embedding_dim == 384:  # MINI_LM model
            dims = [input_dim, 512, 256, 192, 128]
        else:  # Larger models (768 dim)
            dims = [input_dim, 1024, 512, 256, 192]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))  # Add layer normalization
            layers.append(activation)
            layers.append(nn.Dropout(config.dropout))

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
        """Forward pass returning logits per question, split by correctness.

        Args:
            x: [batch_size, 768] - concatenated question and answer embeddings
            question_ids: [batch_size] - question IDs
            is_correct: [batch_size] - correctness indices (0 or 1)
        """
        # Get learnable correctness embeddings
        correct_emb = self.correctness_embedding(is_correct.long())

        # Concatenate embeddings with correctness
        combined = torch.cat([x, correct_emb.squeeze(1) if correct_emb.dim() > 2 else correct_emb], dim=-1)

        # Pass through trunk
        shared_features = self.trunk(combined)

        outputs = {}
        unique_questions = torch.unique(question_ids)

        for qid in unique_questions:
            qid_int = int(qid.item())
            mask = question_ids == qid

            if mask.any():
                question_features = shared_features[mask]
                question_correctness = is_correct[mask]

                # Separate features by correctness
                correct_mask = question_correctness > 0
                incorrect_mask = ~correct_mask

                # Process correct answers through true_heads
                if correct_mask.any() and str(qid_int) in self.true_heads:
                    correct_features = question_features[correct_mask]
                    outputs[(qid_int, True)] = self.true_heads[str(qid_int)](correct_features)

                # Process incorrect answers through false_heads
                if incorrect_mask.any() and str(qid_int) in self.false_heads:
                    incorrect_features = question_features[incorrect_mask]
                    outputs[(qid_int, False)] = self.false_heads[str(qid_int)](incorrect_features)

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
        # Return correctness as index for embedding lookup (0 or 1)
        is_correct_idx = torch.tensor(1 if is_correct else 0, dtype=torch.long)

        label_encoder = self.true_label_encoders.get(qid) if is_correct else self.false_label_encoders.get(qid)

        if label_encoder and prediction in label_encoder.classes_:
            label = label_encoder.transform([prediction])[0]
        else:
            label = 0

        # Return embeddings and correctness index separately
        return self.embeddings[idx], self.question_ids[idx], torch.tensor(label, dtype=torch.long), is_correct_idx


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

    @classmethod
    def get_hyperparameter_search_space(cls, trial) -> dict[str, Any]:
        """Focused 4-hour search space for efficient exploitation.

        Based on insights from previous runs:
        - XLarge architecture performs best (MAP@3 > 0.90)
        - Optimal learning rate: 1e-4 to 3e-4
        - Best batch sizes: 256-384
        - Effective dropout: 0.30-0.40
        """
        if optuna is None:
            msg = "Optuna is required for hyperparameter search"
            raise ImportError(msg)

        # Always use focused exploitation strategy for 4-hour run
        if True:  # Keep structure for easy switching back later
            return {
                # Dense sampling around optimal LR range [8e-5, 3e-4]
                "learning_rate": trial.suggest_float("learning_rate", 8e-5, 3e-4, log=True),
                # Focus on proven batch sizes with fine gradations
                "batch_size": trial.suggest_categorical("batch_size", [224, 256, 288, 320, 384, 448, 512]),
                # Fine-grained dropout exploration around optimum
                "dropout": trial.suggest_float("dropout", 0.30, 0.42),
                # Heavy bias toward xlarge (85%), some large (10%), rare xxlarge (5%)
                "architecture_size": trial.suggest_categorical(
                    "architecture_size", ["xlarge"] * 17 + ["large"] * 2 + ["xxlarge"]
                ),
                # Both optimizers with AdamW preference
                "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adamw", "adamw", "adam"]),
                # Focus on promising weight decay range
                "weight_decay": trial.suggest_float("weight_decay", 3e-3, 1.5e-2, log=True),
                # Test all successful activations
                "activation": trial.suggest_categorical("activation", ["gelu", "silu", "relu", "leaky_relu"]),
                # Focus on successful schedulers
                "scheduler": trial.suggest_categorical("scheduler", ["cosine", "cosine", "onecycle", "none"]),
                # Optimal patience range
                "early_stopping_patience": trial.suggest_int("patience", 16, 22),
                "epochs": trial.suggest_int("epochs", 28, 36),
            }
        return None

    @classmethod
    def _get_architecture_config(cls, size: str) -> dict[str, Any]:
        """Get architecture configuration for different model sizes."""
        configs = {
            "small": {"hidden_dim": 256, "trunk_layers": [800, 512, 256, 128, 96]},
            "medium": {"hidden_dim": 512, "trunk_layers": [800, 1024, 512, 256, 192]},
            "large": {"hidden_dim": 768, "trunk_layers": [800, 1536, 768, 384, 256]},
            "xlarge": {"hidden_dim": 1024, "trunk_layers": [800, 2048, 1024, 512, 384]},
            "xxlarge": {"hidden_dim": 1536, "trunk_layers": [800, 2560, 1536, 768, 512]},
        }
        return configs[size]

    @classmethod
    def create_config_from_hyperparams(cls, hyperparams: dict[str, Any], **base_params: Any) -> TorchConfig:
        """Convert hyperparameters to TorchConfig, including architecture scaling."""
        # Extract architecture size and get config
        arch_size = hyperparams.pop("architecture_size", "medium")
        arch_config = cls._get_architecture_config(arch_size)

        # Merge architecture config with hyperparameters
        config_params = {**hyperparams, **arch_config, **base_params}

        return TorchConfig(**{k: v for k, v in config_params.items() if hasattr(TorchConfig, k)})

    @staticmethod
    def process_mlp_batch_train(
        model: QuestionSpecificMLP,
        batch: tuple,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int]:
        """Process a batch for MLP's multi-head architecture during training.

        Args:
            model: The MLP model with question-specific heads
            batch: Tuple of (embeddings, question_ids, labels, is_correct)
            criterion: Loss function (ListMLELoss or CrossEntropyLoss)
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
            correctness_mask = batch_correctness > 0 if is_correct else batch_correctness == 0
            question_mask = batch_questions == qid
            combined_mask = question_mask & correctness_mask

            if combined_mask.any():
                question_labels = batch_labels[combined_mask]
                if logits.size(0) == question_labels.size(0):
                    total_loss += criterion(logits, question_labels) * logits.size(0)
                    total_samples += logits.size(0)

        if total_samples > 0:
            # Keep gradients for training
            loss = total_loss / total_samples
            return loss, int(batch_embeddings.size(0))
        return None, 0

    @staticmethod
    def process_mlp_batch_val(
        model: QuestionSpecificMLP,
        batch: tuple,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int]:
        """Process a batch for MLP's multi-head architecture during validation.

        Args:
            model: The MLP model with question-specific heads
            batch: Tuple of (embeddings, question_ids, labels, is_correct)
            criterion: Loss function (ListMLELoss or CrossEntropyLoss)
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
            correctness_mask = batch_correctness > 0 if is_correct else batch_correctness == 0
            question_mask = batch_questions == qid
            combined_mask = question_mask & correctness_mask

            if combined_mask.any():
                question_labels = batch_labels[combined_mask]
                if logits.size(0) == question_labels.size(0):
                    total_loss += criterion(logits, question_labels) * logits.size(0)
                    total_samples += logits.size(0)

        if total_samples > 0:
            # Detach for validation to save memory
            loss = (total_loss / total_samples).detach()
            return loss, int(batch_embeddings.size(0))
        return None, 0

    @classmethod
    def fit(cls, **kwargs) -> "MLPStrategy":
        """Fit the MLP strategy on training data."""
        # Get device early to adjust batch size
        device = get_device()

        # Suggest larger batch size for GPU if not explicitly set
        if "batch_size" not in kwargs and str(device) != "cpu":
            kwargs["batch_size"] = 256  # Larger batch for GPU
            logger.info(f"Using batch_size=256 for {device} (override with batch_size parameter)")

        # Use create_config_from_hyperparams if architecture_size is provided (hyperparameter search)
        if "architecture_size" in kwargs:
            config = cls.create_config_from_hyperparams(kwargs, wandb_project="kaggle-map-mlp")
        else:
            config = TorchConfig(
                wandb_project="kaggle-map-mlp", **{k: v for k, v in kwargs.items() if hasattr(TorchConfig, k)}
            )
        logger.info(f"Fitting MLP strategy from {config.train_csv_path} with batch_size={config.batch_size}")

        extra_config = {
            "architecture": "improved_mlp_with_correctness",
            "trunk_layers": [800, 1024, 512, 256, 192],
            "correctness_embedding_dim": 32,
            "include_question": True,
            "loss_function": "ListMLE",
            "layer_norm": True,
            "dropout": config.dropout,
        }
        init_wandb(config, extra_config)

        torch.manual_seed(config.random_seed)

        # Device was already obtained above for batch size adjustment
        logger.info(f"Using device: {device}")
        wandb.config.update({"device": str(device)})

        training_data = parse_training_data(config.train_csv_path)
        train_df = pd.read_csv(config.train_csv_path)

        correct_answers = extract_correct_answers(training_data)
        question_predictions = extract_question_predictions(training_data)

        def compute_embeddings(data):
            logger.info(f"Computing separate question and answer embeddings on device: {device}")
            tokenizer = get_tokenizer(device=str(device))

            # Prepare texts for batch encoding
            question_texts = []
            answer_texts = []
            question_ids_list = []
            predictions_list = []
            mc_answers_list = []

            for row in data:
                question_texts.append(row.question_text)
                answer_texts.append(f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}")
                question_ids_list.append(row.question_id)
                predictions_list.append(str(row.prediction))
                mc_answers_list.append(row.mc_answer)

            # Batch encode all texts at once for better GPU utilization
            logger.info(f"Batch encoding {len(question_texts)} questions and answers...")
            batch_size = 64 if str(device) != "cpu" else 32

            # Encode questions in batches
            question_embeddings = tokenizer.encode(question_texts, batch_size=batch_size, show_progress_bar=True)

            # Encode answers in batches
            answer_embeddings = tokenizer.encode(answer_texts, batch_size=batch_size, show_progress_bar=True)

            # Concatenate question and answer embeddings
            combined_embeddings = np.concatenate([question_embeddings, answer_embeddings], axis=1)
            logger.info(f"Computed embeddings with shape: {combined_embeddings.shape}")

            return (
                combined_embeddings,
                np.array(question_ids_list),
                {"predictions": np.array(predictions_list), "mc_answers": np.array(mc_answers_list)},
            )

        embeddings, question_ids, extra_data = load_embeddings(
            config.embeddings_path, training_data, train_df, compute_embeddings
        )
        predictions = extra_data.get("predictions", np.array([]))
        mc_answers = extra_data.get("mc_answers", np.array([]))

        # Get actual embedding dimension from the data
        embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 768
        logger.info(f"Creating model with embedding dimension: {embedding_dim}")

        model = QuestionSpecificMLP(question_predictions, embedding_dim=embedding_dim, config=config)
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

        # Create data loaders with optimizations for GPU
        # Use num_workers=0 to avoid "too many open files" error during hyperparameter search
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid file handle issues
            pin_memory=str(device) != "cpu",  # Pin memory for faster GPU transfers
            persistent_workers=False,  # Disable to avoid file handle accumulation
            prefetch_factor=None,  # Disable prefetching when num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,  # Larger batch for validation (no gradients)
            shuffle=False,
            num_workers=0,  # Disable multiprocessing to avoid file handle issues
            pin_memory=str(device) != "cpu",
            persistent_workers=False,  # Disable to avoid file handle accumulation
            prefetch_factor=None,  # Disable prefetching when num_workers=0
        )

        # Train the model using the generic training function with ListMLE loss
        model, history = train_torch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            criterion=ListMLELoss(),  # Use ranking loss instead of CrossEntropyLoss
            train_batch_fn=cls.process_mlp_batch_train,  # Training version keeps gradients
            val_batch_fn=cls.process_mlp_batch_val,  # Validation version detaches gradients
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
        # Compute separate embeddings for question and answer (original approach)
        question_emb = self.tokenizer.encode(evaluation_row.question_text)
        answer_text = f"Answer: {evaluation_row.mc_answer}; Explanation: {evaluation_row.student_explanation}"
        answer_emb = self.tokenizer.encode(answer_text)

        # Concatenate question and answer embeddings (768-dimensional)
        combined_emb = torch.FloatTensor(np.concatenate([question_emb, answer_emb]))
        combined_emb = combined_emb.unsqueeze(0).to(self.device)

        # Determine correctness
        is_correct = evaluation_row.mc_answer == self.correct_answers.get(evaluation_row.question_id, "")
        correctness_idx = torch.tensor([1 if is_correct else 0], dtype=torch.long).to(self.device)

        # Get predictions from model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                combined_emb,
                torch.LongTensor([evaluation_row.question_id]).to(self.device),
                correctness_idx,
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

            # Get embedding dimension from saved model state dict
            model_state = checkpoint["model_state_dict"]
            # Extract input dimension from the first layer of trunk
            first_layer_weight = model_state.get("trunk.0.weight")
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]  # [out_features, in_features]
                embedding_dim = input_dim - 32  # Subtract correctness embedding dimension
            else:
                embedding_dim = 768  # Default fallback

            training_data = parse_training_data(train_csv_path)
            mlp_model = QuestionSpecificMLP(
                extract_question_predictions(training_data), embedding_dim=embedding_dim, config=config
            )
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

        results.append(
            {
                "row_id": row.row_id,
                "question_id": row.question_id,
                "is_correct": is_correct,
                "ground_truth": f"{row.prediction.category.value}:{row.misconception}",
                "prediction_1": f"{predictions[0].category.value}:{predictions[0].misconception}",
                "map_score": score,
            }
        )

    # Calculate statistics
    map_scores = [r["map_score"] for r in results]
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    perfect = sum(1 for s in map_scores if s == 1.0)
    partial = sum(1 for s in map_scores if 0 < s < 1)
    misses = sum(1 for s in map_scores if s == 0)

    logger.info(f"\n{'=' * 60}")
    logger.info("PERFORMANCE ANALYSIS RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples analyzed: {len(results)}")
    logger.info(f"Average MAP@3: {avg_map:.4f}")
    logger.info(f"Perfect predictions (MAP=1.0): {perfect} ({perfect / len(results) * 100:.1f}%)")
    logger.info(f"Partial hits (0<MAP<1): {partial} ({partial / len(results) * 100:.1f}%)")
    logger.info(f"Complete misses (MAP=0): {misses} ({misses / len(results) * 100:.1f}%)")
    logger.info(f"Correctness prefix errors: {correctness_errors} ({correctness_errors / len(results) * 100:.1f}%)")

    # Analyze model weights
    logger.info(f"\n{'=' * 60}")
    logger.info("MODEL WEIGHT ANALYSIS")
    logger.info(f"{'=' * 60}")

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
