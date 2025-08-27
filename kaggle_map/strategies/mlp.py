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

import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim
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
    get_device,
    get_split_indices,
    split_training_data,
)


@dataclass
class MLPConfig:
    """Configuration for MLP training."""
    train_split: float = TRAIN_RATIO
    random_seed: int = 42
    train_csv_path: Path = field(default_factory=lambda: Path("datasets/train.csv"))
    embeddings_path: Path | None = field(default_factory=lambda: Path("datasets/train_embeddings.npz"))
    epochs: int = 20
    n_epochs: int = 20  # Alias for epochs
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 5
    wandb_project: str = "kaggle-map-mlp"
    wandb_run_name: str | None = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    # Architecture hyperparameters
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"  # relu, gelu, leaky_relu
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "onecycle"  # none, cosine, step, onecycle
    device: str = "auto"  # auto, cpu, cuda, mps


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific misconception heads."""

    def __init__(
        self,
        question_predictions: dict[QuestionId, list[str]],
        config: MLPConfig | None = None,
    ) -> None:
        super().__init__()

        # Use config or defaults
        if config is None:
            config = MLPConfig()

        # Get activation function
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        activation = activation_map.get(config.activation, nn.ReLU())

        # Build trunk layers dynamically based on config
        layers = []
        input_dim = 384 + 1  # +1 for answer correctness feature

        # Create hidden layers
        hidden_dims = [config.hidden_dim] * config.n_layers
        # Gradually reduce dimensions
        for i, hidden_dim in enumerate(hidden_dims):
            # Adjust dimension for deeper layers
            adjusted_dim = int(hidden_dim * (0.5 ** (i / max(config.n_layers - 1, 1))))
            adjusted_dim = max(adjusted_dim, 192)  # Minimum dimension

            layers.append(nn.Linear(input_dim, adjusted_dim))
            layers.append(activation)
            layers.append(nn.Dropout(config.dropout))
            input_dim = adjusted_dim

        self.trunk = nn.Sequential(*layers)
        self.output_dim = input_dim  # Final dimension from trunk

        # Question-specific heads split by correctness
        self.true_heads = nn.ModuleDict()  # For when answer is correct
        self.false_heads = nn.ModuleDict()  # For when answer is incorrect
        self.true_label_encoders = {}
        self.false_label_encoders = {}

        for question_id, predictions in question_predictions.items():
            # Split predictions by True_ and False_ prefixes
            true_preds = [p for p in predictions if p.startswith("True_")]
            false_preds = [p for p in predictions if p.startswith("False_")]

            # Create heads for True predictions
            if true_preds:
                self.true_heads[str(question_id)] = nn.Linear(self.output_dim, len(true_preds))
                true_encoder = LabelEncoder()
                true_encoder.fit(sorted(true_preds))
                self.true_label_encoders[question_id] = true_encoder

            # Create heads for False predictions
            if false_preds:
                self.false_heads[str(question_id)] = nn.Linear(self.output_dim, len(false_preds))
                false_encoder = LabelEncoder()
                false_encoder.fit(sorted(false_preds))
                self.false_label_encoders[question_id] = false_encoder

    def forward(self, x: torch.Tensor, question_ids: torch.Tensor, is_correct: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass returning logits per question, split by correctness."""
        # Pass through shared trunk
        shared_features = self.trunk(x)

        # Group by question and correctness, apply appropriate head
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

        # Determine if answer is correct
        is_correct = mc_answer == self.correct_answers.get(qid, "")
        is_correct_tensor = torch.tensor([1.0 if is_correct else 0.0], dtype=torch.float32)

        # Add correctness as a feature to embeddings
        enhanced_embedding = torch.cat([self.embeddings[idx], is_correct_tensor], dim=0)

        # Encode prediction based on correctness
        label_encoder = self.true_label_encoders.get(qid) if is_correct else self.false_label_encoders.get(qid)

        if label_encoder and prediction in label_encoder.classes_:
            label = label_encoder.transform([prediction])[0]
        else:
            label = 0  # Default

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
    def extract_question_predictions(training_data: list) -> dict[QuestionId, list[str]]:
        """Extract unique prediction strings per question."""
        question_predictions = defaultdict(list)
        for row in training_data:
            pred_str = str(row.prediction)
            question_predictions[row.question_id].append(pred_str)

        # Return unique predictions per question
        return {
            qid: list(set(preds))
            for qid, preds in question_predictions.items()
        }

    @staticmethod
    def _load_embeddings_and_labels(
        embeddings_path: Path | None,
        training_data: list,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load or compute embeddings and associated labels."""
        if embeddings_path and embeddings_path.exists():
            logger.info(f"Loading precomputed embeddings from {embeddings_path}")
            data = np.load(embeddings_path)
            embeddings = data["embeddings"]
            row_ids = data["row_ids"]
            # We need to reconstruct full predictions from the CSV
            # Get question IDs and predictions from training data
            question_ids = train_df.set_index("row_id").loc[row_ids]["QuestionId"].to_numpy()
            mc_answers = train_df.set_index("row_id").loc[row_ids]["MC_Answer"].to_numpy()
            # Create full prediction strings from the training data
            row_id_to_prediction = {row.row_id: str(row.prediction) for row in training_data}
            predictions = np.array([row_id_to_prediction[rid] for rid in row_ids])
        else:
            logger.info("Computing embeddings from scratch")
            tokenizer = get_tokenizer()
            embeddings = []
            question_ids = []
            predictions = []
            mc_answers = []
            for row in training_data:
                # Create text from only answer and explanation
                text = f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}"
                embedding = tokenizer.encode(text)
                embeddings.append(embedding)
                question_ids.append(row.question_id)
                predictions.append(str(row.prediction))  # Store full prediction string (category:misconception)
                mc_answers.append(row.mc_answer)
            embeddings = np.array(embeddings)
            question_ids = np.array(question_ids)
            predictions = np.array(predictions)
            mc_answers = np.array(mc_answers)
        return embeddings, question_ids, predictions, mc_answers

    @staticmethod
    def _train_epoch(
        model: QuestionSpecificMLP,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        device: str,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> float:
        """Train for one epoch and return average loss."""
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_embeds, batch_quests, batch_lbls, batch_correct in train_loader:
            batch_embeddings = batch_embeds.to(device)
            batch_questions = batch_quests.to(device)
            batch_labels = batch_lbls.to(device)
            batch_correctness = batch_correct.to(device)

            optimizer.zero_grad()
            outputs = model(batch_embeddings, batch_questions, batch_correctness)

            # Calculate loss - outputs is a dict of (qid, is_correct) -> logits for those samples
            total_loss = 0.0
            total_samples = 0
            for (qid, is_correct), logits in outputs.items():
                # Find indices for this question and correctness combination
                correctness_mask = batch_correctness.squeeze() > 0.5 if is_correct else batch_correctness.squeeze() <= 0.5
                question_mask = batch_questions == qid
                combined_mask = question_mask & correctness_mask

                if combined_mask.any():
                    question_labels = batch_labels[combined_mask]
                    # Ensure logits and labels have matching dimensions
                    if logits.size(0) == question_labels.size(0):
                        total_loss += criterion(logits, question_labels) * logits.size(0)
                        total_samples += logits.size(0)

            if total_samples > 0:
                loss = total_loss / total_samples  # Weighted average
                loss.backward()
                optimizer.step()

                # Step OneCycle scheduler after each batch
                if scheduler is not None and hasattr(scheduler, "__class__") and scheduler.__class__.__name__ == "OneCycleLR":
                    scheduler.step()

                train_loss += loss.item() * batch_embeddings.size(0)
                train_samples += batch_embeddings.size(0)

        return train_loss / train_samples if train_samples > 0 else 0

    @staticmethod
    def _validate_epoch(
        model: QuestionSpecificMLP,
        val_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        device: str,
    ) -> float:
        """Validate for one epoch and return average loss."""
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_embeds, batch_quests, batch_lbls, batch_correct in val_loader:
                batch_embeddings = batch_embeds.to(device)
                batch_questions = batch_quests.to(device)
                batch_labels = batch_lbls.to(device)
                batch_correctness = batch_correct.to(device)

                outputs = model(batch_embeddings, batch_questions, batch_correctness)

                # Calculate loss - outputs is a dict of (qid, is_correct) -> logits for those samples
                total_loss = 0.0
                total_samples = 0
                for (qid, is_correct), logits in outputs.items():
                    # Find indices for this question and correctness combination
                    correctness_mask = batch_correctness.squeeze() > 0.5 if is_correct else batch_correctness.squeeze() <= 0.5
                    question_mask = batch_questions == qid
                    combined_mask = question_mask & correctness_mask

                    if combined_mask.any():
                        question_labels = batch_labels[combined_mask]
                        # Ensure logits and labels have matching dimensions
                        if logits.size(0) == question_labels.size(0):
                            total_loss += criterion(logits, question_labels) * logits.size(0)
                            total_samples += logits.size(0)

                if total_samples > 0:
                    loss = total_loss / total_samples  # Weighted average
                    val_loss += loss * total_samples
                    val_samples += total_samples

        return val_loss / val_samples if val_samples > 0 else 0

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> dict:
        """Load a training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # weights_only=False needed for loading custom classes like MLPConfig
        return torch.load(checkpoint_path, weights_only=False)

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        resume_from_checkpoint: Path | None = None,
        config: MLPConfig | None = None,
        **kwargs: object,
    ) -> "MLPStrategy":
        """Fit the MLP strategy on training data."""
        # Use provided config or create one with defaults and overrides
        if config is None:
            config = MLPConfig(
                train_split=train_split,
                random_seed=random_seed,
                train_csv_path=train_csv_path,
                embeddings_path=Path(str(kwargs.get("embeddings_path", "datasets/train_embeddings.npz"))),
                epochs=int(kwargs.get("epochs", 20)),
                batch_size=int(kwargs.get("batch_size", 128)),
                learning_rate=float(kwargs.get("learning_rate", 1e-3)),
                early_stopping_patience=int(kwargs.get("early_stopping_patience", 5)),
                wandb_project=str(kwargs.get("wandb_project", "kaggle-map-mlp")),
                wandb_run_name=str(kwargs.get("wandb_run_name")) if kwargs.get("wandb_run_name") else None,
            )
        else:
            # Update config with provided parameters if not already set
            if not hasattr(config, "train_split"):
                config.train_split = train_split
            if not hasattr(config, "random_seed"):
                config.random_seed = random_seed
            if not hasattr(config, "train_csv_path"):
                config.train_csv_path = train_csv_path

        logger.info(f"Fitting MLP strategy from {config.train_csv_path}")

        # Initialize wandb
        # Auto-generate descriptive run name if not provided
        if not config.wandb_run_name:
            config.wandb_run_name = (
                f"mlp-e{config.epochs}-"
                f"bs{config.batch_size}-"
                f"lr{config.learning_rate:.0e}-"
                f"split{int(config.train_split*100)}-"
                f"seed{config.random_seed}"
            )

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "train_split": config.train_split,
                "random_seed": config.random_seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "early_stopping_patience": config.early_stopping_patience,
                "architecture": "shared_trunk_question_heads",
                "trunk_layers": [384, 768, 384, 192],
                "dropout": 0.3,
            }
        )

        # Set random seeds
        torch.manual_seed(config.random_seed)

        # Detect device
        device = get_device()
        logger.info(f"Using device: {device}")
        wandb.config.update({"device": str(device)})

        # Load training data
        training_data = parse_training_data(config.train_csv_path)
        train_df = pd.read_csv(config.train_csv_path)

        # Extract correct answers and question predictions
        correct_answers = extract_correct_answers(training_data)
        question_predictions = cls.extract_question_predictions(training_data)

        # Load or compute embeddings
        embeddings, question_ids, predictions, mc_answers = cls._load_embeddings_and_labels(
            config.embeddings_path, training_data, train_df
        )

        # Create model with config
        model = QuestionSpecificMLP(question_predictions, config=config)
        model = model.to(device)

        # Log model info to wandb
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_questions": len(question_predictions),
            "total_samples": len(embeddings),
        })

        # Split data using shared utility
        n_samples = len(embeddings)
        train_indices, val_indices, test_indices = get_split_indices(
            n_samples, train_ratio=config.train_split, random_seed=config.random_seed
        )

        # Update wandb with actual dataset sizes
        wandb.config.update({
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
        })

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

        # Training setup - choose optimizer based on config
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Setup scheduler if specified
        scheduler = None
        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif config.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.learning_rate * 10,
                epochs=config.epochs,
                steps_per_epoch=len(train_loader),
            )

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 0

        # Load checkpoint if resuming
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            checkpoint = cls.load_checkpoint(resume_from_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            logger.info(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
            wandb.log({"resumed_from_epoch": start_epoch})

        # Training loop with interrupt handling
        try:
            for epoch in range(start_epoch, config.epochs):
                # Training
                avg_train_loss = cls._train_epoch(model, train_loader, optimizer, criterion, device, scheduler)

                # Validation
                avg_val_loss = cls._validate_epoch(model, val_loader, criterion, device)

                logger.info(
                    f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

                # Step scheduler if using one
                if scheduler is not None and config.scheduler != "onecycle":
                    scheduler.step()  # OneCycle steps per batch, not epoch

                # Early stopping and checkpoint saving
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Log best validation loss
                    wandb.log({"best_val_loss": best_val_loss})

                    # Save checkpoint
                    checkpoint_path = config.checkpoint_dir / f"mlp_best_{config.wandb_run_name or 'model'}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "train_loss": avg_train_loss,
                        "config": config,
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        wandb.log({"early_stopped_epoch": epoch + 1})
                        break
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user!")
            wandb.log({"interrupted": True})

        # Load best checkpoint if it exists
        checkpoint_path = config.checkpoint_dir / f"mlp_best_{config.wandb_run_name or 'model'}.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading best model from checkpoint: {checkpoint_path}")
            checkpoint = cls.load_checkpoint(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

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
        # Determine if answer is correct
        is_correct = evaluation_row.mc_answer == self.correct_answers.get(evaluation_row.question_id, "")
        is_correct_tensor = torch.tensor([[1.0 if is_correct else 0.0]], dtype=torch.float32).to(self.device)

        # Generate embedding for the evaluation row using only answer and explanation
        text = f"Answer: {evaluation_row.mc_answer}; Explanation: {evaluation_row.student_explanation}"
        embedding = self.tokenizer.encode(text)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)

        # Concatenate correctness feature to embedding
        enhanced_embedding = torch.cat([embedding_tensor, is_correct_tensor], dim=1)

        question_tensor = torch.LongTensor([evaluation_row.question_id]).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(enhanced_embedding, question_tensor, is_correct_tensor)

            prediction_strings = []

            # Check which head was used based on correctness
            key = (evaluation_row.question_id, is_correct)
            if key in outputs:
                logits = outputs[key]
                probabilities = functional.softmax(logits, dim=-1)

                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probabilities[0], k=min(3, probabilities.size(-1)))

                # Get appropriate label encoder
                if is_correct:
                    label_encoder = self.model.true_label_encoders.get(evaluation_row.question_id)
                else:
                    label_encoder = self.model.false_label_encoders.get(evaluation_row.question_id)

                if label_encoder:
                    prediction_strings = label_encoder.inverse_transform(top_indices.cpu().numpy())

            # Fallback if no predictions
            if prediction_strings is None or len(prediction_strings) == 0:
                if is_correct:
                    prediction_strings = ["True_Correct:NA", "True_Neither:NA", "True_Neither:NA"]
                else:
                    prediction_strings = ["False_Neither:NA", "False_Misconception:Incomplete", "False_Neither:NA"]

        # Parse prediction strings back to Prediction objects
        predictions = []
        for pred_str in prediction_strings[:3]:
            try:
                prediction = Prediction.from_string(pred_str)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to parse prediction string '{pred_str}': {e}")
                # Fallback prediction based on correctness
                if is_correct:
                    predictions.append(Prediction(category=Category.TRUE_NEITHER, misconception="NA"))
                else:
                    predictions.append(Prediction(category=Category.FALSE_NEITHER, misconception="NA"))

        # Ensure we have exactly 3 predictions
        while len(predictions) < 3:
            if is_correct:
                predictions.append(Prediction(category=Category.TRUE_NEITHER, misconception="NA"))
            else:
                predictions.append(Prediction(category=Category.FALSE_NEITHER, misconception="NA"))

        return SubmissionRow(row_id=evaluation_row.row_id, predicted_categories=predictions)

    def save(self, filepath: Path) -> None:
        """Save fitted model to disk."""
        logger.info(f"Saving MLP model to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save entire strategy object
        with filepath.open("wb") as f:
            pickle.dump(self, f)

        # Save parameters if available
        if self.parameters:
            params_path = filepath.with_suffix(".params.json")
            logger.info(f"Saving model parameters to {params_path}")
            with params_path.open("w") as f:
                json.dump(self.parameters.model_dump(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "MLPStrategy":
        """Load fitted model from disk."""
        logger.info(f"Loading MLP model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        with filepath.open("rb") as f:
            loaded_model = pickle.load(f)

        # Try to load parameters if they exist
        params_path = filepath.with_suffix(".params.json")
        parameters = None
        if params_path.exists():
            logger.info(f"Loading model parameters from {params_path}")
            with params_path.open("r") as f:
                params_data = json.load(f)
                parameters = ModelParameters.model_validate(params_data)
        else:
            logger.warning(f"No parameters file found at {params_path}")
            # Use the parameters from the loaded model if available
            parameters = loaded_model.parameters if hasattr(loaded_model, "parameters") else None

        # If parameters were loaded separately, create a new instance with updated parameters
        # Otherwise return the loaded model as-is
        if parameters and parameters != loaded_model.parameters:
            return cls(
                model=loaded_model.model,
                correct_answers=loaded_model.correct_answers,
                tokenizer=loaded_model.tokenizer,
                device=loaded_model.device,
                parameters=parameters,
            )

        return loaded_model

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
        """Evaluate model on validation split using MAP@3 metric.

        Can either evaluate a provided model or load from a checkpoint.
        If neither is provided, looks for the latest checkpoint.
        """

        # Handle checkpoint loading if model not provided
        if model is None:
            if checkpoint_path is None:
                # Find the latest checkpoint
                checkpoint_dir = Path("checkpoints")
                checkpoints = list(checkpoint_dir.glob("mlp_best_*.pt"))
                if not checkpoints:
                    msg = "No model provided and no checkpoints found!"
                    raise ValueError(msg)
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using latest checkpoint: {checkpoint_path}")

            # Load checkpoint and create model
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            config = checkpoint["config"]

            # Recreate model architecture
            training_data = parse_training_data(train_csv_path)
            question_predictions = cls.extract_question_predictions(training_data)
            correct_answers = extract_correct_answers(training_data)

            # Create and load model with correct configuration
            mlp_model = QuestionSpecificMLP(question_predictions, config=config)
            mlp_model.load_state_dict(checkpoint["model_state_dict"])
            device = get_device()
            mlp_model = mlp_model.to(device)

            # Create strategy instance
            tokenizer = get_tokenizer()
            n_samples = len(training_data)
            train_indices, val_indices, test_indices = get_split_indices(
                n_samples, train_ratio=config.train_split, random_seed=config.random_seed
            )
            parameters = ModelParameters.create(
                train_split=config.train_split,
                random_seed=config.random_seed,
                train_indices=list(train_indices),
                val_indices=list(val_indices),
                test_indices=list(test_indices),
                total_samples=n_samples,
            )

            model = cls(
                model=mlp_model,
                correct_answers=correct_answers,
                tokenizer=tokenizer,
                device=device,
                parameters=parameters,
            )

            logger.info(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

            # Use checkpoint's training config for evaluation
            train_split = config.train_split
            random_seed = config.random_seed

        logger.info("Evaluating MLP model on validation split")
        logger.info(f"Using train_split={train_split}, random_seed={random_seed}")

        # Parse and split data
        training_data = parse_training_data(train_csv_path)
        train_data, val_data, test_data = split_training_data(
            training_data, train_ratio=train_split, random_seed=random_seed
        )

        logger.info(f"Evaluating on {len(val_data)} validation samples")

        # Calculate MAP@3 for each validation sample
        map_scores = []
        for row in val_data:
            # Create evaluation row (without ground truth)
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

            # Calculate MAP@3
            ground_truth = row.prediction
            score = calculate_map_at_3(ground_truth, predictions)
            map_scores.append(score)

        # Calculate average MAP@3
        avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0

        results = {
            "validation_map@3": avg_map,
            "validation_samples": len(val_data),
        }

        logger.info(f"Validation MAP@3: {avg_map:.4f} on {len(val_data)} samples")

        return results


if __name__ == "__main__":
    """Quick test of MLP training and prediction functionality."""
    import random
    from pathlib import Path

    import pandas as pd

    from kaggle_map.core.dataset import parse_training_data
    from kaggle_map.core.models import EvaluationRow

    # Load existing model
    model_path = Path("models/mlp.pkl")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        sys.exit(1)

    model = MLPStrategy.load(model_path)

    # Load training data to get a sample row
    train_csv_path = Path("datasets/train.csv")
    train_df = pd.read_csv(train_csv_path)
    training_data = parse_training_data(train_csv_path)

    # Pick a random row for testing prediction
    random_idx = random.randint(0, len(train_df) - 1)
    sample_row = train_df.iloc[random_idx]

    logger.info(f"Selected random row {sample_row['row_id']} for prediction test")
    logger.info(f"Question ID: {sample_row['QuestionId']}")
    logger.info(f"MC Answer: {sample_row['MC_Answer']}")

    # Create evaluation row
    eval_row = EvaluationRow(
        row_id=sample_row["row_id"],
        question_id=sample_row["QuestionId"],
        question_text=sample_row["QuestionText"],
        mc_answer=sample_row["MC_Answer"],
        student_explanation=sample_row["StudentExplanation"],
    )

    # Make prediction
    logger.info("Making prediction...")
    submission = model.predict(eval_row)

    # Get ground truth for comparison if available
    matching_training_rows = [r for r in training_data if r.row_id == sample_row["row_id"]]
    if matching_training_rows:
        ground_truth = matching_training_rows[0]
        logger.info(f"Ground truth: {ground_truth.prediction.category.value}:{ground_truth.misconception}")

        # Calculate MAP@3 score
        from kaggle_map.core.metrics import calculate_map_at_3
        map_score = calculate_map_at_3(ground_truth.prediction, submission.predicted_categories)
        logger.info(f"MAP@3 score: {map_score:.4f}")

    # Display predictions
    logger.info(f"Predictions for row {submission.row_id}:")
    for i, pred in enumerate(submission.predicted_categories, 1):
        logger.info(f"  {i}. {pred.category.value}:{pred.misconception}")




