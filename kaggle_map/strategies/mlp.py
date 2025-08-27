"""MLP neural network strategy for student misconception prediction.

Architecture Design:
-------------------
Shared Trunk:
  - Input: embedding(384) from sentence-transformers/all-MiniLM-L6-v2
  - Hidden 1: 768 units, ReLU, 0.3 dropout
  - Hidden 2: 384 units, ReLU, 0.3 dropout
  - Shared: 192 units, ReLU, 0.3 dropout

Question-Specific Misconception Heads (nn.ModuleDict):
  - One head per question ID (e.g., Q31772, Q31774, etc.)
  - Each head outputs N classes where N = unique misconceptions + NA
  - Examples:
    - Q31772: 3 outputs → [Incomplete, WNB, NA]
    - Q31774: 4 outputs → [Mult, SwapDividend, FlipChange, NA]

Key Design Decisions:
--------------------
1. NA Handling: "NA" is treated as a separate class in multi-class classification
2. Loss Function: CrossEntropyLoss for multi-class classification per question head
3. Loss Aggregation: Average loss across all question heads during training
4. Optimizer: Adam with learning rate 1e-3
5. Early Stopping: Based on validation loss with patience of 10 epochs
6. Data Split: 70% train, 15% validation, 15% test (configurable)
7. Embeddings: Precomputed 384-dim embeddings from MiniLM stored at datasets/train_embeddings.npz

Training Process:
----------------
1. Load precomputed embeddings or compute on-the-fly
2. Create question-specific label encoders for misconception mapping
3. Train with mini-batches, grouping by question ID for head-specific loss
4. Track progress with Weights & Biases (wandb) integration
5. Apply early stopping based on validation loss
6. Save model with pickle and parameters as JSON
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import wandb
from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    parse_training_data,
)
from kaggle_map.core.embeddings.tokenizer import get_tokenizer
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    Answer,
    Category,
    EvaluationRow,
    Misconception,
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
    batch_size: int = 128
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    wandb_project: str = "kaggle-map-mlp"
    wandb_run_name: str | None = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific misconception heads."""

    def __init__(self, question_misconceptions: dict[QuestionId, list[Misconception]]) -> None:
        super().__init__()

        # Shared trunk layers
        self.trunk = nn.Sequential(
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Question-specific heads
        self.question_heads = nn.ModuleDict()
        self.question_label_encoders = {}

        for question_id, misconceptions in question_misconceptions.items():
            # Always include NA as a class
            unique_misconceptions = sorted(set(misconceptions) | {"NA"})
            num_classes = len(unique_misconceptions)

            # Create head for this question
            self.question_heads[str(question_id)] = nn.Linear(192, num_classes)

            # Create label encoder for this question
            label_encoder = LabelEncoder()
            label_encoder.fit(unique_misconceptions)
            self.question_label_encoders[question_id] = label_encoder

    def forward(self, x: torch.Tensor, question_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass returning logits per question."""
        # Pass through shared trunk
        shared_features = self.trunk(x)

        # Group by question and apply appropriate head
        outputs = {}
        unique_questions = torch.unique(question_ids)

        for qid in unique_questions:
            qid_int = int(qid.item())
            mask = question_ids == qid

            if str(qid_int) in self.question_heads:
                question_features = shared_features[mask]
                outputs[qid_int] = self.question_heads[str(qid_int)](question_features)

        return outputs


class MLPDataset(Dataset):
    """Dataset for MLP training."""

    def __init__(
        self,
        embeddings: np.ndarray,
        question_ids: np.ndarray,
        misconceptions: np.ndarray,
        label_encoders: dict[QuestionId, LabelEncoder],
    ) -> None:
        self.embeddings = torch.FloatTensor(embeddings)
        self.question_ids = torch.LongTensor(question_ids)
        self.misconceptions = misconceptions
        self.label_encoders = label_encoders

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qid = int(self.question_ids[idx].item())
        misconception = self.misconceptions[idx]

        # Encode the misconception for this question
        label = self.label_encoders[qid].transform([misconception])[0] if qid in self.label_encoders else 0

        return self.embeddings[idx], self.question_ids[idx], torch.tensor(label, dtype=torch.long)


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
        return "MLP with shared trunk and question-specific heads for misconception prediction"

    @staticmethod
    def _load_embeddings_and_labels(
        embeddings_path: Path | None,
        training_data: list,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load or compute embeddings and associated labels."""
        if embeddings_path and embeddings_path.exists():
            logger.info(f"Loading precomputed embeddings from {embeddings_path}")
            data = np.load(embeddings_path)
            embeddings = data["embeddings"]
            row_ids = data["row_ids"]
            misconceptions = data["misconceptions"]
            # Get question IDs from training CSV
            question_ids = train_df.set_index("row_id").loc[row_ids]["QuestionId"].to_numpy()
        else:
            logger.info("Computing embeddings from scratch")
            tokenizer = get_tokenizer()
            embeddings = []
            question_ids = []
            misconceptions = []
            for row in training_data:
                text = repr(row)
                embedding = tokenizer.encode(text)
                embeddings.append(embedding)
                question_ids.append(row.question_id)
                misconceptions.append(row.misconception)
            embeddings = np.array(embeddings)
            question_ids = np.array(question_ids)
            misconceptions = np.array(misconceptions)
        return embeddings, question_ids, misconceptions

    @staticmethod
    def _train_epoch(
        model: QuestionSpecificMLP,
        train_loader: DataLoader,
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss,
        device: str,
    ) -> float:
        """Train for one epoch and return average loss."""
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_embeds, batch_quests, batch_lbls in train_loader:
            batch_embeddings = batch_embeds.to(device)
            batch_questions = batch_quests.to(device)
            batch_labels = batch_lbls.to(device)

            optimizer.zero_grad()
            outputs = model(batch_embeddings, batch_questions)

            # Calculate loss per question and aggregate
            loss = 0.0
            for qid, logits in outputs.items():
                mask = batch_questions == qid
                if mask.any():
                    question_labels = batch_labels[mask]
                    loss += criterion(logits, question_labels)

            if len(outputs) > 0:
                loss = loss / len(outputs)  # Average across questions
                loss.backward()
                optimizer.step()
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
            for batch_embeds, batch_quests, batch_lbls in val_loader:
                batch_embeddings = batch_embeds.to(device)
                batch_questions = batch_quests.to(device)
                batch_labels = batch_lbls.to(device)

                outputs = model(batch_embeddings, batch_questions)

                loss = 0.0
                for qid, logits in outputs.items():
                    mask = batch_questions == qid
                    if mask.any():
                        question_labels = batch_labels[mask]
                        loss += criterion(logits, question_labels)

                if len(outputs) > 0:
                    loss = loss / len(outputs)
                    val_loss += loss.item() * batch_embeddings.size(0)
                    val_samples += batch_embeddings.size(0)

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
        **kwargs: object,
    ) -> "MLPStrategy":
        """Fit the MLP strategy on training data."""
        # Create config with defaults and overrides
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

        # Extract correct answers and misconceptions
        correct_answers = extract_correct_answers(training_data)
        question_misconceptions = extract_misconceptions_by_popularity(training_data)

        # Load or compute embeddings
        embeddings, question_ids, misconceptions = cls._load_embeddings_and_labels(
            config.embeddings_path, training_data, train_df
        )

        # Create model
        model = QuestionSpecificMLP(question_misconceptions)
        model = model.to(device)

        # Log model info to wandb
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_questions": len(question_misconceptions),
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
            misconceptions[train_indices],
            model.question_label_encoders,
        )

        val_dataset = MLPDataset(
            embeddings[val_indices],
            question_ids[val_indices],
            misconceptions[val_indices],
            model.question_label_encoders,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Training setup
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
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
                avg_train_loss = cls._train_epoch(model, train_loader, optimizer, criterion, device)

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
                    "learning_rate": config.learning_rate,
                })

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
        logger.debug(f"Making MLP prediction for row {evaluation_row.row_id}")

        # Generate embedding for the evaluation row
        text = repr(evaluation_row)
        embedding = self.tokenizer.encode(text)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        question_tensor = torch.LongTensor([evaluation_row.question_id]).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(embedding_tensor, question_tensor)

            if evaluation_row.question_id in outputs:
                logits = outputs[evaluation_row.question_id]
                probabilities = functional.softmax(logits, dim=-1)

                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probabilities[0], k=min(3, probabilities.size(-1)))

                # Convert to misconceptions
                label_encoder = self.model.question_label_encoders.get(evaluation_row.question_id)
                if label_encoder:
                    misconceptions = label_encoder.inverse_transform(top_indices.cpu().numpy())
                else:
                    misconceptions = ["NA", "NA", "NA"]
            else:
                # Question not seen in training
                misconceptions = ["NA", "NA", "NA"]

        # Determine if answer is correct
        is_correct = evaluation_row.mc_answer == self.correct_answers.get(evaluation_row.question_id, "")

        # Create predictions with appropriate categories
        predictions = []
        for misconception in misconceptions[:3]:
            if misconception == "NA":
                category = Category.TRUE_NEITHER if is_correct else Category.FALSE_NEITHER
            else:
                category = Category.TRUE_MISCONCEPTION if is_correct else Category.FALSE_MISCONCEPTION
            predictions.append(Prediction(category=category, misconception=misconception))

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
            parameters = loaded_model.parameters if hasattr(loaded_model, 'parameters') else None

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
            question_misconceptions = extract_misconceptions_by_popularity(training_data)
            correct_answers = extract_correct_answers(training_data)

            # Create and load model
            mlp_model = QuestionSpecificMLP(question_misconceptions)
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
    """Quick test of MLP training with wandb visualization."""
    from pathlib import Path

    print("🚀 Training MLP with wandb visualization...")
    print("=" * 60)

    # Train with fewer epochs for quick testing
    model = MLPStrategy.fit(
        epochs=5,  # Quick training for testing
        wandb_project="kaggle-map-mlp",
        wandb_run_name="mlp-quicktest",
    )

    # Save the model
    model_path = Path("models/mlp.pkl")
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Quick evaluation
    print("\n📊 Evaluating model on validation split...")
    results = MLPStrategy.evaluate_on_split(model)
    print(f"Validation MAP@3: {results['validation_map@3']:.4f}")
    print(f"Validation samples: {results['validation_samples']}")

    print("\n🎉 Training complete! Check your wandb dashboard for visualizations.")
    print("Dashboard: https://wandb.ai/")
    print("=" * 60)
