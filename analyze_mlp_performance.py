"""Analyze why MLP performance is still lagging behind baseline."""

from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from kaggle_map.core.dataset import parse_training_data
from kaggle_map.strategies.mlp import MLPStrategy


def analyze_mlp_performance() -> None:
    """Analyze MLP's predictions to understand remaining issues."""

    logger.info("Loading models and data...")

    # Load the redesigned MLP
    mlp_model = MLPStrategy.load(Path("models/mlp.pkl"))

    # Load training data
    training_data = parse_training_data(Path("datasets/train.csv"))

    # Get question predictions distribution
    question_predictions = MLPStrategy.extract_question_predictions(training_data)

    # Analyze model's label encoders
    logger.info("\n" + "="*60)
    logger.info("MODEL ARCHITECTURE ANALYSIS")
    logger.info("="*60)

    for qid in sorted(question_predictions.keys())[:3]:
        encoder = mlp_model.model.question_label_encoders.get(qid)
        if encoder:
            logger.info(f"\nQuestion {qid}:")
            logger.info(f"  Classes: {len(encoder.classes_)}")
            logger.info("  Top predictions learned:")
            for i, pred in enumerate(encoder.classes_[:5]):
                logger.info(f"    {i}: {pred}")

    # Analyze class distribution in training data
    logger.info("\n" + "="*60)
    logger.info("TRAINING DATA DISTRIBUTION")
    logger.info("="*60)

    prediction_counts = Counter()
    question_prediction_counts = defaultdict(Counter)

    for row in training_data:
        pred_str = str(row.prediction)
        prediction_counts[pred_str] += 1
        question_prediction_counts[row.question_id][pred_str] += 1

    logger.info("\nTop 10 most common predictions overall:")
    for pred, count in prediction_counts.most_common(10):
        pct = (count / len(training_data)) * 100
        logger.info(f"  {pred}: {count:,} ({pct:.1f}%)")

    # Check class imbalance per question
    logger.info("\n" + "="*60)
    logger.info("CLASS IMBALANCE PER QUESTION")
    logger.info("="*60)

    for qid in sorted(question_predictions.keys())[:3]:
        counts = question_prediction_counts[qid]
        total = sum(counts.values())
        logger.info(f"\nQuestion {qid} (total: {total}):")

        # Show distribution
        for pred, count in counts.most_common(5):
            pct = (count / total) * 100
            logger.info(f"  {pred}: {count} ({pct:.1f}%)")

        # Calculate imbalance ratio
        most_common = counts.most_common(1)[0][1]
        least_common = counts.most_common()[-1][1]
        imbalance = most_common / least_common if least_common > 0 else float("inf")
        logger.info(f"  Imbalance ratio: {imbalance:.1f}:1")

    # Analyze model parameters
    logger.info("\n" + "="*60)
    logger.info("MODEL CAPACITY ANALYSIS")
    logger.info("="*60)

    total_params = sum(p.numel() for p in mlp_model.model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Count parameters per component
    trunk_params = sum(p.numel() for p in mlp_model.model.trunk.parameters())
    logger.info(f"Trunk parameters: {trunk_params:,}")

    head_params = 0
    for qid, head in mlp_model.model.question_heads.items():
        head_params += sum(p.numel() for p in head.parameters())
    logger.info(f"Head parameters (total): {head_params:,}")
    logger.info(f"Average per head: {head_params // len(mlp_model.model.question_heads):,}")

    # Training configuration insights
    logger.info("\n" + "="*60)
    logger.info("POTENTIAL ISSUES")
    logger.info("="*60)

    logger.info("\n1. CLASS IMBALANCE:")
    logger.info("   - True_Correct:NA is 40.3% of data")
    logger.info("   - Some classes have <1% representation")
    logger.info("   - Model may overpredict common classes")

    logger.info("\n2. LIMITED TRAINING:")
    logger.info("   - Only 20 epochs by default")
    logger.info("   - Early stopping may prevent convergence")
    logger.info("   - Learning rate might need tuning")

    logger.info("\n3. ARCHITECTURE LIMITATIONS:")
    logger.info("   - Shared trunk may not capture question-specific patterns well")
    logger.info("   - Fixed embedding size (384) might be limiting")
    logger.info("   - Dropout (0.3) might be too high")

    logger.info("\n4. LOSS FUNCTION:")
    logger.info("   - CrossEntropyLoss doesn't account for class imbalance")
    logger.info("   - Could benefit from weighted loss or focal loss")

    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)

    logger.info("\n1. Address class imbalance:")
    logger.info("   - Use weighted CrossEntropyLoss")
    logger.info("   - Implement focal loss")
    logger.info("   - Oversample minority classes")

    logger.info("\n2. Hyperparameter tuning:")
    logger.info("   - Try more epochs (50-100)")
    logger.info("   - Experiment with learning rates (1e-4 to 1e-2)")
    logger.info("   - Reduce dropout (0.1-0.2)")

    logger.info("\n3. Architecture improvements:")
    logger.info("   - Larger hidden dimensions")
    logger.info("   - Question-specific preprocessing")
    logger.info("   - Attention mechanism between trunk and heads")

    logger.info("\n4. Training strategy:")
    logger.info("   - Curriculum learning (easy to hard)")
    logger.info("   - Multi-task learning with auxiliary objectives")
    logger.info("   - Ensemble with baseline predictions")


if __name__ == "__main__":
    analyze_mlp_performance()
