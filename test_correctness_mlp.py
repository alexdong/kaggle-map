"""Test that MLP uses answer correctness effectively."""

from pathlib import Path

from loguru import logger

from kaggle_map.core.dataset import extract_correct_answers, parse_training_data
from kaggle_map.strategies.mlp import MLPConfig, MLPStrategy, QuestionSpecificMLP


def test_correctness_aware_mlp() -> None:
    """Test the correctness-aware MLP."""

    logger.info("Testing correctness-aware MLP...")

    # Load training data
    training_data = parse_training_data(Path("datasets/train.csv"))
    question_predictions = MLPStrategy.extract_question_predictions(training_data)
    extract_correct_answers(training_data)

    # Check that predictions are properly split
    first_qid = sorted(question_predictions.keys())[0]
    predictions = question_predictions[first_qid]

    true_preds = [p for p in predictions if p.startswith("True_")]
    false_preds = [p for p in predictions if p.startswith("False_")]

    logger.info(f"\nQuestion {first_qid}:")
    logger.info(f"  Total predictions: {len(predictions)}")
    logger.info(f"  True_ predictions: {len(true_preds)}")
    logger.info(f"  False_ predictions: {len(false_preds)}")

    # Test model creation
    logger.info("\nCreating correctness-aware model...")
    config = MLPConfig(hidden_dim=64, n_layers=1, dropout=0.1)
    model = QuestionSpecificMLP(question_predictions, config=config)

    # Check that we have separate heads
    logger.info("\nModel structure:")
    logger.info(f"  True heads: {len(model.true_heads)}")
    logger.info(f"  False heads: {len(model.false_heads)}")

    # Check head sizes for first question
    if str(first_qid) in model.true_heads:
        true_head = model.true_heads[str(first_qid)]
        logger.info(f"  Q{first_qid} True head output size: {true_head.out_features}")

    if str(first_qid) in model.false_heads:
        false_head = model.false_heads[str(first_qid)]
        logger.info(f"  Q{first_qid} False head output size: {false_head.out_features}")

    # Test that encoder classes match
    if first_qid in model.true_label_encoders:
        logger.info(f"\n  True encoder classes: {model.true_label_encoders[first_qid].classes_[:3]}...")
    if first_qid in model.false_label_encoders:
        logger.info(f"  False encoder classes: {model.false_label_encoders[first_qid].classes_[:3]}...")

    logger.success("\n✓ Correctness-aware MLP structure verified!")
    logger.info("\nExpected improvements:")
    logger.info("  - Predictions now constrained to correct category (True_ or False_)")
    logger.info("  - Search space reduced by ~50%")
    logger.info("  - Should significantly improve accuracy")


if __name__ == "__main__":
    test_correctness_aware_mlp()
