#!/usr/bin/env python3
"""Test script to verify MLP dimension mismatch fixes."""

import sys
from pathlib import Path

from loguru import logger

from kaggle_map.strategies.mlp import MLPStrategy


def test_mlp_training() -> bool | None:
    """Test that MLP training works without dimension mismatches."""
    print("Testing MLP training with dimension mismatch fixes...")
    
    # Use a small subset of training data for quick testing
    train_csv_path = Path("datasets/train.csv")
    if not train_csv_path.exists():
        print(f"Training file not found: {train_csv_path}")
        return False
        
    try:
        # Train with a small fraction of data to speed up testing
        logger.info("Starting MLP training test with dimension validation")
        strategy = MLPStrategy.fit(
            train_csv_path=train_csv_path,
            train_split=0.1,  # Use only 10% of data for quick test
            random_seed=42
        )
        
        logger.info("MLP training completed successfully - dimension mismatches are fixed!")
        
        # Test that we can make predictions
        from kaggle_map.models import EvaluationRow
        test_rows = [
            EvaluationRow(
                row_id=999,
                question_id=next(iter(strategy.question_misconceptions.keys())),
                question_text="Test question",
                mc_answer="A",
                student_explanation="Test explanation"
            )
        ]
        
        predictions = strategy.predict(test_rows)
        logger.info(f"Prediction test successful: {len(predictions)} predictions generated")
        
        return True
        
    except Exception as e:
        logger.error(f"MLP training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mlp_training()
    if success:
        print("✓ All tests passed - MLP dimension mismatches are fixed!")
    else:
        print("✗ Tests failed - dimension mismatches still exist")
        sys.exit(1)
