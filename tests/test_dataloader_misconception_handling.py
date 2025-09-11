"""Test that the dataloader correctly handles misconception columns in both train and validation CSVs."""

from pathlib import Path

import pandas as pd
import pytest

from kaggle_map.core.models import Category
from kaggle_map.dataloader import load_validation_data


def test_load_train_csv_misconception_column() -> None:
    """Test that train.csv with 'Misconception' column loads correctly."""
    train_path = Path("datasets/33474_train.csv")
    if not train_path.exists():
        pytest.skip(f"Training data not found at {train_path}")

    # Load validation data (which works for both train and validation CSVs)
    validation_pairs = load_validation_data(train_path)

    # Find row 27118 specifically
    row_27118_found = False
    for eval_row, ground_truth in validation_pairs:
        if eval_row.row_id == 27118:
            row_27118_found = True

            # Verify the misconception is loaded correctly
            assert ground_truth.misconception == "Subtraction", (
                f"Row 27118 should have 'Subtraction' misconception, got '{ground_truth.misconception}'"
            )

            # Also check the category
            assert ground_truth.category == Category.FALSE_MISCONCEPTION, (
                f"Row 27118 should be FALSE_MISCONCEPTION, got {ground_truth.category}"
            )

            print("✓ Row 27118 correctly loaded:")
            print(f"  - Misconception: {ground_truth.misconception}")
            print(f"  - Category: {ground_truth.category.value}")
            break

    assert row_27118_found, "Row 27118 not found in train.csv"


def test_load_validation_csv_actual_misconception_column() -> None:
    """Test that validation.csv with 'actual_misconception' column loads correctly."""
    validation_path = Path("datasets/33474_validation.csv")
    if not validation_path.exists():
        pytest.skip(f"Validation data not found at {validation_path}")

    # Load validation data
    validation_pairs = load_validation_data(validation_path)

    # Check that we can load data successfully
    assert len(validation_pairs) > 0, "Should load validation data"

    # Check a few rows to ensure misconceptions are loaded
    misconception_found = False
    na_found = False

    for _eval_row, ground_truth in validation_pairs[:100]:  # Check first 100 rows
        if ground_truth.misconception != "NA":
            misconception_found = True
        else:
            na_found = True

        # Every row should have a valid category
        assert ground_truth.category in Category, f"Invalid category: {ground_truth.category}"

    print("✓ Validation CSV loaded successfully:")
    print(f"  - Found misconceptions: {misconception_found}")
    print(f"  - Found NA values: {na_found}")


def test_both_column_names_handled() -> None:
    """Test that dataloader handles both 'Misconception' and 'actual_misconception' columns."""
    # Create test dataframes with different column names

    # Test with 'Misconception' column (train.csv format)
    train_format_data = pd.DataFrame({
        "row_id": [1, 2, 3],
        "QuestionId": [100, 100, 100],
        "QuestionText": ["Q1", "Q2", "Q3"],
        "MC_Answer": ["A", "B", "C"],
        "StudentExplanation": ["Exp1", "Exp2", "Exp3"],
        "Category": ["TRUE_MISCONCEPTION", "FALSE_CORRECT", "TRUE_NEITHER"],
        "Misconception": ["Division", "NA", "NA"]
    })

    # Test with 'actual_misconception' column (validation.csv format)
    validation_format_data = pd.DataFrame({
        "row_id": [4, 5, 6],
        "QuestionId": [101, 101, 101],
        "QuestionText": ["Q4", "Q5", "Q6"],
        "MC_Answer": ["D", "E", "F"],
        "StudentExplanation": ["Exp4", "Exp5", "Exp6"],
        "Category": ["FALSE_MISCONCEPTION", "TRUE_CORRECT", "FALSE_NEITHER"],
        "actual_misconception": ["Subtraction", "NA", "NA"]
    })

    # Save temporary test files
    train_test_path = Path("/tmp/test_train.csv")
    validation_test_path = Path("/tmp/test_validation.csv")

    train_format_data.to_csv(train_test_path, index=False)
    validation_format_data.to_csv(validation_test_path, index=False)

    try:
        # Load both formats
        train_pairs = load_validation_data(train_test_path)
        validation_pairs = load_validation_data(validation_test_path)

        # Check train format
        assert len(train_pairs) == 3
        eval_row, ground_truth = train_pairs[0]
        assert eval_row.row_id == 1
        assert ground_truth.misconception == "Division"
        assert ground_truth.category == Category.TRUE_MISCONCEPTION

        # Check validation format
        assert len(validation_pairs) == 3
        eval_row, ground_truth = validation_pairs[0]
        assert eval_row.row_id == 4
        assert ground_truth.misconception == "Subtraction"
        assert ground_truth.category == Category.FALSE_MISCONCEPTION

        print("✓ Both column formats handled correctly")

    finally:
        # Clean up test files
        train_test_path.unlink(missing_ok=True)
        validation_test_path.unlink(missing_ok=True)


def test_missing_misconception_defaults_to_na() -> None:
    """Test that missing misconception values default to 'NA'."""
    # Create test data with missing misconception
    test_data = pd.DataFrame({
        "row_id": [1],
        "QuestionId": [100],
        "QuestionText": ["Question"],
        "MC_Answer": ["A"],
        "StudentExplanation": ["Explanation"],
        "Category": ["TRUE_CORRECT"],
        # No Misconception or actual_misconception column
    })

    test_path = Path("/tmp/test_missing_misconception.csv")
    test_data.to_csv(test_path, index=False)

    try:
        pairs = load_validation_data(test_path)
        assert len(pairs) == 1

        _eval_row, ground_truth = pairs[0]
        assert ground_truth.misconception == "NA", "Missing misconception should default to 'NA'"
        assert ground_truth.category == Category.TRUE_CORRECT

        print("✓ Missing misconceptions correctly default to 'NA'")

    finally:
        test_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("Testing dataloader misconception handling...")
    print("=" * 60)

    # Run all tests
    test_load_train_csv_misconception_column()
    test_load_validation_csv_actual_misconception_column()
    test_both_column_names_handled()
    test_missing_misconception_defaults_to_na()

    print("=" * 60)
    print("All tests passed! ✅")
