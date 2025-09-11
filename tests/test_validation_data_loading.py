"""Test that validation data loading correctly retrieves misconceptions from train.csv."""

from pathlib import Path

import pandas as pd


def test_row_27118_misconception():
    """Verify that row_id 27118 has Subtraction misconception in train.csv."""
    # Load the training data
    train_path = Path("datasets/train.csv")
    assert train_path.exists(), f"Training data not found at {train_path}"

    train_df = pd.read_csv(train_path)

    # Find row with row_id 27118
    row_27118 = train_df[train_df["row_id"] == 27118]
    assert len(row_27118) == 1, f"Expected exactly 1 row with row_id 27118, found {len(row_27118)}"

    # Check the misconception
    misconception = row_27118.iloc[0]["Misconception"]
    print(f"Row 27118 Misconception in train.csv: {misconception}")

    # According to you, it should be "Subtraction"
    assert pd.notna(misconception), "Misconception should not be NaN"
    assert misconception == "Subtraction", f"Expected 'Subtraction', got '{misconception}'"

    # Also check the category
    category = row_27118.iloc[0]["Category"]
    print(f"Row 27118 Category in train.csv: {category}")

    return misconception, category


def test_validation_data_structure():
    """Check the structure of validation data to understand what's available."""
    validation_path = Path("datasets/33474_validation.csv")
    assert validation_path.exists(), f"Validation data not found at {validation_path}"

    val_df = pd.read_csv(validation_path)

    print(f"\nValidation data shape: {val_df.shape}")
    print(f"Validation columns: {list(val_df.columns)}")

    # Check if row_id 27118 exists in validation data
    if "row_id" in val_df.columns:
        row_27118_val = val_df[val_df["row_id"] == 27118]
        if len(row_27118_val) > 0:
            print("\nRow 27118 in validation data:")
            print(row_27118_val.iloc[0].to_dict())

    # Check first few rows to understand structure
    print("\nFirst 3 rows of validation data:")
    for idx, row in val_df.head(3).iterrows():
        print(f"Row {idx}: {row.to_dict()}")

    return val_df


def test_load_validation_data_function():
    """Test the load_validation_data function to see what it returns."""
    from kaggle_map.dataloader import load_validation_data

    validation_path = Path("datasets/33474_validation.csv")
    validation_pairs = load_validation_data(validation_path)

    print(f"\nLoaded {len(validation_pairs)} validation pairs")

    # Find the pair for row_id 27118
    for eval_row, ground_truth in validation_pairs:
        if eval_row.row_id == 27118:
            print("\nFound row 27118:")
            print(f"  EvalRow: {eval_row}")
            print(f"  Ground Truth Category: {ground_truth.category}")
            print(f"  Ground Truth Misconception: {ground_truth.misconception}")
            return eval_row, ground_truth

    print("Row 27118 not found in validation pairs")
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Row 27118 Misconception Loading")
    print("=" * 60)

    # Test 1: Check train.csv
    print("\n1. Checking train.csv for row 27118:")
    try:
        misconception, category = test_row_27118_misconception()
        print(f"✓ Row 27118 has misconception: {misconception}, category: {category}")
    except AssertionError as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Test 2: Check validation data structure
    print("\n2. Checking validation data structure:")
    try:
        val_df = test_validation_data_structure()
        print("✓ Validation data loaded successfully")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Check load_validation_data function
    print("\n3. Testing load_validation_data function:")
    try:
        result = test_load_validation_data_function()
        if result:
            print("✓ Found row 27118 in validation data")
        else:
            print("✗ Row 27118 not found in validation data")
    except Exception as e:
        print(f"✗ Error: {e}")
