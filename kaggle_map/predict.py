"""Model prediction script for the Kaggle student misconception prediction competition."""

from pathlib import Path

import pandas as pd
from loguru import logger

from .models import TestRow, load_model


def make_predictions(
    model_path: Path = Path("baseline_model.json"),
    test_csv_path: Path = Path("dataset/test.csv"),
    output_path: Path = Path("submission.csv"),
) -> int:
    """Load model and make predictions on test data.

    Args:
        model_path: Path to the saved model JSON file
        test_csv_path: Path to the test CSV file
        output_path: Path where to save the submission CSV

    Returns:
        Number of predictions made
    """
    logger.info(f"Loading model from {model_path}")

    # Load the trained model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Load test data
    logger.info(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Convert to TestRow objects
    test_rows = []
    for _, row in test_df.iterrows():
        test_rows.append(
            TestRow(
                row_id=int(row["row_id"]),
                question_id=int(row["QuestionId"]),
                question_text=str(row["QuestionText"]),
                mc_answer=str(row["MC_Answer"]),
                student_explanation=str(row["StudentExplanation"]),
            )
        )

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_rows)
    logger.info(f"Generated predictions for {len(predictions)} rows")

    # Convert to submission format
    submission_data = []
    for pred in predictions:
        # Convert Prediction objects to space-separated string
        pred_strings = [str(p) for p in pred.predicted_categories]
        submission_data.append(
            {"row_id": pred.row_id, "predictions": " ".join(pred_strings)}
        )

    # Save submission file
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    return len(predictions)


def load_and_preview_test_data(test_csv_path: Path) -> pd.DataFrame:
    """Load test data and show a preview.

    Args:
        test_csv_path: Path to test CSV file

    Returns:
        Test data DataFrame
    """
    logger.info(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    print("\n=== TEST DATA PREVIEW ===")
    print(f"Total rows: {len(test_df)}")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Question IDs: {sorted(test_df['QuestionId'].unique())}")
    print("Sample rows:")
    print(test_df.head(3).to_string(index=False))
    print("===========================\n")

    return test_df


if __name__ == "__main__":
    """Load trained model and generate predictions for test data."""
    import sys
    from pathlib import Path

    # Parse command line arguments
    min_args_for_model = 1
    min_args_for_test = 2
    min_args_for_output = 3

    model_path = (
        Path(sys.argv[1])
        if len(sys.argv) > min_args_for_model
        else Path("baseline_model.json")
    )

    test_path = (
        Path(sys.argv[2])
        if len(sys.argv) > min_args_for_test
        else Path("dataset/test.csv")
    )

    output_path = (
        Path(sys.argv[3])
        if len(sys.argv) > min_args_for_output
        else Path("submission.csv")
    )

    logger.info("Running model prediction script")
    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Output: {output_path}")

    try:
        # Check if files exist
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            print("Run fit.py first to train a model")
            sys.exit(1)

        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            sys.exit(1)

        # Preview test data
        test_df = load_and_preview_test_data(test_path)

        # Make predictions
        num_predictions = make_predictions(model_path, test_path, output_path)

        # Display results
        print("\n=== PREDICTION COMPLETED ===")
        print(f"Model: {model_path}")
        print(f"Test data: {test_path} ({len(test_df)} rows)")
        print(f"Predictions: {num_predictions}")
        print(f"Output: {output_path}")

        # Validate output file
        if output_path.exists():
            submission_df = pd.read_csv(output_path)
            print(
                f"Submission file: {len(submission_df)} rows, {output_path.stat().st_size / 1024:.1f} KB"
            )

            # Show sample predictions
            print("\nSample predictions:")
            for _i, row in submission_df.head(3).iterrows():
                print(f"  Row {row['row_id']}: {row['predictions']}")

            print("✅ Prediction completed successfully")
        else:
            print("❌ Submission file not created")
            sys.exit(1)

        print("==============================")
        logger.info("Prediction script completed successfully")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)
