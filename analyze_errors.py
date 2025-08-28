#!/usr/bin/env python3
"""
Script to analyze prediction errors from our best MLP model.
Creates error_prediction.csv with cases where our prediction != actual misconception.
"""

import sys
from pathlib import Path

import pandas as pd

from kaggle_map.core.dataset import parse_training_data
from kaggle_map.core.models import EvaluationRow
from kaggle_map.strategies.mlp import MLPStrategy


def main() -> None:
    print("🔍 Analyzing prediction errors from best MLP model...")

    # Load the trained MLP model
    model_path = Path("models/mlp.pkl")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)

    print(f"📂 Loading model from {model_path}")
    strategy = MLPStrategy.load(model_path)

    # Load the original training data
    train_csv = Path("datasets/train.csv")
    print(f"📊 Loading training data from {train_csv}")
    train_data = parse_training_data(train_csv)

    print(f"📈 Running predictions on {len(train_data)} samples...")

    # Get predictions for all training samples
    predictions = []
    for i, row in enumerate(train_data):
        # Create EvaluationRow for prediction
        eval_row = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )

        # Get predictions (returns SubmissionRow with predictions)
        submission_row = strategy.predict(eval_row)

        # Extract top 3 predictions from submission row
        top_predictions = submission_row.predicted_categories[:3]

        # Extract just the misconception part from the Prediction object
        top_prediction = top_predictions[0].misconception if top_predictions else None

        predictions.append(
            {
                "row_id": row.row_id,
                "QuestionId": row.question_id,
                "QuestionText": row.question_text,
                "StudentExplanation": row.student_explanation,
                "Category": row.category,
                "actual_misconception": row.misconception,
                "predicted_misconception": top_prediction,
                "full_prediction": top_predictions[0] if top_predictions else None,
                "top_3_predictions": top_predictions[:3],  # Top 3 predictions
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"  ✅ Processed {i + 1}/{len(train_data)} samples")

    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)

    # Identify errors (where prediction != actual)
    errors_df = results_df[results_df["actual_misconception"] != results_df["predicted_misconception"]].copy()

    # Format the top 3 predictions for readability
    errors_df["top_3_predictions_formatted"] = errors_df["top_3_predictions"].apply(
        lambda x: " | ".join([str(pred) for pred in x]) if x else ""
    )

    # Save error predictions to CSV
    output_path = Path("datasets/error_prediction.csv")
    errors_df.to_csv(output_path, index=False)

    # Print summary statistics
    total_samples = len(results_df)
    error_count = len(errors_df)
    accuracy = (total_samples - error_count) / total_samples * 100

    print("\n📊 Error Analysis Summary:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Correct predictions: {total_samples - error_count:,}")
    print(f"  Prediction errors: {error_count:,}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Error rate: {(100 - accuracy):.2f}%")
    print(f"\n💾 Error predictions saved to: {output_path}")

    # Show a few example errors
    if not errors_df.empty:
        print("\n🔍 Sample errors (first 3):")
        for _idx, row in errors_df.head(3).iterrows():
            print(f"\n  Row {row['row_id']}:")
            print(f"    Question: {row['QuestionText'][:100]}...")
            print(f"    Student: {row['StudentExplanation'][:100]}...")
            print(f"    Actual: {row['actual_misconception']}")
            print(f"    Predicted: {row['predicted_misconception']}")
            print(f"    Top 3: {row['top_3_predictions_formatted']}")


if __name__ == "__main__":
    main()
