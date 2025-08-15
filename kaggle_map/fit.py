"""Model fitting script for the Kaggle student misconception prediction competition."""

from pathlib import Path

from loguru import logger

try:
    from .models import MAPModel, save_model
except ImportError:
    from models import MAPModel, save_model


def fit_model(
    train_csv_path: Path = Path("dataset/train.csv"),
    model_output_path: Path = Path("baseline_model.json"),
) -> MAPModel:
    """Fit a baseline model from training data and save it.

    Args:
        train_csv_path: Path to the training CSV file
        model_output_path: Path where to save the fitted model

    Returns:
        Fitted MAPModel
    """
    logger.info(f"Starting model fitting from {train_csv_path}")

    # Fit the model
    model = MAPModel.fit(train_csv_path)
    logger.info("Model fitting completed successfully")

    # Display model statistics
    logger.info(f"Model trained on {len(model.correct_answers)} questions")
    logger.info(f"Found correct answers for {len(model.correct_answers)} questions")
    logger.info(
        f"Built category patterns for {len(model.category_frequencies)} questions"
    )

    misconception_count = sum(
        1 for m in model.common_misconceptions.values() if m is not None
    )
    logger.info(f"Extracted misconceptions for {misconception_count} questions")

    # Save the model
    save_model(model, model_output_path)
    logger.info(f"Model saved to {model_output_path}")

    return model


if __name__ == "__main__":
    """Fit baseline model and save it for prediction use."""
    import sys
    from pathlib import Path

    # Parse command line arguments
    min_args_for_train = 1
    min_args_for_output = 2

    train_path = (
        Path(sys.argv[1])
        if len(sys.argv) > min_args_for_train
        else Path("dataset/train.csv")
    )

    if len(sys.argv) > min_args_for_output:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path("baseline_model.json")

    logger.info("Running model fitting script")
    logger.info(f"Training data: {train_path}")
    logger.info(f"Output model: {output_path}")

    try:
        # Fit and save the model
        fitted_model = fit_model(train_path, output_path)

        # Display final model summary
        print("\n=== MODEL FITTING COMPLETED ===")
        print(f"Training data: {train_path}")
        print(f"Model saved to: {output_path}")
        print(f"Questions processed: {len(fitted_model.correct_answers)}")
        print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")

        # Quick validation
        if output_path.exists():
            print("✅ Model file created successfully")
        else:
            print("❌ Model file not found after saving")
            sys.exit(1)

        print("====================================")
        logger.info("Model fitting script completed successfully")

    except Exception as e:
        logger.error(f"Error during model fitting: {e}")
        print(f"❌ Model fitting failed: {e}")
        sys.exit(1)
