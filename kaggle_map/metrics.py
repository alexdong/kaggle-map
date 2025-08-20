"""Evaluation metrics for the Kaggle MAP competition."""

from kaggle_map.models import Prediction


def calculate_map_at_3(ground_truth: Prediction, predictions: list[Prediction]) -> float:
    """Calculate Mean Average Precision at 3 (MAP@3) for a single prediction.

    MAP@3 awards full credit (1.0) if the correct prediction is in position 1,
    half credit (0.5) for position 2, one-third credit (0.333) for position 3,
    and zero credit if not in top 3 positions.

    Args:
        ground_truth: The correct Prediction
        predictions: List of predicted Prediction objects (up to 3)

    Returns:
        Average precision score (0.0 to 1.0)

    Examples:
        >>> from kaggle_map.models import Category, Prediction
        >>> gt = Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")

        >>> # Perfect prediction (position 1)
        >>> pred1 = [Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")]
        >>> calculate_map_at_3(gt, pred1)
        1.0

        >>> # Correct in position 2
        >>> pred2 = [
        ...     Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
        ...     Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
        ... ]
        >>> calculate_map_at_3(gt, pred2)
        0.5

        >>> # Correct in position 3
        >>> pred3 = [
        ...     Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
        ...     Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong"),
        ...     Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
        ... ]
        >>> calculate_map_at_3(gt, pred3)  # doctest: +ELLIPSIS
        0.333...

        >>> # No match
        >>> pred4 = [Prediction(category=Category.FALSE_NEITHER, misconception="NA")]
        >>> calculate_map_at_3(gt, pred4)
        0.0

        >>> # Empty predictions
        >>> calculate_map_at_3(gt, [])
        0.0
    """
    # Check each prediction position (1-indexed for precision calculation)
    for k, prediction in enumerate(predictions, 1):
        if str(ground_truth) == str(prediction):
            # Found correct prediction at position k, precision = 1/k
            return 1.0 / k

    # No correct prediction found
    return 0.0
