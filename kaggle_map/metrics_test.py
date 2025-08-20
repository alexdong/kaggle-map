"""Tests for evaluation metrics."""

import pytest

from kaggle_map.metrics import calculate_map_at_3
from kaggle_map.models import Category, Prediction


def test_calculate_map_at_3():
    """Test the calculate_map_at_3 function for single predictions."""
    ground_truth = Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")

    # First position match
    predictions1 = [
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across"),
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag")
    ]
    assert calculate_map_at_3(ground_truth, predictions1) == 1.0

    # Second position match
    predictions2 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    ]
    assert calculate_map_at_3(ground_truth, predictions2) == 0.5

    # Third position match
    predictions3 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong_tag"),
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    ]
    assert calculate_map_at_3(ground_truth, predictions3) == pytest.approx(1/3)

    # No match
    predictions4 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong_tag")
    ]
    assert calculate_map_at_3(ground_truth, predictions4) == 0.0

    # Empty predictions
    assert calculate_map_at_3(ground_truth, []) == 0.0


def test_calculate_map_at_3_prediction_matching():
    """Test that prediction matching works correctly for different cases."""
    # Test exact matches
    pred1 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across")
    pred2 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across")
    assert calculate_map_at_3(pred1, [pred2]) == 1.0

    # Test different misconceptions
    pred3 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Different_tag")
    assert calculate_map_at_3(pred1, [pred3]) == 0.0

    # Test different categories
    pred4 = Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Adding_across")
    assert calculate_map_at_3(pred1, [pred4]) == 0.0

    # Test NA misconception matches
    pred5 = Prediction(category=Category.TRUE_NEITHER, misconception="NA")
    pred6 = Prediction(category=Category.TRUE_NEITHER, misconception="NA")
    assert calculate_map_at_3(pred5, [pred6]) == 1.0
