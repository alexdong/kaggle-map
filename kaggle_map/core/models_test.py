import pytest

from kaggle_map.core.models import Category, compare_labels


@pytest.mark.parametrize(
    "actual_label, predicted_label, expected",
    [
        # Exact matches
        ("TRUE_CORRECT:NA", "True_Correct:NA", True),
        ("FALSE_MISCONCEPTION:WNB", "False_Misconception:WNB", True),
        ("TRUE_NEITHER:NA", "True_Neither:NA", True),
        
        # Case sensitivity issues
        ("TRUE_MISCONCEPTION:Incomplete", "True_Misconception:incomplete", True),
        ("FALSE_MISCONCEPTION:Adding_across", "False_Misconception:adding_across", True),
        
        # Different formats
        ("TRUE_CORRECT:NA", "True_Correct:NA", True),
        ("FALSE_CORRECT:NA", "False_Correct:NA", True),
        
        # Actual mismatches
        ("TRUE_CORRECT:NA", "False_Correct:NA", False),
        ("FALSE_MISCONCEPTION:WNB", "False_Misconception:Incomplete", False),
        ("TRUE_NEITHER:NA", "True_Misconception:Incomplete", False),
        
        # With Category prefix (should handle if needed)
        ("Category.TRUE_CORRECT:NA", "True_Correct:NA", True),
        ("Category.FALSE_MISCONCEPTION:WNB", "False_Misconception:WNB", True),
    ],
)
def test_compare_labels(actual_label: str, predicted_label: str, expected: bool) -> None:
    assert compare_labels(actual_label, predicted_label) == expected, (
        f"Label comparison failed: actual='{actual_label}' vs predicted='{predicted_label}' should be {expected}"
    )


@pytest.mark.parametrize(
    "csv_value, expected_category",
    [
        ("TRUE_CORRECT", Category.TRUE_CORRECT),
        ("TRUE_NEITHER", Category.TRUE_NEITHER), 
        ("TRUE_MISCONCEPTION", Category.TRUE_MISCONCEPTION),
        ("FALSE_CORRECT", Category.FALSE_CORRECT),
        ("FALSE_NEITHER", Category.FALSE_NEITHER),
        ("FALSE_MISCONCEPTION", Category.FALSE_MISCONCEPTION),
    ],
)
def test_category_from_csv_string(csv_value: str, expected_category: Category) -> None:
    """Test that CSV format strings are correctly converted to Category enums."""
    result = Category.from_csv_string(csv_value)
    assert result == expected_category
    assert result.value == expected_category.value


def test_category_from_csv_string_invalid_format() -> None:
    """Test that invalid CSV format strings raise appropriate errors."""
    
    # Empty string
    with pytest.raises(AssertionError, match="CSV category value cannot be empty"):
        Category.from_csv_string("")
    
    # No underscore
    with pytest.raises(AssertionError, match="Invalid CSV category format"):
        Category.from_csv_string("TRUECORRECT")
    
    # Too many parts
    with pytest.raises(AssertionError, match="Invalid CSV category format"):
        Category.from_csv_string("TRUE_CORRECT_EXTRA")
    
    # Invalid category value
    with pytest.raises(ValueError):
        Category.from_csv_string("INVALID_CATEGORY")