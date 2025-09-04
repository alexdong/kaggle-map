import pytest

from kaggle_map.core.models import compare_labels


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
    assert compare_labels(actual_label, predicted_label) == expected