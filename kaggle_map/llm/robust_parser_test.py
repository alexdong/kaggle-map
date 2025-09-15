"""Unit tests for robust prediction parser."""

import pytest

from kaggle_map.core.models import Category, Prediction
from kaggle_map.llm.robust_parser import (
    find_best_category_match,
    levenshtein_distance,
    parse_predictions_robust,
    parse_single_prediction_robust,
    similarity_ratio,
)


class TestLevenshteinDistance:
    """Test Levenshtein distance calculation."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("hello", "hallo") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("hello", "helllo") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("hello", "hllo") == 1

    def test_multiple_changes(self):
        assert levenshtein_distance("True_Neither", "True_NNeither") == 1
        assert levenshtein_distance("True_Correct", "TrueCorrect") == 1
        assert levenshtein_distance("False_Misconception", "FalseMisconception") == 1

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5


class TestSimilarityRatio:
    """Test similarity ratio calculation."""

    def test_identical_strings(self):
        assert similarity_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        assert similarity_ratio("abc", "xyz") == 0.0

    def test_case_insensitive(self):
        assert similarity_ratio("Hello", "hello") == 1.0
        assert similarity_ratio("TRUE_CORRECT", "true_correct") == 1.0

    def test_partial_similarity(self):
        # "True_Neither" vs "True_NNeither" - 1 char diff in 13 chars
        ratio = similarity_ratio("True_Neither", "True_NNeither")
        assert 0.92 <= ratio <= 0.93

    def test_missing_underscore(self):
        # "True_Correct" vs "TrueCorrect" - 1 char diff in 12 chars
        ratio = similarity_ratio("True_Correct", "TrueCorrect")
        assert 0.91 <= ratio <= 0.92


class TestFindBestCategoryMatch:
    """Test fuzzy category matching."""

    def test_exact_match(self):
        assert find_best_category_match("True_Correct") == Category.TRUE_CORRECT
        assert find_best_category_match("False_Neither") == Category.FALSE_NEITHER

    def test_case_insensitive(self):
        assert find_best_category_match("true_correct") == Category.TRUE_CORRECT
        assert find_best_category_match("FALSE_NEITHER") == Category.FALSE_NEITHER

    def test_common_typos(self):
        # LLM-specific typos
        assert find_best_category_match("True_NNeither") == Category.TRUE_NEITHER
        assert find_best_category_match("TrueCorrect") == Category.TRUE_CORRECT
        assert find_best_category_match("True_Misconcpetion") == Category.TRUE_MISCONCEPTION

    def test_threshold_rejection(self):
        # Completely wrong text should return None
        assert find_best_category_match("RandomText") is None
        assert find_best_category_match("XYZ") is None

    def test_custom_threshold(self):
        # With very high threshold, even small typos are rejected
        assert find_best_category_match("True_NNeither", threshold=0.95) is None
        # With lower threshold, more typos are accepted
        assert find_best_category_match("True_NNeither", threshold=0.70) == Category.TRUE_NEITHER


class TestParseSinglePredictionRobust:
    """Test single token parsing."""

    def test_standard_format(self):
        pred = parse_single_prediction_robust("True_Correct:NA")
        assert pred.category == Category.TRUE_CORRECT
        assert pred.misconception == "NA"

    def test_with_misconception(self):
        pred = parse_single_prediction_robust("True_Misconception:Division")
        assert pred.category == Category.TRUE_MISCONCEPTION
        assert pred.misconception == "Division"

    def test_missing_colon(self):
        pred = parse_single_prediction_robust("True_Neither")
        assert pred.category == Category.TRUE_NEITHER
        assert pred.misconception == "NA"

    def test_typos_in_category(self):
        pred = parse_single_prediction_robust("True_NNeither:Something")
        assert pred.category == Category.TRUE_NEITHER
        assert pred.misconception == "Something"

    def test_missing_underscore(self):
        pred = parse_single_prediction_robust("FalseCorrect:NA")
        assert pred.category == Category.FALSE_CORRECT
        assert pred.misconception == "NA"

    def test_unparseable(self):
        assert parse_single_prediction_robust("CompleteGarbage") is None
        assert parse_single_prediction_robust("") is None

    def test_empty_misconception(self):
        pred = parse_single_prediction_robust("True_Correct:")
        assert pred.category == Category.TRUE_CORRECT
        assert pred.misconception == "NA"


class TestParsePredictionsRobust:
    """Test full response parsing."""

    def test_standard_format(self):
        response = "True_Correct:NA True_Neither:NA True_Misconception:Division"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[0].misconception == "NA"
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[1].misconception == "NA"
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Division"

    def test_with_typos(self):
        response = "True_Correct:NA True_NNeither:NA True_Misconception:Subtraction"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER  # Fixed typo
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Subtraction"

    def test_missing_underscores(self):
        response = "TrueCorrect:NA FalseNeither:Test TrueMisconception:Area"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.FALSE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_comma_separated(self):
        response = "True_Correct:NA, True_Neither:NA, True_Misconception:Division"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_newline_separated(self):
        response = """True_Correct:NA
True_Neither:NA
True_Misconception:Division"""
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_partial_predictions(self):
        # Only 2 predictions provided
        response = "True_Correct:NA False_Neither:Test"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3  # Should pad to 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.FALSE_NEITHER
        assert predictions[2].category == Category.TRUE_CORRECT  # Default
        assert predictions[2].misconception == "NA"

    def test_no_valid_predictions(self):
        response = "garbage text with no valid categories"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3  # All defaults
        assert all(p.category == Category.TRUE_CORRECT for p in predictions)
        assert all(p.misconception == "NA" for p in predictions)

    def test_mixed_format_issues(self):
        # Combination of issues: typo, missing underscore, missing colon
        response = "True_NNeither:Test FalseCorrect True_Misconception:Math"
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_NEITHER
        assert predictions[0].misconception == "Test"
        assert predictions[1].category == Category.FALSE_CORRECT
        assert predictions[1].misconception == "NA"
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Math"

    def test_with_extra_text(self):
        # LLM might include extra explanation
        response = """Here are the predictions:
True_Correct:NA True_Neither:NA True_Misconception:Division
These represent the student's understanding."""
        predictions = parse_predictions_robust(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_custom_max_predictions(self):
        response = "True_Correct:NA True_Neither:NA"
        predictions = parse_predictions_robust(response, max_predictions=2)

        assert len(predictions) == 2
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])