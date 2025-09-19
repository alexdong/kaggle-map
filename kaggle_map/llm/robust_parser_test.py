"""Unit tests for robust prediction parser."""

import re
from typing import cast

import pytest

from kaggle_map.core.models import Category
from kaggle_map.llm.robust_parser import (
    find_best_category_match,
    levenshtein_distance,
    parse_predictions_with_fuzzy_matching,
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

    def test_invalid_input_types(self):
        """Test that non-string inputs fail fast with clear messages."""
        with pytest.raises(AssertionError, match="s1 must be string"):
            levenshtein_distance(cast("str", 123), "hello")

        with pytest.raises(AssertionError, match="s2 must be string"):
            levenshtein_distance("hello", cast("str", 456))


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

    def test_invalid_input_types(self):
        """Test that non-string inputs fail fast."""
        with pytest.raises(AssertionError, match="s1 must be string"):
            similarity_ratio(cast("str", 123), "hello")

        with pytest.raises(AssertionError, match="s2 must be string"):
            similarity_ratio("hello", cast("str", 456))

    def test_ratio_bounds(self):
        """Test that similarity ratio is always in valid range."""
        # Test various combinations
        ratio1 = similarity_ratio("", "")
        assert 0.0 <= ratio1 <= 1.0

        ratio2 = similarity_ratio("hello", "world")
        assert 0.0 <= ratio2 <= 1.0

    def test_empty_string_edge_cases(self):
        # Test edge cases with empty strings
        assert similarity_ratio("", "") == 1.0
        assert similarity_ratio("hello", "") == 0.0
        assert similarity_ratio("", "hello") == 0.0


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

    def test_invalid_threshold(self):
        """Test that invalid thresholds fail fast."""
        threshold_message = re.escape("threshold must be between 0.0 and 1.0")
        with pytest.raises(AssertionError, match=threshold_message):
            find_best_category_match("True_Correct", threshold=-0.1)

        with pytest.raises(AssertionError, match=threshold_message):
            find_best_category_match("True_Correct", threshold=1.5)

    def test_empty_text(self):
        """Test that empty text fails fast."""
        with pytest.raises(AssertionError, match="text cannot be empty"):
            find_best_category_match("")

        with pytest.raises(AssertionError, match="text cannot be empty"):
            find_best_category_match("   ")

    def test_invalid_text_type(self):
        """Test that non-string text fails fast."""
        with pytest.raises(AssertionError, match="text must be string"):
            find_best_category_match(cast("str", 123))

    def test_boundary_threshold_conditions(self):
        # Test exact threshold boundary conditions
        # Find a case where similarity is exactly at threshold
        text = "True_Correct"  # This should have perfect match
        assert find_best_category_match(text, threshold=1.0) == Category.TRUE_CORRECT

        # Test with threshold of 0.0 (should accept anything with partial match)
        assert find_best_category_match("True", threshold=0.0) is not None

        # Test multiple similar matches - should return the best one
        # "False" matches both "False_Correct" and "False_Neither" partially
        match = find_best_category_match("False", threshold=0.0)
        assert match in [Category.FALSE_CORRECT, Category.FALSE_NEITHER]


class TestParseSinglePredictionRobust:
    """Test single token parsing."""

    def test_standard_format(self):
        pred = parse_single_prediction_robust("True_Correct:NA")
        assert pred is not None, "Should successfully parse valid format"
        assert pred.category == Category.TRUE_CORRECT
        assert pred.misconception == "NA"

    def test_with_misconception(self):
        pred = parse_single_prediction_robust("True_Misconception:Division")
        assert pred is not None, "Should successfully parse valid format with misconception"
        assert pred.category == Category.TRUE_MISCONCEPTION
        assert pred.misconception == "Division"

    def test_missing_colon(self):
        pred = parse_single_prediction_robust("True_Neither")
        assert pred is not None, "Should successfully parse category without colon"
        assert pred.category == Category.TRUE_NEITHER
        assert pred.misconception == "NA"

    def test_typos_in_category(self):
        pred = parse_single_prediction_robust("True_NNeither:Something")
        assert pred is not None, "Should successfully parse typo with fuzzy matching"
        assert pred.category == Category.TRUE_NEITHER
        assert pred.misconception == "Something"

    def test_missing_underscore(self):
        pred = parse_single_prediction_robust("FalseCorrect:NA")
        assert pred is not None, "Should successfully parse missing underscore with fuzzy matching"
        assert pred.category == Category.FALSE_CORRECT
        assert pred.misconception == "NA"

    def test_unparseable(self):
        assert parse_single_prediction_robust("CompleteGarbage") is None
        assert parse_single_prediction_robust("") is None

    def test_invalid_token_type(self):
        """Test that non-string tokens fail fast."""
        with pytest.raises(AssertionError, match="token must be string"):
            parse_single_prediction_robust(cast("str", 123))

    def test_empty_misconception(self):
        pred = parse_single_prediction_robust("True_Correct:")
        assert pred is not None, "Should successfully parse empty misconception"
        assert pred.category == Category.TRUE_CORRECT
        assert pred.misconception == "NA"


class TestParsePredictionsRobust:
    """Test full response parsing."""

    def test_standard_format(self):
        response = "True_Correct:NA True_Neither:NA True_Misconception:Division"
        predictions = parse_predictions_with_fuzzy_matching(response, pad_to_max=True)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[0].misconception == "NA"
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[1].misconception == "NA"
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Division"

    def test_with_typos(self):
        response = "True_Correct:NA True_NNeither:NA True_Misconception:Subtraction"
        predictions = parse_predictions_with_fuzzy_matching(response, pad_to_max=True)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER  # Fixed typo
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Subtraction"

    def test_missing_underscores(self):
        response = "TrueCorrect:NA FalseNeither:Test TrueMisconception:Area"
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.FALSE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_comma_separated(self):
        response = "True_Correct:NA, True_Neither:NA, True_Misconception:Division"
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_comma_separated_without_spaces_and_colons(self):
        # Test comma format that doesn't contain both spaces AND colons
        # This should trigger the comma-specific code path
        response = "True_Correct,True_Neither,True_Misconception"
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_newline_separated(self):
        response = """True_Correct:NA
True_Neither:NA
True_Misconception:Division"""
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_partial_predictions(self):
        # Only 2 predictions provided
        response = "True_Correct:NA False_Neither:Test"
        predictions = parse_predictions_with_fuzzy_matching(response, pad_to_max=True)

        assert len(predictions) == 3  # Should pad to 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.FALSE_NEITHER
        assert predictions[2].category == Category.TRUE_CORRECT  # Default
        assert predictions[2].misconception == "NA"

    def test_no_valid_predictions(self):
        response = "garbage text with no valid categories"
        predictions = parse_predictions_with_fuzzy_matching(response, pad_to_max=True)

        assert len(predictions) == 3  # All defaults
        assert all(p.category == Category.TRUE_CORRECT for p in predictions)
        assert all(p.misconception == "NA" for p in predictions)

    def test_mixed_format_issues(self):
        # Combination of issues: typo, missing underscore, missing colon
        response = "True_NNeither:Test FalseCorrect True_Misconception:Math"
        predictions = parse_predictions_with_fuzzy_matching(response)

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
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION

    def test_custom_max_predictions(self):
        response = "True_Correct:NA True_Neither:NA"
        predictions = parse_predictions_with_fuzzy_matching(response, max_predictions=2)

        assert len(predictions) == 2
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER

    def test_invalid_max_predictions(self):
        """Test that invalid max_predictions values fail fast."""
        response = "True_Correct:NA"

        with pytest.raises(AssertionError, match="max_predictions must be positive"):
            parse_predictions_with_fuzzy_matching(response, max_predictions=0)

        with pytest.raises(AssertionError, match="max_predictions must be positive"):
            parse_predictions_with_fuzzy_matching(response, max_predictions=-1)

        with pytest.raises(AssertionError, match="max_predictions seems unreasonable"):
            parse_predictions_with_fuzzy_matching(response, max_predictions=100)

    def test_invalid_response_type(self):
        """Test that non-string responses fail fast."""
        with pytest.raises(AssertionError, match="response must be string"):
            parse_predictions_with_fuzzy_matching(cast("str", 123))

    def test_tab_separated_format(self):
        # Test tab-separated format to improve coverage
        response = "True_Correct:NA\tTrue_Neither:NA\tTrue_Misconception:Division"
        predictions = parse_predictions_with_fuzzy_matching(response)

        assert len(predictions) == 3
        assert predictions[0].category == Category.TRUE_CORRECT
        assert predictions[1].category == Category.TRUE_NEITHER
        assert predictions[2].category == Category.TRUE_MISCONCEPTION
        assert predictions[2].misconception == "Division"

    def test_edge_case_empty_response(self):
        # Test completely empty response
        predictions = parse_predictions_with_fuzzy_matching("", pad_to_max=True)

        assert len(predictions) == 3
        assert all(p.category == Category.TRUE_CORRECT for p in predictions)
        assert all(p.misconception == "NA" for p in predictions)

    def test_whitespace_only_response(self):
        # Test response with only whitespace
        predictions = parse_predictions_with_fuzzy_matching("   \n   \t   ", pad_to_max=True)

        assert len(predictions) == 3
        assert all(p.category == Category.TRUE_CORRECT for p in predictions)
        assert all(p.misconception == "NA" for p in predictions)


class TestContractEnforcement:
    """Test that functions enforce their contracts with assertions."""

    def test_parse_predictions_robust_returns_exact_count(self):
        """Test that parse_predictions_robust always returns exactly max_predictions items."""
        # Empty response should still return exactly max_predictions defaults
        result = parse_predictions_with_fuzzy_matching("", max_predictions=5)
        assert len(result) == 5, "Contract violation: should return exactly max_predictions items"

        # Response with some valid predictions should still pad to max_predictions
        result = parse_predictions_with_fuzzy_matching("True_Correct:NA", max_predictions=3)
        assert len(result) == 3, "Contract violation: should return exactly max_predictions items"

    def test_similarity_ratio_bounds_enforcement(self):
        """Test that similarity_ratio always returns values in [0.0, 1.0]."""
        # Test edge cases that might break bounds
        ratio = similarity_ratio("", "hello")
        assert 0.0 <= ratio <= 1.0, "Contract violation: ratio outside valid bounds"

        ratio = similarity_ratio("a" * 100, "b" * 100)
        assert 0.0 <= ratio <= 1.0, "Contract violation: ratio outside valid bounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
