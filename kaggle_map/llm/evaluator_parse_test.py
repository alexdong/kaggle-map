"""Test cases for parse_predictions function with thinking tags."""

import re
from dataclasses import dataclass

import pytest

from kaggle_map.core.models import Category, Prediction
from kaggle_map.llm.evaluator import parse_predictions


@dataclass
class ParseResult:
    """Result from parsing LLM response."""

    predictions: list[Prediction]
    thinking_trace: str | None = None


def parse_predictions_enhanced(response: str) -> ParseResult:
    """Enhanced parse_predictions that extracts and returns thinking traces.

    This is the proposed implementation that:
    1. Extracts content within <think>...</think> tags
    2. Cleans the response for prediction parsing
    3. Returns both predictions and thinking trace
    """
    # Extract thinking trace if present
    thinking_pattern = r"<think>(.*?)</think>"
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)

    thinking_trace = None
    if thinking_match:
        thinking_trace = thinking_match.group(1).strip()

    # Remove thinking tags from response for prediction parsing
    clean_response = re.sub(thinking_pattern, "", response, flags=re.DOTALL)

    # Parse predictions from cleaned response
    predictions = parse_predictions(clean_response)

    return ParseResult(predictions=predictions, thinking_trace=thinking_trace)


def test_parse_predictions_with_thinking_text():
    """Test parsing when LLM produces thinking text before predictions."""
    # Real response from the LLM
    response = "Let me carefully analyze the student's answer and explanation."

    predictions = parse_predictions(response)

    # Should return 3 default predictions when no valid predictions found
    assert len(predictions) == 3
    assert all(p.category == Category.TRUE_CORRECT for p in predictions)
    assert all(p.misconception == "NA" for p in predictions)


def test_parse_predictions_with_think_tags():
    """Test parsing when response contains <think> tags."""
    # Response with thinking tags
    response = """<think>
The student selected the correct answer (1/3) * (2/3).
Their explanation mentions "1 / 3 of 2 - 3 = 1 1/3" which seems confused.
They may be mixing up the operation.
</think>

True_Neither:NA True_Misconception:Subtraction True_Correct:NA"""

    predictions = parse_predictions(response)

    # Should parse the predictions after the thinking tags
    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_NEITHER
    assert predictions[0].misconception == "NA"
    assert predictions[1].category == Category.TRUE_MISCONCEPTION
    assert predictions[1].misconception == "Subtraction"
    assert predictions[2].category == Category.TRUE_CORRECT
    assert predictions[2].misconception == "NA"


def test_parse_predictions_multiline_with_labels():
    """Test parsing when labels are on different lines."""
    response = """Let me analyze this step by step.

The student got the right answer but the explanation is unclear.

True_Neither:NA True_Correct:NA True_Misconception:Division"""

    predictions = parse_predictions(response)

    # Should find the line with valid categories
    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_NEITHER
    assert predictions[1].category == Category.TRUE_CORRECT
    assert predictions[2].category == Category.TRUE_MISCONCEPTION
    assert predictions[2].misconception == "Division"


def test_parse_predictions_mixed_valid_invalid():
    """Test parsing with mix of valid and invalid tokens."""
    response = "Some random text True_Correct:NA more text True_Neither:NA final text True_Misconception:Fractions"

    predictions = parse_predictions(response)

    # Should extract only the valid predictions
    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[1].category == Category.TRUE_NEITHER
    assert predictions[2].category == Category.TRUE_MISCONCEPTION
    assert predictions[2].misconception == "Fractions"


def test_parse_predictions_categories_without_colons():
    """Test parsing categories without colons (assumes NA misconception)."""
    response = "True_Correct True_Misconception:Division True_Neither"

    predictions = parse_predictions(response)

    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[0].misconception == "NA"
    assert predictions[1].category == Category.TRUE_MISCONCEPTION
    assert predictions[1].misconception == "Division"
    assert predictions[2].category == Category.TRUE_NEITHER
    assert predictions[2].misconception == "NA"


def test_thinking_trace_extraction():
    """Test that we can extract thinking traces for logging."""
    response_with_tags = """<think>
The student's explanation shows confusion about the operation.
They mention subtraction which is incorrect.
</think>

True_Misconception:Subtraction True_Neither:NA True_Correct:NA"""

    # Extract thinking trace (this is what we want to add)
    import re

    thinking_pattern = r"<think>(.*?)</think>"
    thinking_match = re.search(thinking_pattern, response_with_tags, re.DOTALL)

    if thinking_match:
        thinking_trace = thinking_match.group(1).strip()
        assert "confusion about the operation" in thinking_trace
        assert "subtraction which is incorrect" in thinking_trace

    # Clean response for prediction parsing
    clean_response = re.sub(thinking_pattern, "", response_with_tags, flags=re.DOTALL)
    predictions = parse_predictions(clean_response)

    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_MISCONCEPTION
    assert predictions[0].misconception == "Subtraction"


def test_enhanced_parse_predictions_with_thinking():
    """Test the enhanced parse_predictions that returns thinking trace."""
    response = """<think>
The student selected the correct answer (1/3) * (2/3).
However, their explanation "it is 1 / 3 of 2 - 3 = 1 1/3" is confusing.
They seem to be doing subtraction (2-3) instead of multiplication.
This indicates a misconception about the operation.
</think>

True_Misconception:Subtraction True_Neither:NA True_Correct:NA"""

    # This is what we want the enhanced function to do:
    # predictions, thinking_trace = parse_predictions_with_thinking(response)

    # For now, let's simulate what we want:
    import re

    thinking_pattern = r"<think>(.*?)</think>"
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)

    thinking_trace = None
    if thinking_match:
        thinking_trace = thinking_match.group(1).strip()

    clean_response = re.sub(thinking_pattern, "", response, flags=re.DOTALL)
    predictions = parse_predictions(clean_response)

    # Check results
    assert thinking_trace is not None
    assert "doing subtraction (2-3) instead of multiplication" in thinking_trace
    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_MISCONCEPTION
    assert predictions[0].misconception == "Subtraction"


def test_response_with_thinking_tags():
    """Test parsing response with thinking tags."""
    response = """<think>
The student selected the correct answer (1/3) * (2/3).
However, their explanation "it is 1 / 3 of 2 - 3 = 1 1/3" is confusing.
They seem to be doing subtraction (2-3) instead of multiplication.
This indicates a misconception about the operation.
</think>

True_Misconception:Subtraction True_Neither:NA True_Correct:NA"""

    result = parse_predictions_enhanced(response)

    # Check thinking trace was extracted
    assert result.thinking_trace is not None
    assert "doing subtraction (2-3) instead of multiplication" in result.thinking_trace
    assert "misconception about the operation" in result.thinking_trace

    # Check predictions were parsed correctly
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[0].misconception == "Subtraction"
    assert result.predictions[1].category == Category.TRUE_NEITHER
    assert result.predictions[2].category == Category.TRUE_CORRECT


def test_response_without_thinking_tags():
    """Test parsing response without thinking tags."""
    response = "True_Correct:NA True_Neither:NA True_Misconception:Division"

    result = parse_predictions_enhanced(response)

    # No thinking trace
    assert result.thinking_trace is None

    # Predictions parsed normally
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_CORRECT
    assert result.predictions[2].misconception == "Division"


def test_response_with_empty_thinking_tags():
    """Test parsing response with empty thinking tags."""
    response = """<think>
</think>

True_Neither:NA True_Correct:NA True_Misconception:Fractions"""

    result = parse_predictions_enhanced(response)

    # Empty thinking trace
    assert result.thinking_trace == ""

    # Predictions parsed normally
    assert len(result.predictions) == 3
    assert result.predictions[2].misconception == "Fractions"


def test_response_with_multiple_thinking_tags():
    """Test parsing response with multiple thinking tag sections (takes first)."""
    response = """<think>
First thinking section.
</think>

Some text here.

<think>
Second thinking section.
</think>

True_Correct:NA True_Neither:NA True_Misconception:Division"""

    result = parse_predictions_enhanced(response)

    # Should extract first thinking section
    assert result.thinking_trace is not None
    assert "First thinking section" in result.thinking_trace
    assert "Second thinking section" not in result.thinking_trace

    # Predictions parsed normally
    assert len(result.predictions) == 3


def test_response_with_nested_tags_in_thinking():
    """Test that content within thinking tags is preserved as-is."""
    response = """<think>
Looking at the answer, I see:
- Point 1: <important>This is key</important>
- Point 2: The calculation is (1/3) * (2/3)
</think>

True_Correct:NA True_Neither:NA True_Misconception:NA"""

    result = parse_predictions_enhanced(response)

    # Thinking trace preserves nested content
    assert result.thinking_trace is not None
    assert "<important>This is key</important>" in result.thinking_trace
    assert "(1/3) * (2/3)" in result.thinking_trace

    # Predictions parsed normally
    assert len(result.predictions) == 3


def test_logging_with_thinking_trace():
    """Test that thinking traces can be logged appropriately."""
    import io

    from loguru import logger

    # Capture log output
    log_capture = io.StringIO()
    handler_id = logger.add(log_capture, format="{message}", level="DEBUG")

    response = """<think>
The student's approach shows a fundamental misunderstanding.
They are confusing multiplication with subtraction.
</think>

True_Misconception:Subtraction True_Neither:NA True_Correct:NA"""

    result = parse_predictions_enhanced(response)

    # Log the thinking trace
    if result.thinking_trace:
        logger.info(f"LLM Thinking Trace:\n{result.thinking_trace}")

    # Check log output
    log_output = log_capture.getvalue()
    assert "LLM Thinking Trace:" in log_output
    assert "fundamental misunderstanding" in log_output
    assert "confusing multiplication with subtraction" in log_output

    # Clean up logger
    logger.remove(handler_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
