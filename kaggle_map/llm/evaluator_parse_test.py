"""Test cases for parse_predictions function with thinking tags."""

import pytest

from kaggle_map.core.models import Category
from kaggle_map.llm.evaluator import parse_predictions_enhanced


def test_parse_predictions_with_thinking_text():
    """Test parsing when LLM produces thinking text before predictions."""
    # Response with plain text but no valid predictions
    response = "Let me carefully analyze the student's answer and explanation."

    result = parse_predictions_enhanced(response)

    # Should return empty predictions when no valid predictions found
    assert len(result.predictions) == 0
    assert result.thinking_trace is None


def test_parse_predictions_with_think_tags():
    """Test parsing when response contains <think> tags."""
    # Response with thinking tags
    response = """<think>
The student selected the correct answer (1/3) * (2/3).
Their explanation mentions "1 / 3 of 2 - 3 = 1 1/3" which seems confused.
They may be mixing up the operation.
</think>

True_Neither:NA True_Misconception:Subtraction True_Correct:NA"""

    result = parse_predictions_enhanced(response)

    # Should parse the predictions after the thinking tags
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_NEITHER
    assert result.predictions[0].misconception == "NA"
    assert result.predictions[1].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[1].misconception == "Subtraction"
    assert result.predictions[2].category == Category.TRUE_CORRECT
    assert result.predictions[2].misconception == "NA"
    # Check thinking trace was extracted
    assert result.thinking_trace is not None
    assert "mixing up the operation" in result.thinking_trace


def test_parse_predictions_multiline_with_labels():
    """Test parsing when labels are on different lines."""
    response = """Let me analyze this step by step.

The student got the right answer but the explanation is unclear.

True_Neither:NA True_Correct:NA True_Misconception:Division"""

    result = parse_predictions_enhanced(response)

    # Should find the line with valid categories
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_NEITHER
    assert result.predictions[1].category == Category.TRUE_CORRECT
    assert result.predictions[2].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[2].misconception == "Division"
    assert result.thinking_trace is None


def test_parse_predictions_mixed_valid_invalid():
    """Test parsing with mix of valid and invalid tokens."""
    response = "Some random text True_Correct:NA more text True_Neither:NA final text True_Misconception:Fractions"

    result = parse_predictions_enhanced(response)

    # Should extract only the valid predictions
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_CORRECT
    assert result.predictions[1].category == Category.TRUE_NEITHER
    assert result.predictions[2].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[2].misconception == "Fractions"
    assert result.thinking_trace is None


def test_parse_predictions_categories_without_colons():
    """Test parsing categories without colons (assumes NA misconception)."""
    response = "True_Correct True_Misconception:Division True_Neither"

    result = parse_predictions_enhanced(response)

    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_CORRECT
    assert result.predictions[0].misconception == "NA"
    assert result.predictions[1].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[1].misconception == "Division"
    assert result.predictions[2].category == Category.TRUE_NEITHER
    assert result.predictions[2].misconception == "NA"
    assert result.thinking_trace is None


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


def test_parse_gpt_oss_harmony_format_with_analysis():
    """Test parsing GPT-OSS Harmony format with analysis channel."""
    response = """<|channel|>analysis<|message|>
Let me analyze the student's explanation.
The student says "1/3 plus 2/3 is obviously not the answer."
They are rejecting addition.
<|channel|>final<|message|>
True_Neither:NA True_Misconception:Subtraction True_Correct:NA"""

    result = parse_predictions_enhanced(response)

    # Should extract predictions from final channel
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_NEITHER
    assert result.predictions[0].misconception == "NA"
    assert result.predictions[1].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[1].misconception == "Subtraction"

    # Should extract thinking trace from analysis channel
    assert result.thinking_trace is not None
    assert "analyze the student's explanation" in result.thinking_trace
    assert "rejecting addition" in result.thinking_trace


def test_parse_gpt_oss_harmony_format_without_final_channel():
    """Test parsing when GPT-OSS only outputs analysis channel with predictions embedded."""
    response = """<|channel|>analysis<|message|>
The student's reasoning is incorrect.
They should multiply the fractions.
<|end|>
True_Correct:NA True_Neither:NA True_Misconception:Division"""

    result = parse_predictions_enhanced(response)

    # Should parse predictions that appear after the analysis channel
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_CORRECT
    assert result.predictions[2].misconception == "Division"

    # Should extract thinking trace
    assert result.thinking_trace is not None
    assert "reasoning is incorrect" in result.thinking_trace


def test_parse_gpt_oss_harmony_format_with_end_tags():
    """Test parsing GPT-OSS format with <|end|> tags."""
    response = """<|channel|>analysis<|message|>Analyzing the response<|end|>
<|channel|>final<|message|>True_Neither:NA True_Correct:NA<|end|>"""

    result = parse_predictions_enhanced(response)

    # Should parse predictions correctly
    assert len(result.predictions) == 2
    assert result.predictions[0].category == Category.TRUE_NEITHER
    assert result.predictions[1].category == Category.TRUE_CORRECT

    # Should extract thinking trace without end tag
    assert result.thinking_trace == "Analyzing the response"


def test_parse_mixed_harmony_and_regular_content():
    """Test parsing when response has Harmony format mixed with regular text."""
    response = """Some initial text
<|channel|>analysis<|message|>This is the analysis
<|channel|>final<|message|>
True_Misconception:Fractions True_Neither:NA True_Correct:NA
Some trailing text"""

    result = parse_predictions_enhanced(response)

    # Should extract predictions from final channel only
    assert len(result.predictions) == 3
    assert result.predictions[0].misconception == "Fractions"

    # Should extract analysis as thinking trace
    assert result.thinking_trace == "This is the analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
