"""Tests for LLM utilities."""

from kaggle_map.reranker.models import RerankerModelName
from kaggle_map.reranker.utils import format_chat_prompt


def test_gemma_model_formatting() -> None:
    user_content = "What is the capital of France?"
    result = format_chat_prompt(RerankerModelName.GEMMA_3_12B_IT, user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected, "Failed for model Gemma"


def test_qwen_model_formatting() -> None:
    user_content = "Solve 2+2"
    result = format_chat_prompt(RerankerModelName.QWEN3_14B, user_content)
    expected = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    assert result == expected, "Failed for model Qwen3"


def test_gpt_oss_model_formatting() -> None:
    user_content = "Explain quantum physics"
    result = format_chat_prompt(RerankerModelName.GPT_OSS_20B, user_content)
    expected = f"<|start|>user<|message|>{user_content}<|end|><|start|>assistant"
    assert result == expected, "Failed for model gpt-oss"


def test_multiline_user_content() -> None:
    user_content = "Line 1\nLine 2\nLine 3"
    result = format_chat_prompt(RerankerModelName.GEMMA_3_12B_IT, user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected
    assert "Line 1\nLine 2\nLine 3" in result


def test_special_characters_in_content() -> None:
    """Test formatting with special characters in user content."""
    user_content = "What's 2+2? <tag> & \"quotes\" 'apostrophes'"
    result = format_chat_prompt(RerankerModelName.GEMMA_3_12B_IT, user_content)
    assert user_content in result
    assert "<tag>" in result
    assert "&" in result
    assert '"quotes"' in result
