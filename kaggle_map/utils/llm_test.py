"""Tests for LLM utilities."""

import pytest

from kaggle_map.utils.llm import format_chat_prompt, get_stop_tokens

def test_gemma_model_formatting(self):
    user_content = "What is the capital of France?"
    result = format_chat_prompt("gemma-3-12b-it", user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected, f"Failed for model Gemma"

def test_qwen_model_formatting(self):
    user_content = "Solve 2+2"
    result = format_chat_prompt("Qwen3-14B", user_content)
    expected = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    assert result == expected, f"Failed for model Qwen3"

def test_gpt_oss_model_formatting(self):
    user_content = "Explain quantum physics"
    result = format_chat_prompt("gpt-oss-20b", user_content)
    expected = f"<|start|>user<|message|>{user_content}<|end|><|start|>assistant"
    assert result == expected, f"Failed for model gpt-oss"

def test_multiline_user_content(self):
    user_content = "Line 1\nLine 2\nLine 3"
    result = format_chat_prompt("gemma-3-12b-it", user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected
    assert "Line 1\nLine 2\nLine 3" in result

def test_special_characters_in_content(self):
    """Test formatting with special characters in user content."""
    user_content = "What's 2+2? <tag> & \"quotes\" 'apostrophes'"
    result = format_chat_prompt("gemma-3-12b-it", user_content)
    assert user_content in result
    assert "<tag>" in result
    assert "&" in result
    assert '"quotes"' in result

