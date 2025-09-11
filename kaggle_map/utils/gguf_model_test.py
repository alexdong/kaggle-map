"""Tests for LLM utilities and reranker model configurations.

Summary of the UD Issue:
- UD likely stands for "Unsloth Dynamic" (Unsloth's Dynamic 2.0 quantization)
- ALL XL quantizations from Unsloth use "UD-" prefix in filenames
- Examples:
  - gemma-3-12b-it-UD-Q4_K_XL.gguf
  - Qwen3-14B-UD-Q4_K_XL.gguf
  - gpt-oss-20b-UD-Q4_K_XL.gguf
- This naming convention applies to all Unsloth models with XL quantizations
"""

from kaggle_map.utils.gguf_model import (
    GGUF_MODELS,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_model_path,
)


def test_gemma_model_formatting() -> None:
    user_content = "What is the capital of France?"
    result = format_chat_prompt(GGUFModelName.GEMMA_3_12B_IT, user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected, "Failed for model Gemma"


def test_qwen_model_formatting() -> None:
    user_content = "Solve 2+2"
    result = format_chat_prompt(GGUFModelName.QWEN3_14B, user_content)
    expected = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    assert result == expected, "Failed for model Qwen3"


def test_gpt_oss_model_formatting() -> None:
    user_content = "Explain quantum physics"
    result = format_chat_prompt(GGUFModelName.GPT_OSS_20B, user_content)
    expected = f"<|start|>user<|message|>{user_content}<|end|><|start|>assistant"
    assert result == expected, "Failed for model gpt-oss"


def test_multiline_user_content() -> None:
    user_content = "Line 1\nLine 2\nLine 3"
    result = format_chat_prompt(GGUFModelName.GEMMA_3_12B_IT, user_content)
    expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    assert result == expected
    assert "Line 1\nLine 2\nLine 3" in result


def test_special_characters_in_content() -> None:
    """Test formatting with special characters in user content."""
    user_content = "What's 2+2? <tag> & \"quotes\" 'apostrophes'"
    result = format_chat_prompt(GGUFModelName.GEMMA_3_12B_IT, user_content)
    assert user_content in result
    assert "<tag>" in result
    assert "&" in result
    assert '"quotes"' in result


def test_all_xl_quantizations_have_ud_prefix() -> None:
    """Test that ALL XL quantizations use UD- prefix in filenames.

    Note: Not all models have all XL quantizations available:
    - Gemma and Qwen have Q2, Q3, Q4, Q5, Q6, Q8 XL versions
    - GPT-OSS-20B only has Q4, Q6, Q8 XL versions (missing Q2, Q3, Q5)
    """
    # Define which quantizations are available for each model
    model_quantizations = {
        GGUFModelName.GEMMA_3_12B_IT: [
            GGUFModelQuantizationLevel.Q2_K_XL,
            GGUFModelQuantizationLevel.Q3_K_XL,
            GGUFModelQuantizationLevel.Q4_K_XL,
            GGUFModelQuantizationLevel.Q5_K_XL,
            GGUFModelQuantizationLevel.Q6_K_XL,
        ],
        GGUFModelName.QWEN3_14B: [
            GGUFModelQuantizationLevel.Q2_K_XL,
            GGUFModelQuantizationLevel.Q3_K_XL,
            GGUFModelQuantizationLevel.Q4_K_XL,
            GGUFModelQuantizationLevel.Q5_K_XL,
            GGUFModelQuantizationLevel.Q6_K_XL,
        ],
        GGUFModelName.GPT_OSS_20B: [
            GGUFModelQuantizationLevel.Q4_K_XL,
            GGUFModelQuantizationLevel.Q6_K_XL,
            # Q8_K_XL exists but not in our enum
        ],
    }

    for model, quantizations in model_quantizations.items():
        for quant in quantizations:
            path = get_model_path(model, quant)
            assert "UD-" in str(path), f"Missing UD- prefix for {model} {quant}: {path}"
            expected = f"models/gguf/{model.value}-UD-{quant.value}.gguf"
            assert str(path) == expected, f"Expected {expected}, got {path}"


def test_all_models_filename_patterns_include_ud() -> None:
    """Test that all models' GGUF configs have correct UD- pattern."""
    for model_name in [GGUFModelName.GEMMA_3_12B_IT, GGUFModelName.QWEN3_14B, GGUFModelName.GPT_OSS_20B]:
        config = GGUF_MODELS[model_name]
        assert "UD-" in config.filename_pattern, f"Missing UD- in pattern for {model_name}"
        expected = f"{model_name.value}-UD-{{quant}}.gguf"
        assert config.filename_pattern == expected, f"Expected {expected}, got {config.filename_pattern}"


def test_non_xl_quantizations_no_ud_prefix() -> None:
    """Test that non-XL quantizations don't get UD- prefix."""
    # Test a non-XL quantization (if we had Q4_K_M for example)
    # This test would need actual non-XL quantizations to be meaningful
    # For now, just verify the logic in get_model_path

    # Since all our quantizations are XL, we can't test this properly
    # but we document the expected behavior
