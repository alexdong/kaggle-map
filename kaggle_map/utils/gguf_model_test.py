"""Tests for LLM utilities and reranker model configurations.

Summary of the UD Issue:
- UD likely stands for "Unsloth Dynamic" (Unsloth's Dynamic 2.0 quantization)
- Examples:
  - gemma-3-27b-it-UD-Q4_K_XL.gguf
  - Qwen3-14B-UD-Q4_K_XL.gguf
  - gpt-oss-20b-Q4_K_XL.gguf
- This naming convention applies to all Unsloth models with XL quantizations
"""

from kaggle_map.utils.gguf_model import (
    GGUF_MODELS,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_model_path,
    get_stop_tokens,
)


def test_gemma_model_formatting() -> None:
    user_content = "What is the capital of France?"
    # Test both Gemma models
    for model in [GGUFModelName.GEMMA_3_12B_IT, GGUFModelName.GEMMA_3_27B_IT]:
        result = format_chat_prompt(model, user_content)
        expected = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
        assert result == expected, f"Failed for model {model.value}"


def test_qwen_model_formatting() -> None:
    user_content = "Solve 2+2"
    result = format_chat_prompt(GGUFModelName.QWEN3_14B, user_content)
    expected = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
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


def test_all_xl_quantizations_have_correct_local_path() -> None:
    """Test that XL quantizations have correct local paths.

    XL quantizations from Unsloth have "UD-" prefix in download URLs,
    but local files should NOT have the "UD-" prefix after download.

    Note: Not all models have all XL quantizations available.
    We test based on what's configured in GGUF_MODELS.
    """
    # Use the actual available quantizations from GGUF_MODELS
    for model_name, config in GGUF_MODELS.items():
        quantizations = config.available_quantizations

        for quant in quantizations:
            path = get_model_path(model_name, quant)
            # Local files should NOT have UD- prefix
            assert "UD-" not in str(path), f"Unexpected UD- prefix in local path for {model_name} {quant}: {path}"
            expected = f"models/gguf/{model_name.value}-{quant.value}.gguf"
            assert str(path) == expected, f"Expected {expected}, got {path}"


def test_all_models_filename_patterns() -> None:
    """Test that all models' GGUF configs have correct filename patterns.

    Some models use UD- prefix (Unsloth Dynamic), others don't.
    This depends on the specific model and how it's hosted.
    """
    expected_patterns = {
        GGUFModelName.QWEN3_14B: "Qwen3-14B-{quant}.gguf",
        GGUFModelName.QWEN3_30B: "Qwen3-30B-A3B-Instruct-2507-UD-{quant}.gguf",  # Has UD- prefix
        GGUFModelName.QWEN3_30B_Thinking: "Qwen3-30B-A3B-Thinking-2507-UD-{quant}.gguf",  # Has UD- prefix
        GGUFModelName.GEMMA_3_27B_IT: "gemma-3-27b-it-UD-{quant}.gguf",  # Has UD- prefix
        GGUFModelName.GPT_OSS_20B: "gpt-oss-20b-{quant}.gguf",  # No UD- prefix
    }

    for model_name, expected in expected_patterns.items():
        config = GGUF_MODELS[model_name]
        assert config.filename_pattern == expected, (
            f"Model {model_name}: Expected {expected}, got {config.filename_pattern}"
        )


def test_new_models_configuration() -> None:
    """Test that new models (GPT-OSS-20B and GEMMA-3-27B-IT) are properly configured."""
    # Test GPT-OSS-20B
    assert GGUFModelName.GPT_OSS_20B in GGUF_MODELS
    gpt_config = GGUF_MODELS[GGUFModelName.GPT_OSS_20B]
    assert gpt_config.repo == "unsloth/gpt-oss-20b-GGUF"
    assert gpt_config.filename_pattern == "gpt-oss-20b-{quant}.gguf"  # No UD- prefix
    # GPT-OSS-20B has Q2_K_XL, Q3_K_M, Q4_K_M, Q5_K_M (not Q4_K_XL)
    assert GGUFModelQuantizationLevel.Q2_K_XL in gpt_config.available_quantizations
    assert GGUFModelQuantizationLevel.Q4_K_M in gpt_config.available_quantizations

    # Test GEMMA-3-27B-IT
    assert GGUFModelName.GEMMA_3_27B_IT in GGUF_MODELS
    gemma27_config = GGUF_MODELS[GGUFModelName.GEMMA_3_27B_IT]
    assert gemma27_config.repo == "unsloth/gemma-3-27b-it-GGUF"
    assert gemma27_config.filename_pattern == "gemma-3-27b-it-UD-{quant}.gguf"
    assert GGUFModelQuantizationLevel.Q2_K_XL in gemma27_config.available_quantizations
    assert GGUFModelQuantizationLevel.Q3_K_XL in gemma27_config.available_quantizations


def test_get_stop_tokens() -> None:
    """Test stop tokens for all models."""
    # Test Gemma models
    for model in [GGUFModelName.GEMMA_3_12B_IT, GGUFModelName.GEMMA_3_27B_IT]:
        tokens = get_stop_tokens(model)
        assert "<end_of_turn>" in tokens
        assert "\n" in tokens

    # Test Qwen models
    for model in [GGUFModelName.QWEN3_14B, GGUFModelName.QWEN3_30B, GGUFModelName.QWEN3_30B_Thinking]:
        tokens = get_stop_tokens(model)
        assert "<|im_end|>" in tokens

    # Test GPT-OSS model
    tokens = get_stop_tokens(GGUFModelName.GPT_OSS_20B)
    assert "<|end|>" in tokens
    # GPT-OSS doesn't stop on newline according to the actual implementation


def test_models_fit_16gb_vram() -> None:
    """Test that models configured for 16GB VRAM have appropriate quantizations."""
    # Models that should fit in 16GB VRAM
    models_16gb = {
        GGUFModelName.GPT_OSS_20B: [
            GGUFModelQuantizationLevel.Q2_K_XL,
            GGUFModelQuantizationLevel.Q3_K_M,  # GPT-OSS has Q3_K_M, not Q3_K_XL
            GGUFModelQuantizationLevel.Q4_K_M,  # GPT-OSS has Q4_K_M, not Q4_K_XL
        ],
        GGUFModelName.GEMMA_3_27B_IT: [GGUFModelQuantizationLevel.Q2_K_XL, GGUFModelQuantizationLevel.Q3_K_XL],
    }

    for model, expected_quants in models_16gb.items():
        config = GGUF_MODELS[model]
        for quant in expected_quants:
            assert quant in config.available_quantizations, f"{model} should support {quant} for 16GB VRAM"
