"""Tests for LLM utilities and reranker model configurations.

Summary of the UD Issue:
- UD likely stands for "Unsloth Dynamic" (Unsloth's Dynamic 2.0 quantization)
- Examples:
  - gemma-3-27b-it-UD-Q4_K_XL.gguf
  - Qwen3-14B-UD-Q4_K_XL.gguf
  - gpt-oss-20b-Q4_K_XL.gguf
- This naming convention applies to all Unsloth models with XL quantizations
"""

import pytest

from kaggle_map.core.models import Category, Prediction
from kaggle_map.utils.gguf_model import (
    GGUF_MODELS,
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_model_path,
    get_stop_tokens,
    parse_llm_response,
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
    # GPT-OSS-20B has Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M (not Q2_K_XL or Q4_K_XL)
    assert GGUFModelQuantizationLevel.Q2_K_L in gpt_config.available_quantizations
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
            GGUFModelQuantizationLevel.Q2_K_L,  # GPT-OSS has Q2_K_L, not Q2_K_XL
            GGUFModelQuantizationLevel.Q3_K_M,  # GPT-OSS has Q3_K_M, not Q3_K_XL
            GGUFModelQuantizationLevel.Q4_K_M,  # GPT-OSS has Q4_K_M, not Q4_K_XL
        ],
        GGUFModelName.GEMMA_3_27B_IT: [GGUFModelQuantizationLevel.Q2_K_XL, GGUFModelQuantizationLevel.Q3_K_XL],
    }

    for model, expected_quants in models_16gb.items():
        config = GGUF_MODELS[model]
        for quant in expected_quants:
            assert quant in config.available_quantizations, f"{model} should support {quant} for 16GB VRAM"


# =============================================================================
# Tests for LLM Response Parsing
# =============================================================================


def test_parse_llm_response_with_thinking_text():
    """Test parsing when LLM produces thinking text before predictions."""
    # Response with plain text but no valid predictions
    response = "Let me carefully analyze the student's answer and explanation."

    result = parse_llm_response(response, Prediction.parse)

    # Should return empty predictions when no valid predictions found
    assert len(result.predictions) == 0
    assert result.thinking_trace is None


def test_parse_llm_response_with_think_tags():
    """Test parsing when response contains <think> tags."""
    # Response with thinking tags
    response = """<think>
The student selected the correct answer (1/3) * (2/3).
Their explanation mentions "1 / 3 of 2 - 3 = 1 1/3" which seems confused.
They may be mixing up the operation.
</think>

True_Neither:NA True_Misconception:Subtraction True_Correct:NA"""

    result = parse_llm_response(response, Prediction.parse)

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


def test_parse_llm_response_multiline_with_labels():
    """Test parsing when labels are on different lines."""
    response = """Let me analyze this step by step.

The student got the right answer but the explanation is unclear.

True_Neither:NA True_Correct:NA True_Misconception:Division"""

    result = parse_llm_response(response, Prediction.parse)

    # Should find the line with valid categories
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_NEITHER
    assert result.predictions[1].category == Category.TRUE_CORRECT
    assert result.predictions[2].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[2].misconception == "Division"
    assert result.thinking_trace is None


def test_parse_llm_response_mixed_valid_invalid():
    """Test parsing with mix of valid and invalid tokens."""
    response = "Some random text True_Correct:NA more text True_Neither:NA final text True_Misconception:Fractions"

    result = parse_llm_response(response, Prediction.parse)

    # Should extract only the valid predictions
    assert len(result.predictions) == 3
    assert result.predictions[0].category == Category.TRUE_CORRECT
    assert result.predictions[1].category == Category.TRUE_NEITHER
    assert result.predictions[2].category == Category.TRUE_MISCONCEPTION
    assert result.predictions[2].misconception == "Fractions"
    assert result.thinking_trace is None


def test_parse_llm_response_categories_without_colons():
    """Test parsing categories without colons (assumes NA misconception)."""
    response = "True_Correct True_Misconception:Division True_Neither"

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

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

    result = parse_llm_response(response, Prediction.parse)

    # Should extract predictions from final channel only
    assert len(result.predictions) == 3
    assert result.predictions[0].misconception == "Fractions"

    # Should extract analysis as thinking trace
    assert result.thinking_trace == "This is the analysis"


def test_parse_llm_response_without_prediction_parser():
    """Test parsing when no prediction parser is provided."""
    response = """<think>
Some thinking here.
</think>
True_Correct:NA"""

    result = parse_llm_response(response, None)

    # Should extract thinking trace
    assert result.thinking_trace == "Some thinking here."

    # Should return empty predictions when no parser provided
    assert result.predictions == []


def test_static_configs_exist() -> None:
    """Test that static configs exist for common models."""
    # Test GPT-OSS static config
    config = GGUFModelLoadConfig.GPT_OSS_20B
    assert config.model_name == GGUFModelName.GPT_OSS_20B
    assert config.quantization == GGUFModelQuantizationLevel.Q2_K_L
    assert config.n_ctx == 20480
    assert config.n_batch == 512
    assert config.n_gpu_layers == -1
    assert config.n_threads == 8

    # Test GEMMA static config
    config = GGUFModelLoadConfig.GEMMA_3_27B_IT
    assert config.model_name == GGUFModelName.GEMMA_3_27B_IT
    assert config.quantization == GGUFModelQuantizationLevel.Q2_K_L
    assert config.n_ctx == 8192

    # Test Thinking model static config
    config = GGUFModelLoadConfig.QWEN_30B
    assert config.model_name == GGUFModelName.QWEN3_30B
    assert config.quantization == GGUFModelQuantizationLevel.Q4_K_XL
    assert config.n_ctx == 20480


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
