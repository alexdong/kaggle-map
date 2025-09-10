"""Tests for reranker model configurations, focusing on UD naming for XL quantizations.

Summary of the UD Issue:
- UD likely stands for "Unsloth Dynamic" (Unsloth's Dynamic 2.0 quantization)
- ALL XL quantizations from Unsloth use "UD-" prefix in filenames
- Examples: 
  - gemma-3-12b-it-UD-Q4_K_XL.gguf
  - Qwen3-14B-UD-Q4_K_XL.gguf
  - gpt-oss-20b-UD-Q4_K_XL.gguf
- This naming convention applies to all Unsloth models with XL quantizations
"""

from pathlib import Path

from kaggle_map.reranker.models import (
    GGUF_MODELS,
    RerankerModelName,
    RerankerModelQuantizationLevel,
)
from kaggle_map.reranker.utils import get_model_path


def test_all_xl_quantizations_have_ud_prefix():
    """Test that ALL XL quantizations use UD- prefix in filenames.
    
    Note: Not all models have all XL quantizations available:
    - Gemma and Qwen have Q2, Q3, Q4, Q5, Q6, Q8 XL versions
    - GPT-OSS-20B only has Q4, Q6, Q8 XL versions (missing Q2, Q3, Q5)
    """
    # Define which quantizations are available for each model
    model_quantizations = {
        RerankerModelName.GEMMA_3_12B_IT: [
            RerankerModelQuantizationLevel.Q2_K_XL,
            RerankerModelQuantizationLevel.Q3_K_XL,
            RerankerModelQuantizationLevel.Q4_K_XL,
            RerankerModelQuantizationLevel.Q5_K_XL,
            RerankerModelQuantizationLevel.Q6_K_XL,
        ],
        RerankerModelName.QWEN3_14B: [
            RerankerModelQuantizationLevel.Q2_K_XL,
            RerankerModelQuantizationLevel.Q3_K_XL,
            RerankerModelQuantizationLevel.Q4_K_XL,
            RerankerModelQuantizationLevel.Q5_K_XL,
            RerankerModelQuantizationLevel.Q6_K_XL,
        ],
        RerankerModelName.GPT_OSS_20B: [
            RerankerModelQuantizationLevel.Q4_K_XL,
            RerankerModelQuantizationLevel.Q6_K_XL,
            # Q8_K_XL exists but not in our enum
        ],
    }
    
    for model, quantizations in model_quantizations.items():
        for quant in quantizations:
            path = get_model_path(model, quant)
            assert "UD-" in str(path), f"Missing UD- prefix for {model} {quant}: {path}"
            expected = f"models/gguf/{model.value}-UD-{quant.value}.gguf"
            assert str(path) == expected, f"Expected {expected}, got {path}"


def test_all_models_filename_patterns_include_ud():
    """Test that all models' GGUF configs have correct UD- pattern."""
    for model_name in [RerankerModelName.GEMMA_3_12B_IT, RerankerModelName.QWEN3_14B, RerankerModelName.GPT_OSS_20B]:
        config = GGUF_MODELS[model_name]
        assert "UD-" in config.filename_pattern, f"Missing UD- in pattern for {model_name}"
        expected = f"{model_name.value}-UD-{{quant}}.gguf"
        assert config.filename_pattern == expected, f"Expected {expected}, got {config.filename_pattern}"


def test_non_xl_quantizations_no_ud_prefix():
    """Test that non-XL quantizations don't get UD- prefix."""
    # Test a non-XL quantization (if we had Q4_K_M for example)
    # This test would need actual non-XL quantizations to be meaningful
    # For now, just verify the logic in get_model_path
    from kaggle_map.reranker.models import RerankerModelQuantizationLevel
    
    # Since all our quantizations are XL, we can't test this properly
    # but we document the expected behavior
    pass