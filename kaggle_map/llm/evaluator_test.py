"""Tests for LLM evaluator module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_map.llm.evaluator import (
    EvaluationConfig,
    _get_optimal_context,
)
from kaggle_map.utils.gguf_model import GGUFModelName, GGUFModelQuantizationLevel


def test_gpt_oss_uses_suggested_context_when_below_openai_minimum():
    """Test that GPT-OSS models use suggested context size even if below OpenAI's 16k recommendation."""
    # Arrange: Mock suggest_ctx_length to return a value below OpenAI's recommendation
    with patch("kaggle_map.llm.evaluator.suggest_ctx_length") as mock_suggest:
        mock_suggest.return_value = 8192  # Below OpenAI's 16384 recommendation

        # Act: Get optimal context for GPT-OSS model
        context = _get_optimal_context(GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q2_K_L)

        # Assert: Should use suggested value but warn about being below recommendation
        assert context == 8192, "GPT-OSS should use suggested context size when it fits in memory"
        mock_suggest.assert_called_once_with(
            vram_gb=16.0,
            model_name=GGUFModelName.GPT_OSS_20B,
            quantization=GGUFModelQuantizationLevel.Q2_K_L,
            desktop_overhead_gb=0.7,
            safety_margin_gb=1.0,
        )


def test_gpt_oss_falls_back_to_16k_minimum_when_insufficient_vram():
    """Test GPT-OSS falls back to OpenAI's 16k minimum context when model won't fit in VRAM."""
    # Arrange: Mock model not fitting in memory
    with patch("kaggle_map.llm.evaluator.suggest_ctx_length") as mock_suggest:
        mock_suggest.return_value = 0  # Indicates model won't fit

        # Act: Get optimal context for GPT-OSS model with insufficient VRAM
        context = _get_optimal_context(GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q3_K_XL)

        # Assert: Should use OpenAI minimum of 16384 for GPT-OSS
        assert context == 16384, "GPT-OSS should use OpenAI minimum (16k) when insufficient VRAM"


def test_gemma_uses_suggested_context_size():
    """Test that GEMMA models use the suggested context size from memory calculations."""
    # Arrange: Mock a reasonable context size for GEMMA
    with patch("kaggle_map.llm.evaluator.suggest_ctx_length") as mock_suggest:
        mock_suggest.return_value = 3072  # Below standard 4096

        # Act: Get optimal context for GEMMA model
        context = _get_optimal_context(GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q3_K_XL)

        # Assert: Should use the suggested value
        assert context == 3072, "GEMMA should use calculated context size when it fits in memory"


def test_gemma_falls_back_to_2k_minimum_when_insufficient_vram():
    """Test GEMMA falls back to conservative 2k minimum to avoid memory issues."""
    # Arrange: Mock model not fitting in memory
    with patch("kaggle_map.llm.evaluator.suggest_ctx_length") as mock_suggest:
        mock_suggest.return_value = 0  # Indicates model won't fit

        # Act: Get optimal context for GEMMA with insufficient VRAM
        context = _get_optimal_context(GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q4_K_M)

        # Assert: Should use conservative minimum for memory-constrained models
        assert context == 2048, "GEMMA should use 2k minimum to avoid OOM errors when VRAM insufficient"


def test_sufficient_context_returns_suggested_value_without_modification():
    """Test that models with sufficient context use the suggested value as-is."""
    # Arrange: Mock ample context size above all recommendations
    with patch("kaggle_map.llm.evaluator.suggest_ctx_length") as mock_suggest:
        mock_suggest.return_value = 32768  # Well above OpenAI's 16k recommendation

        # Act: Get optimal context with plenty of memory
        context = _get_optimal_context(GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q2_K_L)

        # Assert: Should use suggested value without modification
        assert context == 32768, "Should use full suggested context when ample memory available"


@patch("kaggle_map.llm.evaluator.load_llm_model")
@patch("kaggle_map.llm.evaluator.load_validation_data")
@patch("kaggle_map.llm.evaluator._prepare_dataframe")
@patch("kaggle_map.llm.evaluator._sample_dataframe")
@patch("kaggle_map.llm.evaluator.display_evaluation_details")
@patch("kaggle_map.llm.evaluator.save_evaluation_results_to_csv")
def test_gpt_oss_uses_openai_parameters(  # noqa: PLR0913
    mock_save_csv,  # noqa: ARG001
    mock_display,  # noqa: ARG001
    mock_sample,
    mock_prepare,
    mock_load_data,
    mock_load_llm,
):
    """Test that GPT-OSS models use OpenAI recommended inference parameters."""
    import pandas as pd

    from kaggle_map.core.models import EvaluationRow, Prediction
    from kaggle_map.llm.evaluator import evaluate_with_llm

    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "True_Misconception: Misconception 1"}]}
    mock_load_llm.return_value = mock_llm

    # Arrange: Create realistic mathematical test data
    eval_row = EvaluationRow(
        row_id=1,
        question_id=1,
        question_text="What is the result of 2 + 3 × 4?",  # noqa: RUF001
        mc_answer="C) 14",
        student_explanation="I added 2 + 3 first to get 5, then multiplied by 4 to get 20",  # Order of operations misconception
    )
    from kaggle_map.core.models import Category

    ground_truth = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Misconception 1")

    mock_load_data.return_value = [(eval_row, ground_truth)]
    mock_prepare.return_value = pd.DataFrame(
        [
            {
                "row_id": 1,
                "QuestionId": 1,
                "QuestionText": "What is the result of 2 + 3 × 4?",  # noqa: RUF001
                "MC_Answer": "C) 14",
                "StudentExplanation": "I added 2 + 3 first to get 5, then multiplied by 4 to get 20",
                "Category": "True_Misconception",
                "Misconception": "Misconception 1",
            }
        ]
    )
    mock_sample.return_value = mock_prepare.return_value

    # Create config for GPT-OSS
    config = EvaluationConfig(
        template_path=Path("kaggle_map/llm/prompts/predict.j2"),
        data_path=Path("test_data.csv"),
        sample_ratio=1.0,
        row_ids=None,
        model_name=GGUFModelName.GPT_OSS_20B,
        quantization=GGUFModelQuantizationLevel.Q2_K_L,
    )

    # Mock template file
    with patch("pathlib.Path.read_text") as mock_read:
        mock_read.return_value = "{{ question_text }} {{ mc_answer }} {{ student_explanation }}"

        # Run evaluation
        evaluate_with_llm(config)

    # Assert: Verify that llm was called with OpenAI recommended parameters
    mock_llm.assert_called()
    call_args = mock_llm.call_args[1]
    assert call_args["temperature"] == 1.0, "GPT-OSS should use OpenAI recommended temperature of 1.0"
    assert call_args["top_p"] == 1.0, "GPT-OSS should use OpenAI recommended top_p of 1.0"


@patch("kaggle_map.llm.evaluator.load_llm_model")
@patch("kaggle_map.llm.evaluator.load_validation_data")
@patch("kaggle_map.llm.evaluator._prepare_dataframe")
@patch("kaggle_map.llm.evaluator._sample_dataframe")
@patch("kaggle_map.llm.evaluator.display_evaluation_details")
@patch("kaggle_map.llm.evaluator.save_evaluation_results_to_csv")
def test_gemma_uses_standard_parameters(  # noqa: PLR0913
    mock_save_csv,  # noqa: ARG001
    mock_display,  # noqa: ARG001
    mock_sample,
    mock_prepare,
    mock_load_data,
    mock_load_llm,
):
    """Test that non-GPT-OSS models use standard inference parameters."""
    import pandas as pd

    from kaggle_map.core.models import EvaluationRow, Prediction
    from kaggle_map.llm.evaluator import evaluate_with_llm

    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "False_Misconception: Misconception 2"}]}
    mock_load_llm.return_value = mock_llm

    # Arrange: Create realistic mathematical test data
    eval_row = EvaluationRow(
        row_id=2,
        question_id=2,
        question_text="If a rectangle has area 24 and width 4, what is its length?",
        mc_answer="B) 6",
        student_explanation="I multiplied 24 by 4 to get 96 for the length",  # Area formula misconception
    )
    from kaggle_map.core.models import Category

    ground_truth = Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Misconception 2")

    mock_load_data.return_value = [(eval_row, ground_truth)]
    mock_prepare.return_value = pd.DataFrame(
        [
            {
                "row_id": 2,
                "QuestionId": 2,
                "QuestionText": "If a rectangle has area 24 and width 4, what is its length?",
                "MC_Answer": "B) 6",
                "StudentExplanation": "I multiplied 24 by 4 to get 96 for the length",
                "Category": "False_Misconception",
                "Misconception": "Misconception 2",
            }
        ]
    )
    mock_sample.return_value = mock_prepare.return_value

    # Create config for GEMMA
    config = EvaluationConfig(
        template_path=Path("kaggle_map/llm/prompts/predict.j2"),
        data_path=Path("test_data.csv"),
        sample_ratio=1.0,
        row_ids=None,
        model_name=GGUFModelName.GEMMA_3_27B_IT,
        quantization=GGUFModelQuantizationLevel.Q3_K_XL,
    )

    # Mock template file
    with patch("pathlib.Path.read_text") as mock_read:
        mock_read.return_value = "{{ question_text }} {{ mc_answer }} {{ student_explanation }}"

        # Run evaluation
        evaluate_with_llm(config)

    # Assert: Verify that llm was called with standard (non-OpenAI) parameters
    mock_llm.assert_called()
    call_args = mock_llm.call_args[1]
    assert call_args["temperature"] == 0.1, (
        "GEMMA should use lower temperature (0.1) for more deterministic predictions"
    )
    assert call_args["top_p"] == 0.95, "GEMMA should use standard top_p (0.95) for balanced diversity"
