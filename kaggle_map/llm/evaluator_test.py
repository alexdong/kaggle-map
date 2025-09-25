"""Tests for LLM evaluator module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_map.llm.evaluator import EvaluationConfig
from kaggle_map.utils.gguf_model import (
    GGUFModelInferenceConfig,
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
)


def test_model_has_appropriate_default_context():
    """Test that models have appropriate default context sizes."""
    # Test GPT-OSS default
    gpt_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.GPT_OSS_20B)
    assert gpt_config.n_ctx == 32768, "GPT-OSS should have 32k context"
    assert gpt_config.quantization == GGUFModelQuantizationLevel.Q2_K_L

    # Test GEMMA default
    gemma_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.GEMMA_3_27B_IT)
    assert gemma_config.n_ctx == 8192, "GEMMA should have 8k context"
    assert gemma_config.quantization == GGUFModelQuantizationLevel.Q2_K_L

    # Test QWEN3-30B default (should get substantial context)
    qwen_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.QWEN3_30B)
    assert qwen_config.n_ctx == 32768, "QWEN should have 32k context"
    assert qwen_config.quantization == GGUFModelQuantizationLevel.Q4_K_XL


def test_default_model_configs_exist():
    """Test that default model configurations can be retrieved."""
    # Check that get_default_config works for all models
    gpt_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.GPT_OSS_20B)
    assert isinstance(gpt_config, GGUFModelLoadConfig)

    gemma_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.GEMMA_3_27B_IT)
    assert isinstance(gemma_config, GGUFModelLoadConfig)

    qwen_config = GGUFModelLoadConfig.get_default_config(GGUFModelName.QWEN3_30B)
    assert isinstance(qwen_config, GGUFModelLoadConfig)

    # Check inference configs too
    gpt_inf = GGUFModelInferenceConfig.get_default_config(GGUFModelName.GPT_OSS_20B)
    assert isinstance(gpt_inf, GGUFModelInferenceConfig)

    gemma_inf = GGUFModelInferenceConfig.get_default_config(GGUFModelName.GEMMA_3_27B_IT)
    assert isinstance(gemma_inf, GGUFModelInferenceConfig)

    qwen_inf = GGUFModelInferenceConfig.get_default_config(GGUFModelName.QWEN3_30B)
    assert isinstance(qwen_inf, GGUFModelInferenceConfig)


@patch("kaggle_map.llm.evaluator.load_llm_model")
@patch("kaggle_map.llm.evaluator.MAPDataset")
@patch("kaggle_map.llm.evaluator._prepare_dataframe")
@patch("kaggle_map.llm.evaluator._sample_dataframe")
@patch("kaggle_map.llm.evaluator.display_evaluation_details")
@patch("kaggle_map.llm.evaluator.save_evaluation_results_to_csv")
def test_gpt_oss_uses_openai_parameters(  # noqa: PLR0913
    mock_save_csv,  # noqa: ARG001
    mock_display,  # noqa: ARG001
    mock_sample,
    mock_prepare,
    mock_map_dataset,
    mock_load_llm,
):
    """Test that GPT-OSS models use OpenAI recommended inference parameters."""
    import pandas as pd

    from kaggle_map.core.models import Category, EvaluationRow, Prediction
    from kaggle_map.llm.evaluator import evaluate_with_llm

    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "1"}]}
    mock_load_llm.return_value = mock_llm

    # Arrange: Create realistic mathematical test data
    eval_row = EvaluationRow(
        row_id=1,
        question_id=1,
        question_text="What is the result of 2 + 3 × 4?",  # noqa: RUF001
        mc_answer="C) 14",
        student_explanation="I added 2 + 3 first to get 5, then multiplied by 4 to get 20",  # Order of operations misconception
    )

    ground_truth = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Misconception 1")

    dataset_stub = MagicMock()
    dataset_stub.evaluation_pairs.return_value = [(eval_row, ground_truth)]
    dataset_stub.__len__.return_value = 1
    mock_map_dataset.return_value = dataset_stub
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
    data_path = Path(__file__)
    config = EvaluationConfig(
        template_path=Path("kaggle_map/llm/prompts/predict.j2"),
        data_path=data_path,
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
    # GPT-OSS should use OpenAI recommended parameters from get_default_config
    assert call_args["temperature"] == 1.0, "GPT-OSS should use OpenAI recommended temperature of 1.0"
    assert call_args["top_p"] == 1.0, "GPT-OSS should use OpenAI recommended top_p of 1.0"


@patch("kaggle_map.llm.evaluator.load_llm_model")
@patch("kaggle_map.llm.evaluator.MAPDataset")
@patch("kaggle_map.llm.evaluator._prepare_dataframe")
@patch("kaggle_map.llm.evaluator._sample_dataframe")
@patch("kaggle_map.llm.evaluator.display_evaluation_details")
@patch("kaggle_map.llm.evaluator.save_evaluation_results_to_csv")
def test_gemma_uses_standard_parameters(  # noqa: PLR0913
    mock_save_csv,  # noqa: ARG001
    mock_display,  # noqa: ARG001
    mock_sample,
    mock_prepare,
    mock_map_dataset,
    mock_load_llm,
):
    """Test that non-GPT-OSS models use standard inference parameters."""
    import pandas as pd

    from kaggle_map.core.models import Category, EvaluationRow, Prediction
    from kaggle_map.llm.evaluator import evaluate_with_llm

    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "2"}]}
    mock_load_llm.return_value = mock_llm

    # Arrange: Create realistic mathematical test data
    eval_row = EvaluationRow(
        row_id=2,
        question_id=2,
        question_text="If a rectangle has area 24 and width 4, what is its length?",
        mc_answer="B) 6",
        student_explanation="I multiplied 24 by 4 to get 96 for the length",  # Area formula misconception
    )

    ground_truth = Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Misconception 2")

    dataset_stub = MagicMock()
    dataset_stub.evaluation_pairs.return_value = [(eval_row, ground_truth)]
    dataset_stub.__len__.return_value = 1
    mock_map_dataset.return_value = dataset_stub
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
    data_path = Path(__file__)
    config = EvaluationConfig(
        template_path=Path("kaggle_map/llm/prompts/predict.j2"),
        data_path=data_path,
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
    # GEMMA should use lower temperature from get_default_config
    assert call_args["temperature"] == 0.1, (
        "GEMMA should use lower temperature (0.1) for more deterministic predictions"
    )
    assert call_args["top_p"] == 0.95, "GEMMA should use standard top_p (0.95) for balanced diversity"
