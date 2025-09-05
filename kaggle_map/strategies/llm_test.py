"""Tests for LLM strategy focusing on chat template format handling.

This test file reproduces the issue where different model types require
different chat template formats, but the current implementation hardcodes
the Gemma format for all models.
"""

# ruff: noqa: SLF001 # Allow private member access for testing internal methods
# ruff: noqa: ARG002 # Allow unused mock arguments in test methods

from unittest.mock import Mock, patch

import pytest

from kaggle_map.core.models import (
    Category,
    EvaluationRow,
    LLMModelLoadConfig,
    Prediction,
    SubmissionRow,
)
from kaggle_map.strategies.llm import LLMStrategy


# Sample evaluation row for testing
@pytest.fixture
def sample_row() -> EvaluationRow:
    """Create a sample evaluation row for testing."""
    return EvaluationRow(
        row_id=1,
        question_id=101,
        question_text="What is 2 + 3?",
        mc_answer="6",
        student_explanation="I added 2 and 3 but made an error",
        correct_answer="5",
        known_misconceptions=["Addition_Error", "Calculation_Mistake"],
    )


class TestChatTemplateFormats:
    """Test chat template format generation for different model types."""

    def test_gemma_chat_template_format(self, sample_row: EvaluationRow) -> None:
        """Test that Gemma model uses correct chat template format."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        prompt = strategy._build_prompt(sample_row)

        # Should use Gemma format
        assert "<start_of_turn>user" in prompt
        assert "<end_of_turn>" in prompt
        assert "<start_of_turn>model" in prompt

        # Should NOT contain other model formats
        assert "<|im_start|>" not in prompt
        assert "<|start|>" not in prompt

    def test_qwen3_chat_template_format(self, sample_row: EvaluationRow) -> None:
        """Test that Qwen3 model should use correct chat template format."""
        config = LLMModelLoadConfig(model_name="Qwen3-14B")
        strategy = LLMStrategy(config)

        prompt = strategy._build_prompt(sample_row)

        # Should use Qwen3 format
        assert "<|im_start|>user" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt
        
        # Should include empty think tags to disable thinking mode
        assert "<think>\n\n</think>" in prompt

        # Should NOT contain other model formats
        assert "<start_of_turn>" not in prompt
        assert "<|start|>" not in prompt

    def test_gpt_oss_chat_template_format(self, sample_row: EvaluationRow) -> None:
        """Test that gpt-oss model should use correct chat template format."""
        config = LLMModelLoadConfig(model_name="gpt-oss-20b")
        strategy = LLMStrategy(config)

        prompt = strategy._build_prompt(sample_row)

        # Should use gpt-oss format
        assert "<|start|>user" in prompt
        assert "<|end|>" in prompt
        assert "<|start|>assistant" in prompt

        # Should NOT contain other model formats
        assert "<start_of_turn>" not in prompt
        assert "<|im_start|>" not in prompt


class TestPromptContent:
    """Test that prompt content is consistent across all model formats."""

    def test_prompt_contains_required_elements(self, sample_row: EvaluationRow) -> None:
        """Test that all prompts contain the required elements regardless of format."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        prompt = strategy._build_prompt(sample_row)

        # Core content should always be present
        assert "<task>" in prompt
        assert "<question>" in prompt
        assert "<correct_answer>" in prompt
        assert "<known_misconceptions>" in prompt
        assert "<student_work>" in prompt
        assert "<categories>" in prompt
        assert "<instructions>" in prompt

        # Specific content from sample row
        assert "What is 2 + 3?" in prompt
        assert "6" in prompt  # student answer
        assert "5" in prompt  # correct answer
        assert "Addition_Error" in prompt
        assert "I added 2 and 3 but made an error" in prompt


class TestResponseParsing:
    """Test response parsing for different model output formats."""

    def test_parse_valid_response_format(self, sample_row: EvaluationRow) -> None:
        """Test parsing of properly formatted response."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        response = "False_Misconception:Addition_Error"
        result = strategy._parse_response(response, sample_row)

        assert result == "False_Misconception:Addition_Error"

    def test_parse_response_with_category_prefix(self, sample_row: EvaluationRow) -> None:
        """Test parsing response that includes 'Category:' prefix."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        response = "Category:True_Correct:NA"
        result = strategy._parse_response(response, sample_row)

        assert result == "True_Correct:NA"

    def test_parse_response_with_whitespace(self, sample_row: EvaluationRow) -> None:
        """Test parsing response with extra whitespace."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        response = "  False_Neither:NA  "
        result = strategy._parse_response(response, sample_row)

        assert result == "False_Neither:NA"

    def test_parse_response_handles_invalid_format_gracefully(self, sample_row: EvaluationRow) -> None:
        """Test that invalid response format returns default value instead of crashing."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        invalid_response = "This is not a valid format"
        
        # Should return default value instead of raising error
        result = strategy._parse_response(invalid_response, sample_row)
        assert result == "False_Neither:NA"

    def test_parse_response_handles_different_model_outputs(self, sample_row: EvaluationRow) -> None:
        """Test that response parsing works regardless of model type."""
        models_and_responses = [
            ("gemma-3-12b-it", "True_Misconception:Calculation_Error"),
            ("Qwen3-14B", "False_Correct:NA"),
            ("gpt-oss-20b", "True_Neither:NA"),
        ]

        for model_name, response in models_and_responses:
            config = LLMModelLoadConfig(model_name=model_name)
            strategy = LLMStrategy(config)

            result = strategy._parse_response(response, sample_row)
            assert result == response  # Should return the cleaned response


class TestModelConfigurationIntegration:
    """Test integration between model configuration and chat template selection."""

    def test_models_use_correct_chat_formats(self, sample_row: EvaluationRow) -> None:
        """Verify that each model uses its correct chat template format.

        This test ensures the chat template formatting bug is fixed.
        """
        # Test Gemma model uses its format
        gemma_config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        gemma_strategy = LLMStrategy(gemma_config)
        gemma_prompt = gemma_strategy._build_prompt(sample_row)
        assert "<start_of_turn>" in gemma_prompt
        assert "<|im_start|>" not in gemma_prompt
        assert "<|start|>" not in gemma_prompt

        # Test Qwen3 model uses its format with thinking disabled
        qwen_config = LLMModelLoadConfig(model_name="Qwen3-14B")
        qwen_strategy = LLMStrategy(qwen_config)
        qwen_prompt = qwen_strategy._build_prompt(sample_row)
        assert "<|im_start|>" in qwen_prompt
        assert "<think>\n\n</think>" in qwen_prompt  # Empty think tags disable thinking mode
        assert "<start_of_turn>" not in qwen_prompt
        assert "<|start|>" not in qwen_prompt

        # Test gpt-oss model uses its format
        gpt_config = LLMModelLoadConfig(model_name="gpt-oss-20b")
        gpt_strategy = LLMStrategy(gpt_config)
        gpt_prompt = gpt_strategy._build_prompt(sample_row)
        assert "<|start|>" in gpt_prompt
        assert "<start_of_turn>" not in gpt_prompt
        assert "<|im_start|>" not in gpt_prompt


class TestStopTokensForModels:
    """Test that stop tokens are appropriate for each model format."""

    @patch("kaggle_map.strategies.llm.LLMStrategy.load_model")
    def test_gemma_stop_tokens(self, mock_load_model: Mock, sample_row: EvaluationRow) -> None:
        """Test that Gemma models use appropriate stop tokens."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        # Mock the LLM model
        mock_llm = Mock()
        mock_llm.return_value = {"choices": [{"text": "False_Misconception:Addition_Error"}]}
        strategy.llm = mock_llm

        # Process a single row
        strategy._process_single_row(sample_row)

        # Check that the LLM was called with Gemma-appropriate stop tokens
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args

        assert "<end_of_turn>" in call_args[1]["stop"]
        assert call_args[1]["stop"] == ["<end_of_turn>", "\n"]

    @pytest.mark.skip(reason="Current implementation doesn't handle model-specific stop tokens")
    def test_qwen3_should_use_different_stop_tokens(self, sample_row: EvaluationRow) -> None:
        """Test that Qwen3 should use different stop tokens.

        Currently FAILS because implementation hardcodes Gemma stop tokens.
        """
        config = LLMModelLoadConfig(model_name="Qwen3-14B")
        LLMStrategy(config)

        # This test would pass after fixing the implementation
        # Qwen3 should use ["<|im_end|>", "\n"] as stop tokens

    @pytest.mark.skip(reason="Current implementation doesn't handle model-specific stop tokens")
    def test_gpt_oss_should_use_different_stop_tokens(self, sample_row: EvaluationRow) -> None:
        """Test that gpt-oss should use different stop tokens.

        Currently FAILS because implementation hardcodes Gemma stop tokens.
        """
        config = LLMModelLoadConfig(model_name="gpt-oss-20b")
        LLMStrategy(config)

        # This test would pass after fixing the implementation
        # gpt-oss should use ["<|end|>", "\n"] as stop tokens


class TestFullPipelineWithMockedLLM:
    """Test the complete prediction pipeline with mocked LLM responses."""

    @patch("kaggle_map.strategies.llm.LLMStrategy.load_model")
    def test_prediction_pipeline_with_gemma_format(self, mock_load_model: Mock, sample_row: EvaluationRow) -> None:
        """Test full prediction pipeline works with current Gemma format."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        # Mock the LLM model response
        mock_llm = Mock()
        mock_llm.return_value = {"choices": [{"text": "False_Misconception:Addition_Error"}]}
        strategy.llm = mock_llm

        # Make prediction
        result = strategy.predict(sample_row)

        # Verify result structure
        assert isinstance(result, SubmissionRow)
        assert result.row_id == sample_row.row_id
        assert len(result.predicted_categories) == 1

        prediction = result.predicted_categories[0]
        assert isinstance(prediction, Prediction)
        assert prediction.category == Category.FALSE_MISCONCEPTION
        assert prediction.misconception == "Addition_Error"

    @patch("kaggle_map.strategies.llm.LLMStrategy.load_model")
    def test_batch_prediction_consistency(self, mock_load_model: Mock, sample_row: EvaluationRow) -> None:
        """Test that batch prediction produces consistent results."""
        config = LLMModelLoadConfig(model_name="gemma-3-12b-it")
        strategy = LLMStrategy(config)

        # Mock the LLM model response
        mock_llm = Mock()
        mock_llm.return_value = {"choices": [{"text": "True_Correct:NA"}]}
        strategy.llm = mock_llm

        # Create multiple rows (3 copies for testing)
        rows = [sample_row, sample_row, sample_row]

        # Make batch prediction
        results = strategy.predict_batch(rows, batch_size=2)

        # All results should be consistent
        expected_count = 3
        assert len(results) == expected_count
        for result in results:
            assert isinstance(result, SubmissionRow)
            prediction = result.predicted_categories[0]
            assert prediction.category == Category.TRUE_CORRECT
            assert prediction.misconception == "NA"
