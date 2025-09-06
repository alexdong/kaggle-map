"""LLM-based strategy using GGUF quantized models with llama-cpp-python.

This strategy uses quantized GGUF models (e.g., gemma-3-12b-it Q4_K_M) for
efficient local inference. It processes predictions in batches and uses
XML-structured prompts with context from training data.
"""

import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    load_training_data,
)
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    EvaluationRow,
    LLMModelLoadConfig,
    Prediction,
    QuestionId,
    SubmissionRow,
)
from kaggle_map.reranker.llm_utils import format_chat_prompt, get_model_path, get_stop_tokens, load_llm_model

from .base import Strategy
from .utils import TRAIN_RATIO, VAL_RATIO, split_training_data

if TYPE_CHECKING:
    from llama_cpp import Llama

PROMPT_TEMPLATE = """<task>
Analyze a student's math answer and identify their misconception.
</task>

<question>
{question}
</question>

<correct_answer>
{correct_answer}
</correct_answer>

<known_misconceptions>
{known_misconceptions}
</known_misconceptions>

<student_work>
<answer>{student_answer}</answer>
<explanation>{student_explanation}</explanation>
</student_work>

<categories>
- True_Correct: Student got the right answer with valid reasoning
- True_Misconception: Student got the right answer despite having a misconception
- True_Neither: Student got the right answer but reasoning is unclear
- False_Correct: Student got wrong answer due to calculation error only
- False_Misconception: Student got wrong answer due to conceptual misunderstanding
- False_Neither: Student got wrong answer with unclear reasoning
</categories>

<instructions>
1. Compare student answer to correct answer
2. Analyze the explanation for mathematical errors
3. Match error pattern to known misconceptions if applicable
4. Select the appropriate category

Respond with ONLY the category and misconception in this exact format:
Category:Misconception
</instructions>"""


class LLMStrategy(Strategy):
    """LLM-based misconception prediction using GGUF quantized models."""

    def __init__(self, config: LLMModelLoadConfig) -> None:
        self.config = config
        self.model_path = get_model_path(self.config.model_name, self.config.quantization)
        self.llm: Llama | None = None
        # Store additional context for questions (correct answers, known misconceptions)
        self.question_contexts: dict[QuestionId, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return f"LLM-based prediction using {self.config.model_name} ({self.config.quantization})"

    def load_model(self) -> None:
        """Lazy load the GGUF model, downloading if necessary."""
        if self.llm is None:
            model_context = load_llm_model(self.config)
            self.llm = model_context.__enter__()

    def _build_prompt(self, row: EvaluationRow) -> str:
        """Build prompt with model-specific chat template format."""
        correct_answer = row.correct_answer or "Unknown"
        misconceptions = row.known_misconceptions or []
        known_misconceptions = ", ".join(misconceptions[:5]) if misconceptions else "None identified"
        user_prompt = PROMPT_TEMPLATE.format(
            question=row.question_text,
            correct_answer=correct_answer,
            known_misconceptions=known_misconceptions,
            student_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )
        return format_chat_prompt(self.config.model_name, user_prompt)

    def _parse_response(self, response: str, row: EvaluationRow) -> str:
        """Parse LLM response to extract category and misconception.

        Handles different response formats including Qwen3's thinking tags.
        """
        logger.debug(f"Parsing response for row {row.row_id}: {response}")

        # Handle Qwen3's thinking mode - strip thinking tags if present
        if "<think>" in response:
            # Extract content after thinking section
            if "</think>" in response:
                # Get content after the closing think tag
                parts = response.split("</think>", 1)
                if len(parts) > 1:
                    response = parts[1].strip()
                else:
                    # No content after thinking tag, might be incomplete
                    logger.warning(f"Incomplete thinking response for row {row.row_id}, attempting to parse anyway")
            else:
                # Thinking tag not closed, response might be cut off
                logger.warning(f"Unclosed thinking tag in response for row {row.row_id}")
                # Try to extract any content that might be there
                response = response.replace("<think>", "").strip()

        # If response is empty or just whitespace after cleaning, provide a default
        if not response or response == "<think>":
            logger.warning(f"Empty or incomplete response for row {row.row_id}, using default")
            return "False_Neither:NA"

        cleaned_response = response.replace("Category:", "").strip()
        pattern = r"(True|False)_(Correct|Misconception|Neither):?([\w-]+|NA)?"
        match = re.search(pattern, cleaned_response)

        if not match:
            # Try to be more lenient - look for just True/False patterns
            simple_pattern = r"(True|False)[_\s]*(Correct|Misconception|Neither)"
            match = re.search(simple_pattern, cleaned_response, re.IGNORECASE)
            if match:
                category = f"{match.group(1).title()}_{match.group(2).title()}"
                misconception = "NA"
                result = f"{category}:{misconception}"
                logger.debug(f"Successfully parsed with lenient matching: {result}")
                return result

            # If still no match, log the response and return a default
            logger.error(f"Failed to parse response for row {row.row_id}: '{response}'")
            return "False_Neither:NA"

        category = f"{match.group(1)}_{match.group(2)}"
        misconception = match.group(3) if match.group(3) else "NA"
        result = f"{category}:{misconception}"
        logger.debug(f"Successfully parsed: {result}")
        return result

    def _enrich_row(self, row: EvaluationRow) -> EvaluationRow:
        """Enrich evaluation row with context from training data."""
        if context := self.question_contexts.get(row.question_id):
            row.correct_answer = context.get("correct_answer")
            row.known_misconceptions = context.get("known_misconceptions")
        return row

    def _process_single_row(self, row: EvaluationRow) -> SubmissionRow:
        """Process a single row through the LLM pipeline."""
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(row)
        logger.debug(f"Processing row {row.row_id} (prompt: {len(prompt)} chars)")

        # Run inference
        output = self.llm(
            prompt,
            max_tokens=50,
            temperature=0.1,
            stop=get_stop_tokens(self.config.model_name),
            echo=False,
        )

        # Extract and parse response
        output_dict = dict(output) if hasattr(output, "__iter__") else output
        response = output_dict["choices"][0]["text"].strip()  # type: ignore

        elapsed = time.time() - start_time
        logger.info(f"Row {row.row_id}: '{response}' ({elapsed:.2f}s)")

        # Parse prediction
        prediction = self._parse_response(response, row)
        category, misconception = prediction.split(":", 1) if ":" in prediction else (prediction, "NA")

        return SubmissionRow(
            row_id=row.row_id,
            predicted_categories=[Prediction(category=category, misconception=misconception)],
        )

    def predict_batch(self, rows: list[EvaluationRow], batch_size: int = 8) -> list[SubmissionRow]:
        """Process predictions with parallel data preparation.

        Note: LLM inference remains sequential due to model locking, but we parallelize
        data enrichment and optimize the pipeline for clarity and maintainability.
        """
        self.load_model()
        assert self.llm is not None, "LLM model not loaded"

        logger.info(f"Processing {len(rows)} rows")

        # Handle empty input
        if not rows:
            return []

        # Parallel data enrichment (I/O bound, benefits from threading)
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(rows)))) as executor:
            enriched_rows = list(executor.map(self._enrich_row, rows))

        # Sequential LLM inference (model locks during inference)
        # Process in logical batches for progress reporting
        results = []
        total_batches = (len(enriched_rows) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(enriched_rows), batch_size), 1):
            batch = enriched_rows[i : i + batch_size]
            logger.info(f"Batch {batch_num}/{total_batches}: processing {len(batch)} samples")

            # Process each row in the batch
            batch_results = [self._process_single_row(row) for row in batch]
            results.extend(batch_results)

        logger.info(f"Completed processing {len(results)} predictions")
        return results

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        config: LLMModelLoadConfig,
    ) -> "LLMStrategy":
        """Load training data to extract correct answers and misconceptions."""
        logger.info("Fitting LLM strategy")
        logger.info(f"Loading training data from {train_csv_path}")

        all_training_data = load_training_data(train_csv_path)
        logger.info(f"Parsed {len(all_training_data)} total training rows")

        # When train_split is 1.0, set val_ratio to 0 to avoid exceeding 1.0 total
        val_ratio = 0.0 if train_split == 1.0 else VAL_RATIO
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, val_ratio=val_ratio, random_seed=random_seed
        )
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        correct_answers = extract_correct_answers(train_data)
        logger.info(f"Extracted correct answers for {len(correct_answers)} questions")

        misconceptions_by_question = extract_misconceptions_by_popularity(train_data)
        logger.info(f"Extracted misconceptions for {len(misconceptions_by_question)} questions")

        strategy = cls(config=config)

        # Build problem contexts for all questions
        for question_id in correct_answers:
            strategy.question_contexts[question_id] = {
                "correct_answer": correct_answers[question_id],
                "known_misconceptions": misconceptions_by_question.get(question_id, []),
            }

        return strategy

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make prediction on a single evaluation row."""
        logger.debug(f"Making LLM prediction for row {evaluation_row.row_id}")
        results = self.predict_batch([evaluation_row], batch_size=1)
        return results[0]

    def save(self, filepath: Path) -> None:
        """Save strategy state (not the model itself)."""
        logger.info(f"Saving LLM strategy state to {filepath}")

        state = {
            "question_contexts": self.question_contexts,
            "config": self.config,
        }

        with filepath.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Path) -> "LLMStrategy":
        """Load strategy state from disk."""
        logger.info(f"Loading LLM strategy state from {filepath}")

        with filepath.open("rb") as f:
            state = pickle.load(f)

        strategy = cls(config=state.get("config"))
        strategy.question_contexts = state["question_contexts"]

        return strategy

    @classmethod
    def evaluate(
        cls,
        model: "LLMStrategy",
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
    ) -> dict[str, float]:
        """Evaluate model on validation split with optional sampling."""
        logger.info("Evaluating LLM strategy on validation split")

        # Ensure model is loaded
        if model.llm is None:
            model.load_model()
        all_training_data = load_training_data(train_csv_path)

        # When train_split is 1.0, set val_ratio to 0 to avoid exceeding 1.0 total
        val_ratio = 1.0 if train_split == 0.0 else VAL_RATIO
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, val_ratio=val_ratio, random_seed=random_seed
        )
        assert val_data, "Validation split is empty - cannot evaluate"

        # Sample validation data if specified
        eval_rows = [
            EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )
            for row in val_data
        ]

        predictions = model.predict_batch(eval_rows, batch_size=8)
        ground_truth = {row.row_id: str(row.prediction) for row in val_data}
        predicted = {pred.row_id: pred.predicted_categories for pred in predictions}
        map_score = calculate_map_at_3(predicted, ground_truth)
        logger.info(f"Evaluation complete - MAP@3: {map_score:.4f}")
        return {"map_at_3": map_score, "num_samples": len(val_data)}
