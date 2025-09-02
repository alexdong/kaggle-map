"""LLM-based strategy using GGUF quantized models with llama-cpp-python.

This strategy uses quantized GGUF models (e.g., gemma-3-12b-it Q4_K_M) for
efficient local inference. It processes predictions in batches and uses
XML-structured prompts with context from training data.
"""

import re
import time
from pathlib import Path

from llama_cpp import Llama
from loguru import logger

from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    parse_training_data,
)
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    Answer,
    EvaluationRow,
    Misconception,
    Prediction,
    QuestionId,
    SubmissionRow,
)

from .base import Strategy
from .utils import TRAIN_RATIO, split_training_data

# XML-structured prompt template
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

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize strategy with lazy model loading.

        Args:
            model_path: Path to GGUF model file. If None, uses default Q4_K_M model.
        """
        self.model_path = model_path or "models/gguf/gemma-3-12b-it-Q4_K_M.gguf"
        self.llm = None
        self.correct_answers: dict[QuestionId, Answer] = {}
        self.misconceptions_by_question: dict[QuestionId, list[Misconception]] = {}

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return f"LLM-based prediction using GGUF model: {Path(self.model_path).name}"

    def _load_model(self) -> None:
        """Lazy load the GGUF model."""
        if self.llm is not None:
            return

        logger.info(f"Loading GGUF model from {self.model_path}")

        # Check if model file exists
        model_file = Path(self.model_path)
        assert model_file.exists(), f"Model file not found: {self.model_path}"

        # Load with llama-cpp-python
        self.llm = Llama(
            model_path=str(model_file),
            n_ctx=4096,  # Context window
            n_batch=512,  # Batch size for prompt processing
            n_gpu_layers=-1,  # Use all GPU layers (Metal on Mac, CUDA on GPU)
            verbose=False,  # Quiet mode
            n_threads=8,  # Number of CPU threads
        )

        logger.info(f"Model loaded successfully: {model_file.name}")

    def _build_prompt(self, row: EvaluationRow) -> str:
        """Build XML-structured prompt with training context."""
        # Get correct answer for this question
        correct_answer = self.correct_answers.get(row.question_id, "Unknown")

        # Get top 5 known misconceptions for this question
        misconceptions = self.misconceptions_by_question.get(row.question_id, [])[:5]
        known_misconceptions = ", ".join(misconceptions) if misconceptions else "None identified"

        # Format the prompt
        return PROMPT_TEMPLATE.format(
            question=row.question_text,
            correct_answer=correct_answer,
            known_misconceptions=known_misconceptions,
            student_answer=row.mc_answer,
            student_explanation=row.student_explanation,
        )

    def _parse_response(self, response: str, row: EvaluationRow) -> str:
        """Parse LLM response with comprehensive error logging."""
        logger.debug(f"Parsing response for row {row.row_id}")
        logger.debug(f"Raw response: {response}")

        # Try to extract Category:Misconception pattern
        pattern = r"(True|False)_(Correct|Misconception|Neither):(\w+|NA)"
        match = re.search(pattern, response)

        if not match:
            # Log comprehensive error context
            logger.error(
                "Failed to parse LLM response",
                row_id=row.row_id,
                question_id=row.question_id,
                response_length=len(response),
                response_preview=response[:200],
                response_full=response,
                student_answer=row.mc_answer,
                student_explanation=row.student_explanation[:100],
                question_text=row.question_text[:100],
            )
            # Let it crash with detailed context
            msg = f"Could not parse response for row {row.row_id}. Response: {response}"
            raise ValueError(msg)

        category_misconception = f"{match.group(1)}_{match.group(2)}:{match.group(3)}"
        logger.debug(f"Successfully parsed: {category_misconception}")
        return category_misconception

    def predict_batch(self, rows: list[EvaluationRow], batch_size: int = 8) -> list[SubmissionRow]:
        """Process predictions in batches for efficiency."""
        # Ensure model is loaded
        self._load_model()

        results = []
        total_batches = (len(rows) + batch_size - 1) // batch_size

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")

            # Process each prompt in the batch
            for row in batch:
                prompt = self._build_prompt(row)

                # Generate prediction
                start_time = time.time()
                output = self.llm(
                    prompt,
                    max_tokens=30,  # Just enough for "Category:Misconception"
                    temperature=0.3,
                    stop=["</instructions>", "\n\n"],
                    echo=False,  # Don't include prompt in output
                )
                inference_time = time.time() - start_time

                logger.debug(f"Inference time for row {row.row_id}: {inference_time:.2f}s")

                # Extract response text
                response = output["choices"][0]["text"].strip()

                # Parse response
                try:
                    prediction = self._parse_response(response, row)
                    results.append(SubmissionRow(row_id=row.row_id, predicted_categories=[prediction]))
                except ValueError as e:
                    logger.error(f"Failed to parse response for row {row.row_id}: {e}")
                    # Use fallback prediction
                    fallback = "False_Misconception:Unknown"
                    logger.warning(f"Using fallback prediction for row {row.row_id}: {fallback}")
                    results.append(SubmissionRow(row_id=row.row_id, predicted_categories=[fallback]))

        logger.info(f"Completed batch processing of {len(results)} predictions")
        return results

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        model_path: str | None = None,
    ) -> "LLMStrategy":
        """Load training data to extract correct answers and misconceptions."""
        logger.info("Fitting LLM strategy")
        logger.info(f"Loading training data from {train_csv_path}")

        # Parse training data
        all_training_data = parse_training_data(train_csv_path)
        logger.info(f"Parsed {len(all_training_data)} total training rows")

        # Split the data
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Extract knowledge from training data
        correct_answers = extract_correct_answers(train_data)
        logger.info(f"Extracted correct answers for {len(correct_answers)} questions")

        misconceptions_by_question = extract_misconceptions_by_popularity(train_data)
        logger.info(f"Extracted misconceptions for {len(misconceptions_by_question)} questions")

        # Create instance with optional model path
        strategy = cls(model_path=model_path)
        strategy.correct_answers = correct_answers
        strategy.misconceptions_by_question = misconceptions_by_question

        return strategy

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make prediction on a single evaluation row."""
        logger.debug(f"Making LLM prediction for row {evaluation_row.row_id}")
        # Use batch processing with batch size of 1
        results = self.predict_batch([evaluation_row], batch_size=1)
        return results[0]

    def save(self, filepath: Path) -> None:
        """Save strategy state (not the model itself)."""
        import pickle

        logger.info(f"Saving LLM strategy state to {filepath}")

        # Save only the extracted knowledge and model path
        state = {
            "correct_answers": self.correct_answers,
            "misconceptions_by_question": self.misconceptions_by_question,
            "model_path": self.model_path,
        }

        with filepath.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Path) -> "LLMStrategy":
        """Load strategy state from disk."""
        import pickle

        logger.info(f"Loading LLM strategy state from {filepath}")

        with filepath.open("rb") as f:
            state = pickle.load(f)

        strategy = cls(model_path=state.get("model_path"))
        strategy.correct_answers = state["correct_answers"]
        strategy.misconceptions_by_question = state["misconceptions_by_question"]

        return strategy

    @classmethod
    def evaluate_on_split(
        cls,
        model: "LLMStrategy",
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        sample_size: int | None = 10,  # Small sample for testing
    ) -> dict[str, float]:
        """Evaluate model on validation split with optional sampling."""
        logger.info("Evaluating LLM strategy on validation split")

        # Ensure model is loaded
        if model.llm is None:
            model._load_model()

        # Parse all training data
        all_training_data = parse_training_data(train_csv_path)

        # Split the data
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )

        # Sample validation data if requested
        if sample_size is not None and sample_size < len(val_data):
            import random

            random.seed(random_seed)
            val_data = random.sample(val_data, sample_size)
            logger.info(f"Sampled {sample_size} validation rows for evaluation")

        # Convert to evaluation rows
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

        # Make predictions in batches
        predictions = model.predict_batch(eval_rows, batch_size=8)

        # Extract ground truth
        ground_truth = {row.row_id: str(Prediction.from_ground_truth_row(row)) for row in val_data}

        # Convert predictions to format for metric calculation
        predicted = {pred.row_id: pred.predicted_categories for pred in predictions}

        # Calculate MAP@3
        map_score = calculate_map_at_3(predicted, ground_truth)

        logger.info(f"Evaluation complete - MAP@3: {map_score:.4f}")

        return {"map_at_3": map_score, "num_samples": len(val_data)}
