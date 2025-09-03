"""LLM-based strategy using GGUF quantized models with llama-cpp-python.

This strategy uses quantized GGUF models (e.g., gemma-3-12b-it Q4_K_M) for
efficient local inference. It processes predictions in batches and uses
XML-structured prompts with context from training data.
"""

import pickle
import random
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


type QuantizationType = str
type ModelType = str

# Type -> size_gb
QUANTIZATION_OPTIONS: dict[QuantizationType, float] = {
    "IQ4_XS": 6.55,
    "IQ4_NL": 6.89,
    "Q4_0": 6.91,
    "Q4_1": 7.56,
    "Q4_K_S": 6.94,
    "Q4_K_M": 7.30,
    "Q4_K_XL": 7.43,
}

MODEL_OPTIONS: dict[ModelType, str] = {
    "gemma-3-12b-it": "gemma-3-12b-it",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "qwen-2.5-14b-instruct": "qwen-2.5-14b-instruct",
}


class LLMStrategy(Strategy):
    """LLM-based misconception prediction using GGUF quantized models."""

    def __init__(
        self, 
        model_type: ModelType = "gemma-3-12b-it",
        quantization_type: QuantizationType = "Q4_K_M",
        model_path: str | None = None
    ) -> None:
        """Initialize strategy with lazy model loading.

        Args:
            model_type: Type of model to use (e.g., "gemma-3-12b-it")
            quantization_type: Quantization type (e.g., "Q4_K_M")
            model_path: Explicit path to GGUF model file. If provided, overrides model_type and quantization_type.
        """
        if model_path is not None:
            self.model_path = model_path
        else:
            model_name = MODEL_OPTIONS.get(model_type, model_type)
            self.model_path = f"models/gguf/{model_name}-{quantization_type}.gguf"
        
        self.model_type = model_type
        self.quantization_type = quantization_type
        self.llm: Llama | None = None
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

        model_file = Path(self.model_path)
        assert model_file.exists(), f"Model file not found: {self.model_path}"

        self.llm = Llama(
            model_path=str(model_file),
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,  # Use all GPU layers (Metal on Mac, CUDA on GPU)
            verbose=False,
            n_threads=8,
        )

        logger.info(f"Model loaded successfully: {model_file.name}")

    def _build_prompt(self, row: EvaluationRow) -> str:
        """Build XML-structured prompt with training context."""
        correct_answer = self.correct_answers.get(row.question_id, "Unknown")

        misconceptions = self.misconceptions_by_question.get(row.question_id, [])[:5]
        known_misconceptions = ", ".join(misconceptions) if misconceptions else "None identified"

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

        pattern = r"(True|False)_(Correct|Misconception|Neither):(\w+|NA)"
        match = re.search(pattern, response)

        if not match:
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
            msg = f"Could not parse response for row {row.row_id}. Response: {response}"
            raise ValueError(msg)

        category_misconception = f"{match.group(1)}_{match.group(2)}:{match.group(3)}"
        logger.debug(f"Successfully parsed: {category_misconception}")
        return category_misconception

    def predict_batch(self, rows: list[EvaluationRow], batch_size: int = 8) -> list[SubmissionRow]:
        """Process predictions in batches for efficiency."""
        self._load_model()

        results = []
        total_batches = (len(rows) + batch_size - 1) // batch_size

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")

            for row in batch:
                prompt = self._build_prompt(row)

                start_time = time.time()
                output = self.llm(
                    prompt,
                    max_tokens=30,  # Just enough for "Category:Misconception"
                    temperature=0.3,
                    stop=["</instructions>", "\n\n"],
                    echo=False,
                )
                inference_time = time.time() - start_time

                logger.debug(f"Inference time for row {row.row_id}: {inference_time:.2f}s")

                response = output["choices"][0]["text"].strip()

                try:
                    prediction = self._parse_response(response, row)
                    pred_obj = Prediction(category_misconception=prediction)
                    results.append(SubmissionRow(row_id=row.row_id, predicted_categories=[pred_obj]))
                except ValueError as e:
                    logger.error(f"Failed to parse response for row {row.row_id}: {e}")
                    fallback = "False_Misconception:Unknown"
                    logger.warning(f"Using fallback prediction for row {row.row_id}: {fallback}")
                    fallback_pred = Prediction(category_misconception=fallback)
                    results.append(SubmissionRow(row_id=row.row_id, predicted_categories=[fallback_pred]))

        logger.info(f"Completed batch processing of {len(results)} predictions")
        return results

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        model_type: ModelType = "gemma-3-12b-it",
        quantization_type: QuantizationType = "Q4_K_M",
        model_path: str | None = None,
    ) -> "LLMStrategy":
        """Load training data to extract correct answers and misconceptions."""
        logger.info("Fitting LLM strategy")
        logger.info(f"Loading training data from {train_csv_path}")

        all_training_data = parse_training_data(train_csv_path)
        logger.info(f"Parsed {len(all_training_data)} total training rows")

        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        correct_answers = extract_correct_answers(train_data)
        logger.info(f"Extracted correct answers for {len(correct_answers)} questions")

        misconceptions_by_question = extract_misconceptions_by_popularity(train_data)
        logger.info(f"Extracted misconceptions for {len(misconceptions_by_question)} questions")

        strategy = cls(
            model_type=model_type,
            quantization_type=quantization_type,
            model_path=model_path
        )
        strategy.correct_answers = correct_answers
        strategy.misconceptions_by_question = misconceptions_by_question

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
            "correct_answers": self.correct_answers,
            "misconceptions_by_question": self.misconceptions_by_question,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "quantization_type": self.quantization_type,
        }

        with filepath.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Path) -> "LLMStrategy":
        """Load strategy state from disk."""
        logger.info(f"Loading LLM strategy state from {filepath}")

        with filepath.open("rb") as f:
            state = pickle.load(f)

        strategy = cls(
            model_type=state.get("model_type", "gemma-3-12b-it"),
            quantization_type=state.get("quantization_type", "Q4_K_M"),
            model_path=state.get("model_path")
        )
        strategy.correct_answers = state["correct_answers"]
        strategy.misconceptions_by_question = state["misconceptions_by_question"]

        return strategy

    @classmethod
    def evaluate_on_split(
        cls,
        model: Strategy,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        sample_size: int | None = 10,  # Small sample for testing
    ) -> dict[str, float]:
        """Evaluate model on validation split with optional sampling."""
        logger.info("Evaluating LLM strategy on validation split")

        if model.llm is None:
            model._load_model()

        all_training_data = parse_training_data(train_csv_path)

        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )

        if sample_size is not None and sample_size < len(val_data):

            random.seed(random_seed)
            val_data = random.sample(val_data, sample_size)
            logger.info(f"Sampled {sample_size} validation rows for evaluation")

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
