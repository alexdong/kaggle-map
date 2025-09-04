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
from typing import TYPE_CHECKING

from loguru import logger

from kaggle_map.core.dataset import (
    extract_correct_answers,
    extract_misconceptions_by_popularity,
    parse_training_data,
)
from kaggle_map.core.llm_types import (
    LLMConfig,
    LLMInferenceContext,
    ProblemContext,
    StudentWork,
)
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    EvaluationRow,
    Prediction,
    QuestionId,
    SubmissionRow,
)
from kaggle_map.utils.llm import get_model_path, load_llm_model

from .base import Strategy
from .utils import TRAIN_RATIO, split_training_data

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

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.model_path = get_model_path(self.config.model_name, self.config.quantization)
        self.llm: Llama | None = None
        self.problem_contexts: dict[QuestionId, ProblemContext] = {}

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return f"LLM-based prediction using {self.config.model_name} ({self.config.quantization})"

    def _load_model(self) -> None:
        """Lazy load the GGUF model, downloading if necessary."""
        if self.llm is not None:
            return

        # Use context manager to load model
        model_context = load_llm_model(self.config)
        self.llm = model_context.__enter__()

    def _build_prompt(self, context: LLMInferenceContext) -> str:
        """Build XML-structured prompt with training context."""
        correct_answer = context.problem.correct_answer or "Unknown"
        misconceptions = context.problem.known_misconceptions or []
        known_misconceptions = ", ".join(misconceptions[:5]) if misconceptions else "None identified"

        return PROMPT_TEMPLATE.format(
            question=context.problem.question_text,
            correct_answer=correct_answer,
            known_misconceptions=known_misconceptions,
            student_answer=context.student_work.answer,
            student_explanation=context.student_work.explanation,
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
                # Build inference context
                problem_context = self.problem_contexts.get(row.question_id)
                if problem_context:
                    # Use provided problem context
                    student_work = StudentWork(
                        answer=row.mc_answer,
                        explanation=row.student_explanation
                    )
                    inference_context = LLMInferenceContext(
                        problem=problem_context,
                        student_work=student_work
                    )
                else:
                    # Use default from_evaluation_row
                    inference_context = LLMInferenceContext.from_evaluation_row(row)
                prompt = self._build_prompt(inference_context)

                start_time = time.time()
                assert self.llm is not None, "LLM model not loaded"
                output = self.llm(
                    prompt,
                    max_tokens=30,  # Just enough for "Category:Misconception"
                    temperature=0.3,
                    stop=["</instructions>", "\n\n"],
                    echo=False,
                )
                inference_time = time.time() - start_time

                logger.debug(f"Inference time for row {row.row_id}: {inference_time:.2f}s")

                # Cast to dict to access fields (llama-cpp-python returns iterator/dict)
                output_dict = dict(output) if hasattr(output, "__iter__") else output  # type: ignore
                response = output_dict["choices"][0]["text"].strip()

                prediction = self._parse_response(response, row)
                # Parse category and misconception from "Category:Misconception" format
                if ":" in prediction:
                    category, misconception = prediction.split(":", 1)
                else:
                    category, misconception = prediction, "NA"
                pred_obj = Prediction(category=category, misconception=misconception)
                results.append(SubmissionRow(row_id=row.row_id, predicted_categories=[pred_obj]))

        logger.info(f"Completed batch processing of {len(results)} predictions")
        return results

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        config: LLMConfig | None = None,
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

        strategy = cls(config=config)

        # Build problem contexts for all questions
        for question_id in correct_answers:
            strategy.problem_contexts[question_id] = ProblemContext(
                question_id=question_id,
                question_text="",  # Will be filled from evaluation row
                correct_answer=correct_answers[question_id],
                known_misconceptions=misconceptions_by_question.get(question_id, []),
            )

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
            "problem_contexts": self.problem_contexts,
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
        strategy.problem_contexts = state["problem_contexts"]

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
