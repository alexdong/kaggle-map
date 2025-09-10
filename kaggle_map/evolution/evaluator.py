"""Evaluation pipeline for prompt candidates."""

import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.core.models import Category, Prediction
from kaggle_map.evolution import EvaluationResult, FailureCase, PromptCandidate
from kaggle_map.evolution.sampling import stratified_sample
from kaggle_map.evolution.storage import Storage


def evaluate_candidate(
    candidate: PromptCandidate,
    eval_data_path: Path = Path("datasets/error_prediction.csv"),
    sample_ratio: float = 0.1,
    model_name: str = "gemma-3-12b-it",
    quantization: str = "Q4_K_XL",
) -> EvaluationResult:
    """Evaluate a prompt candidate using benchmark.py.

    Args:
        candidate: Prompt candidate to evaluate
        eval_data_path: Path to evaluation dataset
        sample_ratio: Fraction of data to sample
        model_name: Model to use for evaluation
        quantization: Quantization level

    Returns:
        Evaluation result with MAP@3 score and failure cases
    """
    logger.info(f"Evaluating candidate {candidate.candidate_id}")

    # Save prompt template to temporary file
    storage = Storage()
    storage.save_prompt_template(candidate)
    template_path = storage.get_prompt_template_path(candidate.candidate_id)

    logger.debug(f"Saved template to {template_path}")

    # Run benchmark.py with the template
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "kaggle_map.reranker.benchmark",
        "--model",
        model_name,
        "--quantization",
        quantization,
        "--sample-ratio",
        str(sample_ratio),
        "--prompt-template",
        str(template_path),
        "--use-stratified",
    ]

    logger.info(f"Running benchmark: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse MAP@3 score from output
        map_score = parse_map_score(result.stdout)
        logger.info(f"Candidate {candidate.candidate_id} achieved MAP@3: {map_score:.4f}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed: {e.stderr}")
        map_score = 0.0

    # Load evaluation data and extract failures
    eval_df = pd.read_csv(eval_data_path)

    # Sample the same way benchmark.py does
    eval_df = stratified_sample(
        eval_df,
        sample_ratio=sample_ratio,
        stratify_cols=["QuestionId", "Category", "MC_Answer"],
        min_samples_per_stratum=3,
        random_seed=42,
    )

    # Extract failure cases
    failure_cases = extract_failure_cases(eval_df, max_failures=10)

    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        map_score=map_score,
        failure_samples=failure_cases,
    )


def parse_map_score(output: str) -> float:  # noqa: C901
    """Parse MAP@3 score from benchmark output.

    Args:
        output: Stdout from benchmark.py

    Returns:
        MAP@3 score (0.0 to 1.0)
    """
    # Look for MAP@3 in the output table
    for line in output.split("\n"):
        if "MAP@3" in line or "│" in line:
            # Try to extract a float that looks like a score
            parts = line.split("│")
            for raw_part in parts:
                part = raw_part.strip()
                try:
                    score = float(part)
                    if 0.0 <= score <= 1.0:
                        logger.debug(f"Found MAP@3 score: {score}")
                        return score
                except ValueError:
                    continue

    logger.warning("Could not parse MAP@3 score from output")
    return 0.0


def extract_failure_cases(  # noqa: C901
    eval_df: pd.DataFrame,
    max_failures: int = 10,
) -> list[FailureCase]:
    """Extract diverse failure cases from evaluation results.

    Priority:
    1. Complete misses (correct answer not in top 3)
    2. Wrong category predictions
    3. Wrong misconception predictions
    4. Diverse sampling across (QuestionId, Category, MC_Answer)

    Args:
        eval_df: DataFrame with evaluation results
        max_failures: Maximum number of failures to extract

    Returns:
        List of failure cases
    """
    logger.debug(f"Extracting up to {max_failures} failure cases")

    # Identify failures (where prediction doesn't match ground truth)
    # Assuming eval_df has map_score column or we need to calculate it
    if "map_score" not in eval_df.columns:
        # Simple heuristic: failure if predicted != actual
        eval_df["is_failure"] = (eval_df["predicted_category"] != eval_df["Category"]) | (
            eval_df["predicted_misconception"] != eval_df["actual_misconception"]
        )
    else:
        # Use MAP score - failures are where score < 1.0
        eval_df["is_failure"] = eval_df["map_score"] < 1.0

    # Ensure we get a DataFrame, not a Series
    failure_mask = eval_df["is_failure"]
    failures_df = eval_df.loc[failure_mask, :].copy()

    if len(failures_df) == 0:
        logger.warning("No failures found in evaluation data")
        return []

    # Prioritize complete misses (MAP score = 0)
    if "map_score" in failures_df.columns:
        partial_threshold = 0.5
        # Assign priority based on MAP score
        priority_values = []
        for score in failures_df["map_score"]:
            if score == 0.0:
                priority_values.append(0)
            elif score < partial_threshold:
                priority_values.append(1)
            else:
                priority_values.append(2)
        failures_df["priority"] = priority_values
        # Ensure we're sorting a DataFrame, not a Series
        failures_df = failures_df.sort_values(by=["priority"], axis=0)

    # Sample diversely
    failure_cases = []
    seen_patterns = set()

    for _, row in failures_df.iterrows():
        # Create pattern key for diversity
        pattern = (row.get("QuestionId"), row.get("Category"), row.get("MC_Answer"))

        # Skip if we've seen this pattern too many times
        if pattern in seen_patterns and len(failure_cases) >= max_failures // 2:
            continue

        seen_patterns.add(pattern)

        # Create FailureCase
        actual_category = Category.from_csv_string(str(row["Category"]))
        actual_misconception = row.get("actual_misconception", "NA")

        failure = FailureCase(
            row_id=row["row_id"],
            question_id=row["QuestionId"],
            question_text=row["QuestionText"],
            mc_answer=row["MC_Answer"],
            student_explanation=row["StudentExplanation"],
            prediction=Prediction(
                category=actual_category,
                misconception=actual_misconception,
            ),
            predicted=[
                Prediction(
                    category=Category.from_csv_string(str(row.get("predicted_category", "True_Neither"))),
                    misconception=row.get("predicted_misconception", "NA"),
                )
            ],
        )

        failure_cases.append(failure)

        if len(failure_cases) >= max_failures:
            break

    logger.info(f"Extracted {len(failure_cases)} failure cases")
    return failure_cases


def evaluate_all_candidates(
    candidates: list[PromptCandidate],
    eval_data_path: Path = Path("datasets/error_prediction.csv"),
    sample_ratio: float = 0.1,
) -> list[EvaluationResult]:
    """Evaluate multiple prompt candidates.

    Args:
        candidates: List of candidates to evaluate
        eval_data_path: Path to evaluation dataset
        sample_ratio: Fraction of data to sample

    Returns:
        List of evaluation results, sorted by MAP score descending
    """
    logger.info(f"Evaluating {len(candidates)} candidates")

    results = []
    for i, candidate in enumerate(candidates, 1):
        logger.info(f"Evaluating candidate {i}/{len(candidates)}: {candidate.candidate_id}")
        result = evaluate_candidate(
            candidate,
            eval_data_path=eval_data_path,
            sample_ratio=sample_ratio,
        )
        results.append(result)
        logger.info(f"  MAP@3: {result.map_score:.4f}")

    # Sort by MAP score descending
    results.sort(key=lambda r: r.map_score, reverse=True)

    logger.success(f"Evaluation complete. Best MAP@3: {results[0].map_score:.4f}")
    return results
