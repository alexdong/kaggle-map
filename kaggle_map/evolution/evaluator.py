"""Evaluation pipeline for prompt candidates."""

import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.core.models import Category, Prediction
from kaggle_map.evolution import (
    PARTIAL_HIT_THRESHOLD,
    EvaluationResult,
    FailureCase,
    PromptCandidate,
)
from kaggle_map.evolution.sampling import stratified_sample
from kaggle_map.evolution.storage import Storage


def evaluate_candidate(
    candidate: PromptCandidate,
    eval_data_path: Path = Path("datasets/error_prediction.csv"),
    sample_ratio: float = 0.1,
    model_name: str = "gemma-3-12b-it",
    quantization: str = "Q4_K_XL",
) -> EvaluationResult:
    assert candidate, "Cannot evaluate None candidate"
    assert candidate.prompt, f"Cannot evaluate candidate {candidate.candidate_id} with empty prompt"
    assert 0.0 < sample_ratio <= 1.0, f"Sample ratio must be between 0 and 1, got {sample_ratio}"
    assert eval_data_path.exists(), f"Evaluation data not found at {eval_data_path}"

    logger.info(f"Evaluating candidate {candidate.candidate_id}")

    storage = Storage()
    storage.save_prompt_template(candidate)
    template_path = storage.get_prompt_template_path(candidate.candidate_id)

    assert template_path.exists(), f"Template file not created at {template_path}"

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

    logger.info(f"Running benchmark for {candidate.candidate_id}")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Benchmark failed with exit code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        logger.warning(f"Assigning MAP@3 score of 0.0 to failed candidate {candidate.candidate_id}")
        map_score = 0.0
    else:
        map_score = parse_map_score(result.stdout)
        logger.success(f"Candidate {candidate.candidate_id} achieved MAP@3: {map_score:.4f}")

        if map_score > 0.7:
            logger.info("  Excellent performance (MAP@3 > 0.7)")
        elif map_score > 0.5:
            logger.info("  Good performance (MAP@3 > 0.5)")
        elif map_score < 0.3:
            logger.warning("  Poor performance (MAP@3 < 0.3)")

    eval_df = pd.read_csv(eval_data_path)

    eval_df = stratified_sample(
        eval_df,
        sample_ratio=sample_ratio,
        stratify_cols=["QuestionId", "Category", "MC_Answer"],
        min_samples_per_stratum=3,
        random_seed=42,
    )

    failure_cases = extract_failure_cases(eval_df, max_failures=10)

    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        map_score=map_score,
        failure_samples=failure_cases,
    )


def parse_map_score(output: str) -> float:  # noqa: C901
    assert output, "Cannot parse MAP@3 score from empty output"

    found_scores = []
    for line in output.split("\n"):
        if "MAP@3" in line or "│" in line:
            parts = line.split("│")
            for raw_part in parts:
                part = raw_part.strip()
                if not part or part == "MAP@3":
                    continue
                if part.replace(".", "").replace("-", "").isdigit():
                    score = float(part)
                    if 0.0 <= score <= 1.0:
                        found_scores.append(score)

    if found_scores:
        final_score = found_scores[0]
        logger.info(f"Parsed MAP@3 score: {final_score:.4f}")
        return final_score

    logger.warning("Could not parse MAP@3 score from benchmark output - defaulting to 0.0")
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
    """
    assert eval_df is not None, "Cannot extract failures from None DataFrame"
    assert max_failures > 0, f"Max failures must be positive, got {max_failures}"

    has_map_score = "map_score" in eval_df.columns
    has_predictions = "predicted_category" in eval_df.columns and "predicted_misconception" in eval_df.columns

    if not has_map_score and not has_predictions:
        logger.warning("No prediction columns found in DataFrame - cannot extract failures")
        return []

    if has_map_score:
        eval_df["is_failure"] = eval_df["map_score"] < 1.0
    else:
        eval_df["is_failure"] = (eval_df["predicted_category"] != eval_df["Category"]) | (
            eval_df["predicted_misconception"] != eval_df["actual_misconception"]
        )

    failure_mask = eval_df["is_failure"]
    failures_df = eval_df.loc[failure_mask, :].copy()

    if len(failures_df) == 0:
        logger.warning(
            "No failures found in evaluation data - this may indicate perfect performance or missing predictions"
        )
        return []

    if "map_score" in failures_df.columns:
        partial_threshold = PARTIAL_HIT_THRESHOLD

        priority_values = []
        complete_misses = 0
        partial_hits = 0

        for score in failures_df["map_score"]:
            if score == 0.0:
                priority_values.append(0)
                complete_misses += 1
            elif score < partial_threshold:
                priority_values.append(1)
                partial_hits += 1
            else:
                priority_values.append(2)

        failures_df["priority"] = priority_values
        failures_df = failures_df.sort_values(by=["priority"], axis=0)

    failure_cases = []
    seen_patterns = set()

    for _idx, row in failures_df.iterrows():
        q_id = row.get("QuestionId")
        cat = row.get("Category")
        mc_ans = row.get("MC_Answer")

        pattern = (q_id, cat, mc_ans)

        if pattern in seen_patterns and len(failure_cases) >= max_failures // 2:
            continue

        seen_patterns.add(pattern)

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

    logger.info(f"Extracted {len(failure_cases)} diverse failure cases from {len(failures_df)} total failures")

    return failure_cases


def evaluate_all_candidates(
    candidates: list[PromptCandidate],
    eval_data_path: Path = Path("datasets/error_prediction.csv"),
    sample_ratio: float = 0.1,
) -> list[EvaluationResult]:
    assert candidates, "Cannot evaluate empty candidate list"
    assert all(c.prompt for c in candidates), "All candidates must have prompts"
    assert eval_data_path.exists(), f"Evaluation data not found at {eval_data_path}"

    logger.info(f"Starting evaluation of {len(candidates)} candidates")

    results = []
    scores = []

    for i, candidate in enumerate(candidates, 1):
        logger.info(f"\n[{i}/{len(candidates)}] Evaluating: {candidate.candidate_id}")

        result = evaluate_candidate(
            candidate,
            eval_data_path=eval_data_path,
            sample_ratio=sample_ratio,
        )
        results.append(result)
        scores.append(result.map_score)

        if result.map_score > 0.6:
            logger.success(f"  ✓ Strong result: MAP@3 = {result.map_score:.4f}")
        elif result.map_score > 0.4:
            logger.info(f"  → Moderate result: MAP@3 = {result.map_score:.4f}")
        else:
            logger.warning(f"  ✗ Weak result: MAP@3 = {result.map_score:.4f}")

    results.sort(key=lambda r: (-r.map_score, r.candidate_id))

    best_score = results[0].map_score if results else 0.0
    worst_score = results[-1].map_score if results else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    logger.success(f"\n{'=' * 60}")
    logger.success(f"Evaluation complete for {len(candidates)} candidates")
    logger.success(f"  Best MAP@3:  {best_score:.4f} ({results[0].candidate_id if results else 'N/A'})")
    logger.success(f"  Worst MAP@3: {worst_score:.4f} ({results[-1].candidate_id if results else 'N/A'})")
    logger.success(f"  Average:     {avg_score:.4f}")
    logger.success(f"  Spread:      {best_score - worst_score:.4f}")
    logger.success(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    """Standalone validation of evaluator operations."""
    import sys
    from pathlib import Path

    from kaggle_map.evolution import PromptCandidate

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("=== Evaluator Module Validation ===")

    # Check for baseline prompt
    baseline_path = Path("reranker/prompts/baseline.j2")
    if not baseline_path.exists():
        logger.error(f"Baseline prompt not found at {baseline_path}")
        logger.info("Creating test prompt template...")

        test_prompt = """You are a teacher analyzing student responses to math problems.

Given:
- Question: {{question_text}}
- Student's Answer: {{mc_answer}}
- Student's Explanation: {{student_explanation}}
- Category Type: {{category}}

Identify the most likely misconception the student has based on their explanation and answer choice.

Output format:
Category: [True/False]_[Type]
Misconception: [Number or 'NA']

Be concise and specific."""

        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(test_prompt)
        logger.info(f"Created test prompt at {baseline_path}")

    # Create test candidate
    test_candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_eval_test",
        prompt=baseline_path.read_text(),
        hypothesis="Testing evaluator functionality",
        parent_ids=[]
    )

    logger.info("\n1. Testing single candidate evaluation:")
    logger.info(f"  Candidate: {test_candidate.candidate_id}")
    logger.info(f"  Hypothesis: {test_candidate.hypothesis}")

    # Check if evaluation data exists
    eval_data_path = Path("datasets/error_prediction.csv")
    if not eval_data_path.exists():
        logger.error(f"Evaluation data not found at {eval_data_path}")
        logger.info("Please ensure error_prediction.csv exists before running full evaluation")
        logger.info("\n❌ Evaluator validation incomplete - missing data file")
        sys.exit(1)

    # Test with very small sample for quick validation
    logger.info("\n2. Running micro evaluation (0.1% sample):")
    try:
        result = evaluate_candidate(
            test_candidate,
            eval_data_path=eval_data_path,
            sample_ratio=0.001,  # Very small sample for testing
        )

        logger.info("\n3. Evaluation results:")
        logger.info(f"  MAP@3 Score: {result.map_score:.4f}")
        logger.info(f"  Failure samples: {len(result.failure_samples)}")

        if result.failure_samples:
            logger.info("\n4. Sample failures:")
            for i, failure in enumerate(result.failure_samples[:3], 1):
                logger.info(f"  Failure {i}: {failure}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check if benchmark.py accepts --prompt-template")
        logger.info("  2. Ensure model is available (gemma-3-12b-it)")
        logger.info("  3. Verify error_prediction.csv has correct columns")
        sys.exit(1)

    # Test MAP score parsing
    logger.info("\n5. Testing MAP score parser:")
    test_outputs = [
        "MAP@3: 0.4567",
        "│ MAP@3 │ 0.6789 │",
        "Some text\nMAP@3 score is 0.3456\nMore text",
        "No score here",
    ]

    for output in test_outputs:
        score = parse_map_score(output)
        logger.info(f"  Input: '{output[:30]}...' → Score: {score:.4f}")

    # Test batch evaluation
    logger.info("\n6. Testing batch evaluation:")
    test_candidates = [
        PromptCandidate(
            generation=0,
            candidate_id=f"gen_00_candidate_{i}",
            prompt=test_candidate.prompt,  # Reuse same prompt for testing
            hypothesis=f"Test hypothesis {i}",
            parent_ids=[]
        )
        for i in range(2)
    ]

    logger.info(f"  Evaluating {len(test_candidates)} candidates...")
    results = evaluate_all_candidates(
        test_candidates,
        eval_data_path=eval_data_path,
        sample_ratio=0.001,  # Very small for testing
    )

    logger.info("\n7. Batch results summary:")
    for result in results:
        logger.info(f"  {result.candidate_id}: MAP@3={result.map_score:.4f}")

    # Cleanup test files
    logger.info("\n8. Cleanup:")
    test_template_path = Path(f"reranker/prompts/{test_candidate.candidate_id}.j2")
    if test_template_path.exists():
        test_template_path.unlink()
        logger.info(f"  Removed test template: {test_template_path}")

    logger.info("\n✅ Evaluator validation complete!")
