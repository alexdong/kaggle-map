"""Failure analysis for prompt evolution system."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger


@dataclass
class ErrorPattern:
    """A common error pattern found in the data."""

    question_id: int
    category: str
    mc_answer: str
    actual_misconception: str
    predicted_category: str
    predicted_misconception: str
    count: int
    example_explanations: list[str]

    def __str__(self) -> str:
        return (
            f"Pattern: Q{self.question_id}, Answer={self.mc_answer}, "
            f"Expected={self.category}:{self.actual_misconception}, "
            f"Got={self.predicted_category}:{self.predicted_misconception} "
            f"(occurred {self.count} times)"
        )


def analyze_error_patterns(  # noqa: C901
    error_df: pd.DataFrame,
    max_patterns: int = 10,
) -> list[ErrorPattern]:
    assert error_df is not None, "Cannot analyze patterns from None DataFrame"
    assert max_patterns > 0, f"Max patterns must be positive, got {max_patterns}"

    logger.info(f"Analyzing error patterns from {len(error_df)} rows")

    if len(error_df) == 0:
        logger.warning("Empty DataFrame provided - no patterns to analyze")
        return []

    pattern_cols = [
        "QuestionId",
        "Category",
        "MC_Answer",
        "actual_misconception",
        "predicted_category",
        "predicted_misconception",
    ]

    available_cols = [col for col in pattern_cols if col in error_df.columns]
    assert available_cols, f"No pattern columns found in dataframe. Expected some of: {pattern_cols}, got columns: {list(error_df.columns)}"

    grouped = error_df.groupby(available_cols).size()
    pattern_counts = grouped.reset_index()
    pattern_counts.columns = [*list(pattern_counts.columns[:-1]), "count"]
    pattern_counts = pattern_counts.sort_values("count", ascending=False).head(max_patterns)

    logger.info(f"Found {len(pattern_counts)} unique error patterns")

    if len(pattern_counts) == 0:
        logger.warning("No error patterns found after grouping")
        return []

    patterns = []

    for idx, row in pattern_counts.iterrows():
        mask = True
        for col in available_cols:
            mask = mask & (error_df[col] == row[col])

        assert mask is not None, f"Failed to create mask for pattern at index {idx}"

        matching_rows = error_df[mask]

        examples = []
        if "StudentExplanation" in matching_rows.columns:
            exp_data = matching_rows["StudentExplanation"]
            if hasattr(exp_data, "tolist"):
                all_examples = exp_data.tolist()
            elif hasattr(exp_data, "to_list"):
                all_examples = exp_data.to_list()
            else:
                all_examples = list(exp_data)
            examples = all_examples[:3] if all_examples else []

        q_id = row.get("QuestionId", 0)
        q_id_int = int(q_id) if q_id is not None else 0

        assert q_id_int >= 0, f"Invalid question ID: {q_id_int}"

        count_val = int(row["count"])
        assert count_val > 0, f"Pattern count must be positive, got {count_val}"

        pattern = ErrorPattern(
            question_id=q_id_int,
            category=str(row.get("Category", "Unknown")),
            mc_answer=str(row.get("MC_Answer", "Unknown")),
            actual_misconception=str(row.get("actual_misconception", "NA")),
            predicted_category=str(row.get("predicted_category", "Unknown")),
            predicted_misconception=str(row.get("predicted_misconception", "NA")),
            count=count_val,
            example_explanations=examples,
        )
        patterns.append(pattern)

    logger.success(f"Extracted {len(patterns)} error patterns from {len(error_df)} error rows")

    return patterns


def group_failures_by_type(
    failures_df: pd.DataFrame,
) -> dict[str, list[int]]:
    assert failures_df is not None, "Cannot group failures from None DataFrame"

    if len(failures_df) == 0:
        logger.warning("Empty DataFrame - returning empty groups")
        return {"wrong_category": [], "wrong_misconception": [], "both_wrong": []}

    groups = {
        "wrong_category": [],
        "wrong_misconception": [],
        "both_wrong": [],
    }

    for idx, row in failures_df.iterrows():
        actual_cat = row.get("Category")
        pred_cat = row.get("predicted_category")
        actual_misc = row.get("actual_misconception")
        pred_misc = row.get("predicted_misconception")

        # Check if we have the necessary columns
        if actual_cat is None or pred_cat is None:
            logger.warning(f"Row {idx} missing category columns, skipping")
            continue

        category_match = actual_cat == pred_cat
        misconception_match = actual_misc == pred_misc

        if not category_match and not misconception_match:
            groups["both_wrong"].append(idx)
        elif not category_match:
            groups["wrong_category"].append(idx)
        elif not misconception_match:
            groups["wrong_misconception"].append(idx)

    total_grouped = len(groups["wrong_category"]) + len(groups["wrong_misconception"]) + len(groups["both_wrong"])

    logger.info(
        f"Grouped {total_grouped} failures: "
        f"{len(groups['wrong_category'])} wrong category only, "
        f"{len(groups['wrong_misconception'])} wrong misconception only, "
        f"{len(groups['both_wrong'])} both wrong"
    )

    return groups


def summarize_for_gpt5(  # noqa: C901
    error_df: pd.DataFrame,
    max_patterns: int = 5,
    max_examples: int = 2,
) -> str:
    assert error_df is not None, "Cannot summarize None DataFrame"
    assert max_patterns > 0, f"Max patterns must be positive, got {max_patterns}"
    assert max_examples >= 0, f"Max examples must be non-negative, got {max_examples}"

    if len(error_df) == 0:
        logger.warning("Empty error DataFrame - returning minimal summary")
        return "=== FAILURE ANALYSIS ===\nNo failures to analyze\n"

    # Extract top patterns
    patterns = analyze_error_patterns(error_df, max_patterns=max_patterns)

    # Group by error type
    groups = group_failures_by_type(error_df)

    # Build summary
    summary_parts = [
        "=== FAILURE ANALYSIS ===",
        f"Total failures analyzed: {len(error_df)}",
        "",
        "Error Type Distribution:",
        f"- Wrong category predictions: {len(groups['wrong_category'])}",
        f"- Wrong misconception predictions: {len(groups['wrong_misconception'])}",
        f"- Both wrong: {len(groups['both_wrong'])}",
        "",
        f"Top {len(patterns)} Most Common Error Patterns:",
        "",
    ]

    for i, pattern in enumerate(patterns, 1):
        summary_parts.append(f"{i}. {pattern}")
        logger.debug(f"  Adding pattern {i}: Q{pattern.question_id} ({pattern.count} occurrences)")

        if pattern.example_explanations and max_examples > 0:
            summary_parts.append("   Example student explanations:")
            for j, exp in enumerate(pattern.example_explanations[:max_examples], 1):
                # Truncate long explanations
                max_exp_len = 100
                exp_str = str(exp) if exp is not None else "[None]"
                exp_truncated = exp_str[:max_exp_len] + "..." if len(exp_str) > max_exp_len else exp_str
                summary_parts.append(f"   - {exp_truncated}")
                logger.debug(f"    Example {j}: {len(exp_str)} chars")
        summary_parts.append("")

    # Add insights
    summary_parts.extend(
        [
            "Key Observations:",
        ]
    )

    # Analyze category errors
    if groups["wrong_category"]:
        logger.debug("Analyzing category confusion patterns")
        wrong_cat_df = error_df.iloc[groups["wrong_category"]]
        if "Category" in wrong_cat_df.columns and "predicted_category" in wrong_cat_df.columns:
            most_confused = wrong_cat_df.groupby(["Category", "predicted_category"]).size()
            if not most_confused.empty:
                top_confusion = most_confused.idxmax()
                confusion_count = most_confused.max()
                summary_parts.append(f"- Most common category confusion: {top_confusion[0]} → {top_confusion[1]} ({confusion_count} times)")
                logger.debug(f"  Top confusion: {top_confusion[0]} -> {top_confusion[1]}")

    # Analyze misconception errors
    if groups["wrong_misconception"]:
        logger.debug("Analyzing misconception error patterns")
        wrong_misc_df = error_df.iloc[groups["wrong_misconception"]]
        if "actual_misconception" in wrong_misc_df.columns:
            common_missed = wrong_misc_df["actual_misconception"].value_counts().head(3)
            if not common_missed.empty:
                missed_list = [f"{misc} ({count})" for misc, count in zip(common_missed.index[:3], common_missed.values[:3], strict=False)]
                summary_parts.append(f"- Most commonly missed misconceptions: {', '.join(missed_list)}")
                logger.debug(f"  Top missed: {list(common_missed.index[:3])}")

    summary = "\n".join(summary_parts)

    logger.success(f"Generated GPT-5 summary: {len(summary)} characters, {len(summary_parts)} lines")
    return summary


def load_and_analyze_errors(
    error_path: Path = Path("datasets/error_prediction.csv"),
    output_path: Path = Path("logs/error_analysis.txt"),
) -> str:
    """Load error_prediction.csv and generate analysis.

    Args:
        error_path: Path to error_prediction.csv
        output_path: Path to save analysis results

    Returns:
        Analysis summary string
    """
    assert isinstance(error_path, Path), f"Error path must be Path object, got {type(error_path)}"
    assert isinstance(output_path, Path), f"Output path must be Path object, got {type(output_path)}"
    assert error_path.exists(), f"Error file not found at {error_path}"
    assert error_path.suffix == ".csv", f"Error file must be CSV, got {error_path.suffix}"

    logger.info(f"Loading error predictions from {error_path}")

    # Check file size
    file_size = error_path.stat().st_size
    logger.debug(f"Error file size: {file_size:,} bytes")

    error_df = pd.read_csv(error_path)

    assert not error_df.empty, f"Loaded empty DataFrame from {error_path}"

    logger.success(f"Loaded {len(error_df)} error predictions from {error_path}")
    logger.debug(f"Columns: {list(error_df.columns)}")
    logger.debug(f"Memory usage: {error_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Generate summary
    logger.info("Generating comprehensive error analysis")
    summary = summarize_for_gpt5(error_df, max_patterns=10, max_examples=3)

    assert summary, "Generated empty summary"

    # Save to file
    if not output_path.parent.exists():
        logger.debug(f"Creating output directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(summary)

    # Verify save
    assert output_path.exists(), f"Failed to create output file at {output_path}"
    assert output_path.stat().st_size > 0, f"Output file is empty: {output_path}"

    logger.success(f"Analysis saved to {output_path} ({len(summary)} characters)")

    return summary


if __name__ == "__main__":
    """Run standalone analysis."""
    import sys

    # Configure logging for standalone run
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.info("Starting standalone error analysis")

    # Run analysis
    summary = load_and_analyze_errors()

    assert summary, "Analysis returned empty summary"

    # Print summary
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(summary)
    print("=" * 80)

    logger.success("Standalone analysis complete")
