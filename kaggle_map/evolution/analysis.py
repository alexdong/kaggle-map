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
        """Human-readable description of the pattern."""
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
    """Extract top error patterns from error_prediction.csv.

    Args:
        error_df: DataFrame with error predictions
        max_patterns: Maximum number of patterns to return

    Returns:
        List of most common error patterns, sorted by frequency
    """
    logger.info(f"Analyzing error patterns from {len(error_df)} rows")

    # Group by key columns to find patterns
    pattern_cols = [
        "QuestionId",
        "Category",
        "MC_Answer",
        "actual_misconception",
        "predicted_category",
        "predicted_misconception",
    ]

    # Only use columns that exist
    available_cols = [col for col in pattern_cols if col in error_df.columns]
    assert available_cols, "No pattern columns found in dataframe"

    # Count occurrences of each pattern
    # Group and count patterns
    grouped = error_df.groupby(available_cols).size()
    pattern_counts = grouped.reset_index()
    pattern_counts.columns = [*list(pattern_counts.columns[:-1]), "count"]
    pattern_counts = pattern_counts.sort_values("count", ascending=False).head(max_patterns)

    logger.debug(f"Found {len(pattern_counts)} unique patterns")

    # Convert to ErrorPattern objects with example explanations
    patterns = []
    for _, row in pattern_counts.iterrows():
        # Find matching rows for examples
        mask = True
        for col in available_cols:
            mask = mask & (error_df[col] == row[col])

        matching_rows = error_df[mask]

        # Get example explanations (up to 3)
        examples = []
        if "StudentExplanation" in matching_rows.columns:
            # Get first 3 examples from the Series
            exp_data = matching_rows["StudentExplanation"]
            # Convert to list regardless of whether it's Series or ndarray
            if hasattr(exp_data, "tolist"):
                all_examples = exp_data.tolist()
            elif hasattr(exp_data, "to_list"):
                all_examples = exp_data.to_list()
            else:
                all_examples = list(exp_data)
            examples = all_examples[:3] if all_examples else []

        # Extract values with explicit type conversion
        q_id = row.get("QuestionId", 0)
        q_id_int = int(q_id) if q_id is not None else 0

        pattern = ErrorPattern(
            question_id=q_id_int,
            category=str(row.get("Category", "Unknown")),
            mc_answer=str(row.get("MC_Answer", "Unknown")),
            actual_misconception=str(row.get("actual_misconception", "NA")),
            predicted_category=str(row.get("predicted_category", "Unknown")),
            predicted_misconception=str(row.get("predicted_misconception", "NA")),
            count=int(row["count"]),
            example_explanations=examples,
        )
        patterns.append(pattern)

        logger.debug(f"Pattern: {pattern}")

    logger.info(f"Extracted {len(patterns)} error patterns")
    return patterns


def group_failures_by_type(
    failures_df: pd.DataFrame,
) -> dict[str, list[int]]:
    """Group failures by error type (wrong category vs wrong misconception).

    Args:
        failures_df: DataFrame with failure cases

    Returns:
        Dictionary mapping error type to list of row indices
    """
    logger.debug("Grouping failures by error type")

    groups = {
        "wrong_category": [],
        "wrong_misconception": [],
        "both_wrong": [],
    }

    for idx, row in failures_df.iterrows():
        category_match = row.get("Category") == row.get("predicted_category")
        misconception_match = row.get("actual_misconception") == row.get("predicted_misconception")

        if not category_match and not misconception_match:
            groups["both_wrong"].append(idx)
        elif not category_match:
            groups["wrong_category"].append(idx)
        elif not misconception_match:
            groups["wrong_misconception"].append(idx)

    logger.info(
        f"Grouped failures: {len(groups['wrong_category'])} wrong category, "
        f"{len(groups['wrong_misconception'])} wrong misconception, "
        f"{len(groups['both_wrong'])} both wrong"
    )

    return groups


def summarize_for_gpt5(  # noqa: C901
    error_df: pd.DataFrame,
    max_patterns: int = 5,
    max_examples: int = 2,
) -> str:
    """Generate a concise failure summary for GPT-5 context.

    Args:
        error_df: DataFrame with error predictions
        max_patterns: Maximum number of patterns to include
        max_examples: Maximum examples per pattern

    Returns:
        Human-readable summary string for GPT-5 context
    """
    logger.info("Generating failure summary for GPT-5")

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

        if pattern.example_explanations:
            summary_parts.append("   Example student explanations:")
            for _j, exp in enumerate(pattern.example_explanations[:max_examples], 1):
                # Truncate long explanations
                max_exp_len = 100
                exp_truncated = exp[:max_exp_len] + "..." if len(exp) > max_exp_len else exp
                summary_parts.append(f"   - {exp_truncated}")
        summary_parts.append("")

    # Add insights
    summary_parts.extend(
        [
            "Key Observations:",
        ]
    )

    # Analyze category errors
    if groups["wrong_category"]:
        wrong_cat_df = error_df.iloc[groups["wrong_category"]]
        if "Category" in wrong_cat_df.columns and "predicted_category" in wrong_cat_df.columns:
            most_confused = wrong_cat_df.groupby(["Category", "predicted_category"]).size()
            if not most_confused.empty:
                top_confusion = most_confused.idxmax()
                summary_parts.append(f"- Most common category confusion: {top_confusion[0]} → {top_confusion[1]}")

    # Analyze misconception errors
    if groups["wrong_misconception"]:
        wrong_misc_df = error_df.iloc[groups["wrong_misconception"]]
        if "actual_misconception" in wrong_misc_df.columns:
            common_missed = wrong_misc_df["actual_misconception"].value_counts().head(3)
            if not common_missed.empty:
                summary_parts.append(f"- Most commonly missed misconceptions: {', '.join(common_missed.index[:3])}")

    summary = "\n".join(summary_parts)

    logger.debug(f"Generated summary of {len(summary)} characters")
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
    logger.info(f"Loading errors from {error_path}")

    assert error_path.exists(), f"Error file not found: {error_path}"

    error_df = pd.read_csv(error_path)
    logger.info(f"Loaded {len(error_df)} error predictions")

    # Generate summary
    summary = summarize_for_gpt5(error_df, max_patterns=10, max_examples=3)

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary)
    logger.success(f"Analysis saved to {output_path}")

    return summary


if __name__ == "__main__":
    """Run standalone analysis."""
    import sys

    # Configure logging for standalone run
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Run analysis
    summary = load_and_analyze_errors()

    # Print summary
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)
