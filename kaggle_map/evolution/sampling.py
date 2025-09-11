"""Stratified sampling utilities for evolution system."""

import pandas as pd
from loguru import logger

from kaggle_map.embeddings.sampler import select_diverse_samples

# Constants for display
MAX_DISPLAY_GROUPS = 10
MAX_QUESTION_TEXT_LENGTH = 100


def stratified_sample(  # noqa: C901, PLR0913
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: list[str] | None = None,
    min_samples_per_stratum: int = 3,
    random_seed: int = 42,
    *,
    adaptive_min_samples: bool = True,
) -> pd.DataFrame:
    assert 0.0 < sample_ratio <= 1.0, f"sample_ratio must be between 0 and 1, got {sample_ratio}"

    if stratify_cols is None:
        stratify_cols = ["QuestionId", "Category", "MC_Answer"]

    available_cols = [col for col in stratify_cols if col in df.columns]
    if not available_cols:
        logger.warning("No stratification columns found in dataframe. Using simple random sampling.")
        return df.sample(frac=sample_ratio, random_state=random_seed)

    logger.info(f"Stratified sampling with ratio={sample_ratio}, stratify_by={available_cols}")

    grouped = df.groupby(available_cols, group_keys=False)
    target_total = max(1, int(len(df) * sample_ratio))

    sampled_dfs = []
    total_sampled = 0

    for _name, group in grouped:
        stratum_size = len(group)

        # Adaptive minimum samples based on stratum size
        adaptive_min = max(1, min(3, int(stratum_size * 0.3))) if adaptive_min_samples else min_samples_per_stratum

        target_samples = max(
            min(adaptive_min, stratum_size),
            int(stratum_size * sample_ratio),
        )

        target_samples = min(target_samples, stratum_size)

        if target_samples > 0:
            sampled = group.sample(n=target_samples, random_state=random_seed)
            sampled_dfs.append(sampled)
            total_sampled += len(sampled)

    if not sampled_dfs:
        logger.warning("No samples selected. Returning empty DataFrame.")
        return df.iloc[:0]

    result = pd.concat(sampled_dfs, ignore_index=True)

    if len(result) > target_total * 1.5:
        result = result.sample(n=int(target_total * 1.2), random_state=random_seed)

    # Deterministic ordering for consistent output
    sort_cols = [*available_cols, "row_id"] if "row_id" in result.columns else available_cols
    result = result.sort_values(by=sort_cols).reset_index(drop=True)

    # Sample size warning
    actual_ratio = len(result) / len(df)
    if abs(actual_ratio - sample_ratio) > 0.2 * sample_ratio:
        logger.warning(
            f"Actual sample ratio {actual_ratio:.1%} differs significantly from requested {sample_ratio:.1%}"
        )

    logger.info(f"Sampled {len(result)} rows from {len(df)} ({actual_ratio * 100:.1f}%)")

    return result


def stratification_report(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: list[str],
) -> dict:
    """Generate report comparing original and sampled distributions.

    Returns:
        Dictionary with coverage, balance, and distribution metrics
    """
    assert len(original_df) > 0, "Original DataFrame cannot be empty"
    assert len(sampled_df) > 0, "Sampled DataFrame cannot be empty"

    # Group distributions
    original_groups = original_df.groupby(stratify_cols).size()
    sampled_groups = sampled_df.groupby(stratify_cols).size()

    # Calculate metrics
    coverage = len(sampled_groups) / len(original_groups) * 100

    # Find under/over-represented groups
    all_groups = set(original_groups.index)
    sampled_group_set = set(sampled_groups.index)
    missing_groups = all_groups - sampled_group_set

    # Calculate sampling rates per group
    sampling_rates = {}
    for group in sampled_group_set:
        original_count = original_groups[group]
        sampled_count = sampled_groups[group]
        sampling_rates[group] = sampled_count / original_count

    # Find extremes
    if sampling_rates:
        max_rate = max(sampling_rates.values())
        min_rate = min(sampling_rates.values())
        mean_rate = sum(sampling_rates.values()) / len(sampling_rates)

        oversampled = [g for g, r in sampling_rates.items() if r > mean_rate * 1.5]
        undersampled = [g for g, r in sampling_rates.items() if r < mean_rate * 0.5]
    else:
        max_rate = min_rate = mean_rate = 0
        oversampled = undersampled = []

    report = {
        "coverage": coverage,
        "total_groups": len(original_groups),
        "sampled_groups": len(sampled_groups),
        "missing_groups": len(missing_groups),
        "sample_rate_range": (min_rate, max_rate),
        "mean_sample_rate": mean_rate,
        "oversampled_count": len(oversampled),
        "undersampled_count": len(undersampled),
    }

    logger.info("\n=== Stratification Report ===")
    logger.info(f"Coverage: {coverage:.1f}% ({len(sampled_groups)}/{len(original_groups)} groups)")
    logger.info(f"Missing groups: {len(missing_groups)}")
    logger.info(f"Sample rate range: {min_rate:.1%} to {max_rate:.1%} (mean: {mean_rate:.1%})")

    if oversampled:
        logger.info(f"Oversampled groups: {len(oversampled)}")
    if undersampled:
        logger.info(f"Undersampled groups: {len(undersampled)}")

    return report


if __name__ == "__main__":
    """Standalone validation of sampling operations."""
    import sys
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=== Sampling Module Validation ===")

    # Load sample data
    error_prediction_path = Path("datasets/error_prediction.csv")
    error_data = pd.read_csv(error_prediction_path)
    logger.info(f"Loaded {len(error_data)} rows from {error_prediction_path}")

    # Sample 10% of data
    logger.info("\n1. Testing 10% stratified sampling:")
    sampled = stratified_sample(
        error_data,
        sample_ratio=0.1,
        stratify_cols=["QuestionId", "MC_Answer", "actual_misconception"],
        min_samples_per_stratum=3,
        random_seed=42,
        adaptive_min_samples=True,
    )
    logger.info(f"  10% requested → {len(sampled)} samples ({len(sampled) / len(error_data) * 100:.1f}% actual)")

    # Analyze distribution
    logger.info("\n2. Analyzing sample distribution:")
    grouped = sampled.groupby(["QuestionId", "MC_Answer", "actual_misconception"]).size()
    logger.info(f"  Unique (Question, Answer, Misconception) groups: {len(grouped)}")
    logger.info(f"  Samples per group: min={grouped.min()}, max={grouped.max()}, mean={grouped.mean():.1f}")

    # Generate stratification report
    stratify_cols = ["QuestionId", "MC_Answer", "actual_misconception"]
    report = stratification_report(error_data, sampled, stratify_cols)

    # Show diverse explanations per group
    logger.info("\n3. Selecting diverse explanations per group:")

    # Configuration
    N_DIVERSE_SAMPLES = 3
    MIN_GROUP_SIZE = 3  # Only show groups with enough samples to choose from

    # Create rich table
    console = Console()
    table = Table(title=f"Top {N_DIVERSE_SAMPLES} Diverse Student Explanations per Question/Answer/Misconception Group")

    # Add columns
    table.add_column("Q#", style="cyan", no_wrap=True)
    table.add_column("Question Text", style="blue", overflow="fold", max_width=40)
    table.add_column("Correct Answer", style="green", overflow="fold")
    table.add_column("MC Answer", style="yellow", overflow="fold")
    table.add_column("Category", style="magenta")
    table.add_column("Misconception", style="red")
    table.add_column("Count", style="white")

    for i in range(1, N_DIVERSE_SAMPLES + 1):
        table.add_column(f"Explanation {i}", overflow="fold", min_width=20)

    # Process ALL groups
    groups_processed = 0
    groups_with_diversity = 0

    groupby_cols = ["QuestionId", "MC_Answer", "actual_misconception"]
    for (qid, mc_answer, misconception), group_df in sampled.groupby(groupby_cols):  # type: ignore[attr-defined]
        groups_processed += 1
        explanations = group_df["StudentExplanation"].tolist()

        # Get question text and correct answer from first row of group
        full_question = str(group_df.iloc[0]["QuestionText"])
        question_text = full_question[:MAX_QUESTION_TEXT_LENGTH]
        if len(full_question) > MAX_QUESTION_TEXT_LENGTH:
            question_text += "..."

        # Get correct answer if available
        correct_answer = str(group_df.iloc[0].get("CorrectAnswer", "N/A"))

        # Get the most common category in this group (could be mixed TRUE/FALSE)
        categories = group_df["Category"].value_counts()
        category = categories.index[0] if not categories.empty else "N/A"

        # Build row for table
        row = [
            f"Q{qid}",
            question_text,
            correct_answer,
            str(mc_answer),
            str(category),
            str(misconception),
            str(len(group_df)),
        ]

        if len(group_df) < MIN_GROUP_SIZE:
            # For small groups, just show all explanations
            row.extend(explanations[: min(len(explanations), N_DIVERSE_SAMPLES)])

            # Fill remaining columns
            while len(row) < 7 + N_DIVERSE_SAMPLES:
                row.append("-")
        else:
            # Select diverse samples for larger groups
            groups_with_diversity += 1
            logger.debug(f"Processing Q{qid}, {mc_answer}, {misconception}: {len(explanations)} explanations")
            selected_indices, diversity_score = select_diverse_samples(explanations, n_samples=N_DIVERSE_SAMPLES)

            # Add selected explanations
            row.extend(explanations[idx] for idx in selected_indices)

            # Fill in if we got fewer than requested
            while len(row) < 7 + N_DIVERSE_SAMPLES:
                row.append("-")

        table.add_row(*row)

    # Display table
    console.print("\n")
    console.print(table)

    logger.info(f"\n✅ Showing ALL {groups_processed} groups")
    logger.info(f"  {groups_with_diversity} groups with diversity scores ({MIN_GROUP_SIZE}+ samples)")
    no_diversity = groups_processed - groups_with_diversity
    logger.info(f"  {no_diversity} groups without diversity scores (<{MIN_GROUP_SIZE} samples)")
