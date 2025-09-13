"""Stratified sampling utilities for evolution system."""

import pandas as pd
from loguru import logger

from kaggle_map.dataloader.sampling import stratification_report, stratified_sample
from kaggle_map.embeddings.sampler import select_diverse_samples
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

# Re-export for backward compatibility
__all__ = ["select_diverse_samples", "stratification_report", "stratified_sample"]

# Constants for display
MAX_DISPLAY_GROUPS = 10
MAX_QUESTION_TEXT_LENGTH = 100


if __name__ == "__main__":
    """Standalone validation of sampling operations."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    # Logger is already configured by configure_logger(__name__)

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
