"""Stratified sampling utilities for evolution system."""

import pandas as pd
from loguru import logger


def stratified_sample(  # noqa: C901
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: list[str] | None = None,
    min_samples_per_stratum: int = 3,
    random_seed: int = 42,
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

        target_samples = max(
            min(min_samples_per_stratum, stratum_size),
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

    logger.info(f"Sampled {len(result)} rows from {len(df)} ({len(result) / len(df) * 100:.1f}%)")

    return result


if __name__ == "__main__":
    """Standalone validation of sampling operations."""
    import sys
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from kaggle_map.embeddings.sampler import select_diverse_samples

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=== Sampling Module Validation ===")

    # Load sample data
    error_prediction_path = Path("datasets/error_prediction.csv")
    error_data = pd.read_csv(error_prediction_path)
    logger.info(f"Loaded {len(error_data)} rows from {error_prediction_path}")

    # Sample 10% of data
    logger.info("\n1. Testing 10% stratified sampling:")
    sampled = stratified_sample(error_data, sample_ratio=0.1, random_seed=42)
    logger.info(f"  10% requested → {len(sampled)} samples ({len(sampled)/len(error_data)*100:.1f}% actual)")

    # Analyze distribution
    logger.info("\n2. Analyzing sample distribution:")
    grouped = sampled.groupby(["QuestionId", "Category", "MC_Answer"]).size()
    logger.info(f"  Unique (Question, Category, Answer) groups: {len(grouped)}")
    logger.info(f"  Samples per group: min={grouped.min()}, max={grouped.max()}, mean={grouped.mean():.1f}")

    # Show diverse explanations per group
    logger.info("\n3. Selecting diverse explanations per group:")

    # Configuration
    N_DIVERSE_SAMPLES = 3
    MIN_GROUP_SIZE = 3  # Only show groups with enough samples to choose from

    # Create rich table
    console = Console()
    table = Table(title=f"Top {N_DIVERSE_SAMPLES} Diverse Student Explanations per Question/Answer Group")

    # Add columns
    table.add_column("Question", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("MC Answer", style="yellow")
    table.add_column("Count", style="white")

    for i in range(1, N_DIVERSE_SAMPLES + 1):
        table.add_column(f"Explanation {i}", overflow="fold", min_width=20)

    table.add_column("Div Score", style="green")

    # Process each group
    groups_processed = 0
    for (qid, category, mc_answer), group_df in sampled.groupby(["QuestionId", "Category", "MC_Answer"]):
        if len(group_df) < MIN_GROUP_SIZE:
            continue

        # Get all explanations for this group
        explanations = group_df["StudentExplanation"].tolist()

        # Select diverse samples
        logger.debug(f"Processing Q{qid}, {category}, {mc_answer}: {len(explanations)} explanations")
        selected_indices, diversity_score = select_diverse_samples(
            explanations,
            n_samples=N_DIVERSE_SAMPLES
        )

        # Build row for table
        row = [
            f"Q{qid}",
            str(category),
            str(mc_answer),
            str(len(group_df)),
        ]

        # Add selected explanations
        for idx in selected_indices:
            row.append(explanations[idx])

        # Fill in if we got fewer than requested
        while len(row) < 4 + N_DIVERSE_SAMPLES:
            row.append("-")

        # Add diversity score
        row.append(f"{diversity_score:.2f}")

        table.add_row(*row)
        groups_processed += 1

        # Limit output for readability
        if groups_processed >= 10:
            logger.info(f"  (Showing first 10 groups with {MIN_GROUP_SIZE}+ samples)")
            break

    # Display table
    console.print("\n")
    console.print(table)

    logger.info(f"\n✅ Processed {groups_processed} groups with {MIN_GROUP_SIZE}+ samples")
