"""Stratified sampling utilities for data loading."""

import pandas as pd
from loguru import logger

from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


def stratified_sample(  # noqa: C901, PLR0913
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: list[str] | None = None,
    min_samples_per_stratum: int = 3,
    random_seed: int = 42,
    *,
    adaptive_min_samples: bool = True,
) -> pd.DataFrame:
    """Perform stratified sampling on a DataFrame.

    Args:
        df: Input DataFrame to sample from
        sample_ratio: Target ratio of samples to select (0.0 to 1.0)
        stratify_cols: Columns to use for stratification. Defaults to ["QuestionId", "Category", "MC_Answer"]
        min_samples_per_stratum: Minimum samples per stratum when possible
        random_seed: Random seed for reproducibility
        adaptive_min_samples: Whether to adapt minimum samples based on stratum size

    Returns:
        Sampled DataFrame maintaining stratification
    """
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
