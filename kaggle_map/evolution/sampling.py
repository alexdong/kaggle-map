"""Stratified sampling utilities for evolution system."""

import pandas as pd
from loguru import logger


def stratified_sample(  # noqa: C901, PLR0912
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: list[str] | None = None,
    min_samples_per_stratum: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Perform stratified sampling on a DataFrame.

    Args:
        df: DataFrame to sample from
        sample_ratio: Fraction of data to sample (0.0-1.0)
        stratify_cols: Columns to stratify by. Defaults to ['QuestionId', 'Category', 'MC_Answer']
        min_samples_per_stratum: Minimum samples per stratum (if available)
        random_seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame preserving stratification
    """
    assert 0.0 < sample_ratio <= 1.0, f"sample_ratio must be between 0 and 1, got {sample_ratio}"

    if stratify_cols is None:
        stratify_cols = ["QuestionId", "Category", "MC_Answer"]

    # Filter to only columns that exist in the dataframe
    available_cols = [col for col in stratify_cols if col in df.columns]
    if not available_cols:
        logger.warning("No stratification columns found in dataframe. Using simple random sampling.")
        return df.sample(frac=sample_ratio, random_state=random_seed)

    if len(available_cols) < len(stratify_cols):
        missing = set(stratify_cols) - set(available_cols)
        logger.debug(f"Some stratification columns not found: {missing}. Using: {available_cols}")

    logger.info(f"Stratified sampling with ratio={sample_ratio}, stratify_by={available_cols}")

    # Group by stratification columns
    grouped = df.groupby(available_cols, group_keys=False)

    # Calculate target sample size
    target_total = max(1, int(len(df) * sample_ratio))
    logger.debug(f"Target sample size: {target_total} from {len(df)} total rows")

    # Sample from each stratum
    sampled_dfs = []
    total_sampled = 0

    for name, group in grouped:
        # Calculate how many samples to take from this stratum
        stratum_size = len(group)

        # Proportional sampling with minimum guarantee
        target_samples = max(
            min(min_samples_per_stratum, stratum_size),  # At least min_samples if available
            int(stratum_size * sample_ratio),  # Proportional sampling
        )

        # Don't exceed stratum size
        target_samples = min(target_samples, stratum_size)

        if target_samples > 0:
            sampled = group.sample(n=target_samples, random_state=random_seed)
            sampled_dfs.append(sampled)
            total_sampled += len(sampled)

            small_stratum_size = 5
            if stratum_size <= small_stratum_size:  # Log small strata
                logger.debug(f"Stratum {name}: sampled {len(sampled)}/{stratum_size}")

    if not sampled_dfs:
        logger.warning("No samples selected. Returning empty DataFrame.")
        return df.iloc[:0]

    # Combine all sampled data
    result = pd.concat(sampled_dfs, ignore_index=True)

    # If we oversampled due to minimum requirements, we keep all samples
    # to preserve rare strata (accepting slightly larger sample size)
    if len(result) > target_total * 1.5:  # Only trim if significantly oversampled
        logger.debug(f"Significantly oversampled: {len(result)} > {target_total * 1.5}. Trimming.")
        result = result.sample(n=int(target_total * 1.2), random_state=random_seed)
    elif len(result) > target_total:
        logger.debug(f"Slightly oversampled: {len(result)} > {target_total}. Keeping all to preserve rare strata.")

    logger.info(f"Sampled {len(result)} rows from {len(df)} ({len(result) / len(df) * 100:.1f}%)")

    # Log distribution statistics
    for col in available_cols[:2]:  # Log first 2 columns to avoid spam
        if col in result.columns:
            orig_dist = df[col].value_counts(normalize=True).head(3)
            sample_dist = result[col].value_counts(normalize=True).head(3)
            logger.debug(f"{col} distribution preserved (top 3):")
            for key in orig_dist.index:
                if key in sample_dist.index:
                    logger.debug(f"  {key}: {orig_dist[key]:.1%} -> {sample_dist[key]:.1%}")

    return result
