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

    for name, group in grouped:
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
