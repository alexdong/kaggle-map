"""Stratified sampling utilities for data loading."""

from collections.abc import Sequence

import pandas as pd
from loguru import logger

from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

DEFAULT_STRATIFY_COLS: tuple[str, ...] = ("QuestionId", "Category", "MC_Answer")


def stratified_sample(
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: Sequence[str] = DEFAULT_STRATIFY_COLS,
    min_samples_per_stratum: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    assert 0.0 < sample_ratio <= 1.0, f"sample_ratio must be between 0 and 1, got {sample_ratio}"
    assert min_samples_per_stratum > 0, "min_samples_per_stratum must be positive"
    assert stratify_cols, "stratify_cols must contain at least one column name"

    available_cols = [col for col in stratify_cols if col in df.columns]
    if not available_cols:
        logger.warning("No stratification columns found in dataframe. Using simple random sampling.")
        sampled = df.sample(frac=sample_ratio, random_state=random_seed)
        return sampled.reset_index(drop=True)

    logger.info("Stratified sampling with ratio={} stratify_by={}", sample_ratio, available_cols)

    grouped = df.groupby(available_cols, group_keys=False, sort=False)

    sampled_dfs: list[pd.DataFrame] = []

    for _name, group in grouped:
        stratum_size = len(group)
        requested = max(1, int(stratum_size * sample_ratio))
        guaranteed = min(min_samples_per_stratum, stratum_size)
        target_samples = min(max(guaranteed, requested), stratum_size)

        sampled_group = group.sample(n=target_samples, random_state=random_seed)
        sampled_dfs.append(sampled_group)

    if not sampled_dfs:
        logger.warning("No samples selected. Returning empty DataFrame.")
        return df.iloc[:0]

    result = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)

    actual_ratio = len(result) / len(df)
    logger.info(f"Sampled {len(result)} rows from {len(df)} ({actual_ratio * 100:.1f}%)")

    return result
