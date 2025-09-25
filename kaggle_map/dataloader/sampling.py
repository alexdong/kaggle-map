"""Stratified sampling utilities for data loading.

Example:
    uv run -m kaggle_map.dataloader.sampling

This prints a 1% stratified sample from ``datasets/33474_full_train.csv``.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.core.random_seed import configure_random_seed, get_active_seed
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

DEFAULT_STRATIFY_COLS: tuple[str, ...] = ("QuestionId", "Category", "MC_Answer")
DEFAULT_INPUT_CSV = Path("datasets/33474_full_train.csv")
DEFAULT_SAMPLE_RATIO = 0.01


def stratified_sample(
    df: pd.DataFrame,
    sample_ratio: float = 0.1,
    stratify_cols: Sequence[str] = DEFAULT_STRATIFY_COLS,
    min_samples_per_stratum: int = 3,
) -> pd.DataFrame:
    assert 0.0 < sample_ratio <= 1.0, f"sample_ratio must be between 0 and 1, got {sample_ratio}"
    assert min_samples_per_stratum > 0, "min_samples_per_stratum must be positive"
    assert stratify_cols, "stratify_cols must contain at least one column name"

    seed = get_active_seed()
    available_cols = [col for col in stratify_cols if col in df.columns]
    if not available_cols:
        logger.warning("No stratification columns found in dataframe. Using simple random sampling.")
        sampled = df.sample(frac=sample_ratio, random_state=seed)
        return sampled.reset_index(drop=True)

    logger.info("Stratified sampling with ratio={} stratify_by={}", sample_ratio, available_cols)

    grouped = df.groupby(available_cols, group_keys=False, sort=False)

    sampled_dfs: list[pd.DataFrame] = []

    for _name, group in grouped:
        stratum_size = len(group)
        requested = max(1, int(stratum_size * sample_ratio))
        guaranteed = min(min_samples_per_stratum, stratum_size)
        target_samples = min(max(guaranteed, requested), stratum_size)

        sampled_group = group.sample(n=target_samples, random_state=seed)
        sampled_dfs.append(sampled_group)

    if not sampled_dfs:
        logger.warning("No samples selected. Returning empty DataFrame.")
        return df.iloc[:0]

    result = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)

    actual_ratio = len(result) / len(df)
    logger.info(f"Sampled {len(result)} rows from {len(df)} ({actual_ratio * 100:.1f}%)")

    return result


def main() -> None:
    """Print a stratified 1% sample from the canonical full-train CSV."""

    logger.debug(f"Reading dataframe from {DEFAULT_INPUT_CSV}")
    configure_random_seed()

    df = pd.read_csv(DEFAULT_INPUT_CSV)
    logger.info("Loaded {} rows from {}", len(df), DEFAULT_INPUT_CSV)

    sampled = stratified_sample(
        df=df,
        sample_ratio=DEFAULT_SAMPLE_RATIO,
        stratify_cols=DEFAULT_STRATIFY_COLS,
        min_samples_per_stratum=3,
    )

    logger.success("Sampled {} rows ({}%)", len(sampled), DEFAULT_SAMPLE_RATIO * 100)

    if sampled.empty:
        print("No rows sampled.")
        return

    print(sampled.to_string(index=False))


if __name__ == "__main__":
    main()
