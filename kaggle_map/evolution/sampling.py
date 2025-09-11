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

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("=== Sampling Module Validation ===")

    # Load sample data
    error_prediction_path = Path("datasets/error_prediction.csv")
    df = pd.read_csv(error_prediction_path)
    logger.info(f"Loaded error prediction data: {len(df)} rows")

    logger.info(f"\nDataset columns: {list(df.columns)}")
    logger.info(f"Dataset shape: {df.shape}")

    # Test different sampling ratios
    logger.info("\n1. Testing different sampling ratios:")
    for ratio in [0.01, 0.05, 0.1, 0.2]:
        sampled = stratified_sample(df, sample_ratio=ratio, random_seed=42)
        logger.info(f"  Ratio {ratio:.0%}: {len(sampled)} samples ({len(sampled)/len(df)*100:.1f}% actual)")

    # Test stratification preservation
    logger.info("\n2. Testing stratification preservation (10% sample):")
    sampled = stratified_sample(df, sample_ratio=0.1, random_seed=42)

    if "QuestionId" in df.columns:
        orig_dist = df["QuestionId"].value_counts(normalize=True).head(5)
        sample_dist = sampled["QuestionId"].value_counts(normalize=True).head(5)

        logger.info("  Original distribution (top 5 questions):")
        for qid, pct in orig_dist.items():
            logger.info(f"    Q{qid}: {pct:.1%}")

        logger.info("  Sample distribution (top 5 questions):")
        for qid, pct in sample_dist.items():
            logger.info(f"    Q{qid}: {pct:.1%}")

    # Test minimum samples per stratum
    logger.info("\n3. Testing minimum samples per stratum:")
    for min_samples in [1, 3, 5]:
        sampled = stratified_sample(
            df,
            sample_ratio=0.01,
            min_samples_per_stratum=min_samples,
            random_seed=42
        )
        logger.info(f"  Min {min_samples} samples: {len(sampled)} total samples")

    # Test reproducibility
    logger.info("\n4. Testing reproducibility (same seed):")
    sample1 = stratified_sample(df, sample_ratio=0.1, random_seed=42)
    sample2 = stratified_sample(df, sample_ratio=0.1, random_seed=42)
    logger.info(f"  Sample 1: {len(sample1)} rows")
    logger.info(f"  Sample 2: {len(sample2)} rows")
    logger.info(f"  ✓ Identical: {sample1.equals(sample2)}")

    # Test different seeds
    logger.info("\n5. Testing different seeds:")
    sample3 = stratified_sample(df, sample_ratio=0.1, random_seed=123)
    logger.info(f"  Seed 42: {len(sample1)} rows")
    logger.info(f"  Seed 123: {len(sample3)} rows")
    logger.info(f"  ✓ Different: {not sample1.equals(sample3)}")

    # Test edge cases
    logger.info("\n6. Testing edge cases:")

    # Very small sample
    tiny_sample = stratified_sample(df, sample_ratio=0.001, random_seed=42)
    logger.info(f"  0.1% sample: {len(tiny_sample)} rows")

    # Large sample
    large_sample = stratified_sample(df, sample_ratio=0.5, random_seed=42)
    logger.info(f"  50% sample: {len(large_sample)} rows")

    # Full sample
    full_sample = stratified_sample(df, sample_ratio=1.0, random_seed=42)
    logger.info(f"  100% sample: {len(full_sample)} rows (should equal {len(df)})")

    logger.info("\n✅ Sampling validation complete!")
