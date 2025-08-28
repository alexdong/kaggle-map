#!/usr/bin/env python
"""Rename dataset files to be more self-explanatory."""

import shutil
from pathlib import Path

from loguru import logger

# Define the renaming mappings
RENAME_MAPPINGS = {
    # Original synthetic dataset
    "synth.csv": "synth_original_367k_unbalanced.csv",

    # Undersampled to smallest category (2,330 samples per category)
    "synth_balanced.csv": "synth_undersampled_2330_per_cat.csv",
    "synth_balanced_train.csv": "synth_undersampled_2330_per_cat_train_70pct.csv",
    "synth_balanced_val.csv": "synth_undersampled_2330_per_cat_val_15pct.csv",
    "synth_balanced_test.csv": "synth_undersampled_2330_per_cat_test_15pct.csv",

    # Balanced to median (59,035 samples per category)
    "synth_balanced_hybrid.csv": "synth_median_balanced_59k_per_cat.csv",

    # Custom balanced (5,000 samples per category)
    "synth_balanced_5k.csv": "synth_balanced_5000_per_cat.csv",

    # Original with inverse frequency weights added
    "synth_weighted.csv": "synth_original_with_inverse_freq_weights.csv",
}

def rename_datasets(dataset_dir: Path = Path("dataset"), dry_run: bool = False) -> None:
    """Rename dataset files to be more descriptive.

    Args:
        dataset_dir: Directory containing dataset files
        dry_run: If True, only show what would be renamed without doing it
    """
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Renaming dataset files in {dataset_dir}")

    renamed_count = 0
    for old_name, new_name in RENAME_MAPPINGS.items():
        old_path = dataset_dir / old_name
        new_path = dataset_dir / new_name

        if old_path.exists():
            if new_path.exists():
                logger.warning(f"  Skipping: {new_name} already exists")
                continue

            if dry_run:
                logger.info(f"  Would rename: {old_name} -> {new_name}")
            else:
                shutil.move(str(old_path), str(new_path))
                logger.success(f"  Renamed: {old_name} -> {new_name}")
            renamed_count += 1
        else:
            logger.debug(f"  Not found: {old_name}")

    logger.info(f"{'Would rename' if dry_run else 'Renamed'} {renamed_count} files")

    # Show current dataset files
    logger.info("\nCurrent dataset files:")
    csv_files = sorted(dataset_dir.glob("*.csv"))
    for file in csv_files:
        size_mb = file.stat().st_size / (1024 * 1024)

        # Try to get row count
        try:
            import pandas as pd
            row_count = len(pd.read_csv(file))
            logger.info(f"  {file.name:50} {size_mb:8.1f} MB  {row_count:8,} rows")
        except Exception:
            logger.info(f"  {file.name:50} {size_mb:8.1f} MB")


if __name__ == "__main__":
    import sys

    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    # Perform renaming
    rename_datasets(dry_run=dry_run)

    if dry_run:
        logger.info("\nTo actually rename files, run without --dry-run flag")
