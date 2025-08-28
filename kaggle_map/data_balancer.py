"""Create a balanced dataset by undersampling overrepresented categories."""

import pandas as pd
from loguru import logger


def balance_dataset(
    input_path: str = "dataset/synth_original_367k_unbalanced.csv",
    output_path: str = "dataset/synth_undersampled_2330_per_cat.csv",
    strategy: str = "undersample",
) -> pd.DataFrame:
    """Balance the dataset by undersampling to the smallest category or a specified size.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the balanced CSV file
        strategy: Either 'undersample' (match smallest) or 'equal_n' (equal samples per category)

    Returns:
        Balanced DataFrame
    """
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Total samples: {len(df)}")
    logger.info("Current distribution:")
    category_counts = df["Category"].value_counts()
    for cat, count in category_counts.items():
        logger.info(f"  {cat}: {count} ({count / len(df) * 100:.1f}%)")

    if strategy == "undersample":
        # Match the smallest category size
        min_samples = category_counts.min()
        logger.info(
            f"\nUsing undersampling strategy: matching smallest category ({min_samples} samples)"
        )
    else:
        # Use a fixed number (e.g., 2000 samples per category)
        min_samples = 2000
        logger.info(f"\nUsing equal_n strategy: {min_samples} samples per category")

    # Sample from each category
    balanced_dfs = []
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]

        if len(category_df) >= min_samples:
            # Undersample
            sampled = category_df.sample(n=min_samples, random_state=42)
            logger.debug(f"  {category}: sampled {min_samples} from {len(category_df)}")
        else:
            # Keep all samples if we have fewer than needed
            sampled = category_df
            logger.warning(
                f"  {category}: kept all {len(category_df)} samples (less than {min_samples})"
            )

        balanced_dfs.append(sampled)

    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("\nBalanced dataset created:")
    logger.info(f"  Total samples: {len(balanced_df)}")

    new_counts = balanced_df["Category"].value_counts()
    for cat, count in new_counts.items():
        logger.info(f"  {cat}: {count} ({count / len(balanced_df) * 100:.1f}%)")

    # Save the balanced dataset
    logger.info(f"\nSaving balanced dataset to {output_path}")
    balanced_df.to_csv(output_path, index=False)

    return balanced_df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits maintaining category proportions.

    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, (
        "Ratios must sum to 1"
    )

    # Manual stratified split
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for category in df["Category"].unique():
        category_df = df[df["Category"] == category].copy()
        n_samples = len(category_df)

        # Shuffle within category
        category_df = category_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # Split
        train_dfs.append(category_df.iloc[:train_end])
        val_dfs.append(category_df.iloc[train_end:val_end])
        test_dfs.append(category_df.iloc[val_end:])

    # Combine and shuffle
    train_df = (
        pd.concat(train_dfs, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_dfs, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_dfs, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    logger.info("Created stratified splits:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Create balanced dataset
    balanced_df = balance_dataset()

    # Optionally create train/val/test splits
    train_df, val_df, test_df = create_stratified_splits(balanced_df)

    # Save splits
    train_df.to_csv("dataset/synth_undersampled_2330_per_cat_train_70pct.csv", index=False)
    val_df.to_csv("dataset/synth_undersampled_2330_per_cat_val_15pct.csv", index=False)
    test_df.to_csv("dataset/synth_undersampled_2330_per_cat_test_15pct.csv", index=False)

    logger.success("Balanced dataset and splits created successfully!")
