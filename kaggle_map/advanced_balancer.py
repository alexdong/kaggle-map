"""Advanced dataset balancing with multiple strategies."""

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


def balance_dataset_advanced(
    input_path: str = "dataset/synth.csv",
    output_path: str = "dataset/synth_balanced_advanced.csv",
    strategy: Literal["undersample", "oversample", "hybrid", "weighted"] = "hybrid",
    target_samples_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance dataset using various strategies.

    Args:
        input_path: Path to input CSV
        output_path: Path to save balanced CSV
        strategy: Balancing strategy to use:
            - 'undersample': Reduce to smallest class size
            - 'oversample': Increase to largest class size (with replacement)
            - 'hybrid': Balance to median class size
            - 'weighted': Create weighted samples for training
        target_samples_per_class: Override automatic target calculation
        seed: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Original dataset: {len(df)} samples")
    category_counts = df["Category"].value_counts()

    # Determine target samples per class based on strategy
    if target_samples_per_class is None:
        if strategy == "undersample":
            target_samples_per_class = category_counts.min()
        elif strategy == "oversample":
            target_samples_per_class = category_counts.max()
        elif strategy == "hybrid":
            target_samples_per_class = int(category_counts.median())
        elif strategy == "weighted":
            # For weighted, we keep original data but add weights
            logger.info(
                "Using weighted strategy - keeping original data with sample weights"
            )
            return add_sample_weights(df, output_path)

    logger.info(
        f"Strategy: {strategy}, Target samples per class: {target_samples_per_class}"
    )

    balanced_dfs = []

    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        n_samples = len(category_df)

        if n_samples >= target_samples_per_class:
            # Undersample
            sampled = category_df.sample(n=target_samples_per_class, random_state=seed)
            logger.info(
                f"  {category}: undersampled {target_samples_per_class} from {n_samples}"
            )
        else:
            # Oversample with replacement
            sampled = category_df.sample(
                n=target_samples_per_class, replace=True, random_state=seed
            )
            logger.info(
                f"  {category}: oversampled {target_samples_per_class} from {n_samples}"
            )

        balanced_dfs.append(sampled)

    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"\nBalanced dataset created: {len(balanced_df)} total samples")
    for cat, count in balanced_df["Category"].value_counts().items():
        logger.info(f"  {cat}: {count} ({count / len(balanced_df) * 100:.1f}%)")

    # Save
    logger.info(f"Saving to {output_path}")
    balanced_df.to_csv(output_path, index=False)

    return balanced_df


def add_sample_weights(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add sample weights inversely proportional to class frequency.

    This is useful for weighted loss functions during training.
    """
    category_counts = df["Category"].value_counts()
    total_samples = len(df)

    # Calculate weights: inverse of class frequency
    weights = {}
    for category, count in category_counts.items():
        # Weight = total_samples / (n_classes * class_count)
        weights[category] = total_samples / (len(category_counts) * count)

    # Normalize weights so they average to 1
    mean_weight = np.mean(list(weights.values()))
    weights = {k: v / mean_weight for k, v in weights.items()}

    # Add weight column
    df["sample_weight"] = df["Category"].map(weights)

    logger.info("Sample weights by category:")
    for category, weight in weights.items():
        count = category_counts[category]
        logger.info(f"  {category}: weight={weight:.3f} (count={count})")

    # Save with weights
    df.to_csv(output_path, index=False)

    return df


def create_augmented_balanced_dataset(
    input_path: str = "dataset/synth.csv",
    output_path: str = "dataset/synth_augmented.csv",
    augmentation_factor: float = 1.5,
) -> pd.DataFrame:
    """Create balanced dataset with text augmentation for minority classes.

    This creates synthetic variations of student explanations to increase diversity.
    """
    df = pd.read_csv(input_path)
    category_counts = df["Category"].value_counts()
    target_samples = int(category_counts.median() * augmentation_factor)

    augmented_dfs = []

    for category in df["Category"].unique():
        category_df = df[df["Category"] == category].copy()
        n_samples = len(category_df)

        if n_samples >= target_samples:
            # Undersample
            sampled = category_df.sample(n=target_samples, random_state=42)
            augmented_dfs.append(sampled)
        else:
            # Keep original samples
            augmented_dfs.append(category_df)

            # Create augmented samples
            n_augmented = target_samples - n_samples
            augmented_samples = []

            for _ in range(n_augmented):
                # Sample a random row to augment
                row = category_df.sample(n=1).iloc[0].copy()

                # Simple text augmentation strategies
                explanation = row["StudentExplanation"]

                # Randomly apply one augmentation
                aug_type = np.random.choice(["truncate", "add_filler", "rephrase"])

                if aug_type == "truncate" and len(explanation) > 20:
                    # Truncate explanation
                    cutoff = np.random.randint(15, min(len(explanation), 50))
                    row["StudentExplanation"] = explanation[:cutoff] + "..."

                elif aug_type == "add_filler":
                    # Add filler words
                    fillers = ["Um, ", "Like, ", "So, ", "Well, ", "I think "]
                    row["StudentExplanation"] = (
                        np.random.choice(fillers) + explanation.lower()
                    )

                # Simple rephrasing by changing word order
                elif "because" in explanation.lower():
                    parts = explanation.lower().split("because")
                    if len(parts) == 2:
                        row["StudentExplanation"] = (
                            f"{parts[1].strip()}, so {parts[0].strip()}"
                        )

                # Mark as augmented
                row["row_id"] = f"{row['row_id']}_aug{_}"
                augmented_samples.append(row)

            if augmented_samples:
                aug_df = pd.DataFrame(augmented_samples)
                augmented_dfs.append(aug_df)
                logger.info(
                    f"  {category}: added {len(augmented_samples)} augmented samples"
                )

    # Combine all
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Augmented dataset: {len(final_df)} total samples")
    final_df.to_csv(output_path, index=False)

    return final_df


if __name__ == "__main__":
    # Example 1: Hybrid strategy (balance to median)
    logger.info("=== HYBRID BALANCING ===")
    hybrid_df = balance_dataset_advanced(
        strategy="hybrid", output_path="dataset/synth_balanced_hybrid.csv"
    )

    # Example 2: Weighted samples (for use with weighted loss)
    logger.info("\n=== WEIGHTED SAMPLES ===")
    weighted_df = balance_dataset_advanced(
        strategy="weighted", output_path="dataset/synth_weighted.csv"
    )

    # Example 3: Custom target
    logger.info("\n=== CUSTOM TARGET (5000 samples per class) ===")
    custom_df = balance_dataset_advanced(
        strategy="hybrid",
        target_samples_per_class=5000,
        output_path="dataset/synth_balanced_5k.csv",
    )
