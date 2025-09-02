from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from kaggle_map.core.embeddings.formula import normalize_latex_answer


def rerank_predictions(
    question: str,
    answer: str,
    explanation: str,
    predictions: str,
) -> str:
    """Rerank predictions using LLM via basic HTTP request."""

    normalized_question = normalize_latex_answer(question)
    normalized_answer = normalize_latex_answer(answer)

    prompt = f"""You are a math educator. Your job is to review a student's answer and explanation carefully with the goal to re-order the potential labels.

Question: {normalized_question}

Answer: {normalized_answer}

Explanation: {explanation}

Labels: {predictions}


Reply by re-rank the labels and put the most likely ones to the beginning.
Separated with a |.

Only return the labels in a single line. Nothing else."""

    # Prepare the request payload
    payload = {
        "model": "google/gemma-3-12b",
        "messages": [
            {
                "role": "system",
                "content": "You are a math educator helping to identify student misconceptions.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
    }

    try:
        # Make the HTTP request to LM Studio
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,  # 30 second timeout
        )
        response.raise_for_status()

        # Extract the response text
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content.strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling LLM API: {e}")
        return predictions  # Return original if error
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing LLM response: {e}")
        return predictions  # Return original if error


def process_dataframe(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Process the dataframe with LLM reranking.

    Args:
        df: Input dataframe to process
        sample_size: Number of random rows to process (default: 100)
    """
    # Sample random rows
    logger.info(f"Sampling {sample_size} random rows from {len(df)} total rows")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Add new columns
    df_sample["LLM_top_1"] = ""
    df_sample["LLM_top_3_predictions"] = ""
    df_sample["LLM_correct"] = ""  # New column for emoji indicator
    df_sample["actual_label"] = df_sample.apply(
        lambda row: f"{row['Category']}:{row['actual_misconception'] if pd.notna(row['actual_misconception']) and row['actual_misconception'] else 'NA'}",
        axis=1,
    )

    # Process each row with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Reranking {len(df_sample)} predictions...", total=len(df_sample))

        for _counter, (idx, row) in enumerate(df_sample.iterrows(), 1):
            # Rerank predictions
            reranked = rerank_predictions(
                row["QuestionText"],
                row["MC_Answer"],
                row["StudentExplanation"],
                row["top_3_predictions_formatted"],
            )

            # Parse reranked results
            if reranked and "|" in reranked:
                labels = [label.strip() for label in reranked.split("|")]
                llm_top_1 = labels[0] if labels else ""
                df_sample.at[idx, "LLM_top_1"] = llm_top_1
                df_sample.at[idx, "LLM_top_3_predictions"] = reranked
            else:
                # Fallback to original if parsing fails
                df_sample.at[idx, "LLM_top_3_predictions"] = row["top_3_predictions_formatted"]
                original_labels = [label.strip() for label in row["top_3_predictions_formatted"].split("|")]
                llm_top_1 = original_labels[0] if original_labels else ""
                df_sample.at[idx, "LLM_top_1"] = llm_top_1

            # Add emoji indicator for correctness
            actual_label = df_sample.at[idx, "actual_label"]
            df_sample.at[idx, "LLM_correct"] = "✅" if llm_top_1 == actual_label else "❌"

            progress.update(task, advance=1)

    return df_sample


def main() -> None:
    """Main entry point for reranking error predictions."""
    csv_path = Path("datasets/error_prediction.csv")
    output_path = Path("datasets/error_prediction_llm_reranked_sample.csv")

    if not csv_path.exists():
        logger.error(f"Error prediction file not found: {csv_path}")
        return

    logger.info(f"Loading error predictions from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Total rows in dataset: {len(df)}")
    df_sample = process_dataframe(df, sample_size=100)

    # Save the sampled and reranked dataframe
    logger.info(f"Saving reranked sample to {output_path}")
    df_sample.to_csv(output_path, index=False)

    logger.success(f"Successfully processed and saved {len(df_sample)} rows")

    # Calculate and show accuracy statistics
    correct_count = (df_sample["LLM_correct"] == "✅").sum()
    total_count = len(df_sample)
    accuracy = correct_count / total_count * 100

    logger.info(f"LLM Reranking Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
    logger.info(f"✅ Correct: {correct_count}")
    logger.info(f"❌ Incorrect: {total_count - correct_count}")

    # Show sample results with emojis
    logger.info("\n=== Sample Reranked Results with Emoji Indicators ===")
    sample_cols = ["row_id", "actual_label", "LLM_top_1", "LLM_correct"]

    # Display the first 20 rows to show the emoji indicators
    print("\nFirst 20 results:")
    print(df_sample[sample_cols].head(20).to_string())

    # Also show a summary of correct vs incorrect
    print("\n" + "="*60)
    print("Summary by Correctness:")
    print(df_sample["LLM_correct"].value_counts().to_string())


if __name__ == "__main__":
    main()
