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


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe with LLM reranking."""
    # Add new columns
    df["LLM_top_1"] = ""
    df["LLM_top_3_predictions"] = ""
    df["actual_label"] = df.apply(
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
        task = progress.add_task("Reranking predictions...", total=len(df))
        
        for idx, row in df.iterrows():
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
                df.at[idx, "LLM_top_1"] = labels[0] if labels else ""
                df.at[idx, "LLM_top_3_predictions"] = reranked
            else:
                # Fallback to original if parsing fails
                df.at[idx, "LLM_top_3_predictions"] = row["top_3_predictions_formatted"]
                original_labels = [label.strip() for label in row["top_3_predictions_formatted"].split("|")]
                df.at[idx, "LLM_top_1"] = original_labels[0] if original_labels else ""
            
            progress.update(task, advance=1)
    
    return df


def main() -> None:
    """Main entry point for reranking error predictions."""
    csv_path = Path("datasets/error_prediction.csv")
    
    if not csv_path.exists():
        logger.error(f"Error prediction file not found: {csv_path}")
        return
    
    logger.info(f"Loading error predictions from {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Processing {len(df)} rows...")
    df = process_dataframe(df)
    
    # Save the updated dataframe
    logger.info(f"Saving reranked predictions to {csv_path}")
    df.to_csv(csv_path, index=False)
    
    logger.success(f"Successfully processed and saved {len(df)} rows")
    
    # Show sample results
    logger.info("Sample reranked results:")
    sample_cols = ["row_id", "actual_label", "top_3_predictions_formatted", "LLM_top_1", "LLM_top_3_predictions"]
    print(df[sample_cols].head())


if __name__ == "__main__":
    main()