"""
Quick example for building BERT inputs from Question/Answer/Explanation
using the truncation policy that preserves Q+A and truncates Explanation first.

Run:
    python kaggle_map/tokenizer_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Allow running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from kaggle_map.utils.tokenizer import (
    build_qae_text,
    batch_encode_qae_expl_truncated_first,
)


def main() -> None:
    df = pd.read_csv(
        "datasets/train_original.csv",
        usecols=["QuestionText", "MC_Answer", "StudentExplanation"],
    ).fillna("")

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Demo with a conservative max_length to show truncation behavior
    max_len = 128
    batch = batch_encode_qae_expl_truncated_first(
        tok,
        df["QuestionText"].tolist(),
        df["MC_Answer"].tolist(),
        df["StudentExplanation"].tolist(),
        max_length=max_len,
        pad_to_max_length=True,
    )

    # Simple inspection
    lengths = [sum(m) for m in batch["attention_mask"]]
    over_just_by_padding = sum(1 for L in lengths if L == max_len)
    print(f"Encoded {len(lengths)} rows; max_length={max_len}")
    print(f"Rows at cap (likely truncated or exactly full): {over_just_by_padding}")

    # Show a sample
    i = 0
    s = build_qae_text(
        df.iloc[i]["QuestionText"], df.iloc[i]["MC_Answer"], df.iloc[i]["StudentExplanation"]
    )
    print("\nSample concatenated text:\n", s)
    print("\nTokenized length:", lengths[i])


if __name__ == "__main__":
    main()
