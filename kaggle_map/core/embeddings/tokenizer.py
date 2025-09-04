"""
Utilities for encoding Question/Answer/Explanation text into embeddings using sentence transformers.

Primary entrypoints:
- `row.to_embedding_text()` - Creates normalized Q/A/E text (legacy single embedding)
- `compute_concatenated_embeddings()` - Standard approach for 768-dim concatenated embeddings
- `compute_single_embeddings()` - Legacy approach for single embeddings
- `get_tokenizer()` - Get sentence transformer model
"""

import numpy as np

from kaggle_map.core.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.core.embeddings.utils import compute_concatenated_embeddings, compute_single_embeddings  # noqa: F401
from kaggle_map.core.models import EvaluationRow


def main() -> None:
    # Use the centralized get_tokenizer function
    tokenizer = get_tokenizer()
    row = EvaluationRow(
        row_id=1,
        question_id=1001,
        question_text="What is 2 + 2?",
        mc_answer=r"\( \frac{4}{1} \)",
        student_explanation="The answer is four.",
    )

    # Demonstrate both approaches
    print("=== Legacy Single Embedding Approach ===")
    text = row.to_embedding_text()
    print(f"Text: {text}")
    embeddings = tokenizer.encode(text)
    print(f"Single embedding shape: {embeddings.shape}")
    print(f"Single embedding dtype: {embeddings.dtype}")

    print("\n=== New Concatenated Embedding Approach ===")
    # Separate question and answer
    question_emb = tokenizer.encode(row.question_text)
    answer_text = f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}"
    answer_emb = tokenizer.encode(answer_text)

    concatenated_emb = np.concatenate([question_emb, answer_emb])
    print(f"Question embedding shape: {question_emb.shape}")
    print(f"Answer embedding shape: {answer_emb.shape}")
    print(f"Concatenated embedding shape: {concatenated_emb.shape}")

    print(f"\nModel: {EmbeddingModel.MINI_LM.model_id}")
    print("Expected base dimensions: 384")
    print(f"Expected final dimensions (2x base): {EmbeddingModel.MINI_LM.dim}")


if __name__ == "__main__":
    main()
