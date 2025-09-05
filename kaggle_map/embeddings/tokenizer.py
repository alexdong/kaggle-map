"""
Utilities for encoding Question/Answer/Explanation text into embeddings using Qwen3-Embedding-8B.

Primary entrypoints:
- `row.to_embedding_text()` - Creates normalized Q/A/E text
- `compute_concatenated_embeddings()` - Compute embeddings (using Qwen3)
- `compute_single_embeddings()` - Same as concatenated (single model approach)
"""

from kaggle_map.core.models import EvaluationRow
from kaggle_map.embeddings.embedding_models import QwenEmbeddingModel
from kaggle_map.embeddings.utils import compute_concatenated_embeddings, compute_single_embeddings  # noqa: F401


def main() -> None:
    # Initialize Qwen3-8B model with Q8_0 quantization
    tokenizer = QwenEmbeddingModel()

    row = EvaluationRow(
        row_id=1,
        question_id=1001,
        question_text="What is 2 + 2?",
        mc_answer=r"\( \frac{4}{1} \)",
        student_explanation="The answer is four.",
    )

    # Demonstrate encoding
    print("=== Qwen3-8B Embedding (Q8_0) ===")
    text = row.to_embedding_text()
    print(f"Text: {text}")
    embeddings = tokenizer.encode(text)
    print(f"Single embedding shape: {embeddings.shape}")
    print(f"Single embedding dtype: {embeddings.dtype}")

    print("\n=== Batch Encoding ===")
    # Test batch encoding
    question_text = row.question_text
    answer_text = f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}"

    batch_texts = [question_text, answer_text]
    batch_emb = tokenizer.encode(batch_texts)
    print(f"Batch embedding shape: {batch_emb.shape}")

    print("\nModel: Qwen3-Embedding-8B")
    print("Quantization: Q8_0 (8-bit)")
    print(f"Embedding dimensions: {tokenizer.embedding_dim}")


if __name__ == "__main__":
    main()

