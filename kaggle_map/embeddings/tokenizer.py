"""
Utilities for encoding Question/Answer/Explanation text into embeddings using Qwen3-Embedding-8B.

Primary entrypoints:
- `row.to_embedding_text()` - Creates normalized Q/A/E text
- `compute_concatenated_embeddings()` - Compute embeddings (now using Qwen3)
- `compute_single_embeddings()` - Legacy approach for single embeddings
- `get_tokenizer()` - Get Qwen3 embedding model
"""


from kaggle_map.core.models import EvaluationRow
from kaggle_map.embeddings.embedding_models import QuantizationLevel, get_tokenizer
from kaggle_map.embeddings.utils import compute_concatenated_embeddings, compute_single_embeddings  # noqa: F401


def main() -> None:
    # Use 4-bit quantization for testing as requested
    tokenizer = get_tokenizer(quantization=QuantizationLevel.Q4_K_M)
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

    print("\n=== Qwen3-8B Embedding Approach ===")
    # Test both single and batch encoding
    question_emb = tokenizer.encode(row.question_text)
    answer_text = f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}"
    answer_emb = tokenizer.encode(answer_text)

    print(f"Question embedding shape: {question_emb.shape}")
    print(f"Answer embedding shape: {answer_emb.shape}")

    # Test batch encoding
    batch_texts = [row.question_text, answer_text]
    batch_emb = tokenizer.encode(batch_texts)
    print(f"Batch embedding shape: {batch_emb.shape}")

    print("\nModel: Qwen3-Embedding-8B")
    print("Quantization: 4-bit (Q4_K_M)")
    print(f"Expected dimensions: {tokenizer.embedding_dim}")


if __name__ == "__main__":
    main()
