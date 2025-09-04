"""
Utilities for encoding Question/Answer/Explanation text into embeddings using sentence transformers.

Primary entrypoints:
- `row.to_embedding_text()` - Creates normalized Q/A/E text
- `encode(model, text)` - Encode text into vector embeddings
"""

from kaggle_map.core.embeddings.embedding_models import EmbeddingModel, get_tokenizer
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

    text = row.to_embedding_text()
    print(f"Text: {text}")
    embeddings = tokenizer.encode(text)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Model: {EmbeddingModel.MINI_LM.model_id}")
    print(f"Expected dimensions: {EmbeddingModel.MINI_LM.dim}")


if __name__ == "__main__":
    main()
