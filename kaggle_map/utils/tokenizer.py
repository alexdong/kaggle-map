"""
Utilities for building text from Question/Answer/Explanation triples and encoding them
into embeddings using sentence transformers.

Primary entrypoints:
- `build_qae_text(row)` - Create normalized text from a TrainingRow
- `encode(model, text)` - Encode text into vector embeddings
"""

from kaggle_map.models import EvaluationRow
from kaggle_map.utils.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.utils.formula import normalize_latex_answer, normalize_text


def build_qae_text(row: EvaluationRow) -> str:
    """Compose the canonical Q/A/E string used for embeddings from a TrainingRow.

    Example output:
        "Question: {question}, Answer: {normalized_answer}, Explanation: {explanation}"
    """
    q = normalize_text(row.question_text)
    a = normalize_latex_answer(row.mc_answer)
    e = normalize_text(row.student_explanation)
    return f"Question: {q}, Answer: {a}, Explanation: {e}"


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

    text = build_qae_text(row)
    print(f"Text: {text}")
    embeddings = tokenizer.encode(text)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Model: {EmbeddingModel.MINI_LM.model_id}")
    print(f"Expected dimensions: {EmbeddingModel.MINI_LM.dim}")

if __name__ == "__main__":
    main()
