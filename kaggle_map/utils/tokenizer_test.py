import pytest
import numpy as np

from kaggle_map.utils.tokenizer import build_qae_text, encode, batch_encode
from kaggle_map.utils.embeddings import get_tokenizer, EmbeddingModel
from kaggle_map.models import TrainingRow, Category


def test_build_qae_text():
    """Test that build_qae_text properly normalizes and formats text."""
    row = TrainingRow(
        row_id=1,
        question_id=1001,
        question_text="What is  2 + 2  ?",
        mc_answer=r"\( \frac{4}{1} \)",
        student_explanation="The answer   is   four.",
        category=Category.TRUE_CORRECT,
        misconception=None
    )
    
    result = build_qae_text(row)
    expected = "Question: What is 2 + 2 ?, Answer: 4/1, Explanation: The answer is four."
    assert result == expected


def test_encode_single_text():
    """Test encoding a single text into embeddings."""
    try:
        model = get_tokenizer(EmbeddingModel.MINI_LM)
        text = "Question: What is 2+2?, Answer: 4, Explanation: Basic addition."
        
        embedding = encode(model, text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (EmbeddingModel.MINI_LM.dim,)
        assert embedding.dtype == np.float32
        
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_batch_encode():
    """Test batch encoding multiple texts."""
    try:
        model = get_tokenizer(EmbeddingModel.MINI_LM)
        texts = [
            "Question: What is 2+2?, Answer: 4, Explanation: Basic addition.",
            "Question: What is 3+3?, Answer: 6, Explanation: Another addition.",
        ]
        
        embeddings = batch_encode(model, texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, EmbeddingModel.MINI_LM.dim)
        assert embeddings.dtype == np.float32
        
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_integration_build_and_encode():
    """Test the full pipeline from TrainingRow to embeddings."""
    try:
        model = get_tokenizer(EmbeddingModel.MINI_LM)
        
        row = TrainingRow(
            row_id=2,
            question_id=1002,
            question_text="What is the square root of 16?",
            mc_answer="4",
            student_explanation="Since 4 * 4 = 16, the square root is 4.",
            category=Category.TRUE_CORRECT,
            misconception=None
        )
        
        # Build text
        text = build_qae_text(row)
        assert "Question:" in text
        assert "Answer:" in text
        assert "Explanation:" in text
        
        # Encode text
        embedding = encode(model, text)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == EmbeddingModel.MINI_LM.dim
        
    except ImportError:
        pytest.skip("sentence-transformers not installed")
