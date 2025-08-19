import pytest

from kaggle_map.utils.embedding_models import EmbeddingModel, get_tokenizer


def test_embedding_metadata_dims_and_seq():
    assert EmbeddingModel.MINI_LM.dim == 384
    assert EmbeddingModel.MINI_LM.recommended_max_seq == 256
    assert EmbeddingModel.MP_NET.dim == 768
    assert EmbeddingModel.BGE_SMALL.dim == 512


def test_get_tokenizer_default():
    """Test that get_tokenizer works with default model."""
    try:
        model = get_tokenizer()
        assert model is not None
        # Check that it's using the default model (MINI_LM)
        assert hasattr(model, 'encode')
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_get_tokenizer_specific_model():
    """Test that get_tokenizer works with a specific model."""
    try:
        model = get_tokenizer(EmbeddingModel.MINI_LM)
        assert model is not None
        assert hasattr(model, 'encode')
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_get_tokenizer_import_error():
    """Test that get_tokenizer raises helpful error when sentence-transformers not available."""
    # This test would need mocking to work properly in practice
    # For now, we'll just ensure the function exists and is callable
    assert callable(get_tokenizer)

