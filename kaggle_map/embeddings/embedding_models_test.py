import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from kaggle_map.embeddings.embedding_models import (
    QwenEmbeddingModel,
    QuantizationLevel,
    get_tokenizer,
)


def test_quantization_metadata():
    """Test quantization level metadata."""
    assert QuantizationLevel.Q8_0.spec.size_gb == 8.6
    assert QuantizationLevel.Q4_K_M.spec.size_gb == 5.41
    assert QuantizationLevel.F16.spec.filename == "Qwen3-Embedding-8B-F16.gguf"


def test_embedding_dimensions():
    """Test that Qwen model has correct dimensions."""
    assert QwenEmbeddingModel.EMBEDDING_DIM == 5120


@patch("kaggle_map.embeddings.embedding_models.Llama")
@patch("huggingface_hub.hf_hub_download")
def test_model_initialization(mock_download, mock_llama):
    """Test model initialization with mocked llama.cpp."""
    mock_download.return_value = "/path/to/model.gguf"
    mock_llama_instance = MagicMock()
    mock_llama.return_value = mock_llama_instance
    
    model = QwenEmbeddingModel(quantization=QuantizationLevel.Q4_K_M)
    
    # Check that model was initialized with correct parameters
    mock_llama.assert_called_once()
    call_kwargs = mock_llama.call_args.kwargs
    assert call_kwargs["embedding"] == True
    assert call_kwargs["model_path"] == "/path/to/model.gguf"


@patch("kaggle_map.embeddings.embedding_models.Llama")
@patch("huggingface_hub.hf_hub_download")
def test_encode_single_text(mock_download, mock_llama):
    """Test encoding single text."""
    mock_download.return_value = "/path/to/model.gguf"
    
    # Mock llama instance and embed method
    mock_llama_instance = MagicMock()
    mock_embedding = np.random.randn(5120).tolist()
    mock_llama_instance.embed.return_value = mock_embedding
    mock_llama.return_value = mock_llama_instance
    
    model = QwenEmbeddingModel(quantization=QuantizationLevel.Q4_K_M)
    
    # Test single text encoding
    text = "Test text"
    embedding = model.encode(text)
    
    assert embedding.shape == (5120,)
    assert embedding.dtype == np.float32
    # Check normalization
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01


@patch("kaggle_map.embeddings.embedding_models.Llama")
@patch("huggingface_hub.hf_hub_download")
def test_encode_batch_texts(mock_download, mock_llama):
    """Test batch text encoding."""
    mock_download.return_value = "/path/to/model.gguf"
    
    # Mock llama instance and embed method
    mock_llama_instance = MagicMock()
    mock_llama_instance.embed.side_effect = [
        np.random.randn(5120).tolist(),
        np.random.randn(5120).tolist(),
        np.random.randn(5120).tolist(),
    ]
    mock_llama.return_value = mock_llama_instance
    
    model = QwenEmbeddingModel(quantization=QuantizationLevel.Q6_K)
    
    # Test batch encoding
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = model.encode(texts)
    
    assert embeddings.shape == (3, 5120)
    assert embeddings.dtype == np.float32
    # Check that each embedding is normalized
    for i in range(3):
        assert np.abs(np.linalg.norm(embeddings[i]) - 1.0) < 0.01


def test_get_tokenizer_function():
    """Test the get_tokenizer function interface."""
    with patch("kaggle_map.embeddings.embedding_models.QwenEmbeddingModel") as mock_model:
        tokenizer = get_tokenizer(quantization=QuantizationLevel.Q5_K_M)
        
        mock_model.assert_called_once_with(
            quantization=QuantizationLevel.Q5_K_M,
            verbose=False,
        )