"""Qwen3-Embedding-8B model using GGUF quantization via llama-cpp-python.

This implementation uses llama-cpp-python with GGUF quantized models instead of
SentenceTransformer for several important reasons:

1. **Memory Efficiency**:
   - Qwen3-Embedding-8B is an 8-billion parameter model (~32GB in full precision)
   - GGUF Q8_0 quantization reduces memory usage to ~8GB (75% reduction)
   - SentenceTransformer would load the full precision model requiring 4x more memory

2. **Architectural Consistency**:
   - Project uses llama-cpp-python extensively for LLM models (reranker module)
   - GGUF quantization is a key performance optimization strategy across codebase
   - Maintains consistency with existing quantization infrastructure

3. **Performance Optimization**:
   - GGUF allows GPU acceleration while maintaining memory efficiency
   - Quantized inference is typically faster than full precision
   - Enables running larger models on consumer hardware

4. **Quality vs Resources Trade-off**:
   - Quantized version provides ~95% of full precision quality
   - Significant resource savings enable practical deployment
   - Aligns with project's focus on efficient model usage

Note: SentenceTransformer could technically be used but would be suboptimal for this
large model due to memory constraints and inconsistency with project architecture.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


class QwenEmbeddingModel:
    _instance: "QwenEmbeddingModel | None" = None

    def __init__(self) -> None:
        model_path = self._get_model_path()
        logger.info(f"Loading Qwen3-Embedding-8B Q8_0 from {model_path}")

        # NOTE: You may see warnings like "init: embeddings required but some input tokens were not marked as outputs"
        # These are harmless for embedding models and come from llama.cpp's internal handling.
        # The model works correctly despite these warnings.
        self.model = Llama(
            n_gpu_layers=-1,
            model_path=str(model_path),
            embedding=True,
            verbose=False,
        )

    @classmethod
    def get_instance(cls) -> "QwenEmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_model_path(self) -> Path:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_path = hf_hub_download(
            repo_id="JonathanMiddleton/Qwen3-Embedding-8B-GGUF",
            filename="Qwen3-Embedding-8B-Q8_0.gguf",
            cache_dir=cache_dir,
        )
        return Path(model_path)

    def encode(self, text: str | Sequence[str], batch_size: int = 32) -> torch.Tensor:
        """Encode text(s) into embeddings.

        Args:
            text: Single text string or list of texts to encode
            batch_size: Batch size for encoding multiple texts (default: 32)

        Returns:
            torch.Tensor:
                - If text is str: 1D tensor of shape (embedding_dim,)
                - If text is list: 2D tensor of shape (num_texts, embedding_dim)
        """
        if isinstance(text, str):
            return torch.Tensor(self.model.embed(text))

        # Batch encoding for list of texts
        if not isinstance(text, Sequence):
            msg = f"Expected str or Sequence[str], got {type(text)}"
            raise TypeError(msg)

        texts = list(text)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = [self.model.embed(t) for t in batch]
            embeddings.extend(batch_embeddings)

        return torch.stack([torch.Tensor(emb) for emb in embeddings])


if __name__ == "__main__":
    from loguru import logger

    logger.info("Testing QwenEmbeddingModel")
    model = QwenEmbeddingModel.get_instance()

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "2 + 2 = 4",
    ]

    for text in test_texts:
        logger.info(f"Encoding: {text!r}")
        embedding = model.encode(text)
        logger.info(f"  Shape: {embedding.shape}")
        logger.info(f"  Type: {type(embedding)}")
        logger.info(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
        logger.info(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
        logger.info(f"  First 5 values: {embedding[:5].tolist()}")

        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 1, f"Expected 1D tensor, got {embedding.dim()}D"
        assert embedding.shape[0] > 0, "Embedding dimension should be positive"
        assert not torch.isnan(embedding).any(), "Embedding contains NaN values"
        assert not torch.isinf(embedding).any(), "Embedding contains infinite values"

    logger.success("All tests passed!")
