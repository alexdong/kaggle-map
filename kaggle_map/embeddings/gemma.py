import time
from collections.abc import Sequence

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


class GemmaEmbeddingModel:
    _instance: "GemmaEmbeddingModel | None" = None

    def __init__(self) -> None:
        logger.info("Loading EmbeddingGemma-300M")
        self.model = SentenceTransformer("google/embeddinggemma-300m")

    @classmethod
    def get_instance(cls) -> "GemmaEmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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
            # Single text - keep backward compatibility
            return self.model.encode(text, convert_to_tensor=True)

        if not isinstance(text, Sequence):  # Defensive: should never happen with typing
            msg = f"Expected str or Sequence[str], got {type(text)}"
            raise TypeError(msg)

        texts = list(text)
        assert texts, "Cannot encode empty sequence of texts"
        assert all(isinstance(t, str) for t in texts), "All items in sequence must be strings"

        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)


if __name__ == "__main__":
    logger.info("Testing GemmaEmbeddingModel")

    EMBEDDING_DIMENSION = 768
    SINGLE_VECTOR_DIMENSIONS = 1
    BATCH_TENSOR_DIMENSIONS = 2
    SMALL_BATCH_SIZE = 3
    CONSISTENCY_TOLERANCE = 1e-5
    LARGE_BATCH_MULTIPLIER = 100

    model = GemmaEmbeddingModel.get_instance()
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "2 + 2 = 4",
    ]

    # Test individual encoding (backward compatibility)
    logger.info("\n=== Testing individual encoding ===")
    for text in test_texts:
        logger.info(f"Encoding: {text!r}")
        start = time.time()
        embedding = model.encode(text)
        elapsed = time.time() - start
        logger.info(f"  Time: {elapsed:.3f}s")
        logger.info(f"  Shape: {embedding.shape}")
        logger.info(f"  Type: {type(embedding)}")
        logger.info(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
        logger.info(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
        logger.info(f"  First 5 values: {embedding[:5].tolist()}")

        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == SINGLE_VECTOR_DIMENSIONS, f"Expected 1D tensor, got {embedding.dim()}D"
        assert embedding.shape[0] == EMBEDDING_DIMENSION, (
            f"Expected {EMBEDDING_DIMENSION} dims, got {embedding.shape[0]}"
        )
        assert not torch.isnan(embedding).any(), "Embedding contains NaN values"
        assert not torch.isinf(embedding).any(), "Embedding contains infinite values"

    # Test batch encoding
    logger.info("\n=== Testing batch encoding ===")
    start = time.time()
    batch_embeddings = model.encode(test_texts)
    elapsed = time.time() - start
    logger.info(f"Batch encoded {len(test_texts)} texts in {elapsed:.3f}s")
    logger.info(f"  Shape: {batch_embeddings.shape}")
    logger.info(f"  Type: {type(batch_embeddings)}")

    assert isinstance(batch_embeddings, torch.Tensor)
    assert batch_embeddings.dim() == BATCH_TENSOR_DIMENSIONS, f"Expected 2D tensor, got {batch_embeddings.dim()}D"
    expected_batch_shape = (SMALL_BATCH_SIZE, EMBEDDING_DIMENSION)
    assert batch_embeddings.shape == expected_batch_shape, (
        f"Expected {expected_batch_shape}, got {batch_embeddings.shape}"
    )
    assert not torch.isnan(batch_embeddings).any(), "Batch embeddings contain NaN values"
    assert not torch.isinf(batch_embeddings).any(), "Batch embeddings contain infinite values"

    # Verify batch results match individual results
    logger.info("\n=== Verifying batch vs individual consistency ===")
    for i, text in enumerate(test_texts):
        individual = model.encode(text)
        batch_row = batch_embeddings[i]
        diff = torch.abs(individual - batch_row).max()
        logger.info(f"Text {i}: max difference = {diff:.6f}")
        assert diff < CONSISTENCY_TOLERANCE, f"Batch and individual encodings differ by {diff}"

    # Test larger batch
    logger.info("\n=== Testing larger batch ===")
    large_batch = test_texts * LARGE_BATCH_MULTIPLIER
    start = time.time()
    large_embeddings = model.encode(large_batch, batch_size=64)
    elapsed = time.time() - start
    logger.info(f"Encoded {len(large_batch)} texts in {elapsed:.3f}s ({len(large_batch) / elapsed:.1f} texts/sec)")
    expected_large_batch_shape = (len(test_texts) * LARGE_BATCH_MULTIPLIER, EMBEDDING_DIMENSION)
    assert large_embeddings.shape == expected_large_batch_shape

    logger.success("All tests passed!")
