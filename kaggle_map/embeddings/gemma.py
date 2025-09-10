import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


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

    def encode(self, text: str) -> torch.Tensor:
        return self.model.encode(text)


if __name__ == "__main__":
    from loguru import logger

    logger.info("Testing GemmaEmbeddingModel")
    model = GemmaEmbeddingModel.get_instance()

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
