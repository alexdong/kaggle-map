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
