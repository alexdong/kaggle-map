"""Embedding model using Qwen3-Embedding-8B GGUF with Q8_0 quantization.

This module provides support for the Qwen3-Embedding-8B model in GGUF format
using Q8_0 quantization for efficient embeddings generation.
"""

from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.device import get_device


class QwenEmbeddingModel:
    """Qwen3-Embedding-8B model wrapper for generating embeddings."""

    MODEL_REPO = "JonathanMiddleton/Qwen3-Embedding-8B-GGUF"
    MODEL_FILE = "Qwen3-Embedding-8B-Q8_0.gguf"
    EMBEDDING_DIM = 4096  # Capped to max allowed dimension

    def __init__(
        self,
    ) -> None:
        model_path = self._get_model_path()
        logger.info(f"Loading Qwen3-Embedding-8B Q8_0 from {model_path}")

        self.model = Llama(
            n_gpu_layers=-1 if get_device().type == "cuda" else 0,
            model_path=str(model_path),
            embedding=True,
        )

    def _get_model_path(self) -> Path:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        logger.info(f"Downloading {self.MODEL_FILE} from Hugging Face...")
        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.MODEL_FILE,
            cache_dir=cache_dir,
            resume_download=True,
        )
        return Path(model_path)

    def encode(self, text: str) -> np.ndarray:
        result = self.model.embed(text)
        return np.array(result, dtype=np.float32)


if __name__ == "__main__":
    model = QwenEmbeddingModel()
    text = "hello world"
    embedding = model.encode(text)

    print(f"Text: '{text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Min value: {embedding.min():.4f}")
    print(f"Max value: {embedding.max():.4f}")
    print(f"Mean value: {embedding.mean():.4f}")
    print(f"Std value: {embedding.std():.4f}")
