"""Embedding model using Qwen3-Embedding-8B GGUF with Q8_0 quantization.

This module provides support for the Qwen3-Embedding-8B model in GGUF format
using Q8_0 quantization for efficient embeddings generation.
"""

from pathlib import Path

import numpy as np
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
        model_path: Path | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 means all layers on GPU if available
        verbose: bool = False,
    ) -> None:
        """Initialize the Qwen embedding model with Q8_0 quantization.

        Args:
            model_path: Optional path to pre-downloaded model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Whether to show llama.cpp output
        """
        self.n_ctx = n_ctx

        # Determine device and adjust GPU layers
        device = get_device()
        if device.type == "cpu":
            n_gpu_layers = 0
            logger.debug("CPU detected, disabling GPU acceleration")
        elif n_gpu_layers == -1:
            logger.debug(f"GPU detected ({device}), using all layers on GPU")

        # Get or download model file
        if model_path is None:
            model_path = self._get_model_path()

        logger.info(f"Loading Qwen3-Embedding-8B Q8_0 from {model_path}")

        # Initialize llama.cpp model with embedding mode
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,  # Enable embedding mode
            verbose=verbose,
            seed=-1,  # Random seed
        )

        logger.info(f"Model loaded successfully with context size {n_ctx}")

    def _get_model_path(self) -> Path:
        """Get the path to the model file, downloading if necessary."""
        from huggingface_hub import hf_hub_download

        # Use HF cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        logger.info(f"Downloading {self.MODEL_FILE} from Hugging Face...")

        # Download the model file
        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.MODEL_FILE,
            cache_dir=cache_dir,
            resume_download=True,
        )

        return Path(model_path)

    def encode(self, text: str | list[str], normalize: bool = True) -> np.ndarray:
        """Generate embeddings for text.

        Args:
            text: Single text or list of texts to encode
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Embeddings array of shape (n_texts, EMBEDDING_DIM) or (EMBEDDING_DIM,)
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        embeddings = []
        for t in texts:
            # Get embedding from llama.cpp
            result = self.model.embed(t)
            embedding = np.array(result, dtype=np.float32)

            # Truncate to max dimension if needed
            if len(embedding) > self.EMBEDDING_DIM:
                embedding = embedding[:self.EMBEDDING_DIM]

            if normalize:
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)

        # Stack embeddings
        embeddings_array = np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0].reshape(1, -1)

        # Return single embedding if single input
        if single_input:
            return embeddings_array[0]

        return embeddings_array

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM

    def to(self, device: str) -> "QwenEmbeddingModel":
        """Compatibility method for device movement (no-op for llama.cpp)."""
        logger.debug(f"Device movement to {device} requested (no-op for llama.cpp)")
        return self

