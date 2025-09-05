"""Embedding model using Qwen3-Embedding-8B GGUF with quantization options.

This module provides support for the Qwen3-Embedding-8B model in GGUF format
with various quantization levels for efficient embeddings generation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.device import get_device


@dataclass(frozen=True)
class QuantizationSpec:
    """Specification for a quantization level."""
    filename: str
    level: str
    size_gb: float
    notes: str = ""


class QuantizationLevel(Enum):
    """Available quantization levels for Qwen3-Embedding-8B."""
    F16 = "F16"      # Full precision
    Q8_0 = "Q8_0"    # 8-bit quantization
    Q6_K = "Q6_K"    # 6-bit quantization
    Q5_K_M = "Q5_K_M"  # 5-bit quantization (medium)
    Q4_K_M = "Q4_K_M"  # 4-bit quantization (medium)

    @property
    def spec(self) -> QuantizationSpec:
        specs = {
            QuantizationLevel.F16: QuantizationSpec(
                filename="Qwen3-Embedding-8B-F16.gguf",
                level="F16",
                size_gb=15.1,
                notes="Full precision, highest quality but slowest and largest"
            ),
            QuantizationLevel.Q8_0: QuantizationSpec(
                filename="Qwen3-Embedding-8B-Q8_0.gguf",
                level="Q8_0",
                size_gb=8.6,
                notes="8-bit quantization, excellent quality/speed balance (recommended)"
            ),
            QuantizationLevel.Q6_K: QuantizationSpec(
                filename="Qwen3-Embedding-8B-Q6_K.gguf",
                level="Q6_K",
                size_gb=6.9,
                notes="6-bit quantization, good quality with better speed"
            ),
            QuantizationLevel.Q5_K_M: QuantizationSpec(
                filename="Qwen3-Embedding-8B-Q5_K_M.gguf",
                level="Q5_K_M",
                size_gb=6.16,
                notes="5-bit quantization, balanced quality/speed"
            ),
            QuantizationLevel.Q4_K_M: QuantizationSpec(
                filename="Qwen3-Embedding-8B-Q4_K_M.gguf",
                level="Q4_K_M",
                size_gb=5.41,
                notes="4-bit quantization, fastest but lower quality"
            ),
        }
        return specs[self]

    @property
    def filename(self) -> str:
        return self.spec.filename


class QwenEmbeddingModel:
    """Qwen3-Embedding-8B model wrapper for generating embeddings."""

    MODEL_REPO = "JonathanMiddleton/Qwen3-Embedding-8B-GGUF"
    EMBEDDING_DIM = 5120  # Qwen3-8B embedding dimension

    def __init__(
        self,
        quantization: QuantizationLevel = QuantizationLevel.Q8_0,
        model_path: Path | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 means all layers on GPU if available
        verbose: bool = False,
    ) -> None:
        """Initialize the Qwen embedding model.

        Args:
            quantization: Quantization level to use
            model_path: Optional path to pre-downloaded model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Whether to show llama.cpp output
        """
        self.quantization = quantization
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

        logger.info(f"Loading Qwen3-Embedding-8B with {quantization.value} quantization from {model_path}")

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

        logger.info(f"Downloading {self.quantization.filename} from Hugging Face...")

        # Download the model file
        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.quantization.filename,
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
            Embeddings array of shape (n_texts, embedding_dim) or (embedding_dim,)
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


# Backward compatibility - keep EmbeddingModel enum for existing code
class EmbeddingModel(Enum):
    """Legacy enum for backward compatibility."""
    QWEN3_8B = "qwen3-8b"

    @property
    def model_id(self) -> str:
        return "JonathanMiddleton/Qwen3-Embedding-8B-GGUF"

    @property
    def base_dim(self) -> int:
        """Base dimension of the embedding model."""
        return QwenEmbeddingModel.EMBEDDING_DIM

    @property
    def dim(self) -> int:
        """For compatibility - returns base dim (no concatenation for single model)."""
        return self.base_dim

    @property
    def recommended_max_seq(self) -> int:
        return 2048


def get_tokenizer(
    model: EmbeddingModel | None = None,
    quantization: QuantizationLevel = QuantizationLevel.Q8_0,
    device: str | None = None,
    verbose: bool = False,
) -> QwenEmbeddingModel:
    """Get the Qwen embedding model with specified quantization.

    Args:
        model: Legacy parameter for compatibility (ignored)
        quantization: Quantization level to use (default: Q8_0 for best quality/speed)
        device: Device specification (auto-detected if None)
        verbose: Whether to show llama.cpp output

    Returns:
        QwenEmbeddingModel instance
    """
    if model is not None:
        logger.debug(f"Legacy model parameter {model} ignored, using Qwen3-8B")

    return QwenEmbeddingModel(
        quantization=quantization,
        verbose=verbose,
    )

