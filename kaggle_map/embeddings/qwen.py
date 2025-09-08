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

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.device import get_device


class QwenEmbeddingModel:
    _instance: "QwenEmbeddingModel | None" = None

    def __init__(self) -> None:
        model_path = self._get_model_path()
        logger.info(f"Loading Qwen3-Embedding-8B Q8_0 from {model_path}")

        self.model = Llama(
            n_gpu_layers=-1 if get_device().type == "cuda" else 0,
            model_path=str(model_path),
            embedding=True,
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

    def encode(self, text: str) -> torch.Tensor:
        return torch.Tensor(self.model.embed(text))
