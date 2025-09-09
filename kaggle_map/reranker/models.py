"""Data models and configurations for LLM reranking functionality.

This module contains all GGUF model specifications and configurations
for the reranker module, keeping them separate from core domain models.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple, get_args

import pydash

# LLM operation type aliases
PromptTemplate = str
LLMResponse = str
RerankerModelName = Literal["Qwen3-14B", "gemma-3-12b-it", "gpt-oss-20b"]
# NOTE: Q4_K_XL and Q5_K_XL have sequential loading conflicts in llama-cpp-python
# Use only one quantization per benchmark session to avoid GPU context corruption
RerankerModelQuantizationLevel = Literal["Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]

# Available options derived from type definitions
MODEL_OPTIONS: list[RerankerModelName] = list(get_args(RerankerModelName))
QUANTIZATION_OPTIONS: list[RerankerModelQuantizationLevel] = list(get_args(RerankerModelQuantizationLevel))


class GGUFRepoSpec(NamedTuple):
    """Specification for a GGUF model repository and filename pattern."""

    repo: str  # HuggingFace repository ID
    filename_pattern: str  # Pattern with {quant} placeholder for quantization level
    available_quantizations: list[RerankerModelQuantizationLevel] = QUANTIZATION_OPTIONS


# Model configurations with their HuggingFace patterns
GGUF_MODELS: dict[RerankerModelName, GGUFRepoSpec] = {
    "gpt-oss-20b": GGUFRepoSpec(
        repo="unsloth/gpt-oss-20b-GGUF",
        filename_pattern="gpt-oss-20b-{quant}.gguf",
        available_quantizations=pydash.without(QUANTIZATION_OPTIONS, "Q5_K_XL"),
    ),
    "Qwen3-14B": GGUFRepoSpec(
        repo="unsloth/Qwen3-14B-GGUF",
        filename_pattern="Qwen3-14B-{quant}.gguf",
        # Temporarily test only Q5_K_XL due to sequential loading conflicts
    ),
    "gemma-3-12b-it": GGUFRepoSpec(
        repo="unsloth/gemma-3-12b-it-GGUF",
        filename_pattern="gemma-3-12b-it-{quant}.gguf",
    ),
}


@dataclass
class RerankerLLMLoadConfig:
    """Configuration for loading GGUF models into memory.

    gpt-oss-20b: doesn't follow instruction tuning well.
    Qwen3-14B: Q4: 0.6005; Q6: 0.6021
    gemma-3-12b-it: Q4: 0.6185; Q6: 0.6193

    The Q4 is slightly worse but much much faster, so it's a good trade-off.
    Further, gemma-3 is smaller but slightly better than Qwen3, so it's a good choice
    """

    model_name: RerankerModelName = "gemma-3-12b-it"
    quantization: RerankerModelQuantizationLevel = "Q4_K_XL"
    n_ctx: int = 4096  # Context window size
    n_batch: int = 512  # Batch size for prompt processing
    n_gpu_layers: int = -1  # Use all available GPU layers
    n_threads: int = 8  # CPU threads for inference
    random_seed: int = 42
    verbose: bool = False  # Verbose llama.cpp output

    @property
    def model_filename(self) -> str:
        """Get the GGUF filename for this configuration."""
        return f"{self.model_name}-{self.quantization}.gguf"
