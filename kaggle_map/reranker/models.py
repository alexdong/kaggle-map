"""Data models and configurations for LLM reranking functionality.

This module contains all GGUF model specifications and configurations
for the reranker module, keeping them separate from core domain models.
"""

from dataclasses import dataclass, field
from enum import Enum

import pydash

# LLM operation type aliases
PromptTemplate = str
LLMResponse = str


class RerankerModelName(str, Enum):
    """Available reranker model options."""

    QWEN3_14B = "Qwen3-14B"
    GEMMA_3_12B_IT = "gemma-3-12b-it"
    GPT_OSS_20B = "gpt-oss-20b"


class RerankerModelQuantizationLevel(str, Enum):
    """Available quantization levels for GGUF models.

    NOTE: Q4_K_XL and Q5_K_XL have sequential loading conflicts in llama-cpp-python
    Use only one quantization per benchmark session to avoid GPU context corruption
    """

    Q2_K_XL = "Q2_K_XL"
    Q3_K_XL = "Q3_K_XL"
    Q4_K_XL = "Q4_K_XL"
    Q5_K_XL = "Q5_K_XL"
    Q6_K_XL = "Q6_K_XL"


# Available options derived from enum members
MODEL_OPTIONS: list[RerankerModelName] = list(RerankerModelName.__members__.values())
QUANTIZATION_OPTIONS: list[RerankerModelQuantizationLevel] = list(RerankerModelQuantizationLevel.__members__.values())


@dataclass(frozen=True)
class GGUFRepoSpec:
    """Specification for a GGUF model repository and filename pattern."""

    repo: str  # HuggingFace repository ID
    filename_pattern: str  # Pattern with {quant} placeholder for quantization level
    available_quantizations: list[RerankerModelQuantizationLevel] = field(
        default_factory=lambda: list(RerankerModelQuantizationLevel.__members__.values())
    )


# Model configurations with their HuggingFace patterns
GGUF_MODELS: dict[RerankerModelName, GGUFRepoSpec] = {
    RerankerModelName.GPT_OSS_20B: GGUFRepoSpec(
        repo="unsloth/gpt-oss-20b-GGUF",
        filename_pattern="gpt-oss-20b-{quant}.gguf",
        available_quantizations=pydash.without(
            list(RerankerModelQuantizationLevel.__members__.values()), RerankerModelQuantizationLevel.Q5_K_XL
        ),
    ),
    RerankerModelName.QWEN3_14B: GGUFRepoSpec(
        repo="unsloth/Qwen3-14B-GGUF",
        filename_pattern="Qwen3-14B-{quant}.gguf",
        # Temporarily test only Q5_K_XL due to sequential loading conflicts
    ),
    RerankerModelName.GEMMA_3_12B_IT: GGUFRepoSpec(
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

    model_name: RerankerModelName = RerankerModelName.GEMMA_3_12B_IT
    quantization: RerankerModelQuantizationLevel = RerankerModelQuantizationLevel.Q4_K_XL
    n_ctx: int = 4096  # Context window size
    n_batch: int = 512  # Batch size for prompt processing
    n_gpu_layers: int = -1  # Use all available GPU layers
    n_threads: int = 8  # CPU threads for inference
    random_seed: int = 42
    verbose: bool = False  # Verbose llama.cpp output

    @property
    def model_filename(self) -> str:
        """Get the GGUF filename for this configuration."""
        return f"{self.model_name.value}-{self.quantization.value}.gguf"
