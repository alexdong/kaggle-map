"""Utilities for managing GGUF quantized LLM models with llama-cpp-python.

https://huggingface.co/unsloth/gemma-3-27b-it-GGUF
https://huggingface.co/unsloth/gpt-oss-20b-GGUF
https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.logger_config import configure_logger

# Configure module-specific logging
configure_logger(__name__)

# LLM operation type aliases
PromptTemplate = str
LLMResponse = str


@dataclass
class ParseResult:
    """Result from parsing LLM response."""

    predictions: list  # Will be list[Prediction] but avoiding circular import
    thinking_trace: str | None = None


class GGUFModelName(str, Enum):
    """Available GGUF model options."""

    QWEN3_14B = "Qwen3-14B"
    QWEN3_30B = "Qwen3-30B-A3B-Instruct-2507"
    QWEN3_30B_Thinking = "Qwen3-30B-A3B-Thinking-2507"
    GEMMA_3_12B_IT = "gemma-3-12b-it"
    GEMMA_3_27B_IT = "gemma-3-27b-it"
    GPT_OSS_20B = "gpt-oss-20b"


class GGUFModelQuantizationLevel(str, Enum):
    """Available quantization levels for GGUF models.

    NOTE: Q4_K_XL and Q5_K_XL have sequential loading conflicts in llama-cpp-python
    Use only one quantization per benchmark session to avoid GPU context corruption
    """

    Q2_K_L = "Q2_K_L"
    Q2_K_XL = "Q2_K_XL"
    Q3_K_M = "Q3_K_M"
    Q3_K_XL = "Q3_K_XL"
    Q4_K_M = "Q4_K_M"
    Q4_K_XL = "Q4_K_XL"
    Q5_K_M = "Q5_K_M"
    Q5_K_XL = "Q5_K_XL"
    Q6_K_XL = "Q6_K_XL"


# Available options derived from enum members
MODEL_OPTIONS: list[GGUFModelName] = list(GGUFModelName.__members__.values())
QUANTIZATION_OPTIONS: list[GGUFModelQuantizationLevel] = list(GGUFModelQuantizationLevel.__members__.values())


@dataclass(frozen=True)
class GGUFRepoSpec:
    """Specification for a GGUF model repository and filename pattern."""

    repo: str  # HuggingFace repository ID
    filename_pattern: str  # Pattern with {quant} placeholder for quantization level
    available_quantizations: list[GGUFModelQuantizationLevel] = field(
        default_factory=lambda: list(GGUFModelQuantizationLevel.__members__.values())
    )


# Model configurations with their HuggingFace patterns
GGUF_MODELS: dict[GGUFModelName, GGUFRepoSpec] = {
    GGUFModelName.QWEN3_14B: GGUFRepoSpec(
        repo="unsloth/Qwen3-14B-GGUF",
        filename_pattern="Qwen3-14B-{quant}.gguf",
        # Temporarily test only Q5_K_XL due to sequential loading conflicts
    ),
    GGUFModelName.QWEN3_30B: GGUFRepoSpec(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename_pattern="Qwen3-30B-A3B-Instruct-2507-UD-{quant}.gguf",
        available_quantizations=[GGUFModelQuantizationLevel.Q2_K_XL],
    ),
    GGUFModelName.QWEN3_30B_Thinking: GGUFRepoSpec(
        repo="unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
        filename_pattern="Qwen3-30B-A3B-Thinking-2507-UD-{quant}.gguf",
        available_quantizations=[GGUFModelQuantizationLevel.Q2_K_XL],
    ),
    GGUFModelName.GEMMA_3_27B_IT: GGUFRepoSpec(
        repo="unsloth/gemma-3-27b-it-GGUF",
        filename_pattern="gemma-3-27b-it-UD-{quant}.gguf",
        available_quantizations=[
            GGUFModelQuantizationLevel.Q2_K_XL,
            GGUFModelQuantizationLevel.Q3_K_XL,
        ],
    ),
    GGUFModelName.GPT_OSS_20B: GGUFRepoSpec(
        repo="unsloth/gpt-oss-20b-GGUF",
        filename_pattern="gpt-oss-20b-{quant}.gguf",
        available_quantizations=[
            GGUFModelQuantizationLevel.Q2_K_L,
            GGUFModelQuantizationLevel.Q3_K_M,
            GGUFModelQuantizationLevel.Q4_K_M,
            GGUFModelQuantizationLevel.Q4_K_XL,
            GGUFModelQuantizationLevel.Q5_K_M,
        ],
    ),
}


@dataclass
class GGUFModelLoadConfig:
    """Configuration for loading GGUF models into memory.

    gpt-oss-20b: doesn't follow instruction tuning well.
    Qwen3-14B: Q4: 0.6005; Q6: 0.6021
    gemma-3-12b-it: Q4: 0.6185; Q6: 0.6193

    The Q4 is slightly worse but much much faster, so it's a good trade-off.
    Further, gemma-3 is smaller but slightly better than Qwen3, so it's a good choice
    """

    model_name: GGUFModelName = GGUFModelName.QWEN3_30B_Thinking
    quantization: GGUFModelQuantizationLevel = GGUFModelQuantizationLevel.Q2_K_XL

    # Context window size calculation for 16GB VRAM:
    #
    # Memory breakdown:
    # - Total VRAM: 16GB
    # - Desktop/OS overhead: ~0.7GB
    # - Available for model: ~15.3GB
    #
    # Model memory requirements:
    # - Q2_K_L quantization of 20B model: ~11.8GB (loaded weights)
    # - Q3_K_M quantization of 20B model: ~13.5GB
    # - Q4_K_M quantization of 20B model: ~15GB
    # - Q5_K_M quantization of 20B model: ~17GB (won't fit)
    #
    # KV cache memory per token (for 20B models):
    # - Estimated architecture: 32-40 layers, 32 KV heads, 128 head dim
    # - KV cache per token: ~0.5-1MB (depends on precision)
    # - Conservative estimate: 1MB per 1k tokens
    #
    # Calculation for Q2_K_L (our default):
    # - Model weights: 11.8GB
    # - Remaining for KV cache + overhead: 15.3 - 11.8 = 3.5GB
    # - Reserve 1GB for processing overhead
    # - Available for KV cache: 2.5GB
    # - Max tokens: 2.5GB / 1MB per 1k = ~2,500k tokens
    # - Safe tokens with headroom: 20k (4096 * 5)
    #
    # Multiplier guide by quantization:
    # - Q2_K_L: 4096 * 5 (20k tokens) - current setting
    # - Q3_K_M: 4096 * 3 (12k tokens) - less room due to larger model
    # - Q4_K_M: 4096 * 2 (8k tokens) - minimal context, model barely fits
    # - Q5_K_M: Won't fit in 16GB VRAM
    n_ctx: int = 4096 * 5  # 20,480 tokens - optimal for Q2_K_L on 16GB VRAM
    n_batch: int = 512  # Batch size for prompt processing
    n_gpu_layers: int = -1  # Use all available GPU layers
    n_threads: int = 8  # CPU threads for inference
    random_seed: int = 42
    verbose: bool = False  # Verbose llama.cpp output

    @property
    def model_filename(self) -> str:
        """Get the GGUF filename for this configuration."""
        return f"{self.model_name.value}-{self.quantization.value}.gguf"


def suggest_ctx_length(
    vram_gb: float,
    model_name: GGUFModelName,
    quantization: GGUFModelQuantizationLevel,
    desktop_overhead_gb: float = 0.7,
    safety_margin_gb: float = 1.0,
) -> int:
    """Calculate optimal context length based on available VRAM and model configuration.

    Args:
        vram_gb: Total VRAM in gigabytes (e.g., 16 for 16GB)
        model_name: The GGUF model being used
        quantization: Quantization level of the model
        desktop_overhead_gb: Memory reserved for desktop/OS (default 0.7GB)
        safety_margin_gb: Additional safety margin to prevent OOM (default 1.0GB)

    Returns:
        Optimal n_ctx value (multiple of 4096) that fits in available VRAM

    The calculation follows this formula:
    1. Available VRAM = Total - Desktop overhead
    2. Model weights size based on quantization
    3. Remaining for KV cache = Available - Model weights - Safety margin
    4. Context tokens = KV cache memory / Memory per token
    5. Round down to nearest 4096 multiple
    """
    # Model weight sizes in GB (approximate, based on empirical measurements)
    model_weight_sizes = {
        # 20B parameter models
        (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q2_K_L): 11.8,
        (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q3_K_M): 13.5,
        (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q4_K_M): 15.0,
        (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q5_K_M): 17.0,
        # 27B parameter models (Gemma)
        (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q2_K_XL): 14.5,
        (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q3_K_XL): 16.5,
        # 30B parameter models (Qwen)
        (GGUFModelName.QWEN3_30B, GGUFModelQuantizationLevel.Q2_K_XL): 15.0,
        (GGUFModelName.QWEN3_30B_Thinking, GGUFModelQuantizationLevel.Q2_K_XL): 15.0,
        # 14B parameter models
        (GGUFModelName.QWEN3_14B, GGUFModelQuantizationLevel.Q4_K_XL): 9.5,
        (GGUFModelName.QWEN3_14B, GGUFModelQuantizationLevel.Q6_K_XL): 11.5,
    }

    # KV cache memory per 1k tokens in GB (conservative estimate)
    # For 20B models: ~32-40 layers * 32 heads * 128 dim * 2 (K+V) * 2 bytes = ~1MB per token
    # So 1k tokens = ~120MB of KV cache (empirically tuned for common VRAM sizes)
    kv_cache_per_1k_tokens = {
        GGUFModelName.GPT_OSS_20B: 0.12,  # ~120MB per 1k tokens (tuned for 16GB -> 5*4096)
        GGUFModelName.GEMMA_3_27B_IT: 0.15,  # ~150MB per 1k tokens
        GGUFModelName.QWEN3_30B: 0.16,  # ~160MB per 1k tokens
        GGUFModelName.QWEN3_30B_Thinking: 0.16,  # ~160MB per 1k tokens
        GGUFModelName.QWEN3_14B: 0.085,  # ~85MB per 1k tokens (tuned for more context)
    }

    # Get model weight size
    model_key = (model_name, quantization)
    if model_key not in model_weight_sizes:
        # Default conservative estimate based on quantization alone
        quant_multipliers = {
            GGUFModelQuantizationLevel.Q2_K_L: 0.6,
            GGUFModelQuantizationLevel.Q2_K_XL: 0.6,
            GGUFModelQuantizationLevel.Q3_K_M: 0.7,
            GGUFModelQuantizationLevel.Q3_K_XL: 0.7,
            GGUFModelQuantizationLevel.Q4_K_M: 0.8,
            GGUFModelQuantizationLevel.Q4_K_XL: 0.8,
            GGUFModelQuantizationLevel.Q5_K_M: 0.9,
            GGUFModelQuantizationLevel.Q6_K_XL: 1.0,
        }
        # Rough estimate: 20B model at Q4 is about 15GB
        base_size = 15.0
        multiplier = quant_multipliers.get(quantization, 0.8)
        model_weight_gb = base_size * multiplier
    else:
        model_weight_gb = model_weight_sizes[model_key]

    # Calculate available memory for KV cache
    available_vram = vram_gb - desktop_overhead_gb
    kv_cache_budget = available_vram - model_weight_gb - safety_margin_gb

    # If model doesn't fit at all, return 0
    if kv_cache_budget <= 0:
        return 0

    # Get KV cache requirement per token
    kv_per_1k = kv_cache_per_1k_tokens.get(model_name, 0.125)  # Default to 125MB per 1k tokens

    # Calculate maximum context tokens
    # kv_cache_budget is in GB, kv_per_1k is GB per 1000 tokens
    # So tokens = (budget_GB / kv_per_1k_GB) * 1000
    max_tokens = int((kv_cache_budget / kv_per_1k) * 1000)

    # Round down to nearest multiple of 4096
    # This ensures clean alignment with typical context window sizes
    n_ctx = (max_tokens // 4096) * 4096

    # Ensure minimum context of 4096 if any tokens fit
    min_context_size = 4096
    min_token_threshold = 2048
    if max_tokens >= min_token_threshold and n_ctx < min_context_size:
        n_ctx = min_context_size

    return n_ctx


def format_chat_prompt(model_name: GGUFModelName, user_content: str) -> str:
    """Format chat prompt according to the model's expected template.

    Different models use different chat template formats:
    - Gemma: <start_of_turn>user ... <end_of_turn><start_of_turn>model
    - Qwen3: <|im_start|>user ... <|im_end|><|im_start|>assistant
    - gpt-oss: <|start|>user ... <|end|><|start|>assistant

    Args:
        model_name: The model being used
        user_content: The user's message content

    Returns:
        Formatted prompt string with appropriate chat markers
    """
    if "gemma" in model_name.value.lower():
        return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    if "qwen" in model_name.value.lower():
        # Don't include empty think tags - let the model think if it wants to
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    if "gpt-oss" in model_name.value.lower():
        # gpt-oss uses a more complex format with system/developer messages
        # For simplicity, using basic user/assistant format here
        return f"<|start|>user<|message|>{user_content}<|end|><|start|>assistant"
    # Default to Gemma format for unknown models
    logger.warning(f"Unknown model type {model_name}, defaulting to Gemma chat format")
    return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"


def get_stop_tokens(model_name: GGUFModelName) -> list[str]:
    # Dict-driven configuration for stop tokens
    stop_tokens_config = {
        "gemma": ["<end_of_turn>", "\n"],
        "qwen": ["<|im_end|>"],  # Don't stop on newline for thinking model
        "gpt-oss": ["<|end|>"],  # GPT-OSS uses Harmony format, don't stop on \n
    }

    model_name_lower = model_name.value.lower()

    # Find matching model family
    for model_family, tokens in stop_tokens_config.items():
        if model_family in model_name_lower:
            return tokens

    # If we reach here, model is unknown - use assert to fail early
    supported_families = ", ".join(stop_tokens_config.keys())
    msg = (
        f"Unknown model type '{model_name}'. Model name must contain one of: {supported_families}. "
        f"This is a programming error - the model type should be validated before calling get_stop_tokens."
    )
    raise AssertionError(msg)


def get_model_path(model_name: GGUFModelName, quantization: GGUFModelQuantizationLevel) -> Path:
    """Get the local path for a GGUF model file.

    All XL quantizations from Unsloth use "UD-" prefix in their download URLs,
    but we store them locally WITHOUT the "UD-" prefix for consistency.
    This stands for "Unsloth Dynamic" (Unsloth's Dynamic 2.0 quantization).
    """
    # Always use the same local path format (without UD- prefix)
    return Path(f"models/gguf/{model_name.value}-{quantization.value}.gguf")


def download_model(model_name: GGUFModelName, quantization: GGUFModelQuantizationLevel) -> Path:
    """Download GGUF model from Hugging Face Hub if it doesn't exist."""
    model_path = get_model_path(model_name, quantization)

    if model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        return model_path

    logger.info(f"Model not found locally: {model_path}")

    # Get model configuration
    config = GGUF_MODELS.get(model_name)
    assert config, f"Unknown model type: {model_name}"

    # Assert that the quantization is available for this model (caller's responsibility)
    error_msg = (
        f"Quantization '{quantization}' is not available for model '{model_name}'. "
        f"Available quantizations: {', '.join(config.available_quantizations)}. "
        f"It's the caller's responsibility to check availability before calling download_model."
    )
    assert quantization in config.available_quantizations, error_msg

    repo_id = config.repo
    filename = config.filename_pattern.format(quant=quantization.value)

    logger.info(f"Downloading {filename} from {repo_id}")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Download model
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_path.parent,
        local_dir_use_symlinks=False,  # Copy file instead of symlink
    )

    # Handle filename mismatch between HuggingFace repo and our local naming convention
    # HF repos often have different naming patterns (e.g., "model-UD-Q4_K_XL.gguf")
    # but we want consistent local names (e.g., "model-Q4_K_XL.gguf")
    # This ensures models are stored with predictable names regardless of source
    downloaded_file = Path(downloaded_path)
    if downloaded_file != model_path and downloaded_file.exists():
        downloaded_file.rename(model_path)

    assert model_path.exists(), f"Model file not found after download: {model_path}"
    logger.info(f"Model downloaded successfully: {model_path}")
    return model_path


def load_llm_model(config: GGUFModelLoadConfig) -> Llama:
    """Load a GGUF model with automatic cleanup via context manager."""
    model_path = download_model(config.model_name, config.quantization)
    logger.info(f"Loading GGUF model from {model_path}")
    assert model_path.exists(), f"Model file not found after download: {model_path}"

    return Llama(
        model_path=str(model_path),
        n_ctx=config.n_ctx,
        n_batch=config.n_batch,
        n_gpu_layers=config.n_gpu_layers,
        verbose=config.verbose,
        n_threads=config.n_threads,
    )


def _extract_ranking_from_analysis(thinking_trace: str) -> str | None:
    """Extract label ranking from analysis text."""
    # Pattern 1: Space-separated labels like "False_Neither:NA False_Correct:NA ..."
    ranking_pattern = r'"?([A-Za-z_]+:[A-Za-z_]+(?:\s+[A-Za-z_]+:[A-Za-z_]+)+)"?'
    rankings = re.findall(ranking_pattern, thinking_trace)

    # Pattern 2: Labels separated by > symbols like "False_Neither:NA > False_Correct:NA > ..."
    order_pattern = r"([A-Za-z_]+:[A-Za-z_]+(?:\s*>\s*[A-Za-z_]+:[A-Za-z_]+)+)"
    order_rankings = re.findall(order_pattern, thinking_trace)

    if rankings:
        # Use the last space-separated ranking found
        result = rankings[-1]
        logger.debug(f"Extracted space-separated ranking from analysis: {result}")
        return result
    if order_rankings:
        # Convert > separated to space-separated
        result = order_rankings[-1].replace(" > ", " ").replace(">", " ")
        logger.debug(f"Extracted order ranking from analysis: {result}")
        return result
    return None


def _parse_harmony_format(response: str) -> tuple[str | None, str]:
    """Parse GPT-OSS Harmony format response.

    Args:
        response: Raw response with Harmony format tags

    Returns:
        Tuple of (thinking_trace, clean_response_for_predictions)
    """
    thinking_trace = None
    clean_response = response

    # Extract analysis channel content as thinking trace
    analysis_pattern = r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|channel\|>|<\|end\|>|$)"
    analysis_match = re.search(analysis_pattern, response, re.DOTALL)
    if analysis_match:
        thinking_trace = analysis_match.group(1).strip()
        logger.debug(f"Extracted analysis channel: {len(thinking_trace)} chars")

    # Extract final channel content for predictions
    final_pattern = r"<\|channel\|>final<\|message\|>(.*?)(?=<\|channel\|>|<\|end\|>|$)"
    final_match = re.search(final_pattern, response, re.DOTALL)
    if final_match:
        clean_response = final_match.group(1).strip()
        logger.debug(f"Extracted final channel: {clean_response[:200]}")
    else:
        logger.debug("No final channel found, attempting to extract from analysis")
        # If no final channel, try to extract predictions from the analysis section
        if thinking_trace:
            extracted_ranking = _extract_ranking_from_analysis(thinking_trace)
            if extracted_ranking:
                clean_response = extracted_ranking
            else:
                # If still no rankings, remove analysis and use what's left
                clean_response = re.sub(analysis_pattern, "", response, flags=re.DOTALL).strip()
                logger.debug(f"No rankings found, using cleaned response: {clean_response[:200]}")
        else:
            # No analysis channel either, use cleaned response
            clean_response = re.sub(analysis_pattern, "", response, flags=re.DOTALL).strip()
            logger.debug(f"No analysis channel, using cleaned response: {clean_response[:200]}")

    return thinking_trace, clean_response


def _parse_think_tags(response: str) -> tuple[str | None, str]:
    """Parse standard <think>...</think> format response.

    Args:
        response: Raw response with think tags

    Returns:
        Tuple of (thinking_trace, clean_response_for_predictions)
    """
    thinking_pattern = r"<think>(.*?)</think>"
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)

    thinking_trace = None
    if thinking_match:
        thinking_trace = thinking_match.group(1).strip()

    # Remove thinking tags from response for prediction parsing
    clean_response = re.sub(thinking_pattern, "", response, flags=re.DOTALL)

    return thinking_trace, clean_response


def parse_llm_response(response: str, parse_predictions_fn: Callable[[str], list] | None = None) -> ParseResult:
    """Parse LLM response to extract predictions and thinking trace.

    Handles both standard <think>...</think> tags and GPT-OSS Harmony format
    with <|channel|>analysis<|message|> and <|channel|>final<|message|> tags.

    Args:
        response: Raw LLM response containing predictions and possibly thinking tags
        parse_predictions_fn: Function to parse predictions from cleaned response.
                             If None, returns empty list.

    Returns:
        ParseResult with predictions and optional thinking trace
    """
    # Check for GPT-OSS Harmony format first
    if "<|channel|>" in response:
        thinking_trace, clean_response = _parse_harmony_format(response)
    else:
        # Standard <think>...</think> format
        thinking_trace, clean_response = _parse_think_tags(response)

    # Parse predictions from cleaned response
    predictions = parse_predictions_fn(clean_response) if parse_predictions_fn else []

    return ParseResult(predictions=predictions, thinking_trace=thinking_trace)
