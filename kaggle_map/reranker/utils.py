"""Utilities for managing GGUF quantized LLM models with llama-cpp-python."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kaggle_map.core.models import (
    GGUF_MODELS,
    LLMModelLoadConfig,
    ModelName,
    QuantizationLevel,
)


def format_chat_prompt(model_name: ModelName, user_content: str) -> str:
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
    if "gemma" in model_name.lower():
        return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    if "qwen" in model_name.lower():
        # Include empty think tags to disable thinking mode (per Qwen3 documentation)
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    if "gpt-oss" in model_name.lower():
        # gpt-oss uses a more complex format with system/developer messages
        # For simplicity, using basic user/assistant format here
        return f"<|start|>user<|message|>{user_content}<|end|><|start|>assistant"
    # Default to Gemma format for unknown models
    logger.warning(f"Unknown model type {model_name}, defaulting to Gemma chat format")
    return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"


def get_stop_tokens(model_name: ModelName) -> list[str]:
    """Get the appropriate stop tokens for a model.

    Different models use different stop tokens:
    - Gemma: ["<end_of_turn>", "\n"]
    - Qwen3: ["<|im_end|>", "\n"]
    - gpt-oss: ["<|end|>", "\n"]

    Args:
        model_name: The model being used

    Returns:
        List of stop token strings
    """
    # Dict-driven configuration for stop tokens
    stop_tokens_config = {
        "gemma": ["<end_of_turn>", "\n"],
        "qwen": ["<|im_end|>", "\n"],
        "gpt-oss": ["<|end|>", "\n"],
    }

    model_name_lower = model_name.lower()

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


def get_model_path(model_name: ModelName, quantization: QuantizationLevel) -> Path:
    """Get the local path for a GGUF model file."""
    return Path(f"models/gguf/{model_name}-{quantization}.gguf")


def download_model(model_name: ModelName, quantization: QuantizationLevel) -> Path:
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
    filename = config.filename_pattern.format(quant=quantization)

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


def load_llm_model(config: LLMModelLoadConfig) -> Llama:
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
