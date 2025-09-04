"""Utilities for managing GGUF quantized LLM models with llama-cpp-python."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.core.models import (
    GGUF_MODELS,
    GGUFRepoSpec,
    MODEL_OPTIONS,
    QUANTIZATION_OPTIONS,
    InferenceConfig,
    ModelLoadConfig,
    ModelName,
    QuantizationLevel,
)


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

    # Move to expected location if needed
    if downloaded_path != str(model_path):
        Path(downloaded_path).rename(model_path)

    logger.info(f"Model downloaded successfully: {model_path}")
    return model_path


@contextmanager
def load_llm_model(config: ModelLoadConfig) -> Iterator[Llama]:
    """Load a GGUF model with llama-cpp-python as a context manager, downloading if necessary."""
    model_path = download_model(config.model_name, config.quantization)
    logger.info(f"Loading GGUF model from {model_path}")
    assert model_path.exists(), f"Model file not found after download: {model_path}"

    llm = Llama(
        model_path=str(model_path),
        n_ctx=config.n_ctx,
        n_batch=config.n_batch,
        n_gpu_layers=config.n_gpu_layers,  # Use all GPU layers (Metal on Mac, CUDA on GPU)
        verbose=config.verbose,
        n_threads=config.n_threads,
    )
    logger.info(f"Model loaded successfully: {model_path.name}")

    try:
        yield llm
    finally:
        # Cleanup happens automatically when exiting the context
        del llm
        logger.info(f"Model cleanup completed: {model_path.name}")


if __name__ == "__main__":
    console = Console()

    console.print("🚀 LLM Model Benchmarking Tool", style="bold cyan")
    console.print("=" * 50)

    # Test question
    test_question = "Who is the Bosch in the Haber-Bosch process?"

    # Results storage
    results = []

    # Download and benchmark all model variants
    for model_name in MODEL_OPTIONS:
        for quantization in QUANTIZATION_OPTIONS:
            console.print(f"\n📦 Processing {model_name} - {quantization}", style="bold yellow")
            console.print("-" * 40)

            # Create model loading config
            load_config = ModelLoadConfig(
                model_name=model_name,
                quantization=quantization,
                n_ctx=2048,  # Smaller context for benchmarking
                verbose=False,
            )

            # Create inference config for benchmarking
            inference_config = InferenceConfig(
                max_tokens=100,
                temperature=0.1,
                echo=False,
            )

            # Download model
            model_path = download_model(load_config.model_name, load_config.quantization)
            console.print(f"✅ Model ready: {model_path.name}", style="green")

            # Load model with context manager
            start_load = time.time()
            with load_llm_model(load_config) as llm:
                load_time = time.time() - start_load

                # Benchmark inference
                console.print(f"🧪 Testing with: '{test_question}'")

                start_inference = time.time()
                output = llm(
                    test_question,
                    max_tokens=inference_config.max_tokens,
                    temperature=inference_config.temperature,
                    echo=inference_config.echo,
                )
                total_inference_time = time.time() - start_inference

                # Extract response and calculate metrics
                response = output["choices"][0]["text"].strip()  # type: ignore
                tokens_generated = len(response.split())  # Rough token count

                # Calculate time to first token (approximation)
                time_to_first_token = (
                    total_inference_time / tokens_generated if tokens_generated > 0 else total_inference_time
                )
                tokens_per_sec = tokens_generated / total_inference_time if total_inference_time > 0 else 0

                # Store results
                results.append(
                    {
                        "Model": f"{model_name}",
                        "Quantization": quantization,
                        "Load Time (s)": f"{load_time:.2f}",
                        "Time to 1st Token (s)": f"{time_to_first_token:.3f}",
                        "Tokens/sec": f"{tokens_per_sec:.1f}",
                        "Response Preview": response[:50] + "..." if len(response) > 50 else response,
                    }
                )

                console.print(
                    f"⚡ Performance: {tokens_per_sec:.1f} tok/s, First token: {time_to_first_token:.3f}s", style="blue"
                )
                console.print(f"💬 Response: {response}...")

    # Display results table
    console.print("\n" + "=" * 80, style="bold")
    console.print("📊 BENCHMARK RESULTS", style="bold cyan")
    console.print("=" * 80, style="bold")

    table = Table(title="Model Performance Comparison")

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Quant", style="magenta")
    table.add_column("Load (s)", style="green", justify="right")
    table.add_column("1st Token (s)", style="yellow", justify="right")
    table.add_column("Tok/s", style="red", justify="right")
    table.add_column("Response Preview", style="white")

    for r in results:
        table.add_row(
            r["Model"],
            r["Quantization"],
            r["Load Time (s)"],
            r["Time to 1st Token (s)"],
            r["Tokens/sec"],
            r["Response Preview"],
        )

    console.print(table)

    console.print(f"\n🎯 Test Question: '{test_question}'", style="bold")
    console.print(f"📈 Benchmarked {len(results)} model variants", style="bold")
    console.print(f"📈 Benchmarked {len(results)} model variants", style="bold")
