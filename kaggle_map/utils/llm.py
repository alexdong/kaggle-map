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
    MODEL_OPTIONS,
    InferenceConfig,
    LLMModelLoadConfig,
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
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_path.parent,
        local_dir_use_symlinks=False,  # Copy file instead of symlink
    )
    assert model_path.exists(), f"Model file not found after download: {model_path}"
    logger.info(f"Model downloaded successfully: {model_path}")
    return model_path


@contextmanager
def load_llm_model(config: LLMModelLoadConfig) -> Iterator[Llama]:
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
    from statistics import mean, stdev

    import psutil
    from llama_cpp import llama_supports_gpu_offload

    console = Console()

    console.print("🚀 LLM Model Benchmarking Tool", style="bold cyan")
    console.print("=" * 50)

    # Check GPU support
    gpu_available = llama_supports_gpu_offload()
    console.print(f"\n🎮 GPU Support: {'YES ✅' if gpu_available else 'NO ❌ (CPU only)'}")
    if not gpu_available:
        console.print("   Note: GPU support not detected. Will benchmark CPU only.", style="yellow")
        console.print("   For GPU: rebuild llama-cpp-python with CUDA support", style="yellow")
    console.print("=" * 50)

    # Test question
    test_question = "Who is the Bosch in the Haber-Bosch process?"

    # Benchmark parameters
    WARMUP_RUNS = 3
    MEASUREMENT_RUNS = 10

    # Results storage
    results = []

    # Test configurations - CPU and GPU (if available)
    test_configs = [
        ("CPU", 0),  # No GPU layers
    ]
    if gpu_available:
        test_configs.append(("GPU", -1))  # All layers on GPU

    # Download and benchmark all model variants
    for model_name in MODEL_OPTIONS:
        gguf_repo_spec = GGUF_MODELS[model_name]
        for quantization in gguf_repo_spec.available_quantizations:
            for device_name, n_gpu_layers in test_configs:
                console.print(f"\n📦 Processing {model_name} - {quantization} on {device_name}", style="bold yellow")
                console.print("-" * 40)

                # Create model loading config
                load_config = LLMModelLoadConfig(
                    model_name=model_name,
                    quantization=quantization,
                    n_ctx=2048,  # Smaller context for benchmarking
                    n_gpu_layers=n_gpu_layers,  # Control GPU usage
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
                benchmark_start = time.time()
                with load_llm_model(load_config) as llm:
                    # Track memory usage
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

                    # Warmup runs
                    console.print(f"🔥 Warming up with {WARMUP_RUNS} runs...")
                    for _ in range(WARMUP_RUNS):
                        _ = llm(
                            test_question,
                            max_tokens=inference_config.max_tokens,
                            temperature=inference_config.temperature,
                            echo=inference_config.echo,
                        )

                    # Measurement runs
                    console.print(f"📊 Running {MEASUREMENT_RUNS} measurements...")
                    latencies = []
                    token_counts = []

                    for _i in range(MEASUREMENT_RUNS):
                        start_inference = time.time()
                        output = llm(
                            test_question,
                            max_tokens=inference_config.max_tokens,
                            temperature=inference_config.temperature,
                            echo=inference_config.echo,
                        )
                        latency_ms = (time.time() - start_inference) * 1000  # Convert to ms
                        latencies.append(latency_ms)

                        # Extract response and count tokens
                        response = output["choices"][0]["text"].strip()  # type: ignore
                        tokens = len(response.split())  # Simple word count
                        token_counts.append(tokens)

                    # Track peak memory during inference
                    memory_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
                    memory_used = memory_after - memory_before

                    # Calculate total time for this model/quant combo
                    total_time_s = time.time() - benchmark_start

                    # Calculate statistics
                    mean_latency = mean(latencies)
                    std_latency = stdev(latencies) if len(latencies) > 1 else 0
                    min_latency = min(latencies)
                    max_latency = max(latencies)

                    mean_tokens = mean(token_counts)
                    tokens_per_sec = mean_tokens / (mean_latency / 1000) if mean_latency > 0 else 0

                    # Store results
                    results.append(
                        {
                            "Model": f"{model_name}",
                            "Quant": quantization,
                            "Device": device_name,
                            "Latency (ms)": f"{mean_latency:.0f} ± {std_latency:.0f}",
                            "Tok/s": f"{tokens_per_sec:.1f}",
                            "RAM (GB)": f"{memory_used:.1f}",
                            "Total (s)": f"{total_time_s:.1f}",
                            "min_latency": min_latency,
                            "max_latency": max_latency,
                        }
                    )

                    console.print(
                        f"⚡ {device_name}: Latency {mean_latency:.0f} ± {std_latency:.0f} ms | "
                        f"Throughput: {tokens_per_sec:.1f} tok/s | "
                        f"Memory: {memory_used:.1f} GB | "
                        f"Total: {total_time_s:.1f}s",
                        style="blue",
                    )

    # Display results table
    console.print("\n" + "=" * 80, style="bold")
    console.print("📊 BENCHMARK RESULTS", style="bold cyan")
    console.print("=" * 80, style="bold")

    table = Table()
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Quant", style="magenta")
    table.add_column("Device", style="white")
    table.add_column("Latency (ms)", style="yellow", justify="right")
    table.add_column("Tok/s", style="green", justify="right")
    table.add_column("RAM (GB)", style="red", justify="right")
    table.add_column("Total (s)", style="blue", justify="right")

    for r in results:
        table.add_row(
            r["Model"],
            r["Quant"],
            r["Device"],
            r["Latency (ms)"],
            r["Tok/s"],
            r["RAM (GB)"],
            r["Total (s)"],
        )

    console.print(table)

    console.print(f"\n🎯 Test Question: '{test_question}'")
    console.print(f"📈 {WARMUP_RUNS} warmup runs, {MEASUREMENT_RUNS} measurements per model")
    console.print(f"📊 Benchmarked {len(results)} model variants")
