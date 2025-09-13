#!/usr/bin/env python
"""Test specific GEMMA and GPT-OSS model combinations."""

from pathlib import Path

from benchmark_llm_models import display_results, run_benchmark, save_results
from kaggle_map.utils.gguf_model import GGUFModelName, GGUFModelQuantizationLevel
from kaggle_map.utils.logger_config import configure_logger

# Configure logging
configure_logger(__name__, console_level="INFO")

# Test these 3 specific combinations
combinations = [
    (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q2_K_XL),
    (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q3_K_XL),
]

# Configuration
data_path = Path("datasets/33474_focus_group.csv")
template_path = Path("kaggle_map/llm/prompts/predict.j2")
sample_ratio = 1.0  # Use full dataset

print(f"Testing {len(combinations)} model/quantization combinations")
print(f"Data: {data_path}")
print(f"Sample ratio: {sample_ratio:.1%}")
print("-" * 60)

results = []

for i, (model_name, quantization) in enumerate(combinations, 1):
    print(f"\n[{i}/{len(combinations)}] Processing {model_name.value} {quantization.value}")

    result = run_benchmark(
        model_name=model_name,
        quantization=quantization,
        data_path=data_path,
        template_path=template_path,
        sample_ratio=sample_ratio,
    )

    results.append(result)

    # Save intermediate results
    save_results(results)

    # Display current standings
    display_results(results)

print("\n" + "=" * 60)
print("🎉 All benchmarks completed!")
print("=" * 60)

# Final save and display
csv_path = save_results(results)
display_results(results)

print(f"\nFinal results saved to: {csv_path}")
