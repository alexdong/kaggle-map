#!/usr/bin/env python
"""Benchmark all LLM model/quantization combinations for performance and accuracy."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.llm.evaluator import (
    EvaluationConfig,
    evaluate_with_llm,
)
from kaggle_map.utils.gguf_model import (
    GGUFModelName,
    GGUFModelQuantizationLevel,
)
from kaggle_map.utils.logger_config import configure_logger


@dataclass
class BenchmarkResult:
    model_name: str
    quantization: str
    map_at_3: float
    avg_time_per_question: float
    total_time: float
    num_samples: int
    error: str | None = None


def run_benchmark(
    model_name: GGUFModelName,
    quantization: GGUFModelQuantizationLevel,
    data_path: Path,
    template_path: Path,
    sample_ratio: float = 0.1,
) -> BenchmarkResult:
    """Run a single benchmark for a model/quantization combination."""
    model_str = model_name.value
    quant_str = quantization.value

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting benchmark: {model_str} with {quant_str}")
    logger.info(f"{'='*60}")

    try:
        start_time = time.time()

        config = EvaluationConfig(
            template_path=template_path,
            data_path=data_path,
            sample_ratio=sample_ratio,
            row_ids=None,
            model_name=model_name,
            quantization=quantization,
        )

        # Run evaluation
        avg_map_score = evaluate_with_llm(config)

        total_time = time.time() - start_time

        # Calculate number of samples evaluated
        validation_df = pd.read_csv(data_path)
        num_samples = int(len(validation_df) * sample_ratio)

        avg_time_per_question = total_time / num_samples if num_samples > 0 else 0

        result = BenchmarkResult(
            model_name=model_str,
            quantization=quant_str,
            map_at_3=avg_map_score,
            avg_time_per_question=avg_time_per_question,
            total_time=total_time,
            num_samples=num_samples,
        )

        logger.success(f"Completed: {model_str} {quant_str}")
        logger.success(f"MAP@3: {avg_map_score:.4f}")
        logger.success(f"Avg time/question: {avg_time_per_question:.2f}s")

        return result

    except Exception as e:
        logger.error(f"Failed to benchmark {model_str} {quant_str}: {e}")
        return BenchmarkResult(
            model_name=model_str,
            quantization=quant_str,
            map_at_3=0.0,
            avg_time_per_question=0.0,
            total_time=0.0,
            num_samples=0,
            error=str(e),
        )


def save_results(results: list[BenchmarkResult], output_dir: Path = Path("logs")) -> Path:
    """Save benchmark results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON for full details
    json_path = output_dir / f"llm_benchmark_{timestamp}.json"
    with json_path.open("w") as f:
        json.dump(
            [
                {
                    "model_name": r.model_name,
                    "quantization": r.quantization,
                    "map_at_3": r.map_at_3,
                    "avg_time_per_question": r.avg_time_per_question,
                    "total_time": r.total_time,
                    "num_samples": r.num_samples,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            indent=2,
        )

    # Save as CSV for easy analysis
    csv_path = output_dir / f"llm_benchmark_{timestamp}.csv"
    df = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "quantization": r.quantization,
                "map_at_3": r.map_at_3,
                "avg_time_per_question": r.avg_time_per_question,
                "total_time_minutes": r.total_time / 60,
                "num_samples": r.num_samples,
                "error": r.error or "",
            }
            for r in results
        ]
    )
    df.to_csv(csv_path, index=False)

    logger.info(f"Results saved to {json_path} and {csv_path}")
    return csv_path


def display_results(results: list[BenchmarkResult]) -> None:
    """Display benchmark results in a nice table."""
    console = Console()

    table = Table(
        title="\n🏆 LLM Benchmark Results",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )

    table.add_column("Model", style="cyan")
    table.add_column("Quantization", style="yellow")
    table.add_column("MAP@3", style="green", justify="right")
    table.add_column("Avg Time/Q (s)", style="blue", justify="right")
    table.add_column("Total (min)", style="white", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Status", style="red")

    # Sort by MAP@3 score descending
    sorted_results = sorted(results, key=lambda x: x.map_at_3, reverse=True)

    for r in sorted_results:
        status = "✓" if r.error is None else "✗"
        table.add_row(
            r.model_name,
            r.quantization,
            f"{r.map_at_3:.4f}",
            f"{r.avg_time_per_question:.2f}",
            f"{r.total_time/60:.1f}",
            str(r.num_samples),
            status,
        )

    console.print(table)

    # Print summary statistics
    successful = [r for r in results if r.error is None]
    if successful:
        best = max(successful, key=lambda x: x.map_at_3)
        fastest = min(successful, key=lambda x: x.avg_time_per_question)

        console.print("\n📊 Summary:")
        console.print(f"Best MAP@3: {best.model_name} {best.quantization} = {best.map_at_3:.4f}")
        console.print(
            f"Fastest: {fastest.model_name} {fastest.quantization} = "
            f"{fastest.avg_time_per_question:.2f}s/question"
        )
        console.print(f"Total benchmark time: {sum(r.total_time for r in results)/60:.1f} minutes")


def main() -> None:
    """Run benchmarks for all model/quantization combinations."""

    @click.command()
    @click.option(
        "--data-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("datasets/33474_focus_group.csv"),
        help="Path to validation data CSV",
    )
    @click.option(
        "--template-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("kaggle_map/llm/prompts/predict.j2"),
        help="Path to prompt template",
    )
    @click.option(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of data to sample for benchmarking (0.0-1.0)",
    )
    @click.option(
        "--test-run",
        is_flag=True,
        help="Test with only first combination and 2% of data",
    )
    def run(data_path: Path, template_path: Path, sample_ratio: float, *, test_run: bool) -> None:
        """Benchmark all LLM model/quantization combinations."""

        # Configure logging
        configure_logger(__name__, console_level="INFO")

        # Define all combinations to test
        # Order by likely memory usage (smaller quantizations and models first)
        combinations = [
            # GPT-OSS 20B - Start with smaller quantizations
            (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q2_K_L),
            (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q3_K_M),
            (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q4_K_M),
            (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q4_K_XL),
            (GGUFModelName.GPT_OSS_20B, GGUFModelQuantizationLevel.Q5_K_M),
            # GEMMA 3 27B - Both quantizations
            (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q2_K_XL),
            (GGUFModelName.GEMMA_3_27B_IT, GGUFModelQuantizationLevel.Q3_K_XL),
        ]

        if test_run:
            logger.warning("TEST RUN: Using only first combination with 2% of data")
            combinations = combinations[:1]
            sample_ratio = 0.02

        logger.info(f"Starting benchmark of {len(combinations)} model/quantization combinations")
        logger.info(f"Data: {data_path}")
        logger.info(f"Sample ratio: {sample_ratio:.1%}")

        # Estimate time
        estimated_time_per_combo = 10 * 60  # 10 minutes per combination (rough estimate)
        total_estimated = len(combinations) * estimated_time_per_combo / 60
        logger.info(f"Estimated total time: {total_estimated:.1f} hours")

        results = []

        for i, (model_name, quantization) in enumerate(combinations, 1):
            logger.info(f"\n[{i}/{len(combinations)}] Processing {model_name.value} {quantization.value}")

            result = run_benchmark(
                model_name=model_name,
                quantization=quantization,
                data_path=data_path,
                template_path=template_path,
                sample_ratio=sample_ratio,
            )

            results.append(result)

            # Save intermediate results after each run (in case of crashes)
            save_results(results)

            # Display current standings
            display_results(results)

        logger.success("\n🎉 All benchmarks completed!")

        # Final save and display
        csv_path = save_results(results)
        display_results(results)

        logger.info(f"\nFinal results saved to: {csv_path}")

    run()


if __name__ == "__main__":
    main()
