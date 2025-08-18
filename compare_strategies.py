#!/usr/bin/env python3
"""Comprehensive comparison between baseline and probabilistic strategies."""

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from kaggle_map.models import TestRow
from kaggle_map.strategies.baseline import BaselineStrategy
from kaggle_map.strategies.probabilistic import ProbabilisticStrategy


def main() -> None:
    """Compare baseline and probabilistic strategies."""
    console = Console()
    console.print("[bold blue]⚖️  Strategy Comparison: Baseline vs Probabilistic[/bold blue]")
    
    # Load performance history
    performance_history = _load_performance_history()
    
    if len(performance_history) < 2:
        console.print("[red]Need at least 2 results for comparison. Run both evaluations first.[/red]")
        return
    
    # Display comparison overview
    _display_performance_comparison(console, performance_history)
    
    # Load models for detailed analysis
    baseline_model, probabilistic_model = _load_models(console)
    
    if baseline_model and probabilistic_model:
        # Model architecture comparison
        _compare_model_architectures(console, baseline_model, probabilistic_model)
        
        # Sample a subset for detailed analysis
        console.print("\n[bold blue]🔍 Detailed Prediction Analysis[/bold blue]")
        _analyze_prediction_differences(console, baseline_model, probabilistic_model)
        
        # Performance breakdown by category
        _analyze_category_performance(console, baseline_model, probabilistic_model)


def _load_performance_history() -> list:
    """Load performance history from JSON file."""
    performance_log = Path("performance_history.json")
    if not performance_log.exists():
        return []
    
    with performance_log.open() as f:
        return json.load(f)


def _display_performance_comparison(console: Console, history: list) -> None:
    """Display comprehensive performance comparison."""
    # Find latest baseline and probabilistic results
    baseline_result = None
    probabilistic_result = None
    
    for entry in history:
        if entry["strategy"] == "baseline" and baseline_result is None:
            baseline_result = entry
        elif entry["strategy"] == "probabilistic" and probabilistic_result is None:
            probabilistic_result = entry
    
    if not baseline_result or not probabilistic_result:
        console.print("[red]Missing baseline or probabilistic results[/red]")
        return
    
    # Main comparison table
    comparison_table = Table(title="Strategy Performance Comparison")
    comparison_table.add_column("Metric", style="cyan", no_wrap=True)
    comparison_table.add_column("Baseline", style="blue")
    comparison_table.add_column("Probabilistic", style="magenta")
    comparison_table.add_column("Improvement", style="green")
    
    # MAP@3 Score
    map_improvement = probabilistic_result["map_score"] - baseline_result["map_score"]
    map_percent = (map_improvement / baseline_result["map_score"]) * 100
    
    comparison_table.add_row(
        "MAP@3 Score",
        f"{baseline_result['map_score']:.4f}",
        f"{probabilistic_result['map_score']:.4f}",
        f"+{map_improvement:.4f} (+{map_percent:.1f}%)"
    )
    
    # Perfect predictions
    baseline_perfect_rate = baseline_result["perfect_predictions"] / baseline_result["total_observations"]
    prob_perfect_rate = probabilistic_result["perfect_predictions"] / probabilistic_result["total_observations"]
    perfect_improvement = prob_perfect_rate - baseline_perfect_rate
    
    comparison_table.add_row(
        "Perfect Predictions",
        f"{baseline_result['perfect_predictions']} ({baseline_perfect_rate:.1%})",
        f"{probabilistic_result['perfect_predictions']} ({prob_perfect_rate:.1%})",
        f"+{probabilistic_result['perfect_predictions'] - baseline_result['perfect_predictions']} (+{perfect_improvement:.1%})"
    )
    
    # Execution time
    time_diff = probabilistic_result["total_execution_time"] - baseline_result["total_execution_time"]
    time_percent = (time_diff / baseline_result["total_execution_time"]) * 100
    
    comparison_table.add_row(
        "Execution Time",
        f"{baseline_result['total_execution_time']:.2f}s",
        f"{probabilistic_result['total_execution_time']:.2f}s",
        f"{time_diff:+.2f}s ({time_percent:+.1f}%)"
    )
    
    console.print(comparison_table)
    
    # Summary assessment
    console.print("\n[bold green]🎯 Key Findings:[/bold green]")
    console.print(f"• Probabilistic strategy achieves {map_improvement:.4f} higher MAP@3 score")
    console.print(f"• {probabilistic_result['perfect_predictions'] - baseline_result['perfect_predictions']} more perfect predictions")
    console.print(f"• Execution time difference: {time_diff:+.2f}s")
    
    if map_improvement > 0.05:
        console.print("• [bold green]Significant performance improvement![/bold green]")
    elif map_improvement > 0.01:
        console.print("• [green]Moderate performance improvement[/green]")
    else:
        console.print("• [yellow]Marginal performance difference[/yellow]")


def _load_models(console: Console) -> tuple:
    """Load both models for comparison."""
    baseline_path = Path("baseline_model.json")
    probabilistic_path = Path("probabilistic_model.json")
    
    baseline_model = None
    probabilistic_model = None
    
    try:
        if baseline_path.exists():
            baseline_model = BaselineStrategy.load(baseline_path)
            console.print(f"✅ Loaded baseline model from {baseline_path}")
        else:
            console.print(f"⚠️  Baseline model not found at {baseline_path}")
    except Exception as e:
        console.print(f"❌ Failed to load baseline model: {e}")
    
    try:
        if probabilistic_path.exists():
            probabilistic_model = ProbabilisticStrategy.load(probabilistic_path)
            console.print(f"✅ Loaded probabilistic model from {probabilistic_path}")
        else:
            console.print(f"⚠️  Probabilistic model not found at {probabilistic_path}")
    except Exception as e:
        console.print(f"❌ Failed to load probabilistic model: {e}")
    
    return baseline_model, probabilistic_model


def _compare_model_architectures(console: Console, baseline_model, probabilistic_model) -> None:
    """Compare the architectural differences between models."""
    console.print("\n[bold blue]🏗️  Model Architecture Comparison[/bold blue]")
    
    arch_table = Table()
    arch_table.add_column("Aspect", style="cyan")
    arch_table.add_column("Baseline", style="blue")
    arch_table.add_column("Probabilistic", style="magenta")
    
    # Model type
    arch_table.add_row(
        "Model Type",
        "Frequency-based lookup",
        "Two-stage probabilistic model"
    )
    
    # Complexity metrics
    baseline_questions = len(baseline_model.correct_answers)
    baseline_frequencies = sum(len(freq_dict) for freq_dict in baseline_model.category_frequencies.values())
    baseline_misconceptions = len(baseline_model.common_misconceptions)
    
    prob_contexts = len(probabilistic_model.category_distributions)
    prob_states = len(probabilistic_model.misconception_distributions)
    
    arch_table.add_row(
        "Primary Patterns",
        f"{baseline_questions} questions with {baseline_frequencies} category frequencies",
        f"{prob_contexts} response contexts"
    )
    
    arch_table.add_row(
        "Secondary Structure",
        f"{baseline_misconceptions} common misconceptions",
        f"{prob_states} state-category pairs"
    )
    
    # Fallback mechanisms
    arch_table.add_row(
        "Fallback Strategy",
        "Most frequent category per question",
        "Question-specific → Global priors"
    )
    
    # Memory footprint (rough estimate)
    baseline_size = baseline_questions + baseline_frequencies + baseline_misconceptions
    prob_size = prob_contexts + prob_states + len(probabilistic_model.question_category_priors)
    
    arch_table.add_row(
        "Model Size (approx)",
        f"{baseline_size} parameters",
        f"{prob_size} parameters"
    )
    
    console.print(arch_table)
    
    # Architecture summary
    console.print("\n[bold]Architectural Insights:[/bold]")
    console.print(f"• Baseline: Frequency-based lookup for {baseline_questions} questions")
    console.print(f"• Probabilistic: Two-stage model with {prob_contexts} contexts and {prob_states} state-category pairs")
    console.print(f"• Complexity ratio: {prob_size/baseline_size:.1f}x more parameters in probabilistic model")


def _analyze_prediction_differences(console: Console, baseline_model, probabilistic_model) -> None:
    """Analyze where the models make different predictions."""
    # Load a sample of test data
    train_csv_path = Path("dataset/train.csv")
    train_df = pd.read_csv(train_csv_path)
    
    # Take a representative sample
    sample_size = 100
    sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    
    # Convert to TestRow objects
    test_rows = []
    for _, row in sample_df.iterrows():
        test_row = TestRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )
        test_rows.append(test_row)
    
    # Get predictions from both models
    baseline_predictions = baseline_model.predict(test_rows)
    probabilistic_predictions = probabilistic_model.predict(test_rows)
    
    # Compare predictions
    agreements = 0
    improvements = 0
    degradations = 0
    
    diff_table = Table(title="Sample Prediction Differences")
    diff_table.add_column("Row ID", style="yellow")
    diff_table.add_column("Baseline Top", style="blue")
    diff_table.add_column("Probabilistic Top", style="magenta")
    diff_table.add_column("Ground Truth", style="green")
    diff_table.add_column("Result", style="bold")
    
    shown_diffs = 0
    max_show = 10
    
    for i, (baseline_pred, prob_pred) in enumerate(zip(baseline_predictions, probabilistic_predictions, strict=False)):
        if i >= len(test_rows):
            break
            
        row_id = baseline_pred.row_id
        
        # Get ground truth
        sample_row = sample_df[sample_df["row_id"] == row_id].iloc[0]
        misconception = sample_row["Misconception"] if pd.notna(sample_row["Misconception"]) else "NA"
        ground_truth = f"{sample_row['Category']}:{misconception}"
        
        # Get top predictions
        baseline_top = baseline_pred.predicted_categories[0].value if baseline_pred.predicted_categories else "None"
        prob_top = prob_pred.predicted_categories[0].value if prob_pred.predicted_categories else "None"
        
        # Check agreement
        if baseline_top == prob_top:
            agreements += 1
        else:
            # Check which is better
            baseline_correct = baseline_top == ground_truth
            prob_correct = prob_top == ground_truth
            
            if prob_correct and not baseline_correct:
                improvements += 1
                result = "✅ Improved"
            elif baseline_correct and not prob_correct:
                degradations += 1
                result = "❌ Degraded"
            else:
                result = "↔️ Different"
            
            # Show some examples
            if shown_diffs < max_show:
                diff_table.add_row(
                    str(row_id),
                    baseline_top,
                    prob_top,
                    ground_truth,
                    result
                )
                shown_diffs += 1
    
    console.print(diff_table)
    
    # Summary statistics
    total_compared = len(baseline_predictions)
    agreement_rate = agreements / total_compared
    
    console.print("\n[bold]Prediction Agreement Analysis:[/bold]")
    console.print(f"• Agreement rate: {agreements}/{total_compared} ({agreement_rate:.1%})")
    console.print(f"• Probabilistic improvements: {improvements}")
    console.print(f"• Probabilistic degradations: {degradations}")
    console.print(f"• Net improvement: {improvements - degradations} predictions")


def _analyze_category_performance(console: Console, baseline_model, probabilistic_model) -> None:
    """Analyze performance by category type."""
    console.print("\n[bold blue]📊 Performance by Category Analysis[/bold blue]")
    
    # This would require a more detailed breakdown by running evaluation on subsets
    # For now, show the model's understanding of categories
    
    console.print("[cyan]Category Understanding Comparison:[/cyan]")
    
    # Show global priors from probabilistic model
    if hasattr(probabilistic_model, "global_category_prior"):
        prob_table = Table(title="Probabilistic Model: Global Category Priors")
        prob_table.add_column("Category", style="cyan")
        prob_table.add_column("Prior Probability", style="magenta")
        
        sorted_priors = sorted(
            probabilistic_model.global_category_prior.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for category, prob in sorted_priors:
            prob_table.add_row(category.value, f"{prob:.3f}")
        
        console.print(prob_table)
    
    # Show baseline category frequencies if available
    if hasattr(baseline_model, "category_frequencies"):
        baseline_table = Table(title="Baseline Model: Sample Category Frequencies")
        baseline_table.add_column("Question", style="cyan")
        baseline_table.add_column("Correct Answer", style="blue")
        baseline_table.add_column("Categories (Correct)", style="green")
        baseline_table.add_column("Categories (Incorrect)", style="red")
        
        # Show sample of category frequencies
        sample_questions = list(baseline_model.category_frequencies.keys())[:5]
        
        for question_id in sample_questions:
            correct_answer = baseline_model.correct_answers.get(question_id, "Unknown")
            freq_data = baseline_model.category_frequencies[question_id]
            
            correct_cats = [cat.value for cat in freq_data.get(True, [])][:3]
            incorrect_cats = [cat.value for cat in freq_data.get(False, [])][:3]
            
            baseline_table.add_row(
                str(question_id),
                correct_answer,
                ", ".join(correct_cats) if correct_cats else "None",
                ", ".join(incorrect_cats) if incorrect_cats else "None"
            )
        
        console.print(baseline_table)
    
    console.print("\n[bold]Category Modeling Insights:[/bold]")
    console.print("• Baseline uses direct frequency counting")
    console.print("• Probabilistic models P(Category|Context) with fallbacks")
    console.print("• Probabilistic captures question-specific patterns")


if __name__ == "__main__":
    main()
