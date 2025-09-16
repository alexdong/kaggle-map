"""Example usage of confidence routing data structures.

This demonstrates how the routing system would work in practice,
showing the data flow from MLP predictions through entropy-based
routing to final submission generation.
"""

import math
import time

from kaggle_map.core.models import (
    Category,
    LLMPredictionResult,
    MLPPredictionResult,
    Prediction,
    RoutedPrediction,
    RoutingDecision,
    RoutingSession,
)


def calculate_entropy(probabilities: list[float]) -> float:
    """Calculate entropy from probability distribution.

    Entropy = -sum(p_i * log(p_i)) for all probabilities.
    Higher entropy indicates more uncertainty.
    """
    assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

    entropy = 0.0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log(p)

    return entropy


def simulate_mlp_predictions() -> list[MLPPredictionResult]:
    """Simulate MLP predictions with varying confidence levels."""
    # Simulate different certainty levels
    prediction_scenarios = [
        # Very confident predictions (low entropy)
        {
            "row_id": 1,
            "question_id": 101,
            "predictions": [
                Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
                Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
                Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across_equals")
            ],
            "probabilities": [0.9, 0.08, 0.02]  # Very confident
        },
        # Moderately confident predictions (medium entropy)
        {
            "row_id": 2,
            "question_id": 102,
            "predictions": [
                Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Ignoring_remainder"),
                Prediction(category=Category.FALSE_CORRECT, misconception="NA"),
                Prediction(category=Category.TRUE_CORRECT, misconception="NA")
            ],
            "probabilities": [0.6, 0.25, 0.15]  # Moderate confidence
        },
        # Uncertain predictions (high entropy)
        {
            "row_id": 3,
            "question_id": 103,
            "predictions": [
                Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Converting_units_incorrectly"),
                Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
                Prediction(category=Category.FALSE_CORRECT, misconception="NA")
            ],
            "probabilities": [0.4, 0.35, 0.25]  # High uncertainty
        }
    ]

    results = []
    for scenario in prediction_scenarios:
        entropy = calculate_entropy(scenario["probabilities"])

        result = MLPPredictionResult(
            row_id=scenario["row_id"],
            question_id=scenario["question_id"],
            top_predictions=scenario["predictions"],
            top_probabilities=scenario["probabilities"],
            entropy=entropy,
            prediction_time_ms=2.0  # MLP is fast
        )
        results.append(result)

    return results


def make_routing_decisions(mlp_results: list[MLPPredictionResult]) -> list[RoutingDecision]:
    """Make routing decisions based on entropy threshold."""
    # Sort by entropy (high to low) to prioritize uncertain predictions
    sorted_results = sorted(mlp_results, key=lambda x: x.entropy, reverse=True)

    # Simple routing strategy: route top 50% of entropy scores
    entropy_threshold = 0.5  # Could be dynamic based on time budget

    decisions = []
    routing_rank = 1

    for result in sorted_results:
        should_route = result.entropy > entropy_threshold

        decision = RoutingDecision(
            row_id=result.row_id,
            entropy=result.entropy,
            should_route=should_route,
            routing_rank=routing_rank if should_route else None,
            reason=f"Entropy {result.entropy:.3f} {'>' if should_route else '<='} threshold {entropy_threshold}"
        )

        decisions.append(decision)
        if should_route:
            routing_rank += 1

    return decisions


def simulate_llm_processing(routed_prediction: RoutedPrediction) -> LLMPredictionResult:
    """Simulate LLM processing with chain-of-thought reasoning."""
    # Simulate LLM taking more time but potentially better accuracy
    reasoning = f"""
    Looking at question {routed_prediction.question_id}, I need to analyze the student's reasoning.

    The MLP predicted {routed_prediction.mlp_result.top_predictions[0]} with entropy {routed_prediction.entropy:.3f}.

    Based on chain-of-thought analysis, I believe the student's misconception is actually different.
    The reasoning pattern suggests a specific error in mathematical understanding.
    """

    # Simulate LLM making a different prediction than MLP
    llm_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Conceptual_error_identified"),
        Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA")
    ]

    return LLMPredictionResult(
        row_id=routed_prediction.row_id,
        question_id=routed_prediction.question_id,
        predictions=llm_predictions,
        reasoning=reasoning.strip(),
        prediction_time_ms=1500.0,  # LLM is much slower
        success=True
    )


def demonstrate_routing_pipeline() -> None:
    """Demonstrate the complete confidence-based routing pipeline."""
    print("=== Confidence-Based MLP/LLM Routing Demonstration ===")
    print()

    # Step 1: Generate MLP predictions
    print("Step 1: Generating MLP predictions...")
    mlp_results = simulate_mlp_predictions()

    for result in mlp_results:
        print(f"  Row {result.row_id}: Entropy {result.entropy:.3f}, Top prediction: {result.top_predictions[0]}")
    print()

    # Step 2: Make routing decisions
    print("Step 2: Making routing decisions...")
    routing_decisions = make_routing_decisions(mlp_results)

    # Create routing session
    session = RoutingSession(
        total_time_budget_seconds=10.0,  # 10 second budget for demo
        predictions={},
        entropy_sorted_row_ids=[],
        session_start_time=time.time()
    )

    # Add MLP predictions to session
    decision_map = {d.row_id: d for d in routing_decisions}
    for result in mlp_results:
        decision = decision_map[result.row_id]
        session.add_mlp_prediction(result, decision)

        status = "ROUTED TO LLM" if decision.should_route else "MLP ONLY"
        print(f"  Row {result.row_id}: {status} - {decision.reason}")

    # Set entropy-sorted order
    session.entropy_sorted_row_ids = [r.row_id for r in sorted(mlp_results, key=lambda x: x.entropy, reverse=True)]
    print()

    # Step 3: Process with LLM
    print("Step 3: Processing high-entropy predictions with LLM...")

    while not session.is_time_budget_exhausted:
        next_prediction = session.get_next_prediction_for_llm()
        if next_prediction is None:
            break

        print(f"  Processing Row {next_prediction.row_id} with LLM (entropy: {next_prediction.entropy:.3f})")

        # Simulate LLM processing
        llm_result = simulate_llm_processing(next_prediction)
        session.update_llm_result(next_prediction.row_id, llm_result)

        print(f"    LLM result: {llm_result.predictions[0]} (took {llm_result.prediction_time_ms}ms)")
        print(f"    Time remaining: {session.llm_time_remaining_seconds:.1f}s")

    print()

    # Step 4: Finalize and generate submission
    print("Step 4: Finalizing session and generating submission...")
    session.finalize_session()

    submission_data = session.get_submission_data()
    print(f"Generated submission for {len(submission_data)} rows:")

    for submission_row in submission_data:
        prediction = session.predictions[submission_row.row_id]
        source = "LLM" if prediction.used_llm_prediction else "MLP"
        print(f"  Row {submission_row.row_id}: {submission_row.predicted_categories[0]} (from {source})")

    print()

    # Step 5: Performance summary
    print("Step 5: Performance Summary")
    summary = session.get_performance_summary()

    print(f"  Total predictions: {summary['total_predictions']}")
    print(f"  Routed to LLM: {summary['predictions_routed_to_llm']} ({summary['routing_percentage']:.1%})")
    print(f"  LLM success rate: {summary['llm_success_rate']:.1%}")
    print(f"  Time budget utilization: {summary['time_budget_utilization']:.1%}")
    print(f"  Total LLM time: {summary['total_llm_time_used_seconds']:.1f}s")


if __name__ == "__main__":
    demonstrate_routing_pipeline()
