"""Tests for confidence routing data structures."""

import time

import pytest
from pydantic import ValidationError

from kaggle_map.core.models import (
    Category,
    LLMPredictionResult,
    MLPPredictionResult,
    Prediction,
    PredictionState,
    RoutedPrediction,
    RoutingDecision,
    RoutingSession,
)


def test_mlp_prediction_result_validation():
    """Test MLP prediction result validation."""
    # Valid prediction result
    pred1 = Prediction(category=Category.TRUE_CORRECT, misconception="NA")
    pred2 = Prediction(category=Category.FALSE_NEITHER, misconception="NA")

    mlp_result = MLPPredictionResult(
        row_id=1,
        question_id=101,
        top_predictions=[pred1, pred2],
        top_probabilities=[0.7, 0.3],
        entropy=0.5,
        prediction_time_ms=2.5,
    )

    assert mlp_result.row_id == 1
    assert mlp_result.entropy == 0.5
    assert len(mlp_result.top_predictions) == 2


def test_mlp_prediction_result_validation_failures():
    """Test that MLP prediction validation catches errors."""
    pred1 = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Mismatched predictions and probabilities
    with pytest.raises(ValidationError):
        MLPPredictionResult(
            row_id=1,
            question_id=101,
            top_predictions=[pred1],
            top_probabilities=[0.7, 0.3],  # Too many probabilities
            entropy=0.5,
            prediction_time_ms=2.5,
        )

    # Probabilities don't sum to 1.0
    with pytest.raises(ValidationError):
        MLPPredictionResult(
            row_id=1,
            question_id=101,
            top_predictions=[pred1],
            top_probabilities=[0.5],  # Doesn't sum to 1.0
            entropy=0.5,
            prediction_time_ms=2.5,
        )


def test_llm_prediction_result_validation():
    """Test LLM prediction result validation."""
    pred1 = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Successful LLM result
    llm_result = LLMPredictionResult(
        row_id=1,
        question_id=101,
        predictions=[pred1],
        reasoning="The student correctly identified the concept.",
        prediction_time_ms=1500.0,
        success=True,
    )

    assert llm_result.success
    assert len(llm_result.predictions) == 1
    assert llm_result.reasoning.strip()

    # Failed LLM result (no validation for empty predictions when failed)
    failed_result = LLMPredictionResult(
        row_id=1, question_id=101, predictions=[], reasoning="", prediction_time_ms=500.0, success=False
    )

    assert not failed_result.success
    assert len(failed_result.predictions) == 0


def test_routing_decision_validation():
    """Test routing decision validation."""
    # Valid routing decision
    routing_decision = RoutingDecision(
        row_id=1, entropy=0.8, should_route=True, routing_rank=1, reason="High entropy indicates uncertainty"
    )

    assert routing_decision.should_route
    assert routing_decision.routing_rank == 1

    # Should route without rank fails
    with pytest.raises(ValidationError):
        RoutingDecision(row_id=1, entropy=0.8, should_route=True, routing_rank=None, reason="Missing rank")


def test_routed_prediction_consistency():
    """Test routed prediction consistency validation."""
    pred1 = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    mlp_result = MLPPredictionResult(
        row_id=1, question_id=101, top_predictions=[pred1], top_probabilities=[1.0], entropy=0.0, prediction_time_ms=2.0
    )

    routing_decision = RoutingDecision(row_id=1, entropy=0.0, should_route=False, reason="Low entropy")

    routed_prediction = RoutedPrediction(
        row_id=1,
        question_id=101,
        mlp_result=mlp_result,
        routing_decision=routing_decision,
        state=PredictionState.MLP_ONLY,
    )

    assert routed_prediction.entropy == 0.0
    assert not routed_prediction.was_routed_to_llm
    assert not routed_prediction.used_llm_prediction
    assert routed_prediction.get_best_predictions() == [pred1]


def test_routing_session_workflow():
    """Test complete routing session workflow."""
    # Create session
    session = RoutingSession(
        total_time_budget_seconds=300.0,  # 5 minutes
        predictions={},
        entropy_sorted_row_ids=[],
        session_start_time=time.time(),
    )

    # Add MLP predictions
    pred1 = Prediction(category=Category.TRUE_CORRECT, misconception="NA")
    pred2 = Prediction(category=Category.FALSE_NEITHER, misconception="NA")

    mlp_result_1 = MLPPredictionResult(
        row_id=1,
        question_id=101,
        top_predictions=[pred1],
        top_probabilities=[1.0],
        entropy=0.1,  # Low entropy
        prediction_time_ms=2.0,
    )

    mlp_result_2 = MLPPredictionResult(
        row_id=2,
        question_id=102,
        top_predictions=[pred2],
        top_probabilities=[1.0],
        entropy=0.9,  # High entropy
        prediction_time_ms=2.0,
    )

    # Add to session
    session.add_mlp_prediction(
        mlp_result_1, RoutingDecision(row_id=1, entropy=0.1, should_route=False, reason="Low entropy")
    )

    session.add_mlp_prediction(
        mlp_result_2, RoutingDecision(row_id=2, entropy=0.9, should_route=True, routing_rank=1, reason="High entropy")
    )

    # Set entropy sorting
    session.entropy_sorted_row_ids = [2, 1]  # Sorted by entropy high to low

    # Check session state
    assert session.total_predictions == 2
    assert session.predictions_routed_to_llm == 1
    assert session.llm_time_remaining_seconds == 300.0

    # Get next prediction for LLM
    next_pred = session.get_next_prediction_for_llm()
    assert next_pred is not None
    assert next_pred.row_id == 2
    assert next_pred.state == PredictionState.LLM_PENDING

    # Simulate LLM processing
    llm_result = LLMPredictionResult(
        row_id=2,
        question_id=102,
        predictions=[pred1],  # LLM's prediction
        reasoning="Chain of thought reasoning here.",
        prediction_time_ms=1500.0,
        success=True,
    )

    session.update_llm_result(2, llm_result)

    # Check updated state
    assert session.predictions_completed_by_llm == 1
    assert session.total_llm_time_used_seconds == 1.5
    assert session.predictions[2].state == PredictionState.LLM_COMPLETE
    assert session.predictions[2].used_llm_prediction

    # Finalize session
    session.finalize_session()
    assert session.llm_processing_complete

    # Get submission data
    submission_data = session.get_submission_data()
    assert len(submission_data) == 2
    assert submission_data[0].row_id == 1
    assert submission_data[1].row_id == 2

    # Check performance summary
    summary = session.get_performance_summary()
    assert summary["total_predictions"] == 2
    assert summary["predictions_routed_to_llm"] == 1
    assert summary["predictions_completed_by_llm"] == 1
    assert summary["llm_success_rate"] == 1.0
    assert summary["routing_percentage"] == 0.5
