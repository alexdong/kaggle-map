"""Data models specific to the reranker module."""

from dataclasses import dataclass

from kaggle_map.core.models import EvaluationRow, Prediction


@dataclass(frozen=True)
class RerankingRequest:
    """Complete request for reranking predictions."""

    evaluation_row: EvaluationRow
    candidate_predictions: list[Prediction]

    @property
    def top_prediction(self) -> Prediction | None:
        """Get the current top prediction."""
        return self.candidate_predictions[0] if self.candidate_predictions else None
