"""Label encoding for MLP predictions."""

from sklearn.preprocessing import LabelEncoder

from kaggle_map.core.models import QuestionId

__all__ = ["LabelEncoders"]


class LabelEncoders:
    """Manages label encoding for predictions by question and correctness."""

    def __init__(
        self,
        true_label_encoders: dict[QuestionId, LabelEncoder],
        false_label_encoders: dict[QuestionId, LabelEncoder],
    ) -> None:
        """Initialize with pre-fitted label encoders from the model.

        Args:
            true_label_encoders: Encoders for correct answer predictions
            false_label_encoders: Encoders for incorrect answer predictions
        """
        self.true_label_encoders = true_label_encoders
        self.false_label_encoders = false_label_encoders

    def encode(self, question_id: QuestionId, prediction: str, *, is_correct: bool) -> int:
        """Encode a prediction string to an integer label.

        Args:
            question_id: The question ID
            prediction: The prediction string to encode
            is_correct: Whether the student's answer was correct

        Returns:
            Encoded label as integer, defaults to 0 if encoder not found
        """
        encoder = (
            self.true_label_encoders.get(question_id) if is_correct else self.false_label_encoders.get(question_id)
        )

        if encoder is not None and hasattr(encoder, "classes_") and prediction in getattr(encoder, "classes_", []):
            return int(encoder.transform([prediction])[0])

        return 0  # Default to first class if not found

    def get_encoder(self, question_id: QuestionId, *, is_correct: bool) -> LabelEncoder | None:
        """Get the label encoder for a specific question and correctness.

        Args:
            question_id: The question ID
            is_correct: Whether the student's answer was correct

        Returns:
            The label encoder if found, None otherwise
        """
        return self.true_label_encoders.get(question_id) if is_correct else self.false_label_encoders.get(question_id)
