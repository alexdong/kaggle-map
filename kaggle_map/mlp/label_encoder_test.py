"""Tests for label encoder."""

from sklearn.preprocessing import LabelEncoder

from kaggle_map.mlp.label_encoder import LabelEncoders


def test_label_encoders_encode_correct_predictions() -> None:
    # Create mock encoders
    true_encoder = LabelEncoder()
    true_encoder.fit(["True_Correct:NA", "True_Misconception:123", "True_Neither:NA"])

    false_encoder = LabelEncoder()
    false_encoder.fit(["False_Correct:NA", "False_Misconception:456", "False_Neither:NA"])

    encoders = LabelEncoders(
        true_label_encoders={1: true_encoder},
        false_label_encoders={1: false_encoder},
    )

    # Test encoding for correct answer
    label = encoders.encode(1, "True_Misconception:123", is_correct=True)
    assert label == 1  # Based on alphabetical order in fit

    # Test encoding for incorrect answer
    label = encoders.encode(1, "False_Misconception:456", is_correct=False)
    assert label == 1  # Based on alphabetical order in fit
