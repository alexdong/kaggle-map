"""Tests for dataset utilities and MAPDataset wrapper."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_map.core.models import (
    Category,
    MLPTrainingConfig,
    Prediction,
    TrainingRow,
    default_mlp_training_config,
)
from kaggle_map.core.random_seed import configure_random_seed
from kaggle_map.dataloader.dataset import (
    MAPDataset,
    build_strata,
    extract_correct_answers,
    is_answer_correct,
    load_training_data,
    stratified_splits,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_training_data() -> list[TrainingRow]:
    """Create sample training data for testing."""
    return [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="I added them correctly",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="5",
            student_explanation="I counted wrong",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Adding_across"),
        ),
        TrainingRow(
            row_id=3,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="3",
            student_explanation="I subtracted instead",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Subtraction_error"),
        ),
        TrainingRow(
            row_id=4,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="9",
            student_explanation="Correct multiplication",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
        ),
        TrainingRow(
            row_id=5,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="I added instead",
            prediction=Prediction(
                category=Category.FALSE_MISCONCEPTION, misconception="Addition_instead_multiplication"
            ),
        ),
        TrainingRow(
            row_id=6,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="Same mistake again",
            prediction=Prediction(
                category=Category.FALSE_MISCONCEPTION, misconception="Addition_instead_multiplication"
            ),
        ),
        TrainingRow(
            row_id=7,
            question_id=102,
            question_text="What is 5-2?",
            mc_answer="3",
            student_explanation="Correct subtraction",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
        ),
        TrainingRow(
            row_id=8,
            question_id=102,
            question_text="What is 5-2?",
            mc_answer="7",
            student_explanation="I don't know",
            prediction=Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
        ),
    ]


@pytest.fixture
def temp_training_csv() -> Iterator[Path]:
    """Create temporary training CSV file with comprehensive test data."""
    training_data = {
        "row_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "QuestionId": [100, 100, 100, 101, 101, 101, 102, 102],
        "QuestionText": [
            "What is 2+2?",
            "What is 2+2?",
            "What is 2+2?",
            "What is 3*3?",
            "What is 3*3?",
            "What is 3*3?",
            "What is 5-2?",
            "What is 5-2?",
        ],
        "MC_Answer": ["4", "5", "3", "9", "6", "6", "3", "7"],
        "StudentExplanation": [
            "I added them correctly",
            "I counted wrong",
            "I subtracted instead",
            "Correct multiplication",
            "I added instead",
            "Same mistake again",
            "Correct subtraction",
            "I don't know",
        ],
        "Category": [
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct",
            "False_Neither",
        ],
        "Misconception": [
            None,
            "Adding_across",
            "Subtraction_error",
            None,
            "Addition_instead_multiplication",
            "Addition_instead_multiplication",
            None,
            None,
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        pd.DataFrame(training_data).to_csv(handle.name, index=False)
        yield Path(handle.name)
        Path(handle.name).unlink()


@pytest.fixture(name="sample_csv")
def fixture_sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "row_id": 1,
                "QuestionId": 100,
                "QuestionText": "What is 2+2?",
                "MC_Answer": "4",
                "StudentExplanation": "Because 2+2=4",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 2,
                "QuestionId": 100,
                "QuestionText": "What is 2+2?",
                "MC_Answer": "4",
                "StudentExplanation": "Adding two twos",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 3,
                "QuestionId": 100,
                "QuestionText": "What is 2+2?",
                "MC_Answer": "3",
                "StudentExplanation": "I counted wrong",
                "Category": "False_Misconception",
                "Misconception": "Counting",
            },
            {
                "row_id": 4,
                "QuestionId": 100,
                "QuestionText": "What is 2+2?",
                "MC_Answer": "1",
                "StudentExplanation": "Subtract instead",
                "Category": "False_Misconception",
                "Misconception": "Operator mix-up",
            },
            {
                "row_id": 5,
                "QuestionId": 101,
                "QuestionText": "Simplify 6/9",
                "MC_Answer": "2/3",
                "StudentExplanation": "Divide both by 3",
                "Category": "True_Misconception",
                "Misconception": "Simplification",
            },
            {
                "row_id": 6,
                "QuestionId": 101,
                "QuestionText": "Simplify 6/9",
                "MC_Answer": "2/3",
                "StudentExplanation": "Reduce by gcd",
                "Category": "True_Misconception",
                "Misconception": "Simplification",
            },
            {
                "row_id": 7,
                "QuestionId": 101,
                "QuestionText": "Simplify 6/9",
                "MC_Answer": "6/9",
                "StudentExplanation": "Already simplest",
                "Category": "False_Neither",
                "Misconception": "NA",
            },
            {
                "row_id": 8,
                "QuestionId": 101,
                "QuestionText": "Simplify 6/9",
                "MC_Answer": "4/6",
                "StudentExplanation": "Divide by 1.5",
                "Category": "False_Neither",
                "Misconception": "NA",
            },
        ]
    )
    csv_path = tmp_path / "sample_train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Stratification Helper Tests
# =============================================================================


def _make_row(
    *,
    row_id: int,
    question_id: int,
    category: str,
    mc_answer: str = "A",
    misconception: str = "NA",
) -> TrainingRow:
    series = pd.Series(
        {
            "row_id": row_id,
            "QuestionId": question_id,
            "QuestionText": f"Question {question_id}",
            "MC_Answer": mc_answer,
            "StudentExplanation": f"Explanation {row_id}",
            "Category": category,
            "Misconception": misconception,
        }
    )
    return TrainingRow.from_dataframe_row(series)


def test_build_strata_grouping() -> None:
    rows = [
        _make_row(row_id=1, question_id=10, category="True_Correct"),
        _make_row(row_id=2, question_id=10, category="True_Correct"),
        _make_row(row_id=3, question_id=10, category="False_Misconception", misconception="Arithmetic"),
        _make_row(row_id=4, question_id=11, category="True_Misconception", misconception="Sign error"),
        _make_row(row_id=5, question_id=11, category="True_Misconception", misconception="Sign error"),
        _make_row(row_id=6, question_id=11, category="False_Correct"),
    ]

    strata = build_strata(rows)
    assert len(strata) == 4, f"Expected 4 strata, found {len(strata)}"

    key_counts = {key: len(indices) for key, indices in strata.items()}
    assert sorted(key_counts.values()) == [1, 1, 2, 2]


def test_stratified_splits_partition_sizes() -> None:
    rows = [_make_row(row_id=i, question_id=20 + i // 4, category="True_Correct") for i in range(12)]
    strata = build_strata(rows)

    configure_random_seed(override=99)
    splits = stratified_splits(strata, train_ratio=0.6)
    assert set(splits) == {"train", "val", "test"}

    total_indices = sum(len(indices) for indices in splits.values())
    assert total_indices == len(rows)

    train_fraction = len(splits["train"]) / len(rows)
    assert 0.4 < train_fraction < 0.7

    covered = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    expected = set(np.arange(len(rows)))
    assert covered == expected, "Splits should partition all indices"


# =============================================================================
# load_training_data Function Tests
# =============================================================================


def test_load_training_data_creates_strongly_typed_training_rows(temp_training_csv: Path) -> None:
    """load_training_data creates properly typed TrainingRow objects."""
    training_rows = load_training_data(temp_training_csv)

    assert len(training_rows) == 8
    assert all(isinstance(row, TrainingRow) for row in training_rows)

    first_row = training_rows[0]
    assert first_row.row_id == 1
    assert first_row.question_id == 100
    assert first_row.question_text == "What is 2+2?"
    assert first_row.mc_answer == "4"
    assert first_row.category == Category.TRUE_CORRECT
    assert first_row.misconception == "NA"


def test_load_training_data_handles_nan_misconceptions(temp_training_csv: Path) -> None:
    """load_training_data properly converts pandas NaN to "NA" for misconceptions."""
    training_rows = load_training_data(temp_training_csv)

    rows_without_misconceptions = [row for row in training_rows if row.misconception == "NA"]
    assert len(rows_without_misconceptions) == 4

    rows_with_misconceptions = [row for row in training_rows if row.misconception != "NA"]
    assert len(rows_with_misconceptions) == 4


def test_load_training_data_raises_error_for_missing_file() -> None:
    """load_training_data raises clear error for non-existent files."""
    missing_path = Path("nonexistent_file.csv")

    with pytest.raises(AssertionError, match="Training file not found"):
        load_training_data(missing_path)


def test_load_training_data_raises_error_for_empty_csv() -> None:
    """load_training_data raises error for empty CSV files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        pd.DataFrame().to_csv(handle.name, index=False)
        temp_path = Path(handle.name)

    try:
        with pytest.raises(pd.errors.EmptyDataError):
            load_training_data(temp_path)
    finally:
        temp_path.unlink()


# =============================================================================
# extract_correct_answers Function Tests
# =============================================================================


def test_extract_correct_answers_finds_true_correct_answers(sample_training_data: list[TrainingRow]) -> None:
    """extract_correct_answers identifies correct answers from True_Correct categories."""
    correct_answers = extract_correct_answers(sample_training_data)

    expected_answers = {100: "4", 101: "9", 102: "3"}
    assert correct_answers == expected_answers


def test_extract_correct_answers_handles_single_correct_answer_per_question(
    sample_training_data: list[TrainingRow],
) -> None:
    """extract_correct_answers works when each question has only one correct answer."""
    correct_answers = extract_correct_answers(sample_training_data)

    assert len(correct_answers) == 3
    assert all(isinstance(qid, int) for qid in correct_answers)
    assert all(isinstance(answer, str) for answer in correct_answers.values())


def test_extract_correct_answers_uses_first_correct_answer_when_multiple_exist() -> None:
    """extract_correct_answers uses the first correct answer when multiple exist for same question."""
    conflicting_data = [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="Test",
            mc_answer="A",
            student_explanation="Test",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="Test",
            mc_answer="B",
            student_explanation="Test",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
        ),
    ]

    correct_answers = extract_correct_answers(conflicting_data)
    assert correct_answers[100] == "A"


def test_extract_correct_answers_raises_error_for_empty_data() -> None:
    """extract_correct_answers raises error for empty training data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        extract_correct_answers([])


def test_extract_correct_answers_raises_error_when_no_correct_answers_found() -> None:
    """extract_correct_answers raises error when no True_Correct categories exist."""
    no_correct_data = [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="Test",
            mc_answer="A",
            student_explanation="Test",
            prediction=Prediction(category=Category.FALSE_NEITHER, misconception="NA"),
        ),
    ]

    with pytest.raises(AssertionError, match="Must find at least one correct answer"):
        extract_correct_answers(no_correct_data)


# =============================================================================
# is_answer_correct Function Tests
# =============================================================================


def test_is_answer_correct_returns_true_for_matching_answers() -> None:
    """is_answer_correct returns True when student answer matches correct answer."""
    correct_answers = {100: "4", 101: "9"}

    assert is_answer_correct(100, "4", correct_answers) is True
    assert is_answer_correct(101, "9", correct_answers) is True


def test_is_answer_correct_returns_false_for_non_matching_answers() -> None:
    """is_answer_correct returns False when student answer doesn't match correct answer."""
    correct_answers = {100: "4", 101: "9"}

    assert is_answer_correct(100, "5", correct_answers) is False
    assert is_answer_correct(101, "6", correct_answers) is False


def test_is_answer_correct_returns_false_for_unknown_questions() -> None:
    """is_answer_correct returns False for questions not in correct_answers dict."""
    correct_answers = {100: "4"}

    assert is_answer_correct(999, "4", correct_answers) is False


# =============================================================================
# MAPDataset Integration Tests
# =============================================================================


def _make_config(csv_path: Path) -> MLPTrainingConfig:
    base = default_mlp_training_config()
    return base.model_copy(update={"train_csv_path": csv_path, "batch_size": 8, "train_split": 0.7})


FULL_TRAIN_CSV = Path("datasets/33474_full_train.csv")
TINY_TRAIN_CSV = Path("datasets/33474_tiny_train.csv")


def _build_dataset(csv_path: Path) -> MAPDataset:
    config = default_mlp_training_config().model_copy(update={"train_csv_path": csv_path})
    return MAPDataset(csv_path=csv_path, config=config)


def test_sample_as_df_respects_stratification() -> None:
    dataset = _build_dataset(TINY_TRAIN_CSV)
    sample = dataset.sample_as_df(0.5)
    assert not sample.empty, "Stratified sample should return rows"
    assert {"QuestionId", "Category", "MC_Answer"}.issubset(sample.columns), (
        "Sampled dataframe missing stratify columns"
    )


def test_dataset_split_counts(sample_csv: Path) -> None:
    config = _make_config(sample_csv)
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)

    counts = dataset.split_counts
    assert sum(counts.values()) == len(dataset)
    assert counts["train"] > 0
    assert counts["val"] >= 0
    assert counts["test"] >= 0


def test_split_indexing_returns_training_rows(sample_csv: Path) -> None:
    config = _make_config(sample_csv)
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)
    split_rows = dataset["val"]
    assert isinstance(split_rows, list)
    assert all(isinstance(row, TrainingRow) for row in split_rows)
    assert len(split_rows) == dataset.split_counts["val"]


def test_sampling_helpers_consistent_sizes(sample_csv: Path) -> None:
    config = _make_config(sample_csv)
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)
    sample_df = dataset.sample_as_df(0.2)
    sample_rows = dataset.sample_as_list(0.2)
    assert len(sample_df) == len(sample_rows)
    assert not sample_df.empty


def test_evaluation_pairs_cover_all_rows(sample_csv: Path) -> None:
    config = _make_config(sample_csv)
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)
    pairs = dataset.evaluation_pairs()
    assert len(pairs) == len(dataset)
    assert all(pair[0].row_id == row.row_id for pair, row in zip(pairs, dataset, strict=False))


def test_dataset_supports_indexing_and_slicing(sample_csv: Path) -> None:
    config = _make_config(sample_csv)
    dataset = MAPDataset(csv_path=config.train_csv_path, config=config)

    first_row = dataset[0]
    assert isinstance(first_row, TrainingRow)

    last_row = dataset[-1]
    assert isinstance(last_row, TrainingRow)

    subset = dataset[:3]
    assert isinstance(subset, list)
    assert len(subset) == 3
    assert all(isinstance(row, TrainingRow) for row in subset)

    remainder = dataset[3:]
    assert isinstance(remainder, list)
    assert len(subset) + len(remainder) == len(dataset)

    iterated_ids = [row.row_id for row in dataset]
    assert iterated_ids == [row.row_id for row in subset + remainder]

    val_rows = dataset["val"]
    assert isinstance(val_rows, list)
    assert len(val_rows) == dataset.split_counts["val"]


if __name__ == "__main__":
    raise SystemExit(0)
