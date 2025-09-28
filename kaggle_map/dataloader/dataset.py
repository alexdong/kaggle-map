"""Stratified dataset utilities for MAP competition pipelines."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from loguru import logger

from kaggle_map.core.models import (
    Answer,
    Category,
    EvaluationRow,
    MLPTrainingConfig,
    Prediction,
    QuestionId,
    TrainingRow,
    default_mlp_training_config,
)
from kaggle_map.core.random_seed import configure_random_seed, get_active_seed
from kaggle_map.dataloader.sampling import stratified_sample
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

DEFAULT_SAMPLE_RATIO = 0.1
Split = Literal["train", "val", "test"]
MIN_TRAIN_RATIO = 0.5
MAX_TRAIN_RATIO = 0.9


def load_training_data(csv_path: Path) -> list[TrainingRow]:
    assert csv_path.exists(), f"Training file not found: {csv_path}"

    training_df = pd.read_csv(csv_path)
    logger.debug(f"Loaded CSV with columns: {list(training_df.columns)}")
    assert not training_df.empty, "Training CSV cannot be empty"

    training_rows = [TrainingRow.from_dataframe_row(row) for _, row in training_df.iterrows()]

    logger.debug(f"Parsed {len(training_rows)} training rows")
    assert training_rows, "Must parse at least one training row"
    return training_rows


def extract_correct_answers(
    training_data: Sequence[TrainingRow],
) -> dict[QuestionId, Answer]:
    assert training_data, "Training data cannot be empty"
    correct_answers: dict[QuestionId, Answer] = {}
    for row in training_data:
        if row.category == Category.TRUE_CORRECT:
            correct_answers.setdefault(row.question_id, row.mc_answer)

    logger.debug(f"Extracted correct answers for {len(correct_answers)} questions")
    assert correct_answers, "Must find at least one correct answer"
    assert all(isinstance(qid, int) for qid in correct_answers), "Question IDs must be integers"
    return correct_answers


def is_answer_correct(
    question_id: QuestionId,
    student_answer: Answer,
    correct_answers: Mapping[QuestionId, Answer],
) -> bool:
    correct_answer = correct_answers.get(question_id, "")
    return student_answer == correct_answer


def _stratum_key(row: TrainingRow) -> str:
    category_label = str(row.prediction)
    correctness_flag = "1" if row.category.is_correct_answer else "0"
    return f"{row.question_id}|{correctness_flag}|{category_label}"


def build_strata(rows: Sequence[TrainingRow]) -> dict[str, np.ndarray]:
    assert rows, "Cannot build strata from an empty sequence"

    keys = [_stratum_key(row) for row in rows]
    frame = pd.DataFrame({"key": keys, "index": np.arange(len(rows))})
    grouped = frame.groupby("key", sort=False)["index"]

    strata: dict[str, np.ndarray] = {}
    for key, indices in grouped:
        assert len(indices) > 0, "Each stratum must contain at least one index"
        strata[str(key)] = indices.to_numpy(copy=True)

    logger.debug("Built {} strata", len(strata))
    return strata


def stratified_splits(
    strata: Mapping[str, np.ndarray],
    train_ratio: float,
) -> dict[Split, list[int]]:
    assert strata, "Strata mapping cannot be empty"
    assert MIN_TRAIN_RATIO < train_ratio < MAX_TRAIN_RATIO, "train_ratio must leave room for validation/test"

    rng = np.random.default_rng(get_active_seed())
    splits: dict[Split, list[int]] = {"train": [], "val": [], "test": []}

    for indices in strata.values():
        assert indices.size > 0, "Stratum indices cannot be empty"
        shuffled = rng.permutation(indices)

        train_cutoff = int(len(indices) * train_ratio)
        train_indices = shuffled[:train_cutoff]
        remainder = shuffled[train_cutoff:]
        val_count = len(remainder) // 2

        val_indices = remainder[:val_count]
        test_indices = remainder[val_count:]

        splits["train"].extend(train_indices.tolist())
        splits["val"].extend(val_indices.tolist())
        splits["test"].extend(test_indices.tolist())

    total = sum(len(values) for values in splits.values())
    expected = sum(len(value) for value in strata.values())
    assert total == expected, "Split sizes must match total sample count"

    for split_name, split_indices in splits.items():
        logger.debug("Split {} contains {} indices", split_name, len(split_indices))

    return splits


def _extract_question_predictions(rows: Sequence[TrainingRow]) -> dict[QuestionId, list[str]]:
    question_predictions: dict[QuestionId, set[str]] = {}
    for row in rows:
        question_predictions.setdefault(row.question_id, set()).add(str(row.prediction))
    return {qid: sorted(preds) for qid, preds in question_predictions.items()}


def _derive_correct_answers(rows: Sequence[TrainingRow]) -> dict[QuestionId, str]:
    try:
        return extract_correct_answers(list(rows))
    except AssertionError as exc:
        logger.warning(
            "Falling back to student's multiple-choice answers as correct answers",
            total_rows=len(rows),
        )
        fallback: dict[QuestionId, str] = {}
        for row in rows:
            fallback.setdefault(row.question_id, row.mc_answer)
        if not fallback:
            msg = "Failed to derive fallback correct answers from rows"
            raise RuntimeError(msg) from exc
        return fallback


class MAPDataset(Sequence[TrainingRow]):
    """Lightweight dataset wrapper with stratified splits and helpers."""

    def __init__(self, *, csv_path: Path, config: MLPTrainingConfig) -> None:
        assert csv_path.exists(), f"Training CSV not found: {csv_path}"
        self._csv_path = csv_path

        logger.info("Loading training CSV from {}", csv_path)
        self._dataframe = pd.read_csv(csv_path)
        assert not self._dataframe.empty, "Training CSV cannot be empty"

        self._rows = [TrainingRow.from_dataframe_row(row) for _, row in self._dataframe.iterrows()]
        assert self._rows, "Failed to parse training rows"

        self._correct_answers = _derive_correct_answers(self._rows)
        self._question_predictions = _extract_question_predictions(self._rows)

        logger.info("Preparing stratified splits for {} rows", len(self._rows))
        self._strata = build_strata(self._rows)
        self._split_indices = stratified_splits(self._strata, config.train_split)
        train_count, val_count, test_count = self._split_sizes()
        logger.info(
            "Dataset initialised with {} train/{} val/{} test samples",
            train_count,
            val_count,
            test_count,
        )

    def _split_sizes(self) -> tuple[int, int, int]:
        return (
            len(self._split_indices["train"]),
            len(self._split_indices["val"]),
            len(self._split_indices["test"]),
        )

    @property
    def question_predictions(self) -> dict[QuestionId, list[str]]:
        return self._question_predictions

    @property
    def split_counts(self) -> dict[Split, int]:
        return {split: len(indices) for split, indices in self._split_indices.items()}

    @property
    def split_indices(self) -> dict[Split, list[int]]:
        return self._split_indices

    @property
    def correct_answers(self) -> Mapping[QuestionId, str]:
        return self._correct_answers

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: int | slice | Split) -> TrainingRow | list[TrainingRow]:
        if isinstance(key, str):
            split_key = cast("Split", key)
            assert split_key in self._split_indices, f"Unknown split: {key}"
            return [self._rows[idx] for idx in self._split_indices[split_key]]
        if isinstance(key, int):
            return self._rows[key]
        if isinstance(key, slice):
            return self._rows[key]
        msg = f"Unsupported index type for dataset access: {type(key)}"
        raise TypeError(msg)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def rows_for_question(self, question_id: QuestionId) -> list[TrainingRow]:
        assert isinstance(question_id, int), f"question_id must be int, got {type(question_id)}"
        matches = [row for row in self._rows if row.question_id == question_id]
        assert matches, f"No rows found for question {question_id}"
        return matches

    def evaluation_rows(self) -> list[EvaluationRow]:
        return [self._row_to_pair(row)[0] for row in self._rows]

    def evaluation_pairs(self) -> list[tuple[EvaluationRow, Prediction]]:
        pairs = [self._row_to_pair(row) for row in self._rows]
        assert len(pairs) == len(self._rows), "Each training row must yield exactly one evaluation pair"
        return pairs

    def evaluation_pairs_for_question(self, question_id: QuestionId) -> list[tuple[EvaluationRow, Prediction]]:
        rows = self.rows_for_question(question_id)
        return [self._row_to_pair(row) for row in rows]

    def sample_as_df(self, ratio: float = DEFAULT_SAMPLE_RATIO) -> pd.DataFrame:
        return stratified_sample(
            self._dataframe,
            sample_ratio=ratio,
            stratify_cols=("QuestionId", "Category", "MC_Answer"),
        )

    def sample_as_list(self, ratio: float = DEFAULT_SAMPLE_RATIO) -> list[TrainingRow]:
        sampled_df = self.sample_as_df(ratio)
        return [TrainingRow.from_dataframe_row(row) for _, row in sampled_df.iterrows()]

    def _row_to_pair(self, row: TrainingRow) -> tuple[EvaluationRow, Prediction]:
        correct_answer = self._correct_answers.get(row.question_id)
        evaluation = EvaluationRow(
            row_id=row.row_id,
            question_id=row.question_id,
            question_text=row.question_text,
            mc_answer=row.mc_answer,
            student_explanation=row.student_explanation,
            correct_answer=correct_answer,
        )
        return evaluation, row.prediction


if __name__ == "__main__":
    import click
    from rich.console import Console

    @click.command()
    @click.argument(
        "csv_path",
        type=click.Path(path_type=Path),
        default=Path("datasets/33474_focus_train.csv"),
    )
    @click.option("--train-split", type=float, default=0.7, show_default=True)
    @click.option("--seed", type=int, default=None, show_default=False, help="Override random seed for sampling")
    def main(csv_path: Path, train_split: float, seed: int | None) -> None:
        configure_logger(__name__, console_level="DEBUG")
        active_seed = configure_random_seed(override=seed)
        logger.debug("Dataset CLI configured random seed: {}", active_seed)
        base_config = default_mlp_training_config()
        config = base_config.model_copy(update={"train_csv_path": csv_path, "train_split": train_split})
        dataset = MAPDataset(csv_path=csv_path, config=config)
        console = Console()
        console.print(dataset.split_counts)
        console.print(dataset.sample_as_df(0.05).head())

    main()
