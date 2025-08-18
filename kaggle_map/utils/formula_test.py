from __future__ import annotations

import pytest

from kaggle_map.utils.formula import (
    compose_text_unit,
    normalize_latex_answer,
    normalize_text,
    number_normalize,
)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (r"\( \\frac{3}{6} \)", "3/6"),
        (r"\( \\frac{1}{3} \)", "1/3"),
        (r"\( 10 \)", "10"),
        (r"Not enough information", "Not enough information"),
        (r"\\frac{2}{15}", "2/15"),
        (r"Some \\textbf{bold} thing", "Some bold thing"),
    ],
)
def test_normalize_latex_answer(inp: str, expected: str) -> None:
    assert normalize_latex_answer(inp) == expected


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("  hello   world ", "hello world"),
        ("A\nB\tC", "A B C"),
        ("", ""),
        (None, ""),  # type: ignore[arg-type]
    ],
)
def test_normalize_text(inp: str, expected: str) -> None:
    assert normalize_text(inp) == expected


@pytest.mark.parametrize(
    "q, a, e, expected_contains",
    [
        (
            "What fraction?",
            r"\( \\frac{3}{6} \)",
            "because reasons",
            ["Question: What fraction?", "Provided answer: 3/6", "Student explanation: because reasons"],
        ),
        (
            "Compute value",
            r"\( 192 \) hours",
            "time taken",
            ["Question: Compute value", "Provided answer: 192 hours", "Student explanation: time taken"],
        ),
    ],
)
def test_compose_text_unit(q: str, a: str, e: str, expected_contains: list[str]) -> None:
    out = compose_text_unit(q, a, e)
    for token in expected_contains:
        assert token in out


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("There are 20 apples and 7 pears", "There are <NUM> apples and 7 pears"),
        ("Year 2024-08-19", "Year <NUM>-<NUM>-<NUM>"),
        ("9 cats", "9 cats"),
        ("", ""),
        (None, ""),  # type: ignore[arg-type]
    ],
)
def test_number_normalize(inp: str, expected: str) -> None:
    assert number_normalize(inp) == expected

