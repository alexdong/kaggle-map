import pytest

from kaggle_map.core.normalise import (
    compose_text_unit,
    normalize_latex_answer,
    normalize_text,
    number_normalize,
)


@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        # Basic fractions
        (r"\( \\frac{3}{6} \)", "3/6"),
        (r"\( \\frac{1}{3} \)", "1/3"),
        (r"\( \\frac{2}{15} \)", "2/15"),
        (r"\\frac{2}{15}", "2/15"),
        (r"\( \frac{11}{15} \)", "11/15"),
        (r"\( \frac{1}{12} \)", "1/12"),
        (r"\( \frac{3}{15} \)", "3/15"),
        (r"\( \frac{6}{2} \)", "6/2"),
        (r"\( \frac{10}{15} \)", "10/15"),
        (r"\( \frac{11}{30} \)", "11/30"),
        (r"\( \frac{3}{8} \)", "3/8"),
        (r"\( \frac{3}{9} \)", "3/9"),
        # Simple numbers
        (r"\( 10 \)", "10"),
        (r"\( 24 \)", "24"),
        (r"\( -3 \)", "-3"),
        (r"\( 13 \)", "13"),
        (r"\( 5 \)", "5"),
        (r"\( 12 \)", "12"),
        (r"\( 72 \)", "72"),
        # Numbers with units
        (r"\( 192 \) hours", "192 hours"),
        (r"\( 64 \) hours", "64 hours"),
        (r"\( 48 \) hours", "48 hours"),
        (r"\( 768 \) hours", "768 hours"),
        # Decimal numbers
        (r"\( 6.0001 \)", "6.0001"),
        (r"\( 6.2 \)", "6.2"),
        (r"\( 6.079 \)", "6.079"),
        # Mixed numbers (note: these stay as is after normalization)
        (r"\( 3 \frac{1}{3} \)", "3 1/3"),
        # Operations with fractions
        (r"\( \frac{1}{3} \times \frac{2}{3} \)", "(1/3) * (2/3)"),
        (r"\( \frac{2}{3} \div \frac{1}{3} \)", "(2/3) / (1/3)"),
        (r"\( \frac{1}{3}+\frac{2}{3} \)", "(1/3) + (2/3)"),
        (r"\( \frac{2}{3}-\frac{1}{3} \)", "(2/3) - (1/3)"),
        # Multiplication with \times
        (r"\( 5 \times 6 \)", "5*6"),
        (r"Calculate \( x \times y \)", "Calculate x*y"),
        (r"\( a \times b \times c \)", "a*b*c"),
        # Division with \div
        (r"\( 10 \div 2 \)", "10/2"),
        (r"\( x \div y \)", "x/y"),
        # Dot product with \cdot
        (r"\( 3 \cdot 4 \)", "3*4"),
        (r"\( a \cdot b \)", "a*b"),
        # LaTeX commands
        (r"Some \\textbf{bold} thing", "Some bold thing"),
        (r"\\textit{italic} text", "italic text"),
        (r"\\underline{underlined}", "underlined"),
        # Variables in fractions
        (r"\( \frac{A}{10}=\frac{9}{15} \) What is the value of \( A \) ?", "A/10=9/15 What is the value of A ?"),
        (r"\( \frac{x}{5} = 2 \)", "x/5 = 2"),
        (r"Find \( \frac{n}{12} \) when n = 6", "Find n/12 when n = 6"),
        (r"\( \frac{a+b}{c} \)", "(a+b)/c"),
        (r"\( \frac{x-y}{2} \)", "(x-y)/2"),
        # Edge cases
        (r"Not enough information", "Not enough information"),
        (r"", ""),
        (r"\( \)", ""),
        (r"\\( \\)", ""),
    ],
)
def test_normalize_latex_answer(inp: str, expected: str) -> None:
    assert normalize_latex_answer(inp) == expected, f"LaTeX normalization failed for input '{inp}'"


@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("  hello   world ", "hello world"),
        ("A\nB\tC", "A B C"),
        ("", ""),
        (None, ""),  # type: ignore[arg-type]
    ],
)
def test_normalize_text(inp: str, expected: str) -> None:
    assert normalize_text(inp) == expected, f"Text normalization failed for input '{inp}'"


@pytest.mark.parametrize(
    ("q", "a", "e", "expected_contains"),
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
        assert token in out, f"Expected token '{token}' not found in output: '{out}'"


@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("There are 20 apples and 7 pears", "There are <NUM> apples and 7 pears"),
        ("Year 2024-08-19", "Year <NUM>-<NUM>-<NUM>"),
        ("9 cats", "9 cats"),
        ("", ""),
        (None, ""),  # type: ignore[arg-type]
    ],
)
def test_number_normalize(inp: str, expected: str) -> None:
    assert number_normalize(inp) == expected, f"Number normalization failed for input '{inp}'"
