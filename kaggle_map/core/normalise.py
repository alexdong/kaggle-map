from __future__ import annotations

import re
from re import Match

_FRAC_RE = re.compile(r"\\frac\s*\{\s*([^\}]+)\s*\}\s*\{\s*([^\}]+)\s*\}")
_NUM_RE = re.compile(r"\d{2,}")


def _format_fraction(num: str, den: str) -> str:
    """Format a fraction, adding parentheses if needed."""
    has_ops_in_num = any(op in num for op in ["+", "-", "*", "/"])

    # Check if both parts are numeric (integers)
    if (num.isdigit() or (num.startswith("-") and num[1:].isdigit())) and (
        den.isdigit() or (den.startswith("-") and den[1:].isdigit())
    ):
        # Numeric fraction - simplify if possible
        n, d = int(num), int(den)
        return f"{num}/{den}" if d == 0 else f"{n}/{d}"
    # Contains variables
    if has_ops_in_num:
        return f"({num})/{den}"
    return f"{num}/{den}"


def _replace_fraction_operations(s: str, fractions: list[tuple[int, int, str]]) -> str:
    """Replace fraction placeholders with properly formatted operations."""
    pattern = r"(__FRAC_\d+__)([+\-*/])(__FRAC_\d+__)"

    def replace_op(m: Match[str]) -> str:
        left_idx = int(m.group(1).replace("__FRAC_", "").replace("__", ""))
        op = m.group(2)
        right_idx = int(m.group(3).replace("__FRAC_", "").replace("__", ""))

        # Map operators to their spaced versions
        op_map = {"+": " + ", "-": " - ", "*": " * ", "/": " / "}
        spaced_op = op_map.get(op, f" {op} ")

        return f"({fractions[left_idx][2]}){spaced_op}({fractions[right_idx][2]})"

    if re.search(pattern, s):
        s = re.sub(pattern, replace_op, s)

    return s


def normalize_latex_answer(s: str) -> str:
    if not s:
        return ""
    s = s.replace(r"\(", "").replace(r"\)", "").strip()

    # Track fraction positions for later parenthesization
    fractions = []

    def replace_frac(match: Match[str]) -> str:
        num, den = match.group(1).strip(), match.group(2).strip()
        start_pos = match.start()

        result = _format_fraction(num, den)

        # Store fraction info for later processing
        fractions.append((start_pos, match.end(), result))
        return f"__FRAC_{len(fractions) - 1}__"

    # First pass: replace fractions with placeholders
    s = _FRAC_RE.sub(replace_frac, s)

    # Handle mathematical operators - with spaces for fraction operations
    s = re.sub(r"\s*\\times\s*", "*", s)
    s = re.sub(r"\s*\\div\s*", "/", s)
    s = re.sub(r"\s*\\cdot\s*", "*", s)

    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # remove LaTeX commands like \textbf
    s = s.replace("{", "").replace("}", "")  # drop leftover braces
    s = s.replace("\\", "")  # drop stray backslashes

    # Replace fraction operations with proper formatting
    s = _replace_fraction_operations(s, fractions)

    # Replace any remaining placeholders (single fractions not in operations)
    for i, (_, _, frac_text) in enumerate(fractions):
        s = s.replace(f"__FRAC_{i}__", frac_text)

    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def compose_text_unit(question: str, latex_ans: str, explanation: str) -> str:
    q = normalize_text(question)
    ans = normalize_latex_answer(latex_ans)
    exp = normalize_text(explanation)
    return f"Question: {q} | Provided answer: {ans} | Student explanation: {exp}"


def number_normalize(s: str) -> str:
    return _NUM_RE.sub("<NUM>", s or "")
