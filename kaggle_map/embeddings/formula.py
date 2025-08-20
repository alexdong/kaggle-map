from __future__ import annotations

import re

_FRAC_RE = re.compile(r"\\frac\s*\{\s*([^\}]+)\s*\}\s*\{\s*([^\}]+)\s*\}")
_NUM_RE = re.compile(r"\d{2,}")


def normalize_latex_answer(s: str) -> str:
    """Turn LaTeX like \\( \frac{3}{6} \\) into '3/6'.

    Fallback: remove simple LaTeX commands and whitespace normalize.
    """
    if not s:
        return ""
    s = s.replace(r"\(", "").replace(r"\)", "").strip()
    m = _FRAC_RE.search(s)
    if m:
        num, den = m.group(1), m.group(2)
        # Validate that numerator and denominator are valid integers
        assert num.isdigit() or (num.startswith("-") and num[1:].isdigit()), (
            f"Invalid numerator in fraction: {num}"
        )
        assert den.isdigit() or (den.startswith("-") and den[1:].isdigit()), (
            f"Invalid denominator in fraction: {den}"
        )

        n, d = int(num), int(den)
        assert d != 0, f"Denominator cannot be zero in fraction: {num}/{den}"
        return f"{n}/{d}"
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # remove LaTeX commands like \textbf
    s = s.replace("{", "").replace("}", "")  # drop leftover braces
    s = s.replace("\\", "")  # drop stray backslashes
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
