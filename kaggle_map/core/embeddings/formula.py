from __future__ import annotations

import re

_FRAC_RE = re.compile(r"\\frac\s*\{\s*([^\}]+)\s*\}\s*\{\s*([^\}]+)\s*\}")
_NUM_RE = re.compile(r"\d{2,}")


def normalize_latex_answer(s: str) -> str:
    if not s:
        return ""
    s = s.replace(r"\(", "").replace(r"\)", "").strip()

    # Process all fractions in the string
    def replace_frac(match) -> str:
        num, den = match.group(1).strip(), match.group(2).strip()
        # Check if both parts are numeric (integers)
        if (num.isdigit() or (num.startswith("-") and num[1:].isdigit())) and \
           (den.isdigit() or (den.startswith("-") and den[1:].isdigit())):
            # Numeric fraction - simplify if possible
            n, d = int(num), int(den)
            if d == 0:
                return f"{num}/{den}"  # Keep original for zero denominator
            return f"{n}/{d}"
        # Contains variables - keep as is
        return f"{num}/{den}"

    s = _FRAC_RE.sub(replace_frac, s)

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
