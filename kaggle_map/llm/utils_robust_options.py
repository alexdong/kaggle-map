"""Alternative robust parsing strategies for predictions."""

import contextlib
import re

from loguru import logger

from kaggle_map.core.models import Category, Prediction


# Option 2: Smart token splitting with category detection
def parse_predictions_option2(response: str) -> list[Prediction]:
    """Use regex to find category boundaries."""
    max_predictions = 3
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Category pattern: (True|False)_(Correct|Neither|Misconception)
    category_pattern = r"(True|False)_?(Correct|Neither|NNeither|Misconception)"

    # Find all category occurrences
    matches = list(re.finditer(category_pattern, response, re.IGNORECASE))
    predictions = []

    for i, match in enumerate(matches[:max_predictions]):
        # Extract category
        prefix = match.group(1).capitalize()
        suffix = match.group(2).capitalize()

        # Fix common typos
        if suffix == "Nneither":
            suffix = "Neither"

        category_str = f"{prefix}_{suffix}"

        # Extract misconception (between this match and next, or end)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)

        remainder = response[start:end].strip()

        # Check for colon
        misconception = remainder[1:].strip() or "NA" if remainder.startswith(":") else "NA"

        try:
            category = Category(category_str)
            predictions.append(Prediction(category=category, misconception=misconception))
        except ValueError:
            logger.debug(f"Invalid category: {category_str}")

    # Pad with defaults
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions


# Option 3: Fallback chain with multiple strategies
def parse_predictions_option3(response: str) -> list[Prediction]:
    """Try multiple parsing strategies in order."""
    max_predictions = 3
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    def try_standard_parse(text: str) -> list[Prediction]:
        """Standard space-split parsing."""
        predictions = []
        for part in text.split():
            if ":" in part:
                with contextlib.suppress(Exception):
                    predictions.append(Prediction.from_string(part))
        return predictions

    def try_newline_parse(text: str) -> list[Prediction]:
        """Try parsing by newlines (some LLMs use this)."""
        predictions = []
        for line in text.strip().split("\n")[:max_predictions]:
            if ":" in line:
                with contextlib.suppress(Exception):
                    predictions.append(Prediction.from_string(line.strip()))
        return predictions

    def try_comma_parse(text: str) -> list[Prediction]:
        """Try parsing by commas."""
        predictions = []
        for part in text.split(",")[:max_predictions]:
            if ":" in part:
                with contextlib.suppress(Exception):
                    predictions.append(Prediction.from_string(part.strip()))
        return predictions

    # Try strategies in order
    strategies = [
        try_standard_parse,
        try_newline_parse,
        try_comma_parse,
    ]

    predictions = []
    for strategy in strategies:
        predictions = strategy(response)
        if len(predictions) > 0:
            logger.debug(f"Parsed {len(predictions)} predictions using {strategy.__name__}")
            break

    # If still no predictions, try regex extraction
    if not predictions:
        predictions = parse_predictions_option2(response)[:max_predictions]

    # Pad with defaults
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions[:max_predictions]


# Option 4: Pre-process and clean before parsing
def parse_predictions_option4(response: str) -> list[Prediction]:
    """Clean and normalize input before parsing."""
    max_predictions = 3
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Common typo fixes
    typo_map = {
        "True_NNeither": "True_Neither",
        "False_NNeither": "False_Neither",
        "TrueCorrect": "True_Correct",
        "FalseCorrect": "False_Correct",
        "TrueNeither": "True_Neither",
        "FalseNeither": "False_Neither",
        "TrueMisconception": "True_Misconception",
        "FalseMisconception": "False_Misconception",
    }

    # Clean the response
    cleaned = response
    for typo, fix in typo_map.items():
        cleaned = cleaned.replace(typo, fix)

    # Add missing colons after categories without them
    for category in Category:
        # If we find category followed by space or end, add :NA
        pattern = f"{category.value}(?=\\s|$)"
        cleaned = re.sub(pattern, f"{category.value}:NA", cleaned)

    # Now try standard parsing on cleaned text
    predictions = []
    for part in cleaned.split()[:max_predictions]:
        if ":" in part:
            try:
                predictions.append(Prediction.from_string(part))
            except Exception as e:
                logger.debug(f"Failed to parse '{part}': {e}")

    # Pad with defaults
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions


# Option 5: Statistical approach with confidence scoring
def parse_predictions_option5(response: str) -> list[Prediction]:
    """Parse with confidence scoring for each extraction."""
    from difflib import SequenceMatcher

    max_predictions = 3
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Valid categories
    valid_categories = [cat.value for cat in Category]

    # Find all potential category-like strings
    tokens = re.findall(r"\S+", response)

    predictions = []

    for token in tokens:
        if len(predictions) >= max_predictions:
            break

        # Check similarity to valid categories
        best_match = None
        best_score = 0

        for category in valid_categories:
            # Check if token contains or is similar to category
            category_part = token.split(":")[0] if ":" in token else token

            score = SequenceMatcher(None, category_part.lower(), category.lower()).ratio()

            if score > best_score and score > 0.7:  # 70% similarity threshold
                best_score = score
                best_match = category

        if best_match:
            # Extract misconception
            if ":" in token:
                _, misconception = token.split(":", 1)
            else:
                misconception = "NA"

            with contextlib.suppress(Exception):
                predictions.append(Prediction(
                    category=Category(best_match),
                    misconception=misconception.strip() or "NA"
                ))

    # Pad with defaults
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "True_Correct:NA True_Neither:NA True_Misconception:Division",  # Normal
        "True_NNeither False_Correct:NA True_Misconception:Area",  # Typo
        "TrueCorrect FalseNeither:Something TrueMisconception:Math",  # Missing underscores
        "True_Correct True_Neither True_Misconception",  # Missing colons
        "True_Correct:NA,True_Neither:NA,True_Misconception:Division",  # Comma separated
        "True_Correct:NA\nTrue_Neither:NA\nTrue_Misconception:Division",  # Newline separated
    ]

    print("Testing different parsing strategies:\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test!r}")
        print("-" * 60)

        for option_num, parser in enumerate([
            parse_predictions_option2,
            parse_predictions_option3,
            parse_predictions_option4,
            parse_predictions_option5,
        ], 2):
            try:
                results = parser(test)
                print(f"Option {option_num}: {[str(p) for p in results]}")
            except Exception as e:
                print(f"Option {option_num}: ERROR - {e}")

        print()
