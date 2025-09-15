"""Robust prediction parser using Levenshtein distance for typo tolerance."""

from loguru import logger

from kaggle_map.core.models import Category, Prediction


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


def find_best_category_match(text: str, threshold: float = 0.7) -> Category | None:
    """Find the best matching category for a given text using fuzzy matching.

    Args:
        text: Text that might be a category (possibly with typos)
        threshold: Minimum similarity ratio to accept a match (0.0 to 1.0)

    Returns:
        Best matching Category or None if no good match found
    """
    best_category = None
    best_score = 0.0

    for category in Category:
        score = similarity_ratio(text, category.value)
        if score > best_score and score >= threshold:
            best_score = score
            best_category = category

    if best_category and best_score < 1.0:
        logger.debug(f"Fuzzy matched '{text}' to '{best_category.value}' (similarity: {best_score:.2%})")

    return best_category


def parse_single_prediction_robust(token: str) -> Prediction | None:
    """Parse a single token into a Prediction, handling typos and format issues.

    Args:
        token: A single token that should be Category:Misconception

    Returns:
        Parsed Prediction or None if unparseable
    """
    # Split on colon if present
    if ":" in token:
        category_part, misconception_part = token.split(":", 1)
        misconception = misconception_part.strip() or "NA"
    else:
        category_part = token
        misconception = "NA"

    # Try exact match first (fast path)
    try:
        category = Category(category_part)
        return Prediction(category=category, misconception=misconception)
    except ValueError:
        pass

    # Try fuzzy matching
    category = find_best_category_match(category_part)
    if category:
        return Prediction(category=category, misconception=misconception)

    logger.debug(f"Could not parse token: '{token}'")
    return None


def parse_predictions_robust(response: str, max_predictions: int = 3) -> list[Prediction]:
    """Parse LLM response to extract predictions with typo tolerance.

    Expected format: "Category1:Misconception1 Category2:Misconception2 Category3:Misconception3"

    Handles:
    - Typos in category names (True_NNeither -> True_Neither)
    - Missing underscores (TrueCorrect -> True_Correct)
    - Missing colons (categories without misconceptions get NA)
    - Multiple separators (spaces, newlines, commas)

    Args:
        response: Raw LLM response
        max_predictions: Maximum number of predictions to return

    Returns:
        List of exactly max_predictions Prediction objects (padded with defaults)
    """
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Extract the main prediction line (first non-empty line with content)
    prediction_line = response.strip()
    if "\n" in prediction_line:
        # If multiple lines, take the first one that looks like predictions
        for line in prediction_line.split("\n"):
            if line.strip() and any(cat.value.split("_")[0].lower() in line.lower() for cat in Category):
                prediction_line = line.strip()
                break

    # Try multiple separators
    tokens = []
    if " " in prediction_line and ":" in prediction_line:
        # Standard space-separated format
        tokens = prediction_line.split()
    elif "," in prediction_line:
        # Comma-separated format
        tokens = [t.strip() for t in prediction_line.split(",")]
    elif "\t" in prediction_line:
        # Tab-separated format
        tokens = prediction_line.split("\t")
    else:
        # Last resort - just split on whitespace
        tokens = prediction_line.split()

    # Parse tokens into predictions
    predictions = []
    for token in tokens:
        if len(predictions) >= max_predictions:
            break

        pred = parse_single_prediction_robust(token)
        if pred:
            predictions.append(pred)

    # Pad with defaults to ensure exactly max_predictions
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions[:max_predictions]