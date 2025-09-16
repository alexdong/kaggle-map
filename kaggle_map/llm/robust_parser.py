"""Robust prediction parser using Levenshtein distance for typo tolerance."""

from loguru import logger

from kaggle_map.core.models import Category, Prediction


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    assert isinstance(s1, str), f"s1 must be string, got {type(s1)}"
    assert isinstance(s2, str), f"s2 must be string, got {type(s2)}"

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
    assert isinstance(s1, str), f"s1 must be string, got {type(s1)}"
    assert isinstance(s2, str), f"s2 must be string, got {type(s2)}"

    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))

    # Both strings empty - they're identical
    if max_len == 0:
        return 1.0

    ratio = 1.0 - (distance / max_len)
    assert 0.0 <= ratio <= 1.0, f"Similarity ratio {ratio} out of valid range [0.0, 1.0]"
    return ratio


def find_best_category_match(text: str, threshold: float = 0.7) -> Category | None:
    """Find the best matching category for a given text using fuzzy matching.

    Args:
        text: Text that might be a category (possibly with typos)
        threshold: Minimum similarity ratio to accept a match (0.0 to 1.0)

    Returns:
        Best matching Category or None if no good match found
    """
    assert isinstance(text, str), f"text must be string, got {type(text)}"
    assert 0.0 <= threshold <= 1.0, f"threshold must be between 0.0 and 1.0, got {threshold}"
    assert text.strip(), "text cannot be empty or whitespace-only"
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
    assert isinstance(token, str), f"token must be string, got {type(token)}"

    token = token.strip()
    if not token:
        logger.debug("Cannot parse empty token")
        return None
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


def extract_tokens_from_response(response: str) -> list[str]:
    """Extract prediction tokens from LLM response, handling various formats."""
    prediction_line = response.strip()
    lines = prediction_line.split("\n")

    # Newline-separated format (3 lines with categories)
    if len(lines) >= 3 and all(":" in line or any(cat.value in line for cat in Category) for line in lines[:3]):
        return [line.strip() for line in lines if line.strip()]

    # Space-separated format
    if " " in prediction_line and ":" in prediction_line:
        return prediction_line.split()

    # Comma-separated format
    if "," in prediction_line:
        return [t.strip() for t in prediction_line.split(",")]

    # Tab-separated format
    if "\t" in prediction_line:
        return prediction_line.split("\t")

    # Default: split on whitespace
    return prediction_line.split()


def parse_predictions_with_fuzzy_matching(
    response: str, max_predictions: int = 3, pad_to_max: bool = True
) -> list[Prediction]:
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
        pad_to_max: If True, pad with defaults to ensure exactly max_predictions returned

    Returns:
        List of Prediction objects (up to max_predictions, padded if pad_to_max=True)
    """
    assert isinstance(response, str), f"response must be string, got {type(response)}"
    assert max_predictions > 0, f"max_predictions must be positive, got {max_predictions}"
    assert max_predictions <= 10, f"max_predictions seems unreasonable: {max_predictions}"
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")
    tokens = extract_tokens_from_response(response)

    predictions = []
    for token in tokens:
        if len(predictions) >= max_predictions:
            break

        pred = parse_single_prediction_robust(token)
        if pred:
            predictions.append(pred)

    # Pad with defaults if requested
    if pad_to_max:
        while len(predictions) < max_predictions:
            predictions.append(default_prediction)

    result = predictions[:max_predictions]

    # Only enforce exact count if padding is enabled
    if pad_to_max:
        assert len(result) == max_predictions, f"Expected {max_predictions} predictions, got {len(result)}"

    assert all(isinstance(p, Prediction) for p in result), "All results must be Prediction objects"

    return result
