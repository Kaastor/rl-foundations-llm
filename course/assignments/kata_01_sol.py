from __future__ import annotations

from course.core.scoring import PARSE_ERROR_CODES


def classify(outcome_code: str) -> str:
    """
    Return one of:
      - model_math
      - model_format
      - data_invalid
      - unknown
    """

    if outcome_code == "wrong_answer":
        return "model_math"
    if outcome_code == "invalid_example":
        return "data_invalid"
    if outcome_code in PARSE_ERROR_CODES:
        return "model_format"
    return "unknown"
