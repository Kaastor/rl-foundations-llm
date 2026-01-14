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

    # TODO: implement the mapping described in Project.md.
    # Hints:
    # - "wrong_answer" -> model_math
    # - "invalid_example" -> data_invalid
    # - parse error codes (PARSE_ERROR_CODES) -> model_format
    return "unknown"
