"""Compatibility wrapper.

The canonical implementation lives in `course.core.scoring`.
"""

from course.core.scoring import (  # noqa: F401
    PARSE_ERROR_CODES,
    REWARD_SPEC,
    RESULT_CODES,
    SCORER_NAME,
    SCORER_VERSION,
    score,
)

__all__ = [
    "SCORER_NAME",
    "SCORER_VERSION",
    "REWARD_SPEC",
    "PARSE_ERROR_CODES",
    "RESULT_CODES",
    "score",
]
