"""Core library for the RL Foundations harness.

The goal of `course.core` is to provide **boring primitives**:
- stable data types
- strict file IO / schema checks
- deterministic scoring contract
- run artifacts (manifest + results)

It is deliberately not a framework.
"""

from .types import Example, RolloutSample
from .scoring import score, SCORER_NAME, SCORER_VERSION

__all__ = [
    "Example",
    "RolloutSample",
    "score",
    "SCORER_NAME",
    "SCORER_VERSION",
]
