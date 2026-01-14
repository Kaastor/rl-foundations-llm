"""Assignment: Best-of-N selection policy (template)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from course.core.scoring import score as default_score
from course.core.types import Example, RolloutSample


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Result of selecting one sample from a list."""

    best_index: int
    best_sample: RolloutSample
    scored: dict[str, Any]


def tie_break_key(sample: RolloutSample) -> tuple[float, int, str]:
    """Deterministic tie-break key.

    TODO: implement tie-break precedence:
    - higher sum_logprob (if present)
    - shorter completion length
    - lexicographic order of completion text
    """

    # Placeholder: implement the real key.
    return (0.0, len(sample.completion), sample.completion)


def pick_best(
    example: Example,
    samples: list[RolloutSample],
    *,
    scorer: Callable[[Example, Any], dict[str, Any]] = default_score,
) -> SelectionResult:
    """Pick the best sample from a list.

    Sorting policy (expected):
    - highest reward first
    - then tie_break_key(sample) (ascending)
    - then earliest index (ascending)
    """

    # TODO: implement scoring + deterministic selection.
    if not samples:
        empty = RolloutSample(completion="")
        return SelectionResult(0, empty, scorer(example, empty.completion))

    raise NotImplementedError("Implement pick_best with deterministic ranking.")
