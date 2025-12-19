"""Assignment: Best-of-N selection policy.

This file is a deliberately small, student-editable surface.

Rules:
- Keep it deterministic.
- Keep it explainable.
- Do not import heavy libraries.

Default behavior:
- score each sample
- choose max reward
- tie-break by (lexicographically smallest completion text, then earliest index)

Production relevance:
Selection (best-of-N, reranking, filtering, rejection sampling) is extremely common
in real LLM stacks. It can improve the chosen output without changing the underlying
model weights.

Your goal in this assignment is to make tie-breaking and selection behavior explicit
and easy to inspect.
"""

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


def tie_break_key(sample: RolloutSample) -> tuple[int, str]:
    """Deterministic tie-break key.

    The default uses the completion text.

    You can change this to prefer shorter completions, penalize weird whitespace,
    prefer higher logprob, etc — as long as it stays deterministic.
    """

    return (0, sample.completion)


def pick_best(
    example: Example,
    samples: list[RolloutSample],
    *,
    scorer: Callable[[Example, Any], dict[str, Any]] = default_score,
) -> SelectionResult:
    """Pick the best sample from a list.

    Sorting policy (default):
    - highest reward first
    - then tie_break_key(sample) (ascending)
    - then earliest index (ascending)

    Keep this function small. The point is for a reader to understand it in one glance.
    """

    if not samples:
        empty = RolloutSample(completion="")
        return SelectionResult(0, empty, scorer(example, empty.completion))

    scored_rows: list[tuple[float, tuple[int, str], int, RolloutSample, dict[str, Any]]] = []
    for i, s in enumerate(samples):
        out = scorer(example, s.completion)
        r = float(out.get("reward", 0.0))
        scored_rows.append((-r, tie_break_key(s), i, s, out))

    scored_rows.sort()
    _neg_r, _tb, best_i, best_s, best_out = scored_rows[0]
    return SelectionResult(best_i, best_s, best_out)
