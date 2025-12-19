from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Example:
    """A single dataset item for the course.

    The anchor task is intentionally narrow:
    - the prompt describes a math problem
    - the expected answer is a base-10 integer
    - the scorer verifies the completion against expected_answer (it does not solve math)
    """

    id: str
    prompt: str
    expected_answer: int
    metadata: JsonDict = field(default_factory=dict)

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "Example":
        """Parse an Example from a dict-like record.

        This is strict about required fields but will never raise a cryptic exception:
        it converts obvious type issues into a ValueError with context.
        """
        missing = [k for k in ("id", "prompt", "expected_answer") if k not in record]
        if missing:
            raise ValueError(f"Example record missing required fields: {missing}. Record={dict(record)!r}")

        ex_id = str(record["id"])
        prompt = str(record["prompt"])

        expected_raw = record["expected_answer"]
        try:
            expected_int = int(expected_raw)
        except Exception as e:  # pragma: no cover (defensive)
            raise ValueError(
                f"expected_answer must be int-coercible. Got {expected_raw!r} (type={type(expected_raw)})."
            ) from e

        meta = dict(record.get("metadata", {}))
        return Example(id=ex_id, prompt=prompt, expected_answer=expected_int, metadata=meta)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Return type for scoring.

    The public scorer API returns a plain dict:
        {"reward": float, "details": dict}

    But internally we sometimes use this dataclass for structure.
    """

    reward: float
    details: JsonDict


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Minimal run metadata saved alongside outputs."""

    run_id: str
    created_utc: str
    scorer_name: str
    scorer_version: str
    dataset_path: str
    extra: JsonDict = field(default_factory=dict)
