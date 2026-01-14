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

        This is strict about required fields but raises friendly ValueErrors.
        """
        missing = [k for k in ("id", "prompt", "expected_answer") if k not in record]
        if missing:
            raise ValueError(f"Example record missing required fields: {missing}. Record={dict(record)!r}")

        ex_id = str(record["id"])
        prompt = str(record["prompt"])

        expected_raw = record["expected_answer"]
        try:
            expected_int = int(expected_raw)
        except Exception as e:
            raise ValueError(
                f"expected_answer must be int-coercible. Got {expected_raw!r} (type={type(expected_raw)})."
            ) from e

        meta = dict(record.get("metadata", {}))
        return Example(id=ex_id, prompt=prompt, expected_answer=expected_int, metadata=meta)


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One sampled completion (a.k.a. a rollout fragment).

    This repo keeps rollouts simple: single-turn completions.

    Optional fields are included to make the artifacts look more like production:
    - `sum_logprob`: sum of token logprobs under the current policy
    - `sum_ref_logprob`: sum of token logprobs under a reference policy

    If both are present, you can estimate per-sample KL as:
        kl_est = sum_logprob - sum_ref_logprob

    (This is a common bookkeeping identity in RLHF/RLVR pipelines.)
    """

    completion: str
    sum_logprob: Optional[float] = None
    sum_ref_logprob: Optional[float] = None
    meta: JsonDict = field(default_factory=dict)

    def kl_estimate(self) -> Optional[float]:
        if self.sum_logprob is None or self.sum_ref_logprob is None:
            return None
        return float(self.sum_logprob) - float(self.sum_ref_logprob)
