from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from course.core.io import read_jsonl
from course.core.types import RolloutSample


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _first_present(obj: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in obj:
            return obj.get(k)
    return None


def coerce_sample(obj: Any) -> RolloutSample:
    """Coerce a JSON-ish object into a RolloutSample.

    Accepted formats:
    - string: "Final: 123"
    - dict: {
        "completion": "Final: 123",
        "sum_logprob": -12.3,   # or "logprob" / "total_logprob"
        "sum_ref_logprob": -13.1,  # or "ref_logprob" / "total_ref_logprob"
        ...
      }
    """

    if isinstance(obj, str):
        return RolloutSample(completion=obj)
    if obj is None:
        return RolloutSample(completion="")

    if isinstance(obj, dict):
        comp = obj.get("completion")
        completion = "" if comp is None else str(comp)

        # Accept a few common names. The harness only uses these as optional
        # bookkeeping (e.g., KL estimates), never as part of the reward.
        sum_logprob = _coerce_float(_first_present(obj, ["sum_logprob", "logprob", "total_logprob"]))
        sum_ref_logprob = _coerce_float(_first_present(obj, ["sum_ref_logprob", "ref_logprob", "total_ref_logprob"]))

        strip_keys = {
            "completion",
            "sum_logprob",
            "logprob",
            "total_logprob",
            "sum_ref_logprob",
            "ref_logprob",
            "total_ref_logprob",
        }
        return RolloutSample(
            completion=completion,
            sum_logprob=sum_logprob,
            sum_ref_logprob=sum_ref_logprob,
            meta={k: v for k, v in obj.items() if k not in strip_keys},
        )

    # Fallback: string coercion, but keep it explicit.
    return RolloutSample(completion=str(obj))


def load_frozen_rollouts(path: Path) -> dict[str, RolloutSample]:
    """Load a JSONL file mapping example id -> completion.

    Each line must be an object with:
    - id: str
    - completion: str (optional; missing/None becomes empty string)

    Optional fields (names accepted):
    - sum_logprob / logprob / total_logprob
    - sum_ref_logprob / ref_logprob / total_ref_logprob
    """

    records = read_jsonl(path)
    m: dict[str, RolloutSample] = {}
    for rec in records:
        if "id" not in rec:
            raise ValueError(f"Completion record missing 'id' in {path}: {rec!r}")
        ex_id = str(rec["id"])
        if ex_id in m:
            raise ValueError(f"Duplicate completion id {ex_id!r} in {path}")

        sample = coerce_sample({k: v for k, v in rec.items() if k != "id"})
        m[ex_id] = sample
    return m


def load_selection_pack(path: Path) -> dict[str, list[RolloutSample]]:
    """Load a selection pack JSONL: id -> list of samples.

    Each line:
      {"id": "...", "samples": ["Final: 123", {"completion": "Final: 123", ...}, ...]}
    """

    records = read_jsonl(path)
    m: dict[str, list[RolloutSample]] = {}
    for rec in records:
        if "id" not in rec:
            raise ValueError(f"Selection pack record missing 'id' in {path}: {rec!r}")
        ex_id = str(rec["id"])
        if ex_id in m:
            raise ValueError(f"Duplicate selection-pack id {ex_id!r} in {path}")

        samples_obj = rec.get("samples")
        if not isinstance(samples_obj, list):
            raise ValueError(
                f"Selection pack record must include list 'samples'. id={ex_id!r} rec={rec!r}"
            )

        m[ex_id] = [coerce_sample(s) for s in samples_obj]
    return m
