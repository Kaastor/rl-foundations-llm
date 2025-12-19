from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from course.core.io import read_jsonl


@dataclass(frozen=True, slots=True)
class RunStats:
    run_dir: Path
    created_utc: str
    dataset_path: str
    scorer_name: str
    scorer_version: str
    n_examples: int
    pass_rate: float
    outcome_counts: Dict[str, int]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _outcome_code(rec: Dict[str, Any]) -> str:
    if "outcome_code" in rec:
        return str(rec.get("outcome_code") or "unknown")
    details = rec.get("details") or {}
    result = details.get("result") or {}
    code = result.get("code")
    if code:
        return str(code)
    parse = details.get("parse") or {}
    p = parse.get("error_code")
    if p:
        return str(p)
    return "unknown"


def load_run_stats(run_dir: Path) -> RunStats:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Run dir must contain summary.json and results.jsonl: {run_dir}")

    summary = _read_json(summary_path)
    run = summary.get("run") or {}
    scorer = run.get("scorer") or {}

    created_utc = str(run.get("created_utc") or "")
    dataset_path = str(run.get("dataset_path") or "")
    scorer_name = str(scorer.get("name") or "")
    scorer_version = str(scorer.get("version") or "")

    records = read_jsonl(results_path)
    n = len(records)
    rewards = [float(r.get("reward", 0.0) or 0.0) for r in records]
    pass_rate = (sum(rewards) / n) if n else 0.0

    counts: Dict[str, int] = {}
    for r in records:
        c = _outcome_code(r)
        counts[c] = counts.get(c, 0) + 1

    return RunStats(
        run_dir=run_dir,
        created_utc=created_utc,
        dataset_path=dataset_path,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
        n_examples=n,
        pass_rate=pass_rate,
        outcome_counts=counts,
    )


def gate(
    *,
    baseline: RunStats,
    candidate: RunStats,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    """Production-style gate: PROMOTE / REJECT with reasons.

    This is intentionally simple:
    - enforce Locked Room Rule (dataset + scorer)
    - compare pass_rate
    """

    reasons: list[str] = []

    if baseline.scorer_name != candidate.scorer_name or baseline.scorer_version != candidate.scorer_version:
        reasons.append(
            "LockedRoomViolation: scorer mismatch "
            f"(baseline={baseline.scorer_name} v{baseline.scorer_version}, candidate={candidate.scorer_name} v{candidate.scorer_version})"
        )

    if baseline.dataset_path and candidate.dataset_path and baseline.dataset_path != candidate.dataset_path:
        reasons.append(
            "LockedRoomViolation: dataset_path mismatch "
            f"(baseline={baseline.dataset_path}, candidate={candidate.dataset_path})"
        )

    if baseline.n_examples != candidate.n_examples:
        reasons.append(
            "LockedRoomViolation: n_examples mismatch "
            f"(baseline={baseline.n_examples}, candidate={candidate.n_examples})"
        )

    delta = candidate.pass_rate - baseline.pass_rate

    decision: str
    if reasons:
        decision = "REJECT"
    else:
        decision = "PROMOTE" if delta >= min_delta else "REJECT"
        if decision == "REJECT":
            reasons.append(f"DeltaTooSmall: delta={delta:.4f} < min_delta={min_delta:.4f}")

    return {
        "decision": decision,
        "reasons": reasons,
        "baseline": {
            "run_dir": str(baseline.run_dir),
            "created_utc": baseline.created_utc,
            "pass_rate": baseline.pass_rate,
            "n": baseline.n_examples,
        },
        "candidate": {
            "run_dir": str(candidate.run_dir),
            "created_utc": candidate.created_utc,
            "pass_rate": candidate.pass_rate,
            "n": candidate.n_examples,
        },
        "delta": delta,
    }
