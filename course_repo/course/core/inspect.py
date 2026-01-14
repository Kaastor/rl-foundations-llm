from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from course.core.io import read_jsonl


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    run_dir: Path
    results_path: Path
    summary_path: Optional[Path]


def resolve_run(path: Path) -> ResolvedRun:
    """Resolve a user-provided path into (run_dir, results.jsonl, summary.json)."""
    path = path.expanduser().resolve()

    if path.is_dir():
        results = path / "results.jsonl"
        if not results.exists():
            raise FileNotFoundError(f"Run dir {path} does not contain results.jsonl")
        summary = path / "summary.json"
        return ResolvedRun(run_dir=path, results_path=results, summary_path=summary if summary.exists() else None)

    if path.is_file():
        if path.name == "results.jsonl":
            run_dir = path.parent
            summary = run_dir / "summary.json"
            return ResolvedRun(run_dir=run_dir, results_path=path, summary_path=summary if summary.exists() else None)
        raise ValueError("Please provide a run directory or a results.jsonl file.")

    raise FileNotFoundError(f"Path not found: {path}")


def load_summary(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_mode(first_record: Mapping[str, Any]) -> str:
    """Infer which script wrote the results.jsonl."""
    if "reward" in first_record and "details" in first_record:
        return "eval"
    if "baseline" in first_record and "best_of_n" in first_record:
        return "selection"
    return "unknown"


def eval_outcome_code(rec: Mapping[str, Any]) -> str:
    # Prefer explicit field if present.
    if "outcome_code" in rec:
        return str(rec.get("outcome_code"))

    try:
        reward = float(rec.get("reward", 0.0))
    except Exception:
        reward = 0.0

    if reward == 1.0:
        return "ok"

    details = rec.get("details") or {}
    result = details.get("result") or {}
    code = result.get("code")
    if code:
        return str(code)

    parse = details.get("parse") or {}
    pcode = parse.get("error_code")
    if pcode:
        return str(pcode)

    return "unknown"


def analyze_eval(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    n_pass = sum(1 for r in records if float(r.get("reward", 0.0)) == 1.0)
    n_fail = n - n_pass

    by_code: Dict[str, list[Dict[str, Any]]] = {}
    for r in records:
        code = eval_outcome_code(r)
        by_code.setdefault(code, []).append(r)

    items = sorted(by_code.items(), key=lambda kv: len(kv[1]), reverse=True)
    if "ok" in by_code:
        items = [("ok", by_code["ok"])] + [(k, v) for k, v in items if k != "ok"]

    return {
        "mode": "eval",
        "n": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate": (n_pass / n) if n else 0.0,
        "by_code": {k: len(v) for k, v in items},
        "groups": by_code,
    }


def analyze_selection(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    rescued = 0
    baseline_pass = 0
    best_pass = 0
    rescue_examples: list[Dict[str, Any]] = []

    for r in records:
        b = r.get("baseline") or {}
        bn = float(b.get("reward", 0.0) or 0.0)
        bo = r.get("best_of_n") or {}
        br = float(bo.get("reward", 0.0) or 0.0)

        if bn == 1.0:
            baseline_pass += 1
        if br == 1.0:
            best_pass += 1
        if bn == 0.0 and br == 1.0:
            rescued += 1
            rescue_examples.append(r)

    return {
        "mode": "selection",
        "n": n,
        "pass_at_1": baseline_pass / n if n else 0.0,
        "pass_at_n": best_pass / n if n else 0.0,
        "rescued": rescued,
        "rescue_examples": rescue_examples,
    }


def analyze_run(run: ResolvedRun) -> Dict[str, Any]:
    records = read_jsonl(run.results_path)
    if not records:
        return {"mode": "empty", "n": 0}

    mode = infer_mode(records[0])
    if mode == "eval":
        return analyze_eval(records)
    if mode == "selection":
        return analyze_selection(records)
    return {"mode": "unknown", "n": len(records)}
