from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from course.io_utils import read_jsonl
from course.score import SCORER_NAME, SCORER_VERSION


# -----------------------------------------------------------------------------
# Goal
# -----------------------------------------------------------------------------
# This script is intentionally small and "boringly helpful":
# - read a run directory (runs/<something>/)
# - summarize failures in an inspectable way
# - print a few concrete examples per failure type
#
# It's built for the course's core habit:
# "Don't trust a number until you can point at the raw artifacts and explain failures."


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    run_dir: Path
    results_path: Path
    summary_path: Optional[Path]


def _truncate(s: str, limit: int = 160) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= limit:
        return s
    return s[:limit] + f"... <truncated {len(s) - limit} chars>"


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


def _load_summary(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_mode(first_record: Mapping[str, Any]) -> str:
    """Infer which script wrote the results.jsonl."""
    if "reward" in first_record and "details" in first_record:
        return "eval"
    if "baseline" in first_record and "best_of_n" in first_record:
        return "selection"
    return "unknown"


def _error_code_for_eval_record(rec: Mapping[str, Any]) -> str:
    try:
        reward = float(rec.get("reward", 0.0))
    except Exception:
        reward = 0.0

    if reward == 1.0:
        return "ok"

    details = rec.get("details") or {}
    parse = details.get("parse") or {}
    code = parse.get("error_code")
    if code:
        return str(code)
    return "unknown"


def analyze_eval(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    n_pass = sum(1 for r in records if float(r.get("reward", 0.0)) == 1.0)
    n_fail = n - n_pass

    by_code: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        code = _error_code_for_eval_record(r)
        by_code.setdefault(code, []).append(r)

    # Order by count desc, but keep "ok" first if present.
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


def analyze_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Selection results are structured differently; summarize 'rescues'."""
    n = len(records)
    rescued = 0
    baseline_pass = 0
    best_pass = 0
    rescue_examples: list[dict[str, Any]] = []

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


def analyze_run(run: ResolvedRun) -> dict[str, Any]:
    records = read_jsonl(run.results_path)
    if not records:
        return {"mode": "empty", "n": 0}

    mode = _infer_mode(records[0])
    if mode == "eval":
        return analyze_eval(records)
    if mode == "selection":
        return analyze_selection(records)
    return {"mode": "unknown", "n": len(records)}


def _print_header(run: ResolvedRun, summary: Optional[dict[str, Any]]) -> None:
    print(f"Run directory: {run.run_dir}")
    print(f"Results file:  {run.results_path}")
    if run.summary_path and run.summary_path.exists():
        print(f"Summary file:  {run.summary_path}")

    if not summary:
        print("\n(No readable summary.json found.)")
        return

    run_meta = summary.get("run") or {}
    created = run_meta.get("created_utc")
    scorer = run_meta.get("scorer") or {}

    print("\nRun metadata:")
    if created:
        print(f"- created_utc: {created}")
    if scorer:
        print(f"- scorer: {scorer.get('name')} v{scorer.get('version')}")

        # Spec drift warning
        if scorer.get("name") != SCORER_NAME or scorer.get("version") != SCORER_VERSION:
            print("  WARNING: This run used a different scorer version than the current code.")
            print(f"  current scorer: {SCORER_NAME} v{SCORER_VERSION}")
            print("  Locked Room Rule: don't compare metrics across scorer/spec versions.")

    # Print a few common fields if present
    for k in ("dataset_path", "completions_path", "samples_path"):
        if k in run_meta:
            print(f"- {k}: {run_meta[k]}")


def print_eval_report(analysis: dict[str, Any], *, top_k: int, show: int, only_fails: bool, inspect_id: Optional[str]) -> None:
    groups: dict[str, list[dict[str, Any]]] = analysis["groups"]

    # Single-id inspection mode
    if inspect_id is not None:
        for code, recs in groups.items():
            for r in recs:
                if str(r.get("id")) == inspect_id:
                    print(f"\n=== Inspect id={inspect_id} (error_code={code}) ===\n")
                    # Pretty JSON, but avoid huge outputs.
                    completion = str(r.get("completion", ""))
                    r2 = dict(r)
                    r2["completion"] = _truncate(completion, limit=800)
                    details = r2.get("details") or {}
                    # Keep details but truncate completion previews inside it if present.
                    comp = (details.get("completion") or {})
                    if "raw_preview" in comp:
                        comp["raw_preview"] = _truncate(str(comp["raw_preview"]), limit=400)
                    if "normalized_preview" in comp:
                        comp["normalized_preview"] = _truncate(str(comp["normalized_preview"]), limit=400)
                    details["completion"] = comp
                    r2["details"] = details
                    print(json.dumps(r2, indent=2, ensure_ascii=False, sort_keys=True))
                    return
        print(f"\nNo record found with id={inspect_id!r}")
        return

    n = analysis["n"]
    n_pass = analysis["n_pass"]
    n_fail = analysis["n_fail"]
    pass_rate = analysis["pass_rate"]

    print("\nMetrics (from results.jsonl):")
    print(f"- n: {n}")
    print(f"- pass_rate: {pass_rate:.3f}  (pass={n_pass}, fail={n_fail})")

    # Error code distribution
    by_code = analysis["by_code"]
    codes = [(c, by_code[c]) for c in by_code.keys()]
    if only_fails:
        codes = [(c, cnt) for c, cnt in codes if c != "ok"]

    print("\nTop error codes:")
    for code, cnt in codes[:top_k]:
        print(f"- {code}: {cnt}")

    # Print examples for each top failure code
    printed = 0
    for code, cnt in codes[:top_k]:
        if code == "ok":
            continue
        recs = groups.get(code, [])
        if not recs:
            continue

        print(f"\n=== {code} (showing up to {show}) ===")
        for r in recs[:show]:
            ex_id = r.get("id")
            expected = r.get("expected_answer")
            completion = str(r.get("completion", ""))
            details = r.get("details") or {}
            parse = details.get("parse") or {}
            msg = parse.get("error_message") or ""
            ans = parse.get("answer_str")
            print(f"- id={ex_id} expected={expected} parsed={ans!r}")
            print(f"  completion: {_truncate(completion)}")
            if msg:
                print(f"  why: {msg}")

        printed += 1

    if printed == 0 and only_fails:
        print("\nNo failures found (all ok).")


def print_selection_report(analysis: dict[str, Any], *, show: int) -> None:
    print("\nMetrics (from results.jsonl):")
    print(f"- n: {analysis['n']}")
    print(f"- pass@1: {analysis['pass_at_1']:.3f}")
    print(f"- pass@N: {analysis['pass_at_n']:.3f}")
    print(f"- rescued (baseline fail -> best success): {analysis['rescued']}")

    rescues: list[dict[str, Any]] = analysis["rescue_examples"]
    if rescues:
        print(f"\n=== Rescue examples (showing up to {show}) ===")
        for r in rescues[:show]:
            ex_id = r.get("id")
            expected = r.get("expected_answer")
            b = r.get("baseline") or {}
            bo = r.get("best_of_n") or {}
            print(f"- id={ex_id} expected={expected}")
            print(f"  baseline: { _truncate(str(b.get('completion',''))) }")
            print(f"  best:     { _truncate(str(bo.get('completion',''))) }")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect a run directory: group failures, print examples, and sanity-check scorer version."
    )
    p.add_argument("--run", type=Path, required=True, help="Run dir (runs/...) or a results.jsonl path")
    p.add_argument("--top-k", type=int, default=5, help="Top K error codes to display")
    p.add_argument("--show", type=int, default=3, help="How many examples to show per group")
    p.add_argument("--only-fails", action="store_true", help="Hide the 'ok' group and focus on failures")
    p.add_argument("--id", type=str, default=None, help="Inspect a single example id (prints its full record)")
    args = p.parse_args()

    run = resolve_run(args.run)
    summary = _load_summary(run.summary_path)

    _print_header(run, summary)

    analysis = analyze_run(run)
    mode = analysis.get("mode")

    if mode == "eval":
        print_eval_report(
            analysis,
            top_k=args.top_k,
            show=args.show,
            only_fails=args.only_fails,
            inspect_id=args.id,
        )
    elif mode == "selection":
        if args.id is not None:
            print("Single-id inspection is currently supported only for eval-mode results.")
        print_selection_report(analysis, show=args.show)
    elif mode == "empty":
        print("\nRun contained no records.")
    else:
        print(f"\nUnknown results format. Found {analysis.get('n')} records, but couldn't infer mode.")


if __name__ == "__main__":
    main()
