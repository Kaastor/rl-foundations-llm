from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from course.core.inspect import analyze_run, load_summary, resolve_run
from course.core.scoring import SCORER_NAME, SCORER_VERSION


def _truncate(s: str, limit: int = 160) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= limit:
        return s
    return s[:limit] + f"... <truncated {len(s) - limit} chars>"


def _print_header(run_dir: Path, summary: Optional[Dict[str, Any]]) -> None:
    print(f"Run directory: {run_dir}")
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

        if scorer.get("name") != SCORER_NAME or scorer.get("version") != SCORER_VERSION:
            print("  WARNING: This run used a different scorer version than the current code.")
            print(f"  current scorer: {SCORER_NAME} v{SCORER_VERSION}")
            print("  Locked Room Rule: don't compare metrics across scorer/spec versions.")

    for k in ("dataset_path", "completions_path", "samples_path"):
        if k in run_meta:
            print(f"- {k}: {run_meta[k]}")


def print_eval_report(analysis: Dict[str, Any], *, top_k: int, show: int, only_fails: bool, inspect_id: Optional[str]) -> None:
    groups: Dict[str, list[Dict[str, Any]]] = analysis["groups"]

    # Single-id inspection mode
    if inspect_id is not None:
        for code, recs in groups.items():
            for r in recs:
                if str(r.get("id")) == inspect_id:
                    print(f"\n=== Inspect id={inspect_id} (outcome_code={code}) ===\n")
                    r2 = dict(r)
                    comp = str(r2.get("completion", ""))
                    r2["completion"] = _truncate(comp, limit=800)
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

    by_code = analysis["by_code"]
    codes = [(c, by_code[c]) for c in by_code.keys()]
    if only_fails:
        codes = [(c, cnt) for c, cnt in codes if c != "ok"]

    print("\nTop outcome codes:")
    for code, cnt in codes[:top_k]:
        print(f"- {code}: {cnt}")

    printed = 0
    for code, _cnt in codes[:top_k]:
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
            msg = parse.get("error_message") or (details.get("result") or {}).get("message") or ""
            ans = parse.get("answer_str")
            print(f"- id={ex_id} expected={expected} parsed={ans!r}")
            print(f"  completion: {_truncate(completion)}")
            if msg:
                print(f"  why: {msg}")

        printed += 1

    if printed == 0 and only_fails:
        print("\nNo failures found (all ok).")


def print_selection_report(analysis: Dict[str, Any], *, show: int) -> None:
    print("\nMetrics (from results.jsonl):")
    print(f"- n: {analysis['n']}")
    print(f"- pass@1: {analysis['pass_at_1']:.3f}")
    print(f"- pass@N: {analysis['pass_at_n']:.3f}")
    print(f"- rescued (baseline fail -> best success): {analysis['rescued']}")

    rescues: list[Dict[str, Any]] = analysis["rescue_examples"]
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
    p = argparse.ArgumentParser(description="Inspect a run directory: group failures, print examples, and sanity-check scorer version.")
    p.add_argument("--run", type=Path, required=True, help="Run dir (runs/...) or a results.jsonl path")
    p.add_argument("--top-k", type=int, default=5, help="Top K outcome codes to display")
    p.add_argument("--show", type=int, default=3, help="How many examples to show per group")
    p.add_argument("--only-fails", action="store_true", help="Hide the 'ok' group and focus on failures")
    p.add_argument("--id", type=str, default=None, help="Inspect a single example id (prints its full record)")
    args = p.parse_args()

    run = resolve_run(args.run)
    summary = load_summary(run.summary_path)

    _print_header(run.run_dir, summary)

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
