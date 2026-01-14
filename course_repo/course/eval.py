from __future__ import annotations

import argparse
import sys
from pathlib import Path

from course.core.eval import run_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Loop A: evaluate frozen rollouts using the deterministic verifier.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSONL")
    p.add_argument("--completions", type=Path, required=True, help="Path to completions JSONL (id -> completion)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/eval_<timestamp>)")
    p.add_argument("--max", type=int, default=None, dest="max_examples", help="Optional cap for quick runs")
    args = p.parse_args()

    out_dir, summary = run_eval(
        dataset_path=args.dataset,
        completions_path=args.completions,
        out_dir=args.outdir,
        max_examples=args.max_examples,
        argv=sys.argv,
        args=vars(args),
    )

    print(f"Wrote results to: {out_dir}")
    print(f"pass_rate={summary['metrics']['pass_rate']:.3f}  n={summary['run']['n_examples']}")
    if summary["run"]["n_missing_completions"]:
        print(f"WARNING: missing completions for {summary['run']['n_missing_completions']} examples")


if __name__ == "__main__":
    main()
