from __future__ import annotations

import argparse
import sys
from pathlib import Path

from course.assignments.selection_policy import pick_best
from course.core.selection import run_selection_demo


def main() -> None:
    p = argparse.ArgumentParser(description="Loop B: Best-of-N selection using the deterministic verifier.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSONL")
    p.add_argument("--samples", type=Path, required=True, help="Path to selection pack JSONL (id -> samples)")
    p.add_argument("--n", type=int, default=None, help="Use only the first N samples per example")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/selection_<timestamp>)")
    p.add_argument("--max", type=int, default=None, dest="max_examples", help="Optional cap for quick runs")
    args = p.parse_args()

    out_dir, summary = run_selection_demo(
        dataset_path=args.dataset,
        samples_path=args.samples,
        pick_best=pick_best,
        out_dir=args.outdir,
        n=args.n,
        max_examples=args.max_examples,
        argv=sys.argv,
        args=vars(args),
    )

    print(f"Wrote results to: {out_dir}")
    print(f"pass@1={summary['metrics']['pass_at_1']:.3f}  pass@N={summary['metrics']['pass_at_n']:.3f}")


if __name__ == "__main__":
    main()
