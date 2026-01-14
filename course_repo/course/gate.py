from __future__ import annotations

import argparse
import json
from pathlib import Path

from course.core.gate import gate, load_run_stats


def main() -> None:
    p = argparse.ArgumentParser(description="Gate a candidate eval run vs a baseline (PROMOTE/REJECT).")
    p.add_argument("--baseline", type=Path, required=True, help="Baseline run directory (runs/...)")
    p.add_argument("--candidate", type=Path, required=True, help="Candidate run directory (runs/...)")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum pass_rate improvement required to PROMOTE")
    p.add_argument("--json", action="store_true", help="Print decision as JSON")
    args = p.parse_args()

    base = load_run_stats(args.baseline)
    cand = load_run_stats(args.candidate)

    decision = gate(baseline=base, candidate=cand, min_delta=args.min_delta)

    if args.json:
        print(json.dumps(decision, indent=2, sort_keys=True))
    else:
        print(f"Decision: {decision['decision']}")
        print(f"Baseline pass_rate:  {decision['baseline']['pass_rate']:.3f}  (n={decision['baseline']['n']})")
        print(f"Candidate pass_rate: {decision['candidate']['pass_rate']:.3f}  (n={decision['candidate']['n']})")
        print(f"Delta: {decision['delta']:.3f}")
        if decision["reasons"]:
            print("Reasons:")
            for r in decision["reasons"]:
                print(f"- {r}")

    raise SystemExit(0 if decision["decision"] == "PROMOTE" else 1)


if __name__ == "__main__":
    main()
