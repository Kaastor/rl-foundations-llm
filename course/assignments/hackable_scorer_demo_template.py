from __future__ import annotations

import argparse


def score_naive(expected: int, completion: str) -> float:
    """Naive verifier: reward if expected integer appears anywhere."""
    # TODO: implement a naive substring or regex check.
    raise NotImplementedError


def score_patched(expected: int, completion: str) -> float:
    """Patched verifier: enforce exact 'Final: <int>' with strict integer rules."""
    # TODO: implement strict format enforcement (no extra text, no leading zeros, no -0).
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Hackable scorer demo.")
    p.add_argument("--expected", type=int, default=22)
    p.add_argument("--naive", action="store_true", help="Use naive substring verifier.")
    args = p.parse_args()

    completions = [
        "Final: 22",
        "The answer is 22.",
        "Final:  22",
        "I think 22 is right. Final: 0",
        "Final: 022",
    ]

    scorer = score_naive if args.naive else score_patched
    for c in completions:
        r = scorer(args.expected, c)
        print(f"{c!r} -> reward={r}")


if __name__ == "__main__":
    main()
