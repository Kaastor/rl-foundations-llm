from __future__ import annotations

import argparse
import re


def score_naive(expected: int, completion: str) -> float:
    """Naive verifier: reward if expected integer appears anywhere."""
    return 1.0 if str(expected) in completion else 0.0


def score_patched(expected: int, completion: str) -> float:
    """Patched verifier: enforce exact 'Final: <int>' with strict integer rules."""
    m = re.fullmatch(r"Final: (-?\d+)", completion.strip())
    if not m:
        return 0.0
    value_str = m.group(1)
    if value_str.startswith("+"):
        return 0.0
    if value_str == "-0":
        return 0.0
    if value_str.startswith("-"):
        digits = value_str[1:]
        if len(digits) > 1 and digits.startswith("0"):
            return 0.0
    else:
        if len(value_str) > 1 and value_str.startswith("0"):
            return 0.0
    return 1.0 if int(value_str) == expected else 0.0


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
