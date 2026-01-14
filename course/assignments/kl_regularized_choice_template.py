from __future__ import annotations


def choose(beta: float) -> dict[str, float | str]:
    candidates = [
        {"name": "c1", "reward": 1.0, "kl": 0.5},
        {"name": "c2", "reward": 0.9, "kl": 0.1},
        {"name": "c3", "reward": 1.2, "kl": 1.5},
        {"name": "c4", "reward": 0.7, "kl": 0.05},
        {"name": "c5", "reward": 1.1, "kl": 0.9},
        {"name": "c6", "reward": 0.8, "kl": 0.2},
    ]

    # TODO: choose the candidate that maximizes reward - beta * kl.
    # Add deterministic tie-breaking if scores are equal.
    return {}


def main() -> None:
    # TODO: print the best choice for beta=0.1 and beta=1.0.
    raise NotImplementedError


if __name__ == "__main__":
    main()
