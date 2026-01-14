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

    best = None
    for c in candidates:
        score = float(c["reward"]) - beta * float(c["kl"])
        row = dict(c)
        row["score"] = score
        if best is None:
            best = row
            continue
        if score > float(best["score"]):
            best = row
        elif score == float(best["score"]) and str(row["name"]) < str(best["name"]):
            best = row
    return best or {}


def main() -> None:
    betas = [0.1, 1.0]
    for beta in betas:
        best = choose(beta)
        print(f"beta={beta:.1f} -> {best['name']} (reward={best['reward']}, kl={best['kl']}, score={best['score']:.3f})")


if __name__ == "__main__":
    main()
