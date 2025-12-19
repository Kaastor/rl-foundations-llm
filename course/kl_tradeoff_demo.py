from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from course.io_utils import make_run_dir, write_json, atomic_write_text, utc_now_iso


def normalize(probs: Sequence[float]) -> list[float]:
    s = sum(probs)
    if s <= 0:
        raise ValueError("Probabilities must sum to > 0")
    return [p / s for p in probs]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) for categorical distributions (natural log)."""
    out = 0.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        out += pi * math.log(pi / qi)
    return out


def expected_reward(p: Sequence[float], r: Sequence[float]) -> float:
    return sum(pi * ri for pi, ri in zip(p, r))


def constrained_optimal_policy(pi_ref: Sequence[float], rewards: Sequence[float], beta: float) -> list[float]:
    """Closed-form optimum for: max_pi E_pi[r] - beta * KL(pi || pi_ref).

    For beta > 0:
        pi*(a) ∝ pi_ref(a) * exp(r(a) / beta)

    This is the simplest “KL constraint” mental model.
    """
    if beta <= 0:
        # In the limit beta -> 0+, this collapses to argmax reward while respecting support.
        # For pedagogy, we clamp to a tiny beta.
        beta = 1e-8

    unnorm = [pref * math.exp(r / beta) for pref, r in zip(pi_ref, rewards)]
    return normalize(unnorm)


def run_demo(out_dir: Path, *, plot: bool) -> dict[str, Any]:
    actions = [f"a{i}" for i in range(6)]
    # Reference policy: a mild bias toward a0/a1 (think: “pretrained prior”).
    pi_ref = normalize([0.30, 0.25, 0.15, 0.10, 0.10, 0.10])

    # Rewards: the environment/verifier prefers later actions.
    rewards = [0.0, 0.1, 0.4, 0.6, 0.8, 1.0]

    betas = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]

    rows: list[dict[str, Any]] = []
    for beta in betas:
        pi = constrained_optimal_policy(pi_ref, rewards, beta)
        kl = kl_divergence(pi, pi_ref)
        er = expected_reward(pi, rewards)
        top3 = sorted(zip(actions, pi), key=lambda t: t[1], reverse=True)[:3]
        rows.append(
            {
                "beta": beta,
                "expected_reward": er,
                "kl": kl,
                "top3": [{"action": a, "p": p} for a, p in top3],
                "policy": pi,
            }
        )

    # Save CSV (easy to chart in any tool)
    csv_path = out_dir / "kl_tradeoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beta", "expected_reward", "kl"])
        for r in rows:
            w.writerow([r["beta"], r["expected_reward"], r["kl"]])

    summary = {
        "run": {"created_utc": utc_now_iso(), "script": "kl_tradeoff_demo"},
        "reference_policy": {"actions": actions, "pi_ref": pi_ref},
        "rewards": rewards,
        "rows": rows,
        "csv": str(csv_path),
    }
    write_json(out_dir / "summary.json", summary)

    md = []
    md.append("# KL tradeoff demo (categorical policy)\n\n")
    md.append("We optimize: `E_pi[r] - beta * KL(pi || pi_ref)`\n\n")
    md.append("Interpretation for LLM RL:\n")
    md.append("- `pi_ref` ≈ reference model (pretrained/SFT)\n")
    md.append("- KL penalty discourages drifting into weird space\n")
    md.append("- smaller beta → more reward-seeking, more drift\n\n")
    md.append("See `kl_tradeoff.csv` for the reward/KL curve.\n")
    atomic_write_text(out_dir / "summary.md", "".join(md))

    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            print("matplotlib not available; skipping plot.")
        else:
            xs = [r["kl"] for r in rows]
            ys = [r["expected_reward"] for r in rows]
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("KL(pi || pi_ref)")
            plt.ylabel("Expected reward under pi")
            plt.title("KL–Reward tradeoff (beta sweep)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            png_path = out_dir / "kl_tradeoff.png"
            plt.savefig(png_path)
            print(f"Wrote plot: {png_path}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="KL-constrained policy improvement demo (no LLMs)." )
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/kl_<timestamp>)")
    p.add_argument("--plot", action="store_true", help="Write a kl_tradeoff.png (requires matplotlib)")
    args = p.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="kl")

    run_demo(out_dir, plot=args.plot)
    print(f"Wrote KL demo outputs to: {out_dir}")


if __name__ == "__main__":
    main()
