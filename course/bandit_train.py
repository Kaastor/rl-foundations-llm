from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from course.io_utils import make_run_dir, write_json, write_jsonl, atomic_write_text, utc_now_iso


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_categorical(probs: Sequence[float], rng) -> int:
    """Sample an index from a categorical distribution."""
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1  # numerical edge


@dataclass
class BanditEnv:
    """A tiny contextual-bandit-like environment.

    This is NOT the course math task. It's a microscope for the REINFORCE mechanism.

    Each action `a` yields reward 1 with probability `p[a]`, else 0.
    """

    action_names: list[str]
    reward_probs: list[float]

    def reward(self, action_idx: int, rng) -> float:
        p = self.reward_probs[action_idx]
        return 1.0 if rng.random() < p else 0.0


def reinforce_step(
    theta: list[float],
    action_idx: int,
    probs: list[float],
    advantage: float,
    lr: float,
) -> list[float]:
    """One REINFORCE update for a categorical softmax policy.

    For a selected action a:
        grad log pi(a) = one_hot(a) - pi

    Update:
        theta <- theta + lr * advantage * (one_hot(a) - pi)

    This is exactly the “increase log-prob of sampled actions if advantage>0” story.
    """
    k = len(theta)
    updated = theta[:]
    for i in range(k):
        grad_i = (1.0 if i == action_idx else 0.0) - probs[i]
        updated[i] = theta[i] + lr * advantage * grad_i
    return updated


def run_bandit_train(
    *,
    steps: int,
    seed: int,
    lr: float,
    use_baseline: bool,
    slow: bool,
    out_dir: Path,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)

    env = BanditEnv(
        action_names=["A", "B", "C", "D"],
        reward_probs=[0.10, 0.25, 0.80, 0.05],  # action C is best
    )

    theta = [0.0 for _ in env.action_names]
    baseline = 0.0
    n = 0

    logs: list[dict[str, Any]] = []
    rewards: list[float] = []

    if slow:
        print("Advantage sign intuition: A = reward - baseline")
        print("- If A > 0: increase probability of sampled action (relative to others)")
        print("- If A < 0: decrease probability of sampled action (relative to others)\n")

    for t in range(1, steps + 1):
        probs = softmax(theta)
        action_idx = sample_categorical(probs, rng)
        reward = env.reward(action_idx, rng)

        # Running-mean baseline (simple, deterministic, no critic).
        if use_baseline:
            n += 1
            baseline = baseline + (reward - baseline) / n
        else:
            baseline = 0.0

        advantage = reward - baseline

        # Update.
        theta_new = reinforce_step(theta, action_idx, probs, advantage, lr)
        theta = theta_new

        rewards.append(reward)

        log = {
            "step": t,
            "theta": [round(x, 6) for x in theta],
            "probs": [round(p, 6) for p in probs],
            "action_idx": action_idx,
            "action": env.action_names[action_idx],
            "reward": reward,
            "baseline": round(baseline, 6),
            "advantage": round(advantage, 6),
        }
        logs.append(log)

        if slow:
            probs_str = ", ".join(f"{name}:{p:.3f}" for name, p in zip(env.action_names, probs))
            print(f"step {t:03d} | pi=({probs_str}) | a={env.action_names[action_idx]} | r={reward:.0f} | b={baseline:.3f} | A={advantage:.3f}")

    # Summary metrics
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    last_k = min(100, len(rewards))
    mean_last = sum(rewards[-last_k:]) / last_k if last_k else 0.0

    summary = {
        "run": {
            "created_utc": utc_now_iso(),
            "script": "bandit_train",
            "seed": seed,
            "steps": steps,
            "lr": lr,
            "baseline": use_baseline,
        },
        "metrics": {
            "mean_reward": mean_reward,
            f"mean_reward_last_{last_k}": mean_last,
        },
        "env": {
            "action_names": env.action_names,
            "reward_probs": env.reward_probs,
        },
    }

    write_jsonl(out_dir / "log.jsonl", logs)
    write_json(out_dir / "summary.json", summary)

    md = []
    md.append("# Bandit training demo (REINFORCE, categorical policy)\n\n")
    md.append(f"- Created (UTC): `{summary['run']['created_utc']}`\n")
    md.append(f"- Steps: `{steps}`\n")
    md.append(f"- Seed: `{seed}`\n")
    md.append(f"- Learning rate: `{lr}`\n")
    md.append(f"- Baseline: `{use_baseline}`\n")
    md.append("\n## Metrics\n")
    md.append(f"- mean_reward: **{mean_reward:.3f}**\n")
    md.append(f"- mean_reward_last_{last_k}: **{mean_last:.3f}**\n")
    md.append("\n## Interpretation\n")
    md.append(
        "This script is a microscope for the mechanism: probabilities shift toward\n"
        "actions that yield higher reward. The baseline reduces variance; it does not\n"
        "change the expected update direction (it changes noise).\n"
    )
    atomic_write_text(out_dir / "summary.md", "".join(md))

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Toy policy-gradient demo: categorical bandit + REINFORCE.")
    p.add_argument("--steps", type=int, default=200, help="Training steps")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    p.add_argument("--baseline", action="store_true", help="Use a running-mean baseline")
    p.add_argument("--slow", action="store_true", help="Print step-by-step logs to stdout")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/bandit_<timestamp>)")
    args = p.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="bandit")

    summary = run_bandit_train(
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        use_baseline=args.baseline,
        slow=args.slow,
        out_dir=out_dir,
    )

    print(f"Wrote bandit logs to: {out_dir}")
    print(f"mean_reward={summary['metrics']['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
