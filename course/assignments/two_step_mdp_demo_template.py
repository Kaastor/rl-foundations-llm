from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# Toggle for the credit assignment sabotage.
UPDATE_STEP1 = True


@dataclass
class PolicyState:
    logits_step1: List[float]
    logits_step2_a: List[float]
    logits_step2_b: List[float]


def softmax(logits: List[float]) -> List[float]:
    # TODO: implement a numerically stable softmax.
    raise NotImplementedError


def sample_action(rng: random.Random, actions: List[str], probs: List[float]) -> Tuple[str, int]:
    # TODO: sample an action index based on probs.
    raise NotImplementedError


def update_logits(logits: List[float], probs: List[float], action_idx: int, advantage: float, lr: float) -> None:
    # TODO: implement REINFORCE-style update.
    raise NotImplementedError


def run(steps: int, seed: int, lr: float, use_baseline: bool, outdir: Path) -> Dict[str, float]:
    rng = random.Random(seed)
    state = PolicyState(logits_step1=[0.0, 0.0], logits_step2_a=[0.0, 0.0], logits_step2_b=[0.0, 0.0])

    actions1 = ["A", "B"]
    actions2 = ["X", "Y"]

    baseline = 0.0
    total_reward = 0.0
    rows = []

    for t in range(1, steps + 1):
        # TODO: sample step-1 action from policy.
        # TODO: sample step-2 action conditioned on step-1.
        # TODO: compute terminal reward.
        # TODO: compute advantage and update policies.
        raise NotImplementedError

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "traj.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "steps": steps,
        "mean_reward": total_reward / steps if steps else 0.0,
        "update_step1": UPDATE_STEP1,
        "baseline": use_baseline,
        "seed": seed,
        "lr": lr,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Two-step MDP REINFORCE demo.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    summary = run(args.steps, args.seed, args.lr, args.baseline, args.outdir)
    print(f"Wrote results to: {args.outdir}")
    print(f"mean_reward={summary['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
