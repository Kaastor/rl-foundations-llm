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
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sample_action(rng: random.Random, actions: List[str], probs: List[float]) -> Tuple[str, int]:
    r = rng.random()
    total = 0.0
    for i, p in enumerate(probs):
        total += p
        if r <= total:
            return actions[i], i
    return actions[-1], len(actions) - 1


def update_logits(logits: List[float], probs: List[float], action_idx: int, advantage: float, lr: float) -> None:
    for i in range(len(logits)):
        grad = (1.0 if i == action_idx else 0.0) - probs[i]
        logits[i] += lr * advantage * grad


def run(steps: int, seed: int, lr: float, use_baseline: bool, outdir: Path) -> Dict[str, float]:
    rng = random.Random(seed)
    state = PolicyState(logits_step1=[0.0, 0.0], logits_step2_a=[0.0, 0.0], logits_step2_b=[0.0, 0.0])

    actions1 = ["A", "B"]
    actions2 = ["X", "Y"]

    baseline = 0.0
    total_reward = 0.0
    rows = []

    for t in range(1, steps + 1):
        p1 = softmax(state.logits_step1)
        a1, idx1 = sample_action(rng, actions1, p1)

        logits2 = state.logits_step2_a if a1 == "A" else state.logits_step2_b
        p2 = softmax(logits2)
        a2, idx2 = sample_action(rng, actions2, p2)

        reward = 1.0 if (a1 == "A" and a2 == "X") else 0.0
        total_reward += reward

        advantage = reward - baseline if use_baseline else reward

        if UPDATE_STEP1:
            update_logits(state.logits_step1, p1, idx1, advantage, lr)
        update_logits(logits2, p2, idx2, advantage, lr)

        if use_baseline:
            baseline += (reward - baseline) / t

        rows.append(
            {
                "step": t,
                "a1": a1,
                "a2": a2,
                "reward": reward,
                "baseline": baseline,
                "advantage": advantage,
                "p1": p1,
                "p2": p2,
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "traj.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "steps": steps,
        "mean_reward": total_reward / steps if steps else 0.0,
        "final_p1": softmax(state.logits_step1),
        "final_p2_a": softmax(state.logits_step2_a),
        "final_p2_b": softmax(state.logits_step2_b),
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
