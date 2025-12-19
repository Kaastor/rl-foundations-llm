from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from course.datasets.loader import load_examples, index_by_id
from course.io_utils import make_run_dir, read_jsonl, write_json, write_jsonl, atomic_write_text, utc_now_iso
from course.score import SCORER_NAME, SCORER_VERSION, score
from course.types import Example


def _load_samples_map(path: Path) -> dict[str, list[str]]:
    records = read_jsonl(path)
    m: dict[str, list[str]] = {}
    for rec in records:
        if "id" not in rec:
            raise ValueError(f"Selection pack record missing 'id' in {path}: {rec!r}")
        ex_id = str(rec["id"])
        samples = rec.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Selection pack record must include list 'samples'. id={ex_id!r} rec={rec!r}")
        m[ex_id] = ["" if s is None else str(s) for s in samples]
    return m


def _tie_break_key(completion: str) -> tuple[int, str]:
    """Deterministic tie-break: prefer lexicographically smallest completion.

    The first element can be extended later if you want to incorporate length penalties, etc.
    Keep it deterministic and explainable.
    """
    return (0, completion)


def pick_best(example: Example, samples: list[str]) -> tuple[int, str, dict[str, Any]]:
    """Pick the best sample for an example.

    This is the only function students are allowed/encouraged to edit (Milestone 0.5).
    Keep it small (5–15 lines) and deterministic.

    Current behavior:
    - score all samples
    - choose max reward
    - deterministic tie-break by lexicographic order of completion text
    - if still tied, earliest index wins (stable)

    Returns: (best_index, best_completion, best_scored_record)
    """
    scored_samples: list[tuple[float, tuple[int, str], int, str, dict[str, Any]]] = []
    for i, s in enumerate(samples):
        scored = score(example, s)
        r = float(scored["reward"])
        scored_samples.append((r, _tie_break_key(s), i, s, scored))

    # max by reward then tie-break key; then stable index.
    best = max(scored_samples, key=lambda t: (t[0], t[1], -t[2]))
    _r, _tb, idx, comp, scored = best
    return idx, comp, scored


def selection_demo(
    dataset_path: Path,
    samples_path: Path,
    *,
    out_dir: Path,
    n: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> dict[str, Any]:
    examples = load_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    examples_by_id = index_by_id(examples)

    samples_map = _load_samples_map(samples_path)

    rows: list[dict[str, Any]] = []
    pass1 = 0
    passN = 0
    missing = 0

    for ex in examples:
        samples = samples_map.get(ex.id)
        if not samples:
            missing += 1
            samples = [""]  # fail safe

        if n is not None:
            samples = samples[:n]

        # Baseline: take first sample.
        baseline_completion = samples[0] if samples else ""
        baseline_scored = score(ex, baseline_completion)
        baseline_reward = float(baseline_scored["reward"])
        if baseline_reward == 1.0:
            pass1 += 1

        # Best-of-N selection.
        best_idx, best_completion, best_scored = pick_best(ex, samples)
        best_reward = float(best_scored["reward"])
        if best_reward == 1.0:
            passN += 1

        rows.append(
            {
                "id": ex.id,
                "prompt": ex.prompt,
                "expected_answer": ex.expected_answer,
                "n_samples_used": len(samples),
                "baseline": {
                    "index": 0,
                    "completion": baseline_completion,
                    "reward": baseline_reward,
                },
                "best_of_n": {
                    "index": best_idx,
                    "completion": best_completion,
                    "reward": best_reward,
                },
                "all_samples": samples,
            }
        )

    n_ex = len(examples)
    summary: dict[str, Any] = {
        "run": {
            "created_utc": utc_now_iso(),
            "scorer": {"name": SCORER_NAME, "version": SCORER_VERSION},
            "dataset_path": str(dataset_path),
            "samples_path": str(samples_path),
            "n_examples": n_ex,
            "n_missing_samples": missing,
            "n": n,
        },
        "metrics": {
            "pass_at_1": pass1 / n_ex if n_ex else 0.0,
            "pass_at_n": passN / n_ex if n_ex else 0.0,
            "delta": (passN - pass1) / n_ex if n_ex else 0.0,
            "n_pass_at_1": pass1,
            "n_pass_at_n": passN,
        },
    }

    write_jsonl(out_dir / "results.jsonl", rows)
    write_json(out_dir / "summary.json", summary)

    md = []
    md.append("# Selection demo (Best-of-N)\n\n")
    md.append(f"- Created (UTC): `{summary['run']['created_utc']}`\n")
    md.append(f"- Scorer: `{SCORER_NAME}` v`{SCORER_VERSION}`\n")
    md.append(f"- Dataset: `{dataset_path}`\n")
    md.append(f"- Samples: `{samples_path}`\n")
    md.append(f"- n_samples_used (cap): `{n}`\n")
    md.append("\n## Metrics\n")
    md.append(f"- pass@1: **{summary['metrics']['pass_at_1']:.3f}**\n")
    md.append(f"- pass@N: **{summary['metrics']['pass_at_n']:.3f}**\n")
    md.append(f"- delta: **{summary['metrics']['delta']:.3f}**\n")
    md.append("\n## Interpretation reminder\n")
    md.append(
        "Best-of-N improves the *chosen output* by spending more sampling compute.\n"
        "It does **not** change the underlying policy distribution.\n"
    )
    atomic_write_text(out_dir / "summary.md", "".join(md))

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Loop B: Best-of-N selection using the deterministic verifier.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSONL")
    p.add_argument("--samples", type=Path, required=True, help="Path to selection pack JSONL (id -> samples)")
    p.add_argument("--n", type=int, default=None, help="Use only the first N samples per example")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/selection_<timestamp>)")
    p.add_argument("--max", type=int, default=None, dest="max_examples", help="Optional cap for quick runs")
    args = p.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="selection")

    summary = selection_demo(
        dataset_path=args.dataset,
        samples_path=args.samples,
        out_dir=out_dir,
        n=args.n,
        max_examples=args.max_examples,
    )

    print(f"Wrote results to: {out_dir}")
    print(f"pass@1={summary['metrics']['pass_at_1']:.3f}  pass@N={summary['metrics']['pass_at_n']:.3f}")


if __name__ == "__main__":
    main()
