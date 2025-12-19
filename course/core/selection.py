from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from course.core.datasets import load_examples
from course.core.io import atomic_write_text, make_run_dir, utc_now_iso, write_json, write_jsonl, write_manifest
from course.core.rollouts import load_selection_pack
from course.core.scoring import SCORER_NAME, SCORER_VERSION, score
from course.core.types import Example, RolloutSample


def _as_selection_triplet(obj: Any) -> Tuple[int, RolloutSample, Dict[str, Any]]:
    """Normalize different possible return styles from student policies."""

    # Tuple style: (idx, sample, scored)
    if isinstance(obj, tuple) and len(obj) == 3:
        idx, sample, scored = obj
        return int(idx), sample, dict(scored)

    # Dataclass / object style (duck typing)
    idx = getattr(obj, "best_index")
    sample = getattr(obj, "best_sample")
    scored = getattr(obj, "scored")
    return int(idx), sample, dict(scored)


SelectionPolicyFn = Callable[[Example, list[RolloutSample]], Any]


def selection_demo(
    *,
    dataset_path: Path,
    samples_path: Path,
    pick_best: Callable[..., Any],
    out_dir: Path,
    n: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Loop B: Best-of-N selection using the deterministic verifier."""

    created_utc = utc_now_iso()

    examples = load_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    samples_map = load_selection_pack(samples_path)

    rows: list[Dict[str, Any]] = []
    pass1 = 0
    passN = 0
    missing = 0
    rescued = 0

    for ex in examples:
        samples = samples_map.get(ex.id)
        if not samples:
            missing += 1
            samples = [RolloutSample(completion="")]  # fail safe

        if n is not None:
            samples = samples[:n]

        # Baseline: take first sample.
        baseline_sample = samples[0] if samples else RolloutSample(completion="")
        baseline_scored = score(ex, baseline_sample.completion)
        baseline_reward = float(baseline_scored.get("reward", 0.0))
        baseline_outcome = (baseline_scored.get("details") or {}).get("result") or {}
        baseline_code = str(baseline_outcome.get("code") or "unknown")
        if baseline_reward == 1.0:
            pass1 += 1

        # Best-of-N selection.
        pick_obj = pick_best(ex, samples, scorer=score)
        best_idx, best_sample, best_scored = _as_selection_triplet(pick_obj)
        best_reward = float(best_scored.get("reward", 0.0))
        best_outcome = (best_scored.get("details") or {}).get("result") or {}
        best_code = str(best_outcome.get("code") or "unknown")
        if best_reward == 1.0:
            passN += 1

        if baseline_reward == 0.0 and best_reward == 1.0:
            rescued += 1

        def _sample_brief(s: RolloutSample) -> Dict[str, Any]:
            return {
                "completion": s.completion,
                "sum_logprob": s.sum_logprob,
                "sum_ref_logprob": s.sum_ref_logprob,
                "kl_est": s.kl_estimate(),
            }

        rows.append(
            {
                "id": ex.id,
                "prompt": ex.prompt,
                "expected_answer": ex.expected_answer,
                "n_samples_used": len(samples),
                "baseline": {
                    "index": 0,
                    "completion": baseline_sample.completion,
                    "reward": baseline_reward,
                    "outcome_code": baseline_code,
                },
                "best_of_n": {
                    "index": best_idx,
                    "completion": best_sample.completion,
                    "reward": best_reward,
                    "outcome_code": best_code,
                },
                "all_samples": [_sample_brief(s) for s in samples],
            }
        )

    n_ex = len(examples)
    summary: Dict[str, Any] = {
        "run": {
            "created_utc": created_utc,
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
            "rescued": rescued,
        },
    }

    write_jsonl(out_dir / "results.jsonl", rows)
    write_json(out_dir / "summary.json", summary)

    md = []
    md.append("# Selection demo (Best-of-N)\n\n")
    md.append(f"- Created (UTC): `{created_utc}`\n")
    md.append(f"- Scorer: `{SCORER_NAME}` v`{SCORER_VERSION}`\n")
    md.append(f"- Dataset: `{dataset_path}`\n")
    md.append(f"- Samples: `{samples_path}`\n")
    md.append(f"- n_samples_used (cap): `{n}`\n")
    md.append("\n## Metrics\n")
    md.append(f"- pass@1: **{summary['metrics']['pass_at_1']:.3f}**\n")
    md.append(f"- pass@N: **{summary['metrics']['pass_at_n']:.3f}**\n")
    md.append(f"- delta: **{summary['metrics']['delta']:.3f}**\n")
    md.append(f"- rescued: `{summary['metrics']['rescued']}`\n")
    md.append("\n## Interpretation reminder\n")
    md.append(
        "Best-of-N improves the *chosen output* by spending more sampling compute.\n"
        "It does **not** change the underlying model distribution.\n"
    )
    atomic_write_text(out_dir / "summary.md", "".join(md))

    write_manifest(
        out_dir,
        created_utc=created_utc,
        script="selection_demo",
        inputs=[dataset_path, samples_path],
        scorer={"name": SCORER_NAME, "version": SCORER_VERSION},
        extra={"n": n, "n_missing_samples": missing},
    )

    return summary


def run_selection_demo(
    *,
    dataset_path: Path,
    samples_path: Path,
    pick_best: Callable[..., Any],
    out_dir: Optional[Path] = None,
    n: Optional[int] = None,
    max_examples: Optional[int] = None,
    argv: Optional[list[str]] = None,
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Path, Dict[str, Any]]:
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="selection")

    summary = selection_demo(
        dataset_path=dataset_path,
        samples_path=samples_path,
        pick_best=pick_best,
        out_dir=out_dir,
        n=n,
        max_examples=max_examples,
    )

    if argv is not None or args is not None:
        created = summary.get("run", {}).get("created_utc") or utc_now_iso()
        write_manifest(
            out_dir,
            created_utc=created,
            script="selection_demo",
            argv=argv,
            args=args,
            inputs=[dataset_path, samples_path],
            scorer={"name": SCORER_NAME, "version": SCORER_VERSION},
            extra={"n": n},
        )

    return out_dir, summary
