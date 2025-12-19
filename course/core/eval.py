from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from course.core.completion_sources import CompletionSource, JsonlCompletionSource
from course.core.datasets import load_examples
from course.core.io import atomic_write_text, make_run_dir, utc_now_iso, write_json, write_jsonl, write_manifest
from course.core.scoring import SCORER_NAME, SCORER_VERSION, score


def evaluate_examples(
    *,
    dataset_path: Path,
    completion_source: CompletionSource,
    out_dir: Path,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a dataset using a completion source, writing run artifacts.

    This is the core of Loop A.
    """

    created_utc = utc_now_iso()

    examples = load_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    results: list[Dict[str, Any]] = []
    outcome_codes: Counter[str] = Counter()
    missing = 0

    total_reward = 0.0

    kl_vals: list[float] = []

    for ex in examples:
        sample = completion_source.get(ex.id)
        missing_completion = sample is None
        if missing_completion:
            missing += 1
            completion = None
            sum_logprob = None
            sum_ref_logprob = None
            kl_est = None
        else:
            completion = sample.completion
            sum_logprob = sample.sum_logprob
            sum_ref_logprob = sample.sum_ref_logprob
            kl_est = sample.kl_estimate()
            if kl_est is not None:
                kl_vals.append(float(kl_est))

        scored = score(ex, completion)
        reward = float(scored.get("reward", 0.0))
        details = scored.get("details", {})

        # Enforce binary reward assumption.
        assert reward in (0.0, 1.0), f"Reward must be 0.0 or 1.0, got {reward} for example {ex.id}"

        total_reward += reward

        # Stable classification for grouping.
        outcome = (details or {}).get("result") or {}
        code = str(outcome.get("code") or "unknown")
        
        # Override outcome_code for missing completions to maintain semantic honesty.
        if missing_completion:
            code = "missing_completion"
        
        outcome_codes[code] += 1

        results.append(
            {
                "id": ex.id,
                "prompt": ex.prompt,
                "expected_answer": ex.expected_answer,
                "completion": "" if completion is None else completion,
                "missing_completion": bool(missing_completion),
                "reward": reward,
                "outcome_code": code,
                "sum_logprob": sum_logprob,
                "sum_ref_logprob": sum_ref_logprob,
                "kl_est": kl_est,
                "details": details,
            }
        )

    n = len(examples)
    n_pass = sum(1 for r in results if r["reward"] == 1.0)
    pass_rate = (n_pass / n) if n else 0.0

    # Summaries for failures only (excluding ok)
    failures = {k: v for k, v in outcome_codes.items() if k != "ok"}

    summary: Dict[str, Any] = {
        "run": {
            "created_utc": created_utc,
            "scorer": {"name": SCORER_NAME, "version": SCORER_VERSION},
            "dataset_path": str(dataset_path),
            "completion_source": completion_source.describe(),
            "n_examples": n,
            "n_missing_completions": missing,
        },
        "metrics": {
            "mean_reward": pass_rate,
            "pass_rate": pass_rate,
            "n_pass": n_pass,
            "n_fail": n - n_pass,
        },
        "outcomes": {
            "counts": dict(outcome_codes.most_common()),
            "failures": dict(sorted(failures.items(), key=lambda kv: kv[1], reverse=True)),
        },
    }

    if kl_vals:
        mean_kl = sum(kl_vals) / len(kl_vals)
        summary["metrics"]["n_with_kl"] = len(kl_vals)
        summary["metrics"]["mean_kl_est"] = mean_kl

    write_jsonl(out_dir / "results.jsonl", results)
    write_json(out_dir / "summary.json", summary)

    # Human-readable summary
    top_failures = [r for r in results if r["reward"] == 0.0][:10]

    md_lines = []
    md_lines.append("# Eval run\n")
    md_lines.append(f"- Created (UTC): `{created_utc}`\n")
    md_lines.append(f"- Scorer: `{SCORER_NAME}` v`{SCORER_VERSION}`\n")
    md_lines.append(f"- Dataset: `{dataset_path}`\n")
    md_lines.append(f"- Completion source: `{summary['run']['completion_source']}`\n")
    md_lines.append(f"- Examples: `{n}` (missing completions: `{missing}`)\n")

    md_lines.append("\n## Metrics\n")
    md_lines.append(f"- pass_rate: **{pass_rate:.3f}**\n")
    md_lines.append(f"- n_pass: `{summary['metrics']['n_pass']}`\n")
    md_lines.append(f"- n_fail: `{summary['metrics']['n_fail']}`\n")

    if kl_vals:
        md_lines.append(f"- mean_kl_est: **{summary['metrics']['mean_kl_est']:.3f}** (n={len(kl_vals)})\n")

    md_lines.append("\n## Outcome codes (grouping)\n")
    for code, cnt in outcome_codes.most_common():
        md_lines.append(f"- `{code}`: {cnt}\n")

    md_lines.append("\n## First 10 failures (inspect details in results.jsonl)\n")
    for r in top_failures:
        preview = str(r["completion"])[:80].replace("\n", " ")
        md_lines.append(f"- `{r['id']}` [{r['outcome_code']}] completion preview: `{preview}...`\n")

    atomic_write_text(out_dir / "summary.md", "".join(md_lines))

    return summary


def run_eval(
    *,
    dataset_path: Path,
    completions_path: Path,
    out_dir: Optional[Path] = None,
    max_examples: Optional[int] = None,
    argv: Optional[list[str]] = None,
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Path, Dict[str, Any]]:
    """Convenience wrapper used by the CLI.

    Creates a run directory (if needed), evaluates, and writes a manifest with CLI args.
    """

    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="eval")

    source = JsonlCompletionSource(completions_path)
    summary = evaluate_examples(dataset_path=dataset_path, completion_source=source, out_dir=out_dir, max_examples=max_examples)

    # Manifest: hashes + args + env snapshot.
    # NOTE: We always write this, even for programmatic use, because it's a
    # central part of the "production-like" discipline.
    created = summary.get("run", {}).get("created_utc") or utc_now_iso()
    write_manifest(
        out_dir,
        created_utc=created,
        script="eval",
        argv=argv or [],
        args=args or {},
        inputs=[dataset_path, completions_path],
        scorer={"name": SCORER_NAME, "version": SCORER_VERSION},
        extra={"completion_source": source.describe()},
    )

    return out_dir, summary
