from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from course.datasets.loader import load_examples
from course.io_utils import make_run_dir, read_jsonl, write_json, write_jsonl, atomic_write_text, utc_now_iso
from course.score import SCORER_NAME, SCORER_VERSION, score


def _load_completions_map(path: Path) -> dict[str, str]:
    """Load a JSONL file mapping example id -> completion string."""
    records = read_jsonl(path)
    m: dict[str, str] = {}
    for rec in records:
        if "id" not in rec:
            raise ValueError(f"Completion record missing 'id' in {path}: {rec!r}")
        ex_id = str(rec["id"])
        completion = rec.get("completion", "")
        m[ex_id] = "" if completion is None else str(completion)
    return m


def evaluate(
    dataset_path: Path,
    completions_path: Path,
    *,
    out_dir: Path,
    max_examples: Optional[int] = None,
) -> dict[str, Any]:
    examples = load_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    completion_map = _load_completions_map(completions_path)

    results: list[dict[str, Any]] = []
    parse_error_codes: Counter[str] = Counter()
    missing = 0

    total_reward = 0.0
    for ex in examples:
        if ex.id not in completion_map:
            missing += 1
            completion = None  # scorer treats None as empty string
        else:
            completion = completion_map[ex.id]

        scored = score(ex, completion)
        reward = float(scored.get("reward", 0.0))
        details = scored.get("details", {})

        total_reward += reward

        parse = (details or {}).get("parse") or {}
        code = parse.get("error_code") or "ok" if reward == 1.0 else "unknown"
        if reward == 0.0 and code:
            parse_error_codes[str(code)] += 1

        results.append(
            {
                "id": ex.id,
                "prompt": ex.prompt,
                "expected_answer": ex.expected_answer,
                "completion": "" if completion is None else completion,
                "reward": reward,
                "details": details,
            }
        )

    n = len(examples)
    pass_rate = (total_reward / n) if n else 0.0

    summary: dict[str, Any] = {
        "run": {
            "created_utc": utc_now_iso(),
            "scorer": {"name": SCORER_NAME, "version": SCORER_VERSION},
            "dataset_path": str(dataset_path),
            "completions_path": str(completions_path),
            "n_examples": n,
            "n_missing_completions": missing,
        },
        "metrics": {
            "mean_reward": pass_rate,
            "pass_rate": pass_rate,  # same in 0/1 scoring
            "n_pass": int(total_reward),
            "n_fail": n - int(total_reward),
        },
        "failures": {
            "parse_error_codes": dict(parse_error_codes.most_common()),
        },
    }

    # Write outputs
    write_jsonl(out_dir / "results.jsonl", results)
    write_json(out_dir / "summary.json", summary)

    # Human-readable summary
    top_failures = [r for r in results if r["reward"] == 0.0][:10]
    md_lines = []
    md_lines.append(f"# Eval run\n")
    md_lines.append(f"- Created (UTC): `{summary['run']['created_utc']}`\n")
    md_lines.append(f"- Scorer: `{SCORER_NAME}` v`{SCORER_VERSION}`\n")
    md_lines.append(f"- Dataset: `{dataset_path}`\n")
    md_lines.append(f"- Completions: `{completions_path}`\n")
    md_lines.append(f"- Examples: `{n}` (missing completions: `{missing}`)\n")
    md_lines.append("\n## Metrics\n")
    md_lines.append(f"- pass_rate: **{pass_rate:.3f}**\n")
    md_lines.append(f"- n_pass: `{summary['metrics']['n_pass']}`\n")
    md_lines.append(f"- n_fail: `{summary['metrics']['n_fail']}`\n")
    md_lines.append("\n## Failure codes (reward==0)\n")
    for code, cnt in parse_error_codes.most_common():
        md_lines.append(f"- `{code}`: {cnt}\n")
    md_lines.append("\n## First 10 failures (inspect details in results.jsonl)\n")
    for r in top_failures:
        preview = r["completion"][:80].replace("\n", " ")
        md_lines.append(f"- `{r['id']}` completion preview: `{preview}...`\n")
    atomic_write_text(out_dir / "summary.md", "".join(md_lines))

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Loop A: evaluate frozen rollouts using the deterministic verifier.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSONL")
    p.add_argument("--completions", type=Path, required=True, help="Path to completions JSONL (id -> completion)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/eval_<timestamp>)")
    p.add_argument("--max", type=int, default=None, dest="max_examples", help="Optional cap for quick runs")
    args = p.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="eval")

    summary = evaluate(
        dataset_path=args.dataset,
        completions_path=args.completions,
        out_dir=out_dir,
        max_examples=args.max_examples,
    )

    print(f"Wrote results to: {out_dir}")
    print(f"pass_rate={summary['metrics']['pass_rate']:.3f}  n={summary['run']['n_examples']}")
    if summary["run"]["n_missing_completions"]:
        print(f"WARNING: missing completions for {summary['run']['n_missing_completions']} examples")


if __name__ == "__main__":
    main()
