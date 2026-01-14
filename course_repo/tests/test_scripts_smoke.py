from __future__ import annotations

import json
from pathlib import Path

from course.core.eval import run_eval
from course.core.inspect import analyze_run, resolve_run
from course.core.selection import run_selection_demo

try:
    from course.assignments.selection_policy_sol import pick_best as _pick_best
except ImportError:  # student repo uses the template file
    from course.assignments.selection_policy import pick_best as _pick_best


def test_eval_smoke(tmp_path: Path):
    out_dir, summary = run_eval(
        dataset_path=Path("data/datasets/math_dev.jsonl"),
        completions_path=Path("data/rollouts/frozen_rollouts_dev.jsonl"),
        out_dir=tmp_path / "eval_run",
        max_examples=5,
        argv=["python", "-m", "course.eval"],
        args={"dataset": "data/datasets/math_dev.jsonl"},
    )
    assert out_dir.exists()
    assert summary["run"]["n_examples"] == 5
    assert 0.0 <= summary["metrics"]["pass_rate"] <= 1.0

    counts = summary["outcomes"]["counts"]
    # Regression: codes should be informative (not all 'unknown')
    assert "wrong_answer" in counts
    assert "missing_prefix" in counts

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["inputs"]) >= 2
    assert all("sha256" in x or x.get("missing") for x in manifest["inputs"])

    run = resolve_run(tmp_path / "eval_run")
    analysis = analyze_run(run)
    assert analysis["mode"] == "eval"
    assert analysis["n"] == 5


def test_selection_smoke(tmp_path: Path):
    out_dir, summary = run_selection_demo(
        dataset_path=Path("data/datasets/math_dev.jsonl"),
        samples_path=Path("data/rollouts/selection_pack_dev.jsonl"),
        out_dir=tmp_path / "sel_run",
        n=4,
        max_examples=5,
        argv=["python", "-m", "course.selection_demo"],
        args={"dataset": "data/datasets/math_dev.jsonl"},
        pick_best=_pick_best,
    )
    assert summary["run"]["n_examples"] == 5
    assert summary["metrics"]["pass_at_n"] >= summary["metrics"]["pass_at_1"]

    run = resolve_run(tmp_path / "sel_run")
    analysis = analyze_run(run)
    assert analysis["mode"] == "selection"
    assert analysis["n"] == 5


def test_eval_logprob_kl_support(tmp_path: Path):
    # Create a tiny completions file with optional logprob bookkeeping.
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "\n".join(
            [
                json.dumps({"id": "dev-0001", "completion": "Final: 22", "sum_logprob": -1.0, "sum_ref_logprob": -2.0}),
                json.dumps({"id": "dev-0002", "completion": "Final: 108", "sum_logprob": -3.0, "sum_ref_logprob": -3.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir, summary = run_eval(
        dataset_path=Path("data/datasets/math_dev.jsonl"),
        completions_path=rollouts,
        out_dir=tmp_path / "eval_kl",
        max_examples=2,
    )
    assert "mean_kl_est" in summary["metrics"]

    # Results should include per-sample kl_est when provided.
    lines = (out_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec0 = json.loads(lines[0])
    assert rec0["kl_est"] is not None
