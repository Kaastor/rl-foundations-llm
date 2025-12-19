from pathlib import Path

from course.eval import evaluate
from course.selection_demo import selection_demo
from course.inspect_run import resolve_run, analyze_run


def test_eval_smoke(tmp_path: Path):
    summary = evaluate(
        dataset_path=Path("data/datasets/math_dev.jsonl"),
        completions_path=Path("data/rollouts/frozen_rollouts_dev.jsonl"),
        out_dir=tmp_path / "eval_run",
        max_examples=5,
    )
    assert summary["run"]["n_examples"] == 5
    assert 0.0 <= summary["metrics"]["pass_rate"] <= 1.0

    run = resolve_run(tmp_path / "eval_run")
    analysis = analyze_run(run)
    assert analysis["mode"] == "eval"
    assert analysis["n"] == 5


def test_selection_smoke(tmp_path: Path):
    summary = selection_demo(
        dataset_path=Path("data/datasets/math_dev.jsonl"),
        samples_path=Path("data/rollouts/selection_pack_dev.jsonl"),
        out_dir=tmp_path / "sel_run",
        n=4,
        max_examples=5,
    )
    assert summary["run"]["n_examples"] == 5
    assert summary["metrics"]["pass_at_n"] >= summary["metrics"]["pass_at_1"]

    run = resolve_run(tmp_path / "sel_run")
    analysis = analyze_run(run)
    assert analysis["mode"] == "selection"
    assert analysis["n"] == 5
