# AGENTS.md

## Purpose
- This repo is a small, scope-limited harness for teaching RL concepts in LLM systems, with emphasis on measurement discipline.
- Core contract: `score(example, completion) -> {"reward": float, "details": dict}`; the scorer verifies, it does not solve math.

## Key paths
- `course/core/`: core library (IO, scoring, eval/selection logic, artifacts). Keep changes deliberate and versioned.
- `course/assignments/`: student-editable surfaces (start here for policy tweaks).
- `course/`: CLI entrypoints (`poetry run python -m course.eval`, etc.).
- `data/`: datasets, rollouts, golden sets.
- `runs/`: generated outputs (safe to delete).

## Scoring and measurement invariants
- Scoring is strict: exactly one line `Final: <int>`; no extra whitespace, no plus sign, no leading zeros, no negative zero.
- Scoring must be deterministic and total (never raise).
- If reward behavior changes, update `SCORER_VERSION` and `REWARD_SPEC` in `course/core/scoring.py`.
- Follow `MEASUREMENT_RULES.md` (Locked Room Rule: keep dataset, prompt, scorer, sampling fixed for comparisons).

## Golden validation (scorer changes)
- Use `data/golden/golden_correct.jsonl` and `data/golden/golden_exploits.jsonl` with:
  - `poetry run python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_correct.jsonl`
  - `poetry run python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits.jsonl`

## Data formats
- Dataset JSONL: `{"id","prompt","expected_answer"}`.
- Completions JSONL: `{"id","completion"}` with optional logprob fields.
- Selection pack JSONL: `{"id","samples":[...]}`

## Running and tests
- Typical commands: `poetry run python -m course.eval`, `poetry run python -m course.selection_demo`, `poetry run python -m course.bandit_train`, `poetry run python -m course.gate`.
- Tests: `poetry run pytest -q`.
