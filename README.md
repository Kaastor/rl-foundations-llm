# RL Foundations for LLMs — Course Harness (CPU, local, verifiable rewards)

This repo is the **teacher-provided harness** for the course described in the prompt.
It is intentionally small and boring — because the goal is **measurement discipline**,
not “ship an RL stack”.

Everything revolves around one interface:

```py
score(example, completion) -> {"reward": float, "details": dict}
```

Where:

- `example` contains `{id, prompt, expected_answer}`
- `completion` is the model output string
- the scorer **does not solve math** — it only verifies by comparing the parsed final answer
  to `expected_answer`.

## Quickstart

From the repo root:

```bash
# 1) Run eval (Loop A) on teacher-provided frozen rollouts
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl

# 1b) Inspect a run (group failures, show examples)
python -m course.inspect_run --run runs/<your_run_dir> --only-fails

# 2) Run selection demo (Loop B, Best-of-N)
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4

# 3) Run toy bandit learning (Loop C, conceptual)
python -m course.bandit_train --slow --steps 15
python -m course.bandit_train --steps 500 --baseline

# 4) (Optional) KL tradeoff demo (categorical policy, no LLMs)
python -m course.kl_tradeoff_demo --plot
```

Each command creates a new folder under `runs/` containing:

- raw per-example scored records (`results.jsonl`)
- `summary.json` and `summary.md`
- run metadata (dataset/scorer versions, seeds, etc.)

## Repo layout (what to touch)

- `course/` — teacher-owned code. Read it, inspect it, but avoid refactors.
- `tests/` — **student adds tests here** (especially in Milestone 4/5).
- `notes/` — student-written notes (not included by default; add as you go).
- `data/` — teacher-provided datasets + frozen rollouts.
- `runs/` — generated outputs (safe to delete).

## File formats (JSONL)

### Dataset (`data/datasets/*.jsonl`)

Each line:

```json
{"id":"dev-0001","prompt":"Compute 17*19. Output exactly one line: Final: <int>","expected_answer":323}
```

### Frozen rollouts (`data/rollouts/frozen_rollouts_*.jsonl`)

Each line:

```json
{"id":"dev-0001","completion":"Final: 323"}
```

### Selection pack (`data/rollouts/selection_pack_*.jsonl`)

Each line:

```json
{"id":"dev-0001","samples":["Final: 323","Final:  323","The answer is 323.","Final: 999"]}
```

## Design notes (why it’s built this way)

- Deterministic scoring is treated like production code: versioned, tested, inspectable.
- Scripts are small and explicit; outputs are readable JSONL.
- No RL frameworks. No training infra. The point is to understand the loop and the failure modes.

## Running tests

If you have `pytest`:

```bash
pytest -q
```

If you don’t, you can still run the course scripts — tests are optional but strongly recommended.
