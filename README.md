# RL Foundations for LLMs

This repository is a **scope-limited** teaching harness for Reinforcement Learning (RL) ideas as they show up in LLM systems.

It is intentionally small and boring.

The goal is not to ship an RL stack — it is to teach the habits that make RL/LLM work **trustworthy**:

- treat reward/eval like a **measurement instrument**
- keep the instrument stable when comparing runs (Locked Room Rule)
- save artifacts so failures have names, not vibes
- separate **selection gains** (best-of-N) from **learning gains** (policy changes)

Everything revolves around one contract:

```py
score(example, completion) -> {"reward": float, "details": dict}
```

Where:

- `example` contains `{id, prompt, expected_answer}`
- `completion` is the model output string
- the scorer does **not** solve math — it only verifies by comparing the parsed final answer to `expected_answer`

## Setup

**Requirements:** Python 3.10 or higher, [Poetry](https://python-poetry.org/docs/#installation)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rl-foundations-llm
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. (Optional) Install with development dependencies for testing and visualization:
   ```bash
   poetry install --with dev
   ```

That's it. The harness is ready to use. Run commands with `poetry run` prefix or activate the shell with `poetry shell`.

## Quickstart

From the repo root:

```bash
# Loop A: eval frozen rollouts
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl

# Inspect a run: group failures and show examples
python -m course.inspect_run --run runs/<your_run_dir> --only-fails

# Loop B: Best-of-N selection (uses student-editable selection policy)
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4

# Gate candidate vs baseline (production-ish “should we promote?” check)
python -m course.gate \
  --baseline runs/<baseline_eval_run> \
  --candidate runs/<candidate_eval_run> \
  --min-delta 0.00

# Loop C: tiny policy-gradient microscope (no LLMs)
python -m course.bandit_train --steps 200 --baseline

# Optional: KL tradeoff demo (no LLMs)
python -m course.kl_tradeoff_demo --plot
```

Each command creates a new folder under `runs/` containing:

- `results.jsonl` — per-example scored records
- `summary.json` and `summary.md` — human + machine summaries
- `manifest.json` — environment + input hashes (production-ish reproducibility)

## Repo layout (what to touch)

- `course/core/` — **tiny library**: IO, schemas, scoring contract, eval/selection logic, artifacts
- `course/` — CLI entrypoints (`python -m course.eval`, etc.)
- `course/assignments/` — **student-editable surfaces** (start here)
- `tests/` — tests (optional but recommended)
- `data/` — teacher-provided datasets and frozen rollouts
- `runs/` — generated outputs (safe to delete)

## Assignments

This harness is set up so students can change behavior without refactoring infrastructure.

Example: Best-of-N selection policy lives in:

- `course/assignments/selection_policy.py`

All scripts in `course/` call into `course/core/`.

## File formats (JSONL)

### Dataset (`data/datasets/*.jsonl`)

Each line:

```json
{"id":"dev-0001","prompt":"Compute 17*19. Output exactly one line: Final: <int>","expected_answer":323}
```

### Completions (`data/rollouts/frozen_rollouts_*.jsonl`)

Each line:

```json
{"id":"dev-0001","completion":"Final: 323"}
```

Optional production-ish metadata is allowed:

```json
{"id":"dev-0001","completion":"Final: 323","sum_logprob":-12.34,"sum_ref_logprob":-12.80}
```

(Synonyms accepted: `logprob`/`ref_logprob`/`total_logprob`/`total_ref_logprob`.)

### Selection pack (`data/rollouts/selection_pack_*.jsonl`)

Each line:

```json
{"id":"dev-0001","samples":["Final: 323","Final:  323","The answer is 323.","Final: 999"]}
```

Samples may also be objects with metadata:

```json
{"id":"dev-0001","samples":[{"completion":"Final: 323","logprob":-3.2},"Final: 999"]}
```

## Running tests

If you have `pytest`:

```bash
pytest -q
```

(Tests are optional; the scripts run without installing anything.)
