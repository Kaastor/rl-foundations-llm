# RL Foundations for LLMs — Mastery Track Project Roadmap

---

## Project Overview

**Project Title:** Verifier-Driven Math QA — Construction, Deliberate Disruption, and Systematic Hardening

This track is designed to enforce understanding through repeated application of a structured experimental cycle:

1. **Build (Baseline Establishment):** Establish a clean, deterministic baseline over which the student exercises complete control.
2. **Sabotage (Controlled Perturbation):** Introduce precisely one explicit modification that violates a core principle.
3. **Reflect (Forensic Analysis):** Observe the resulting failure through examination of artifacts (`manifest.json`, `summary.json`, `results.jsonl`) rather than through intuitive assessment.
4. **Repair and Lock (Mechanism Correction):** Correct the underlying mechanism and encode the fix as tests and gates to prevent regression.

The Sabotage steps constitute the essential pedagogical component of this track and must not be omitted.

---

## Why This Exists

LLM training and evaluation pipelines are easy to misread. Metrics can move because you changed the scorer, dataset, or selection rule rather than the model itself. This track trains measurement discipline: build a clean baseline, make one controlled change, inspect artifacts, and lock the fix. The goal is reliable evidence, not just higher numbers.

## Where It Matters in the LLM Lifecycle

- Data curation and eval: keep dataset, prompt, and scorer fixed so comparisons are valid.
- Reward specification and verification: scorers and reward models are production instruments that must be deterministic and versioned.
- Inference-time selection: Best-of-N can boost results without learning, so attribution must be explicit.
- Training (SFT/RLHF/DPO): credit assignment, variance, and KL constraints determine whether learning is stable and meaningful.
- Deployment and monitoring: regression tests, golden sets, and gates prevent silent metric drift or reward hacking.
- Incident analysis: artifact-based forensics let you prove what changed and why.

---

## Scope note (practicality)

These assignments do **not** fine-tune or sample a real LLM. The "policy" work is synthetic to make the mechanics clear, so transfer to LLM systems is conceptual rather than hands-on.

If you want practical grounding, an optional extension is to plug in an LLM, generate fresh rollouts, and re-run Loop A/B using the same measurement discipline.

### Optional extension: real rollouts via Groq (minimal change)

Add `GROQ_API_KEY` to a `.env` file, then:

```bash
# Loop A: sample completions, then eval
python -m course.rollout_sample \
  --dataset data/datasets/math_dev.jsonl \
  --outdir runs/rollouts_groq_dev

python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions runs/rollouts_groq_dev/completions.jsonl \
  --outdir runs/l0_build_eval_groq

# Loop B: sample a selection pack (N samples per prompt)
python -m course.rollout_sample \
  --dataset data/datasets/math_dev.jsonl \
  --n 4 \
  --format selection \
  --outdir runs/rollouts_groq_sel

python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples runs/rollouts_groq_sel/selection_pack.jsonl \
  --n 4 \
  --outdir runs/l0_5_groq_sel
```

Default model is a small Groq model; override with `--model` or `GROQ_MODEL` if needed.

---

## Foundational Rules

These principles should be internalized before commencing practical work.

### The Three Operational Loops

This terminology should be applied consistently throughout all documentation and discussion:

* **Loop A (Evaluation):** Measurement operations exclusively (frozen inputs, deterministic scorer).
* **Loop B (Selection):** Best-of-N selection (observable behavior improves; the underlying policy distribution remains unchanged).
* **Loop C (Learning):** Policy parameter updates (probability distributions shift → generative behavior shifts).

### The Locked Room Principle (Valid Experimental Comparisons)

Before comparing metrics between two experimental runs, the following conditions must be verified as identical:

* Dataset split (identical file fingerprint recorded in `manifest.json`)
* Prompt text (if applicable to the experimental design)
* Scorer **name and version**
* Sampling configuration (if stochastic sampling is employed)

If any of these conditions differ between runs, the experimental environment or measurement instrument has been altered. Such comparisons are methodologically invalid.

### Reward Specification as Production Code

The scorer functions as a measurement instrument and must therefore exhibit the following properties:
**deterministic**, **total (never terminates abnormally)**, **explainable**, **computationally efficient**, **version-controlled**.

The `score()` function should be treated with the same rigor as production software.

---

## Workflow Discipline

The following practices are strongly recommended:

Create a dedicated branch and commit incrementally to maintain a verifiable record of modifications.

```bash
git checkout -b mastery-track
pytest -q
```

For each level, execute the following commit sequence:

* Commit the **Build** state (baseline establishment).
* Commit the **Sabotage** modification (even though it introduces deliberate failure).
* Commit the **Repair** (fix and preventive measures).

---

## Prerequisites and quick checks

Before Level 0, verify the environment and entrypoints:

```bash
python --version   # must be 3.10+
python -m course.eval --help
python -m course.selection_demo --help
python -m course.bandit_train --help
```

If you use Poetry, run commands with `poetry run ...` or activate `poetry shell`.

## Start Here (New Students)

If you are brand new, read "Foundational Rules" once, then jump to "Assignments Start Here" below. Each level follows the same four moves:

1) Build a clean baseline so you have a trusted reference.
2) Sabotage one variable on purpose to see the failure mode.
3) Reflect using artifacts so your conclusion is evidence-based.
4) Repair and lock the fix with tests/notes so it cannot regress.

Every step below includes a short explanation of why you are doing it. If you get lost, return to the "Conceptual Foundation" section at the top of the level.

### Assignment files

Assignment files include TODOs where you should implement missing logic. Tests are provided as scaffolds and will fail until you complete the assignments.

### Common pitfalls to avoid (with descriptions)

* **Python < 3.10:** crashes on `dataclass(slots=True)` before you even reach the assignments.
* **Locked Room violations:** comparing runs with different dataset/scorer/sampling; manifests disagree, so comparisons are invalid.
* **Nondeterministic selection:** random tie-breaks cause result drift and flaky tests; hashes of `results.jsonl` will differ.
* **Forgetting sabotage rollback:** leaving a sabotage change in place contaminates repair results and later levels.
* **Artifact blind spots:** reading only `summary.json` and ignoring `results.jsonl`/`manifest.json` hides failure modes.
* **Format vs math confusion:** treating parse errors as math errors leads to the wrong fixes.
* **Advantage sign mistakes:** negative advantage should decrease action probability; reversing it yields anti-learning.
* **Credit assignment gaps:** updating only the final step keeps early decisions random and stalls learning.
* **KL ignored:** maximizing reward without divergence control produces brittle, high-drift policies.
* **JSONL/ID mismatches:** dataset `id` not matching rollout `id` yields missing samples and silent failures.
* **Missing matplotlib:** KL plots are skipped without it; CSV and summaries still write.

---

## Artifact checklist (what to inspect)

Each run produces evidence; use it in your README and comparisons:

* `manifest.json` — inputs + SHA256 hashes, scorer name/version (Locked Room evidence).
* `results.jsonl` — per-example records and outcome codes.
* `summary.json` — aggregate metrics.
* `summary.md` — human-readable recap.

Some assignments add extra artifacts (e.g., `traj.jsonl`, `kl_tradeoff.csv`).

If you need to prove determinism, hash `results.jsonl` across repeated runs.

---

## Student self-check (validation)

If the checks below pass, you have likely implemented the assignments correctly. If you made a mistake, at least one check should fail or produce an unexpected artifact/result.

### 1) Run the assignment tests

```bash
pytest -q
```

### 2) Validate scorer changes (Level 4)

```bash
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_correct.jsonl
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits.jsonl
```

### 3) Determinism check (Level 0.5 sabotage/repair)

After fixing selection determinism, repeated runs should match:

```bash
for i in 1 2 3; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/selfcheck_sel_$i
done

for i in 1 2 3; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/selfcheck_sel_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

If the hashes match, your selection policy is deterministic.

### 4) Artifact sanity check

For any run in `runs/<name>/`, confirm:
- `manifest.json` exists and includes dataset + scorer info.
- `results.jsonl` exists and includes per-example records.
- `summary.json` exists and matches the expected metric direction for that level.

## Assignments Start Here (Step-by-step)

From here down, follow the levels in order. Each level is a step-by-step narrative with a short explanation for why you are doing each action. Do not skip sabotage or repair; the contrast is the lesson.

---

# Level 0: Measurement Hygiene — Loop A

## Conceptual Foundation

**Incorrect intuition:** "The model is performing poorly."

**Correct mental model:** "Loop A constitutes a measurement instrument verification. If the measured value changed, either (a) the policy changed, or (b) the measurement conditions changed. The objective is to determine which."


## First-run checklist

- Inputs: `data/datasets/math_dev.jsonl`, `data/rollouts/frozen_rollouts_dev.jsonl`
- Outputs: `runs/l0_build_eval`, `runs/l0_sabotage_eval_tampered` with manifest/results/summary
- Edits: `notes/mental_map_v1.md`, `course/assignments/kata_01.py`, `tests/test_kata_01.py`

---

## Step-by-step

### Step 1 - Build: Execute a Clean Loop A Evaluation

You are creating a clean baseline run so later comparisons have a trustworthy reference.

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

### Step 2 - Sabotage: Introduce Dataset Tampering (Locked Room Violation)

You intentionally change one label to show how metrics can move without any policy change.

```bash
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Manually modify exactly ONE record's expected_answer in the tampered file.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

### Step 3 - Reflect: Invoke the Gate to Assess Comparability

You use the gate to prove the runs are not comparable; the expected REJECT is the guardrail.

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

The expected output is **REJECT** with diagnostic information indicating Locked Room incompatibility.

---

## Capstone Tasks

Purpose: document the mental model and lock concepts with a small kata.

**Permitted Modifications:** Files under `notes/`, `course/assignments/`, `tests/`, and new files under `data/`.
Scorer code modifications are not permitted at this level.

### Task A — Mental Model Documentation

Create `notes/mental_map_v1.md` (1–2 pages):

* Define the following terms: reward, metric, objective, loss, policy, environment
* Diagram Loop A/B/C with the primary variable identified for each

### Task B — Debugging Kata (Automated Assessment)

Create `course/assignments/kata_01.py`:

```python
def classify(outcome_code: str) -> str:
    """
    Return one of:
      - model_math
      - model_format
      - data_invalid
      - unknown
    """
```

Create `tests/test_kata_01.py` with appropriate test cases. Minimum required mappings:

* `"wrong_answer"` → `model_math`
* Any parse-format code (e.g., `"missing_prefix"`, `"extra_whitespace"`, `"not_single_line"`, `"leading_zeros"`) → `model_format`
* `"invalid_example"` → `data_invalid`

Execute verification:

```bash
pytest -q
```

---

## Completion Criteria

* Given two evaluation runs with differing results, can you enumerate the valid causes and provide evidence from `manifest.json`?
* Can you articulate the distinction between a **format failure** and a **mathematical failure** using outcome codes?
* Can you explain why "the tampered dataset improved pass_rate" does not constitute a valid improvement?

---

# Level 0.5: Selection Without Learning — Loop B

## Conceptual Foundation

**Incorrect intuition:** "pass@N improved, therefore the model learned."

**Correct mental model:** "Loop B modifies the decision rule over samples (selection compute). The underlying policy distribution remains unchanged."


## First-run checklist

- Inputs: `data/datasets/math_dev.jsonl`, `data/rollouts/selection_pack_dev.jsonl`
- Outputs: `runs/l0_5_build_sel_n4`, `runs/l0_5_sabotage_sel_random_*`, `runs/l0_5_fixed_sel_n4`
- Edits: `course/assignments/selection_policy.py`, `tests/test_selection_policy.py`

---

## Step-by-step

### Step 1 - Build: Execute Selection Demo on Provided Data

You are establishing a baseline selection run to compare against later changes.

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

### Step 2 - Sabotage: Introduce Nondeterminism into Tie-Breaking

You inject randomness to show how nondeterministic tie-breaks create unstable artifacts.

Edit `course/assignments/selection_policy.py`.

Modify `tie_break_key` to incorporate explicit randomness:

```python
import random

def tie_break_key(sample):
    return (random.random(), sample.completion)
```

Execute the identical command five times:

```bash
for i in 1 2 3 4 5; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/l0_5_sabotage_sel_random_$i
done
```

### Step 3 - Reflect: Demonstrate Variance Through Artifact Analysis

You confirm nondeterminism by showing that run outputs hash differently.

Compute a hash of each `results.jsonl`. If selection is nondeterministic, hashes will differ:

```bash
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

---

## Capstone Tasks

Purpose: repair the selection policy and lock determinism with tests.

**Permitted Modifications:** `course/assignments/selection_policy.py` and associated tests only.

### Repair and Lock

1. Remove randomness. Implement deterministic tie-breaking with the following precedence:

   * Higher `sum_logprob` (if present)
   * Shorter completion length
   * Lexicographic ordering of text

Example deterministic key:

```python
def tie_break_key(sample):
    lp = sample.sum_logprob
    lp_key = 0.0 if lp is None else -float(lp)   # negative to sort higher logprob first
    return (lp_key, len(sample.completion), sample.completion)
```

2. Create `tests/test_selection_policy.py`:

   * Determinism test: repeated invocations select the identical index
   * Tie-break test: given two equal-reward samples, the intended one is selected

Execute verification:

```bash
pytest -q
python -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l0_5_fixed_sel_n4
```

---

## Completion Criteria

* Why can Loop B improve outputs without modifying the policy distribution?
* What specific variable was modified during sabotage, and how was this demonstrated through artifact analysis?
* Why must selection code be deterministic in production systems?

---

# Level 1: REINFORCE Algorithm Analysis — Loop C (Simplified Domain)

## Conceptual Foundation

**Incorrect intuition:** "RL is supervised learning with additional complexity."

**Correct mental model:** "RL adjusts probability distributions. The update direction is controlled by advantage = reward − baseline. Baselines reduce variance without altering the optimization objective."


## First-run checklist

- Inputs: none (synthetic bandit)
- Outputs: `runs/l1_build_bandit`, `runs/l1_sabotage_lr2`, `runs/l1_sabotage_lrneg`, `runs/l1_build_slow`, `runs/l1_sabotage_slow_lrneg`
- Edits: `notes/reinforce_forensics.md`

---

## Step-by-step

### Step 1 - Build: Stable Learning with Baseline

You establish a healthy learning baseline so you can recognize later failures.

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

### Step 2 - Sabotage: Over-Optimization and Inverted Learning

You intentionally destabilize learning to observe two distinct failure modes.

Execute two sabotage experiments:

**(A) Excessive Learning Rate**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
```

**(B) Negative Learning Rate (Inverted Update Direction)**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

### Step 3 - Reflect: Step-by-Step Mechanism Observation

Slow mode lets you inspect how advantage signs change probabilities step by step.

```bash
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

---

## Capstone Tasks

Purpose: write a short forensics note linking advantage signs to update direction.

**Permitted Modifications:** Files under `notes/` and `course/assignments/`. Modification of `course/bandit_train.py` is not permitted.

Create `notes/reinforce_forensics.md`:

* Select 10 consecutive entries from the `--slow` output
* For each entry, annotate:

  * Action, reward, baseline, advantage sign
  * The required change to that action's probability

Conclude with one paragraph explaining the behavioral differences observed in sabotage runs.

---

## Completion Criteria

* If advantage is negative, what occurs to the probability of the sampled action?
* Why does a baseline reduce noise without altering the learning direction?
* What did negative learning rate demonstrate regarding probability adjustment?

---

# Level 2: Credit Assignment and Token-Text Boundary

## Conceptual Foundation

**Incorrect intuition:** "Only the terminal token matters because reward is assigned at sequence end."

**Correct mental model:** "In sequential generation, early decisions constrain subsequent states. Terminal reward becomes a training signal propagated across the sampled trajectory (credit assignment). Additionally: policy operates in token space; verification operates in text space."


## First-run checklist

- Inputs: none (token inspection uses fixed strings)
- Outputs: `runs/l2_build_two_step`, `runs/l2_sabotage_no_credit_step1`, `runs/l2_fixed_two_step`
- Edits: `course/assignments/two_step_mdp_demo.py`, `notes/credit_assignment.md`

---

## Step-by-step

### Step 1 - Build: Token Boundary Observation

You observe how small formatting changes create different tokenizations, which explains why format rules matter.

```bash
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

---

## Capstone Tasks

Purpose: build a two-step MDP demo, break it, then fix it and explain why.

**Permitted Modifications:** Create `course/assignments/two_step_mdp_demo.py` and associated notes. Scorer modification is not permitted.

### Step 2 - Build: Correct Two-Step REINFORCE Implementation

You implement a two-step environment so credit assignment across time steps is visible.

Create `course/assignments/two_step_mdp_demo.py`:

* Step 1: Select A or B
* Step 2: Select X or Y conditioned on step 1 outcome
* Reward assigned only at termination
* Update **both** step policies using the terminal reward (baseline optional)

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```

### Step 3 - Sabotage: Remove Credit Assignment for Step 1

You remove step-1 updates to show what breaks when credit assignment is missing.

Modify the script such that **only step 2 receives updates** (step 1 remains uniformly random):

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```

### Step 4 - Repair: Restore Step 1 Updates

You restore step-1 updates to recover learning and compare artifacts.

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```

Create `notes/credit_assignment.md`:

Explain in plain language what failed during sabotage and why the repair works. Include:

* A short description of the failure and its cause
* Include a diagram: tokens → detokenization → text → parsing → reward

---

## Completion Criteria

* Why can step 1 influence outcomes even when reward is assigned at termination?
* What distinguishes macro-actions from micro-actions in the LLM context?
* Why is parsing/formatting considered part of the environment boundary?

---

# Level 3: KL Divergence as Regularization Constraint

## Conceptual Foundation

**Incorrect intuition:** "KL divergence is purely mathematical abstraction."

**Correct mental model:** "KL divergence functions as a practical constraint: maximize reward while penalizing excessive divergence from a reference distribution."


## First-run checklist

- Inputs: none (demo uses synthetic policies)
- Outputs: `runs/l3_build_kl_demo`, `runs/l3_build_kl_choice.txt`, `runs/l3_sabotage_no_kl.txt`
- Edits: `course/assignments/kl_regularized_choice.py`, `notes/kl_tradeoff.md`

---

## Step-by-step

### Step 1 - Build: Execute KL Tradeoff Demonstration

You run the demo to see the reward vs KL tradeoff before writing code.

```bash
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

---

## Capstone Tasks

Purpose: implement a tiny KL-regularized choice rule, break it, and explain the difference.

**Permitted Modifications:** Create assignment scripts and notes. Core demonstration code modification is not permitted.

### Step 2 - Build: KL-Regularized Selection on Synthetic Data

You implement the selection rule so the KL penalty is concrete.

Create `course/assignments/kl_regularized_choice.py`:

* Define 6 candidate options, each with `(reward, kl)` values
* Implement selection rule: maximize `reward - beta * kl`
* Output selected candidate for `beta = 0.1` and `beta = 1.0`

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```

### Step 3 - Sabotage: Remove Regularization Constraint (beta = 0)

You remove the constraint to see how the choice shifts when KL is ignored.

Modify the script to set beta = 0:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```

### Step 4 - Reflect

You document why the unconstrained choice is tempting but risky.

Create `notes/kl_tradeoff.md`:

* Explain why beta = 0 represents an attractive but problematic configuration
* Connect this to why unconstrained optimization tends toward extreme, high-divergence solutions

---

## Completion Criteria

* What does KL divergence constrain in operational terms?
* Why can removing KL preference lead to reward-exploiting behavior even when measured reward increases?
* What constitutes the "reference" in this regularization framework?

---

# Level 4: Reward Specification as Production Code

## Conceptual Foundation

**Incorrect intuition:** "The scorer is merely an assertion statement."

**Correct mental model:** "The scorer defines the task specification and must be treated as production measurement software."


## First-run checklist

- Inputs: `data/datasets/math_dev.jsonl`, `data/golden/golden_correct.jsonl`, `data/golden/golden_exploits.jsonl`
- Outputs: `data/golden/golden_exploits_extra.jsonl`, `tests/test_reward_regressions.py`
- Edits: `course/core/scoring.py`, `notes/reward_spec_blackbox.md`

---

## Step-by-step

### Step 1 - Build: Validate Scorer Against Golden Test Cases

You establish the current scorer behavior so later changes are measurable.

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_correct.jsonl

python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

---

## Capstone Tasks

Purpose: probe the scorer, introduce a deliberate weakness, then restore and lock it.

**Permitted Modifications:** `course/core/scoring.py`, `tests/`, `data/golden/`, `notes/`.
Selection and learning code modifications are not permitted at this level.

### Step 2 - Build: Black-Box Specification Probing

You infer the spec by observation, which is how you reason about black-box verifiers.

Create `notes/reward_spec_blackbox.md`:

* Document the format rules you believe exist based on observed behavior
* Include 8 probe strings with predicted outcome codes

### Step 3 - Sabotage: Relax One Specification Rule

You weaken one rule to see how exploits slip through.

Select one strict rule in `course/core/scoring.py` and deliberately weaken it (examples):

* Permit additional whitespace after `Final:`
* Permit leading zeros
* Permit multi-line outputs

Execute golden validation and observe failures:

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

### Step 4 - Repair and Lock: Restore Strictness and Expand Test Coverage

You restore the spec and lock it with tests and golden cases.

1. Restore correct specification behavior.
2. Create `tests/test_reward_regressions.py` with **at least 6** test cases:

   * 2 verified correct cases
   * 4 exploit/edge cases (format manipulation)
3. Create `data/golden/golden_exploits_extra.jsonl` with **at least 5** additional exploit cases.
4. If reward behavior differs from baseline, increment `SCORER_VERSION`.

Execute verification:

```bash
pytest -q
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits_extra.jsonl
```

---

## Completion Criteria

* What constitutes a false positive versus false negative in this verification context?
* Why must the scorer be versioned upon behavioral change, invalidating comparisons with prior run metrics?
* Why is "more lenient parsing" typically counterproductive?

---

# Level 5: Reward Exploitation Analysis

## Conceptual Foundation

**Incorrect intuition:** "Reward hacking indicates a malicious optimizer."

**Correct mental model:** "Optimizers exploit proxy measures. If the specification contains loopholes, optimization will discover them. The solution is to patch mechanism classes and lock them with tests."


## First-run checklist

- Inputs: optional baseline uses `data/datasets/math_dev.jsonl`, `data/rollouts/frozen_rollouts_dev.jsonl`
- Outputs: `runs/l5_build_eval_context` (optional)
- Edits: `course/assignments/hackable_scorer_demo.py`, `notes/red_team_report.md`

---

## Step-by-step

### Step 1 - Build: Establish Baseline Context (Optional)

You refresh context on how the current evaluator behaves; this is optional baseline evidence.

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l5_build_eval_context

python -m course.inspect_run --run runs/l5_build_eval_context --only-fails --top-k 5 --show 2
```

---

## Capstone Tasks

Purpose: build a naive verifier, exploit it, then patch the exploit class.

**Permitted Modifications:** New assignment files, notes, and tests. Modifications to the production scorer should be treated as instrument changes requiring version increment.

### Step 2 - Build: Implement a Deliberately Naive Verifier

You create a deliberately weak verifier so you can study how it fails.

Create `course/assignments/hackable_scorer_demo.py`:

* Assign reward = 1 if the expected integer appears anywhere in the completion (naive substring/regex approach)
* Demonstrate that it produces correct results on honest completions

### Step 3 - Sabotage: Generate 5 Exploits

You craft exploit completions that pass the naive check without solving the problem.

Create 5 completions that achieve reward = 1 without constituting valid solutions (e.g., number enumeration).

### Step 4 - Repair: Patch the Exploit Class

You patch the exploit class (not just the specific strings) and document the fix.

Modify the demonstration verifier to close the exploit class (not merely the specific exploit strings).

Create `notes/red_team_report.md`:

* Document the 5 exploit strings
* Explain why they succeeded
* Describe the patch strategy
* Specify what tests would prevent regression

---

## Completion Criteria

* Provide a concrete example of "proxy metric increases, true objective decreases."
* What does "patch the class, not the instance" mean?
* Why does this analysis matter for subsequent Loop C work?

---

# Level 6: Promotion Committee (Three Controlled Experimental Traps)

## Conceptual Foundation

**Incorrect intuition:** "Higher metric value equals improvement."

**Correct mental model:** "A promotable improvement must satisfy Locked Room comparability and be attributable to exactly one controlled variable."


## First-run checklist

- Inputs: `runs/l0_build_eval`, `runs/l0_sabotage_eval_tampered`, `data/rollouts/selection_pack_dev.jsonl`
- Outputs: `runs/l6_trap_sel_n1`, `runs/l6_trap_sel_n4`, `runs/l6_trap_learning`
- Edits: `notes/promotion_memo.md`

---

## Step-by-step

### Step 1 - Trap 1 (Sabotage): Invalid Promotion via Instrument Change

You attempt a promotion based on a Locked Room violation to show why the gate exists.

Use the previously tampered dataset run (or create a new one), then invoke the gate:

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

### Step 2 - Trap 2: Selection Improvement Without Learning

You show how metrics can improve without any learning, so promotion is not justified.

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 1 \
  --outdir runs/l6_trap_sel_n1

python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l6_trap_sel_n4
```

### Step 3 - Trap 3: Learning Modifies Behavior

You show a real learning run, then argue promotion based on evidence and holdout risk.

```bash
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

---

## Capstone Tasks

Purpose: write a short promotion memo that argues from artifacts, not intuition.

Create `notes/promotion_memo.md` (maximum 1 page):

* For each trap, document:

  * Loop classification (A/B/C)
  * Variable modified
  * Comparison validity assessment
  * Evidence (quotation from `manifest.json` / scorer version / parameter modification)
* Concluding paragraph: what would you PROMOTE versus REJECT today, with justification

---

## Completion Criteria

* If pass_rate increased, how do you demonstrate it is not a Locked Room violation?
* How do you distinguish selection improvement from learning improvement using artifacts exclusively?
* Why does Loop C inevitably create novel failure modes that frozen rollouts cannot fully reveal?

---

## Track Completion Summary

Successful completion of this track establishes the core competency this course targets: the ability to execute and interpret RL-for-LLMs experiments without conflating **measurement**, **selection**, and **learning**, and without accepting methodologically invalid comparisons.

---

## Appendix

### Rubric (what “correct” looks like)

Use this as a checklist to evaluate your work without external help. If a row fails, investigate artifacts and fix before moving on.

#### Level 0 — Loop A (Measurement hygiene)
- **Build run:** `pass_rate` in `runs/l0_build_eval/summary.json` is between 0 and 1.
- **Sabotage run:** `runs/l0_sabotage_eval_tampered` has a different dataset hash than the build run (see `manifest.json`).
- **Gate:** `python -m course.gate` outputs `REJECT` with a Locked Room violation.
- **Mental map:** `notes/mental_map_v1.md` defines required terms and includes A/B/C distinctions.
- **Kata:** `classify()` maps format errors to `model_format` and `"wrong_answer"` to `model_math`.

#### Level 0.5 — Loop B (Selection)
- **Build:** `pass_at_n` > `pass_at_1` in `runs/l0_5_build_sel_n4/summary.json`.
- **Sabotage:** hashes of `results.jsonl` differ across repeated runs with random tie-breaks.
- **Repair:** hashes match across repeated runs; selection deterministic.
- **Tie-break policy:** uses logprob → length → lexicographic order.

#### Level 1 — Loop C (REINFORCE)
- **Build:** `mean_reward` improves over time (see `summary.json` or slow logs).
- **Sabotage (lr negative):** `mean_reward` is much lower than the build run.
- **Forensics:** `notes/reinforce_forensics.md` shows correct advantage sign reasoning.

#### Level 2 — Credit assignment + token boundary
- **Token inspect:** format variants tokenize differently.
- **Two-step build:** `mean_reward` near 1.0 and step‑1 policy becomes non‑uniform.
- **Sabotage:** with no step‑1 updates, step‑1 policy stays ~uniform and `mean_reward` drops.
- **Repair:** build behavior returns.

#### Level 3 — KL tradeoff
- **KL demo:** `runs/l3_build_kl_demo/kl_tradeoff.csv` exists.
- **KL choice:** chosen action differs for `beta=0.1` vs `beta=1.0`.
- **Notes:** `notes/kl_tradeoff.md` explains why `beta=0` is risky.

#### Level 4 — Scorer as production code
- **Golden tests:** both golden sets pass after repair.
- **Sabotage evidence:** one exploit fails (incorrectly passes) when a rule is weakened.
- **Lock:** regression tests include both correct cases and exploit cases.

#### Level 5 — Reward exploitation
- **Naive verifier:** exploit strings receive reward under naive rules.
- **Patched verifier:** same exploit strings fail.
- **Report:** `notes/red_team_report.md` lists exploits and patch strategy.

#### Level 6 — Promotion committee
- **Trap 1:** REJECT with Locked Room violation evidence.
- **Trap 2:** selection improvement without learning; should be rejected for promotion.
- **Trap 3:** learning run shows improved reward; promotion is conditional on holdout.

### Run Documentation Template

The following template should be completed for each experimental run in `runs/<name>/README.md`:

* **Loop:** A / B / C
* **Variable Modified (select exactly one):** measurement instrument / selection compute / policy / environment
* **Locked Room Evidence:** (dataset fingerprints + scorer version; direct quotation from manifest)
* **Expected Outcome:**
* **Observed Outcome (quantitative):**
* **Two Concrete Examples:** (example id + outcome code + observed behavior)
* **Repair Plan:** (single sentence)
