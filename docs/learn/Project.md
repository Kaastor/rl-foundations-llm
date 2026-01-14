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

## Assignment Descriptions

## Rubric (what “correct” looks like)

Use this as a checklist to evaluate your work without external help. If a row fails, investigate artifacts and fix before moving on.

### Level 0 — Loop A (Measurement hygiene)
- **Build run:** `pass_rate` in `runs/l0_build_eval/summary.json` is between 0 and 1.
- **Sabotage run:** `runs/l0_sabotage_eval_tampered` has a different dataset hash than the build run (see `manifest.json`).
- **Gate:** `python -m course.gate` outputs `REJECT` with a Locked Room violation.
- **Mental map:** `notes/mental_map_v1.md` defines required terms and includes A/B/C distinctions.
- **Kata:** `classify()` maps format errors to `model_format` and `"wrong_answer"` to `model_math`.

### Level 0.5 — Loop B (Selection)
- **Build:** `pass_at_n` > `pass_at_1` in `runs/l0_5_build_sel_n4/summary.json`.
- **Sabotage:** hashes of `results.jsonl` differ across repeated runs with random tie-breaks.
- **Repair:** hashes match across repeated runs; selection deterministic.
- **Tie-break policy:** uses logprob → length → lexicographic order.

### Level 1 — Loop C (REINFORCE)
- **Build:** `mean_reward` improves over time (see `summary.json` or slow logs).
- **Sabotage (lr negative):** `mean_reward` is much lower than the build run.
- **Forensics:** `notes/reinforce_forensics.md` shows correct advantage sign reasoning.

### Level 2 — Credit assignment + token boundary
- **Token inspect:** format variants tokenize differently.
- **Two-step build:** `mean_reward` near 1.0 and step‑1 policy becomes non‑uniform.
- **Sabotage:** with no step‑1 updates, step‑1 policy stays ~uniform and `mean_reward` drops.
- **Repair:** build behavior returns.

### Level 3 — KL tradeoff
- **KL demo:** `runs/l3_build_kl_demo/kl_tradeoff.csv` exists.
- **KL choice:** chosen action differs for `beta=0.1` vs `beta=1.0`.
- **Notes:** `notes/kl_tradeoff.md` explains why `beta=0` is risky.

### Level 4 — Scorer as production code
- **Golden tests:** both golden sets pass after repair.
- **Sabotage evidence:** one exploit fails (incorrectly passes) when a rule is weakened.
- **Lock:** regression tests include both correct cases and exploit cases.

### Level 5 — Reward exploitation
- **Naive verifier:** exploit strings receive reward under naive rules.
- **Patched verifier:** same exploit strings fail.
- **Report:** `notes/red_team_report.md` lists exploits and patch strategy.

### Level 6 — Promotion committee
- **Trap 1:** REJECT with Locked Room violation evidence.
- **Trap 2:** selection improvement without learning; should be rejected for promotion.
- **Trap 3:** learning run shows improved reward; promotion is conditional on holdout.

---

## Run Documentation Template

The following template should be completed for each experimental run in `runs/<name>/README.md`:

* **Loop:** A / B / C
* **Variable Modified (select exactly one):** measurement instrument / selection compute / policy / environment
* **Locked Room Evidence:** (dataset fingerprints + scorer version; direct quotation from manifest)
* **Expected Outcome:**
* **Observed Outcome (quantitative):**
* **Two Concrete Examples:** (example id + outcome code + observed behavior)
* **Repair Plan:** (single sentence)

---

# Level 0: Measurement Hygiene — Loop A

## Conceptual Foundation

**Incorrect intuition:** "The model is performing poorly."

**Correct mental model:** "Loop A constitutes a measurement instrument verification. If the measured value changed, either (a) the policy changed, or (b) the measurement conditions changed. The objective is to determine which."

---

## Procedural Steps

### Build Phase: Execute a Clean Loop A Evaluation

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

### Sabotage Phase: Introduce Dataset Tampering (Locked Room Violation)

```bash
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Manually modify exactly ONE record's expected_answer in the tampered file.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

### Reflect Phase: Invoke the Gate to Assess Comparability

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

The expected output is **REJECT** with diagnostic information indicating Locked Room incompatibility.

---

## Capstone Tasks

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

---

## Procedural Steps

### Build Phase: Execute Selection Demo on Provided Data

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

### Sabotage Phase: Introduce Nondeterminism into Tie-Breaking

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

### Reflect Phase: Demonstrate Variance Through Artifact Analysis

Compute a hash of each `results.jsonl`. If selection is nondeterministic, hashes will differ:

```bash
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

---

## Capstone Tasks

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

---

## Procedural Steps

### Build Phase: Stable Learning with Baseline

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

### Sabotage Phase: Over-Optimization and Inverted Learning

Execute two sabotage experiments:

**(A) Excessive Learning Rate**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
```

**(B) Negative Learning Rate (Inverted Update Direction)**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

### Reflect Phase: Step-by-Step Mechanism Observation

```bash
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

---

## Capstone Tasks

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

---

## Procedural Steps

### Build Phase: Token Boundary Observation

```bash
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

---

## Capstone Tasks

**Permitted Modifications:** Create `course/assignments/two_step_mdp_demo.py` and associated notes. Scorer modification is not permitted.

### Build: Correct Two-Step REINFORCE Implementation

Create `course/assignments/two_step_mdp_demo.py`:

* Step 1: Select A or B
* Step 2: Select X or Y conditioned on step 1 outcome
* Reward assigned only at termination
* Update **both** step policies using the terminal reward (baseline optional)

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```

### Sabotage: Remove Credit Assignment for Step 1

Modify the script such that **only step 2 receives updates** (step 1 remains uniformly random):

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```

### Repair: Restore Step 1 Updates

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```

Create `notes/credit_assignment.md`:

* Explain what failed during sabotage and the underlying cause
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

---

## Procedural Steps

### Build Phase: Execute KL Tradeoff Demonstration

```bash
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

---

## Capstone Tasks

**Permitted Modifications:** Create assignment scripts and notes. Core demonstration code modification is not permitted.

### Build: KL-Regularized Selection on Synthetic Data

Create `course/assignments/kl_regularized_choice.py`:

* Define 6 candidate options, each with `(reward, kl)` values
* Implement selection rule: maximize `reward - beta * kl`
* Output selected candidate for `beta = 0.1` and `beta = 1.0`

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```

### Sabotage: Remove Regularization Constraint (beta = 0)

Modify the script to set beta = 0:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```

### Reflect

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

---

## Procedural Steps

### Build Phase: Validate Scorer Against Golden Test Cases

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

**Permitted Modifications:** `course/core/scoring.py`, `tests/`, `data/golden/`, `notes/`.
Selection and learning code modifications are not permitted at this level.

### Build: Black-Box Specification Probing

Create `notes/reward_spec_blackbox.md`:

* Document the format rules you believe exist based on observed behavior
* Include 8 probe strings with predicted outcome codes

### Sabotage: Relax One Specification Rule

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

### Repair and Lock: Restore Strictness and Expand Test Coverage

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

---

## Procedural Steps

### Build Phase: Establish Baseline Context (Optional)

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l5_build_eval_context

python -m course.inspect_run --run runs/l5_build_eval_context --only-fails --top-k 5 --show 2
```

---

## Capstone Tasks

**Permitted Modifications:** New assignment files, notes, and tests. Modifications to the production scorer should be treated as instrument changes requiring version increment.

### Build: Implement a Deliberately Naive Verifier

Create `course/assignments/hackable_scorer_demo.py`:

* Assign reward = 1 if the expected integer appears anywhere in the completion (naive substring/regex approach)
* Demonstrate that it produces correct results on honest completions

### Sabotage: Generate 5 Exploits

Create 5 completions that achieve reward = 1 without constituting valid solutions (e.g., number enumeration).

### Repair: Patch the Exploit Class

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

---

## Procedural Steps

### Trap 1 (Sabotage): Invalid Promotion via Instrument Change

Use the previously tampered dataset run (or create a new one), then invoke the gate:

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

### Trap 2: Selection Improvement Without Learning

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

### Trap 3: Learning Modifies Behavior

```bash
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

---

## Capstone Tasks

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
