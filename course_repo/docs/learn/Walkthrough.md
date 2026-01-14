# RL Foundations for LLMs — Instructor Guide

## Session Overview

**Duration:** Approximately 45–60 minutes  
**Format:** Hands-on terminal demonstration with interactive discussion  
**Learning Objectives:** Establish the foundational conceptual framework before students encounter the codebase

---

## Pre-Session Preparation

### Technical Requirements

Verify the following conditions are satisfied before the session begins:

```bash
cd path/to/rl-foundations-llm
poetry run python -m course.eval --help
poetry run python -m course.selection_demo --help
poetry run python -m course.bandit_train --help
```

If any command fails, ensure Python environment configuration is complete.

### Cognitive Preparation

Students frequently arrive with the following mental model:

> "We will simply instruct the model to improve and measure the results."

The session objective is to replace this intuition with the following framework:

> "Measurement, selection, and learning constitute three distinct operations with fundamentally different mechanisms and implications."

---

## Session Progression

### Segment 1: Loop A — Measurement as Scientific Instrumentation (15 minutes)

#### Conceptual Foundation

**Instructor Opening Statement:**

> "Before any discussion of improvement, we must establish what we are measuring and how. Loop A concerns measurement exclusively."

**Key Conceptual Points:**

- A scorer functions as a scientific instrument, not merely a utility function
- When metrics change between runs, either the policy changed OR the measurement conditions changed
- It is the experimenter's responsibility to determine which

#### Live Demonstration

Execute the following sequence:

```bash
poetry run python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/demo_eval_clean
```

Navigate to the output directory and examine artifacts:

```bash
cat runs/demo_eval_clean/manifest.json
head -5 runs/demo_eval_clean/results.jsonl
cat runs/demo_eval_clean/summary.json
```

**Discussion Points:**

- `manifest.json` records experimental conditions (analogous to laboratory notebook documentation)
- `results.jsonl` contains per-example outcomes
- `summary.json` provides aggregate statistics

#### Perturbation Demonstration

Introduce a controlled modification to the dataset:

```bash
cp data/datasets/math_dev.jsonl /tmp/math_dev_modified.jsonl
# Modify one expected_answer in the temporary file

poetry run python -m course.eval \
  --dataset /tmp/math_dev_modified.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/demo_eval_modified
```

Execute the gate comparison:

```bash
poetry run python -m course.gate \
  --baseline runs/demo_eval_clean \
  --candidate runs/demo_eval_modified \
  --min-delta 0.00
```

**Expected Outcome:** REJECT with diagnostic information indicating Locked Room violation.

**Instructor Explanation:**

> "The gate rejected this comparison. Why? Because the experimental conditions differed. The metric changed, but not due to any change in the policy. This is precisely the confusion we must prevent."

---

### Segment 2: Loop B — Selection Without Learning (15 minutes)

#### Conceptual Foundation

**Instructor Statement:**

> "Loop B demonstrates that apparent improvement is achievable without any modification to model parameters."

**Key Conceptual Points:**

- Best-of-N selection chooses among existing samples
- The underlying probability distribution remains unchanged
- Improved outcomes result from increased computational expenditure, not learned behavior

#### Live Demonstration

Execute selection with N=1:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 1 \
  --outdir runs/demo_sel_n1
```

Execute selection with N=4:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/demo_sel_n4
```

Compare the results:

```bash
cat runs/demo_sel_n1/summary.json
cat runs/demo_sel_n4/summary.json
```

**Instructor Analysis:**

> "The pass rate increased from N=1 to N=4. Has the model learned anything? No. We simply examined more samples and selected the best one. The probability distribution that generated these samples is unchanged."

#### Interactive Verification

Pose the following question to students:

> "If we reset and run N=4 selection again, will we obtain identical results?"

Execute verification:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/demo_sel_n4_repeat

diff runs/demo_sel_n4/results.jsonl runs/demo_sel_n4_repeat/results.jsonl
```

**Expected Outcome:** No difference. Selection is deterministic given identical inputs.

---

### Segment 3: Loop C — Learning Modifies Distribution (15 minutes)

#### Conceptual Foundation

**Instructor Statement:**

> "Loop C is where probability distributions change. This is actual learning."

**Key Conceptual Points:**

- Policy gradient methods modify the probability distribution over actions
- The update direction is determined by the advantage: reward minus baseline
- This is fundamentally different from selection—the generator itself changes

#### Live Demonstration

Execute bandit training in verbose mode:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/demo_bandit_slow
```

**Instructor Analysis:**

Direct attention to the step-by-step output:

> "Observe the log entries. For each step, note: which action was sampled, what reward was received, what the baseline value was, and what the resulting advantage is."

> "When advantage is positive, the probability of that action increases. When advantage is negative, the probability decreases. This is the REINFORCE gradient."

#### Perturbation Demonstration

Execute training with inverted learning rate:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/demo_bandit_negative
```

**Instructor Analysis:**

> "The learning rate is negative. What happens? The update direction reverses. Actions with positive advantage have their probabilities decreased rather than increased. The system learns the inverse of the intended objective."

---

### Segment 4: Integration and Synthesis (10–15 minutes)

#### Summary Framework

Present the following taxonomy:

| Loop | Primary Mechanism | What Changes | What Remains Constant |
|------|-------------------|--------------|----------------------|
| A | Measurement | Recorded metrics | Policy, environment, instrument |
| B | Selection | Observed outputs | Underlying distribution |
| C | Learning | Probability distribution | (Potentially nothing—this is the fundamental change) |

#### Locked Room Principle

**Instructor Statement:**

> "Before comparing any two experimental runs, verify that the following conditions are identical: dataset, scorer version, sampling configuration. If any differ, the comparison is methodologically invalid."

Demonstrate by examining manifests:

```bash
cat runs/demo_eval_clean/manifest.json | jq '.dataset_fingerprint, .scorer_version'
cat runs/demo_eval_modified/manifest.json | jq '.dataset_fingerprint, .scorer_version'
```

---

## Common Student Misconceptions

### Misconception: "Higher pass@N means the model improved"

**Correction:** pass@N improvement may indicate selection (Loop B) rather than learning (Loop C). Verify by examining whether policy parameters changed.

### Misconception: "The scorer is just a grading function"

**Correction:** The scorer defines the optimization target. Poorly specified scorers lead to reward exploitation. The scorer must be treated as production code.

### Misconception: "We can compare any two runs with the same metric"

**Correction:** Comparisons are valid only under Locked Room conditions. Different datasets, scorer versions, or sampling configurations invalidate comparison.

---

## Session Conclusion

### Student Takeaways

Ensure students depart with understanding of:

1. The three loops (A/B/C) represent categorically different operations
2. Locked Room conditions are prerequisites for valid experimental comparison
3. The scorer functions as the specification of what behavior we seek

### Transition to Project Work

**Instructor Closing Statement:**

> "The project exercises will require you to deliberately violate these principles, observe the resulting failures, and then implement corrections. The Build-Sabotage-Reflect-Repair cycle is designed to transform theoretical understanding into operational competence."

---

## Appendix: Diagnostic Commands

### Environment Verification

```bash
poetry run python --version
poetry run python -c "import course; print('Course package available')"
ls data/datasets/
ls data/rollouts/
```

### Quick Reset

```bash
rm -rf runs/demo_*
```

### Artifact Inspection

```bash
poetry run python -m course.inspect_run --run <run_dir> --top-k 5 --show 2
poetry run python -m course.inspect_run --run <run_dir> --only-fails --top-k 5 --show 2
```
