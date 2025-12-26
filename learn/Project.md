# RL Foundations for LLMs — Mastery Track Project Roadmap

**Project:** *Verifier-Driven Math QA — Build a “working car,” then sabotage it, then harden it.*

This track is designed to “force” understanding by making you repeatedly do the same cycle:

1. **Build (Working Car):** get a clean, deterministic baseline that you fully own.
2. **Sabotage (Break It):** make one small, explicit change that breaks a core principle.
3. **Reflect (Forensics):** observe the break using artifacts (`manifest.json`, `summary.json`, `results.jsonl`) rather than vibes.
4. **Repair + Lock:** fix the *mechanism* and encode it as tests + gates so it can’t regress.

**Do not skip the Sabotage steps.** They’re the entire point.

---

## Core rules (memorize these early)

### The three loops (use this terminology)

* **Loop A (Eval):** measurement only (frozen inputs, deterministic scorer).
* **Loop B (Selection):** choose best-of-N (behavior improves; the underlying policy distribution does not).
* **Loop C (Learning):** update a policy (probabilities shift → distribution shifts).

### Locked Room Rule (valid comparisons)

Before you compare two run numbers, you must verify these are identical:

* dataset split (same file fingerprint in `manifest.json`)
* prompt text (if applicable)
* scorer **name + version**
* sampling settings (if applicable)

If any differ, you changed the environment/instrument. That comparison is invalid.

### Reward = Spec (treat `score()` like production code)

Your scorer is a measurement instrument. It must be:
**deterministic**, **total (never crashes)**, **explainable**, **fast**, **versioned**.

---

## Workflow discipline (strongly recommended)

Create a branch and make small commits so you can prove what changed.

```bash
git checkout -b mastery-track
pytest -q
```

For every level, do:

* Commit the **Build** state.
* Commit the **Sabotage** change (even though it’s “wrong”).
* Commit the **Fix**.

---

## Run note template (put this in every `runs/<name>/README.md`)

* **Loop:** A / B / C
* **Knob turned (pick ONE):** measurement instrument / selection compute / policy / environment
* **Locked Room evidence:** (dataset fingerprints + scorer version; quote from manifest)
* **What I expected:**
* **What happened (numbers):**
* **Two concrete examples:** (id + outcome code + what you saw)
* **Repair plan:** (one sentence)

---

# Level 0: Measurement Hygiene (Build → Tamper → Catch It) — Loop A

## 1) The Mindset Shift

**Wrong intuition:** “The model is dumb.”
**Correct mental model:** “Loop A is a measurement instrument check. If the number changed, either (a) the policy changed, or (b) the measurement conditions changed. My job is to prove which.”

## 2) Student actions (exact CLI)

### Build (working car): run a clean Loop A eval

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

### Sabotage (break it): tamper the dataset (Locked Room violation)

```bash
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Manually edit exactly ONE record’s expected_answer in the tampered file.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

### Reflect (forensics): force the gate to judge comparability

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

You should see **REJECT** with reasons that point to Locked Room incompatibility.

## 3) The Capstone Task (Skillset)

**Knob Budget:** Allowed changes: `notes/`, `course/assignments/`, `tests/`, new files under `data/`.
Do not change scorer code yet.

### Task A — Write the mental model

Create `notes/mental_map_v1.md` (1–2 pages):

* define: reward, metric, objective, loss, policy, environment
* draw Loop A/B/C and label the knob for each

### Task B — Debugging Kata (autograded)

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

Create `tests/test_kata_01.py` with a small answer key. Minimum required mapping:

* `"wrong_answer"` → `model_math`
* any parse-format code (e.g. `"missing_prefix"`, `"extra_whitespace"`, `"not_single_line"`, `"leading_zeros"`) → `model_format`
* `"invalid_example"` → `data_invalid`

Run:

```bash
pytest -q
```

## 4) The “Passed” Checklist

* If two eval runs differ, can you name the only valid causes and point to evidence in `manifest.json`?
* Can you explain the difference between a **format failure** and a **math failure** using outcome codes?
* Can you explain why “tampered dataset improved pass_rate” is not an improvement?

---

# Level 0.5: Selection Is Not Learning (Build → Randomize → Measure Variance → Fix) — Loop B

## 1) The Mindset Shift

**Wrong intuition:** “pass@N improved, so the model learned.”
**Correct mental model:** “Loop B changes the decision rule over samples (selection compute). The underlying policy distribution is unchanged.”

## 2) Student actions (exact CLI)

### Build (working car): run selection demo on the provided pack

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

### Sabotage (break it): inject nondeterminism into tie-breaking

Edit `course/assignments/selection_policy.py`.

Change `tie_break_key` to something explicitly random (on purpose):

```python
import random

def tie_break_key(sample):
    return (random.random(), sample.completion)
```

Now run the exact same command 5 times:

```bash
for i in 1 2 3 4 5; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/l0_5_sabotage_sel_random_$i
done
```

### Reflect (forensics): prove variance exists using run artifacts

Compute a hash of each `results.jsonl`. If selection is nondeterministic, hashes will differ.

```bash
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** Only change `course/assignments/selection_policy.py` and its tests.

### Repair + Lock

1. Remove randomness. Implement a deterministic tie-break that prefers:

   * higher `sum_logprob` (if present)
   * then shorter completion
   * then lexicographic text

Example deterministic key:

```python
def tie_break_key(sample):
    lp = sample.sum_logprob
    lp_key = 0.0 if lp is None else -float(lp)   # negative so “higher logprob” sorts first
    return (lp_key, len(sample.completion), sample.completion)
```

2. Add `tests/test_selection_policy.py`:

   * determinism test: repeated calls pick the same index
   * tie-break test: two equal-reward samples pick the intended one

Run:

```bash
pytest -q
python -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l0_5_fixed_sel_n4
```

## 4) The “Passed” Checklist

* Why can Loop B improve outputs without changing the policy distribution?
* What exact “knob” did you turn during sabotage, and how did you prove it using hashes/artifacts?
* Why does selection code need to be deterministic in production?

---

# Level 1: REINFORCE Under a Microscope (Build → Over-optimize → Diagnose) — Loop C (toy)

## 1) The Mindset Shift

**Wrong intuition:** “RL is supervised learning with extra steps.”
**Correct mental model:** “RL nudges probabilities. The update direction is controlled by advantage = reward − baseline. Baselines reduce variance, not the goal.”

## 2) Student actions (exact CLI)

### Build (working car): stable-ish learning with baseline

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

### Sabotage (break it): over-optimize and/or invert learning

Run two sabotage experiments:

**(A) Too aggressive learning rate**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
```

**(B) Negative learning rate (learn the wrong direction)**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

### Reflect (forensics): watch the step-by-step mechanism

```bash
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** You may create files under `notes/` and `course/assignments/`. Do not modify `course/bandit_train.py`.

Create `notes/reinforce_forensics.md`:

* pick 10 consecutive entries from the `--slow` output (copy/paste)
* for each entry, annotate:

  * action, reward, baseline, advantage sign
  * what must happen to that action’s probability next

Then write 1 paragraph explaining why the sabotage runs behave differently.

## 4) The “Passed” Checklist

* If advantage is negative, what happens to the probability of the sampled action?
* Why does a baseline reduce noise but not change the “direction” of learning?
* What did negative learning rate teach you about “probability nudging”?

---

# Level 2: Credit Assignment + Token/Text Boundary (Build → Remove Credit → Observe Failure → Fix)

## 1) The Mindset Shift

**Wrong intuition:** “Only the last token/action matters because reward is at the end.”
**Correct mental model:** “In sequential generation, early decisions shape later states. End reward becomes a training signal applied across the sampled trajectory (credit assignment). Also: policy acts on tokens; verifier acts on text.”

## 2) Student actions (exact CLI)

### Build (working car): observe token boundary

```bash
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** Create `course/assignments/two_step_mdp_demo.py` and notes. Do not change scorer.

### Build: implement correct two-step REINFORCE

Create `course/assignments/two_step_mdp_demo.py`:

* step 1 choose A/B
* step 2 choose X/Y conditioned on step1
* reward only at end
* update **both** step policies using the same end reward (with baseline optional)

Run:

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```

### Sabotage: remove credit assignment for step 1

Edit your script so **only step 2 gets updated** (step 1 stays random). Run again:

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```

### Reflect + Fix: restore updates for step 1 and re-run

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```

Write `notes/credit_assignment.md`:

* explain what failed in sabotage and why
* include one diagram: tokens → detokenize → text → parse → reward

## 4) The “Passed” Checklist

* Why can step 1 matter even when reward arrives at the end?
* What’s macro-action vs micro-action in an LLM context?
* Why is parsing/formatting part of the environment boundary?

---

# Level 3: KL as a Leash (Build → Drop the Leash → See Drift Preference)

## 1) The Mindset Shift

**Wrong intuition:** “KL is decorative math.”
**Correct mental model:** “KL is a practical leash: improve reward, but penalize moving too far from a reference distribution.”

## 2) Student actions (exact CLI)

### Build (working car): run the KL tradeoff demo

```bash
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** Create a small assignment script + notes. Do not modify core demo code.

### Build: implement KL-regularized selection on a tiny synthetic table

Create `course/assignments/kl_regularized_choice.py`:

* hardcode 6 “candidates,” each with `(reward, kl)`
* implement choice rule: maximize `reward - beta * kl`
* print chosen candidate for `beta = 0.1` and `beta = 1.0`

Run:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```

### Sabotage: remove the leash (beta = 0)

Modify the script so beta = 0 and re-run:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```

### Reflect

Create `notes/kl_tradeoff.md`:

* explain why “beta = 0” is an attractive disaster
* connect it to why unconstrained optimization tends to prefer extreme/high-drift solutions

## 4) The “Passed” Checklist

* What does KL constrain in plain English?
* Why can removing KL preference lead to reward-hack-y behavior (even if reward rises)?
* What is the “reference” in this mental model?

---

# Level 4: Reward = Spec (Build Golden Gates → Loosen Rule → Catch False Positives → Fix + Version)

## 1) The Mindset Shift

**Wrong intuition:** “The scorer is just an assertion.”
**Correct mental model:** “The scorer defines the task and must be treated like production measurement software.”

## 2) Student actions (exact CLI)

### Build (working car): validate the scorer against goldens

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_correct.jsonl

python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** Allowed changes: `course/core/scoring.py`, `tests/`, `data/golden/`, `notes/`.
(Do not touch selection or learning code in this level.)

### Build: black-box probes (before reading the code)

Create `notes/reward_spec_blackbox.md`:

* write the format rules you believe exist
* include 8 probe strings and predicted outcome codes

### Sabotage: loosen one rule in the scorer

Pick one strict rule in `course/core/scoring.py` and deliberately weaken it (examples):

* allow extra whitespace after `Final:`
* allow leading zeros
* allow multi-line outputs

Now re-run golden checks and observe failures:

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

### Repair + Lock: restore strictness, then expand tests

1. Restore the correct behavior.
2. Add `tests/test_reward_regressions.py` with **at least 6** cases:

   * 2 “known correct” cases
   * 4 exploit/edge cases (format tricks)
3. Add `data/golden/golden_exploits_extra.jsonl` with **at least 5** new exploit cases.
4. If reward behavior changed vs baseline, bump `SCORER_VERSION`.

Validate:

```bash
pytest -q
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits_extra.jsonl
```

## 4) The “Passed” Checklist

* What is a false positive vs false negative in this verifier?
* Why must you version the scorer on behavior change and stop comparing old run numbers?
* Why is “more lenient parsing” usually a trap?

---

# Level 5: Goodhart Dungeon (Build a Naive Verifier → Hack It → Patch the Class)

## 1) The Mindset Shift

**Wrong intuition:** “Reward hacking means the optimizer is evil.”
**Correct mental model:** “Optimizers exploit proxies. If the spec has loopholes, optimization will find them. Patch mechanisms, then lock with tests.”

## 2) Student actions (exact CLI)

### Build baseline context (optional but grounding)

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l5_build_eval_context

python -m course.inspect_run --run runs/l5_build_eval_context --only-fails --top-k 5 --show 2
```

## 3) The Capstone Task (Skillset)

**Knob Budget:** Add new assignment file(s) + notes + tests. Avoid editing the real scorer unless you treat it as an instrument change.

### Build: write a deliberately naive verifier demo

Create `course/assignments/hackable_scorer_demo.py`:

* reward = 1 if the expected integer appears anywhere in the completion (naive substring/regex style)
* show it “works” on honest completions

### Sabotage: generate 5 hacks

Create 5 cheating completions that score 1 without actually being a proper answer (e.g., number spray).

### Repair: patch the class of exploit

Modify the demo verifier logic to close the exploit class, not just one string.

Write `notes/red_team_report.md`:

* the 5 exploit strings
* why they worked
* the patch strategy
* what tests would lock it permanently

## 4) The “Passed” Checklist

* Give a concrete example of “proxy up, true goal down.”
* What does “patch the class, not the string” mean?
* Why does this matter for Loop C later?

---

# Level 6: Promotion Committee (Three Controlled Traps) — Integrate A/B/C

## 1) The Mindset Shift

**Wrong intuition:** “Higher number = better.”
**Correct mental model:** “A promotable improvement must be comparable under Locked Room and attributable to exactly one knob.”

## 2) Student actions (exact CLI)

### Trap 1 (Sabotage): invalid promotion by instrument change (Loop A comparability break)

Use your earlier tampered dataset run (or make a fresh one), then gate it:

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

### Trap 2: selection improvement without learning (Loop B)

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

### Trap 3: learning changes behavior (Loop C mechanism)

```bash
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

## 3) The Capstone Task (Skillset)

Write `notes/promotion_memo.md` (≤ 1 page):

* Section for each trap:

  * loop (A/B/C)
  * knob turned
  * whether comparison is valid
  * evidence (quote the relevant part of `manifest.json` / scorer version / parameter change)
* Final paragraph: what you would PROMOTE vs REJECT today and why

## 4) The “Passed” Checklist

* If pass_rate is higher, how do you prove it’s not a Locked Room violation?
* How do you distinguish selection improvement from learning improvement using artifacts only?
* Why does Loop C inevitably create new failure modes that frozen rollouts can’t fully reveal?

---

## If you complete this track correctly

You’ll have the core skill this course is aiming for: the ability to run and interpret RL-for-LLMs experiments without confusing **measurement**, **selection**, and **learning**, and without trusting invalid comparisons.
