# Solution Walkthrough

This walkthrough follows the assignments in `docs/learn/Project.md`. Each section includes the assignment description (copied), a step-by-step record of what was done and why, and a note on how it supports real-world / future-engineer readiness.

Note: All commands were run with `python3.11` because `python3` defaults to 3.8 on this machine and fails on `dataclass(slots=True)`.

Note: Student templates live in `*_template.py`. The student repo only contains template versions.

---

## Level 0: Measurement Hygiene — Loop A

### Assignment description (copied)
Build:
```
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

Sabotage:
```
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Manually modify exactly ONE record's expected_answer in the tampered file.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

Reflect:
```
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

Capstone Tasks:
- Create `notes/mental_map_v1.md`
- Create `course/assignments/kata_01.py` and `tests/test_kata_01.py`

### What I did and why
1) Build eval:
```
python3.11 -m course.eval --dataset data/datasets/math_dev.jsonl --completions data/rollouts/frozen_rollouts_dev.jsonl --outdir runs/l0_build_eval
python3.11 -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```
Why: establishes a clean, deterministic baseline and inspects failure modes.

2) Sabotage dataset:
- Created `data/datasets/math_dev_TAMPERED.jsonl` by incrementing one `expected_answer`.
- Ran eval against the tampered dataset:
```
python3.11 -m course.eval --dataset data/datasets/math_dev_TAMPERED.jsonl --completions data/rollouts/frozen_rollouts_dev.jsonl --outdir runs/l0_sabotage_eval_tampered
```
Why: a single, explicit Locked Room violation is the pedagogical core.

3) Reflect with gate:
```
python3.11 -m course.gate --baseline runs/l0_build_eval --candidate runs/l0_sabotage_eval_tampered --min-delta 0.00
```
Why: verify that the gate rejects comparisons when instrument conditions differ.

4) Capstone tasks:
- Created `notes/mental_map_v1.md` with definitions + Loop A/B/C diagram.
- Implemented assignment logic in `course/assignments/kata_01.py` and `tests/test_kata_01.py` for outcome code classification (templates in `*_template.py`).

### Artifacts and code changes
Artifacts:
- `runs/l0_build_eval/manifest.json`
- `runs/l0_build_eval/results.jsonl`
- `runs/l0_build_eval/summary.json`
- `runs/l0_build_eval/summary.md`
- `runs/l0_sabotage_eval_tampered/manifest.json`
- `runs/l0_sabotage_eval_tampered/results.jsonl`
- `runs/l0_sabotage_eval_tampered/summary.json`
- `data/datasets/math_dev_TAMPERED.jsonl`

Code changes:
- `course/assignments/kata_01.py`
- `course/assignments/kata_01_template.py`
- `tests/test_kata_01.py`
- `tests/kata_01_template.py`
- `notes/mental_map_v1.md`

### Real-world readiness
This level builds the habit of using manifests and gates for valid comparisons, which is how real eval pipelines prevent false claims of improvement.

---

## Level 0.5: Selection Without Learning — Loop B

### Assignment description (copied)
Build:
```
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

Sabotage:
```
import random

def tie_break_key(sample):
    return (random.random(), sample.completion)
```
Run five times:
```
for i in 1 2 3 4 5; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/l0_5_sabotage_sel_random_$i
done
```

Reflect (hash results):
```
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

Repair:
```
def tie_break_key(sample):
    lp = sample.sum_logprob
    lp_key = 0.0 if lp is None else -float(lp)
    return (lp_key, len(sample.completion), sample.completion)
```
Add tests in `tests/test_selection_policy.py`.

### What I did and why
1) Build selection demo:
```
python3.11 -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l0_5_build_sel_n4
python3.11 -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```
Why: establish best-of-N improvement without learning.

2) Sabotage tie-break with randomness:
- Temporarily changed `tie_break_key` in `course/assignments/selection_policy.py` to use `random.random()`.
- Ran five selection demos and hashed `results.jsonl` to show nondeterminism.
Why: demonstrate that nondeterministic selection is unstable and unauditable.

3) Repair:
- Restored deterministic tie-break: higher logprob, shorter length, lexicographic.
- Added `tests/test_selection_policy.py` for determinism and tie-break behavior (template in `tests/selection_policy_template.py`).
- Ran `runs/l0_5_fixed_sel_n4`.

### Artifacts and code changes
Artifacts:
- `runs/l0_5_build_sel_n4/summary.json`
- `runs/l0_5_build_sel_n4/results.jsonl`
- `runs/l0_5_sabotage_sel_random_1/results.jsonl`
- `runs/l0_5_sabotage_sel_random_2/results.jsonl`
- `runs/l0_5_sabotage_sel_random_3/results.jsonl`
- `runs/l0_5_sabotage_sel_random_4/results.jsonl`
- `runs/l0_5_sabotage_sel_random_5/results.jsonl`
- `runs/l0_5_fixed_sel_n4/summary.json`
- `runs/l0_5_fixed_sel_n4/results.jsonl`

Code changes:
- `course/assignments/selection_policy.py`
- `course/assignments/selection_policy_template.py`
- `tests/test_selection_policy.py`
- `tests/selection_policy_template.py`

### Real-world readiness
Production selection layers must be deterministic to avoid flaky outputs and hard-to-debug regressions. This mirrors real ranking/tie-break rules in LLM systems.

---

## Optional Extension: Real Rollouts via Groq (Minimal Change)

### Assignment description (copied)
```
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

### What I did and why
1) Set `GROQ_API_KEY` in `.env` (provided by the user).
2) Sampled real-model completions (full dataset) and evaluated:
```
python3.11 -m course.rollout_sample --dataset data/datasets/math_dev.jsonl --outdir runs/rollouts_groq_dev
python3.11 -m course.eval --dataset data/datasets/math_dev.jsonl --completions runs/rollouts_groq_dev/completions.jsonl --outdir runs/l0_build_eval_groq
```
Why: establishes a real-model baseline using the same measurement instrument.

3) Sampled a real selection pack with a cap (`--max 5`) to keep runtime bounded, then ran selection:
```
python3.11 -m course.rollout_sample --dataset data/datasets/math_dev.jsonl --n 4 --format selection --max 5 --outdir runs/rollouts_groq_sel
python3.11 -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples runs/rollouts_groq_sel/selection_pack.jsonl --n 4 --max 5 --outdir runs/l0_5_groq_sel
```
Why: demonstrates Loop B with real samples while avoiding long API runs.

### Artifacts and code changes
Artifacts:
- `runs/rollouts_groq_dev/completions.jsonl`
- `runs/rollouts_groq_dev/summary.json`
- `runs/rollouts_groq_dev/manifest.json`
- `runs/l0_build_eval_groq/summary.json`
- `runs/l0_build_eval_groq/results.jsonl`
- `runs/rollouts_groq_sel/selection_pack.jsonl`
- `runs/rollouts_groq_sel/summary.json`
- `runs/rollouts_groq_sel/manifest.json`
- `runs/l0_5_groq_sel/summary.json`
- `runs/l0_5_groq_sel/results.jsonl`

Code changes:
- `course/rollout_sample.py`
- `pyproject.toml` (added `groq` and `python-dotenv`)

### Real-world readiness
Adds a true model sampling loop while preserving the same measurement discipline, which is how production systems integrate new models into existing evaluation pipelines.

---

## Level 1: REINFORCE Algorithm Analysis — Loop C

### Assignment description (copied)
Build:
```
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

Sabotage:
```
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

Reflect:
```
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

Capstone:
- Create `notes/reinforce_forensics.md` with 10 annotated slow steps.

### What I did and why
1) Build bandit run:
```
python3.11 -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```
Why: baseline learning behavior.

2) Sabotage learning rate:
```
python3.11 -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
python3.11 -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```
Why: show instability and inverted updates.

3) Slow runs for inspection:
```
python3.11 -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow | tee notes/l1_build_slow.log
python3.11 -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg | tee notes/l1_sabotage_slow_lrneg.log
```
4) Forensics note:
- Added `notes/reinforce_forensics.md` with 10-step annotations and a summary.

### Artifacts and code changes
Artifacts:
- `runs/l1_build_bandit/summary.json`
- `runs/l1_sabotage_lr2/summary.json`
- `runs/l1_sabotage_lrneg/summary.json`
- `runs/l1_build_slow/summary.json`
- `runs/l1_sabotage_slow_lrneg/summary.json`
- `notes/l1_build_slow.log`
- `notes/l1_sabotage_slow_lrneg.log`

Code changes:
- `notes/reinforce_forensics.md`

### Real-world readiness
Engineers need intuition for advantage signs and update directions; it prevents silent training failures and helps diagnose training instability.

---

## Level 2: Credit Assignment and Token-Text Boundary

### Assignment description (copied)
Build (token inspection):
```
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

Capstone:
- Create `course/assignments/two_step_mdp_demo.py`
- Run build, sabotage (no step 1 updates), and repair.
- Create `notes/credit_assignment.md`

### What I did and why
1) Token inspection:
```
python3.11 -m course.token_inspect "Final: 323"
python3.11 -m course.token_inspect "Final:  323"
python3.11 -m course.token_inspect $'Final:\n323'
python3.11 -m course.token_inspect "Final: 0323"
```
Why: show that small textual changes map to distinct tokens.

2) Two-step MDP demo:
- Implemented `course/assignments/two_step_mdp_demo.py`.
- Build:
```
python3.11 course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```
- Sabotage (disabled step 1 updates):
```
python3.11 course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```
- Repair (restore step 1 updates):
```
python3.11 course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```
- Added `notes/credit_assignment.md`.

### Artifacts and code changes
Artifacts:
- `runs/l2_build_two_step/summary.json`
- `runs/l2_build_two_step/traj.jsonl`
- `runs/l2_sabotage_no_credit_step1/summary.json`
- `runs/l2_sabotage_no_credit_step1/traj.jsonl`
- `runs/l2_fixed_two_step/summary.json`
- `runs/l2_fixed_two_step/traj.jsonl`
- `notes/credit_assignment.md`

Code changes:
- `course/assignments/two_step_mdp_demo.py`

### Real-world readiness
This makes it explicit that terminal reward must credit earlier actions, and that token-level policy choices affect text-level reward outcomes.

---

## Level 3: KL Divergence as Regularization Constraint

### Assignment description (copied)
Build:
```
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

Capstone:
- Create `course/assignments/kl_regularized_choice.py`
- Build + sabotage with beta=0
- Create `notes/kl_tradeoff.md`

### What I did and why
1) KL demo:
```
python3.11 -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```
Note: matplotlib was unavailable, so plotting was skipped; CSV and summaries were still written.

2) KL regularized choice:
- Implemented `course/assignments/kl_regularized_choice.py`.
- Build:
```
python3.11 course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```
- Sabotage (beta=0):
```
python3.11 course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```
- Wrote `notes/kl_tradeoff.md`.

### Artifacts and code changes
Artifacts:
- `runs/l3_build_kl_demo/summary.json`
- `runs/l3_build_kl_demo/summary.md`
- `runs/l3_build_kl_demo/kl_tradeoff.csv`
- `runs/l3_build_kl_choice.txt`
- `runs/l3_sabotage_no_kl.txt`
- `notes/kl_tradeoff.md`

Code changes:
- `course/assignments/kl_regularized_choice.py`
- `course/kl_tradeoff_demo.py` (added output directory creation)

### Real-world readiness
Makes the KL constraint concrete: reward can be optimized while bounding distribution drift, a key production RLHF guardrail.

---

## Level 4: Reward Specification as Production Code

### Assignment description (copied)
Build:
```
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_correct.jsonl

python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

Capstone:
- Create `notes/reward_spec_blackbox.md`
- Sabotage: relax a spec rule and observe failures
- Repair: add `tests/test_reward_regressions.py` and `data/golden/golden_exploits_extra.jsonl`

### What I did and why
1) Validated golden sets:
```
python3.11 -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_correct.jsonl
python3.11 -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits.jsonl
```

2) Black-box spec note:
- Added `notes/reward_spec_blackbox.md`.

3) Sabotage:
- Temporarily allowed leading zeros in `course/core/scoring.py`.
- Validation failed against `golden_exploits.jsonl`.

4) Repair and lock:
- Restored strictness.
- Added `tests/test_reward_regressions.py` (template in `tests/reward_regressions_template.py`).
- Added `data/golden/golden_exploits_extra.jsonl`.
- Validated:
```
python3.11 -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits_extra.jsonl
```

### Artifacts and code changes
Artifacts:
- `notes/reward_spec_blackbox.md`
- `data/golden/golden_exploits_extra.jsonl`
- `tests/test_reward_regressions.py`
- `tests/reward_regressions_template.py`

Code changes:
- `tests/test_reward_regressions.py`
- `tests/reward_regressions_template.py`
- `data/golden/golden_exploits_extra.jsonl`
- `course/core/scoring.py` (temporarily loosened for sabotage, then restored)

### Real-world readiness
Treats scoring as production code: versioned, tested, and guarded against regression and exploitation.

---

## Level 5: Reward Exploitation Analysis

### Assignment description (copied)
- Create `course/assignments/hackable_scorer_demo.py` with a naive verifier.
- Generate 5 exploits.
- Patch the exploit class.
- Create `notes/red_team_report.md`.

### What I did and why
1) Implemented naive and patched verifiers in `course/assignments/hackable_scorer_demo.py`.
2) Documented 5 exploit strings and patch strategy in `notes/red_team_report.md`.

### Artifacts and code changes
Artifacts:
- `course/assignments/hackable_scorer_demo.py`
- `notes/red_team_report.md`

Code changes:
- `course/assignments/hackable_scorer_demo.py`
- `notes/red_team_report.md`

### Real-world readiness
Builds the ability to identify proxy-metric exploits and to patch entire exploit classes, not individual strings.

---

## Level 6: Promotion Committee (Experimental Traps)

### Assignment description (copied)
Trap 1:
```
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

Trap 2:
```
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

Trap 3:
```
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

Capstone:
- Create `notes/promotion_memo.md`

### What I did and why
1) Trap 1 gate:
```
python3.11 -m course.gate --baseline runs/l0_build_eval --candidate runs/l0_sabotage_eval_tampered --min-delta 0.00
```

2) Trap 2 selection comparison:
```
python3.11 -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 1 --outdir runs/l6_trap_sel_n1
python3.11 -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l6_trap_sel_n4
```

3) Trap 3 learning:
```
python3.11 -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

4) Promotion memo:
- Added `notes/promotion_memo.md`.

### Artifacts and code changes
Artifacts:
- `runs/l6_trap_sel_n1/summary.json`
- `runs/l6_trap_sel_n4/summary.json`
- `runs/l6_trap_learning/summary.json`
- `runs/l0_build_eval/manifest.json`
- `runs/l0_sabotage_eval_tampered/manifest.json`
- `notes/promotion_memo.md`

Code changes:
- `notes/promotion_memo.md`

### Real-world readiness
Teaches release governance: promotion decisions must be grounded in valid comparisons and artifacts, not raw metric bumps.

---

## Tests
```
python3.11 -m pytest -q
```

---

## Summary of created artifacts
- Assignments: `course/assignments/kata_01.py`, `course/assignments/two_step_mdp_demo.py`, `course/assignments/kl_regularized_choice.py`, `course/assignments/hackable_scorer_demo.py`, `course/assignments/selection_policy.py`
- Assignment templates: `course/assignments/kata_01_template.py`, `course/assignments/two_step_mdp_demo_template.py`, `course/assignments/kl_regularized_choice_template.py`, `course/assignments/hackable_scorer_demo_template.py`, `course/assignments/selection_policy_template.py`
- Updated code: `course/assignments/kata_01.py`, `course/assignments/two_step_mdp_demo.py`, `course/assignments/kl_regularized_choice.py`, `course/assignments/hackable_scorer_demo.py`, `course/assignments/selection_policy.py`, `course/kl_tradeoff_demo.py`, `course/rollout_sample.py`, `pyproject.toml`
- Notes: `notes/mental_map_v1.md`, `notes/reinforce_forensics.md`, `notes/credit_assignment.md`, `notes/kl_tradeoff.md`, `notes/reward_spec_blackbox.md`, `notes/red_team_report.md`, `notes/promotion_memo.md`, `notes/student_results.md`
- Tests: `tests/test_kata_01.py`, `tests/test_selection_policy.py`, `tests/test_reward_regressions.py`
- Test templates: `tests/kata_01_template.py`, `tests/selection_policy_template.py`, `tests/reward_regressions_template.py`
- Data: `data/datasets/math_dev_TAMPERED.jsonl`, `data/golden/golden_exploits_extra.jsonl`
