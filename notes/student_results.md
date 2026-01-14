# Student Results Summary

## Level 0 (Loop A)
- Build eval: `runs/l0_build_eval` pass_rate=0.20 (4/20).
- Sabotage (tampered dataset): `runs/l0_sabotage_eval_tampered` pass_rate=0.15 (3/20).
- Gate: REJECT due to dataset hash mismatch (Locked Room violation).

## Level 0.5 (Loop B)
- Build selection: `runs/l0_5_build_sel_n4` pass@1=0.25, pass@N=0.75.
- Sabotage (random tie-break): hashes differed across `runs/l0_5_sabotage_sel_random_1..5`.
- Repair: deterministic tie-break restored; `runs/l0_5_fixed_sel_n4` pass@N=0.75.

## Level 1 (Loop C)
- Build bandit: `runs/l1_build_bandit` mean_reward=0.725.
- Sabotage lr=2.0: `runs/l1_sabotage_lr2` mean_reward=0.765 (unstable but not catastrophic here).
- Sabotage lr=-0.5: `runs/l1_sabotage_lrneg` mean_reward=0.115.
- Slow logs captured in `notes/l1_build_slow.log` and `notes/l1_sabotage_slow_lrneg.log`.

## Level 2 (Credit Assignment + Token Boundary)
- token_inspect outputs ran for strict format variants (fallback tokenizer).
- Two-step MDP build: `runs/l2_build_two_step` mean_reward=0.925.
- Sabotage (no step1 updates): `runs/l2_sabotage_no_credit_step1` mean_reward=0.435.
- Repair: `runs/l2_fixed_two_step` mean_reward=0.925.

## Level 3 (KL)
- KL demo: `runs/l3_build_kl_demo` (plot skipped: matplotlib unavailable).
- KL choice build: `runs/l3_build_kl_choice.txt`.
- KL choice sabotage (beta=0): `runs/l3_sabotage_no_kl.txt`.

## Level 4 (Scorer as Production Code)
- Golden sets: `data/golden/golden_correct.jsonl`, `data/golden/golden_exploits.jsonl`, and `data/golden/golden_exploits_extra.jsonl` validated under v1.0.0.
- Sabotage (leading zeros allowed) caused exploit to pass; repaired to strict behavior.

## Level 5 (Reward Exploitation)
- Implemented naive vs patched verifier demo in `course/assignments/hackable_scorer_demo.py`.
- Exploit list + patch strategy in `notes/red_team_report.md`.

## Level 6 (Promotion Committee)
- Trap 1: REJECT (Locked Room violation).
- Trap 2: REJECT for model promotion (selection-only improvement).
- Trap 3: PROVISIONAL PROMOTE pending holdout eval.
