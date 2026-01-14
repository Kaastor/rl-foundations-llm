# Promotion Memo

## Trap 1: Instrument change (Locked Room violation)
- Loop: A (evaluation)
- Variable modified: measurement instrument/environment (dataset file)
- Comparison validity: invalid (Locked Room violated)
- Evidence: `runs/l0_build_eval/manifest.json` dataset sha256 `af5c8a6f...` vs `runs/l0_sabotage_eval_tampered/manifest.json` sha256 `91981188...` with same scorer v1.0.0
- Decision: REJECT

## Trap 2: Selection improvement without learning
- Loop: B (selection)
- Variable modified: selection compute (n=1 vs n=4)
- Comparison validity: valid for selection compute change; does not indicate learning
- Evidence: `runs/l6_trap_sel_n1/summary.json` pass@N=0.25 vs `runs/l6_trap_sel_n4/summary.json` pass@N=0.75 with same dataset/samples
- Decision: REJECT for model promotion; acceptable for product-level selection tuning

## Trap 3: Learning modifies behavior
- Loop: C (learning)
- Variable modified: policy parameters via REINFORCE updates
- Comparison validity: valid learning run (same environment)
- Evidence: `runs/l6_trap_learning/summary.json` mean_reward=0.745 (mean_reward_last_100=0.84)
- Decision: PROVISIONAL PROMOTE pending additional eval on held-out data

## Conclusion
Promote only Trap 3 after verifying Locked Room conditions and holdout performance. Reject Trap 1 and Trap 2 as invalid for policy promotion.
