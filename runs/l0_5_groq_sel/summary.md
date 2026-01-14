# Selection demo (Best-of-N)

- Created (UTC): `2026-01-14T19:18:20+00:00`
- Scorer: `math_final_line_verifier` v`1.0.0`
- Dataset: `data/datasets/math_dev.jsonl`
- Samples: `runs/rollouts_groq_sel/selection_pack.jsonl`
- n_samples_used (cap): `4`

## Metrics
- pass@1: **0.000**
- pass@N: **0.000**
- delta: **0.000**
- rescued: `0`

## Interpretation reminder
Best-of-N improves the *chosen output* by spending more sampling compute.
It does **not** change the underlying model distribution.
