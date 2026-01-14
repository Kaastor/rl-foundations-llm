# Selection demo (Best-of-N)

- Created (UTC): `2026-01-14T18:21:56+00:00`
- Scorer: `math_final_line_verifier` v`1.0.0`
- Dataset: `data/datasets/math_dev.jsonl`
- Samples: `data/rollouts/selection_pack_dev.jsonl`
- n_samples_used (cap): `4`

## Metrics
- pass@1: **0.250**
- pass@N: **0.750**
- delta: **0.500**
- rescued: `10`

## Interpretation reminder
Best-of-N improves the *chosen output* by spending more sampling compute.
It does **not** change the underlying model distribution.
