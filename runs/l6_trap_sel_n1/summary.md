# Selection demo (Best-of-N)

- Created (UTC): `2026-01-14T18:26:53+00:00`
- Scorer: `math_final_line_verifier` v`1.0.0`
- Dataset: `data/datasets/math_dev.jsonl`
- Samples: `data/rollouts/selection_pack_dev.jsonl`
- n_samples_used (cap): `1`

## Metrics
- pass@1: **0.250**
- pass@N: **0.250**
- delta: **0.000**
- rescued: `0`

## Interpretation reminder
Best-of-N improves the *chosen output* by spending more sampling compute.
It does **not** change the underlying model distribution.
