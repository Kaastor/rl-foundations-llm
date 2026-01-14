# Eval run
- Created (UTC): `2026-01-14T18:21:01+00:00`
- Scorer: `math_final_line_verifier` v`1.0.0`
- Dataset: `data/datasets/math_dev_TAMPERED.jsonl`
- Completion source: `{'type': 'jsonl', 'path': 'data/rollouts/frozen_rollouts_dev.jsonl', 'n': 20}`
- Examples: `20` (missing completions: `0`)

## Metrics
- pass_rate: **0.150**
- n_pass: `3`
- n_fail: `17`

## Outcome codes (grouping)
- `wrong_answer`: 5
- `missing_prefix`: 4
- `extra_whitespace`: 4
- `not_single_line`: 4
- `ok`: 3

## First 10 failures (inspect details in results.jsonl)
- `dev-0001` [wrong_answer] completion preview: `Final: 22...`
- `dev-0002` [wrong_answer] completion preview: `Final: 109...`
- `dev-0003` [missing_prefix] completion preview: `The answer is 208....`
- `dev-0004` [extra_whitespace] completion preview: `Final:  56...`
- `dev-0005` [not_single_line] completion preview: `Final: 165...`
- `dev-0007` [wrong_answer] completion preview: `Final: 103...`
- `dev-0008` [missing_prefix] completion preview: `The answer is 234....`
- `dev-0009` [extra_whitespace] completion preview: `Final:  19...`
- `dev-0010` [not_single_line] completion preview: `Final: 160...`
- `dev-0012` [wrong_answer] completion preview: `Final: 67...`
