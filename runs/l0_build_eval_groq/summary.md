# Eval run
- Created (UTC): `2026-01-14T19:14:25+00:00`
- Scorer: `math_final_line_verifier` v`1.0.0`
- Dataset: `data/datasets/math_dev.jsonl`
- Completion source: `{'type': 'jsonl', 'path': 'runs/rollouts_groq_dev/completions.jsonl', 'n': 20}`
- Examples: `20` (missing completions: `0`)

## Metrics
- pass_rate: **0.000**
- n_pass: `0`
- n_fail: `20`

## Outcome codes (grouping)
- `non_digit_characters`: 20

## First 10 failures (inspect details in results.jsonl)
- `dev-0001` [non_digit_characters] completion preview: `Final: 22....`
- `dev-0002` [non_digit_characters] completion preview: `Final: 108....`
- `dev-0003` [non_digit_characters] completion preview: `Final: 208....`
- `dev-0004` [non_digit_characters] completion preview: `Final: 56....`
- `dev-0005` [non_digit_characters] completion preview: `Final: 165....`
- `dev-0006` [non_digit_characters] completion preview: `Final: 288....`
- `dev-0007` [non_digit_characters] completion preview: `Final: 102....`
- `dev-0008` [non_digit_characters] completion preview: `Final: 234....`
- `dev-0009` [non_digit_characters] completion preview: `Final: 19....`
- `dev-0010` [non_digit_characters] completion preview: `Final: 160....`
