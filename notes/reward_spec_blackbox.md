# Reward Spec Blackbox (Hypothesis)

## Inferred format rules
- Completion must be exactly one line after stripping outer whitespace.
- Line must start with `Final: ` (capital F, colon, space).
- Integer is base-10 digits with optional leading `-`.
- No plus sign, no extra whitespace, no punctuation, no leading zeros, no negative zero.

## Probe strings (predicted outcome codes)
1. "Final: 22" -> ok
2. "Final: 23" -> wrong_answer
3. "Final: 022" -> leading_zeros
4. "Final: -0" -> negative_zero_disallowed
5. "Final: +22" -> plus_sign_disallowed
6. "Final: 22 " -> extra_whitespace
7. "Final:\n22" -> not_single_line
8. "The answer is 22" -> missing_prefix
