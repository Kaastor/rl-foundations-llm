# Red Team Report (Hackable Scorer Demo)

## Exploit completions (expected=22)
1. "The answer is 22." (no `Final:` prefix)
2. "Final: 220" (wrong answer, but substring match)
3. "Final: 022" (leading zeros)
4. "Final: 0 and also 22" (extra text, substring match)
5. "22" (bare number, missing format)

## Why they succeeded (naive verifier)
The naive verifier only checks whether the expected integer appears anywhere in the completion. It ignores format, line boundaries, and extra text.

## Patch strategy
Enforce the exact `Final: <int>` format with full-string match and strict integer rules (no plus sign, no leading zeros, no negative zero). This closes the entire class of substring/format exploits.

## Regression tests to add
- Reject completions containing the target number in extra text.
- Reject format variants like missing prefix, extra whitespace, and leading zeros.
- Accept only the exact canonical format for the expected answer.
