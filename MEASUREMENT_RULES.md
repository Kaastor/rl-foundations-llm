# Measurement Rules (read this before trusting any number)

This repo treats evaluation and reward as a **measurement instrument**.
Most “RL failures” in practice are measurement failures wearing a trench coat.

## Locked Room Rule (valid comparisons)

When comparing runs, keep the environment fixed:

- same dataset split
- same prompt text
- same scoring spec/version
- same sampling settings (temperature/top_p/seeds) **if sampling exists**

If you changed the scorer/spec, you changed the instrument.
Do not compare numbers across instrument versions as if they mean the same thing.

## Holdout discipline

- Keep a small holdout set you do **not** tune on.
- Only celebrate improvements that show up on holdout, not just dev.

## Save artifacts, not just aggregates

Always save:

- raw completions
- parsed fields (what did the scorer *think* it saw?)
- per-sample rewards

Averages hide reward hacking and parser bugs.

## Golden gates (scorer quality)

Track both:

- **False negatives**: in-spec correct outputs that get reward 0  
  → usually parsing/normalization/spec bugs.
- **False positives**: wrong outputs that get reward 1  
  → reward hacking surface.

Before trusting a scorer change:

1) Run the **Golden Correct Set** (known-correct, in-spec completions).
   If any golden case fails, fix the scorer/spec first.

2) Run the **Golden Exploit Set** (known exploits / tricky cases).
   If any exploit passes, patch and add/upgrade tests.

## Change one thing at a time

Until measurement is stable, changing:
- prompt + scorer + dataset + sampling settings

…teaches you nothing.

## Noise rule

On tiny datasets (N < 100), 2–3% changes are usually just luck.
Treat improvements as “real” only if they are **large** (rule of thumb: >5%) and stable across seeds/splits.
