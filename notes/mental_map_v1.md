# Mental Map v1

## Definitions
- reward: scalar feedback from the scorer for a completion.
- metric: aggregate statistic computed from rewards (e.g., pass_rate).
- objective: the target behavior implied by the reward specification.
- loss: the optimization signal used to update a policy (often derived from reward).
- policy: a distribution over actions/completions conditioned on context.
- environment: everything outside the policy (prompts, datasets, scorer, sampling config).

## Loop Diagram (A/B/C)

Loop A (Evaluation): measurement instrument
  policy (fixed) -> completions -> scorer -> reward -> metrics
  primary variable: measurement instrument + environment must stay fixed

Loop B (Selection): selection compute
  policy (fixed) -> N samples -> selection rule -> chosen completion -> reward
  primary variable: decision rule over samples (selection compute)

Loop C (Learning): policy update
  policy -> sample -> reward -> update -> policy'
  primary variable: policy parameters (distribution shifts)
