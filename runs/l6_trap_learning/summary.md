# Bandit training demo (REINFORCE, categorical policy)

- Created (UTC): `2026-01-14T18:27:01+00:00`
- Steps: `200`
- Seed: `1`
- Learning rate: `0.5`
- Baseline: `True`

## Metrics
- mean_reward: **0.745**
- mean_reward_last_100: **0.840**

## Interpretation
This script is a microscope for the mechanism: probabilities shift toward
actions that yield higher reward. The baseline reduces variance; it does not
change the expected update direction (it changes noise).
