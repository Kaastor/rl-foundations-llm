# Bandit training demo (REINFORCE, categorical policy)

- Created (UTC): `2026-01-14T18:22:18+00:00`
- Steps: `200`
- Seed: `0`
- Learning rate: `-0.5`
- Baseline: `True`

## Metrics
- mean_reward: **0.115**
- mean_reward_last_100: **0.110**

## Interpretation
This script is a microscope for the mechanism: probabilities shift toward
actions that yield higher reward. The baseline reduces variance; it does not
change the expected update direction (it changes noise).
