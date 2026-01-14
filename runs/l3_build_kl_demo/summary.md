# KL tradeoff demo (categorical policy)

We optimize: `E_pi[r] - beta * KL(pi || pi_ref)`

Interpretation for LLM RL:
- `pi_ref` ≈ reference model (pretrained/SFT)
- KL penalty discourages drifting into weird space
- smaller beta → more reward-seeking, more drift

See `kl_tradeoff.csv` for the reward/KL curve.
