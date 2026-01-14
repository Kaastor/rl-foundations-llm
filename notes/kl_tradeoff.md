# KL Tradeoff Notes

## Why beta = 0 is attractive but problematic
When beta = 0, the selector ignores divergence and picks the highest raw reward. This often looks good on paper, but it removes any constraint on how far the policy drifts from the reference distribution.

## Why unconstrained optimization goes extreme
Without a KL penalty, the optimizer can concentrate probability mass on narrow, brittle behaviors that exploit the proxy reward. The measured reward can rise while robustness and safety degrade.

## Operational interpretation
KL acts as a practical brake: "improve reward, but stay close to the reference." Removing it creates a search for extreme, high-divergence solutions.
