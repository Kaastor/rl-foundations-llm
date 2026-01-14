# Credit Assignment Notes

## What failed in sabotage
- Step 1 remained uniform, so half the time we never reached the rewarding branch (A -> X).
- Step 2 learned to prefer X given A, but could not compensate for random step 1.
- Result: mean reward plateaued near 0.5 instead of approaching 1.0.

## Underlying cause
Terminal reward depends on both decisions. If step 1 receives no update, the policy cannot assign credit to the earlier choice that enables reward later.

## Boundary diagram

tokens -> detokenization -> text -> parsing -> reward

In LLMs, the policy acts in token space while the verifier acts on text after decoding and parsing. Credit must flow from the terminal reward back through the sampled token trajectory.
