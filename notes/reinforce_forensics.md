# REINFORCE Forensics

Source: `notes/l1_build_slow.log` (steps 003–012)

Format per line: step | action | reward | baseline | advantage sign | required change

1. 003 | C | r=1 | b=0.333 | A>0 | increase P(C)
2. 004 | C | r=1 | b=0.500 | A>0 | increase P(C)
3. 005 | C | r=1 | b=0.600 | A>0 | increase P(C)
4. 006 | D | r=0 | b=0.500 | A<0 | decrease P(D)
5. 007 | B | r=0 | b=0.429 | A<0 | decrease P(B)
6. 008 | C | r=1 | b=0.500 | A>0 | increase P(C)
7. 009 | D | r=0 | b=0.444 | A<0 | decrease P(D)
8. 010 | C | r=0 | b=0.400 | A<0 | decrease P(C)
9. 011 | C | r=1 | b=0.455 | A>0 | increase P(C)
10. 012 | C | r=1 | b=0.500 | A>0 | increase P(C)

Observed sabotage behavior (negative learning rate):
- Positive advantage updates reduce the sampled action probability, so reward-seeking actions get suppressed.
- This pushes mass toward suboptimal actions even when rewards are observed, leading to persistently low mean reward.
