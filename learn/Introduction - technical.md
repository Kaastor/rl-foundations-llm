# Introduction

## What you actually do in this course

This course has three concrete scripts:

1. **`eval.py`** — Measure how well a model performs right now
2. **`selection_demo.py`** — Generate N answers, pick the best one
3. **`bandit_train.py`** — Actually train something to get better

They all use the same **scorer**: a function that checks if the model's output has the exact format `Final: <int>` with the right number.

That's it. That's the skeleton.

---

## From formal to practical (concept map)

If you've read the formal introduction, here's how those concepts show up here:

| Formal Concept | Technical Name | Where You See It |
|----------------|----------------|------------------|
| **Policy** | "The model" | All three scripts use the model to generate outputs |
| **Reward function** | "The scorer" | All scripts call `score()` to check outputs |
| **Objective (expected reward)** | "Performance" or "pass rate" | eval.py measures this |
| **Empirical mean reward** | "Pass rate" or "mean reward" | eval.py's summary statistics |
| **Argmax selection** | "Pick the best one" | selection_demo.py chooses highest-scoring sample |
| **Policy gradient update** | "Update the model" | bandit_train.py changes model weights |
| **Advantage** | "Reward minus baseline" | bandit_train.py's learning signal |
| **KL divergence penalty** | "Don't drift too far" | Prevents reward hacking |

The formal version teaches you **why** these work. This guide shows you **what** they do in practice.

---

## The core idea: feedback instead of labels

Normally you train a model by showing it examples:
- Input: "What is 2+2?"
- Correct output: "4"

**RL for LLMs** is different. You give the model a prompt, let it generate an answer, then give it **feedback**:
- ✅ "Correct"
- ❌ "Wrong"
- 👍 "This is better than that"

In this course:
- Prompt: math problem
- Model generates: some text
- Scorer checks: "Does it say `Final: <correct number>`?"
- Feedback: 1 (correct) or 0 (wrong)

Simple. Boring. **That's the point.**

Boring feedback = you can see what's actually happening.

---

## Three ways to use the same scorer

### 1. **Evaluation** (`eval.py`)

**What:** Run the model, score every answer, calculate pass rate.

**Purpose:** "How good is the model right now?"

No training. No tricks. Just measurement.

Example output:
```
Pass rate: 23/100 = 23%
Failures:
  - Wrong answer: 65
  - Bad format: 12
```

This is **Loop A**: pure measurement.

*(Technical note: This computes the empirical mean reward you learned about in formal - it estimates the objective "expected reward.")*

---

### 2. **Selection** (`selection_demo.py`)

**What:** Generate N answers for the same question, pick the best one.

**Purpose:** "Can I get better results without changing the model?"

Yes! If the model gets it right 23% of the time:
- 1 sample → 23% chance
- 10 samples → ~93% chance at least one is right

This is **best-of-N** search.

The model didn't get smarter. You just searched harder.

This is **Loop B**: optimization at inference time.

*(Technical note: This implements the argmax selection operation - you're maximizing over samples from the same policy.)*

**Key metrics:**
- **pass@1** — how often the first try works
- **pass@10** — how often at least 1 of 10 tries works

Selection improves pass@10. It barely helps pass@1.

---

### 3. **Learning** (`bandit_train.py`)

**What:** Generate answers, score them, **update the model** to make good answers more likely.

**Purpose:** "Can I make pass@1 better?"

This is actual RL:
1. Sample an answer
2. Score it
3. If reward > expected → increase probability of that answer
4. If reward < expected → decrease probability

This is **Loop C**: changing the model itself.

*(Technical note: This implements policy gradient updates using advantage - the model's parameters are actually changing to increase expected reward.)*

---

## The mental model: one thermometer, three uses

```
┌─────────┐
│ Dataset │ ← math problems
└────┬────┘
     │
     v
┌─────────┐
│  Model  │ ← generates text
└────┬────┘
     │
     v
┌─────────┐
│ Scorer  │ ← checks format + answer
└────┬────┘
     │
     v
┌─────────┐
│ Results │
└────┬────┘
     │
     ├─→ Loop A: summarize (eval)
     ├─→ Loop B: pick best (selection)
     └─→ Loop C: improve model (learning)
```

**Critical insight:** The scorer is your measuring instrument.

If you change the scorer, you changed what "correct" means.

---

## Why this matters: the reward spec is everything

Example: what if you scored "any string containing the right number"?

Problem:
```
Final: 42
But also 43 might work?
Actually I'm not sure
The answer could be 42
```

This string contains "42" → reward = 1 ✅

But it's garbage.

**Strict format = strict feedback signal.**

In this course: reward is 1 **only if** the output is exactly:
```
Final: 42
```

No extra lines. No extra words. One integer.

This strictness isn't pedantry—it's **environment design**.

Loose spec → model learns to exploit loopholes.
Strict spec → model learns to follow rules.

---

## The one-sentence guide to RL training

**"Push up what worked better than expected, push down what worked worse."**

Details:
- Sample action → get reward
- Compare to **baseline** (your expectation)
- **Advantage** = reward − baseline
- If advantage > 0 → increase probability
- If advantage < 0 → decrease probability

That's REINFORCE.

---

## Why you need a leash (KL penalty)

If you only optimize reward, the model will find loopholes.

Example:
- Reward: "output contains the number 42"
- Model learns: "42 42 42 42 42 42 42"
- Technically correct! Reward = 1!
- But it's not actually solving the problem.

**Solution:** Add a penalty for drifting too far from the original model.

**KL divergence** measures "how different is the new policy from the old one?"

Objective becomes:
```
maximize (reward − β × KL)
```

Where:
- Reward says "go this way"
- KL says "but don't go insane"

This tradeoff (reward vs drift) is the heart of RLHF tuning.

---

## Common pitfall: confusing the three loops

All three loops look similar:
- Take prompts
- Generate completions
- Score them
- Get numbers

But they mean **completely different things:**

| Loop | What it does | What changes |
|------|-------------|--------------|
| **A: Eval** | Measure current performance | Nothing |
| **B: Selection** | Pick best of N | Which output you use |
| **C: Learning** | Train the model | The model itself |

If you're comparing two runs and can't tell which loop changed, you're lying to yourself.

---

## What you should know before touching code

Say these out loud:

1. **"The scorer is my thermometer. If I change it, I changed what I'm measuring."**

2. **"Loop A measures. Loop B searches. Loop C trains."**

3. **"Selection improves pass@N. Training improves pass@1."**

4. **"Advantage = reward − baseline. That's the learning signal."**

5. **"KL penalty prevents reward hacking."**

6. **"Text has tokens. Tiny format changes = big probability changes."**

If those make sense, you're ready for the code.

If not, re-read the sections above until they click.

**The code is just details. The mental map is what matters.**
