# 🎓 First Meeting: Introduction to RL for LLMs

> **For the teacher:** This document is designed for a ~45-60 minute introductory session with students who have **no prior Reinforcement Learning knowledge**. Use it as a guide for explaining the project before students start working.

---

## 🧭 Agenda

1. [Why This Matters Now](#-why-this-matters-now-5-min)
2. [The Core Problem We're Solving](#-the-core-problem-were-solving-10-min)
3. [Key Concepts (RL Crash Course)](#-key-concepts-rl-crash-course-15-min)
4. [The Three Loops Framework](#-the-three-loops-framework-10-min)
5. [What Makes This Project Different](#-what-makes-this-project-different-5-min)
6. [Project Overview & First Task](#-project-overview--first-task-10-min)

---

## 🌍 Why This Matters Now (5 min)

### The AI Landscape Has Changed

ChatGPT, Claude, Gemini — all the LLMs that students interact with daily — are not just trained on text. They go through a process called **RLHF (Reinforcement Learning from Human Feedback)**.

**Key insight for students:**
> "Every time you talk to ChatGPT, you're talking to a model that was shaped by reinforcement learning. Understanding RL is now essential to understanding how modern AI actually works."

### Real-World Impact

| Company | What RL Enables |
|---------|-----------------|
| OpenAI | ChatGPT follows instructions and refuses harmful requests |
| Anthropic | Claude maintains helpful but safe behavior |
| DeepMind | AlphaCode solves competitive programming problems |
| Meta | LLaMA models are fine-tuned for specific tasks |

**The punchline:** If you want to understand, debug, or build modern AI systems — you need to understand RL for LLMs.

---

## 🎯 The Core Problem We're Solving (10 min)

### The Training Data Problem

Traditional machine learning works like this:
```
Input → Model → Output
  ↑               ↓
  └───── Compare ─┘ (with labeled "correct" answer)
```

**But what if there's no single "correct" output?**

Consider these tasks:
- "Write a poem about autumn" — infinite valid answers
- "Explain quantum physics to a 5-year-old" — many good ways to do it
- "Solve this math problem step by step" — multiple valid reasoning paths

We can't give the model a "correct output string" to copy. Instead, we can only tell it whether its output was **good or bad**.

### The RL Solution

Reinforcement Learning lets us train models with **feedback** instead of **examples**:

```
Prompt → Model generates output → Scorer gives feedback
                ↑                           ↓
                └─── Model learns ──────────┘
```

**The key shift:**
- ❌ Old: "Copy this exact answer"
- ✅ New: "Try something, I'll tell you if it's good"

### Our Concrete Example

In this project, we focus on **math word problems**:

```
Prompt: "A farmer has 15 apples. She sells 7. How many remain?"
Model output: "Let me calculate... 15 - 7 = 8. Final: 8"
Scorer: ✅ Reward = 1 (correct!)
```

**Why math?**
- Clear right/wrong answer (no ambiguity)
- We can build a deterministic scorer
- Students can see the mechanism clearly

---

## 📚 Key Concepts (RL Crash Course) (15 min)

### 1. Policy (The Model as Decision-Maker)

In RL terminology, a **policy** is a rule for choosing actions.

For LLMs:
- **Policy** = the model itself
- **Action** = generating a completion (sequence of tokens)
- **State** = the prompt

Mathematical notation: $\pi_\theta(y | x)$

> "Given prompt $x$, what's the probability of generating completion $y$?"

**Analogy for students:** Think of the model as a student taking an exam. The policy is their "strategy" for answering questions. Some strategies work better than others.

### 2. Reward (The Feedback Signal)

The **reward function** $R(x, y)$ takes a prompt and completion and returns a score.

In our project: $R(x, y) \in \{0, 1\}$
- 1 = correct answer in correct format
- 0 = wrong answer or bad format

**Critical insight:**
> "The reward function is your measurement instrument. If it's broken, you're training toward garbage."

### 3. Objective (What We're Optimizing)

The goal of RL: **maximize expected reward**

$$J(\theta) = \mathbb{E}[R(x, y)]$$

Translation: "On average, how often does the model get it right?"

### 4. The Learning Signal (How Updates Happen)

The core idea of **REINFORCE** algorithm:

- If an action got **high reward** → make it **more likely**
- If an action got **low reward** → make it **less likely**

Visual:
```
Model generates: "Final: 42"
Reward: 1 (correct!)
→ Increase probability of generating "Final: 42" for similar prompts

Model generates: "The answer is maybe 42?"
Reward: 0 (bad format!)
→ Decrease probability of this pattern
```

### 5. Baseline (The "Expected" Reward)

Raw rewards are noisy. We use a **baseline** $b$ to stabilize learning:

$$\text{Advantage} = R - b$$

- If $R > b$: "Better than expected!" → reinforce
- If $R < b$: "Worse than expected!" → discourage

**Analogy:** If you usually score 70% on tests, getting 75% is a success. Getting 65% is a failure. The baseline (70%) gives context to the raw score.

### 6. KL Divergence (The Safety Net)

**Problem:** If you only optimize for reward, the model might find "hacks" — technically valid but unintended solutions.

**Solution:** Keep the model close to its original behavior:

$$J_{KL}(\theta) = \mathbb{E}[R] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

Translation: "Maximize reward, but don't change too much from the original model."

**Analogy:** "Improve your exam scores, but don't cheat." The KL penalty keeps the model from going off the rails.

---

## 🔄 The Three Loops Framework (10 min)

This is the **central mental model** of the course. All operations fall into one of three loops:

### Loop A — Evaluation (Measurement)

**Question:** "How good is the model right now?"

```
Prompts → Model → Completions → Scorer → Statistics
```

- No learning, no optimization
- Pure measurement
- Output: pass rate, failure mode breakdown

**Real-world analogy:** Taking a practice test to see where you stand.

### Loop B — Selection (Best-of-N)

**Question:** "Can we do better by trying multiple times?"

```
1 Prompt → Model → N Completions → Scorer → Pick the best one
```

- The model doesn't change
- We just cherry-pick the best output
- Trades compute for quality

**Real-world analogy:** Writing 5 drafts of an essay and submitting the best one. You didn't become a better writer — you just filtered your attempts.

**Key insight:** Selection improves **pass@N** but not **pass@1**.

### Loop C — Learning (Training)

**Question:** "Can we make the model actually better?"

```
Prompts → Model → Completions → Scorer → Gradient Update → Better Model
```

- Model parameters change
- Future generations improve
- This is actual reinforcement learning

**Real-world analogy:** Studying from your mistakes to actually improve your skills.

### The Three Loops Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     SCORER                              │
│              (Measurement Instrument)                   │
└─────────────────────────────────────────────────────────┘
                          ↑
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
    │ LOOP A  │     │ LOOP B  │     │ LOOP C  │
    │  EVAL   │     │ SELECT  │     │  LEARN  │
    └─────────┘     └─────────┘     └─────────┘
    Measure         Choose best      Update model
    (nothing        (output          (parameters
     changes)        changes)         change)
```

---

## ⚠️ What Makes This Project Different (5 min)

### The "Measurement First" Philosophy

Most RL courses start with algorithms (PPO, DPO, etc.). We start with **measurement**.

**Why?**

> "Most RL failures in practice are measurement failures wearing a trench coat."

If your scorer is broken, your training is broken. If your comparisons are invalid, your conclusions are wrong.

### The "Locked Room" Rule

Before comparing two experiments, verify:
- ✅ Same dataset
- ✅ Same prompts
- ✅ Same scorer version
- ✅ Same sampling settings

If any of these differ, **you changed the experiment**. The comparison is invalid.

**Analogy:** You can't compare two students' test scores if they took different tests.

### Artifacts Over Vibes

Every run produces:
- `results.jsonl` — per-example scores
- `summary.json` — aggregate statistics
- `manifest.json` — environment fingerprint (hashes of inputs)

We don't say "it seems better." We say "pass rate increased from 23% to 31% on the same dataset with the same scorer."

---

## 🛠️ Project Overview & First Task (10 min)

### What Students Will Do

1. **Level 0:** Measurement Hygiene
   - Run evaluation on frozen completions
   - Tamper with data, observe what breaks
   - Learn to trust artifacts, not intuition

2. **Level 1:** Build a Scorer
   - Implement reward specification
   - Handle edge cases
   - Learn that small spec changes → big behavior changes

3. **Level 2:** Selection Experiments
   - Implement Best-of-N selection
   - Measure pass@1 vs pass@N
   - Understand inference-time compute scaling

4. **Level 3:** Learning Experiments
   - Observe policy gradient updates
   - See advantage/baseline in action
   - Understand KL tradeoffs

### The Core Contract

Everything revolves around one function:

```python
score(example, completion) -> {"reward": float, "details": dict}
```

Where:
- `example` = prompt + expected answer + id
- `completion` = model's output string
- `reward` = 0 or 1
- `details` = diagnostic information (parse status, error codes, etc.)

### First Commands to Run

```bash
# Setup
poetry install

# Loop A: Evaluate frozen completions
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl

# Inspect failures
python -m course.inspect_run --run runs/<run_dir> --only-fails --top-k 5
```

---

## 💡 Key Takeaways for Students

### Before they leave this meeting, students should understand:

1. **RL is how modern LLMs are trained** — not just text prediction, but reward optimization

2. **The three loops** — Evaluation, Selection, Learning — each has a distinct purpose

3. **Measurement matters most** — a broken scorer means broken training

4. **Artifacts over vibes** — always save results, always compare fairly

5. **Strictness is intentional** — tight specifications force correct behavior

### The Mindset Shift

❌ **Wrong:** "The model is dumb, let's train it more"

✅ **Right:** "Something changed. Was it the model, the data, or my measurement?"

---

## 📋 Pre-Work for Next Session

1. Clone the repository and run `poetry install`
2. Execute the first eval command and inspect results
3. Read the first 2 pages of [Introduction.md](Introduction.md)
4. Write down 3 questions you have about RL

---

## 🎯 Learning Objectives Check

By the end of this project, students will be able to:

- [ ] Explain why RL is used for LLM training
- [ ] Distinguish between the three operational loops
- [ ] Implement and validate a reward function
- [ ] Compare experiments using the Locked Room principle
- [ ] Interpret selection vs learning gains
- [ ] Explain KL regularization and why it matters

---

## 📖 Glossary (Quick Reference)

| Term | Meaning |
|------|---------|
| **Policy** $\pi_\theta$ | The model; a probability distribution over completions |
| **Reward** $R(x,y)$ | Score for a (prompt, completion) pair |
| **Objective** $J(\theta)$ | Expected reward; what we maximize |
| **Advantage** $A = R - b$ | Reward relative to baseline |
| **pass@1** | Success probability of single sample |
| **pass@N** | Probability at least 1 of N samples succeeds |
| **KL Divergence** | Measure of how much policy changed |
| **Locked Room** | Principle: keep environment fixed for valid comparisons |

---

> **Teacher note:** After this session, students should feel that RL for LLMs is *demystified* — not fully understood, but no longer intimidating. The technical details come later. The goal here is conceptual grounding and motivation.
