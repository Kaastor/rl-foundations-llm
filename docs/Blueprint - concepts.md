  

---

  

# RL Foundations for LLMs

  

## Concept‑First, Verifiable Rewards Track

  

LLMs have reached the point where raw “make it bigger” progress is no longer the whole game. The frontier is increasingly about **alignment between intent and behavior**: turning a general-purpose model into something reliable, useful, and safe in specific contexts. That’s what modern post-training is—choosing what “good” means, measuring it honestly, and pushing the model toward it without breaking everything else.

  

This course is your on-ramp to that reality. It focuses on the core capability behind RLHF, RLVR, and even simple sampling tricks: **closing the feedback loop**. You’ll learn how to define success in a way a machine can check, how to tell real improvement from metric theater, and how to think clearly about optimization when the system will happily exploit loopholes. It’s less about memorizing algorithms and more about gaining the mindset that lets you navigate the current LLM landscape without getting hypnotized by tooling, hype, or dashboards.

  

This is a **foundations course** using one small anchor project to teach the mental model of RL and RL-for-LLMs without drowning the student in tools, infrastructure, or configuration. The project exists to **teach thinking**, not ship a product.

  
  

---

## Course at a glance (what this is / isn’t)

  

**This course is about mental models + measurement discipline.**

We will **not** build training infrastructure or a multi-turn agent.

  

- **Compute:** runs on a laptop (CPU). No GPU required.

- **Primary artifact:** a tiny, testable `score()` harness + a hardened reward spec.

- **Primary skill:** tell the difference between

- a model failure,

- a measurement/spec failure,

- and a “we tricked ourselves” failure.

- **End state:** the student can explain RL-for-LLMs clearly enough to *design* a correct experiment, not just run a framework command.

- In this course, “**verifier**” just means “the deterministic scorer we run locally” (the `score()` harness), not a specific library.

- Each dataset item includes `{prompt, expected_answer}`; the scorer compares the parsed final answer to `expected_answer`. The scorer does **not** solve math.

  

### Verifier vs Reward Model (do not mix these up)

  

**Rule-based reward (Verifier) — used in this course**

- A deterministic program (your `score()` harness).

- Fast, reproducible, versioned like software.

- Doesn’t “drift,” but it *can* be wrong due to spec bugs, parsing bugs, or loopholes.

  

**Model-based reward (Reward Model / RM) — conceptual only**

- A learned scorer trained to predict human preference or approximate a rule.

- Can drift, be biased, hallucinate, and be reward-hacked in its own special ways.

- Requires its own holdouts, audits, and monitoring like any other model.

  

**Pedagogical rule for this course:**

If you can write the scoring rule directly (cheaply and clearly) in Python, do **not** train a reward model to guess it.

  
  

## Design principles (guardrails against distraction)

  

1. **Concept > tooling.** Tools are introduced only when they illuminate a concept.

2. **One moving project, tiny scope.** Single‑turn task, deterministic/verifiable reward. **No multi‑turn agents.** No training infra required.

3. Every milestone produces two artifacts:

- a **1–2 page note** (student’s words + diagrams), and

- No plumbing/framework wiring. Required code is small and local: tests + explicitly marked exercise stubs.

1. **The one scorer interface (used everywhere)**

  

To avoid confusion, this course uses exactly one scoring signature:

`score(example, completion) -> {reward: float, details: dict}`

Where `example` is a small record (from the dataset) that includes:

- `prompt`

- `expected_answer`

- (optional) `id` or metadata for logging

Important:

- The scorer receives the **completion** (model output) and compares it to `expected_answer`.

- The scorer does **not** solve the math; it only verifies.

- If you see any other `score(...)` signature in notes or code, treat it as a bug and fix the wording.

  
  

5. **Good mindset throughout:**

- treat reward/eval as a **spec**

- make failures **inspectable**

- build **regression tests**

- avoid self‑deception: “did we improve the system, or just the metric?”

6. **Most real-world RL failures are not “algorithm problems.”**

They’re **reward/spec**, **data**, and **measurement** problems.

The algorithm is usually the boring part — and that’s a feature.

  

---

  

### Stop Doing Extra Things (seriously)

  

This course is designed to be small on purpose. To prevent scope creep:

  

- Do **not** add new datasets or new task types.

- Do **not** widen accepted answer formats unless a milestone explicitly says to.

- Do **not** change prompts *and* scorer/spec in the same run.

- Do **not** implement PPO, actor-critic, reward models, or any RL framework wiring.

  

Passing this course = measurement discipline + reward-as-spec thinking + correct mental models.

Shipping tools is explicitly out of scope.

  

---

  

## What “success” means

  

By the end, the student can confidently explain:

  

1. RL as **behavior optimization under feedback** (not label fitting).

2. The RL loop with the right objects: **policy, trajectory/episode, reward, return, horizon/termination, exploration**.

3. **LLM = policy over token sequences** and why credit assignment is hard when reward arrives at the end

4. The two compatible ways to model LLM behavior:

  

* **macro-action**: completion as a single action (great for eval + selection),

* **micro-action**: tokens as actions (what training optimizes).

5. The LLM RL pipeline: **rollouts → score → update under constraints** (and what each box is responsible for).

6. How to build **verifiable reward** (RLVR-style thinking) and harden it against **reward hacking**.

7. How to evaluate reliably and separate:

- model failures

- dataset/spec issues

- verifier/parser bugs

8. The difference between the **true objective** (“maximize expected reward/return”)

and the **training surrogate** (what we actually optimize, e.g., via log-prob tricks + constraints).

In particular: why you usually **can’t backprop through** a verifier/reward.

  

9. Why **Best-of-N** is “optimization at inference time” (selection), and how it differs from

**training-time** improvement (learning): one spends compute to pick better samples,

the other changes the policy distribution itself.

  

---

  

## The mental map (the 3 loops you must not confuse)

  

This map is repeated until it’s automatic.

  

### Loop A — Evaluation (measurement)

  

Generate outputs → score them → inspect failures.

No learning yet. Just **truthful measurement**.

  

### Loop B — Selection (optimization without gradients)

  

Sample N outputs → pick best by reward/verifier.

This is already “policy improvement” behaviorally.

> Best-of-N does not change the policy. It changes the _decision rule_ applied to samples from the same policy.

  

### Loop C — Learning (RL training)

  

Sample outputs → score → update policy probabilities (often with constraints like KL).

This is where PPO/policy gradients live _conceptually_.

  

**Reality check:** always ask “Which loop am I in right now?”

Also name the knob you turned: **policy** (learning), **selection compute** (best-of-N), **measurement instrument** (scorer/spec), or **environment** (prompt/dataset).

  

---

  

## The “do-not-confuse” set (terminology that must stay untangled)

  

**Reward `r`**

- A number produced *by the environment / verifier* after an action (or after a whole completion).

- In this course: produced by `score(prompt, completion)`.

  

**Return `G`**

- The total outcome signal over time (sum of rewards along a trajectory).

- In single-turn tasks: `return = reward` (helpful simplification—don’t forget it’s a special case).

  

**Objective `J(θ)`**

- **Formal objective `J(θ)`**

- What optimization actually targets: maximize **expected return under the implemented reward spec**.

- Think: “average reward if we sampled from this policy many times, using *this scorer*.”

  

- **True goal (latent)**

- What humans actually want (often richer than any proxy).

- The reward spec is an attempt to approximate this goal, but it can diverge under optimization (Goodhart).

  

- Practical rule:

- When reward and the true goal disagree, the optimizer will follow **reward**.

- Our job is to shrink that gap via better specs, tests, holdouts, and red-teaming.

  
  

**Metric**

- A measurement we track (pass rate, mean reward, etc.).

- A metric can be *correlated with* the objective, but it can also drift into nonsense if the spec is wrong.

  

**Loss / surrogate**

- The math trick we optimize to shift probabilities in the direction that increases expected reward.

- Key idea: reward is usually not differentiable → learning happens via sampling + probability shifts.

  

**Policy**

- A distribution over actions.

- For LLMs: a distribution over token sequences given a prompt.

  

**Environment (LLM RL version)**

- In classic RL: the environment is the thing you act in; it returns observations and rewards.

- In many RL-for-LLMs setups (especially single-turn): the “environment” is effectively:

- a **prompt distribution** (dataset), plus

- a **scorer** (verifier or reward model), plus

- optional constraints/rules (formatting, safety filters).

- The key idea: the environment defines **what gets rewarded**, and that is what optimization will exploit.

  

**Baseline & advantage**

- Baseline: a reference level used to reduce variance.

- Advantage: “better/worse than baseline.”

Sign intuition: advantage > 0 reinforces what was sampled; advantage < 0 suppresses it.

- **Baseline scope rule (no critic in this course):**

- In this course, “baseline” means a simple deterministic reference (e.g., a running mean reward, or per-prompt mean).

- We are **not** learning a value function / critic network here. The goal is variance intuition, not actor-critic.

  

**One sentence sanity check**

If you can’t say which of these you’re looking at (reward/return/objective/metric/loss), you are not doing RL—you are doing word salad.

  
  

---

  

## Measurement hygiene

  

These rules prevent the most common failure: “we improved the number, not the system.”

  

* Keep a small **holdout set** that you do not tune on.

* Version your scorer/spec (even just a git commit hash) alongside every eval run.

* Never compare metrics across **different scorer/spec versions** as if they were the same number.

If the spec changed, treat it like a new measurement instrument.

* Prefer deterministic scoring. If randomness exists, record seeds.

* Save raw model outputs for inspection (not just aggregate metrics).

* Look at the **distribution**, not just the mean:

- pass rate, mean reward, and a quick “top 5 / bottom 5” failure inspection.

Averages hide reward hacking and spec bugs.

  

- **Ignore small changes (Noise Rule):**

- On small datasets (N < 100), a 2–3% improvement is usually just luck.

- Only treat an improvement as "real" if it is **large (>5%)**

  

* Add “golden” regression cases and run them before trusting new results.

**Verifier quality has two failure modes (track both):**

- **False positives:** wrong outputs get rewarded (reward hacking surface).

- **False negatives:** _in-spec_ correct outputs get 0 (overly strict parsing/normalization).

Maintain a tiny **Golden Correct Set** (known-correct, in-spec completions) and run it before trusting any new scorer change. If any golden case gets 0 reward, treat it as a scorer/spec bug (or a spec clarification you must make explicit).

* Keep an **experiment log** per run:

- dataset version

- scorer/spec version

- prompt/system prompt version

- sampling settings (temperature, top_p, seeds)

- the single change you made vs last run

  

* Change **one thing at a time** until you have stable measurement.

If you changed the prompt *and* the scorer *and* the dataset, you learned nothing.

* **Locked Room Rule (valid comparisons):** When you are measuring **Selection (Loop B)** or reasoning about **Learning (Loop C)**, the **environment is fixed**: same prompt(s), same scorer/spec version, same dataset split.

- Changing the prompt is an **environment change**, not “the policy learned.”

- If you change the prompt, label it as a new environment version and restart the comparison from a clean baseline.

This prevents confusing _prompt engineering_ with _policy improvement_.

  

---

  

## The hidden simplification: most single-turn LLM RL is a contextual bandit

  

For this course’s anchor project (single-turn, reward on completion), the **macro-action view** is a **contextual bandit**:

  

- Context/state: the prompt

- Action: the whole completion

- Reward: verifier score for that completion

- No environment “transition” beyond choosing the action

  

This matters because it explains why these work so well *without* heavy RL machinery:

- **Loop A (evaluation):** score completions honestly

- **Loop B (best-of-N):** sample more, pick best (optimization without learning)

- **Loop C (learning):** shift probability mass toward high-reward completions

  

Under the hood, the **micro-action token view** is still sequential (tokens are actions), but the bandit lens keeps the foundations clean: reward engineering + measurement discipline first, then learning.

  

### Important nuance: bandits are a special case of RL (not a different thing)

  

RL in full generality is *sequential decision-making* (actions affect future states and future rewards).

  

This course’s anchor project is a **single-step special case** (a contextual bandit):

- there is no state transition to model,

- the “episode” length is 1 at the macro-action level,

- and return = reward.

  

However, for LLMs the **token-level view is still sequential**:

- tokens are actions,

- prefixes are states,

- and reward often arrives at the end (delayed/sparse credit assignment).

  

**Reality check**: **Bandit framing is a simplification for clarity, not a denial of sequential RL.**

You must be able to move between:

- bandit lens (debugging + measurement + selection), and

- sequential lens (token credit assignment + training).

  
  

---

## RL methods: the 1-page family tree (so you don’t get lost later)

  

We only need a *map*, not the full math.

  

- **Value-based RL** (e.g., Q-learning): learn a value of actions, derive a policy from it.

*Not covered here* (great later, not needed for RL-for-LLMs foundations).

  

- **Policy-gradient RL** (e.g., REINFORCE → PPO-style): directly shift policy probabilities

toward rewarded behavior. **This is the main conceptual tool for LLM RL.**

  

- **Model-based RL**: learn or use a world model to plan.

*Not covered here* (orthogonal complexity).

  

**Why this matters:** “RL” is a category, not a single technique.

This course teaches the mental model you’ll reuse when you later see PPO/GAE/critics.

  

## The other mental map (macro-actions vs micro-actions)

  

This course uses single-turn tasks to keep scope small, but we must keep two compatible views in mind:

  

**Macro-action view (completion as the action)**

  

* State ≈ prompt/context

* Action ≈ the whole completion

* Reward ≈ verifier score for that completion

* This framing is perfect for: evaluation, best-of-N, reward engineering, debugging.

  

**Micro-action view (tokens as actions)**

  

* State ≈ prefix (prompt + generated tokens so far)

* Action ≈ next token

* Trajectory ≈ full token sequence

* Reward often arrives at the end → credit assignment becomes hard

* This framing is what RL training (e.g., PPO-style updates) actually operates on.

  

**Reality check mantra**: **know which granularity you’re reasoning in**.

  
  

---

  

## Project (single anchor throughout)

  

**Project:** Use a tiny “Math Answers” scoring harness with a **verifiable rubric** (single-turn) to teach:

- measurement discipline, reward-as-spec thinking, and how RL-for-LLMs maps onto this loop.

- The teacher provides the harness code; the student uses it, inspects it, and learns to reason about it.

  

### Anchor task constraints (to prevent scope creep)

  

To keep the project *foundational* (not “build a grading product”), the task is intentionally narrow:

  

- Output format is **exactly one line**: `Final: <int>`

- `<int>` is a **base-10 integer** (no words, no commas, no units, no extra punctuation).

- Verification is **exact string/number equality** after minimal normalization (e.g., trimming whitespace).

- No partial credit. Default reward is **0/1 pass-fail**.

- Prompts are written to guarantee a **single unambiguous** correct answer.

  

Out of scope (on purpose):

- symbolic equivalence, simplification rules, units, rounding tolerances, multiple valid formats.

If we ever add these later, it will be teacher-provided and explicitly scoped as a new environment/spec version.

  
  

#### Why this project?

  

* Math gives you **objective verification** (RLVR-style) without preference-model complexity.

* Single-turn removes multi-turn agent complexity.

* It forces correct thinking about **reward specs, parsers, and failure modes**, which is *the* real-world RL skill.

  

**The tool-agnostic core interface (the real curriculum)**

  

Everything in the course can be expressed as:

  

`score(example, completion) -> {reward: float, details: dict}`

`

  

We keep everything inside the course harness (no external RL/eval frameworks).

This preserves the mental model and removes wiring/config as a failure mode.

  

> Important: we’re not “training a model.” We’re building the *evaluation + reward machinery* and the *mental model*.

  
  

---

  

## Teacher role (high leverage, low time)

  

At each milestone: async review or a 30–45 minute check‑in focused on:

  

- “Explain the design choices in plain English.”

- “Show me 2 failures. Diagnose root cause.”

- "Propose one fix. **Encode it as a spec change + a regression test**. Re‑measure."

- “How could a policy exploit this reward?”

**Exploit mindset:** Think like a lazy student optimizing for points, not a malicious attacker.

Ask: “What is the absolute lowest-effort way to get a 1 from this scorer?”

  
  

No long lectures. The artifacts do the teaching.

  

### Teacher invariant checklist (mental map must be stable)

  

By Milestone 3, the student should be able to say (without notes):

  

1) **Which loop are we in** (A eval / B selection / C learning) and what changes in each.

2) What is the **policy** in this system (a distribution), and what is an “action” at macro vs micro level.

3) What is the **environment** here (prompt distribution + scorer + constraints).

4) Where does the **data** come from (current policy vs frozen/offline) and why that matters.

5) Why reward is not a label: **objective vs reward spec vs metric vs surrogate loss**.

6) How reward hacking happens (Goodhart) and how to defend (tests + red teaming + holdout).

  
  

---

  

# Milestones (order is the point)

  

## Milestone 0 — Orientation + vocabulary map + first measurement (1–2 days)

  

### Goal

  

Install the right mental objects **before** code feels “real,” and make Loop A concrete immediately.

  

### Teacher provides (start tiny; expand only when needed)

  

A repo with a **sealed core** (teacher-owned) and a **student workspace**:

  

- `course/` — teacher-owned code (read/inspect, do not refactor)

- `score.py` (the scorer contract + implementation)

- `eval.py` (runs scoring over datasets; saves inspectable outputs)

- `datasets/` (loading + schema; tiny and boring on purpose)

- `data/` — frozen rollouts + tiny datasets (teacher-provided)

- `runs/` — saved eval outputs (raw completions + scored details)

- `tests/` — regression tests (student adds cases; teacher keeps harness stable)

- `MEASUREMENT_RULES.md`

  

### Student writes (1–2 pages, diagrams encouraged)

  

**“RL mental map v1”** answering:

  

- agent, environment, action, observation/state, episode/trajectory, reward, return

- reward vs loss vs metric (and why confusing them breaks reasoning)

- policy = **a distribution**, not a single answer

- diagram: RL loop + “LLM as policy generating tokens”

  

### Loop A immediately (low friction)

  

Teacher provides a small file of **frozen rollouts** (prompt → model output).

Student runs scoring + prints failures (no model setup required yet).

  

### Guardrail: frozen rollouts are for Loop A/B clarity, not a general training pattern

  

Frozen rollouts let us start with evaluation and debugging immediately.

  

But conceptually, **learning changes the policy, and the policy changes what data you see next**:

policy → rollouts → scores → update → new policy → new rollouts.

  

So:

- Frozen rollouts are perfect for **Loop A (evaluation)** and **Loop B (selection demos)**.

- Real RL learning is typically **on-policy** (or carefully handled offline RL, which we do not cover here).

  

**Reality check**: always ask **“Who generated this data—this policy, or a different one?”**

  

**Frozen rollouts lie to you about the future (distribution shift warning).**

  

Frozen rollouts are an **offline snapshot**: “data generated by some earlier policy.”

That’s perfect for Loop A (evaluation) and Loop B (selection), but it hides the most dangerous RL reality:

  

- In Loop C (learning), the policy changes.

- When the policy changes, the distribution of generated completions changes.

- New policies discover **new failure modes** (mode collapse, reward hacks, weird formats) that never appear in the frozen snapshot.

  

Concrete rule:

> You cannot evaluate a Step‑100 policy on Step‑1 data and think you measured Step‑100 behavior.

  

This is why on-policy rollouts (or carefully managed offline RL, which we do not cover here) matter,

and why KL constraints exist: they limit how fast the policy can drift into “new weird space.”

**Diagram trigger (Frozen Data / Distribution Shift):**

Draw two clouds: `π0(completions)` and `πt(completions)` drifting apart over training steps,

with the frozen dataset sampling only from `π0`. Show a “blind spot” region where `πt` puts mass

but the frozen set has no coverage.

  

### Good habits introduced

  

**Failure taxonomy early:**

  

- model wrong

- verifier wrong

- data ambiguous

- spec unclear

  

### Debugging kata

  

Teacher provides 10 scored examples where *something* is wrong, but the reason differs:

  

- 4 are genuine model failures

- 2 are verifier/parser bugs

- 2 are ambiguous prompts/spec ambiguity

- 2 are “reward got hacked” style failures

  

Student task (30–45 minutes):

- For each example, label the failure type and justify in 2–3 sentences.

- Propose one concrete fix (spec change, test, prompt tightening, or “don’t include this example”).

  

This kata repeats later in Milestone 4/5 with harder cases.

  

### Teacher checklist

  

- Are they mixing up **reward vs objective vs metric**?

- Do they understand RL as sequential decision-making **in general**, and that our anchor task is a **single-step special case** (contextual bandit) while token-level behavior remains sequential?

  

---

## Milestone 0.5 — Selection is already optimization (Best-of-N) (0.5–1 day)

  

### Goal

Make Loop B real *immediately* using frozen rollouts (no model setup).

Student learns the difference between:

- “the policy improved” vs

- “we spent more compute to pick a better sample.”

  

### Student does

Teacher provides a small pack: for each prompt, **N candidate completions** (N=4 or 8) + the scorer,

plus a ready-to-run script `course/selection_demo.py`.

  

Student either:

- (A) runs it as-is and explains the results, or

- (B) edits only one clearly-marked function:

`pick_best(samples) -> best_sample` (5–15 lines, no plumbing).

  

**Determinism rule for selection:**

If multiple samples have the same reward, break ties by a fixed rule

(e.g., choose the lexicographically smallest completion or the first by a stable id).

Log the chosen id so results are reproducible.

  

- Eval comparing:

- **N=1** (take first sample)

- **N>1** (choose max reward)

- **Invariant (repeat until automatic):**

- Selection (Best-of-N) improves the **chosen output**, not the **policy distribution**.

- Learning (RL) changes the **policy distribution** so good outputs become more likely even at N=1.

  

> **Mental Check: “Why not just train on the Best-of-N?” (Selection vs RFT vs RL)**

>

> A smart instinct is: “If Best-of-N finds good samples, why not just fine-tune on those?”

> That idea is common and often useful — but it is **not the same thing** as RL.

>

> **Best-of-N (Selection, Loop B):** spend more sampling compute, pick the best output **at inference time**.

> **Rejection-sampling fine-tuning (RFT / filtered SFT):** take the best samples you found, and do supervised fine-tuning to **clone those specific trajectories**.

> **RL (Learning, Loop C):** update the policy to shift probability mass toward **kinds of behaviors** that yield high reward (in expectation), typically using **fresh rollouts from the current policy**.

>

> **Why this matters (especially with frozen rollouts):**

> - With **frozen/offline** rollouts, you are limited by what the *old* policy happened to explore. Filtering + SFT cannot reliably improve beyond the dataset’s coverage.

> - In real RL, the policy changes → it must generate **new rollouts** → it can discover new high-reward behaviors *and* new reward hacks/failure modes.

>

> **Course note:** We use frozen rollouts for speed and clarity (Loops A/B). When we talk about Loop C, assume the real setting: the policy must generate its own data to improve beyond the initial snapshot.

  
  

- Reports:

- average reward

- pass@1 vs pass@N

- 2–3 failure cases where selection still fails (why?)

  

### Deliverables

- `runs/run_00X/selection_demo.jsonl` (or whatever format your harness writes)

- short comparison summary in `runs/run_00X/README.md`

  

### Good habit introduced

Always label your improvement:

- **selection improvement** (more samples) or

- **learning improvement** (policy changed)

  

### Exploration in LLM terms (no jargon, just reality)

  

In LLMs, “exploration” often looks like **sampling diversity**:

- temperature / top_p / seeds change the distribution you sample from,

- increasing N in Best-of-N is “pay more exploration compute,”

- picking the best by a scorer is “exploit what you found.”

  

This course treats sampling settings as first-class experiment parameters:

record them, version them, and never compare runs that silently changed them.

  

## Milestone 1 — RL update step in slow motion (bandit, no calculus) (2–3 days)

  

### Goal

  

Make RL feel non-magical: **probabilities shift toward rewarded behavior**.

  

### Student does

  

**Part A — Slow-motion policy gradient demo (teacher script)**

  

- policy params (`theta`)

- action probs

- sampled action

- reward

- baseline → advantage sign intuition

- gradient direction intuition (why log-prob of rewarded actions increases)

- updated params and probs

**Mini-clarifier: objective vs reward vs training surrogate**

  

- **Reward**: a number the environment/verifier gives you after an action/trajectory.

- **Objective**: what we *wish* to maximize: expected reward/return under the policy.

- **Surrogate training loss**: the mathematical trick we optimize to *move probabilities*

in the direction that increases expected reward (because reward is not a differentiable label).

  

Rule of thumb: if your “reward” is produced by a parser/verifier, you usually can’t

backprop through it — you learn by **sampling + probability shifts**.

  

#### How a single completion reward trains token probabilities (the log‑prob trick)

  

Even if `score()` returns **one scalar** reward for the entire completion, the policy is still

a **distribution over token sequences**.

  

Key identity (no calculus required):

  

- A completion is tokens: `y = (y1, y2, ..., yT)`

- The model assigns conditional probabilities per token:

`πθ(y) = ∏t πθ(yt | prompt, y< t)`

- Taking logs turns the product into a sum:

`log πθ(y) = Σt log πθ(yt | prompt, y< t)`

  

**Policy-gradient intuition (REINFORCE view):**

- We don’t “teach the correct answer.”

- We **increase the likelihood of the sampled trajectory** if it beat baseline, and decrease it if it was worse.

  

A simple surrogate you can read like English:

  

- Let `A = (reward − baseline)` be the advantage.

- Optimize something proportional to:

  

`loss = − A * Σt log πθ(yt | prompt, y< t)`

  

> **Note:** `log` here is the natural log (`ln`). The base does not change the gradient direction, but it changes the numeric scale you’ll see in logs.

  
  

Interpretation:

- If `A > 0`, minimizing `loss` pushes the summed log-prob **up** → the sampled tokens become more likely.

- If `A < 0`, it pushes the summed log-prob **down** → the sampled tokens become less likely.

  

This is the concrete answer to:

“How does Reward: 1.0 change the probability of token ‘4’ generated 50 tokens ago?”

Because that token contributed one term in the sum, and **the same advantage scalar multiplies the whole sum**.

  

**Diagram trigger (Gradient Connection):**

Draw a token chain `prompt → y1 → y2 → ... → yT → score() → A`,

then show `A` as a single scalar “gain knob” multiplying a vector labeled

`∇ log πθ(y)` (the “log-prob gradient”), which flows back into the model update.

  
  

**Critical requirement:**

Before each step the student writes 1–2 lines predicting:

  

- “Which probability should go up/down and why?”

**Part B — Fast run**

  

* Run the same demo script in normal/fast mode and observe learning curves.

* Student writes 5–8 lines answering:

  

- What improved over time (reward/accuracy/return)?

- What stayed noisy, and why?

- What changed when baseline was on vs off?

  
  

**Part C — Student audits a tiny bandit learner (teacher code)**

Teacher provides `course/bandit_train.py` with:

- explicit probabilities

- explicit sampled action

- explicit reward

- baseline + advantage shown

- one-step logs that can be read like a story

  

Student task:

- For 10 consecutive steps, annotate the log in plain English:

“what got reinforced/suppressed and why.”

- Run the **baseline on/off** ablation and explain the variance difference.

  

Optional (small edit, concept-only):

- Change *one* line (learning rate or baseline choice) and predict the effect before running.

  

### Deliverables

  

- `runs/run_00X/bandit_train_log.txt` (or `.jsonl`)

  

### Good habits introduced

  

- **Observability:** if you can’t explain one update step, you don’t understand the system.

- **Ablations:** remove baseline; observe variance/instability.

  

### Teacher checklist

  

- Can they explain exploration vs exploitation without slogans?

- Do they understand learning as **changing probabilities**, not “finding the answer”?

  

**Scale intuition (“Toy Tax”):**

The bandit demos learn fast because the action space is tiny and the reward is clean. Real LLM post-training is often **noisy and slow**: the space of possible token sequences is enormous and the reward signal can be sparse or brittle.

Treat toy learning curves as a microscope for the mechanism—not a promise about how quickly real systems improve.

  

---

  

## Milestone 2 — From bandits to sequences: credit assignment (2–3 days)

  

### Goal

  

Correctly frame the hard part for LLM RL: **many actions, reward often at the end.**

  

### Student does

  

1. **Two-step toy MDP (delayed reward) — teacher script**

- 2 steps, 2 actions per step, reward only at the end

- Student explains why step-1 action matters despite delayed reward

  

2. **Connect delayed reward to the token view**

- 5–8 lines explaining how “reward at the end” maps to:

“many token actions happened before we saw a reward signal.”

- Add one diagram:

prompt → tokens → end reward (sparse) → credit assignment problem

  

3. **Return + horizon intuition**

- return = cumulative outcome signal

- horizon/termination = how far consequences propagate

  

### Deliverables

  

- runs/run_00X/toy_mdp_log.txt

### Good habits introduced

  

- **Credit assignment is a debugging problem:** ask where the reward signal actually comes from.

  

### Teacher checklist

  

- Can they explain sparse/delayed reward and why learning becomes unstable?

- Do they understand why “completion = many actions” matters?

  

---

  

## Milestone 3 — LLM mapping + constrained optimization mental model (2–3 days)

  

### Goal

  

Map RL objects onto LLMs cleanly, without framework distractions.

  

### Student writes (no heavy implementation)

  

**“LLM as policy”** note including:

  

- state ≈ prompt/context

  

**State vs observation (LLM-friendly framing)**

- For clean mental models, we often treat the **prompt** as “the state.”

- More precisely, the token-level “state” is the full **prefix**:

prompt + generated tokens so far.

- This is why LLM RL is sequential at the micro level even when scoring is completion-level.

  

- action ≈ next token (or completion as macro-action)

  

- trajectory ≈ token sequence

- reward ≈ verifier score (often sparse/delayed)

  

#### Task: See the micro-actions (token inspection) (5–10 minutes)

  

Students often hear “tokens are actions” but never *see* the action space once. Do it once here.

  

**Teacher provides:** a tiny script that prints the tokenizer output for a string.

  

**Student does:**

- Run the token inspection script on a few completions (including near-misses):

- `Final: 323`

- `Final: 323` (double space)

- `Final: 0323` (leading zero)

- `Final:\n323` (newline)

- Record the resulting token list for at least two variants.

  

**What to notice:**

- Token boundaries change the effective action space. For example, `" 323"` may be a single token while `"3","2","3"` are three tokens.

- The verifier/scorer operates on **text**, but the optimizer operates on **token log-probs**.

If parsing/normalization is strict, small tokenization/whitespace differences can create “phantom failures” that look like model incompetence but are actually interface/spec bugs.

  

**One-sentence takeaway:** The policy makes a decision at *every token boundary*, even if reward arrives only at the end.

  
  

#### The “scalar reward → token update” mechanism (one paragraph, no magic)

  

Even when reward is only computed on the final detokenized completion,

the **probability of the whole completion** is still built from token probabilities:

  

`log πθ(completion) = Σt log πθ(token_t | prefix_<t)`

  

Policy-gradient updates use this decomposition: the reward signal (via advantage) acts like a single

multiplier on the summed log-prob of the sampled tokens, pushing the policy to make that sampled

token trajectory more (or less) likely next time.

  

This is reinforcement, not correction: we’re not backpropagating through `score()`;

we’re shifting probability mass toward trajectories that scored well under the verifier.

  
  

**Also cover (conceptually):**

  

- why KL-to-reference exists (“improve reward without drifting into nonsense”)

- why reward noise breaks training

- why “format rewards” create perverse incentives

### Mini-demo (teacher-provided): KL-constrained policy improvement (30–45 min)

  

To make “KL to a reference policy” feel concrete (not slogan-y), the teacher provides

  

`course/kl_tradeoff_demo.py` (no LLMs, just a categorical policy over K actions).

  

**What it demonstrates**

  

* A policy can increase expected reward by shifting probability mass toward higher-reward actions.

  

* A KL penalty keeps the learned policy near a **reference policy** (the “don’t drift into nonsense” constraint).

  

* Students see the **tradeoff curve**: as constraint weakens → reward ↑, KL ↑.

  

**Setup (teacher code)**

  

* Define a reference policy `pi_ref(a)` over K actions.

  

* Define a reward per action `r(a)` (or a tiny stochastic reward sampler).

  

* For a sweep of penalty strengths `beta` (or `kl_coef`), compute an updated policy that optimizes:

  

`E_pi[r(a)] - beta * KL(pi || pi_ref)`.

  

* For each `beta`, log:

  

* mean reward under `pi`

  

* `KL(pi || pi_ref)`

  

* the top-3 actions by probability (so the change is interpretable)

  

**Student task**

  

* Run the script once.

  

* Write 6–10 lines interpreting the curve and linking it back to LLM RL:

  

* “reference policy” ≈ the pretrained/SFT model

  

* KL penalty ≈ “improve reward without drifting into weird space”

  

* stronger KL → safer/closer behavior but less reward improvement

  

---

  

### Deliverables

  

- `notes/llm_as_policy.md` with token-trajectory diagram + end reward

- minimal pseudocode (~15 lines) for rollouts → score → constrained update (conceptual)

* `runs/run_00X/kl_tradeoff.csv` (or `.jsonl`) and optionally `runs/run_00X/kl_tradeoff.png`

  

### Good habits introduced

  

- Separate concerns: generation vs scoring vs update.

- Always name your loop (A/B/C).

  

### Teacher checklist

  

- Can they explain why constraints (KL, penalties) are there without math?

- Do they understand the difference between token-level actions vs completion-level scoring?

### 10-minute oral exam (teacher uses this to detect “word-salad RL”)

  

Ask the student to answer in plain English:

  

1) What are the **actions** in the macro-action view vs micro-action view?

2) What exactly does `score()` represent: reward, metric, objective, or loss?

3) Why can’t we usually backprop through `score()`?

4) What changed between Loop B and Loop C?

5) What does KL constrain, and what failure mode is it trying to prevent?

6) If reward goes up but holdout pass rate doesn’t, name 3 likely causes.

  

---

  

## Milestone 4 — Build verifiable reward as a spec (tests first) (3–5 days)

  

### Goal

  

Teach the most transferable RL skill: **reward is a specification**, engineered like production code.

No external frameworks. No adapters. Just one harness and ruthless clarity.

  

### Student does (order matters)

  

1. **Write / refine the reward spec first (plain English, testable)**

- what counts as correct (format + semantics)

- what must not get reward (common hacks)

- what is intentionally *out of scope* (avoid accidental complexity)

  

**Golden Correct Set (anti–false-negative gate)**

Before you evaluate models or celebrate “reward went up,” validate the scorer against a tiny set of **known-correct, in-spec answers** (provided by the teacher or written by a human).

- If any golden case scores 0, you have a **false negative**: fix parsing/normalization/spec/tests first.

- Only proceed once the scorer reliably rewards _in-spec_ correct completions.

This stays within your existing “tests-first hardening” logic—just makes the gate explicit.

  

2. **Read the teacher’s scorer like a spec**

Teacher provides `course/score.py` (clean, typed, heavily commented).

Student must be able to point to:

- where parsing happens

- where rubric checks happen

- how failures become `details` (inspectability)

- how determinism is enforced

##### The scorer contract (non-negotiable)

  

Your `score(prompt, completion)` must be:

  

1. **Deterministic** (same input → same output).

2. **Total** (never crashes; returns `{reward, details}` even on garbage input).

3. **Explainable** (details show *why* the score happened: parse result, checks passed/failed).

4. **Fast** (target: <50ms per sample on laptop CPU; no network calls).

5. **Versionable** (spec changes are tracked; scorer version recorded with eval runs).

  

##### A concrete `score()` example (anchor the mental model)

  

- Prompt: `Compute 17*19. Output exactly one line: Final: <int>.`

- Good completion: `Final: 323` → reward 1

- Bad completion: `323` → reward 0 (format)

- Bad completion: `The answer is 323.` → reward 0 (extra text)

  

The point: **reward comes from the environment (the scorer/verifier), not from “knowing the answer.”**

The scorer is a spec. The model will optimize whatever the scorer actually rewards.

  
  

3. **Tests-first hardening (student adds cases, teacher keeps core clean)**

- Student adds unit/regression tests for:

- correct cases

- formatting edge cases

- ambiguous cases (expected behavior is “no reward” unless spec says otherwise)

- Every test must assert both:

- `reward`, and

- at least one key `details` field (so the scorer stays explainable)

4. **Run the harness and do failure forensics (Loop A)**

- Run `course/eval.py` on frozen rollouts.

- Save outputs under `runs/run_00X/` (raw completions + scored details).

- Produce a short failure report:

- 3 model failures

- 2 spec ambiguities

- 2 scorer bugs (or suspicious details)

- 1 reward-hacking attempt (even if it fails)

5. **No framework integration**

The course ends with a rock-solid mental model and a rock-solid scorer harness.

Frameworks come later, when the student can smell measurement bugs from 30 meters away.

  

### The Token ↔ Text boundary (why detokenization/parsing is part of “the environment”)

  

In this course, the verifier operates on **text**:

  

`score(prompt, completion_str) -> {reward, details}`

  

But the policy we conceptually optimize is **token-based** (logits → token IDs).

There is always a boundary component between these worlds:

  

`token IDs → detokenizer → completion string → parser/rubric → reward`

  

If the detokenizer, normalization, or parser is buggy or inconsistent, your RL loop “breaks” in subtle ways:

- The policy optimizes for quirks of whitespace, Unicode, or formatting.

- The scorer may “see” something different than what you think the model produced.

- Correct answers can be mis-scored due to extraction bugs (and the policy will learn to route around them).

  

**Engineering rule:** treat detokenization + normalization + parsing as part of the environment/spec.

Harden them like production code.

  

**Suggested regression tests (add a few):**

- whitespace variants (extra newlines, trailing spaces, double spaces) — this is the #1 source of “model looks dumb but the parser/spec is too strict”

- Unicode lookalikes (“−” vs “-”, non-breaking spaces)

- multiple answers / answer laundering (“Final: …” repeated)

- number formatting variants (leading zeros, scientific notation) *only if* spec allows it

  

**Diagram trigger (Token/Text Interface):**

Draw a vertical stack:

`Policy (tokens/logits)` → `Detokenize` → `Text completion` → `Parse/Extract` → `Rubric checks` → `Reward`,

and annotate: “gradients live above the line; the verifier lives below the line.”

  
  

### Two different “test sets” (do not mix them up)

  

1) **Scorer/spec tests (normative):**

- Unit + regression tests are the *definition* of correct scoring behavior.

- If the spec changes, tests change with it.

- You absolutely *do* “fix” bugs found by these tests.

  

2) **Evaluation holdout set (descriptive):**

- A small prompt set you do **not** tune the scorer/spec/prompt on.

- It exists to detect overfitting, reward hacking, and spec drift.

- If holdout improves but dev doesn’t (or vice versa), investigate before celebrating.

  

### Deliverables

  

- `tests/test_rubric.py` (student-added cases)

- `runs/run_001/` saved outputs + scores

- `notes/eval_vs_reward.md` (metric vs reward vs spec; how they diverge)

  

### Good habits introduced

  

- Measurement is production code.

- Determinism where possible.

- Don’t tune on the test set.

  

### Teacher checklist

  

- Are they measuring parser quirks instead of correctness?

- Can they debug a low score: model vs verifier vs prompt vs data?

  

---

  

## Milestone 5 — Red team the reward (exploit → fix → regression) (3–5 days)

  

### Goal

  

Internalize the unpleasant truth: **policies exploit reward functions.**

  

### Goodhart’s Law (reward hacking is often competence, not stupidity)

  

Reward hacking is usually the policy doing exactly what optimization pressures demand:

it finds a way to score highly on the **proxy** you implemented, even when that diverges from your **true intent**.

  

Two critical reframes:

- “The model is dumb” is often the wrong diagnosis.

- The real diagnosis is: **the proxy metric became misaligned with the goal under optimization**.

  

Early on, your reward often correlates with the real goal (progress feels real).

Past a point, pushing the proxy harder can actively degrade the true goal

(e.g., brittle formatting tricks, loopholes, spec gaming, mode collapse).

  

**Diagram trigger (Goodhart Curve):**

Plot training steps on x-axis.

Plot two curves on y-axis:

- Proxy reward keeps rising.

- True goal rises at first, then bends downward after the optimizer finds loopholes.

Label the divergence point: “reward hacking regime.”

  

### The “Math Privilege” Warning (Proxy Gap)

  

We use Math because the verifier is unusually close to ground truth:

- **Ground truth:** `17 * 19 = 323`

- **Verifier:** equality check (after minimal normalization like `.strip()`)

- **Proxy gap:** near zero

  

Most real RL applications do **not** have this property. The verifier is a proxy for what you actually want.

  

Example: “Write high-quality Python code”

- **Ground truth:** correct, maintainable, readable, safe, solves the user’s real problem.

- **Verifier:** unit tests + linter + type checks.

- **Proxy gap:** large. A policy can produce unreadable or brittle code that still passes tests.

  

**Lesson:** Math hides how much of practical RL work is “fighting the proxy.”

In real deployments, a large fraction of effort goes into improving the measurement device (spec/tests/holdouts/red-teaming), not the optimizer.

  
  

### Student does

  

**Part A — Break it**

  

- **Teacher provides intentionally hackable rubric variants**

- Student produces 2–5 “reward hacking” outputs that score high but are wrong/unhelpful.

  

**Part B — Regression harness**

  

- every exploit becomes a permanent test

- re-run eval; compare before/after

- * Add one “golden exploit suite” file that can be run in <5 seconds and must always pass before you trust a new eval.

  

  

### Deliverable

  

**Red Team Report**

  

- exploit outputs (exact strings)

- root cause analysis (what assumption failed?)

- patch summary

- regression tests added

- tradeoffs (false positives/negatives)

  

### Good habits introduced

  

- Threat modeling for rewards: assume the policy is adversarial.

- Patch the _class_ of failures, not one string.

  

### Teacher checklist

  

- Did they fix the mechanism, not the symptom?

- Did they avoid making the verifier brittle?

  

---

  

## Milestone 6 — From evaluation to optimization: best‑of‑N + RLHF/RLVR mental model (2–4 days)

  

### Goal

  

Connect loops A/B/C: measurement → selection → learning.

  

### Loop C reality check: learning changes the data you will see next

  

Loop A and B can be demonstrated with frozen rollouts, but Loop C fundamentally cannot be understood

without the distribution-shift warning:

  

- A learning update changes the policy.

- A changed policy generates different completions.

- Different completions expose new reward hacks and new failure modes.

  

So any “evaluation” that only uses a fixed, frozen snapshot can badly mislead you about live behavior.

  

Practical takeaway:

- Use frozen rollouts for pedagogy and for debugging the scorer/spec.

- When you reason about learning, assume **on-policy** data unless you are explicitly doing offline RL

(out of scope here).

- KL constraints aren’t just “stability glue” — they are a partial defense against runaway distribution shift.

  

### Student does

  

1. **Best‑of‑N (Loop B)**

  

- for each prompt: sample N, score, pick best

- compare N=1 baseline vs N>1

- explain why this is already “policy improvement”

  

2. **RLHF vs RLVR (conceptual, clean)**

**Critical mental guardrail:**

A reward model (RLHF) is **a learned measurement device**, not ground truth.

Treat it like any other metric: it can be biased, hackable, and drift—so you validate it with holdouts, audits, and adversarial tests the same way you validate a verifier.

**Industry terminology sanity check**

- RLVR-style setups often use a **verifier** (rule-based, deterministic reward).

- RLHF-style setups often use a **reward model** (learned proxy reward from preferences).

**Lifecycle difference:**

- Verifier: version it, test it, harden it like production code.

- Reward model: validate it like a model (holdouts, drift checks, adversarial probes).

In this course we use a verifier to keep rewards inspectable and non-drifting.

  

- SFT: teach base behavior/format

- RL step: optimize reward (preferences or verification)

- constraints: KL, penalties, safety rules

- diagram: rollouts → scoring → update

3. **Minimal pseudocode**

~15 lines showing the loop (conceptual).

  

### Optional (only if it won’t derail focus)

  

Run the teacher’s toy “policy improvement” demo in two modes:

- **selection-only** (Best-of-N on frozen rollouts), and

- **learning** (the tiny bandit learner shifting probabilities).

  

No LLM training runs required.

  

### Final deliverable

  

**“RL for LLMs: Mental Map v2”** (2 pages max)

  

- definitions in their own words

- the 3 loops diagram

- common failure modes + how to detect them

- what changes moving from verification to human preferences

  

### Good habits introduced

  

- Don’t confuse metric improvement with system improvement.

- Prefer small controlled experiments over big confusing ones.

  

# Transfer: how this maps to “real” RL-for-LLMs work later

  

This course stays single-turn on purpose. Here’s what changes later — and what doesn’t.

  

**What stays the same**

- The 3 loops (Eval / Selection / Learning)

- Reward/spec as production code

- Debugging taxonomy (model vs spec vs data vs measurement)

- Reward hacking mindset (assume adversarial optimization)

  

**What gets more complex later**

- Multi-turn agents: trajectory includes tool calls + environment feedback

- Preference-based reward: reward comes from a learned model (and can drift)

- Non-determinism: stochastic environments, noisy graders, human variance

- Credit assignment: longer horizons amplify instability and reward hacking

  

If you can keep your measurement honest and your reward spec hard-to-cheat,

the “real stack” becomes a scaling problem, not a conceptual one.

  

---

  

# Materials (teacher-provided)

  

## 0) Teacher code standards

  

Teacher-provided code should be:

- small files with clear names (no clever architecture)

- typed where it helps readability (lightweight annotations)

- pure functions where possible (especially `score`)

- deterministic by default (no hidden randomness)

- explicit about failure modes (never crash; always return explainable `details`)

- instrumented for learning (logs that tell a coherent story, not spam)

- dependency-minimal (prefer stdlib; add a dependency only if it buys clarity)

  
  

## 1) Policy gradient demo script

  

Keep your slow-motion script; it’s doing real pedagogical work.

  

**Restored micro-tweaks (from your original):**

  

- add a flag to print a one-line summary each step even in fast mode

- add a one-time “advantage sign interpretation” explanation line at the top

  

## 2) Minimal course harness starter repo (friction killer)

  

Teacher provides a working, batteries-included harness that runs on laptop CPU:

- `course/score.py` (scorer contract + implementation)

- `course/eval.py` (runs scoring; saves inspectable artifacts)

- `course/selection_demo.py` (Best-of-N)

- `course/bandit_train.py` (toy learning demo)

  

Student focuses on reading, reasoning, and diagnosing—no wiring.

  
  

## 3) Golden cases pack (to avoid tool/API friction)

  

Teacher provides a small set of:

  

* prompts

* model completions (some correct, some wrong, some adversarial)

* expected score outcomes

  

This lets the student validate the scorer deterministically without needing model access or long eval runs.

  
  

## 4) Frozen rollouts pack (keeps Milestone 0 painless)

  

A small set of pre-generated outputs so evaluation + debugging starts immediately.

  

---

  

# Minimal reading (only what’s necessary)

  

Core:

- The course notes + the scorer contract + the measurement rules (everything local/offline)

- The harness README: “how to run evals, where outputs land, how to inspect failures”

  

  

Optional:

  

- short policy gradient intuition (no proofs)

- one RLHF/RLVR overview (pipeline-level)

  

---

  

# Explicit non-goals (scope stays minimal)

  

- PPO math details, actor‑critic derivations, GAE

- distributed RL infra / production training stacks

- multi‑turn agent architectures and tool use

- training reward models from preference data (conceptual only)

  

---