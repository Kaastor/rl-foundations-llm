# Introduction

---

## Fundamental Concepts

A Large Language Model (LLM) functions as a probabilistic system: when provided with a prompt x, it does not generate a singular, deterministic answer. Rather, it defines a **conditional probability distribution P(y | x)** over possible completions y, where each token in the completion is sampled sequentially according to the model's learned distribution.

Reinforcement Learning for LLMs (RL-for-LLMs) is employed in scenarios where it is not feasible to provide the model with a labeled "correct output string," yet it remains possible to furnish **feedback** regarding the quality of the output it has generated.

This feedback may take various forms:

* Binary correctness indicators ("Correct / incorrect")
* Comparative preferences ("This is preferred over that")
* Rule compliance verification ("This follows the rules")
* Specification conformance ("This matches a spec")
* Safety assessments ("This is safe")

In this course, the feedback mechanism has been deliberately designed to be deterministic and straightforward: **a verifier** evaluates whether the model's completion contains a correctly formatted final integer that corresponds to the expected answer.

The rationale for this simplicity is pedagogical: deterministic feedback allows students to observe the underlying mechanism clearly, without the confounding complexity introduced by a learned reward model.

The foundational principle that guides this course is:

**Reward and evaluation mechanisms function as measurement instruments.**
If your measurement instrument lacks precision, your reinforcement learning system will confidently optimize toward meaningless objectives.

Consequently, this course is structured around the development of a reliable measurement instrument first, followed by its application in three distinct operational modes.

---

## The Environment: A Controlled System of Prompts and Verification

Classical reinforcement learning theory describes an "environment" as a Markov Decision Process (MDP) that provides observations and rewards. In LLM reinforcement learning, particularly for single-turn tasks, the "environment" typically comprises the following components:

1. A **prompt distribution** p(x) (your dataset)
2. A **policy** πθ(y | x) (the LLM parameterized by θ that samples completions)
3. A **reward function** R(x, y) (verifier or reward model) that maps (prompt, output) pairs to scalar rewards
4. Optional constraints (format rules, safety rules, etc.)

This course makes these components explicit and rigorously defined, creating a simplified MDP where:
- State space: the prompt x
- Action space: the space of possible completions y
- Transition dynamics: deterministic (single-turn)
- Reward: R(x, y) ∈ {0, 1} for this binary verification task

### The Specification as Environmental Design

The format requirement "Output exactly one line: `Final: <int>`" is not merely a stylistic preference. It constitutes **environment design**.

The specification you enforce determines the behavioral outcomes:
- Rewarding "any string that contains the correct number somewhere" produces one pattern of behavior.
- Rewarding "exactly one line with a specific prefix and a strictly formatted integer" produces an entirely different pattern.

Therefore, the strictness of specifications is not pedantic; it represents a controlled experimental methodology.

---

## The Central Contract

All operations in this system revolve around a single interface that implements the reward function:

**`score(example, completion) -> {reward, details}`**

Formally, this implements R: (X, Y) → ℝ, where:
* `example` ∈ X represents one dataset item (prompt + expected answer + identifier)
* `completion` ∈ Y represents the model's generated output
* `reward` ∈ {0, 1} is the scalar reward signal (binary in this framework)
* `details` contains diagnostic information (parse success status, error codes, extracted values, etc.)

This contract serves as the "measurement instrument." All other operations constitute different methods of **utilizing** this instrument to compute objectives, metrics, or learning signals.

---

## Conceptual Framework: Three Distinct Operational Loops

All three loops execute variations of the same basic process:

**prompt → completion(s) → score → numeric result(s)**

This similarity can lead to conceptual confusion. This course emphasizes the distinctions until they become intuitive:

### Loop A — Evaluation (Measurement)

**Objective:** "What is the current state of system performance?"

You select a set of prompts {x₁, x₂, ..., xₙ}, obtain completions yᵢ ~ πθ(·|xᵢ) (often frozen or previously recorded initially), execute the `score` function, and aggregate the results.

This loop involves no optimization, no learning, and no sophisticated techniques. It is purely measurement.

**Formally, you compute empirical statistics:**

* **Mean reward**: R̄ = (1/N) Σᵢ₌₁ᴺ R(xᵢ, yᵢ)
* **Pass rate** (for binary rewards): P̂_pass = (1/N) Σᵢ₌₁ᴺ 𝟙[R(xᵢ, yᵢ) = 1]
* **Failure mode distribution**: Group by error type and compute proportions

Practical outputs include:

* Pass rate or mean reward (empirical estimates of 𝔼[R])
* Distribution of failure modes (incorrect answers versus formatting failures versus parse errors)
* Concrete examples available for inspection

This loop teaches what is arguably the most critical skill in reinforcement learning that does not involve mathematics:
**Systematically categorizing failure modes rather than making intuitive assessments.**

If you cannot articulate the reason for failure, you cannot improve systematically.

---

### Loop B — Selection (Optimization Without Model Modification)

**Objective:** "Can performance be enhanced through increased computational resources at inference time?"

Rather than generating a single completion per prompt, you generate **N samples** {y₁, y₂, ..., yₙ} ~ πθ(·|x) from the *same* model for each prompt, score all samples, and select the highest-scoring one.

**Formally, the selection operation is:**

**y* = argmax_{y ∈ {y₁,...,yₙ}} R(x, y)**

This approach is known as Best-of-N sampling (along with related techniques: reranking, rejection sampling, verifier-guided search).

Key concept:

* The model's capabilities (the distribution πθ) do not improve.
* You are simply exploring a larger portion of what it already knows how to generate.
* This is inference-time compute scaling, not training-time improvement.

**Metrics formalized:**

* **pass@1** = ℙ[R(x, y) = 1 | y ~ πθ(·|x)] — probability a single sample succeeds
* **pass@k** = ℙ[max{R(x, y₁), ..., R(x, yₖ)} = 1 | y₁,...,yₖ ~ πθ(·|x)] — probability at least one of k samples succeeds

Selection improves pass@N. It may have minimal impact on pass@1, since πθ itself is unchanged.

Practical considerations:

* Selection is frequently the most expedient method for deploying performance improvements safely.
* However, it incurs costs in runtime computation and latency.
* Additionally, it may obscure the fact that the base policy remains weak.

---

### Loop C — Learning (Reinforcement Learning Training)

**Objective:** "Can pass@1 performance be improved by modifying the policy itself?"

This loop executes the same sampling and scoring procedures, but adds one critical step:

**Update the model parameters θ such that the policy becomes more likely to generate high-reward completions in future iterations.**

This is where reinforcement learning "actually occurs."

Even when the reward is a simple binary 0/1 terminal value, the update procedure must assign credit to the sequence of token-level decisions that produced the completion.

The fundamental conceptual transformation is:

* Treat the model as a **policy** πθ(y | x) parameterized by weights θ
* Define the objective as **expected reward** under that policy: J(θ) = 𝔼_{x~p(x), y~πθ(·|x)}[R(x, y)]
* Employ policy gradient methods to maximize J(θ) by adjusting θ in the direction ∇θJ(θ)

---

## The Underlying Mechanism: Reinforcement of Successful Actions

In the course framework, the learning mechanism is initially demonstrated using a simple bandit problem rather than a full LLM. This is not an evasion of complexity; it is a pedagogical strategy for clarity.

### REINFORCE Algorithm in Summary

The REINFORCE algorithm (Williams, 1992) is a Monte Carlo policy gradient method. When you sample an action and observe a reward:

* If the reward was **better than expected**, increase the probability of the sampled action.
* If the reward was **worse than expected**, decrease the probability of the sampled action.

The policy gradient updates parameters according to:

**∇θJ(θ) ≈ (R − b) · ∇θ log πθ(y | x)**

where:
* **R** is the observed reward for completion y
* **b** is a baseline (typically estimated as the running mean: b ← (1−α)b + αR)
* **A = R − b** is the advantage function
* The advantage determines the *direction* and *magnitude* of the probability adjustment

**Computing the baseline:**
In practice, the baseline is often computed as:
- **Batch mean**: b = (1/N) Σᵢ Rᵢ over current batch
- **Exponential moving average**: b_{t+1} = (1−α) b_t + α R_t with decay α ∈ (0,1)
- **Value function**: In actor-critic methods, b = V_φ(x) is a learned neural network

The importance of baselines:

* Raw rewards contain substantial noise; baselines center the learning signal around zero.
* Baselines reduce gradient variance without introducing bias (since 𝔼[b · ∇θ log πθ] = 0).
* Lower variance enables more stable learning with fewer samples.

Application to LLMs:

* An LLM completion y = (y₁, y₂, ..., yₜ) consists of a sequential chain of token-level actions.
* The policy factorizes as πθ(y | x) = ∏ᵢ πθ(yᵢ | x, y₁:ᵢ₋₁)
* The gradient ∇θ log πθ(y | x) = Σᵢ ∇θ log πθ(yᵢ | x, y₁:ᵢ₋₁) pushes up the log-probability of each token in high-reward trajectories

Therefore, conceptually, Loop C executes:
**sample completions y ~ πθ(·|x) → score R(x,y) → compute advantage A = R − b → update θ ← θ + α·A·∇θ log πθ(y|x)**

---

## Regularization: The Role of KL Divergence in RLHF Systems

If you optimize solely for reward, you invite a problematic phenomenon:

**Goodhart's Law:** When a measure becomes a target, it ceases to be a good measure.

Optimizers are highly effective at identifying specification loopholes. If your reward specification is imperfect (which it invariably is), the policy will exploit those imperfections.

One of the most common stabilization techniques is a **KL divergence penalty**:

* You maintain a reference policy πref (typically the original supervised fine-tuned model).
* You penalize the learned policy πθ for diverging excessively from the reference.

The constrained objective becomes:

**J_KL(θ) = 𝔼_{x,y}[R(x, y)] − β · 𝔼_x[D_KL(πθ(·|x) || πref(·|x))]**

where:
* **D_KL** is the Kullback-Leibler divergence: D_KL(πθ || πref) = 𝔼_{y~πθ}[log πθ(y|x) − log πref(y|x)]
* **β** is the KL penalty coefficient controlling the strength of regularization

Intuition:

* Reward term says "move in this direction to maximize R."
* KL penalty says "but don't deviate too far from πref."

This serves multiple purposes beyond "safety":

* Preventing catastrophic behavioral shifts (mode collapse)
* Maintaining language fluency and coherence
* Reducing over-optimization on brittle reward specifications
* Improving learning stability through implicit trust regions

The course's KL demonstration provides a clear mental model: in a small discrete action space, you can directly observe the Pareto frontier between "higher reward" and "lower divergence from reference."

This tradeoff curve—essentially a constrained optimization problem—is fundamental to tuning real RL-for-LLMs systems (PPO, RLHF, DPO).

---

## The Critical Detail: Token-Space Versus Text-Space

A significant source of confusion in LLM reinforcement learning arises from the fact that we *reason* in text, but the policy operates in **tokens**.

**Formally:** The LLM operates over a discrete vocabulary V (typically |V| ≈ 30k-100k tokens). A completion is a sequence y = (y₁, ..., y_T) where each yᵢ ∈ V, and the policy is an autoregressive distribution:

**πθ(y | x) = ∏ᵢ₌₁ᵀ πθ(yᵢ | x, y₁:ᵢ₋₁)**

where each token probability πθ(yᵢ | context) is computed via a softmax over V:

**πθ(yᵢ = v | context) = exp(f_θ(v, context)) / Σ_{v'∈V} exp(f_θ(v', context))**

Two consequences that should be understood before examining implementation code:

1. **Small textual differences can correspond to large policy differences.**
   A space character, a newline, a different colon variant, or "Final:323" versus "Final: 323" may tokenize differently and receive substantially different probabilities.

2. **A strict reward specification becomes a behavioral shaping mechanism.**
   If reward is provided only for exactly `Final: <int>`, the training signal will heavily reinforce:

   * Formatting correctness
   * Appropriate stopping behavior (not generating extra lines)
   * Producing a valid integer without extraneous characters

This is why the course framework includes:

* Strict parsing requirements
* Explicit parse error categorization
* "Golden exploit" test cases

These are not peripheral concerns. They prevent your "measurement instrument" from being deceived by the very optimization process you are deploying.

---

## Measurement Discipline: The "Locked Room" Principle

This aspect may appear non-technical until you have experienced its violation firsthand:

If you modify the measurement instrument, you have changed what the numeric results signify.

Therefore, when comparing two experimental runs, you must maintain a controlled environment:

* Identical dataset split
* Identical prompts
* Identical scoring specification and version
* Identical sampling settings (if stochastic sampling is employed)

This practice prevents the most common pattern of self-deception in reinforcement learning research:

> "The metric improved! We have made progress!"
> (In reality, you modified the prompt, relaxed parsing constraints, and increased sample count.)

The course enforces this discipline through:

* Versioning the scorer as you would version software
* Storing run artifacts (results, summaries, manifests, checksums)
* Maintaining a small holdout set that is not used for tuning
* Executing "golden" tests to verify that the scorer has not inadvertently become permissive

This is what transforms your course work into a coherent *narrative* rather than a collection of disconnected techniques:
the central focus is not "PPO algorithm implementation" — it is **epistemic rigor**.

---

## System Architecture: A Unified Mental Model

Conceptualize the system as a single pipeline with three operational "modes":

```
           ┌──────────────┐
           │  Dataset D    │   (prompts + expected answers)
           └──────┬───────┘
                  │  sample prompt x
                  v
           ┌──────────────┐
           │ Policy πθ     │   (LLM distribution over completions)
           └──────┬───────┘
                  │  generate 1 or N completions
                  v
           ┌──────────────┐
           │ Completions   │   (strings / token sequences)
           └──────┬───────┘
                  │  score each completion
                  v
           ┌──────────────┐
           │ Scorer S      │   (verifier; your measurement instrument)
           └──────┬───────┘
                  │  reward + details
                  v
           ┌──────────────┐
           │ Results       │   (per-example records)
           └──────┬───────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      v           v           v
  Loop A       Loop B       Loop C
 (EVAL)     (SELECTION)   (LEARNING)
 summarize   choose best    update θ
 inspect     deploy output  (often with KL regularization)
```

This diagram represents the complete conceptual framework. Every component in your repository is either:

* One of these architectural elements,
* Or a safeguard to prevent self-deception regarding experimental changes.

---

## Translation to Real-World Practice

Although the course framework is minimal, it mirrors actual industrial workflows:

* **Development teams allocate substantial resources to the scorer and reward specification**, as this is where the majority of system failures originate.
* **Selection methods** (best-of-N, reranking, verifier-guided search) are frequently deployed before training-based approaches because they are more interpretable and reversible.
* **Training** is computationally expensive and can amplify specification errors into severe policy pathologies, so it is undertaken only after measurement is reliable.
* **KL regularization and constraints** are standard practice because unconstrained optimization frequently produces policies that exploit specification loopholes.
* **Inspection and artifact preservation** are critical because reinforcement learning failures rarely manifest as a single obvious error; they typically emerge as a cascade of small deviations.

---

## Expected Understanding After This Introduction

Before you examine implementation code, you should be able to articulate both conceptually and mathematically:

* "The scorer implements the reward function R(x, y); if R changes, the optimum policy π* changes."
* "Loop A computes empirical statistics 𝔼[R], Loop B performs inference-time optimization max_{y∈samples} R(x,y), Loop C performs learning-time optimization ∇θ 𝔼[R]."
* "Selection improves pass@N (existence of good sample) without necessarily improving pass@1 (quality of typical sample)."
* "REINFORCE updates log-probabilities using advantage-weighted gradients: Δθ ∝ A · ∇θ log πθ."
* "KL regularization D_KL(πθ || πref) constrains the policy to a trust region around the reference."
* "Tokenization makes the action space discrete; formatting differences like whitespace affect πθ(y|x) non-trivially."

This conceptual and mathematical framework is sufficient to approach the code as implementation detail rather than as an incomprehensible maze.

You have essentially constructed a controlled laboratory environment where reinforcement learning is not an arcane procedure—it is a rigorous experiment with:
- A calibrated measurement instrument (R: X × Y → ℝ)
- A systematic optimization objective (J(θ) = 𝔼[R])
- A precise definition of success (empirical pass rate on held-out data)
