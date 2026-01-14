# Introduction

---

## Why This Course?

This course teaches you the thinking that prevents expensive mistakes. Most RL tutorials focus on algorithms, but most real-world failures come from broken measurement—you changed the eval instead of improving the model, your reward got hacked, your results aren't reproducible. Here you'll learn to treat reward as a specification (test it like code), separate real improvement from self-deception (Locked Room Rule), and understand *why* RL works (policy gradients, credit assignment, KL constraints) through small experiments you deliberately break and fix. You won't leave with a PPO implementation, but you'll leave able to read production RLHF systems critically and ask the questions that matter: "Is this metric trustworthy? Did the policy actually improve, or did we just pick better samples? What happens when optimization pressure finds the loopholes?" That's the skill gap between running a framework and understanding what it's doing.

---

## Fundamental Concepts

A Large Language Model (LLM) functions as a probabilistic system: when provided with a prompt $x$, it does not generate a singular, deterministic answer. Rather, it defines a **conditional probability distribution $P(y | x)$** over possible completions $y$, where each token in the completion is sampled sequentially according to the model's learned distribution.

> **Math translation:** $P(y | x)$ is read as "the probability of $y$ given $x$". It captures the idea that the output $y$ depends entirely on the input context $x$.

Reinforcement Learning for LLMs (RL-for-LLMs) is employed in scenarios where it is not feasible to provide the model with a labeled "correct output string," yet it remains possible to furnish **feedback** regarding the quality of the output it has generated.

This feedback may take various forms:

* Binary correctness indicators ("Correct / incorrect")
* Comparative preferences ("This is preferred over that")
* Rule compliance verification ("This follows the rules")
* Specification conformance ("This matches a spec")
* Safety assessments ("This is safe")

In this course, the feedback mechanism is implemented as **a verifier** (also called a **scorer**). The verifier is a deterministic program that serves a dual purpose:

1. **A Measurement Instrument (The "Checker")**: It provides a truthful, objective report on whether the model succeeded. It is your "red pen" for evaluation.
2. **A Shaping Mechanism (The "Gate")**: It defines the environment's rewards. Because RL optimizes for reward, the verifier acts as a gate that "shapes" the model's behavior toward the desired goal.

For the anchor project, the verifier evaluates whether the model's completion contains a correctly formatted final integer that corresponds to the expected answer.

The rationale for this simplicity is pedagogical: deterministic feedback allows students to observe the underlying mechanism clearly, without the confounding complexity introduced by a learned reward model.

The foundational principle that guides this course is:

**Reward and evaluation mechanisms function as measurement instruments.**
If your measurement instrument lacks precision, your reinforcement learning system will confidently optimize toward meaningless objectives.

Consequently, this course is structured around the development of a reliable measurement instrument first, followed by its application in three distinct operational modes.

---

## Practical setup for assignments

Before you start the project levels, keep these concrete details in mind:

- **Python version:** Use Python 3.10+ (if `python3` is older, use `python3.11` or `poetry run`).
- **How to run:** Scripts are invoked as modules (e.g., `python -m course.eval`). With Poetry: `poetry run python -m course.eval`.
- **Where outputs go:** Runs are written under `runs/<name>/` and typically include `manifest.json`, `results.jsonl`, `summary.json`, and `summary.md`.
- **Where you edit:** Student surfaces live in `course/assignments/`, reflections in `notes/`, and tests in `tests/`.
- **Determinism expectation:** Runs should be reproducible; if in doubt, hash `results.jsonl` across repeated runs.
- **Optional plots:** `--plot` in the KL demo requires matplotlib; if missing, it will skip plotting.
- **Optional Groq sampling:** Put `GROQ_API_KEY` in a `.env` file and use `python -m course.rollout_sample` to generate real-model rollouts.

## The Environment: A Controlled System of Prompts and Verification

Classical reinforcement learning theory describes an "environment" as a Markov Decision Process (MDP) that provides observations and rewards. In LLM reinforcement learning, particularly for single-turn tasks, the "environment" typically comprises the following components:

1. A **prompt distribution** $p(x)$ (your dataset)
2. A **policy** $\pi_\theta(y | x)$ (the LLM parameterized by $\theta$ that samples completions)
3. A **reward function** $R(x, y)$ (verifier or reward model) that maps (prompt, output) pairs to scalar rewards
4. Optional constraints (format rules, safety rules, etc.)
> **Math translation:**
> * $p(x)$: How often different prompts appear in the real world (or your dataset).
> * $\pi_\theta(y | x)$: The AI model itself. $\theta$ represents the billions of parameters (weights) inside the neural network.
> * $R(x, y)$: The rulebook. It takes the prompt and the answer, reads them, and spits out a score.

This course makes these components explicit and rigorously defined, creating a simplified MDP where:
- State space: the prompt $x$
- Action space: the space of possible completions $y$
- Transition dynamics: deterministic (single-turn)
- Reward: $R(x, y) \in \{0, 1\}$ for this binary verification task

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

Formally, this implements $R: (X, Y) \to \mathbb{R}$, where:
* `example` $\in X$ represents one dataset item (prompt + expected answer + identifier)
* `completion` $\in Y$ represents the model's generated output
* `reward` $\in \{0, 1\}$ is the scalar reward signal (binary in this framework)
* `details` contains diagnostic information (parse success status, error codes, extracted values, etc.)

> **Math translation:** $R: (X, Y) \to \mathbb{R}$ means "R is a function that maps a pair (from set X and set Y) to a Real number." In our case, the real number is just 0 or 1.

This contract serves as the "measurement instrument." All other operations constitute different methods of **utilizing** this instrument to compute objectives, metrics, or learning signals.

---

## Conceptual Framework: Three Distinct Operational Loops

All three loops execute variations of the same basic process:

**prompt → completion(s) → score → numeric result(s)**

This similarity can lead to conceptual confusion. This course emphasizes the distinctions until they become intuitive:

### Loop A — Evaluation (Measurement)

**Objective:** "What is the current state of system performance?"

You select a set of prompts $\{x_1, x_2, \dots, x_n\}$, obtain completions $y_i \sim \pi_\theta(\cdot|x_i)$ (often frozen or previously recorded initially), execute the `score` function, and aggregate the results.

This loop involves no optimization, no learning, and no sophisticated techniques. It is purely measurement.

**Formally, you compute empirical statistics:**

* **Mean reward**: $\bar{R} = (1/N) \sum_{i=1}^N R(x_i, y_i)$
* **Pass rate** (for binary rewards): $\hat{P}_{\text{pass}} = (1/N) \sum_{i=1}^N \mathbb{1}[R(x_i, y_i) = 1]$

> **Math translation:**
> * $\Sigma$ (Sigma) means "sum up all the values".
> * $\mathbb{1}[\dots]$ is the **Indicator Function**. It equals $1$ if the condition inside is true, and $0$ if false.
> * So, $\hat{P}_{\text{pass}}$ is just counting how many times the model got it right, divided by the total number of attempts $N$.
* **Failure mode distribution**: Group by error type and compute proportions

Practical outputs include:

* Pass rate or mean reward (empirical estimates of $\mathbb{E}[R]$)
* Distribution of failure modes (incorrect answers versus formatting failures versus parse errors)
* Concrete examples available for inspection

This loop teaches what is arguably the most critical skill in reinforcement learning that does not involve mathematics:
**Systematically categorizing failure modes rather than making intuitive assessments.**

If you cannot articulate the reason for failure, you cannot improve systematically.

---

### Loop B — Selection (Optimization Without Model Modification)

**Objective:** "Can performance be enhanced through increased computational resources at inference time?"

Rather than generating a single completion per prompt, you generate **N samples** $\{y_1, y_2, \dots, y_n\} \sim \pi_\theta(\cdot|x)$ from the *same* model for each prompt, score all samples, and select the highest-scoring one.

**Formally, the selection operation is:**

**$y^* = \text{argmax}_{y \in \{y_1,\dots,y_n\}} R(x, y)$**

> **Math translation:** $\text{argmax}$ asks "which argument (input) produces the maximum output?"
> In plain English: We generate $n$ options. We score them all. We pick the one ($y^*$) that got the highest score. We don't change the model; we just cherry-pick its best work.

This approach is known as Best-of-N sampling (along with related techniques: reranking, rejection sampling, verifier-guided search).

Key concept:

* The model's capabilities (the distribution $\pi_\theta$) do not improve.
* You are simply exploring a larger portion of what it already knows how to generate.
* This is inference-time compute scaling, not training-time improvement.

**Metrics formalized:**

* **pass@1** = $\mathbb{P}[R(x, y) = 1 \mid y \sim \pi_\theta(\cdot|x)]$ — probability a single sample succeeds
* **pass@k** = $\mathbb{P}[\max\{R(x, y_1), \dots, R(x, y_k)\} = 1 \mid y_1,\dots,y_k \sim \pi_\theta(\cdot|x)]$ — probability at least one of k samples succeeds

> **Math translation:** $ \sim $ means "sampled from". $\pi_\theta(\cdot|x)$ is the model's distribution.
> pass@k asks: "If I let the model try $k$ times, what is the probability that *at least one* of those tries is correct?"

Selection improves pass@N. It may have minimal impact on pass@1, since $\pi_\theta$ itself is unchanged.

Practical considerations:

* Selection is frequently the most expedient method for deploying performance improvements safely.
* However, it incurs costs in runtime computation and latency.
* Additionally, it may obscure the fact that the base policy remains weak.

---

### Loop C — Learning (Reinforcement Learning Training)

**Objective:** "Can pass@1 performance be improved by modifying the policy itself?"

This loop executes the same sampling and scoring procedures, but adds one critical step:

**Update the model parameters $\theta$ such that the policy becomes more likely to generate high-reward completions in future iterations.**

This is where reinforcement learning "actually occurs."

Even when the reward is a simple binary 0/1 terminal value, the update procedure must assign credit to the sequence of token-level decisions that produced the completion.

The fundamental conceptual transformation is:

* Treat the model as a **policy** $\pi_\theta(y | x)$ parameterized by weights $\theta$
* Define the objective as **expected reward** under that policy: $J(\theta) = \mathbb{E}_{x \sim p(x), y \sim \pi_\theta(\cdot|x)}[R(x, y)]$

> **Math translation:**
> * $\mathbb{E}$ stands for **Expectation** (the mathematical average).
> * The goal $J(\theta)$ is simply "the average score we expect the model to get."
> * $\nabla_\theta$ (Nabla or Gradient) points in the direction of steepest ascent. We want to climb the hill of higher rewards.
* Employ policy gradient methods to maximize $J(\theta)$ by adjusting $\theta$ in the direction $\nabla_\theta J(\theta)$

---

## The Underlying Mechanism: Reinforcement of Successful Actions

In the course framework, the learning mechanism is initially demonstrated using a simple bandit problem rather than a full LLM. This is not an evasion of complexity; it is a pedagogical strategy for clarity.

### REINFORCE Algorithm in Summary

The REINFORCE algorithm (Williams, 1992) is a Monte Carlo policy gradient method. When you sample an action and observe a reward:

* If the reward was **better than expected**, increase the probability of the sampled action.
* If the reward was **worse than expected**, decrease the probability of the sampled action.

The policy gradient updates parameters according to:

**$\nabla_\theta J(\theta) \approx (R - b) \cdot \nabla_\theta \log \pi_\theta(y | x)$**

> **Math translation:** This formula says: "Take the gradient of the log-probability of the action we just took. Scale it by how good the result was ($R-b$). If the result was better than usual (positive), push the gradient to make this action *more* likely. If worse (negative), push it to make it *less* likely."

where:
* **$R$** is the observed reward for completion $y$
* **$b$** is a baseline (typically estimated as the running mean: $b \leftarrow (1-\alpha)b + \alpha R$)
* **$A = R - b$** is the advantage function
* The advantage determines the *direction* and *magnitude* of the probability adjustment

**Computing the baseline:**
In practice, the baseline is often computed as:
- **Batch mean**: $b = (1/N) \sum_i R_i$ over current batch
- **Exponential moving average**: $b_{t+1} = (1-\alpha) b_t + \alpha R_t$ with decay $\alpha \in (0,1)$
- **Value function**: In actor-critic methods, $b = V_\phi(x)$ is a learned neural network

The importance of baselines:

* Raw rewards contain substantial noise; baselines center the learning signal around zero.
* Baselines reduce gradient variance without introducing bias (since $\mathbb{E}[b \cdot \nabla_\theta \log \pi_\theta] = 0$).
* Lower variance enables more stable learning with fewer samples.

Application to LLMs:

* An LLM completion $y = (y_1, y_2, \dots, y_t)$ consists of a sequential chain of token-level actions.
* The policy factorizes as $\pi_\theta(y | x) = \prod_i \pi_\theta(y_i | x, y_{1:i-1})$
* The gradient $\nabla_\theta \log \pi_\theta(y | x) = \sum_i \nabla_\theta \log \pi_\theta(y_i | x, y_{1:i-1})$ pushes up the log-probability of each token in high-reward trajectories

Therefore, conceptually, Loop C executes:
**sample completions $y \sim \pi_\theta(\cdot|x)$ → score $R(x,y)$ → compute advantage $A = R - b$ → update $\theta \leftarrow \theta + \alpha \cdot A \cdot \nabla_\theta \log \pi_\theta(y|x)$**

---

## Regularization: The Role of KL Divergence in RLHF Systems

If you optimize solely for reward, you invite a problematic phenomenon:

**Goodhart's Law:** When a measure becomes a target, it ceases to be a good measure.

Optimizers are highly effective at identifying specification loopholes. If your reward specification is imperfect (which it invariably is), the policy will exploit those imperfections.

One of the most common stabilization techniques is a **KL divergence penalty**:

* You maintain a reference policy $\pi_{\text{ref}}$ (typically the original supervised fine-tuned model).
* You penalize the learned policy $\pi_\theta$ for diverging excessively from the reference.

The constrained objective becomes:

**$J_{KL}(\theta) = \mathbb{E}_{x,y}[R(x, y)] - \beta \cdot \mathbb{E}_x[D_{KL}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x))]$**

> **Math translation:**
> * $J_{KL}$ is our new, safer goal.
> * term 1 ($\mathbb{E}[R]$): "Get high rewards."
> * term 2 ($\beta \cdot D_{KL}$): "Minus a penalty for being weird."
> * $D_{KL}$ measures the "distance" between the new model behavior and the old one. We want to maximize reward, but minimize distance.

where:
* **$D_{KL}$** is the Kullback-Leibler divergence: $D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)]$
* **$\beta$** is the KL penalty coefficient controlling the strength of regularization

Intuition:

* Reward term says "move in this direction to maximize R."
* KL penalty says "but don't deviate too far from $\pi_{\text{ref}}$."

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

**Formally:** The LLM operates over a discrete vocabulary $V$ (typically $|V| \approx 30k-100k$ tokens). A completion is a sequence $y = (y_1, \dots, y_T)$ where each $y_i \in V$, and the policy is an autoregressive distribution:

**$\pi_\theta(y | x) = \prod_{i=1}^T \pi_\theta(y_i | x, y_{1:i-1})$**

where each token probability $\pi_\theta(y_i | \text{context})$ is computed via a softmax over $V$:

**$\pi_\theta(y_i = v | \text{context}) = \frac{\exp(f_\theta(v, \text{context}))}{\sum_{v' \in V} \exp(f_\theta(v', \text{context}))}$**

> **Math translation:** This is the **Softmax** function.
> * $f_\theta$ outputs raw numbers (logits) which can be negative or huge.
> * $\exp$ (exponential) makes everything positive.
> * Dividing by the sum $\sum$ guarantees that all the numbers add up to exactly 1.0, making them valid probabilities.

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

* "The scorer implements the reward function $R(x, y)$; if $R$ changes, the optimum policy $\pi^*$ changes."
* "Loop A computes empirical statistics $\mathbb{E}[R]$, Loop B performs inference-time optimization $\max_{y \in \text{samples}} R(x,y)$, Loop C performs learning-time optimization $\nabla_\theta \mathbb{E}[R]$."
* "Selection improves pass@N (existence of good sample) without necessarily improving pass@1 (quality of typical sample)."
* "REINFORCE updates log-probabilities using advantage-weighted gradients: $\Delta \theta \propto A \cdot \nabla_\theta \log \pi_\theta$."
* "KL regularization $D_{KL}(\pi_\theta || \pi_{\text{ref}})$ constrains the policy to a trust region around the reference."
* "Tokenization makes the action space discrete; formatting differences like whitespace affect $\pi_\theta(y|x)$ non-trivially."

This conceptual and mathematical framework is sufficient to approach the code as implementation detail rather than as an incomprehensible maze.

You have essentially constructed a controlled laboratory environment where reinforcement learning is not an arcane procedure—it is a rigorous experiment with:
- A calibrated measurement instrument ($R: X \times Y \to \mathbb{R}$)
- A systematic optimization objective ($J(\theta) = \mathbb{E}[R]$)
- A precise definition of success (empirical pass rate on held-out data)

---

## From Theory to Implementation: The Course Scripts

The preceding sections establish the formal framework. This section maps those concepts to the concrete scripts you will execute. All three scripts utilize the same scorer—a function that verifies whether the model's output contains the exact format `Final: <int>` with the correct integer.

### Concept Mapping Table

| Formal Concept | Implementation Term | Location in Codebase |
|----------------|---------------------|---------------------|
| **Policy** $\pi_\theta(y \mid x)$ | "The model" | All scripts invoke the model to generate outputs |
| **Reward function** $R(x, y)$ | "The scorer" | `score.py` — central contract implementation |
| **Objective** $J(\theta) = \mathbb{E}[R]$ | "Performance" or "pass rate" | `eval.py` computes this empirical estimate |
| **Empirical mean reward** $\bar{R}$ | "Pass rate" or "mean reward" | `eval.py` summary statistics |
| **Argmax selection** $y^* = \text{argmax } R$ | "Pick the best one" | `selection_demo.py` selects highest-scoring sample |
| **Policy gradient update** $\nabla_\theta J$ | "Update the model" | `bandit_train.py` modifies model weights |
| **Advantage** $A = R - b$ | "Reward minus baseline" | `bandit_train.py` learning signal |
| **KL divergence penalty** $D_{KL}$ | "Don't drift too far" | `kl_tradeoff_demo.py` demonstrates this constraint |

### Script 1: Evaluation (`eval.py`) — Loop A

**Purpose:** Determine the current state of system performance.

**Formal operation:** Compute the empirical mean reward $\bar{R} = (1/N) \sum_{i=1}^N R(x_i, y_i)$.

**What occurs:**
1. Execute the model on a set of prompts
2. Score every generated answer
3. Aggregate results into pass rate and failure mode distribution

**Output example:**
```
Pass rate: 23/100 = 23%
Failures:
  - Wrong answer: 65
  - Bad format: 12
```

This script performs pure measurement. No optimization. No training. The pass rate reported is your empirical estimate of $\mathbb{E}[R]$.

---

### Script 2: Selection (`selection_demo.py`) — Loop B

**Purpose:** Enhance performance through increased inference-time computation without modifying the model.

**Formal operation:** Compute $y^* = \text{argmax}_{y \in \{y_1,\dots,y_n\}} R(x, y)$ over $N$ samples.

**What occurs:**
1. Generate $N$ completions for the same prompt
2. Score all completions
3. Select the highest-scoring completion

**Key insight:** If the model succeeds with probability $p = 0.23$ per sample:
- 1 sample → 23% success probability
- 10 samples → $1 - (1 - 0.23)^{10} \approx 93\%$ probability that at least one succeeds

The model's distribution $\pi_\theta$ remains unchanged. You are exploring a larger portion of its existing capability space.

**Relevant metrics:**
* **pass@1** — probability the first sample succeeds
* **pass@N** — probability at least one of $N$ samples succeeds

Selection improves pass@N substantially. It has minimal effect on pass@1, since $\pi_\theta$ itself is unchanged.

---

### Script 3: Learning (`bandit_train.py`) — Loop C

**Purpose:** Improve pass@1 by modifying the policy itself.

**Formal operation:** Update $\theta \leftarrow \theta + \alpha \cdot A \cdot \nabla_\theta \log \pi_\theta(y|x)$ where $A = R - b$.

**What occurs:**
1. Sample a completion from the current policy
2. Score it to obtain reward $R$
3. Compute advantage: $A = R - b$ (reward relative to baseline expectation)
4. If $A > 0$: increase probability of that completion
5. If $A < 0$: decrease probability of that completion

This is the REINFORCE algorithm in action. The model's parameters $\theta$ are modified to make high-reward completions more probable.

---

### The KL Tradeoff Demonstration (`kl_tradeoff_demo.py`)

**Purpose:** Visualize the reward-divergence tradeoff formalized as $J_{KL}(\theta) = \mathbb{E}[R] - \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$.

In a simplified discrete action space, this demonstration allows direct observation of the Pareto frontier between "higher reward" and "lower divergence from reference policy."

---

## The Measurement Instrument in Practice

The scorer constitutes the implementation of your reward function $R(x, y)$. Its behavior determines what the optimization process targets.

**Example of specification strictness:**

Consider a relaxed specification: "reward any string containing the correct number."

A completion such as:
```
Final: 42
But also 43 might work?
Actually I'm not sure
The answer could be 42
```

This string contains "42" → $R = 1$ under the relaxed specification.

However, this output is not a valid solution. The model has exploited a loophole in the reward specification.

**The course enforces strict specifications:**

Reward is 1 **only if** the output is exactly:
```
Final: 42
```

No additional lines. No extraneous text. One integer in the prescribed format.

This strictness is not pedantic—it is **environment design**. A loose specification permits reward hacking. A strict specification forces the model to learn the intended behavior.

---

## Distinguishing the Three Loops

All three loops share superficial similarity:
- Accept prompts
- Generate completions
- Score outputs
- Produce numeric results

This similarity obscures fundamental differences:

| Loop | Operation | What Changes |
|------|-----------|--------------|
| **A: Evaluation** | Measure current performance | Nothing |
| **B: Selection** | Select best of $N$ samples | Which output is deployed |
| **C: Learning** | Train via policy gradient | The model parameters $\theta$ |

If you cannot identify which loop changed between two experimental runs, your comparison is invalid.

---

## Verification Checklist Before Examining Code

Articulate the following statements until they are intuitive:

1. **"The scorer is my measurement instrument. If I modify it, I have changed what I am measuring."**

2. **"Loop A measures $\mathbb{E}[R]$. Loop B computes $\max R$ over samples. Loop C optimizes $\nabla_\theta \mathbb{E}[R]$."**

3. **"Selection improves pass@N. Training improves pass@1."**

4. **"Advantage $A = R - b$ is the learning signal. Positive advantage increases probability; negative advantage decreases it."**

5. **"KL penalty $D_{KL}(\pi_\theta || \pi_{\text{ref}})$ prevents reward hacking by constraining policy drift."**

6. **"Text has tokens. Minor formatting differences correspond to substantial probability differences in token space."**

If these statements are clear, you possess the conceptual framework to approach the implementation as engineering detail rather than as an opaque system.

The code implements these concepts. The mental model determines whether you understand what the code is doing.
