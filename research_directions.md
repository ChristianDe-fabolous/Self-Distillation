# Research Directions: SDFT Extensions

## Primary Direction: KL Divergence as a Transfer Saliency Map

### Core Idea

After SDFT training, compare two models on identical inference inputs:

- **Student**: base model (no SDFT training)
- **Teacher**: SDFT-trained model (has internalized demonstrations)

The per-token KL divergence between their output distributions is a map of **what SDFT training actually changed and where in the reasoning process**. High KL at token position T means training specifically redirected the model's computation at that point.

### What the KL Map Reveals

By inspecting which token positions carry high KL across many examples, you can determine what kind of thing SDFT internalizes:

| High KL at | Interpretation |
|---|---|
| Reasoning transition tokens ("therefore", "because", "thus") | SDFT learned a reasoning structure |
| Domain-specific content tokens (facts, numbers, entities) | SDFT learned domain knowledge |
| Formatting / structural tokens | SDFT learned output style |

This gives a mechanistic account of what distillation actually does — not just whether it works, but where in generation it takes effect.

### Transferability Experiment

**Setup:**
1. Train SDFT on domain A (e.g., tool use)
2. Run both student and teacher on domain B (e.g., science problems)
3. Compute per-token KL at each position across many domain B examples

**What it measures:**
- High KL in domain B → training on domain A transferred here
- Low KL in domain B → model reasons identically to base model here

**Key property:** Transfer is measured without any labeled data in domain B. The KL between student and teacher is the transfer signal itself, not accuracy on a benchmark.

**Expected finding:** If SDFT transfers a reasoning structure rather than domain-specific knowledge, you would see high KL at reasoning step tokens (transition words, logical connectives) but low KL at content tokens (facts specific to domain B). This would demonstrate that SDFT generalizes a *way of reasoning*, not just what to reason about.

### Why This Is Novel

Most transfer learning research measures transfer via accuracy on a target domain benchmark. This approach measures it via distributional divergence between trained and untrained model, giving a finer-grained picture of *what* transferred and *where* in the generation process — not just *whether* it worked. The SDFT framework is uniquely suited for this because it already provides a principled student/teacher split with a clear semantic difference: one model has internalized demonstrations, the other has not.

### Experiment Steps

1. Train a SDFT model on the existing tool-use dataset
2. Take the base model as student, SDFT-trained model as teacher
3. Run both on science eval questions (out-of-domain), collect full logit distributions
4. Compute per-token KL for each position across all eval examples
5. Aggregate KL by token type (content vs. reasoning transition vs. structural)
6. Compare cross-domain accuracy: SDFT-trained vs. base model
7. Check whether positions with high KL correlate with positions where the answer improves

---

## Secondary Direction: KL-Based Self-Checking

### Core Idea

Instead of majority voting over multiple rollouts (self-consistency), use KL divergence between rollouts as a selection and weighting signal. The output with minimum average KL to all others is the distributional centroid — the most consistent output across the model's own sampling distribution.

### Why It Differs from Majority Voting

- Majority voting operates on the final answer token only
- KL comparison operates on the full token distribution across the entire sequence
- Two outputs can agree on the final answer but disagree on the reasoning — KL catches this, voting does not
- Outputs can be weighted by their KL distance to the consensus rather than binary selected

### Mid-Rollout Checking

Rather than comparing complete outputs, KL can be computed at each reasoning step. High KL at an intermediate step indicates a distributional branching point — a position where the model is uncertain. At these positions, multiple continuations can be sampled and the lowest-KL branch selected, catching errors before the model over-commits to a wrong reasoning path.

This is related to process reward models (PRMs) but requires no trained verifier — the signal comes from the model's own distributional uncertainty.

### The Critical Assumption

For KL-based selection to outperform random selection, correct outputs must be more distributionally consistent with each other than incorrect ones. This holds when:
- The task has a verifiable correct answer (math, code, factual QA)
- Errors are factual rather than systematic reasoning-chain errors
- N rollouts is large enough for a meaningful consensus to form (N ≥ 5)

It breaks when the model is confidently and consistently wrong — all rollouts agree on the same wrong answer, KL is low, selection fails.

### Clarification: This Is Not the Same as Entropy

Pure entropy of the student measures: "how uncertain is the model at this token?"

KL between rollouts measures: "how much do the model's own outputs disagree here?"

These are related but not identical. KL between rollouts collapses to a sampling variance measure without an external anchor. The meaningful KL in SDFT (student vs. teacher) has a semantic anchor — the teacher genuinely has more information. Pure self-checking via rollout comparison does not have this anchor, which is its fundamental limitation.

### Diagnostic Experiment (First Step)

Before building the full pipeline:
1. Generate N rollouts per question on existing eval sets
2. Compute pairwise KL between output distributions
3. Select the minimum-KL centroid output
4. Measure accuracy vs. random selection vs. majority voting

If minimum-KL selection outperforms random selection, the assumption holds and the direction is viable.

---

## Other Directions Considered

### Compute Efficiency
Engineering improvements to the training loop:
- **KV cache the teacher prefix**: demonstrations are shared across examples; compute the KV cache once per epoch and reuse
- **Batch student and teacher together**: single forward pass with masked attention instead of two separate passes
- **Quantize the teacher**: teacher runs under `no_grad`; INT8/INT4 reduces memory without affecting gradients
- **Adaptive token budget**: instead of a fixed entropy quantile, backpropagate through only the top-K highest-KL tokens per batch

Low novelty, high practical value. A prerequisite for scaling the other experiments.

### Activation Map Distillation

Instead of matching output token distributions (logits), match intermediate hidden states between student and teacher. The residual stream at layer N encodes richer semantic signal than the output distribution. Requires student and teacher to share architecture depth. Related to FitNets-style distillation in vision but less explored for LLMs in this on-policy setting.

### Repo / Style Internalization

Give the teacher a style guide or codebase conventions as the demonstration context. After SDFT, the student follows those conventions without the guide in context. Clean evaluation metric: does the output conform to the rules when the guide is absent? Practical and compelling but relatively straightforward to implement given existing infrastructure.

### VLMs and Robotics

Apply SDFT to multimodal models where the teacher sees a visual or trajectory demonstration in context and the student learns to act without it. Interesting question: how much of what is in the KV cache (compressed representation of the visual demonstration) can be baked into the weights? High ambition, high infrastructure cost.
