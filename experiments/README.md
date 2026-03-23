# Experiments

This directory contains research scripts written independently of the original SDFT authors' code.
The original codebase (`distil_trainer.py`, `distil_config.py`, `main.py`, `eval_*.py`) is left untouched.

---

## Model

Both scripts use **`Qwen/Qwen2.5-7B-Instruct`** — the same default as the authors' `main.py`.

**Why this model:**
- Instruction-tuned, produces structured `<reasoning>` / `<answer>` output needed for region-level KL analysis
- 7B fits on a single 24GB GPU in bfloat16 for inference
- Already validated on both datasets by the original authors

**Local vs. HuggingFace:** If the model is already cached, pass the local path. If not, the scripts will download it automatically via HuggingFace Hub on first run.

---

## LoRA vs. Full Fine-Tuning

This only applies to `transfer_saliency.py`, which requires an SDFT-trained checkpoint. The self-checking diagnostic needs no training at all.

### For the SDFT training checkpoint (via `main.py`):

| Setup | Recommendation | Why |
|---|---|---|
| Single GPU < 24GB | **4-bit QLoRA** (`--quantization 4bit`) | Qwen2.5-7B needs ~14GB in bfloat16; full fine-tuning with optimizer states requires ~56GB+ |
| Single GPU 24GB (e.g. A100 40GB) | **LoRA in bfloat16** (`--quantization none`, LoRA applied automatically) | Fast, no quality loss vs. full fine-tuning for short training runs |
| Multi-GPU ≥ 4 × 24GB | **Full fine-tuning** (`--quantization none`) | Full fine-tuning gives the cleanest signal for the transfer saliency analysis — no adapter approximation error |

**Recommendation for the transfer saliency experiment specifically:** prefer full fine-tuning if hardware allows. QLoRA introduces quantization noise that may partially confound the KL measurement — you want weight changes to reflect what the training learned, not quantization artifacts.

### LoRA config (applied automatically when quantization is enabled in `main.py`):
- Rank: `r=16`, alpha: `32`, dropout: `0.05`
- Target: all linear layers (`target_modules="all-linear"`)

---

## Requirements

All dependencies are already in the root `requirements.txt`. The key ones for the experiment scripts:

```
torch==2.9.0
transformers==4.57.1
accelerate==1.11.0
peft==0.17.1
datasets==4.3.0
numpy==2.2.6
wandb==0.22.2
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Scripts

### `self_checking.py` — KL-Based Self-Checking Diagnostic

Tests whether KL divergence between rollouts is a meaningful correctness signal, and whether the signal lives in the reasoning region or the answer region (memorization diagnostic).

**What it does:**
- Generates N rollouts per science question at temperature T
- Computes pairwise symmetric KL between all rollout distributions
- Classifies tokens into `reasoning` / `answer` / `other` regions via XML tag tracking
- Compares three selection strategies against ground truth:
  - Random selection
  - Majority voting (most common extracted answer)
  - KL centroid (rollout with minimum average KL to all others)
- Reports KL and entropy separately per region, split by whether the question was answered correctly

**Key questions:**
1. Does `accuracy(kl_centroid) > accuracy(random)`? — is KL a correctness signal at all?
2. Is `reasoning_KL(correct) < reasoning_KL(wrong)`? — does the model reason more consistently when it knows the answer?
3. Or is only `answer_KL(correct) < answer_KL(wrong)`? — pattern-matching signature

```bash
# activate the virtualenv first
source self-distillation/bin/activate

python experiments/self_checking.py \
    --model_path     Qwen/Qwen2.5-7B-Instruct \
    --n_rollouts     8 \
    --temperature    0.7 \
    --n_samples      100 \
    --max_new_tokens 512 \
    --seed           42 \
    --output_dir     experiments/results/self_checking
```

**Recommended parameters:**
- `--n_rollouts 8`: enough for a meaningful consensus, not too expensive
- `--temperature 0.7`: enough diversity without collapsing to noise
- `--n_samples 100` for a first run, `507` for full eval set

**Outputs** (`experiments/results/self_checking/`):
- `summary.json` — aggregated accuracy and KL metrics per region and correctness
- `samples.jsonl` — one JSON record per question, written incrementally (safe to interrupt)

---

### `transfer_saliency.py` — KL Transfer Saliency Map

Measures what SDFT training on tool-use transferred to science, and where in the generation process.

**What it does:**
- Loads base model and SDFT-trained model (same architecture, different weights)
- Runs both on science eval questions (greedy decoding)
- Computes per-token KL(base || sdft) on the base model's completions
- Classifies each token as `structural`, `reasoning`, or `content`
- Aggregates mean KL per category

**Key question:** Is KL highest at `reasoning` tokens (reasoning structure transferred) or `content` tokens (domain knowledge transferred)?

**Requires:** An SDFT checkpoint trained on tool-use via `main.py`. Train it first:
```bash
python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name tooluse \
    --quantization 4bit \
    --output_dir checkpoints/sdft-tooluse
```

Then run the saliency analysis:
```bash
source self-distillation/bin/activate

python experiments/transfer_saliency.py \
    --base_model_path  Qwen/Qwen2.5-7B-Instruct \
    --sdft_model_path  checkpoints/sdft-tooluse \
    --n_samples        100 \
    --max_new_tokens   512 \
    --output_dir       experiments/results/transfer_saliency
```

**Outputs** (`experiments/results/transfer_saliency/`):
- `summary.json` — accuracy of both models, mean KL per token category
- `samples.jsonl` — one record per question with per-token KL and category labels

---

## Suggested Order

1. **Run `self_checking.py` now** — no training needed, pure inference, runs today
2. **While it runs:** start SDFT training on tool-use with `main.py`
3. **After training:** run `transfer_saliency.py` with the checkpoint

---

## Interpreting Results

### Self-checking

| Pattern | Interpretation |
|---|---|
| `kl_centroid > random` and `kl_centroid > majority` | KL adds signal beyond voting |
| `kl_centroid ≈ majority` | KL adds no signal over surface consistency |
| `reasoning_KL(correct) << reasoning_KL(wrong)` | Model genuinely reasons on known questions |
| `reasoning_KL(correct) ≈ reasoning_KL(wrong)` | Reasoning region is noisy regardless — pattern-matching |
| Only `answer_KL(correct) << answer_KL(wrong)` | Pure pattern-matching: answer clusters but reasoning doesn't |

### Transfer saliency

| Highest KL at | Interpretation |
|---|---|
| `reasoning` tokens | SDFT transferred a reasoning structure, not facts |
| `content` tokens | SDFT transferred domain knowledge |
| `structural` tokens | SDFT only changed output formatting |

---

## Token Category Definitions

| Category | Examples |
|---|---|
| `structural` | `<reasoning>`, `</answer>`, `<think>` |
| `reasoning` | therefore, because, thus, hence, consequently, implies, since, so |
| `content` | everything else (facts, numbers, domain-specific words) |
