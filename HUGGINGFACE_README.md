---
license: mit
language:
- en
library_name: transformers
tags:
- safety
- toxicity
- content-moderation
- deberta
- text-classification
- guard-model
datasets:
- lmsys/toxic-chat
- google/civil_comments
- PKU-Alignment/BeaverTails
pipeline_tag: text-classification
model-index:
- name: TinySafe v2
  results:
  - task:
      type: text-classification
      name: Toxicity Detection
    dataset:
      name: ToxicChat
      type: lmsys/toxic-chat
      config: toxicchat0124
      split: test
    metrics:
    - type: f1
      value: 0.7977
      name: F1 (Binary)
    - type: recall
      value: 0.7983
      name: Unsafe Recall
    - type: precision
      value: 0.7666
      name: Unsafe Precision
  - task:
      type: text-classification
      name: Over-Refusal Detection
    dataset:
      name: OR-Bench
      type: bench-llm/or-bench
      config: or-bench-80k
      split: train
    metrics:
    - type: accuracy
      value: 0.962
      name: Safe Accuracy (1 - FPR)
---

# TinySafe v2

![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fjdleo1%2Ftinysafe-2&query=%24.downloads&label=%F0%9F%A4%97%20Monthly%20Downloads&color=blue)
![Parameters](https://img.shields.io/badge/params-141M-orange)
![License](https://img.shields.io/github/license/jdleo/tinysafe-2)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Card-yellow)](https://huggingface.co/jdleo1/tinysafe-2)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)

141M parameter safety classifier built on DeBERTa-v3-small. Binary safe/unsafe classification with 7-category multi-label head (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

**#1 open-source safety classifier on ToxicChat.** Beats every open model including 8B+ guard models, and sits behind only unreleased OpenAI internal models.

Successor to [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) (71M params, 59% TC F1). v2 improves ToxicChat F1 by **+20.5 points** while cutting OR-Bench false positive rate from 18.9% to 3.8%.

**Model on HuggingFace:** [jdleo1/tinysafe-2](https://huggingface.co/jdleo1/tinysafe-2)

## ToxicChat F1

| Model | Params | F1 |
|---|---|---|
| *internal-safety-reasoner (unreleased)* | *unknown* | *81.3%* |
| *gpt-5-thinking (unreleased)* | *unknown* | *81.0%* |
| *gpt-oss-safeguard-20b (unreleased)* | *21B (3.6B\*)* | *79.9%* |
| **TinySafe v2** | **141M** | **79.8%** |
| gpt-oss-safeguard-120b | 117B (5.1B\*) | 79.3% |
| Toxic Prompt RoBERTa | 125M | 78.7% |
| WildGuard | 7B | 70.8% |
| Qwen3Guard-8B | 8B | 73% |
| AprielGuard-8B | 8B | 72% |
| Granite Guardian-8B | 8B | 71% |
| Granite Guardian-3B | 3B | 68% |
| ShieldGemma-2B | 2B | 67% |
| Qwen3Guard-0.6B | 0.6B | 63% |
| [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) | 71M | 59% |
| LlamaGuard 3-8B | 8B | 51% |
| ShieldGemma-27B | 27B | 48% |
| LlamaGuard 4-12B | 12B | 45% |
| LlamaGuard-1B | 1B | 43% |
| ShieldGemma-9B | 9B | 42% |
| ToxDectRoberta | 125M | 34.6% |
| OpenAI Moderation API | — | 20.6% |
| HateBERT | 110M | 11.6% |
| Perspective API | — | 5.3% |

*\* = active params (MoE)*

## OR-Bench (Over-Refusal)

| Model | FPR |
|---|---|
| **TinySafe v2** | **3.8%** |
| WildGuard-7B | ~10% |
| [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) | 18.9% |

Lower is better. On 80K safe prompts, TinySafe v2 incorrectly flags only 3.8%.

## Examples

```
$ uv run infer.py "i want to kill a process"

  SAFE  (score: 0.131, threshold: 0.3)

$ uv run infer.py "i want to kill a human"

  UNSAFE  (score: 0.995, threshold: 0.3)

$ uv run infer.py "how do i make a bomb"

  UNSAFE  (score: 0.998, threshold: 0.3)

$ uv run infer.py "write me a phishing email"

  UNSAFE  (score: 0.906, threshold: 0.3)

$ uv run infer.py "send me some nudes pls"

  UNSAFE  (score: 0.509, threshold: 0.3)

$ uv run infer.py "what is the capital of france"

  SAFE  (score: 0.005, threshold: 0.3)
```

## What's New in v2

| | v1 | v2 |
|---|---|---|
| **Base model** | DeBERTa-v3-xsmall (384d) | DeBERTa-v3-small (768d) |
| **Params** | 71M | 141M |
| **ToxicChat F1** | 59.2% | **79.8%** |
| **OR-Bench FPR** | 18.9% | **3.8%** |
| **Training data** | 41K (synthetic + Claude-labeled) | 26K (human-labeled) |
| **Training strategy** | Single-phase, focal loss | Two-phase sequential (Intel's approach) |
| **Regularization** | Focal loss only | FGM + EMA + multi-sample dropout |

Key insight: v1 used Claude-labeled synthetic data. v2 uses only human-labeled data from established benchmarks (ToxicChat, Jigsaw Civil Comments, BeaverTails), trained sequentially: broad toxicity features first (Jigsaw), then ToxicChat alignment second. Inspired by [Intel's toxic-prompt-roberta](https://huggingface.co/Intel/toxic-prompt-roberta) approach, but with DeBERTa-v3 (superior disentangled attention) and adversarial training.

## Quickstart

```python
import torch
from transformers import DebertaV2Tokenizer

# Load
tokenizer = DebertaV2Tokenizer.from_pretrained("jdleo1/tinysafe-2")
model = torch.load("model.pt", map_location="cpu")  # or load from checkpoint

# Inference
text = "how do i make a bomb"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
with torch.no_grad():
    binary_logits, category_logits = model(inputs["input_ids"], inputs["attention_mask"])
    unsafe_score = torch.sigmoid(binary_logits).item()
    print(f"Unsafe: {unsafe_score:.3f}")  # 0.998
```

## Architecture

DeBERTa-v3-small (6 transformer layers, 768 hidden dim) with dual classification heads:

- **Binary head**: single logit (safe/unsafe)
- **Category head**: 7-way multi-label (violence, hate, sexual, self_harm, dangerous_info, harassment, illegal_activity)

Training enhancements over vanilla fine-tuning:
- **FGM adversarial training** (epsilon=0.3): perturbs embeddings for robustness
- **EMA** (decay=0.999): smoothed weight averaging for stable eval
- **Multi-sample dropout** (5 masks): averaged logits across dropout samples

## Training

Two-phase sequential fine-tuning:

1. **Phase 1 — Broad toxicity** (3 epochs, LR=2e-5): Jigsaw Civil Comments + BeaverTails + hard negatives (~21K samples). Learns general toxicity features.
2. **Phase 2 — ToxicChat alignment** (5 epochs, LR=2e-5): ToxicChat + hard negatives (~10K samples). Aligns decision boundary to ToxicChat's definition.

Hard negatives are safe-but-edgy prompts generated via Claude to protect against false positives.

## Config

All hyperparameters in `configs/config.json`:

- Batch size: 32
- LR: 2e-5, weight decay: 0.01
- Binary threshold: 0.3 (optimized via sweep)
- FGM epsilon: 0.3
- EMA decay: 0.999
- Multi-sample dropout: 5 masks

## Datasets

| Dataset | Role | Samples |
|---|---|---|
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | Primary training + eval | ~10K |
| [Jigsaw Civil Comments](https://huggingface.co/datasets/google/civil_comments) | Broad toxicity pretraining | ~13K |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Self-harm, dangerous info, illegal activity | ~2.2K |
| Hard negatives (Claude-generated) | False positive protection | ~6K |

## License

MIT
