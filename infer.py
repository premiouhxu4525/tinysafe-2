#!/usr/bin/env python3
"""
Usage:
  uv run infer.py "single query"
  uv run infer.py "first query" "second query" "third query"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import DebertaV2Tokenizer

from src.model import SafetyClassifierV2
from src.utils import CATEGORIES, load_config

if len(sys.argv) < 2:
    print('Usage: uv run infer.py "query1" "query2" ...')
    sys.exit(1)

queries = sys.argv[1:]

# Load model
t0 = time.perf_counter()
config = load_config()
threshold = config.get("inference", {}).get("binary_threshold", 0.5)
cat_thresholds = config.get("inference", {}).get("category_thresholds", {})
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
model = SafetyClassifierV2(
    base_model_name=config["base_model"],
    num_categories=config["num_categories"],
    layers_to_keep=None,  # Full model, no pruning
    num_dropout_samples=1,
)
ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
load_time = time.perf_counter() - t0
print(f"  Model loaded in {load_time*1000:.0f}ms ({device})\n")

# Inference
for text in queries:
    t0 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    result = model.predict(inputs["input_ids"], inputs["attention_mask"])
    infer_time = time.perf_counter() - t0

    score = result["unsafe_score"].item()
    cat_scores = result["category_scores"].squeeze(0)
    label = "UNSAFE" if score > threshold else "SAFE"

    print(f'  > "{text}"')
    print(f"  {label}  (score: {score:.3f}, threshold: {threshold}, {infer_time*1000:.1f}ms)\n")

    if score > threshold:
        for i, cat in enumerate(CATEGORIES):
            s = cat_scores[i].item()
            t = cat_thresholds.get(cat, 0.5)
            if s > t:
                bar = "\u2588" * int(s * 20)
                print(f"  {cat:<20} {s:.3f}  {bar}")
        print()
