#!/usr/bin/env python3
"""
Step 6 (GPU): Binary + per-category threshold optimization.
Sweeps on val set AND ToxicChat test for external calibration.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import SafetyClassifierV2
from src.dataset import SafetyDataset
from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def get_predictions(model, dataloader, device):
    model.eval()
    all_binary_probs, all_binary_labels = [], []
    all_cat_probs, all_cat_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, category_logits = model(input_ids, attention_mask, multi_sample=False)

            all_binary_probs.extend(torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy())
            # Binarize soft labels for eval
            all_binary_labels.extend((batch["binary_label"] > 0.5).float().numpy())
            all_cat_probs.extend(torch.sigmoid(category_logits).cpu().numpy())
            all_cat_labels.extend(batch["category_labels"].numpy())

    return (
        np.array(all_binary_probs),
        np.array(all_binary_labels),
        np.array(all_cat_probs),
        np.array(all_cat_labels),
    )


def sweep_binary_threshold(probs, labels, name=""):
    thresholds = np.arange(0.30, 0.61, 0.01)
    results = []
    best_f1 = 0
    best_threshold = 0.5

    print(f"\n{'Thresh':<10} {'F1-B':<10} {'U-Rec':<10} {'U-Prec':<10} {'FPR':<10}")
    print("-" * 50)

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1_b = f1_score(labels, preds, average="binary", zero_division=0)
        u_rec = recall_score(labels, preds, pos_label=1, zero_division=0)
        u_prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        fpr = 1 - recall_score(labels, preds, pos_label=0, zero_division=0)

        marker = ""
        if f1_b > best_f1:
            best_f1 = f1_b
            best_threshold = t
            marker = " *"

        print(f"{t:<10.2f} {f1_b:<10.4f} {u_rec:<10.4f} {u_prec:<10.4f} {fpr:<10.4f}{marker}")
        results.append({
            "threshold": round(float(t), 2),
            "f1_binary": float(f1_b),
            "unsafe_recall": float(u_rec),
            "unsafe_precision": float(u_prec),
            "fpr": float(fpr),
        })

    print(f"\nBest {name} F1: {best_f1:.4f} at threshold {best_threshold:.2f}")
    return best_threshold, best_f1, results


def sweep_category_thresholds(cat_probs, cat_labels):
    thresholds = np.arange(0.20, 0.71, 0.05)
    best_thresholds = {}

    print(f"\nPer-category threshold sweep:")
    print(f"{'Category':<18} {'Best Thresh':<15} {'F1':<10} {'Recall':<10} {'Prec':<10}")
    print("-" * 63)

    for i, cat in enumerate(CATEGORIES):
        if cat_labels[:, i].sum() == 0:
            best_thresholds[cat] = 0.5
            print(f"{cat:<18} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        best_f1 = 0
        best_t = 0.5
        best_rec = 0
        best_prec = 0

        for t in thresholds:
            preds = (cat_probs[:, i] > t).astype(int)
            f1 = f1_score(cat_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_rec = recall_score(cat_labels[:, i], preds, zero_division=0)
                best_prec = precision_score(cat_labels[:, i], preds, zero_division=0)

        best_thresholds[cat] = round(float(best_t), 2)
        print(f"{cat:<18} {best_t:<15.2f} {best_f1:<10.4f} {best_rec:<10.4f} {best_prec:<10.4f}")

    return best_thresholds


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=config["num_categories"],
        layers_to_keep=None,  # Full model, no pruning
        num_dropout_samples=config["training"]["multi_sample_dropout_count"],
    )
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"
    nw = config["training"]["num_workers"] if use_cuda else 0

    # Val set sweep
    val_ds = SafetyDataset("data/processed/val.jsonl", tokenizer, config["max_length"])
    val_loader = DataLoader(val_ds, batch_size=512 if use_cuda else 64, shuffle=False,
                            num_workers=nw, pin_memory=use_cuda)

    print("=" * 60)
    print("Val Set — Binary Threshold Sweep (0.30 - 0.60)")
    print("=" * 60)
    binary_probs, binary_labels, cat_probs, cat_labels = get_predictions(model, val_loader, device)
    best_val_t, best_val_f1, val_results = sweep_binary_threshold(binary_probs, binary_labels, "val")

    # ToxicChat test sweep
    print("\n" + "=" * 60)
    print("ToxicChat Test — Binary Threshold Sweep")
    print("=" * 60)

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    tc_samples = []
    for row in ds:
        text = row.get("user_input", "")
        if text.strip():
            label = "unsafe" if row.get("toxicity", 0) == 1 else "safe"
            tc_samples.append(normalize_sample(text, label, source="tc_test"))

    tmp_path = Path("data/eval/tc_test_sweep.jsonl")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(tc_samples, tmp_path)
    tc_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    tc_loader = DataLoader(tc_ds, batch_size=512 if use_cuda else 64, shuffle=False,
                           num_workers=nw, pin_memory=use_cuda)

    tc_probs, tc_labels, _, _ = get_predictions(model, tc_loader, device)
    best_tc_t, best_tc_f1, tc_results = sweep_binary_threshold(tc_probs, tc_labels, "ToxicChat")

    # Per-category threshold sweep (on val set)
    print("\n" + "=" * 60)
    print("Per-Category Threshold Sweep (Val Set)")
    print("=" * 60)
    best_cat_thresholds = sweep_category_thresholds(cat_probs, cat_labels)

    # Use val-optimized threshold as primary (more generalizable than TC-specific)
    # But report both so user can decide
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Best val threshold:       {best_val_t:.2f} (F1={best_val_f1:.4f})")
    print(f"  Best ToxicChat threshold: {best_tc_t:.2f} (F1={best_tc_f1:.4f})")
    print(f"  Category thresholds:      {best_cat_thresholds}")

    # Save results
    output = {
        "best_binary_threshold": round(float(best_val_t), 2),
        "best_val_f1": float(best_val_f1),
        "best_tc_threshold": round(float(best_tc_t), 2),
        "best_tc_f1": float(best_tc_f1),
        "category_thresholds": best_cat_thresholds,
        "val_sweep": val_results,
        "tc_sweep": tc_results,
    }
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(output, f, indent=2)

    # Update config with best thresholds
    config["inference"]["binary_threshold"] = round(float(best_val_t), 2)
    config["inference"]["category_thresholds"] = best_cat_thresholds
    with open("configs/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nUpdated configs/config.json with optimized thresholds")
    print(f"Results saved to {RESULTS_DIR / 'threshold_sweep.json'}")


if __name__ == "__main__":
    main()
