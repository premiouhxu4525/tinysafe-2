#!/usr/bin/env python3
"""
Step 5 (GPU): Two-phase sequential training (Intel's approach + our edges).
Phase 1: Jigsaw + BeaverTails + hard negatives (broad toxicity features)
Phase 2: ToxicChat + hard negatives (align to ToxicChat's definition)

Edges over Intel: DeBERTa-v3 > RoBERTa, EMA, multi-sample dropout, FGM.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import SafetyClassifierV2, EMAModel, FGM
from src.dataset import SafetyDataset
from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl, load_jsonl

CHECKPOINT_DIR = Path("checkpoints")


def build_toxicchat_eval(tokenizer, max_length):
    """Build ToxicChat test set for external validation."""
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    samples = []
    for row in ds:
        text = row.get("user_input", "")
        if text.strip():
            label = "unsafe" if row.get("toxicity", 0) == 1 else "safe"
            samples.append(normalize_sample(text, label, source="tc_test"))

    tmp_path = Path("data/eval/tc_test.jsonl")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(samples, tmp_path)
    return SafetyDataset(tmp_path, tokenizer, max_length)


def evaluate(model, dataloader, device, threshold=0.5):
    """Evaluate model, return metrics dict."""
    model.eval()
    all_probs, all_labels = [], []
    all_cat_probs, all_cat_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, cat_logits = model(input_ids, attention_mask, multi_sample=False)

            all_probs.extend(torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy())
            all_labels.extend(batch["binary_label"].numpy())
            all_cat_probs.extend(torch.sigmoid(cat_logits).cpu().numpy())
            all_cat_labels.extend(batch["category_labels"].numpy())

    probs = np.array(all_probs)
    labels = (np.array(all_labels) > 0.5).astype(int)
    preds = (probs > threshold).astype(int)

    metrics = {
        "f1_binary": f1_score(labels, preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(labels, preds, pos_label=0, zero_division=0),
    }

    cat_probs = np.array(all_cat_probs)
    cat_labels = np.array(all_cat_labels)
    for i, cat in enumerate(CATEGORIES):
        if cat_labels[:, i].sum() > 0:
            cat_preds = (cat_probs[:, i] > 0.5).astype(int)
            metrics[f"{cat}_f1"] = f1_score(cat_labels[:, i], cat_preds, zero_division=0)
            metrics[f"{cat}_precision"] = precision_score(cat_labels[:, i], cat_preds, zero_division=0)
            metrics[f"{cat}_recall"] = recall_score(cat_labels[:, i], cat_preds, zero_division=0)

    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum,
                    fgm, ema, num_categories):
    """Simple BCE training — one epoch."""
    model.train()
    total_loss = 0
    step_count = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        binary_labels = batch["binary_label"].to(device).float()
        category_labels = batch["category_labels"].to(device).float()

        binary_logits, cat_logits = model(input_ids, attention_mask)

        # Simple BCE loss — no fancy stuff
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits.squeeze(-1), binary_labels)
        cat_loss = F.binary_cross_entropy_with_logits(cat_logits, category_labels)
        loss = binary_loss + 0.5 * cat_loss

        loss = loss / grad_accum
        loss.backward()

        # FGM adversarial training (our edge over Intel)
        if fgm is not None:
            fgm.attack()
            adv_binary, adv_cat = model(input_ids, attention_mask)
            adv_loss = F.binary_cross_entropy_with_logits(
                adv_binary.squeeze(-1), binary_labels)
            (adv_loss / grad_accum).backward()
            fgm.restore()

        total_loss += loss.item() * grad_accum
        step_count += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        if (step + 1) % 100 == 0:
            avg = total_loss / step_count
            lr = scheduler.get_last_lr()[0]
            print(f"    Step {step+1}/{len(loader)} | Loss: {avg:.4f} | LR: {lr:.2e}")

    return total_loss / max(step_count, 1)


def run_phase(phase_name, model, train_ds, val_loader, tc_loader, device,
              num_epochs, lr, batch_size, grad_accum, fgm, ema, num_categories,
              save_metric="tc_f1", nw=4):
    """Run a training phase. Returns best metric value."""
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)

    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * 0.06)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps, pct_start=max(warmup_steps / total_steps, 0.01),
    )

    # Reset EMA to current model weights
    ema_tracker = EMAModel(model, decay=0.999)
    best_val = 0

    for epoch in range(num_epochs):
        print(f"\n{phase_name} - Epoch {epoch+1}/{num_epochs}")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum, fgm, ema_tracker, num_categories,
        )
        elapsed = time.time() - t0
        print(f"  Train loss: {train_loss:.4f} ({elapsed:.1f}s)")

        # Eval with EMA weights
        ema_tracker.apply_shadow(model)

        val_metrics = evaluate(model, val_loader, device)
        tc_metrics = evaluate(model, tc_loader, device)

        print(f"  Internal  F1: {val_metrics['f1_binary']:.4f} | Recall: {val_metrics['unsafe_recall']:.4f} | Prec: {val_metrics['unsafe_precision']:.4f} | Safe-R: {val_metrics['safe_recall']:.4f}")
        print(f"  ToxicChat F1: {tc_metrics['f1_binary']:.4f} | Recall: {tc_metrics['unsafe_recall']:.4f} | Prec: {tc_metrics['unsafe_precision']:.4f}")

        # Pick save metric
        if save_metric == "tc_f1":
            current = tc_metrics["f1_binary"]
        else:
            current = val_metrics["f1_binary"]

        if current > best_val:
            best_val = current
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"phase": phase_name},
                "epoch": epoch,
                "metrics": val_metrics,
                "tc_metrics": tc_metrics,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  * New best ({save_metric}): {best_val:.4f}")

        ema_tracker.restore(model)

    print(f"\n{phase_name} complete. Best {save_metric}: {best_val:.4f}")
    return best_val


def main(phase2_only=False):
    config = load_config()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    use_cuda = device.type == "cuda"
    nw = 4 if use_cuda else 0

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
    max_len = config["max_length"]
    num_cats = config["num_categories"]

    # Build ToxicChat eval set
    tc_test_ds = build_toxicchat_eval(tokenizer, max_len)

    # Load ALL training data
    all_train = load_jsonl("data/processed/train.jsonl")
    print(f"Total training samples: {len(all_train)}")

    # Split by source for sequential training
    jigsaw_samples = [s for s in all_train if "jigsaw" in s.get("source", "")]
    beavertails_samples = [s for s in all_train if "beavertails" in s.get("source", "")]
    tc_samples = [s for s in all_train if s.get("source", "") == "toxicchat"]
    hardneg_samples = [s for s in all_train if "hard_neg" in s.get("source", "")]

    print(f"  Jigsaw: {len(jigsaw_samples)}")
    print(f"  BeaverTails: {len(beavertails_samples)}")
    print(f"  ToxicChat: {len(tc_samples)}")
    print(f"  Hard negatives: {len(hardneg_samples)}")

    # Phase 1 data: Jigsaw + BeaverTails + hard negatives (broad toxicity)
    phase1_data = jigsaw_samples + beavertails_samples + hardneg_samples
    phase1_path = Path("data/processed/phase1_broad.jsonl")
    save_jsonl(phase1_data, phase1_path)

    # Phase 2 data: ToxicChat + hard negatives (alignment)
    phase2_data = tc_samples + hardneg_samples
    phase2_path = Path("data/processed/phase2_align.jsonl")
    save_jsonl(phase2_data, phase2_path)

    print(f"\nPhase 1 (broad): {len(phase1_data)} samples")
    print(f"Phase 2 (align): {len(phase2_data)} samples")

    # Build model — NO PRUNING, full DeBERTa-v3-small
    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=num_cats,
        layers_to_keep=None,  # NO PRUNING
        num_dropout_samples=5,
        dropout_rate=0.1,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} (full DeBERTa-v3-small, no pruning)")

    # FGM adversarial training (our edge)
    fgm = FGM(model, epsilon=0.3)

    # Val + TC loaders
    val_ds = SafetyDataset(Path("data/processed/val.jsonl"), tokenizer, max_len)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=nw, pin_memory=use_cuda)
    tc_loader = DataLoader(tc_test_ds, batch_size=256, shuffle=False, num_workers=nw, pin_memory=use_cuda)

    # ============================================================
    # PHASE 1: Broad Toxicity (Jigsaw + BeaverTails + hard negs)
    # ============================================================
    if phase2_only:
        print("\n*** Skipping Phase 1 (--phase2-only) ***")
        assert (CHECKPOINT_DIR / "best_model.pt").exists(), "No Phase 1 checkpoint!"
    else:
        phase1_ds = SafetyDataset(phase1_path, tokenizer, max_len)
        run_phase(
            "Phase 1 (Broad)", model, phase1_ds, val_loader, tc_loader, device,
            num_epochs=3, lr=2e-5, batch_size=32, grad_accum=1,
            fgm=fgm, ema=True, num_categories=num_cats,
            save_metric="internal_f1",  # Phase 1 saves on internal metric
            nw=nw,
        )

    # ============================================================
    # PHASE 2: ToxicChat Alignment (TC + hard negs)
    # ============================================================
    # Reload best Phase 1 model
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    p1_tc = ckpt.get("tc_metrics", {}).get("f1_binary", "N/A")
    print(f"\nLoaded Phase 1 model. TC F1 at end of Phase 1: {p1_tc}")

    phase2_ds = SafetyDataset(phase2_path, tokenizer, max_len)
    best_tc_f1 = run_phase(
        "Phase 2 (TC Align)", model, phase2_ds, val_loader, tc_loader, device,
        num_epochs=5, lr=2e-5, batch_size=32, grad_accum=1,
        fgm=fgm, ema=True, num_categories=num_cats,
        save_metric="tc_f1",  # Phase 2 saves on ToxicChat F1
        nw=nw,
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best ToxicChat F1: {best_tc_f1:.4f}")
    print(f"Target: >0.787 (Intel)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-only", action="store_true",
                        help="Skip Phase 1, load checkpoint, run Phase 2 only")
    args = parser.parse_args()
    main(phase2_only=args.phase2_only)
