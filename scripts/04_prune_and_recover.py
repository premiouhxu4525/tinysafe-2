#!/usr/bin/env python3
"""
Step 4 (GPU): Prune DeBERTa-v3-small layers and do recovery fine-tune.
Drop layers 2 & 3, keep [0, 1, 4, 5]. 2-epoch recovery on training data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, recall_score

from src.model import SafetyClassifierV2
from src.dataset import SafetyDataset
from src.utils import load_config

CHECKPOINT_DIR = Path("checkpoints")


def main():
    config = load_config()
    recovery_config = config["training"]["recovery"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    print(f"Loading tokenizer: {config['base_model']}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    # Load data
    train_ds = SafetyDataset("data/processed/train.jsonl", tokenizer, config["max_length"])
    val_ds = SafetyDataset("data/processed/val.jsonl", tokenizer, config["max_length"])

    use_cuda = device.type == "cuda"
    nw = config["training"]["num_workers"] if use_cuda else 0
    train_loader = DataLoader(train_ds, batch_size=recovery_config["batch_size"],
                              shuffle=True, num_workers=nw, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["eval_batch_size"],
                            shuffle=False, num_workers=nw, pin_memory=use_cuda)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Build pruned model
    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=config["num_categories"],
        layers_to_keep=config["pruning"]["layers_to_keep"],
        num_dropout_samples=1,  # single dropout for recovery
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"Layers kept: {config['pruning']['layers_to_keep']}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=recovery_config["learning_rate"],
                                  weight_decay=recovery_config["weight_decay"])

    grad_accum = recovery_config["gradient_accumulation_steps"]
    total_steps = len(train_loader) * recovery_config["num_epochs"] // grad_accum
    warmup_steps = int(total_steps * recovery_config["warmup_ratio"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=recovery_config["learning_rate"],
        total_steps=total_steps, pct_start=warmup_steps/total_steps,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"\nRecovery: {recovery_config['num_epochs']} epochs, LR={recovery_config['learning_rate']}")

    import time
    for epoch in range(recovery_config["num_epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels = batch["binary_label"].to(device)

            binary_logits, _ = model(input_ids, attention_mask, multi_sample=False)
            loss = loss_fn(binary_logits.squeeze(-1), binary_labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"    Step {step+1}/{len(train_loader)} | Loss: {avg:.4f}")

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits, _ = model(input_ids, attention_mask, multi_sample=False)
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend((batch["binary_label"] > 0.5).int().numpy())

        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: Train loss={avg_loss:.4f} | Val F1={f1:.4f} | Recall={rec:.4f} | ({elapsed:.0f}s)")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, CHECKPOINT_DIR / "recovered_model.pt")
    print(f"\nSaved recovered model to {CHECKPOINT_DIR / 'recovered_model.pt'}")


if __name__ == "__main__":
    main()
