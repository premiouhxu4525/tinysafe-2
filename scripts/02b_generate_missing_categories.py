#!/usr/bin/env python3
"""
Step 2b: Pull self_harm, dangerous_info, illegal_activity from BeaverTails.
Civil_comments doesn't have these categories — BeaverTails does, with human labels.
No synthetic generation needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from src.utils import CATEGORIES, BEAVERTAILS_CATEGORY_MAP, load_config, normalize_sample, save_jsonl

DATA_DIR = Path("data/raw")

# Only pull samples for categories missing from civil_comments
MISSING_CATEGORIES = {"self_harm", "dangerous_info", "illegal_activity"}
TARGET_PER_CATEGORY = 2000


def main():
    load_config()  # validate config exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BeaverTails dataset...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    print(f"  Loaded {len(ds)} rows")

    # Collect samples per missing category
    category_samples = {cat: [] for cat in MISSING_CATEGORIES}

    for row in tqdm(ds, desc="Processing BeaverTails"):
        text = row.get("prompt", "")
        if not text or not text.strip():
            continue

        is_safe = row.get("is_safe", True)
        if is_safe:
            continue  # only want unsafe samples for missing categories

        # Map BeaverTails categories to ours
        bt_labels = row.get("category", {})
        if isinstance(bt_labels, str):
            import json
            try:
                bt_labels = json.loads(bt_labels)
            except Exception:
                continue

        our_cats = {c: False for c in CATEGORIES}
        matched_missing = set()

        for bt_cat, is_flagged in bt_labels.items():
            if is_flagged:
                our_cat = BEAVERTAILS_CATEGORY_MAP.get(bt_cat)
                if our_cat and our_cat in MISSING_CATEGORIES:
                    our_cats[our_cat] = True
                    matched_missing.add(our_cat)

        # Add to each matched missing category's pool
        for cat in matched_missing:
            if len(category_samples[cat]) < TARGET_PER_CATEGORY:
                category_samples[cat].append(
                    normalize_sample(text, "unsafe", our_cats, source=f"beavertails_{cat}")
                )

    # Deduplicate across categories (same text might appear in multiple)
    seen_texts = set()
    all_samples = []
    for cat in MISSING_CATEGORIES:
        for s in category_samples[cat]:
            text_key = s["text"].strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_samples.append(s)

    save_jsonl(all_samples, DATA_DIR / "synthetic_missing_categories.jsonl")

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_samples)} samples from BeaverTails")
    print(f"{'='*60}")
    for cat in MISSING_CATEGORIES:
        count = sum(1 for s in all_samples if s.get("categories", {}).get(cat))
        print(f"  {cat}: {count}")
    print(f"Saved to {DATA_DIR / 'synthetic_missing_categories.jsonl'}")


if __name__ == "__main__":
    main()
