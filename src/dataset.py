"""PyTorch Dataset for safety classification v2 with dual-head labels."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SafetyDataset(Dataset):
    CATEGORIES = [
        "violence", "hate", "sexual", "self_harm",
        "dangerous_info", "harassment", "illegal_activity",
    ]

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(data_path)

    def _load(self, path: str | Path) -> list[dict]:
        path = Path(path)
        if path.suffix == ".jsonl":
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        elif path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = sample["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Support soft labels from Jigsaw continuous scores
        if "soft_label" in sample:
            binary_label = float(sample["soft_label"])
        else:
            binary_label = 1.0 if sample["label"] == "unsafe" else 0.0

        category_labels = torch.zeros(len(self.CATEGORIES), dtype=torch.float32)
        if "categories" in sample:
            for i, cat in enumerate(self.CATEGORIES):
                if sample["categories"].get(cat, False):
                    category_labels[i] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "binary_label": torch.tensor(binary_label, dtype=torch.float32),
            "category_labels": category_labels,
        }

    def get_category_counts(self) -> list[int]:
        """Return per-category positive sample counts for class-balanced loss."""
        counts = [0] * len(self.CATEGORIES)
        for sample in self.samples:
            cats = sample.get("categories", {})
            for i, cat in enumerate(self.CATEGORIES):
                if cats.get(cat, False):
                    counts[i] += 1
        return counts
