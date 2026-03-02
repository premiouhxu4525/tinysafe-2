#!/usr/bin/env python3
"""
Step 8: Export v2 model to ONNX + optional INT8 quantization.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import DebertaV2Tokenizer

from src.model import SafetyClassifierV2
from src.utils import load_config

CHECKPOINT_DIR = Path("checkpoints")
EXPORT_DIR = Path("exported")


def main():
    config = load_config()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # export on CPU

    print(f"Loading model from {CHECKPOINT_DIR / 'best_model.pt'}...")
    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=config["num_categories"],
        layers_to_keep=None,  # Full model, no pruning
        num_dropout_samples=1,  # single dropout for export
    )
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Save tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
    tokenizer.save_pretrained(EXPORT_DIR / "tokenizer")
    print(f"Tokenizer saved to {EXPORT_DIR / 'tokenizer'}")

    # ONNX export
    print("\nExporting to ONNX...")
    dummy_input = tokenizer(
        "test input for export",
        return_tensors="pt",
        padding="max_length",
        max_length=config["max_length"],
        truncation=True,
    )

    onnx_path = EXPORT_DIR / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["binary_logit", "category_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "binary_logit": {0: "batch_size"},
            "category_logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {onnx_path}")

    # INT8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process

        int8_path = EXPORT_DIR / "model_int8.onnx"
        preprocessed_path = EXPORT_DIR / "model_preprocessed.onnx"

        # Preprocess to fix shape inference issues
        try:
            quant_pre_process(str(onnx_path), str(preprocessed_path))
            source_for_quant = str(preprocessed_path)
        except Exception as e:
            print(f"  Preprocessing failed ({e}), quantizing directly")
            source_for_quant = str(onnx_path)

        quantize_dynamic(
            source_for_quant,
            str(int8_path),
            weight_type=QuantType.QInt8,
        )
        # Clean up preprocessed
        if preprocessed_path.exists():
            preprocessed_path.unlink()
        print(f"INT8 model saved to {int8_path}")
    except Exception as e:
        print(f"INT8 quantization failed: {e}, fp32 ONNX still available")

    # Save inference config
    inference_config = {
        "model_path": "model.onnx",
        "model_int8_path": "model_int8.onnx",
        "tokenizer_path": "tokenizer",
        "max_length": config["max_length"],
        "binary_threshold": config["inference"]["binary_threshold"],
        "category_thresholds": config["inference"]["category_thresholds"],
        "categories": config["categories"],
    }
    with open(EXPORT_DIR / "config.json", "w") as f:
        json.dump(inference_config, f, indent=2)
    print(f"Inference config saved to {EXPORT_DIR / 'config.json'}")

    # File sizes
    for p in [onnx_path, EXPORT_DIR / "model_int8.onnx"]:
        if p.exists():
            size_mb = os.path.getsize(p) / 1024 / 1024
            print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
