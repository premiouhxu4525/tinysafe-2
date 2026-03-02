"""Dual-head safety classifier v2: DeBERTa-v3-small with layer pruning and multi-sample dropout."""

import copy

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SafetyClassifierV2(nn.Module):
    """
    DeBERTa-v3-small with layers 2 & 3 dropped (~98M params).
    Binary head: safe/unsafe from [CLS].
    Category head: 7-way multi-label from [CLS].
    Multi-sample dropout: N forward passes through dropout, averaged loss.
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-small",
        num_categories: int = 7,
        layers_to_keep: list[int] | None = None,
        num_dropout_samples: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_dropout_samples = num_dropout_samples
        self.layers_to_keep = layers_to_keep

        # Load full model
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden = self.backbone.config.hidden_size  # 768 for small

        # Prune layers if specified
        if layers_to_keep is not None:
            self._prune_layers(layers_to_keep)

        # Multi-sample dropout heads
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_dropout_samples)
        ])
        self.binary_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, num_categories)

    def _prune_layers(self, layers_to_keep: list[int]):
        """Remove transformer layers not in layers_to_keep."""
        encoder = self.backbone.encoder
        original_layers = encoder.layer
        num_original = len(original_layers)

        # Validate
        for idx in layers_to_keep:
            assert 0 <= idx < num_original, f"Layer {idx} out of range [0, {num_original})"

        # Keep only specified layers
        new_layers = nn.ModuleList([original_layers[i] for i in layers_to_keep])
        encoder.layer = new_layers

        # Update config
        self.backbone.config.num_hidden_layers = len(layers_to_keep)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_sample: bool = True,
    ):
        """
        Returns (binary_logits, category_logits).
        During training with multi_sample=True, returns averaged logits across dropout masks.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0]  # [CLS] token

        if multi_sample and self.training and self.num_dropout_samples > 1:
            # Multi-sample dropout: average logits across N dropout masks
            binary_logits_list = []
            category_logits_list = []
            for dropout in self.dropouts:
                h = dropout(cls_embed)
                binary_logits_list.append(self.binary_head(h))
                category_logits_list.append(self.category_head(h))
            binary_logit = torch.stack(binary_logits_list).mean(dim=0)
            category_logits = torch.stack(category_logits_list).mean(dim=0)
        else:
            # Inference or single-sample: use first dropout
            h = self.dropouts[0](cls_embed)
            binary_logit = self.binary_head(h)
            category_logits = self.category_head(h)

        return binary_logit, category_logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            binary_logit, category_logits = self.forward(
                input_ids, attention_mask, multi_sample=False
            )
        unsafe_score = torch.sigmoid(binary_logit).squeeze(-1)
        category_scores = torch.sigmoid(category_logits)
        return {
            "unsafe_score": unsafe_score,
            "category_scores": category_scores,
        }


class EMAModel:
    """Exponential moving average of model weights for smoother eval."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA weights (for eval)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights (after eval)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class FGM:
    """Fast Gradient Method adversarial training on embedding layer."""

    def __init__(self, model: nn.Module, epsilon: float = 0.3, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """Remove adversarial perturbation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
