"""V2 loss functions: focal loss, asymmetric loss, class-balanced loss, R-Drop."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss with optional label smoothing."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Different focusing parameters for positive vs negative samples.
    Particularly effective for long-tail multi-label problems.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Asymmetric clipping for negatives
        probs_neg = probs.clamp(min=self.clip) if self.clip > 0 else probs

        # Positive and negative losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            pt_pos = probs
            focal_pos = (1 - pt_pos) ** self.gamma_pos
            loss_pos = focal_pos * loss_pos

        if self.gamma_neg > 0:
            pt_neg = 1 - probs
            focal_neg = (1 - pt_neg) ** self.gamma_neg
            loss_neg = focal_neg * loss_neg

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Wraps a per-sample loss with class-balanced (CB) weighting.
    Weights are inversely proportional to effective number of samples.
    """

    def __init__(self, samples_per_class: list[int], beta: float = 0.9999, base_loss: nn.Module = None):
        super().__init__()
        self.base_loss = base_loss or AsymmetricLoss(reduction="none")

        # Compute effective number and weights
        effective_num = [1.0 - beta ** n if n > 0 else 1.0 for n in samples_per_class]
        weights = [1.0 / en if en > 0 else 0.0 for en in effective_num]
        # Normalize so weights sum to num_classes
        total = sum(weights)
        num_classes = len(samples_per_class)
        weights = [w * num_classes / total for w in weights]

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # base_loss returns (B, C) with reduction="none"
        per_sample_loss = self.base_loss(logits, targets)  # (B, C)

        # Apply per-class weights
        weighted = per_sample_loss * self.weights.unsqueeze(0).to(logits.device)
        return weighted.mean()


def compute_rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    R-Drop: KL divergence between two forward passes with different dropout masks.
    Symmetric KL: 0.5 * (KL(p1||p2) + KL(p2||p1))
    """
    p1 = F.log_softmax(torch.cat([logits1, -logits1], dim=-1), dim=-1)
    p2 = F.log_softmax(torch.cat([logits2, -logits2], dim=-1), dim=-1)

    # For binary logits, create 2-class distribution
    q1 = torch.softmax(torch.cat([logits1, -logits1], dim=-1), dim=-1)
    q2 = torch.softmax(torch.cat([logits2, -logits2], dim=-1), dim=-1)

    kl_1 = F.kl_div(p1, q2, reduction="batchmean", log_target=False)
    kl_2 = F.kl_div(p2, q1, reduction="batchmean", log_target=False)

    return alpha * 0.5 * (kl_1 + kl_2)


class DualHeadLossV2(nn.Module):
    """
    Combined loss for v2:
    - Binary: focal loss with label smoothing
    - Category: asymmetric loss with class-balanced weighting
    - R-Drop: KL divergence consistency (applied externally during training)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        category_weight: float = 0.7,
        asl_gamma_pos: float = 1.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        samples_per_class: list[int] | None = None,
    ):
        super().__init__()
        self.binary_loss = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
        self.category_weight = category_weight

        if samples_per_class is not None:
            self.category_loss = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                base_loss=AsymmetricLoss(
                    gamma_pos=asl_gamma_pos,
                    gamma_neg=asl_gamma_neg,
                    clip=asl_clip,
                    reduction="none",
                ),
            )
        else:
            self.category_loss = AsymmetricLoss(
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
            )

    def forward(
        self,
        binary_logits: torch.Tensor,
        category_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        category_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_binary = self.binary_loss(binary_logits.squeeze(-1), binary_targets)
        l_category = self.category_loss(category_logits, category_targets)
        total = l_binary + self.category_weight * l_category

        return {
            "loss": total,
            "binary_loss": l_binary.detach(),
            "category_loss": l_category.detach(),
        }
