from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyWithPerClassFocal(nn.Module):
    """Cross entropy combined with a per-class averaged focal component."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        focal_gamma: float = 3.0,
        focal_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            if weight_tensor.numel() != self.num_classes:
                raise ValueError(
                    f"Class weights length ({weight_tensor.numel()}) must match num_classes ({self.num_classes})."
                )
            self.register_buffer("ce_weight", weight_tensor)
        else:
            self.register_buffer("ce_weight", torch.empty(0,device=logits.device, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_weight = self.ce_weight.to(logits.device) if self.ce_weight.numel() > 0 else None
        ce = F.cross_entropy(
            logits,
            targets,
            weight=ce_weight,
            ignore_index=self.ignore_index,
        )
        if self.focal_weight == 0.0:
            return ce

        focal = self._per_class_focal_loss(logits, targets)
        return ce + self.focal_weight * focal

    def _per_class_focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            if self.ignore_index is not None:
                valid_mask &= targets != self.ignore_index

        if not valid_mask.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        probs = torch.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.view(-1)
        valid_mask_flat = valid_mask.view(-1)

        probs = probs[valid_mask_flat]
        targets_flat = targets_flat[valid_mask_flat]
        if probs.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        gathered = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        gathered = torch.clamp(gathered, min=1e-6, max=1.0)
        focal_terms = (-torch.log(gathered)) * (1.0 - gathered) ** self.focal_gamma

        per_class_losses = []
        for class_idx in range(self.num_classes):
            class_mask = targets_flat == class_idx
            if class_mask.any():
                per_class_losses.append(focal_terms[class_mask].mean())

        if not per_class_losses:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        stacked = torch.stack(per_class_losses)
        return stacked.mean()
    



class FocalLossCEAligned(nn.Module):
    """
    CE-aligned Focal Loss (micro-averaged).

    Computes:
        L = mean_i [ - w_{y_i} * (1 - p_t)^gamma * log(p_t) ]
    where p_t is the softmax probability of the true class.

    - Reduces to CrossEntropyLoss when gamma = 0
    - Uses class weights (optional) like CrossEntropyLoss
    - Handles ignore_index
    - Mean reduction (micro-average) over all valid pixels

    Args:
        gamma (float): Focusing parameter. Default = 2.0.
        weight (Tensor, optional): Class weights tensor of shape [num_classes].
                                   Same as `weight` in CrossEntropyLoss.
        ignore_index (int, optional): Target value to ignore.
        reduction (str, optional): Only 'mean' and 'none' are supported. Default = 'mean'.
    """

    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor | None = None,
                 ignore_index: int | None = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else torch.tensor([]))
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        # --- valid mask (like CE) ---
        with torch.no_grad():
            valid = torch.ones_like(targets, dtype=torch.bool)
            if self.ignore_index is not None:
                valid &= (targets != self.ignore_index)

        if not valid.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        # --- log softmax and gather ---
        log_probs = F.log_softmax(logits, dim=1)                 # [B,C,H,W]
        log_p_t = log_probs.gather(1, targets.unsqueeze(1))      # [B,1,H,W]
        log_p_t = log_p_t.squeeze(1)[valid]                      # [N]
        p_t = log_p_t.exp()                                      # [N]

        # --- class weights ---
        if self.weight.numel() > 0:
            w = self.weight.to(logits.device)[targets[valid]]    # [N]
        else:
            w = 1.0

        # --- focal factor ---
        focal_factor = (1.0 - p_t).pow(self.gamma)

        # --- per-pixel loss ---
        loss = -(w * focal_factor * log_p_t)                     # [N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

