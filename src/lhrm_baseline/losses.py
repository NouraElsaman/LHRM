"""
Loss functions for LHRM baseline.

Eq.17 (PAPER): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
where:
  p_t = sigmoid probability assigned to the TRUE class (1 for positive, 0 for negative)
  alpha_t = alpha for positives (y=1), 1-alpha for negatives (y=0) [ASSUMED A9]
  gamma = focusing parameter [ASSUMED A9: gamma=2]

Applied to CSAM outputs (before sigmoid) via BCEWithLogitsLoss formulation.

Note: The paper does NOT report alpha_t or gamma. We use standard Lin et al. 2017
values: gamma=2, alpha=0.25 (positive class), 0.75 (negative class). [ASSUMED A9]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Applied independently per label (binary classification per label).
    
    PAPER: Eq.17  FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t)
    ASSUMED (A9): gamma=2, alpha=0.25 (positive), 0.75 (negative)
    
    Args:
        gamma: focusing parameter (ASSUMED: 2.0)
        alpha: weight for POSITIVE class (ASSUMED: 0.25)
        reduction: 'mean' over all (doc, label) pairs
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, L] — raw scores before sigmoid (CSAM Eq.8 outputs)
            targets: [B, L] — binary labels (0 or 1)
        
        Returns:
            Scalar focal loss.
        """
        # p_t: probability of the TRUE class
        # For y=1 (positive): p_t = sigmoid(logit)
        # For y=0 (negative): p_t = 1 - sigmoid(logit) = sigmoid(-logit)
        # Use stable formulation via BCE
        
        p = torch.sigmoid(logits)  # [B, L] — predicted probability of positive class
        
        # CE term: -log(p_t)
        # For positive (y=1): -log(p)
        # For negative (y=0): -log(1-p)
        # This is standard BCE loss per element
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B, L]
        
        # p_t for the true class
        p_t = p * targets + (1 - p) * (1 - targets)  # [B, L]
        
        # alpha_t: weight for the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # [B, L]
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)  # [B, L]
        
        # Focal loss
        loss = alpha_t * focal_weight * ce  # [B, L]
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, alpha={self.alpha} [ASSUMED A9], reduction={self.reduction}"


class BCELoss(nn.Module):
    """
    Standard Binary Cross-Entropy loss.
    Control condition for ablation (A9 control experiment).
    REPOSITORY-DEFINED (original implementation).
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Factory function for loss modules."""
    if loss_name == "focal":
        gamma = kwargs.get("focal_gamma", 2.0)
        alpha = kwargs.get("focal_alpha", 0.25)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_name == "bce":
        return BCELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choices: focal, bce")
