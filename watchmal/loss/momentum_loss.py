import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumLoss(nn.Module):
    """
    Relative Euclidean loss for 3D momentum regression.

    Computes SmoothL1 on the relative 3D error:
        ||p_pred - p_true||_2 / ||p_true||_2

    Scale-invariant: the same fractional accuracy is required across all
    momentum magnitudes. Assumes targets are normalized by dividing by the
    dataset maximum (target_norm = [p_max, 0]), so zero momentum stays at
    the origin and the relative error is preserved under normalization.

    Args:
        beta      : SmoothL1 transition point in fractional error units
                    (e.g. 0.1 = quadratic below 10% relative error).
        eps       : Minimum ||p_true_norm|| to avoid instability at very low
                    momenta. In normalized units — e.g. 0.01 corresponds to
                    1% of p_max.
        reduction : 'mean', 'sum', or 'none'.
    """

    def __init__(self, beta: float = 0.1, eps: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norm_true = torch.norm(target, dim=-1).clamp(min=self.eps)
        rel_err = torch.norm(pred - target, dim=-1) / norm_true
        return F.smooth_l1_loss(
            rel_err,
            torch.zeros_like(rel_err),
            beta=self.beta,
            reduction=self.reduction,
        )
