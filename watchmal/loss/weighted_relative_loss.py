
import torch
import torch.nn as nn
from typing import Optional, Union


class RelativeWeightedLoss(nn.Module):
    """
    A flexible “batch-weighted” regression loss.

    loss_i = base_loss(pred_i, target_i)
    rel_i  = |pred_i - target_i| / (|target_i| + eps)
    combined_i = rel_i * loss_i

    Final loss: reduction(combined_i * sample_weights) * batch_weight

    Args:
        base_criterion: either an nn.Module (e.g. nn.MSELoss(reduction='none'))
                        or a string 'mse' / 'mae'.
        eps: small constant to avoid division by zero.
        reduction: one of 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        base_criterion: Union[str, nn.Module] = 'mse',
        eps: float = 1e-8,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        if isinstance(base_criterion, nn.Module):
            self.criterion = base_criterion
        else:
            if base_criterion.lower() == 'mse':
                self.criterion = nn.MSELoss(reduction='none')
            elif base_criterion.lower() == 'mae':
                self.criterion = nn.L1Loss(reduction='none')
            else:
                raise ValueError(f"Unsupported base_criterion: {base_criterion!r}")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if preds.shape != targets.shape:
            raise ValueError(f"preds {preds.shape} vs targets {targets.shape} mismatch")

        # flatten to (batch_size,)
        base = self.criterion(preds, targets)
        base = base.view(base.size(0), -1).mean(dim=1)

        rel = torch.abs(preds - targets) / (torch.abs(targets) + self.eps)
        rel = rel.view(rel.size(0), -1).mean(dim=1)

        combined = base * rel

        if self.reduction == 'mean':
            loss = combined.mean()
        elif self.reduction == 'sum':
            loss = combined.sum()
        elif self.reduction == 'none':
            loss = combined
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction!r}")

        # ---- apply batch weight ----
        if batch_weight is not None:
            loss = loss * batch_weight

        return loss

