from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


def _head_name(i: int) -> str:
    """Semantic name for a head index, matching the per-index branching in the loss forward() methods."""
    if i == 0:
        return "energy"
    if i == 1:
        return "time"
    if i == 2:
        return "vertex"
    if i == 3:
        return "direction"
    return f"head_{i}"


class MultiTaskLoss(_WeightedLoss):
    """
    Multi-head Huber loss with learned uncertainty weighting (log_vars) for concatenated outputs.

    Each active head contributes an unweighted Huber-style loss that is then scaled by learned
    uncertainty: loss_i / sigma_i + log(sigma_i), where sigma_i = exp(log_var_i).
    The final loss is the mean over active heads.

    Head types (by index):
        percent_head_idx : percentage-based Huber loss.
        logratio_head_idx: log-ratio Huber loss, log1p(pred / true), for strictly positive targets (energies)
        1                : scalar Huber loss (vertex times)
        2                : 3D vertex loss (Huber on Euclidean distance) (vertex positions)
        3                : 3D direction loss (Huber on arccos angular distance) (directions)
        default          : Huber loss

    Args:
        head_dims          : output dimensions per head, e.g. [1, 4, 3]. Use 0 to skip a head.
        percent_head_idx   : index of the head to apply percentage-based Huber loss.
        logratio_head_idx  : index of the head to apply log-ratio Huber loss.
        delta              : Huber delta — float (shared) or list (per head).
        static_head        : reserved for future use.
        reduction          : 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        head_dims: List[int],
        percent_head_idx: Optional[int] = None,
        logratio_head_idx: Optional[int] = None,
        momentum_head_idx: Optional[int] = None,
        delta: Any = 1.0,
        static_head: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        self.head_dims = head_dims
        self.percent_head_idx = percent_head_idx
        self.static_head = static_head
        self.reduction = reduction
        self.logratio_head_idx = logratio_head_idx
        self.momentum_head_idx = momentum_head_idx
        if isinstance(delta, (float, int)):
            self.delta = [float(delta)] * len(head_dims)
        elif isinstance(delta, (list, ListConfig)):
            self.delta = [float(d) for d in delta]
            if len(self.delta) != len(head_dims):
                raise ValueError(
                    f"Length of delta ({len(self.delta)}) must match number of heads ({len(head_dims)})"
                )
        else:
            raise TypeError("delta must be a float or list of floats")

    def _head_loss(self, i: int, y_pred: Tensor, y_true: Tensor, delta: float) -> Tensor:
        """Compute unweighted loss for a single head."""
        if i == self.percent_head_idx:
            norm_true = torch.norm(y_true, dim=-1, keepdim=True).clamp(min=1e-6)
            rel_error = torch.abs(y_pred - y_true) / norm_true
            return F.smooth_l1_loss(rel_error, torch.zeros_like(rel_error), beta=delta, reduction="mean")
        if i == self.logratio_head_idx:
            log_ratio = torch.abs(torch.log((torch.abs(y_pred) / y_true.clamp(min=1e-6))))
            return F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=delta, reduction="mean")
        if i == self.momentum_head_idx:
            dist = torch.norm(y_pred - y_true, dim=1)
            return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=delta, reduction="mean")
            
        if i == 1:  # scalar
            return F.smooth_l1_loss(y_pred, y_true, beta=delta, reduction="mean")

        if i == 2:  # 3D vertex — loss on Euclidean distance
            dist = torch.norm(y_pred - y_true, dim=1)
            return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=delta, reduction="mean")

        if i == 3:  # 3D direction — loss on angular distance
            dot = torch.sum(y_pred * y_true, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
            dist = torch.arccos(dot)
            return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=delta, reduction="mean")

        return F.smooth_l1_loss(y_pred, y_true, beta=delta, reduction="mean")

    def forward(self, preds: Tensor, targets: Tensor, log_vars: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        active_head_indices = [i for i, d in enumerate(self.head_dims) if d > 0]
        head_to_logvar = {head_idx: lv_idx for lv_idx, head_idx in enumerate(active_head_indices)}
        n_active = len(active_head_indices)
        single_head = n_active == 1

        total_loss = 0.0
        head_losses: Dict[str, Tensor] = {}
        start = 0

        for i, dim in enumerate(self.head_dims):
            if dim == 0:
                continue

            end = start + dim
            y_pred = preds[:, start:end]
            y_true = targets[:, start:end]

            unweighted = self._head_loss(i, y_pred, y_true, self.delta[i])
            head_losses[_head_name(i)] = unweighted.detach()

            if single_head:
                total_loss += unweighted
            else:
                sigma_i = torch.exp(log_vars[head_to_logvar[i]])
                total_loss += unweighted / sigma_i + torch.log1p((sigma_i))

            start = end

        return total_loss / n_active, head_losses

