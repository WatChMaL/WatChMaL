from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


def _huber_loss(y_pred: Tensor, y_true: Tensor, delta: float, reduction: str) -> Tensor:
    """Elementwise Huber loss."""
    return F.smooth_l1_loss(y_pred, y_true, beta=delta, reduction=reduction)


def _percent_loss(y_pred: Tensor, y_true: Tensor, delta: float, reduction: str) -> Tensor:
    """Huber loss on |pred - true| / ||true|| (relative-magnitude error)."""
    norm_true = torch.norm(y_true, dim=-1, keepdim=True).clamp(min=1e-6)
    rel_error = torch.abs(y_pred - y_true) / norm_true
    return F.smooth_l1_loss(rel_error, torch.zeros_like(rel_error), beta=delta, reduction=reduction)


def _euclidean_loss(y_pred: Tensor, y_true: Tensor, delta: float, reduction: str) -> Tensor:
    """Huber loss on Euclidean distance between pred and true vectors (e.g. vertex position, momentum)."""
    dist = torch.norm(y_pred - y_true, dim=1)
    return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=delta, reduction=reduction)


def _angular_loss(y_pred: Tensor, y_true: Tensor, delta: float, reduction: str) -> Tensor:
    """Huber loss on arccos angular distance between pred and true unit vectors (e.g. direction)."""
    dot = torch.sum(y_pred * y_true, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    dist = torch.arccos(dot)
    return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=delta, reduction=reduction)


_LOSS_REGISTRY: Dict[str, Callable[[Tensor, Tensor, float, str], Tensor]] = {
    "huber": _huber_loss,
    "percent": _percent_loss,
    "euclidean": _euclidean_loss,
    "angular": _angular_loss,
}


class MultiTaskLoss(_WeightedLoss):
    """
    Multi-head loss with learned uncertainty weighting (log_vars) for concatenated outputs.

    `head_names` and `loss_types` are independent, explicit per-head lists, indexed the same
    way as `head_dims` — the label a head is given and the loss form it uses are both stated
    directly in config rather than inferred from each other or from head position.

    Each active head contributes an unweighted loss that is then scaled by learned
    uncertainty: loss_i / sigma_i + log1p(sigma_i), where sigma_i = exp(log_var_i).
    The final loss is the mean over active heads.

    Loss types (see functions above for exact definitions):
        huber     : elementwise Huber loss
        percent   : Huber on relative-magnitude error, |pred-true| / ||true||
        euclidean : Huber on Euclidean distance between pred/true vectors (e.g. vertex position, momentum)
        angular   : Huber on arccos angular distance between pred/true unit vectors (e.g. direction)

    Args:
        head_dims  : output dimensions per head, e.g. [1, 1, 3, 3]. Use 0 to skip a head.
        head_names : label per head (any string), same length as head_dims. Used as the key
                     in the returned per-head loss dict.
        loss_types : loss type per head (one of _LOSS_REGISTRY's keys), same length as head_dims.
        delta      : Huber delta — float (shared) or list (per head).
        reduction  : 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        head_dims: List[int],
        head_names: List[str],
        loss_types: List[str],
        delta: Any = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        if len(head_names) != len(head_dims):
            raise ValueError(
                f"Length of head_names ({len(head_names)}) must match number of heads ({len(head_dims)})"
            )
        if len(loss_types) != len(head_dims):
            raise ValueError(
                f"Length of loss_types ({len(loss_types)}) must match number of heads ({len(head_dims)})"
            )
        unknown = sorted(set(loss_types) - _LOSS_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Unknown loss type(s) {unknown}; available: {sorted(_LOSS_REGISTRY)}")

        self.head_dims = head_dims
        self.head_names = list(head_names)
        self.loss_types = list(loss_types)
        self.reduction = reduction

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

    def _head_loss(self, i: int, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute unweighted loss for a single head, dispatched by its configured loss type."""
        loss_fn = _LOSS_REGISTRY[self.loss_types[i]]
        return loss_fn(y_pred, y_true, self.delta[i], self.reduction)

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

            unweighted = self._head_loss(i, y_pred, y_true)
            head_losses[self.head_names[i]] = unweighted.detach()

            if single_head:
                total_loss += unweighted
            else:
                sigma_i = torch.exp(log_vars[head_to_logvar[i]])
                total_loss += unweighted / sigma_i + torch.log1p(sigma_i)

            start = end

        return total_loss / n_active, head_losses
