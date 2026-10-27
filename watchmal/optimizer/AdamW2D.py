"""
AdamW variant that automatically exempts biases and normalisation parameters from weight decay.
"""

import torch


class AdamW2D(torch.optim.AdamW):
    """AdamW with automatic weight-decay exemption for biases and norm parameters.

    When a plain parameter iterable is passed, parameters are split into two groups:

    - ``ndim >= 2`` (weight matrices, conv kernels): receive the configured ``weight_decay``.
    - ``ndim < 2``  (biases, BatchNorm/LayerNorm scale and shift): ``weight_decay`` is forced
      to 0.0, because decaying these parameters introduces an incorrect prior and can hurt
      normalisation layers.

    If explicit param groups (list of dicts) are already supplied, they are forwarded unchanged
    so that callers retain full control.

    Usage in config
    ---------------
    Replace ``_target_: torch.optim.AdamW`` with
    ``_target_: watchmal.optimizer.AdamW2D.AdamW2D``.  No other changes are needed.
    """

    def __init__(self, params, **kwargs):
        params = list(params)
        decay    = [p for p in params if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in params if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": decay}]
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0})
        super().__init__(param_groups, **kwargs)

