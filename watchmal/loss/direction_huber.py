import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionHuberLoss(nn.Module):
    """Huber-like loss for direction vectors using an angular proxy error."""

    def __init__(self, delta=0.1, eps=1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        pred_normalized = F.normalize(pred, p=2, dim=-1, eps=self.eps)
        target_normalized = F.normalize(target, p=2, dim=-1, eps=self.eps)
        dot_product = torch.sum(pred_normalized * target_normalized, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        one_minus_cosine = 1.0 - dot_product
        angular_error_proxy = torch.sqrt(2 * one_minus_cosine + self.eps)
        abs_error = angular_error_proxy
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()
