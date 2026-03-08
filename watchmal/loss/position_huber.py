import torch
import torch.nn as nn


class PositionHuberLoss(nn.Module):
    """Huber-like loss on 3D position error magnitude."""

    def __init__(self, delta=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        error_vector = pred - target
        euclidean_distance = torch.linalg.vector_norm(error_vector, dim=-1)
        abs_error = euclidean_distance
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic
        loss_per_sample = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss_per_sample.mean()
