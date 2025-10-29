'''
Relative Huber Loss
This loss function is basically used for momentum regression.
It uses the relative error instead of the absolute error.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, input, target):
        relative_error = (input - target) / (target + self.eps)
        abs_error = torch.abs(relative_error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()
