import math

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """Kendall-style uncertainty-weighted combination of task losses."""

    def __init__(self, task_losses: dict, initial_variances: dict = None):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.tasks = list(task_losses.keys())
        num_tasks = len(self.tasks)
        if initial_variances is None:
            initial_log_vars = torch.zeros(num_tasks)
        else:
            initial_log_vars = torch.tensor([
                math.log(initial_variances.get(task, 1.0) + 1e-8) for task in self.tasks
            ])
        self.log_vars = nn.Parameter(initial_log_vars)

    def forward(self, preds: dict, targets: dict):
        total_loss = 0
        log_dict = {}
        for i, task in enumerate(self.tasks):
            pred = preds[task]
            target = targets[task]
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            base_loss = self.task_losses[task](pred, target)
            task_loss = precision * base_loss + log_var
            total_loss += task_loss
            log_dict[f"loss_{task}"] = base_loss
            log_dict[f"variance_{task}"] = torch.exp(log_var)
        return total_loss, log_dict
