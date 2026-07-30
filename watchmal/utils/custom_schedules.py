import torch
from torch.optim.lr_scheduler import _LRScheduler

class WSDScheduler(_LRScheduler):
    '''
    Warmup phase (linear) -> Stable phase (constant) -> Decay phase (can be whatever you want!)
    '''
    def __init__(
        self,
        optimizer,
        max_lr: float,          # set here, overrides optimizer lr
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        start_lr: float = 1e-4,
        min_lr: float = 1e-6,
        exponent: float = 2.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps  = decay_steps
        self.start_lr    = start_lr
        self.min_lr       = min_lr
        self.max_lr       = max_lr
        self.exponent     = exponent

        # override optimizer lr before super().__init__ reads base_lrs
        for group in optimizer.param_groups:
            group['lr'] = max_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        warmup_end = self.warmup_steps
        stable_end = self.warmup_steps + self.stable_steps
        decay_end  = self.warmup_steps + self.stable_steps + self.decay_steps

        if step < warmup_end:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (step / max(1, warmup_end))
        elif step < stable_end:
            lr = self.max_lr
        elif step < decay_end:
            progress = (step - stable_end) / max(1, self.decay_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1.0 - progress) ** self.exponent
        else:
            lr = self.min_lr

        return [lr for _ in self.base_lrs]
    
class StretchedCosineScheduler(_LRScheduler):
    """
    Modified OneCycle-style schedule:
    1. Linear warmup to max_lr
    2. Upper half of cosine (max_lr → mid_lr) over upper_steps  
    3. Lower half of cosine (mid_lr → min_lr) stretched over lower_steps
    
    The lower half operates over more steps than the upper half,
    spending more training time in the gentle low-LR refinement region.
    """
    def __init__(
        self,
        optimizer,
        max_lr: float,
        warmup_steps: int,
        upper_steps: int,    # steps for upper half of cosine (steep part)
        lower_steps: int,    # steps for lower half of cosine (stretched gentle part)
        start_lr: float = None,
        min_lr: float = None,
        last_epoch: int = -1,
    ):
        self.max_lr       = max_lr
        self.warmup_steps = warmup_steps
        self.upper_steps  = upper_steps
        self.lower_steps  = lower_steps
        self.start_lr     = start_lr if start_lr is not None else max_lr / 25
        self.min_lr       = min_lr   if min_lr   is not None else max_lr / 1e4
        self.mid_lr       = (self.max_lr + self.min_lr) / 2  # midpoint of cosine

        for group in optimizer.param_groups:
            group['lr'] = max_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        warmup_end = self.warmup_steps
        upper_end  = self.warmup_steps + self.upper_steps
        lower_end  = self.warmup_steps + self.upper_steps + self.lower_steps

        if step < warmup_end:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (step / max(1, warmup_end))

        elif step < upper_end:
            progress = (step - warmup_end) / max(1, self.upper_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * ((1 + torch.cos(torch.tensor(torch.pi / 2 * progress))) / 2).item()

        elif step < lower_end:
            progress = (step - upper_end) / max(1, self.lower_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * ((1 + torch.cos(torch.tensor(torch.pi / 2 + torch.pi / 2 * progress))) / 2).item()

        else:
            lr = self.min_lr

        return [lr for _ in self.base_lrs]
class WarmupExponentialScheduler(_LRScheduler):
    """
    Linear warmup to max_lr, then exponential decay to min_lr over total_steps.
    gamma is computed automatically from max_lr, min_lr, and total decay steps.
    """
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        start_lr: float = None,
        min_lr: float = None,
        last_epoch: int = -1,
    ):
        self.max_lr       = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.start_lr     = start_lr if start_lr is not None else max_lr / 25
        self.min_lr       = min_lr   if min_lr   is not None else max_lr / 1e4

        decay_steps = max(1, total_steps - warmup_steps)
        self.gamma  = (self.min_lr / self.max_lr) ** (1.0 / decay_steps)

        for group in optimizer.param_groups:
            group['lr'] = self.start_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (step / max(1, self.warmup_steps))
        elif step < self.total_steps:
            decay_step = step - self.warmup_steps
            lr = max(self.max_lr * (self.gamma ** decay_step), self.min_lr)
        else:
            lr = self.min_lr

        return [lr for _ in self.base_lrs]
    
class WarmupBlendedScheduler(_LRScheduler):
    """
    1. Linear warmup to max_lr
    2. Blended cosine + exponential decay
       - Early: mostly cosine (fast, smooth descent)
       - Late: mostly exponential (logarithmic tail)
       - Blend weight transitions smoothly via a sigmoid
       - Continuous everywhere by construction
    """
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        blend_centre: float = 0.5,   # fraction of decay steps where blend is 50/50
        blend_width: float = 0.15,   # width of sigmoid transition (smaller = sharper)
        start_lr: float = None,
        min_lr: float = None,
        last_epoch: int = -1,
    ):
        self.max_lr       = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.start_lr     = start_lr if start_lr is not None else max_lr / 25
        self.min_lr       = min_lr   if min_lr   is not None else max_lr / 1e4
        self.blend_centre = blend_centre
        self.blend_width  = blend_width

        decay_steps  = max(1, total_steps - warmup_steps)
        self.gamma   = (self.min_lr / self.max_lr) ** (1.0 / decay_steps)
        self.decay_steps = decay_steps

        for group in optimizer.param_groups:
            group['lr'] = self.start_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (step / max(1, self.warmup_steps))

        elif step < self.total_steps:
            decay_step = step - self.warmup_steps
            progress   = decay_step / self.decay_steps   # 0 → 1

            # cosine component: max_lr → min_lr over full decay
            cosine_val = self.min_lr + (self.max_lr - self.min_lr) * \
                         ((1 + torch.cos(torch.tensor(torch.pi * progress))) / 2).item()

            # exponential component: max_lr → min_lr over full decay
            exp_val = max(self.max_lr * (self.gamma ** decay_step), self.min_lr)

            # sigmoid blend: 0 = full cosine, 1 = full exponential
            blend = torch.sigmoid(torch.tensor(
                (progress - self.blend_centre) / self.blend_width
            )).item()

            lr = (1 - blend) * cosine_val + blend * exp_val
        else:
            lr = self.min_lr
        return [lr for _ in self.base_lrs]