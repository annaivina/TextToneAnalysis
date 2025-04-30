import torch
import math



class WarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, hold_steps, start_lr, target_lr, alpha=0.0, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.alpha = alpha
        self.last_epoch = last_epoch
        super(WarmupCosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
          step_ratio = step / self.warmup_steps
          lr = self.start_lr + 0.5 * (self.target_lr - self.start_lr) * (1 - math.cos(math.pi * step_ratio))
        elif step < self.warmup_steps + self.hold_steps:
          lr = self.target_lr
        else:
          decay_steps = self.total_steps - self.warmup_steps - self.hold_steps
          decay_progress = (step - self.warmup_steps - self.hold_steps) / decay_steps
          cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
          lr = self.target_lr * ((1 - self.alpha) * cosine_decay + self.alpha)

        return [lr for _ in self.optimizer.param_groups]