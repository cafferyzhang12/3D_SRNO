"""Optimizer-step warmup/cosine and an independent EMA evaluation model."""
import copy
import math

import torch


class ModelEMA:
    def __init__(self, model, decay=.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        for target, value in zip(self.model.parameters(),model.parameters(),strict=True):
            target.lerp_(value.detach(),1-self.decay)
        for target, value in zip(self.model.buffers(),model.buffers(),strict=True):
            target.copy_(value.detach())
        self.updates += 1


def warmup_cosine(optimizer, settings, updates_per_epoch):
    warmup = settings.get("warmup_epochs",2)*updates_per_epoch
    total = settings.get("schedule_epochs",settings["epochs"])*updates_per_epoch
    minimum = settings.get("min_lr_ratio",.05)
    if not 0 <= warmup < total or not 0 <= minimum <= 1:
        raise ValueError("Invalid warmup/cosine horizon or minimum learning rate")
    def factor(step):
        if step < warmup:
            return (step+1)/max(1,warmup)
        phase = min(1.,(step-warmup)/max(1,total-warmup-1))
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*phase))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
