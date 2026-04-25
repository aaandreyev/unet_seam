from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # torch.compile wraps the model in OptimizedModule (_orig_mod attribute).
        # state_dict() on a compiled model returns '_orig_mod.*' prefixed keys.
        # Always update from the underlying raw module so keys match the EMA copy.
        source = getattr(model, "_orig_mod", model)
        ema_state = self.model.state_dict()
        for key, value in source.state_dict().items():
            ema_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
