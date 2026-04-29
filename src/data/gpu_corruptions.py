from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.data.corruptions import apply_random_corruptions


class GPUCorruption(nn.Module):
    """CUDA wrapper that keeps training corruption behavior aligned with SyntheticStripDataset.

    This now delegates to the same procedural corruption engine used by CPU/export code,
    so cached triplets and GPU training samples follow the same corruption family.
    """

    def __init__(self, p_c: float = 0.5, p_d: float = 0.2) -> None:
        super().__init__()
        self.p_c = p_c
        self.p_d = p_d

    @torch.no_grad()
    def forward(self, inner: Tensor, gen: torch.Generator | None = None) -> Tensor:
        outputs = []
        for sample in inner.split(1, dim=0):
            result = apply_random_corruptions(sample, gen if gen is not None else torch.Generator(device=sample.device))
            outputs.append(result.image)
        return torch.cat(outputs, dim=0).clamp(0.0, 1.0)
