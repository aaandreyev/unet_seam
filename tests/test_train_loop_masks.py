from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.losses.seam_losses import SeamLossComputer
from src.train.train_loop import run_epoch


class _OneStripDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        del idx
        height, width, seam_x = 64, 160, 144
        input_rgb = torch.zeros(3, height, width)
        target = input_rgb.clone()
        inner = torch.zeros(1, height, width)
        inner[:, :, seam_x:] = 1.0
        boundary = torch.zeros(1, height, width)
        boundary[:, :, seam_x - 8 : seam_x + 8] = 1.0
        return {
            "input": torch.cat([input_rgb, inner, torch.zeros_like(inner)], dim=0),
            "input_rgb": input_rgb,
            "target": target,
            "inner_region_mask": inner,
            "boundary_band_mask": boundary,
        }


class _LeakyResidualModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :3], 0.1)


def test_run_epoch_preserves_full_outer_mask() -> None:
    loader = DataLoader(_OneStripDataset(), batch_size=1)
    result, _ = run_epoch(
        _LeakyResidualModel(),
        loader,
        optimizer=None,
        device=torch.device("cpu"),
        loss_computer=SeamLossComputer(lpips_enabled=False),
    )
    assert result.metrics["outer_identity_error"] == 0.0
