from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from src.models.blocks import gaussian_blur_tensor

try:
    import lpips
except Exception:
    lpips = None


class BoundaryLPIPSLoss:
    def __init__(self, enabled: bool = True, max_batch: int = 4, resize: int | None = 256) -> None:
        self.enabled = enabled
        self.max_batch = max(1, int(max_batch))
        self.resize = resize if resize and resize > 0 else None
        if not enabled or lpips is None:
            self.model = None
        else:
            try:
                # LPIPS still triggers torchvision UserWarnings (pretrained) on AlexNet; narrow scope.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.model = lpips.LPIPS(net="alex", pnet_rand=True)
            except Exception:
                self.model = None

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return pred.new_tensor(0.0)
        device = pred.device
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
        self.model.eval()
        hf_pred = pred - gaussian_blur_tensor(pred, sigma=4.0)
        hf_target = target - gaussian_blur_tensor(target, sigma=4.0)
        hf_pred = (hf_pred * band_mask) * 2.0 - 1.0
        hf_target = (hf_target * band_mask) * 2.0 - 1.0
        if hf_pred.shape[0] > self.max_batch:
            hf_pred = hf_pred[: self.max_batch]
            hf_target = hf_target[: self.max_batch]
        if self.resize is not None and max(hf_pred.shape[-2:]) > self.resize:
            hf_pred = F.interpolate(hf_pred, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
            hf_target = F.interpolate(hf_target, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return self.model(hf_pred, hf_target).mean()
