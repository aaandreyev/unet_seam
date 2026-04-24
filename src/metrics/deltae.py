from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def boundary_ciede2000(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred_lab = rgb2lab(pred.clip(0.0, 1.0))
    target_lab = rgb2lab(target.clip(0.0, 1.0))
    delta = deltaE_ciede2000(pred_lab, target_lab)
    valid = mask.squeeze() > 0.5
    return float(delta[valid].mean()) if np.any(valid) else 0.0
