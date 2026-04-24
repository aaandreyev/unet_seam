from __future__ import annotations

import numpy as np


def bootstrap_ci(values: list[float], n_samples: int = 200, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        sample = rng.choice(arr, size=arr.shape[0], replace=True)
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
