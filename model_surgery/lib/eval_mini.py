"""Fast mini-eval for cyclic surgery loop.

Uses a fixed reproducible subset of the val split.
On M1 MPS ~10s for 60 strips with batch_size=4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch
from src.models.harmonizer import SeamHarmonizerV3

from model_surgery.lib.checkpoint_io import free_memory


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(state_dict: dict[str, torch.Tensor], meta: dict[str, Any],
                device: torch.device) -> SeamHarmonizerV3:
    cfg = meta.get("config") or {}
    mcfg = cfg.get("model") or {}
    dcfg = cfg.get("dataset") or {}
    correction_limits = mcfg.get("correction_limits")
    model = SeamHarmonizerV3(
        in_channels=int(mcfg.get("in_channels", 9)),
        channels=tuple(mcfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(mcfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int(dcfg.get("outer_width", 128)),
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        correction_limits=correction_limits,
    ).to(device)
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in state_dict: {result.unexpected_keys[:5]}")
    model.eval()
    return model


def build_loader(manifest: Path, n_strips: int, outer_width: int, inner_width: int,
                 strip_height: int, boundary_band_px: int, batch_size: int,
                 seed: int, materialized_dir: Path | None = None) -> DataLoader:
    """Build eval DataLoader.

    If materialized_dir is provided and contains a manifest.jsonl, uses
    MaterializedStripDataset (fast, no on-the-fly corruption) for reproducible eval.
    Falls back to SyntheticStripDataset otherwise.
    """
    if materialized_dir is not None:
        mat_manifest = Path(materialized_dir) / "manifest.jsonl"
        if mat_manifest.exists():
            from src.data.materialized_strip_dataset import MaterializedStripDataset
            ds = MaterializedStripDataset(
                mat_manifest,
                split=None,  # use all splits
                boundary_band_px=boundary_band_px,
                preload=False,
            )
            n = min(n_strips, len(ds))
            rng = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
            subset = Subset(ds, indices)
            return DataLoader(subset, batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_strip_batch)

    spec = StripSpec(strip_height=strip_height, outer_width=outer_width,
                     inner_width=inner_width, seam_jitter_px=0)
    ds = SyntheticStripDataset(
        manifest, split="val", strips_per_image=1,
        seed=seed + 7919, spec=spec,
        boundary_band_px=boundary_band_px,
        inner_widths=[inner_width],
        apply_corruption=True,
    )
    n = min(n_strips, len(ds))
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
    subset = Subset(ds, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_strip_batch)


def run_eval(
    state_dict: dict[str, torch.Tensor],
    meta: dict[str, Any],
    loader: DataLoader,
    device: torch.device | None = None,
    outer_width: int = 128,
) -> dict[str, float]:
    """Run eval on loader, return metric dict. Frees model from device after."""
    if device is None:
        device = _pick_device()
    model = build_model(state_dict, meta, device)
    loss_computer = HarmonizerLossComputer(outer_width=outer_width)
    agg: dict[str, float] = {}
    steps = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["input"])
            m = evaluate_harmonizer_batch(
                out["corrected_strip"], batch["input_rgb"], batch["target"], out,
                outer_width=outer_width,
            )
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            steps += 1
    del model
    free_memory()
    return {k: v / steps for k, v in agg.items()} if steps > 0 else {}


def quality_score(metrics: dict[str, float]) -> float:
    """Surgery-safe quality score.

    _quality() from train_harmonizer uses 1.0 as default for missing metrics like
    overcorrection_mae, delta_luma_profile_mae, delta_chroma_profile_mae — this
    massively penalises older checkpoints that weren't evaluated for those metrics.
    Surgery needs consistent relative ranking, so we substitute 0.0 for missing
    penalty-only metrics (no penalty when unknown, rather than maximum penalty).
    """
    from scripts.train_harmonizer import _quality
    safe = dict(metrics)
    # These terms scale linearly and default to 1.0 in _quality() — use 0 when missing.
    for key in ("overcorrection_mae", "delta_luma_profile_mae", "delta_chroma_profile_mae",
                "confidence_alignment_mae"):
        if key not in safe:
            safe[key] = 0.0
    return _quality(safe)


def metrics_summary(metrics: dict[str, float]) -> str:
    de = metrics.get("boundary_ciede2000_16", float("nan"))
    mae = metrics.get("boundary_mae_16", float("nan"))
    conf = metrics.get("confidence_mean", float("nan"))
    low = metrics.get("lowfreq_mae", float("nan"))
    overcorr = metrics.get("overcorrection_mae", float("nan"))
    q = quality_score(metrics)
    return (f"ΔE={de:.3f} mae={mae:.4f} conf={conf:.3f} "
            f"low={low:.4f} overcorr={overcorr:.4f} Q={q:.2f}")
