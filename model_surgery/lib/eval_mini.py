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
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch, evaluate_harmonizer_batch_fast
from src.models.harmonizer import SeamHarmonizerV3

from model_surgery.lib.checkpoint_io import free_memory


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer_channels_blocks(
    state_dict: dict[str, torch.Tensor],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer (channels, blocks) from state_dict key shapes.

    Reads encoder.stages.{level}.{block}.norm1.weight — works for NAFEncoderLite.
    Falls back to empty tuples if the expected keys are absent.
    """
    channels: list[int] = []
    blocks: list[int] = []
    for level in range(10):
        k = f"encoder.stages.{level}.0.norm1.weight"
        if k not in state_dict:
            break
        channels.append(int(state_dict[k].shape[0]))
        n = 1
        while f"encoder.stages.{level}.{n}.norm1.weight" in state_dict:
            n += 1
        blocks.append(n)
    return tuple(channels), tuple(blocks)


def build_model(state_dict: dict[str, torch.Tensor], meta: dict[str, Any],
                device: torch.device) -> SeamHarmonizerV3:
    cfg = meta.get("config") or {}
    mcfg = cfg.get("model") or {}
    dcfg = cfg.get("dataset") or {}
    correction_limits = mcfg.get("correction_limits")
    in_channels   = int(mcfg.get("in_channels", 9))
    channels      = tuple(mcfg.get("channels", [32, 64, 128, 192]))
    blocks        = tuple(mcfg.get("blocks", [2, 2, 4, 6]))
    outer_width   = int(dcfg.get("outer_width", 128))
    boundary_band = int(dcfg.get("boundary_band_px", 24))

    def _make(ch, bl):
        return SeamHarmonizerV3(
            in_channels=in_channels, channels=ch, blocks=bl,
            outer_width=outer_width, boundary_band_px=boundary_band,
            correction_limits=correction_limits,
        ).to(device)

    try:
        model = _make(channels, blocks)
        result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        if "size mismatch" not in str(e):
            raise
        # Infer architecture from state_dict and retry
        inf_ch, inf_bl = _infer_channels_blocks(state_dict)
        if not inf_ch:
            raise RuntimeError(
                f"Cannot infer architecture from state_dict (size mismatch): {e}"
            ) from e
        print(f"[build_model] size mismatch with meta channels={channels} — "
              f"inferred channels={inf_ch}, blocks={inf_bl}")
        model = _make(inf_ch, inf_bl)
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


def preload_batches(loader: DataLoader, device: torch.device) -> list[dict]:
    """Load all batches to device once and cache them."""
    batches = []
    for batch in loader:
        batches.append({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        })
    return batches


def cache_coarse_outputs(
    model: SeamHarmonizerV3,
    batches: list[dict],
) -> list[dict]:
    """Run encoder+decoder+coarse_head ONCE per batch, cache raw low-res outputs.

    For S1 gate/limits search: correction_limits only affect `reconstruct_corrected_strip`,
    NOT the network weights. So we can run the heavy encoder/decoder once, cache the
    coarse outputs, then for each parameter combination call only the cheap reconstruction.

    Speedup for S1: ~1000× (5 full forward passes vs 1600 × 5 reconstruction-only calls).

    Returns list of dicts with keys:
      gain_lowres, gamma_lowres, bias_lowres, mix_lowres, detail_lowres, gate_lowres,
      attention_lowres, x_rgb (input strip RGB), input_rgb, target
    """
    cached: list[dict] = []
    with torch.inference_mode():
        for batch in batches:
            x = batch["input"]
            out = model(x)
            cached.append({
                # Raw coarse outputs (small tensors, cheap to store)
                "gain_lowres":      out["gain_lowres"].clone(),
                "gamma_lowres":     out["gamma_lowres"].clone(),
                "bias_lowres":      out["bias_lowres"].clone(),
                "mix_lowres":       out["mix_lowres"].clone(),
                "detail_lowres":    out["detail_lowres"].clone(),
                "gate_lowres":      out["gate_lowres"].clone(),
                "attention_lowres": out["attention_lowres"].clone(),
                # Full-res data needed for metrics
                "x_rgb":     x[:, :3].clone(),
                "input_rgb": batch["input_rgb"],
                "target":    batch["target"],
            })
    return cached


def eval_from_cache(
    cached: list[dict],
    correction_limits: dict[str, float],
    outer_width: int = 128,
) -> dict[str, float]:
    """Eval using cached coarse outputs — runs only reconstruction, no encoder/decoder.

    Uses evaluate_harmonizer_batch_fast (pure-torch, no CIEDE2000) so everything
    stays on the accelerator device.  Use quality_score() for ranking — it handles
    the absent boundary_ciede2000_16 gracefully.

    Call cache_coarse_outputs() first to build the cache.
    """
    from src.models.harmonizer import reconstruct_corrected_strip
    agg: dict[str, float] = {}
    steps = 0
    with torch.inference_mode():
        for entry in cached:
            lowres = {
                "gain_lowres":   entry["gain_lowres"],
                "gamma_lowres":  entry["gamma_lowres"],
                "bias_lowres":   entry["bias_lowres"],
                "mix_lowres":    entry["mix_lowres"],
                "detail_lowres": entry["detail_lowres"],
                "gate_lowres":   entry["gate_lowres"],
            }
            recon = reconstruct_corrected_strip(
                entry["x_rgb"], lowres, outer_width=outer_width, **correction_limits
            )
            out_for_metrics = {**recon, "attention_lowres": entry["attention_lowres"]}
            m = evaluate_harmonizer_batch_fast(
                recon["corrected_strip"], entry["input_rgb"], entry["target"],
                out_for_metrics, outer_width=outer_width,
            )
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            steps += 1
    return {k: v / steps for k, v in agg.items()} if steps > 0 else {}


def eval_with_model(
    model: SeamHarmonizerV3,
    loader: DataLoader,
    device: torch.device,
    outer_width: int = 128,
) -> dict[str, float]:
    """Run eval on an already-built model. Does NOT free the model — caller owns it."""
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
    return {k: v / steps for k, v in agg.items()} if steps > 0 else {}


def eval_with_preloaded(
    model: SeamHarmonizerV3,
    batches: list[dict],
    outer_width: int = 128,
) -> dict[str, float]:
    """Eval on pre-loaded device batches — zero H→D transfer cost.

    Use inside tight loops where the same dataset is evaluated many times
    (S2 ablation inner loop).
    """
    agg: dict[str, float] = {}
    steps = 0
    with torch.inference_mode():
        for batch in batches:
            out = model(batch["input"])
            m = evaluate_harmonizer_batch(
                out["corrected_strip"], batch["input_rgb"], batch["target"], out,
                outer_width=outer_width,
            )
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            steps += 1
    return {k: v / steps for k, v in agg.items()} if steps > 0 else {}


def run_eval(
    state_dict: dict[str, torch.Tensor],
    meta: dict[str, Any],
    loader: DataLoader,
    device: torch.device | None = None,
    outer_width: int = 128,
) -> dict[str, float]:
    """Build model, run eval, free model. Use when each eval has a different state_dict."""
    if device is None:
        device = _pick_device()
    model = build_model(state_dict, meta, device)
    result = eval_with_model(model, loader, device, outer_width)
    del model
    free_memory()
    return result


def quality_score(metrics: dict[str, float]) -> float:
    """Surgery-safe quality score.

    _quality() from train_harmonizer uses 1.0 as default for missing penalty metrics
    and float("inf") for missing CIEDE2000 — leading to inf/inf=nan when both
    boundary_ciede2000_16 and its baseline are absent (e.g. older checkpoints,
    fast-eval cache path that skips CIEDE2000).

    Surgery needs consistent relative ranking, so:
    - Penalty-only terms (overcorrection_mae etc.) → 0.0 when missing (no false penalty)
    - CIEDE2000 terms → 0.0 when missing (neutralise; ranking driven by MAE terms)
    """
    from scripts.train_harmonizer import _quality
    safe = dict(metrics)
    for key in ("overcorrection_mae", "delta_luma_profile_mae", "delta_chroma_profile_mae",
                "confidence_alignment_mae"):
        if key not in safe:
            safe[key] = 0.0
    # Avoid inf/inf = nan: neutralise CIEDE2000 terms when absent.
    # When present (full eval), they contribute normally.
    if "boundary_ciede2000_16" not in safe:
        safe["boundary_ciede2000_16"] = 0.0
        safe.setdefault("baseline_boundary_ciede2000_16", 0.0)
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
