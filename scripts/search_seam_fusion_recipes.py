from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.materialized_strip_dataset import MaterializedStripDataset
from src.data.synthetic_strip_dataset import collate_strip_batch
from src.infer.cv_mask_harmonize import harmonize_by_mask_torch
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch, evaluate_harmonizer_batch_fast
from src.models.harmonizer import SeamHarmonizerV3
from src.train.checkpoint import load_checkpoint


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[SeamHarmonizerV3, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model") or {}
    dataset_cfg = cfg.get("dataset") or {}
    model = SeamHarmonizerV3(
        in_channels=int(model_cfg.get("in_channels", 9)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int(dataset_cfg.get("outer_width", 128)),
        boundary_band_px=int(dataset_cfg.get("boundary_band_px", 24)),
        correction_limits=model_cfg.get("correction_limits"),
    ).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, cfg


def _candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ident = 0

    def add(family: str, **params: Any) -> None:
        nonlocal ident
        ident += 1
        candidates.append({"id": ident, "family": family, **params})

    # 144 recipes: ML + low-frequency residual from (cv - ml)
    for sigma, corridor, scale, power in itertools.product((4.0, 8.0, 12.0, 16.0), (8.0, 16.0, 24.0, 32.0), (0.25, 0.5, 0.75), (0.75, 1.0, 1.5)):
        add("residual_mlcv", sigma=sigma, corridor=corridor, scale=scale, power=power)
    # 48 recipes: ML + low-frequency residual from (cv - input)
    for sigma, corridor, scale, power in itertools.product((4.0, 8.0, 12.0, 16.0), (8.0, 16.0, 24.0, 32.0), (0.25, 0.5, 0.75), (0.75,)):
        add("residual_inputcv", sigma=sigma, corridor=corridor, scale=scale, power=power)
    # 36 recipes: direct local mix near seam
    for corridor, cv_weight, power in itertools.product((4.0, 8.0, 16.0, 24.0), (0.1, 0.2, 0.35), (0.75, 1.0, 1.5)):
        add("local_rgb_mix", corridor=corridor, cv_weight=cv_weight, power=power)
    return candidates


def _gaussian_blur_cpu(x: torch.Tensor, sigma: float) -> torch.Tensor:
    from scipy.ndimage import gaussian_filter

    arr = x.detach().cpu().numpy().astype("float32")
    blurred = gaussian_filter(arr, sigma=(0.0, 0.0, sigma, sigma), mode="nearest")
    return torch.from_numpy(blurred).to(dtype=x.dtype)


def _seam_weight(mask: torch.Tensor, corridor: float, power: float) -> torch.Tensor:
    from scipy.ndimage import distance_transform_edt

    weights = []
    for i in range(mask.shape[0]):
        dist_in = distance_transform_edt(mask[i, 0].detach().cpu().numpy() > 0.5)
        w = torch.from_numpy(dist_in).unsqueeze(0).unsqueeze(0).to(dtype=mask.dtype)
        w = torch.exp(-0.5 * torch.square(w / max(corridor, 1e-6))).clamp(0.0, 1.0)
        weights.append(w.pow(power))
    return torch.cat(weights, dim=0)


def _apply_candidate(candidate: dict[str, Any], input_rgb: torch.Tensor, ml_rgb: torch.Tensor, cv_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    seam_w = _seam_weight(mask, float(candidate["corridor"]), float(candidate["power"]))
    if candidate["family"] == "residual_mlcv":
        residual = _gaussian_blur_cpu(cv_rgb - ml_rgb, float(candidate["sigma"]))
        return ml_rgb + residual * seam_w * float(candidate["scale"]) * mask
    if candidate["family"] == "residual_inputcv":
        residual = _gaussian_blur_cpu(cv_rgb - input_rgb, float(candidate["sigma"]))
        return ml_rgb + residual * seam_w * float(candidate["scale"]) * mask
    if candidate["family"] == "local_rgb_mix":
        cv_weight = seam_w * float(candidate["cv_weight"])
        return ml_rgb * (1.0 - cv_weight) + cv_rgb * cv_weight
    raise ValueError(f"unsupported family: {candidate['family']}")


def _coarse_score(metrics: dict[str, float]) -> float:
    return (
        metrics["boundary_mae_16"] * 4.0
        + metrics["lowfreq_mae"] * 2.0
        + metrics["delta_luma_profile_mae"] * 1.5
        + metrics["overcorrection_mae"] * 1.0
    )


def _full_score(metrics: dict[str, float]) -> float:
    return (
        metrics["boundary_ciede2000_16"] * 1.5
        + metrics["boundary_mae_16"] * 6.0
        + metrics["lowfreq_mae"] * 2.5
        + metrics["delta_luma_profile_mae"] * 2.0
        + metrics["overcorrection_mae"] * 1.5
    )


def _aggregate(sums: dict[int, dict[str, float]], counts: dict[int, int]) -> list[dict[str, Any]]:
    rows = []
    for ident, metric_sums in sums.items():
        n = max(counts.get(ident, 1), 1)
        avg = {k: v / n for k, v in metric_sums.items()}
        rows.append({"id": ident, "metrics": avg})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Search many seam fusion recipes on canonical training strips.")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best_harmonizer_quality.pt")
    ap.add_argument("--manifest", default="outputs/synthetic_triplets_100/manifest.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--rerank-topk", type=int, default=16)
    ap.add_argument("--out", default="outputs/fusion_search/fusion_recipe_search.json")
    ap.add_argument("--cv-blur-sigma", type=float, default=20.0)
    ap.add_argument("--cv-strip-width", type=int, default=8)
    ap.add_argument("--cv-falloff", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(Path(args.checkpoint), device)
    dataset = MaterializedStripDataset(Path(args.manifest), split=args.split, boundary_band_px=int((cfg.get("dataset") or {}).get("boundary_band_px", 24)), preload=False)
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_strip_batch)

    candidates = _candidate_grid()
    sample_count = len(dataset)
    batch_count = len(loader)
    print(
        json.dumps(
            {
                "event": "fusion_search_begin",
                "split": args.split,
                "samples": sample_count,
                "batches": batch_count,
                "candidate_count": len(candidates),
                "rerank_topk": args.rerank_topk,
                "device": device.type,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    coarse_sums: dict[int, dict[str, float]] = {c["id"]: {} for c in candidates}
    coarse_counts: dict[int, int] = {c["id"]: 0 for c in candidates}

    coarse_bar = tqdm(loader, desc="fusion coarse", dynamic_ncols=True, leave=True)
    for batch_idx, batch in enumerate(coarse_bar, start=1):
        model_in = batch["input"].to(device)
        with torch.inference_mode():
            out = model(model_in)
        ml_rgb = out["corrected_strip"].detach().cpu()
        input_rgb = batch["input_rgb"].cpu()
        target = batch["target"].cpu()
        mask = batch["mask"].cpu()
        cv_rgb = harmonize_by_mask_torch(
            input_rgb.permute(0, 2, 3, 1),
            mask[:, 0],
            mode="inside",
            strip_width=args.cv_strip_width,
            blur_sigma=args.cv_blur_sigma,
            falloff=args.cv_falloff,
            correction_strength=1.0,
            luminance_strength=1.0,
            chroma_strength=1.0,
            mask_threshold=0.5,
            max_workdim=1024,
        ).permute(0, 3, 1, 2).cpu()

        for candidate in candidates:
            fused = _apply_candidate(candidate, input_rgb, ml_rgb, cv_rgb, mask)
            metrics = evaluate_harmonizer_batch_fast(fused, input_rgb, target, out, outer_width=128)
            acc = coarse_sums[candidate["id"]]
            for key, value in metrics.items():
                acc[key] = acc.get(key, 0.0) + float(value)
            coarse_counts[candidate["id"]] += 1
        coarse_bar.set_postfix(
            samples=min(batch_idx * args.batch_size, sample_count),
            recipes=len(candidates),
            stage="coarse",
        )

    coarse_rows = _aggregate(coarse_sums, coarse_counts)
    coarse_index = {row["id"]: row["metrics"] for row in coarse_rows}
    ranked = sorted(candidates, key=lambda c: _coarse_score(coarse_index[c["id"]]))
    topk = ranked[: args.rerank_topk]
    print(
        json.dumps(
            {
                "event": "fusion_search_coarse_done",
                "evaluated_candidates": len(candidates),
                "rerank_topk": len(topk),
                "best_coarse_candidate": ranked[0]["id"] if ranked else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    full_sums: dict[int, dict[str, float]] = {c["id"]: {} for c in topk}
    full_counts: dict[int, int] = {c["id"]: 0 for c in topk}
    full_bar = tqdm(loader, desc="fusion rerank", dynamic_ncols=True, leave=True)
    for batch_idx, batch in enumerate(full_bar, start=1):
        model_in = batch["input"].to(device)
        with torch.inference_mode():
            out = model(model_in)
        ml_rgb = out["corrected_strip"].detach().cpu()
        input_rgb = batch["input_rgb"].cpu()
        target = batch["target"].cpu()
        mask = batch["mask"].cpu()
        cv_rgb = harmonize_by_mask_torch(
            input_rgb.permute(0, 2, 3, 1),
            mask[:, 0],
            mode="inside",
            strip_width=args.cv_strip_width,
            blur_sigma=args.cv_blur_sigma,
            falloff=args.cv_falloff,
            correction_strength=1.0,
            luminance_strength=1.0,
            chroma_strength=1.0,
            mask_threshold=0.5,
            max_workdim=1024,
        ).permute(0, 3, 1, 2).cpu()
        for candidate in topk:
            fused = _apply_candidate(candidate, input_rgb, ml_rgb, cv_rgb, mask)
            metrics = evaluate_harmonizer_batch(fused, input_rgb, target, out, outer_width=128)
            acc = full_sums[candidate["id"]]
            for key, value in metrics.items():
                acc[key] = acc.get(key, 0.0) + float(value)
            full_counts[candidate["id"]] += 1
        full_bar.set_postfix(
            samples=min(batch_idx * args.batch_size, sample_count),
            topk=len(topk),
            stage="rerank",
        )

    full_rows = _aggregate(full_sums, full_counts)
    full_index = {row["id"]: row["metrics"] for row in full_rows}
    best = min(topk, key=lambda c: _full_score(full_index[c["id"]]))

    report = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "split": args.split,
        "limit": args.limit,
        "candidate_count": len(candidates),
        "rerank_topk": args.rerank_topk,
        "best_candidate": {**best, "metrics": full_index[best["id"]], "full_score": _full_score(full_index[best["id"]])},
        "top_coarse": [{**c, "metrics": coarse_index[c["id"]], "coarse_score": _coarse_score(coarse_index[c["id"]])} for c in ranked[: min(24, len(ranked))]],
        "top_full": [
            {**c, "metrics": full_index[c["id"]], "full_score": _full_score(full_index[c["id"]])}
            for c in sorted(topk, key=lambda c: _full_score(full_index[c["id"]]))
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "fusion_search_done", "report": str(out_path), "best_candidate": report["best_candidate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
