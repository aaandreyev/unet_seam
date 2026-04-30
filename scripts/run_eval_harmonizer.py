from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.materialized_strip_dataset import MaterializedStripDataset
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.models.harmonizer import SeamHarmonizerV3
from src.train.checkpoint import load_checkpoint
from src.train.harmonizer_loop import run_harmonizer_epoch
from src.utils.device import pick_device


def _build_dataset(train_cfg: dict[str, Any], eval_cfg: dict[str, Any]) -> SyntheticStripDataset:
    dcfg = dict(train_cfg.get("dataset") or {})
    materialized_manifest = eval_cfg.get("materialized_manifest") or dcfg.get("materialized_manifest")
    if materialized_manifest:
        return MaterializedStripDataset(
            Path(materialized_manifest),
            split="val",
            boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
            preload=bool(dcfg.get("materialized_preload", False)),
        )
    dcfg["source_manifest"] = eval_cfg.get("source_manifest", dcfg.get("source_manifest"))
    return SyntheticStripDataset(
        Path(dcfg["source_manifest"]),
        split="val",
        strips_per_image=int(eval_cfg.get("strips_per_image", 1)),
        seed=int(train_cfg.get("seed", 42)) + 10_000,
        spec=StripSpec(
            strip_height=int(dcfg.get("strip_height", 1024)),
            outer_width=int(dcfg.get("outer_width", 128)),
            inner_width=int(dcfg.get("inner_width", 128)),
            seam_jitter_px=0,
        ),
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        inner_widths=[int(dcfg.get("inner_width", 128))],
        corruption_cfg=dcfg.get("corruptions"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_harmonizer_v1.yaml")
    args = parser.parse_args()
    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = load_checkpoint(Path(eval_cfg["checkpoint"]), map_location="cpu")
    train_cfg = ckpt.get("config") or {}
    model_cfg = train_cfg.get("model") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    device = pick_device()
    model = SeamHarmonizerV3(
        in_channels=int(model_cfg.get("in_channels", 9)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int(dataset_cfg.get("outer_width", 128)),
        boundary_band_px=int(dataset_cfg.get("boundary_band_px", 24)),
        correction_limits=model_cfg.get("correction_limits"),
    ).to(device)
    try:
        result = model.load_state_dict(ckpt["ema"], strict=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Checkpoint EMA weights are incompatible with SeamHarmonizerV3 evaluation. "
            "Use a v3-trained checkpoint or retrain via --load-weights first."
        ) from exc
    if result.missing_keys or result.unexpected_keys:
        print(
            json.dumps(
                {
                    "event": "load_state_dict_partial",
                    "missing": list(result.missing_keys)[:8],
                    "unexpected": list(result.unexpected_keys)[:8],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    model.eval()
    dataset = _build_dataset(train_cfg, eval_cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_strip_batch,
        pin_memory=device.type == "cuda",
    )
    loss_computer = HarmonizerLossComputer(
        outer_width=int(dataset_cfg.get("outer_width", 128)),
        weights={k: float(v) for k, v in ((train_cfg.get("loss") or {}).get("weights") or {}).items()},
    )
    result, _ = run_harmonizer_epoch(
        model,
        loader,
        None,
        device,
        loss_computer,
        desc="eval harmonizer",
        console_log_interval=25,
    )
    report_root = Path(eval_cfg["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": eval_cfg["checkpoint"],
        "losses": result.losses,
        "metrics": result.metrics,
        "primary_metric": eval_cfg.get("primary_metric", "boundary_ciede2000_16"),
    }
    out = report_root / "summary_harmonizer.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "metrics": result.metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
