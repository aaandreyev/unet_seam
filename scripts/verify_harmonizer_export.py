from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from src.models.harmonizer import SeamHarmonizerV3
from src.train.checkpoint import load_checkpoint


def _build_from_sidecar(sidecar: dict) -> SeamHarmonizerV3:
    arch = sidecar["architecture"]
    return SeamHarmonizerV3(
        in_channels=arch["in_channels"],
        channels=tuple(arch["channels"]),
        blocks=tuple(arch["blocks"]),
        outer_width=sidecar["strip"]["outer_width"],
        boundary_band_px=sidecar["strip"].get("boundary_band_px", 24),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export_harmonizer_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(cfg["export_root"]) / f"{cfg['model_name']}.safetensors"
    sidecar = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar["architecture"]["name"] != "seam_harmonizer_v3":
        raise RuntimeError("not a SeamHarmonizerV3 export")
    exported = _build_from_sidecar(sidecar)
    exported.load_state_dict(load_file(str(model_path), device="cpu"))
    exported.eval()
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location="cpu")
    raw = _build_from_sidecar(sidecar)
    raw.load_state_dict(ckpt["ema"])
    raw.eval()
    generator = torch.Generator().manual_seed(123)
    x = torch.rand(1, 9, 1024, 256, generator=generator)
    with torch.inference_mode():
        a = exported(x)
        b = raw(x)
    diffs = {
        key: float((a[key] - b[key]).abs().max().item())
        for key in ("corrected_strip", "corrected_inner", "gain_lowres", "gate_lowres")
    }
    max_diff = max(diffs.values())
    if max_diff >= 1e-5:
        raise RuntimeError(f"verify_harmonizer_export failed: max_diff={max_diff}, diffs={diffs}")
    print(json.dumps({"max_diff": max_diff, "diffs": diffs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
