from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from src.models.resunet import SeamResUNet
from src.train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(cfg["export_root"]) / f"{cfg['model_name']}.safetensors"
    sidecar = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    a = sidecar["architecture"]
    state = load_file(str(model_path), device="cpu")
    model = SeamResUNet(
        in_channels=a["in_channels"],
        out_channels=a["out_channels"],
        base_channels=a["base_channels"],
        residual_cap=a["residual_cap_tanh_scale"],
    )
    model.load_state_dict(state)
    model.eval()
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location="cpu")
    raw_model = SeamResUNet()
    raw_model.load_state_dict(ckpt["ema"])
    x = torch.rand(1, 5, 1024, 256)
    diff = (model(x) - raw_model(x)).abs().max().item()
    if diff >= 1e-5:
        raise RuntimeError(f"verify_export failed: max diff={diff}")
    print({"max_diff": diff})


if __name__ == "__main__":
    main()
