from __future__ import annotations

import argparse
import json
import time

import torch

from src.models.harmonizer import SeamHarmonizerV1
from src.utils.device import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = pick_device()
    model = SeamHarmonizerV1().to(device).eval()
    x = torch.rand(args.batch_size, 5, 1024, 256, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / max(args.iters, 1)
    print(json.dumps({"device": str(device), "batch_size": args.batch_size, "latency_ms": ms}, ensure_ascii=False))


if __name__ == "__main__":
    main()
