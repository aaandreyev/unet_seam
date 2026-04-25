from __future__ import annotations

import contextlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch
from src.train.ema import EMA


@dataclass
class HarmonizerEpochResult:
    losses: dict[str, float]
    metrics: dict[str, float]
    per_sample_metrics: list[dict[str, float]]


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def run_harmonizer_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_computer: HarmonizerLossComputer,
    ema: EMA | None = None,
    scaler: GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    use_amp: bool = False,
    desc: str | None = None,
    tb_writer: Any = None,
    tb_prefix: str = "train",
    tb_global_step: int = 0,
    tb_log_interval: int = 20,
    console_log_interval: int = 25,
    gpu_corruption: torch.nn.Module | None = None,
    outer_width: int = 128,
) -> tuple[HarmonizerEpochResult, int]:
    train_mode = optimizer is not None
    model.train(train_mode)
    agg_losses: dict[str, float] = {}
    agg_metrics: dict[str, float] = {}
    per_sample_metrics: list[dict[str, float]] = []
    steps = 0
    n_batches = len(loader)
    progress = tqdm(loader, desc=desc or ("train" if train_mode else "val"), disable=not sys.stderr.isatty(), leave=False)
    t0 = time.monotonic()
    if console_log_interval > 0:
        print(json.dumps({"event": "harmonizer_iter_begin", "desc": desc, "batches": n_batches}, ensure_ascii=False), flush=True)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx_factory = autocast if device.type in {"cuda", "cpu", "mps"} else None
    for batch in progress:
        batch = _move(batch, device)
        if train_mode and gpu_corruption is not None:
            clean_strip = batch["input"][:, :3]
            inner = clean_strip[:, :, :, outer_width:]
            corrupted_inner = gpu_corruption(inner)
            corrupted_strip = torch.cat([clean_strip[:, :, :, :outer_width], corrupted_inner], dim=-1)
            batch = {**batch, "input": torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)}
        ctx = torch.inference_mode() if not train_mode else contextlib.nullcontext()
        with ctx:
            if amp_ctx_factory is not None:
                amp_ctx = amp_ctx_factory(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
            else:
                amp_ctx = contextlib.nullcontext()
            with amp_ctx:
                outputs = model(batch["input"])
                losses = loss_computer(outputs, batch["target"])
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
        with torch.inference_mode():
            metrics = evaluate_harmonizer_batch(
                outputs["corrected_strip"].detach(),
                batch["input_rgb"],
                batch["target"],
                outputs["curves"].detach(),
                outputs["shading"].detach(),
                outer_width=loss_computer.outer_width,
            )
        per_sample_metrics.append(metrics)
        steps += 1
        for key, value in losses.items():
            agg_losses[key] = agg_losses.get(key, 0.0) + float(value.detach().item())
        for key, value in metrics.items():
            agg_metrics[key] = agg_metrics.get(key, 0.0) + float(value)
        if tb_writer is not None and train_mode and (steps == 1 or steps % tb_log_interval == 0 or steps == n_batches):
            gs = tb_global_step + steps
            for key, value in agg_losses.items():
                tb_writer.add_scalar(f"{tb_prefix}/loss/{key}", value / steps, gs)
            for key, value in agg_metrics.items():
                tb_writer.add_scalar(f"{tb_prefix}/metric/{key}", value / steps, gs)
            tb_writer.flush()
        if console_log_interval > 0 and (steps == 1 or steps % console_log_interval == 0 or steps == n_batches):
            print(
                json.dumps(
                    {
                        "event": "harmonizer_step",
                        "desc": desc,
                        "step": steps,
                        "batches": n_batches,
                        "loss_total": round(agg_losses["total"] / steps, 6),
                        "mae16": round(agg_metrics["boundary_mae_16"] / steps, 6),
                        "de16": round(agg_metrics["boundary_ciede2000_16"] / steps, 4),
                        "sec": int(time.monotonic() - t0),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if steps == 0:
        return HarmonizerEpochResult({}, {}, []), tb_global_step
    return (
        HarmonizerEpochResult(
            losses={k: v / steps for k, v in agg_losses.items()},
            metrics={k: v / steps for k, v in agg_metrics.items()},
            per_sample_metrics=per_sample_metrics,
        ),
        tb_global_step + steps if train_mode else tb_global_step,
    )
