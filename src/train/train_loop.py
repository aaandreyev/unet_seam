from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses.seam_losses import SeamLossComputer
from src.metrics.seam_metrics import evaluate_batch
from src.train.ema import EMA


@dataclass
class EpochResult:
    losses: dict[str, float]
    metrics: dict[str, float]
    per_sample_metrics: list[dict[str, float]]


def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ema: EMA | None = None,
    scaler: GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    use_amp: bool = False,
    desc: str | None = None,
    tb_writer: Any = None,
    tb_prefix: str = "train",
    tb_global_step: int = 0,
    tb_log_interval: int = 1,
) -> tuple[EpochResult, int]:
    loss_computer = SeamLossComputer()
    train_mode = optimizer is not None
    model.train(train_mode)
    agg_losses: dict[str, float] = {}
    agg_metrics: dict[str, float] = {}
    per_sample_metrics: list[dict[str, float]] = []
    steps = 0
    progress = tqdm(loader, desc=desc or ("train" if train_mode else "eval"), dynamic_ncols=True, leave=False)
    for batch in progress:
        batch = _move(batch, device)
        inputs = batch["input"]
        input_rgb = batch["input_rgb"]
        target = batch["target"]
        inner_mask = batch["inner_region_mask"]
        boundary = batch["boundary_band_mask"]
        with autocast(enabled=use_amp):
            residual = model(inputs)
            pred = (input_rgb + residual).clamp(0.0, 1.0)
            pred[:, :, :, :128] = input_rgb[:, :, :, :128]
            losses = loss_computer(pred, target, input_rgb, inner_mask, boundary, residual)
        if train_mode:
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
        metrics = evaluate_batch(pred.detach(), target, input_rgb, inner_mask, boundary, residual.detach())
        per_sample_metrics.append(metrics)
        for key, value in losses.items():
            agg_losses[key] = agg_losses.get(key, 0.0) + float(value.detach().item())
        for key, value in metrics.items():
            agg_metrics[key] = agg_metrics.get(key, 0.0) + float(value)
        steps += 1
        if tb_writer is not None and train_mode and tb_log_interval > 0:
            gs = tb_global_step + steps
            if steps % tb_log_interval == 0 or steps == len(loader):
                w = float(agg_losses.get("total", 0.0) / steps)
                tb_writer.add_scalar(f"{tb_prefix}/loss/total", w, gs)
                for lk, lv in agg_losses.items():
                    if lk == "total":
                        continue
                    tb_writer.add_scalar(f"{tb_prefix}/loss/{lk}", float(lv) / steps, gs)
                for mk, mv in agg_metrics.items():
                    tb_writer.add_scalar(f"{tb_prefix}/metric/{mk}", float(mv) / steps, gs)
                if scheduler is not None and optimizer is not None:
                    lr = optimizer.param_groups[0].get("lr", 0.0)
                    tb_writer.add_scalar(f"{tb_prefix}/lr", float(lr), gs)
        progress.set_postfix(
            loss=f"{agg_losses.get('total', 0.0) / steps:.4f}",
            b_ciede=f"{agg_metrics.get('boundary_ciede2000', 0.0) / steps:.3f}",
            rel=f"{agg_metrics.get('relative_improvement', 0.0) / steps:.3f}",
        )
    progress.close()
    if steps == 0:
        return EpochResult({}, {}, []), tb_global_step
    end_step = tb_global_step + steps if train_mode else tb_global_step
    return (
        EpochResult(
            losses={key: value / steps for key, value in agg_losses.items()},
            metrics={key: value / steps for key, value in agg_metrics.items()},
            per_sample_metrics=per_sample_metrics,
        ),
        end_step,
    )
