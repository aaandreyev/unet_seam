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
    console_log_interval: int = 0,
    loss_computer: SeamLossComputer | None = None,
    wall_t0: float | None = None,
    current_epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[EpochResult, int]:
    if loss_computer is None:
        loss_computer = SeamLossComputer()
    train_mode = optimizer is not None
    model.train(train_mode)
    agg_losses: dict[str, float] = {}
    agg_metrics: dict[str, float] = {}
    per_sample_metrics: list[dict[str, float]] = []
    steps = 0
    use_tqdm = sys.stderr.isatty()
    n_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(
        loader,
        desc=desc or ("train" if train_mode else "eval"),
        dynamic_ncols=True,
        leave=False,
        disable=not use_tqdm,
    )
    t_epoch_start = time.monotonic()
    if not train_mode and console_log_interval > 0 and n_batches is not None:
        vpre: dict[str, Any] = {
            "event": "val_iter_begin",
            "desc": desc,
            "batches_in_epoch": n_batches,
        }
        if wall_t0 is not None:
            vpre["sec_since_train_start"] = int(time.perf_counter() - wall_t0)
        print(json.dumps(vpre, ensure_ascii=False), flush=True)
    if train_mode and console_log_interval > 0 and n_batches is not None:
        tpre: dict[str, Any] = {
            "event": "train_iter_begin",
            "desc": desc,
            "batches_in_epoch": n_batches,
            "note": "first DataLoader batch + 1st forward/backward can take minutes; next: train_step",
        }
        if wall_t0 is not None:
            tpre["sec_since_train_start"] = int(time.perf_counter() - wall_t0)
        print(json.dumps(tpre, ensure_ascii=False), flush=True)
    _amp_device_types = frozenset({"cuda", "cpu", "mps", "hpu", "xpu", "mtia"})
    for batch in progress:
        batch = _move(batch, device)
        inputs = batch["input"]
        input_rgb = batch["input_rgb"]
        target = batch["target"]
        inner_mask = batch["inner_region_mask"]
        boundary = batch["boundary_band_mask"]
        ad = inputs.device.type
        if ad in _amp_device_types:
            amp_ctx = autocast(device_type=ad, enabled=use_amp)
        else:
            amp_ctx = contextlib.nullcontext()
        with amp_ctx:
            residual = model(inputs)
            pred = (input_rgb + residual).clamp(0.0, 1.0)
            pred[:, :, :, :128] = input_rgb[:, :, :, :128]
            losses = loss_computer(pred, target, input_rgb, inner_mask, boundary, residual)
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            did_optim_step = False
            if scaler is not None and use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                _prev = optimizer._step_count
                scaler.step(optimizer)
                scaler.update()
                did_optim_step = optimizer._step_count > _prev
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                _prev = optimizer._step_count
                optimizer.step()
                did_optim_step = optimizer._step_count > _prev
            if scheduler is not None and did_optim_step:
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
            if (
                steps == 1
                or steps % tb_log_interval == 0
                or (n_batches is not None and steps == n_batches)
            ):
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
                tb_writer.flush()
        progress.set_postfix(
            loss=f"{agg_losses.get('total', 0.0) / steps:.4f}",
            b_ciede=f"{agg_metrics.get('boundary_ciede2000', 0.0) / steps:.3f}",
            rel=f"{agg_metrics.get('relative_improvement', 0.0) / steps:.3f}",
        )
        if console_log_interval > 0:
            if steps == 1 or steps % console_log_interval == 0 or (n_batches is not None and steps == n_batches):
                gs = tb_global_step + steps
                elapsed_epoch = time.monotonic() - t_epoch_start
                eta_epoch: float | None = None
                if n_batches and steps > 0 and steps < n_batches:
                    sec_per = elapsed_epoch / steps
                    eta_epoch = sec_per * (n_batches - steps)
                out: dict[str, Any] = {
                    "event": "train_step" if train_mode else "val_step",
                    "desc": desc,
                    "epoch": current_epoch,
                    "of_epochs": total_epochs,
                    "global_step": gs,
                    "step_in_epoch": steps,
                    "batches_in_epoch": n_batches,
                    "progress_in_epoch": round(100.0 * steps / n_batches, 1) if n_batches and n_batches > 0 else None,
                    "loss_total": round(agg_losses.get("total", 0.0) / steps, 6),
                    "b_ciede": round(agg_metrics.get("boundary_ciede2000", 0.0) / steps, 4),
                }
                if wall_t0 is not None:
                    out["sec_since_train_start"] = int(time.perf_counter() - wall_t0)
                if eta_epoch is not None:
                    out["eta_sec_this_epoch"] = int(eta_epoch)
                print(json.dumps(out, ensure_ascii=False), flush=True)
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
