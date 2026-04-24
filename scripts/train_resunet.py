from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.cached_strip_dataset import CachedStripDataset
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.models.resunet import SeamResUNet
from src.train.checkpoint import load_checkpoint, restore_rng_state, save_training_checkpoint
from src.train.ema import EMA
from src.train.scheduler import cosine_with_warmup
from src.train.train_loop import run_epoch
from src.utils.device import amp_enabled, pick_device
from src.utils.seed import seed_everything, worker_init_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_synth_v1.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = pick_device()
    num_epochs = args.max_epochs or cfg["train"]["num_epochs"]
    print(
        json.dumps({"device": str(device), "epochs": num_epochs, "batch_size": cfg["train"]["batch_size"]}, ensure_ascii=False),
        flush=True,
    )
    cache_root = Path(cfg["dataset"].get("cache_root", "outputs/strip_cache"))
    train_cache_manifest = Path(cfg["dataset"].get("train_cache_manifest", "manifests/strip_train_cache.jsonl"))
    val_cache_manifest = Path(cfg["dataset"].get("val_cache_manifest", "manifests/strip_val_cache.jsonl"))
    if cache_root.exists() and train_cache_manifest.exists() and val_cache_manifest.exists():
        train_ds = CachedStripDataset(train_cache_manifest, cache_root)
        val_ds = CachedStripDataset(val_cache_manifest, cache_root)
    else:
        train_ds = SyntheticStripDataset(Path(cfg["dataset"]["source_manifest"]), split="train", strips_per_image=cfg["dataset"]["strips_per_image"])
        val_ds = SyntheticStripDataset(Path(cfg["dataset"]["source_manifest"]), split="val", strips_per_image=max(1, cfg["dataset"]["strips_per_image"] // 4))
    train_workers = int(cfg["train"]["num_workers"])
    val_workers = min(max(train_workers // 2, 0), 8)
    common_loader = {
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_strip_batch,
    }
    if train_workers > 0:
        common_loader["persistent_workers"] = True
        common_loader["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=train_workers, worker_init_fn=worker_init_fn, **common_loader)
    val_loader = DataLoader(val_ds, batch_size=max(1, cfg["train"]["batch_size"] // 2), shuffle=False, num_workers=val_workers, worker_init_fn=worker_init_fn if val_workers > 0 else None, **common_loader)
    model = SeamResUNet(residual_mode=cfg["model"]["residual_mode"], low_freq_sigma=cfg["model"]["low_freq_sigma"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], betas=tuple(cfg["train"]["betas"]))
    ema = EMA(model, decay=cfg["ema"]["decay"])
    scaler = GradScaler(enabled=amp_enabled(device, cfg["train"]["precision"]))
    total_steps = max(len(train_loader) * num_epochs, 1)
    scheduler = cosine_with_warmup(optimizer, warmup_steps=cfg["scheduler"]["warmup_steps"], total_steps=total_steps)
    start_epoch = 0
    best = {
        "boundary_ciede2000": float("inf"),
        "boundary_mae": float("inf"),
        "outer_identity_error": float("inf"),
        "relative_improvement": float("-inf"),
    }
    if args.resume:
        state = load_checkpoint(Path(args.resume), map_location=device.type)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(json.dumps({"event": "resumed", "start_epoch": start_epoch}, ensure_ascii=False), flush=True)
    log_cfg: dict = cfg.get("logging") or {}
    console_iv = int(log_cfg.get("console_log_interval", 25))
    print(
        json.dumps(
            {
                "event": "train_start",
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "batches_per_epoch": len(train_loader),
                "val_batches_per_epoch": len(val_loader),
                "num_epochs": num_epochs,
                "start_epoch": start_epoch,
                "end_epoch_excl": num_epochs,
                "console_log_interval": console_iv,
                "note": "In Colab subprocess tqdm is off; use these JSON lines + TensorBoard.",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tb_on = bool(log_cfg.get("tensorboard", True))
    log_dir: Path = Path(log_cfg.get("log_dir", "outputs/logs/tensorboard"))
    log_interval = int(log_cfg.get("log_interval", 20))
    tb_writer: Any = None
    if tb_on:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(str(log_dir))
            print(json.dumps({"tensorboard_logdir": str(log_dir.resolve())}, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"tensorboard": "disabled", "reason": str(e)}, ensure_ascii=False), flush=True)
    global_step = 0
    epoch_use_tqdm = sys.stderr.isatty()
    epoch_bar = tqdm(
        range(start_epoch, num_epochs),
        desc="epochs",
        dynamic_ncols=True,
        disable=not epoch_use_tqdm,
    )
    for epoch in epoch_bar:
        epoch_start = time.time()
        print(json.dumps({"event": "epoch_begin", "epoch": epoch + 1, "of": num_epochs}, ensure_ascii=False), flush=True)
        train_result, global_step = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ema=ema,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=scaler.is_enabled(),
            desc=f"train e{epoch+1}/{num_epochs}",
            tb_writer=tb_writer,
            tb_prefix="train",
            tb_global_step=global_step,
            tb_log_interval=max(1, log_interval),
            console_log_interval=console_iv,
        )
        val_result, _ = run_epoch(
            ema.model,
            val_loader,
            None,
            device,
            use_amp=False,
            desc=f"val e{epoch+1}/{num_epochs}",
            console_log_interval=0,
        )
        if tb_writer is not None:
            for k, v in val_result.losses.items():
                tb_writer.add_scalar(f"val/loss_{k}", v, global_step)
            for k, v in val_result.metrics.items():
                tb_writer.add_scalar(f"val/metric_{k}", v, global_step)
            tb_writer.add_scalar("epoch", float(epoch), global_step)
            tb_writer.flush()
        metrics = {"train": train_result.metrics, "val": val_result.metrics}
        save_training_checkpoint(Path("outputs/checkpoints/last.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        if val_result.metrics.get("boundary_ciede2000", float("inf")) < best["boundary_ciede2000"]:
            best["boundary_ciede2000"] = val_result.metrics["boundary_ciede2000"]
            save_training_checkpoint(Path("outputs/checkpoints/best_boundary_ciede2000.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        if val_result.metrics.get("boundary_mae", float("inf")) < best["boundary_mae"]:
            best["boundary_mae"] = val_result.metrics["boundary_mae"]
            save_training_checkpoint(Path("outputs/checkpoints/best_boundary_mae.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        if val_result.metrics.get("outer_identity_error", float("inf")) < best["outer_identity_error"]:
            best["outer_identity_error"] = val_result.metrics["outer_identity_error"]
            save_training_checkpoint(Path("outputs/checkpoints/best_outer_identity.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        if val_result.metrics.get("relative_improvement", float("-inf")) > best["relative_improvement"]:
            best["relative_improvement"] = val_result.metrics["relative_improvement"]
            save_training_checkpoint(Path("outputs/checkpoints/best_relative_improvement.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        epoch_bar.set_postfix(
            sec=f"{time.time()-epoch_start:.1f}",
            val_b=f"{val_result.metrics.get('boundary_ciede2000', 0.0):.3f}",
            rel=f"{val_result.metrics.get('relative_improvement', 0.0):.3f}",
        )
        print(
            json.dumps(
                {
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "sec": round(time.time() - epoch_start, 1),
                    "train_loss": train_result.losses,
                    "val_metrics": val_result.metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
