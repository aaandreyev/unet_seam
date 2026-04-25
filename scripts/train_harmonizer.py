from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.data.gpu_corruptions import GPUCorruption
from src.data.real_strip_dataset import RealPairedStripDataset
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.models.harmonizer import SeamHarmonizerV3
from src.train.checkpoint import load_checkpoint, restore_rng_state, save_training_checkpoint
from src.train.ema import EMA
from src.train.harmonizer_loop import run_harmonizer_epoch
from src.train.scheduler import cosine_with_warmup
from src.utils.device import amp_enabled, pick_device
from src.utils.seed import seed_everything, worker_init_fn


# NAFBlockLite: pw1 maps -> (channels * expansion * 2) = 4 * channels[0] when expansion=2.
# PyTorch CUDA conv requires canUse32BitIndexMath: numel < 2^31 (fails at B=64, 128ch, 1024×256).
_NAF_INTERMEDIATE_FACTOR = 4
_INT32_MAX_ELEMS = 2_147_483_647


def _assert_batch_within_cuda_index_limit(
    ch0: int, strip_h: int, strip_w: int, train_bs: int, val_bs: int,
) -> None:
    inter_c = ch0 * _NAF_INTERMEDIATE_FACTOR
    nmax = _INT32_MAX_ELEMS
    cap = max(1, nmax // (inter_c * strip_h * strip_w))
    for name, bs in (("train", train_bs), ("val", val_bs)):
        n = bs * inter_c * strip_h * strip_w
        if n > nmax:
            raise ValueError(
                f"{name} batch_size={bs} is too large for {strip_h}×{strip_w} and channels[0]={ch0}: "
                f"first NAF block activations use {n} elements (PyTorch/CUDA index limit 2^31-1). "
                f"Use {name} batch_size <= {cap}."
            )


def _quality(metrics: dict[str, float]) -> float:
    de = metrics.get("boundary_ciede2000_16", float("inf"))
    base = metrics.get("baseline_boundary_ciede2000_16", de)
    mae = metrics.get("boundary_mae_16", 1.0)
    low = metrics.get("lowfreq_mae", 1.0)
    no_improve = max(de - base, 0.0) * 2.0
    return de + 200.0 * mae + 50.0 * low + no_improve


def _load_matching_state(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    current = module.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in current and isinstance(value, torch.Tensor) and current[key].shape == value.shape
    }
    current.update(matched)
    module.load_state_dict(current)
    return {"matched": len(matched), "skipped": len(state_dict) - len(matched)}


def _build_dataset(cfg: dict[str, Any], split: str, apply_corruption: bool = True) -> SyntheticStripDataset:
    dcfg = cfg["dataset"]
    spec = StripSpec(
        strip_height=int(dcfg.get("strip_height", 1024)),
        outer_width=int(dcfg.get("outer_width", 128)),
        inner_width=int(dcfg.get("inner_width", 128)),
        seam_jitter_px=int(dcfg.get("seam_jitter_px", 0)),
    )
    if split == "train":
        strips = int(dcfg.get("strips_per_image", 25))
    else:
        strips = int(dcfg.get("val_strips_per_image", 1))
    return SyntheticStripDataset(
        Path(dcfg["source_manifest"]),
        split=split,
        strips_per_image=strips,
        seed=int(cfg.get("seed", 42)),
        spec=spec,
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        inner_widths=[int(dcfg.get("inner_width", 128))],
        apply_corruption=apply_corruption,
    )


def _build_real_dataset(cfg: dict[str, Any], split: str) -> RealPairedStripDataset | None:
    dcfg = cfg["dataset"]
    manifest = dcfg.get("real_manifest")
    if not manifest:
        return None
    spec = StripSpec(
        strip_height=int(dcfg.get("strip_height", 1024)),
        outer_width=int(dcfg.get("outer_width", 128)),
        inner_width=int(dcfg.get("inner_width", 128)),
        seam_jitter_px=0,
    )
    return RealPairedStripDataset(
        Path(manifest),
        split=split,
        spec=spec,
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        structural_threshold=float(dcfg.get("structural_threshold", 0.6)),
    )


def main() -> None:
    if __name__ == "__main__":
        print("train_harmonizer: loading PyTorch and project code...", flush=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_harmonizer_v1.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--load-weights", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--additional-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.resume and args.load_weights:
        raise ValueError("--resume and --load-weights are mutually exclusive")
    if args.max_epochs is not None and args.additional_epochs is not None:
        raise ValueError("--max-epochs and --additional-epochs are mutually exclusive")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(cfg.get("seed", 42)))
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    synthetic_train_ds = _build_dataset(cfg, "train", apply_corruption=False)
    real_train_ds = _build_real_dataset(cfg, "train")
    train_ds = synthetic_train_ds
    sampler = None
    shuffle = True
    if real_train_ds is not None and len(real_train_ds) > 0:
        train_ds = ConcatDataset([real_train_ds, synthetic_train_ds])
        real_ratio = float(cfg["dataset"].get("real_batch_ratio", 0.8))
        weights = [real_ratio / max(len(real_train_ds), 1)] * len(real_train_ds)
        weights += [(1.0 - real_ratio) / max(len(synthetic_train_ds), 1)] * len(synthetic_train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        shuffle = False
    val_ds = _build_dataset(cfg, "val")
    real_val_ds = _build_real_dataset(cfg, "val")
    if real_val_ds is not None and len(real_val_ds) > 0:
        val_ds = ConcatDataset([real_val_ds, val_ds])
    train_cfg = cfg["train"]
    num_workers = int(train_cfg.get("num_workers", 4))
    common = {"collate_fn": collate_strip_batch, "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        **common,
    )
    val_workers = min(max(num_workers // 2, 0), 4)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=val_workers,
        worker_init_fn=worker_init_fn if val_workers > 0 else None,
        **common,
    )
    model_cfg = cfg.get("model") or {}
    model = SeamHarmonizerV3(
        in_channels=int(model_cfg.get("in_channels", 9)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int(cfg["dataset"].get("outer_width", 128)),
        boundary_band_px=int(cfg["dataset"].get("boundary_band_px", 24)),
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    dset = cfg["dataset"]
    strip_h = int(dset.get("strip_height", 1024))
    strip_w = int(dset.get("outer_width", 128)) + int(dset.get("inner_width", 128))
    train_bs = int(train_cfg["batch_size"])
    val_bs = int(train_cfg.get("val_batch_size", train_cfg["batch_size"]))
    if device.type == "cuda":
        _assert_batch_within_cuda_index_limit(model.channels[0], strip_h, strip_w, train_bs, val_bs)
    gpu_corruption = GPUCorruption().to(device) if device.type == "cuda" else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.99])),
    )
    precision = train_cfg.get("precision", "bf16")
    use_amp = amp_enabled(device, precision)
    # GradScaler only needed for fp16; bf16 has fp32-range exponent and doesn't need loss scaling
    scaler_enabled = use_amp and precision == "fp16"
    scaler = GradScaler("cuda", enabled=scaler_enabled) if device.type == "cuda" else GradScaler("cpu", enabled=False)
    base_total_epochs = int(train_cfg["num_epochs"])
    total_epochs = int(args.max_epochs if args.max_epochs is not None else base_total_epochs)
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=int(cfg.get("scheduler", {}).get("warmup_steps", 1000)),
        total_steps=max(len(train_loader) * total_epochs, 1),
        min_lr_scale=float(cfg.get("scheduler", {}).get("min_lr_scale", 0.005)),
    )
    # EMA uses deepcopy — must be created before torch.compile so it copies the raw nn.Module.
    ema = EMA(model, decay=float(cfg.get("ema", {}).get("decay", 0.999)))
    if device.type == "cuda" and hasattr(torch, "compile") and train_cfg.get("compile", True):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(json.dumps({"event": "torch_compile", "status": "ok"}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"event": "torch_compile", "status": "skipped", "reason": str(exc)}, ensure_ascii=False), flush=True)
    start_epoch = 0
    best_quality = float("inf")
    if args.resume:
        state = load_checkpoint(Path(args.resume), map_location=device.type)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        restore_rng_state(state.get("rng_state", {}))
        start_epoch = int(state["epoch"]) + 1
        best_quality = _quality(((state.get("metrics") or {}).get("val") or {}))
        if args.additional_epochs is not None:
            total_epochs = start_epoch + int(args.additional_epochs)
        print(json.dumps({"event": "resumed", "start_epoch": start_epoch}, ensure_ascii=False), flush=True)
    elif args.load_weights:
        state = load_checkpoint(Path(args.load_weights), map_location=device.type)
        model_stats = _load_matching_state(model, state["model"])
        ema_stats = _load_matching_state(ema.model, state["ema"])
        if args.additional_epochs is not None:
            total_epochs = int(args.additional_epochs)
        print(
            json.dumps(
                {
                    "event": "loaded_weights",
                    "checkpoint": args.load_weights,
                    "epochs": total_epochs,
                    "model_match": model_stats,
                    "ema_match": ema_stats,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if total_epochs <= start_epoch:
        raise ValueError(
            f"Nothing to train: start_epoch={start_epoch}, total_epochs={total_epochs}. "
            "Use --additional-epochs for resume or increase --max-epochs / train.num_epochs."
        )
    if args.resume or args.load_weights:
        scheduler = cosine_with_warmup(
            optimizer,
            warmup_steps=int(cfg.get("scheduler", {}).get("warmup_steps", 1000)),
            total_steps=max(len(train_loader) * total_epochs, 1),
            min_lr_scale=float(cfg.get("scheduler", {}).get("min_lr_scale", 0.005)),
        )
        if args.resume:
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
    loss_cfg = cfg.get("loss") or {}
    loss_computer = HarmonizerLossComputer(
        outer_width=int(cfg["dataset"].get("outer_width", 128)),
        low_sigma=float(loss_cfg.get("low_sigma", 5.0)),
        weights={k: float(v) for k, v in (loss_cfg.get("weights") or {}).items()},
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "batch_size": train_cfg["batch_size"],
                "val_batch_size": train_cfg.get("val_batch_size", train_cfg["batch_size"]),
                "epochs": total_epochs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tb_writer = None
    log_cfg = cfg.get("logging") or {}
    if log_cfg.get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(str(Path(log_cfg.get("log_dir", "outputs/logs/tensorboard_harmonizer"))))
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, total_epochs):
        outer_width = int(cfg["dataset"].get("outer_width", 128))
        train_result, global_step = run_harmonizer_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_computer,
            ema=ema,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=use_amp,
            desc=f"train e{epoch+1}/{total_epochs}",
            tb_writer=tb_writer,
            tb_global_step=global_step,
            tb_log_interval=int(log_cfg.get("log_interval", 20)),
            console_log_interval=int(log_cfg.get("console_log_interval", 25)),
            gpu_corruption=gpu_corruption,
            outer_width=outer_width,
            boundary_band_px=int(cfg["dataset"].get("boundary_band_px", 24)),
        )
        val_result, _ = run_harmonizer_epoch(
            ema.model,
            val_loader,
            None,
            device,
            loss_computer,
            use_amp=False,
            desc=f"val e{epoch+1}/{total_epochs}",
            tb_global_step=global_step,
            console_log_interval=int(log_cfg.get("console_log_interval", 25)),
        )
        quality = _quality(val_result.metrics)
        val_result.metrics["quality_score"] = quality
        if tb_writer is not None:
            for k, v in val_result.losses.items():
                tb_writer.add_scalar(f"val/loss/{k}", v, global_step)
            for k, v in val_result.metrics.items():
                tb_writer.add_scalar(f"val/metric/{k}", v, global_step)
            tb_writer.flush()
        metrics = {"train": train_result.metrics, "val": val_result.metrics}
        save_training_checkpoint(Path("outputs/checkpoints/last_harmonizer.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        if quality < best_quality:
            best_quality = quality
            save_training_checkpoint(Path("outputs/checkpoints/best_harmonizer_quality.pt"), model=model, ema_state=ema.state_dict(), optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, config=cfg, metrics=metrics)
        print(json.dumps({"event": "epoch_end", "epoch": epoch + 1, "train_loss": train_result.losses, "val_metrics": val_result.metrics}, ensure_ascii=False), flush=True)
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
