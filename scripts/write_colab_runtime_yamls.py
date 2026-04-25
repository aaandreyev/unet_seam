#!/usr/bin/env python3
"""Write colab/runtime_configs/{train,eval,export}.yaml. Used by the Colab notebook (cell 7 and as fallback before TRAIN)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--train-batch-size", type=int, required=True)
    ap.add_argument("--val-batch-size", type=int, required=True)
    ap.add_argument("--train-epochs", type=int, required=True)
    ap.add_argument("--train-num-workers", type=int, required=True)
    ap.add_argument("--primary-checkpoint", type=str, default="best_boundary_ciede2000.pt")
    args = ap.parse_args()
    pr: Path = args.project_root
    dr: Path = args.data_root
    cfg_dir = pr / "runtime_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    local_outputs = pr / "outputs"
    local_ckpt = local_outputs / "checkpoints"
    local_eval = local_outputs / "eval_reports"
    local_export = local_outputs / "exports"
    train_cfg = yaml.safe_load((pr / "configs" / "train_synth_v1.yaml").read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((pr / "configs" / "eval_v1.yaml").read_text(encoding="utf-8"))
    export_cfg = yaml.safe_load((pr / "configs" / "export_v1.yaml").read_text(encoding="utf-8"))
    train_cfg["dataset"]["cache_root"] = str(dr / "outputs" / "strip_cache")
    train_cfg["dataset"]["train_cache_manifest"] = str(dr / "manifests" / "strip_train_cache.jsonl")
    train_cfg["dataset"]["val_cache_manifest"] = str(dr / "manifests" / "strip_val_cache.jsonl")
    train_cfg["train"]["batch_size"] = args.train_batch_size
    train_cfg["train"]["val_batch_size"] = args.val_batch_size
    train_cfg["train"]["num_epochs"] = args.train_epochs
    train_cfg["train"]["num_workers"] = args.train_num_workers
    eval_cfg["checkpoint"] = str(local_ckpt / args.primary_checkpoint)
    eval_cfg["report_root"] = str(local_eval)
    eval_cfg["batch_size"] = args.val_batch_size
    eval_cfg["cache_root"] = str(dr / "outputs" / "strip_cache")
    eval_cfg["val_cache_manifest"] = str(dr / "manifests" / "strip_val_cache.jsonl")
    export_cfg["checkpoint"] = str(local_ckpt / args.primary_checkpoint)
    export_cfg["export_root"] = str(local_export)
    (cfg_dir / "train.yaml").write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    (cfg_dir / "eval.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")
    (cfg_dir / "export.yaml").write_text(yaml.safe_dump(export_cfg, sort_keys=False), encoding="utf-8")
    print("Wrote", cfg_dir / "train.yaml", cfg_dir / "eval.yaml", cfg_dir / "export.yaml", sep=" ")


if __name__ == "__main__":
    main()
