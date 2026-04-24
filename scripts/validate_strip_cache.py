"""
Проверка материализованного strip-cache.

По умолчанию (файлы):
  — каждая строка train/val manifest: папка существует, все обязательные файлы, meta.id;
  — число папок в train/val совпадает с manifest;
  — нет «лишних» папок (id на диске, которого нет в manifest).

Опционально (тензоры, чтение PNG):
  --max-per-split N: случайные N сэмплов на split;
  --full-tensor: все сэмплы (долго, ~десятки минут на ~14k сэмплов).

Пример:
  python -m scripts.validate_strip_cache \\
    --cache-root outputs/strip_cache_final \\
    --train-manifest manifests/strip_train_cache.jsonl \\
    --val-manifest manifests/strip_val_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.data.cached_strip_dataset import CachedStripDataset
from src.data.manifest import read_jsonl

_REQUIRED_NAMES = frozenset({"input.png", "target.png", "mask.png", "distance.png", "meta.json"})


def _check_sample(ds: CachedStripDataset, idx: int, outer_px: int = 128) -> list[str]:
    err: list[str] = []
    s = ds[idx]
    inp = s["input_rgb"]
    tgt = s["target"]
    m = s["inner_region_mask"]
    if not torch.isfinite(inp).all() or not torch.isfinite(tgt).all():
        err.append(f"idx={idx} non-finite rgb")
    if inp.shape[0] != 3 or tgt.shape[0] != 3:
        err.append(f"idx={idx} bad ch {inp.shape}")
    h, w = inp.shape[-2], inp.shape[-1]
    if h != 1024 or w < outer_px + 64:
        err.append(f"idx={idx} unexpected HxW {h}x{w}")
    outer_mae = (inp[..., :outer_px] - tgt[..., :outer_px]).abs().mean().item()
    if outer_mae > 1e-5:
        err.append(f"idx={idx} outer not byte-consistent mae={outer_mae:.6f}")
    if m.min() < -0.01 or m.max() > 1.01:
        err.append(f"idx={idx} mask out of [0,1]")
    return err


def _validate_all_files(
    cache: Path,
    name: str,
    man: Path,
    strict_only_required: bool,
) -> list[str]:
    out: list[str] = []
    rows = read_jsonl(man)
    manifest_ids = {r["id"] for r in rows}
    split_dir = cache / name
    if not split_dir.is_dir():
        return [f"{name}: missing split directory {split_dir}"]

    on_disk = {p.name for p in split_dir.iterdir() if p.is_dir()}
    if len(rows) != len(on_disk):
        out.append(f"{name}: manifest rows={len(rows)} sample-dirs={len(on_disk)}")
    for orphan in on_disk - manifest_ids:
        out.append(f"extra sample dir (not in manifest): {split_dir / orphan}")
    for r in tqdm(rows, desc=f"files_{name}", dynamic_ncols=True):
        d = cache / r["split"] / r["id"]
        if not d.is_dir():
            out.append(f"missing dir {d}")
            continue
        for fn in _REQUIRED_NAMES:
            if not (d / fn).is_file():
                out.append(f"missing {d / fn}")
        if strict_only_required:
            for p in d.iterdir():
                if p.is_file() and p.name not in _REQUIRED_NAMES:
                    out.append(f"unexpected file: {p}")
        if d.is_dir() and (d / "meta.json").is_file():
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                out.append(f"bad meta.json {d}: {e}")
                continue
            if meta.get("id") != r["id"]:
                out.append(f"id mismatch dir={r['id']} meta={meta.get('id')}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument(
        "--max-per-split",
        type=int,
        default=100,
        help="random tensor checks per split (0 = skip random tensor pass)",
    )
    p.add_argument(
        "--full-tensor",
        action="store_true",
        help="load every sample via PNG (slow; use with --max-per-split 0 to only do full file pass + full tensor)",
    )
    p.add_argument(
        "--strict-files",
        action="store_true",
        help="fail if a sample directory contains any file not in the required five",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cache = args.cache_root.resolve()
    if not cache.is_dir():
        print("ERROR: cache-root is not a directory", file=sys.stderr)
        sys.exit(1)

    all_err: list[str] = []
    for name, man in [("train", args.train_manifest), ("val", args.val_manifest)]:
        all_err.extend(
            _validate_all_files(
                cache,
                name,
                man,
                strict_only_required=args.strict_files,
            )
        )

    if all_err:
        print("FAILED (files/layout)", len(all_err), "issue(s), first 50:")
        for e in all_err[:50]:
            print(" ", e)
        if len(all_err) > 50:
            print("  ...")
        sys.exit(1)

    for name, man in [("train", args.train_manifest), ("val", args.val_manifest)]:
        ds = CachedStripDataset(man, cache)
        n = len(ds)
        if args.full_tensor:
            for idx in tqdm(range(n), desc=f"tensor_all_{name}", dynamic_ncols=True):
                all_err.extend(_check_sample(ds, idx))
        elif args.max_per_split > 0:
            rng = random.Random(args.seed + (0 if name == "train" else 1))
            idxs = list(range(n))
            rng.shuffle(idxs)
            for idx in idxs[: args.max_per_split]:
                all_err.extend(_check_sample(ds, idx))

    if all_err:
        print("FAILED (tensor)", len(all_err), "issue(s), first 50:")
        for e in all_err[:50]:
            print(" ", e)
        if len(all_err) > 50:
            print("  ...")
        sys.exit(1)
    print(
        "OK: all manifest rows have dirs+required files+meta; counts/orphans match; "
        "tensor: "
        + (
            "full pass"
            if args.full_tensor
            else (f"random {args.max_per_split}/split" if args.max_per_split else "skipped")
        )
    )


if __name__ == "__main__":
    main()
