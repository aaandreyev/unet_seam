from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def _add_tree(tf: tarfile.TarFile, source_dir: Path, arc_prefix: str) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir)
        tf.add(path, arcname=f"{arc_prefix}/{rel.as_posix()}", filter=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack strip cache, manifests, and configs for Colab. "
        "By default, cache on disk is outputs/strip_cache/; it is always stored in the "
        "archive as outputs/strip_cache/ (train, val) so the Colab notebook paths match."
    )
    parser.add_argument("--output", default="outputs/training_bundle/seam_residual_corrector_training_bundle.tar.gz")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--strip-cache",
        default="outputs/strip_cache",
        help="Directory under --root with train/ and val/ (e.g. outputs/strip_cache_final). "
        "Contents are packed as outputs/strip_cache/... in the archive.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip_src = (root / args.strip_cache).resolve()
    if not strip_src.is_dir():
        raise FileNotFoundError(f"strip cache dir: {strip_src}")
    for split in ("train", "val"):
        if not (strip_src / split).is_dir():
            raise FileNotFoundError(f"missing {strip_src / split} (run materialize for both splits)")

    static_rel = [
        "manifests/strip_train_cache.jsonl",
        "manifests/strip_val_cache.jsonl",
        "configs/model_resunet_v1.yaml",
        "configs/train_synth_v1.yaml",
        "configs/eval_v1.yaml",
        "configs/export_v1.yaml",
    ]
    manifest: dict = {"root": str(root), "files": list(static_rel), "strip_cache_source": str(strip_src)}

    for rel in static_rel:
        if not (root / rel).exists():
            raise FileNotFoundError(f"missing required bundle path: {root / rel}")

    use_pigz = bool(shutil.which("tar") and shutil.which("pigz"))
    # pigz+tar can only pack real paths; remapping final->strip_cache needs Python tar for trees.
    use_python_tar = (args.strip_cache.rstrip("/") != "outputs/strip_cache") or (not use_pigz)

    if not use_python_tar and use_pigz:
        rels = list(static_rel) + ["outputs/strip_cache/train", "outputs/strip_cache/val"]
        for rel in rels:
            if not (root / rel).exists():
                raise FileNotFoundError(f"missing: {root / rel}")
        cmd = ["tar", "-C", str(root), "-I", "pigz -9", "-cf", str(out_path)] + rels
        subprocess.run(cmd, check=True)
    else:
        with tarfile.open(out_path, "w:gz", compresslevel=6) as tf:
            for rel in static_rel:
                tf.add(root / rel, arcname=rel, filter=None)
            for split in ("train", "val"):
                _add_tree(tf, strip_src / split, f"outputs/strip_cache/{split}")

    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    size_gb = round(out_path.stat().st_size / (1024**3), 3)
    print(
        json.dumps(
            {
                "bundle": str(out_path),
                "size_gb": size_gb,
                "strip_cache_packed_from": str(strip_src),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
