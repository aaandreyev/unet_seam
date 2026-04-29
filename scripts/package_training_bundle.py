from __future__ import annotations

import argparse
import shutil
import json
import tarfile
import tempfile
from pathlib import Path


def _iter_rewritten_rows(root: Path, manifest_path: Path) -> list[tuple[dict, Path, str]]:
    rows: list[tuple[dict, Path, str]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_path = Path(row["source_path"])
        if not source_path.is_absolute():
            source_path = root / source_path
        arcname = f"data/source_images/{source_path.name}"
        row["source_path"] = arcname
        rows.append((row, source_path, arcname))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack clean images, source manifest, and harmonizer configs for Colab. "
        "Synthetic seam strips are built on the fly; no cached strip dataset is packaged."
    )
    parser.add_argument("--output", default="outputs/training_bundle/seam_harmonizer_training_bundle.tar.gz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_arg = Path(args.manifest)
    manifest_path = manifest_arg if manifest_arg.is_absolute() else (root / manifest_arg).resolve()
    manifest_arcname = "manifests/input_raw_manifest.jsonl"
    static_rel = [
        "configs/model_harmonizer_v1.yaml",
        "configs/train_harmonizer_v1.yaml",
        "configs/eval_harmonizer_v1.yaml",
        "configs/export_harmonizer_v1.yaml",
    ]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    for rel in static_rel:
        if not (root / rel).exists():
            raise FileNotFoundError(root / rel)
    rewritten_rows = _iter_rewritten_rows(root, manifest_path)
    sidecar = {"root": str(root), "files": [manifest_arcname, *static_rel], "dataset": "input_raw_manifest"}

    if args.output_dir:
        out_dir_arg = Path(args.output_dir)
        out_dir = out_dir_arg if out_dir_arg.is_absolute() else (root / out_dir_arg).resolve()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (out_dir / "data/source_images").mkdir(parents=True, exist_ok=True)
        (out_dir / "configs").mkdir(parents=True, exist_ok=True)
        for rel in static_rel:
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dst)
        manifest_lines = []
        for row, source_path, arcname in rewritten_rows:
            manifest_lines.append(json.dumps(row, ensure_ascii=False))
            shutil.copy2(source_path, out_dir / arcname)
        (out_dir / manifest_arcname).write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        (out_dir / "bundle.manifest.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"bundle_dir": str(out_dir), "images": len(rewritten_rows)}, ensure_ascii=False))
        return

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz", compresslevel=6) as tf:
        for rel in static_rel:
            tf.add(root / rel, arcname=rel, filter=None)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("\n".join(json.dumps(row, ensure_ascii=False) for row, _, _ in rewritten_rows) + "\n")
            tmp.flush()
            tf.add(tmp.name, arcname=manifest_arcname, filter=None)
        for _, source_path, arcname in rewritten_rows:
            tf.add(source_path, arcname=arcname, filter=None)
    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(out_path), "size_gb": round(out_path.stat().st_size / (1024**3), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
