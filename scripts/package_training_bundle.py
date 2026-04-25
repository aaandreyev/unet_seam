from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack clean images, source manifest, and harmonizer configs for Colab. "
        "Synthetic seam strips are built on the fly; no cached strip dataset is packaged."
    )
    parser.add_argument("--output", default="outputs/training_bundle/seam_harmonizer_training_bundle.tar.gz")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests/input_raw_manifest.jsonl"
    static_rel = [
        "manifests/input_raw_manifest.jsonl",
        "configs/model_harmonizer_v1.yaml",
        "configs/train_harmonizer_v1.yaml",
        "configs/eval_harmonizer_v1.yaml",
        "configs/export_harmonizer_v1.yaml",
    ]
    for rel in static_rel:
        if not (root / rel).exists():
            raise FileNotFoundError(root / rel)
    with tarfile.open(out_path, "w:gz", compresslevel=6) as tf:
        for rel in static_rel[1:]:
            tf.add(root / rel, arcname=rel, filter=None)
        rewritten_rows = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_path = Path(row["source_path"])
            if not source_path.is_absolute():
                source_path = root / source_path
            arcname = f"data/source_images/{source_path.name}"
            row["source_path"] = arcname
            rewritten_rows.append(json.dumps(row, ensure_ascii=False))
            tf.add(source_path, arcname=arcname, filter=None)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("\n".join(rewritten_rows) + "\n")
            tmp.flush()
            tf.add(tmp.name, arcname="manifests/input_raw_manifest.jsonl", filter=None)
    sidecar = {"root": str(root), "files": static_rel, "dataset": "input_raw_manifest"}
    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(out_path), "size_gb": round(out_path.stat().st_size / (1024**3), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
