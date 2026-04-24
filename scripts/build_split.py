from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data.manifest import read_jsonl, write_jsonl
from src.utils.phash import hamming_distance


def cluster_rows(rows: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for row in rows:
        placed = False
        for cluster in clusters:
            if hamming_distance(row["phash64"], cluster[0]["phash64"]) <= 6:
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(Path(args.manifest))
    clusters = cluster_rows(rows)
    random.Random(42).shuffle(clusters)
    n = len(clusters)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.92)
    for idx, cluster in enumerate(clusters):
        split = "train" if idx < train_cut else "val" if idx < val_cut else "bench"
        for row in cluster:
            row["cluster_id"] = idx
            row["split"] = split
    write_jsonl(Path("manifests/input_raw_manifest.jsonl"), rows)
    for split in ("train", "val", "bench"):
        write_jsonl(Path(f"manifests/source_{split}.jsonl"), [row for row in rows if row["split"] == split])


if __name__ == "__main__":
    main()
