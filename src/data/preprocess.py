from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.image_io import center_crop_square, open_rgb_image, resize_square, save_png, sha256_file
from src.utils.phash import compute_phash64


CRITICAL_SCENE_TAGS = {
    "sky",
    "skin",
    "gradient",
    "night",
    "wall",
    "water",
    "glass",
    "architecture",
    "leaves",
}


@dataclass
class PreparedSource:
    row: dict
    excluded_reason: Optional[dict] = None


def extract_scene_tags(caption: str) -> list[str]:
    tokens = {token.lower() for token in re.findall(r"[a-zA-Z_]+", caption)}
    return sorted(token for token in CRITICAL_SCENE_TAGS if token in tokens)


def prepare_single_source(
    path: Path,
    output_dir: Path,
    sample_id: str,
) -> PreparedSource:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return PreparedSource({}, {"path": str(path), "reason": "unsupported_extension"})
    try:
        image = open_rgb_image(path)
    except Exception as exc:
        return PreparedSource({}, {"path": str(path), "reason": str(exc)})
    if min(image.size) < 512:
        return PreparedSource({}, {"path": str(path), "reason": "too_small"})
    image = resize_square(center_crop_square(image), 1024)
    out_path = output_dir / f"{sample_id}.png"
    save_png(image, out_path)
    caption_path = path.with_suffix(".txt")
    caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
    row = {
        "id": sample_id,
        "source_path": str(out_path).replace("\\", "/"),
        "original_path": str(path).replace("\\", "/"),
        "caption_path": str(caption_path).replace("\\", "/") if caption_path.exists() else None,
        "caption": caption,
        "scene_tags": extract_scene_tags(caption),
        "phash64": compute_phash64(out_path),
        "cluster_id": None,
        "split": None,
        "width": 1024,
        "height": 1024,
        "has_icc": bool(image.info.get("icc_profile")),
        "sha256": sha256_file(out_path),
        "source_domain": "photo",
    }
    return PreparedSource(row=row)
