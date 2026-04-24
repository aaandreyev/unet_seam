from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageCms, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def open_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        raise ValueError("rgba_not_allowed")
    if "icc_profile" in image.info:
        try:
            src = ImageCms.ImageCmsProfile(BytesIO(image.info["icc_profile"]))
            dst = ImageCms.createProfile("sRGB")
            image = ImageCms.profileToProfile(image, src, dst, outputMode=image.mode)
        except Exception:
            pass
    if image.mode in {"L", "P", "CMYK"}:
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_square(image: Image.Image, size: int = 1024) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
