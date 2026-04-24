from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


def _phash_from_image(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    img_size = hash_size * highfreq_factor
    image = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_full = dct(dct_rows, axis=1, norm="ortho")
    lowfreq = dct_full[:hash_size, :hash_size]
    med = np.median(lowfreq[1:, 1:])
    bits = lowfreq > med
    flat = "".join("1" if bit else "0" for bit in bits.flatten())
    return f"{int(flat, 2):0{hash_size * hash_size // 4}x}"


def compute_phash64(path: Path) -> str:
    with Image.open(path) as image:
        return _phash_from_image(image, hash_size=8)


def hamming_distance(hex_a: str, hex_b: str) -> int:
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")
