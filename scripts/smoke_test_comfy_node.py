from __future__ import annotations

import torch

from comfy_node.seam_corrector_node import SeamResidualCorrectorNode


def main() -> None:
    node = SeamResidualCorrectorNode()
    image = torch.rand(1, 1024, 1024, 3)
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 256:768, 256:768] = 1.0
    try:
        node.run(image, mask, "outputs/exports/seam_residual_corrector_v1.safetensors", 128, 1.0, True, True, True, True, False)
    except FileNotFoundError:
        print("export not found; smoke test skipped")


if __name__ == "__main__":
    main()
