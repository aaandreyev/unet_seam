from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NODE_ROOT = Path(__file__).resolve().parent
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

try:
    from .seam_corrector_node import SeamHarmonizerV3Node
    from .seam_harmonize_hybrid_node import SeamHarmonizerHybridNode
except ImportError:
    from seam_corrector_node import SeamHarmonizerV3Node
    from seam_harmonize_hybrid_node import SeamHarmonizerHybridNode

NODE_CLASS_MAPPINGS = {
    "SeamHarmonizerV3": SeamHarmonizerV3Node,
    "SeamHarmonizerHybrid": SeamHarmonizerHybridNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamHarmonizerV3": "Seam Harmonizer v3",
    "SeamHarmonizerHybrid": "Seam Harmonizer Hybrid",
}
