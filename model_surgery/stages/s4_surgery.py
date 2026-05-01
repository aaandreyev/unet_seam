"""Stage 4: Head transplant surgery.

For top-K base checkpoints × all checkpoint donors × each head:
transplant that head's weights from donor into base, eval.

Also tries full coarse_head transplant.

Outputs:
  outputs/s4_surgery.json
  outputs/s4_best.pt
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from tqdm import tqdm

from model_surgery.lib.checkpoint_io import (
    free_memory, load_ema, save_surgery_checkpoint, transplant_head, HEAD_SLICES,
)
from model_surgery.lib.eval_mini import (
    _pick_device, build_loader, metrics_summary, preload_batches,
    quality_score, run_eval_on_preloaded,
)
from model_surgery.lib.reporting import RunLog, save_json


def surgery(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    top_k_base: int = 4,
    top_k_donors: int = 8,
) -> list[dict[str, Any]]:
    s_cfg = cfg["s4_surgery"]
    eval_cfg = cfg["eval"]
    heads = s_cfg["heads_to_transplant"]
    bases = survey_rows[:top_k_base]
    donors = survey_rows[:top_k_donors]

    mat_dir = Path(eval_cfg["materialized_dir"]) if eval_cfg.get("materialized_dir") else None
    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"],
        materialized_dir=mat_dir,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        materialized_preload=bool(eval_cfg.get("materialized_preload", False)),
    )
    device = _pick_device()
    preloaded = preload_batches(loader, device)

    combos = [
        (b, d, h)
        for b, d, h in itertools.product(range(len(bases)), range(len(donors)), heads)
        if bases[b]["path"] != donors[d]["path"]
    ]
    total = len(combos)
    log.log("surgery_start", n_bases=len(bases), n_donors=len(donors),
            heads=heads, total_combos=total)
    print(f"\n[S4] Surgery: {len(bases)} bases × {len(donors)} donors × {len(heads)} heads "
          f"= {total} transplants")

    all_results: list[dict[str, Any]] = []
    best_q = float("inf")
    best_state: dict | None = None
    best_meta: dict | None = None
    best_result: dict | None = None

    pbar = tqdm(total=total, desc="S4 surgery",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    _cache: dict[str, tuple] = {}

    def _load(path: str):
        if path not in _cache:
            _cache[path] = load_ema(Path(path))
        return _cache[path]

    for bi, di, head in combos:
        base_row, donor_row = bases[bi], donors[di]
        base_state, base_meta = _load(base_row["path"])
        donor_state, _ = _load(donor_row["path"])

        try:
            new_state = transplant_head(base_state, donor_state, head)
            m = run_eval_on_preloaded(
                new_state, base_meta, preloaded, device=device, outer_width=eval_cfg["outer_width"]
            )
            q = quality_score(m)
            result = {
                "base": base_row["name"], "donor": donor_row["name"],
                "head": head, "quality": q,
                "base_quality": base_row.get("quality_score"),
                "donor_quality": donor_row.get("quality_score"),
                **m,
            }
            all_results.append(result)
            if q < best_q:
                best_q = q
                best_state = new_state
                best_meta = base_meta
                best_result = result
                pbar.set_postfix_str(
                    f"BEST head={head} base={base_row['name'][:20]} "
                    f"donor={donor_row['name'][:20]} {metrics_summary(m)}"
                )
            else:
                del new_state
            free_memory()
        except Exception as e:
            pass
        pbar.update(1)

    for state, meta in _cache.values():
        del state, meta
    _cache.clear()
    free_memory()
    pbar.close()

    all_results.sort(key=lambda r: r.get("quality", float("inf")))
    save_json(all_results[:400], out_dir / "s4_surgery.json")

    if best_state is not None and best_meta is not None:
        save_surgery_checkpoint(best_state, best_meta, out_dir / "s4_best.pt")
        del best_state
        free_memory()
        log.log("surgery_done", best_quality=best_q, best=best_result)
        print(f"\n[S4] Best surgery: head={best_result['head']} "
              f"base={best_result['base']} donor={best_result['donor']} Q={best_q:.3f}")

    return all_results
