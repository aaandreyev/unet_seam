"""Stage 3: Cross-checkpoint merging (linear, SLERP, selective head blend).

For every pair of top-K checkpoints × alpha grid: merge weights, eval, track best.
Also does selective head merging: take head from donor, keep rest from base.

Outputs:
  outputs/s3_merge.json   — all merge results sorted by quality
  outputs/s3_best.pt      — best merged model state
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from tqdm import tqdm

from model_surgery.lib.checkpoint_io import (
    clone_state, free_memory, linear_merge, load_ema,
    save_surgery_checkpoint, selective_head_merge, slerp_merge, HEAD_SLICES,
)
from model_surgery.lib.eval_mini import (
    ReusableModelEvaluator, _pick_device, build_loader, metrics_summary,
    preload_batches, quality_score,
)
from model_surgery.lib.reporting import RunLog, save_json


def merge(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    s_cfg = cfg["s3_merge"]
    eval_cfg = cfg["eval"]
    alphas = s_cfg["alphas"]
    sel_heads = s_cfg["selective_heads"]
    use_fast_proxy = bool(s_cfg.get("use_fast_proxy", True))
    rerank_top_k = int(s_cfg.get("rerank_top_k", 12))
    candidates = survey_rows[:top_k]

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
    evaluator = ReusableModelEvaluator(preloaded, device, outer_width=eval_cfg["outer_width"])

    pairs = [(i, j) for i, j in itertools.combinations(range(len(candidates)), 2)]
    n_linear = len(pairs) * len(alphas)
    n_slerp = len(pairs) * len(alphas)
    n_selective = len(pairs) * len(sel_heads) * len(alphas)
    total = n_linear + n_slerp + n_selective

    log.log("merge_start", top_k=len(candidates), pairs=len(pairs),
            alphas=len(alphas), total_evals=total)
    print(f"\n[S3] Merge search: {len(pairs)} pairs × "
          f"({len(alphas)} linear + {len(alphas)} slerp + {len(sel_heads)}heads×{len(alphas)}) "
          f"= {total} evals")

    proxy_results: list[dict[str, Any]] = []
    best_q = float("inf")
    best_state: dict | None = None
    best_meta: dict | None = None
    best_result: dict | None = None

    pbar = tqdm(total=total, desc="S3 merge",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    # Cache loaded states to avoid redundant I/O
    _cache: dict[str, tuple] = {}

    def _load(path: str):
        if path not in _cache:
            _cache[path] = load_ema(Path(path))
        return _cache[path]

    for i, j in pairs:
        row_a, row_b = candidates[i], candidates[j]
        state_a, meta_a = _load(row_a["path"])
        state_b, meta_b = _load(row_b["path"])
        name_a, name_b = row_a["name"], row_b["name"]

        for alpha in alphas:
            for merge_fn, merge_type in [
                (lambda a, b, al: linear_merge(a, b, al), "linear"),
                (lambda a, b, al: slerp_merge(a, b, al), "slerp"),
            ]:
                try:
                    merged = merge_fn(state_a, state_b, alpha)
                    m = evaluator.evaluate(merged, meta_a, fast=use_fast_proxy)
                    q = quality_score(m)
                    result = {"type": merge_type, "a": name_a, "b": name_b,
                              "alpha": alpha, "quality": q, "proxy": use_fast_proxy, **m}
                    proxy_results.append(result)
                    del merged
                    if q < best_q:
                        best_q = q
                        best_result = result
                        pbar.set_postfix_str(f"BEST_PROXY {merge_type} α={alpha:.2f} {metrics_summary(m)}")
                    free_memory()
                except Exception:
                    pass
                pbar.update(1)

            # Selective head merges
            for head in sel_heads:
                try:
                    merged = selective_head_merge(state_a, state_b, head, alpha)
                    m = evaluator.evaluate(merged, meta_a, fast=use_fast_proxy)
                    q = quality_score(m)
                    result = {"type": f"head_blend_{head}", "a": name_a, "b": name_b,
                              "alpha": alpha, "head": head, "quality": q, "proxy": use_fast_proxy, **m}
                    proxy_results.append(result)
                    del merged
                    if q < best_q:
                        best_q = q
                        best_result = result
                        pbar.set_postfix_str(f"BEST_PROXY head={head} α={alpha:.2f} {metrics_summary(m)}")
                    free_memory()
                except Exception:
                    pass
                pbar.update(1)

    rerank_candidates = sorted(proxy_results, key=lambda r: r.get("quality", float("inf")))[: max(1, rerank_top_k)]
    all_results: list[dict[str, Any]] = []
    best_q = float("inf")
    best_result = None

    for candidate in rerank_candidates:
        row_a = next(r for r in candidates if r["name"] == candidate["a"])
        row_b = next(r for r in candidates if r["name"] == candidate["b"])
        state_a, meta_a = _load(row_a["path"])
        state_b, _ = _load(row_b["path"])
        if candidate["type"] == "linear":
            merged = linear_merge(state_a, state_b, candidate["alpha"])
        elif candidate["type"] == "slerp":
            merged = slerp_merge(state_a, state_b, candidate["alpha"])
        else:
            merged = selective_head_merge(state_a, state_b, candidate["head"], candidate["alpha"])
        m = evaluator.evaluate(merged, meta_a, fast=False)
        q = quality_score(m)
        full_result = {**candidate, "proxy_quality": candidate["quality"], "quality": q, "proxy": False, **m}
        all_results.append(full_result)
        if q < best_q:
            if best_state is not None:
                del best_state
            best_q = q
            best_state = merged
            best_meta = meta_a
            best_result = full_result
        else:
            del merged
        free_memory()

    # Free cache
    evaluator.close()
    for state, meta in _cache.values():
        del state, meta
    _cache.clear()
    free_memory()
    pbar.close()

    all_results.sort(key=lambda r: r.get("quality", float("inf")))
    save_json({
        "proxy_top": sorted(proxy_results, key=lambda r: r.get("quality", float("inf")))[:300],
        "reranked": all_results,
    }, out_dir / "s3_merge.json")

    if best_state is not None and best_meta is not None:
        save_surgery_checkpoint(best_state, best_meta, out_dir / "s3_best.pt")
        del best_state
        free_memory()
        log.log("merge_done", best_quality=best_q, best=best_result)
        print(f"\n[S3] Best merge: {best_result}")

    return all_results
