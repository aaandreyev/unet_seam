#!/usr/bin/env python3
"""Open a local HTML report for harmonizer TensorBoard runs.

Builds a single dashboard over all `events.out.tfevents.*` files under the log
directory, preserving event-file boundaries as visual segments on the charts.

  python scripts/harmonizer_metrics_dashboard.py
  python scripts/harmonizer_metrics_dashboard.py --logdir outputs/logs/tensorboard_harmonizer
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_ANAL = Path(__file__).resolve().parent / "analyze_tfevents_harmonizer.py"
_spec = importlib.util.spec_from_file_location("analyze_tfevents_harmonizer", _ANAL)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
analyze = _mod.analyze
_merge_runs = _mod._merge_runs
_find_event_files = _mod._find_event_files
_load_scalars = _mod._load_scalars

# Chart.js (CDN) — при блокировке file:// откройте через: python -m http.server
CHART_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def _best_row_by_key(
    val_epochs: list[dict[str, Any]], key: str, *, minimize: bool = True
) -> dict[str, Any] | None:
    rows = [r for r in val_epochs if isinstance(r.get(key), (int, float)) and not math.isnan(float(r[key]))]
    if not rows:
        return None
    fn = min if minimize else max
    return fn(rows, key=lambda r: float(r[key]))


def _integrated_score_0_95(r: dict[str, Any] | None) -> float | None:
    if not r:
        return None
    bm, bb = r.get("boundary_mae_16"), r.get("baseline_boundary_mae_16")
    de, bd = r.get("boundary_ciede2000_16"), r.get("baseline_boundary_ciede2000_16")
    if not all(isinstance(x, (int, float)) for x in (bm, bb, de, bd)) or bb <= 0 or bd <= 0:
        return None
    rel_mae = (1.0 - float(bm) / float(bb)) * 100.0
    rel_de = (1.0 - float(de) / float(bd)) * 100.0
    bonus = 0.0
    if bm < 0.02 and de < 2.5:
        bonus = 15.0
    elif bm < 0.025 and de < 3.0:
        bonus = 12.0
    elif bm < 0.03 and de < 3.5:
        bonus = 8.0
    elif bm < 0.035 and de < 4.5:
        bonus = 5.0
    s = 35.0 + 0.45 * rel_mae + 0.35 * rel_de + bonus
    return max(0.0, min(95.0, round(s, 1)))


def _collect_best_rows(val_epochs: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    best_score: dict[str, Any] | None = None
    best_score_value: float | None = None
    for row in val_epochs:
        sc = _integrated_score_0_95(row)
        if sc is None:
            continue
        if best_score_value is None or sc > best_score_value:
            best_score_value = sc
            best_score = dict(row)
            best_score["integrated_score_0_95"] = sc
    return {
        "mae": _best_row_by_key(val_epochs, "boundary_mae_16", minimize=True),
        "de": _best_row_by_key(val_epochs, "boundary_ciede2000_16", minimize=True),
        "quality": _best_row_by_key(val_epochs, "quality_score", minimize=True),
        "score": best_score,
    }


def _html_escape(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _downsample(pairs: list[tuple[int, float]], max_points: int = 1800) -> list[tuple[int, float]]:
    if len(pairs) <= max_points:
        return pairs
    n = len(pairs)
    step = max(1, n // max_points)
    out = [pairs[i] for i in range(0, n, step)]
    if pairs[-1] not in out:
        out.append(pairs[-1])
    return out


def _bar_pct(label: str, value: float, cap: float = 100.0) -> str:
    w = max(0.0, min(100.0, (value / cap) * 100.0 if cap else 0.0))
    return (
        f'<div class="barrow"><span class="blab">{_html_escape(label)}</span>'
        f'<div class="bar"><i style="width:{w:.1f}%"></i></div>'
        f'<span class="bval">{value:.1f} / {cap:.0f}</span></div>'
    )


def _json_for_script(obj: Any) -> str:
    """Safe embed in <script> — avoid breaking closing tags."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _metric_or_none(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    v = row.get(key)
    if isinstance(v, (int, float)) and not math.isnan(float(v)):
        return float(v)
    return None


def _goal_status_from_best(best: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not best:
        return []
    target_mae = 0.02
    target_de = 2.5
    target_rel_mae = 50.0
    target_rel_de = 40.0
    goal_bars: list[dict[str, Any]] = []

    bm = _metric_or_none(best, "boundary_mae_16")
    bbm = _metric_or_none(best, "baseline_boundary_mae_16")
    de = _metric_or_none(best, "boundary_ciede2000_16")
    bde = _metric_or_none(best, "baseline_boundary_ciede2000_16")

    if bm is not None:
        if bbm is not None and bbm > target_mae:
            distance = _clamp01((bm - target_mae) / max(1e-8, bbm - target_mae))
        else:
            distance = 0.0 if bm <= target_mae else 1.0
        goal_bars.append(
            {
                "id": "mae_abs",
                "label": "Гейт к 95: MAE@16 ≤ 0.02",
                "distance_pct": round(100.0 * distance, 1),
                "reached": bm <= target_mae,
            }
        )

    if de is not None:
        if bde is not None and bde > target_de:
            distance = _clamp01((de - target_de) / max(1e-8, bde - target_de))
        else:
            distance = 0.0 if de <= target_de else 1.0
        goal_bars.append(
            {
                "id": "de_abs",
                "label": "Гейт к 95: ΔE@16 ≤ 2.5",
                "distance_pct": round(100.0 * distance, 1),
                "reached": de <= target_de,
            }
        )

    if bbm is not None and bbm > 0 and bm is not None:
        imp = (1.0 - bm / bbm) * 100.0
        distance = _clamp01((target_rel_mae - imp) / max(target_rel_mae, 1e-8))
        goal_bars.append(
            {
                "id": "imp_mae",
                "label": f"Гейт к 95: Rel. MAE ≥ {target_rel_mae:.0f}% (сейчас {imp:.0f}%)",
                "distance_pct": round(100.0 * distance, 1),
                "reached": imp >= target_rel_mae,
            }
        )

    if bde is not None and bde > 0 and de is not None:
        imp = (1.0 - de / bde) * 100.0
        distance = _clamp01((target_rel_de - imp) / max(target_rel_de, 1e-8))
        goal_bars.append(
            {
                "id": "imp_de",
                "label": f"Гейт к 95: Rel. ΔE ≥ {target_rel_de:.0f}% (сейчас {imp:.0f}%)",
                "distance_pct": round(100.0 * distance, 1),
                "reached": imp >= target_rel_de,
            }
        )
    return goal_bars


def _build_run_segments(files: list[Path]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, path in enumerate(sorted(files, key=lambda p: p.stat().st_mtime), start=1):
        scalars = _load_scalars(path)
        train_loss = scalars.get("train/loss/total") or []
        val_steps = sorted({s for tag, pts in scalars.items() if tag.startswith("val/") for s, _ in pts})
        train_markers: dict[str, dict[str, float] | None] = {}
        for tag in (
            "train/loss/total",
            "train/metric/boundary_mae_16",
            "train/metric/lowfreq_mae",
            "train/metric/gradient_mae",
        ):
            pts = scalars.get(tag) or []
            train_markers[tag] = {"x": float(pts[-1][0]), "y": float(pts[-1][1])} if pts else None
        segment = {
            "index": idx,
            "file": str(path),
            "file_name": path.name,
            "start_step": train_loss[0][0] if train_loss else (val_steps[0] if val_steps else None),
            "end_step": train_loss[-1][0] if train_loss else (val_steps[-1] if val_steps else None),
            "train_points": len(train_loss),
            "val_points": len(val_steps),
            "val_start_step": val_steps[0] if val_steps else None,
            "val_end_step": val_steps[-1] if val_steps else None,
            "markers": train_markers,
        }
        segments.append(segment)

    train_boundaries = [s["start_step"] for s in segments[1:] if s.get("start_step") is not None]
    ordered_val_end_steps = sorted(
        {s["val_end_step"] for s in segments if s.get("val_end_step") is not None}
    )
    val_epoch_map = {st: i + 1 for i, st in enumerate(ordered_val_end_steps)}
    for segment in segments:
        vstart = segment.get("val_start_step")
        vend = segment.get("val_end_step")
        segment["val_start_epoch"] = val_epoch_map.get(vstart)
        segment["val_end_epoch"] = val_epoch_map.get(vend)
        segment["train_boundary"] = segment.get("start_step") if segment.get("index", 0) > 1 else None
        segment["val_boundary_epoch"] = segment["val_start_epoch"] if segment.get("index", 0) > 1 else None
    return segments


def _build_chart_payload(
    report: dict[str, Any],
    data: dict[str, list[tuple[int, float]]],
    best_rows: dict[str, dict[str, Any] | None],
    score: float | None,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = report.get("val_epochs") or []

    val_epochs: list[dict[str, Any]] = []
    best_ep_mae = _metric_or_none(best_rows.get("mae"), "epoch_idx")
    best_ep_de = _metric_or_none(best_rows.get("de"), "epoch_idx")
    best_ep_score = _metric_or_none(best_rows.get("score"), "epoch_idx")
    for r in rows:
        val_epochs.append(
            {
                "ep": r.get("epoch_idx"),
                "step": r.get("epoch_end_step"),
                "mae16": r.get("boundary_mae_16"),
                "baseline_mae": r.get("baseline_boundary_mae_16"),
                "de16": r.get("boundary_ciede2000_16"),
                "baseline_de": r.get("baseline_boundary_ciede2000_16"),
                "quality": r.get("quality_score"),
            }
        )

    def series(tag: str) -> list[dict[str, float]]:
        pts = _downsample(data.get(tag) or [], 2500)
        return [{"x": float(s), "y": float(v)} for s, v in pts]

    loss_parts: list[dict[str, Any]] = []
    br = report.get("train_loss_breakdown_last_step")
    if br and br.get("components"):
        for name, d in sorted(br["components"].items(), key=lambda x: -x[1].get("weighted", 0)):
            loss_parts.append(
                {
                    "name": str(name),
                    "weighted": float(d.get("weighted", 0)),
                    "raw": float(d.get("raw", 0)),
                    "pct": float(d.get("pct_of_weighted_sum", 0)),
                }
            )

    val_segments = [
        {
            "index": seg["index"],
            "file_name": seg["file_name"],
            "boundary_epoch": seg.get("val_boundary_epoch"),
            "end_epoch": seg.get("val_end_epoch"),
        }
        for seg in segments
        if seg.get("val_end_epoch") is not None
    ]
    train_segments = [
        {
            "index": seg["index"],
            "file_name": seg["file_name"],
            "boundary_step": seg.get("train_boundary"),
            "markers": seg.get("markers", {}),
        }
        for seg in segments
    ]

    return {
        "val_epochs": val_epochs,
        "best_ep_mae": int(best_ep_mae) if best_ep_mae is not None else None,
        "best_ep_de": int(best_ep_de) if best_ep_de is not None else None,
        "best_ep_score": int(best_ep_score) if best_ep_score is not None else None,
        "train_loss": series("train/loss/total"),
        "train_mae16": series("train/metric/boundary_mae_16"),
        "train_lowfreq": series("train/metric/lowfreq_mae"),
        "train_grad": series("train/metric/gradient_mae"),
        "loss_doughnut": loss_parts,
        "goals": _goal_status_from_best(best_rows.get("score")),
        "segments": {"train": train_segments, "val": val_segments},
        "T_mae": 0.02,
        "T_de": 2.5,
        "meta": {
            "score": score,
            "T_rel_mae": 50.0,
            "T_rel_de": 40.0,
        },
    }


def build_html(
    report: dict[str, Any],
    data: dict[str, list[tuple[int, float]]],
    source_files: list[Path],
    best_rows: dict[str, dict[str, Any] | None],
    score: float | None,
    segments: list[dict[str, Any]],
) -> str:
    rows = report.get("val_epochs") or []
    bpe = report.get("inferred_batches_per_epoch")
    tmax = report.get("train_global_step_max")
    target_mae = 0.02
    target_de = 2.5
    target_rel_mae = 50.0
    target_rel_de = 40.0
    latest = max(source_files, key=lambda p: p.stat().st_mtime) if source_files else None
    best_mae = best_rows.get("mae")
    best_de = best_rows.get("de")
    best_quality = best_rows.get("quality")
    best_score = best_rows.get("score")
    sc = score if score is not None else 0.0
    sc_bar = min(100.0, (sc / 95.0) * 100.0) if sc else 0.0
    gap_to_95 = max(0.0, 95.0 - sc) if sc else 95.0

    ch = _build_chart_payload(report, data, best_rows, score, segments)
    ch_json = _json_for_script(ch)

    insights: list[str] = [
        "Дашборд объединяет все event-файлы из logdir; пунктирные разделители на графиках показывают границы между отдельными файлами.",
        "Train-кривые остаются кумулятивными средними на момент лога, поэтому визуально они гладче, чем пошаговые batch-loss.",
        "Интегральная оценка 0–95 остаётся эвристикой: финальная цель дашборда — 95/95, а остальные метрики ниже трактуются как гейты и блокеры на пути к этому уровню.",
    ]
    vol = report.get("val_volatility_pstdev") or {}
    if vol.get("boundary_mae_16") is not None:
        insights.append(
            f"Разброс val MAE@16 между эпохами: pstdev={vol['boundary_mae_16']:.4f}. Одиночные пики стоит проверять по соседним эпохам, а не трактовать как устойчивый тренд."
        )
    if len(source_files) > 1:
        insights.append(
            f"В анализ попало {len(source_files)} event-файлов. Если после resume динамика изменилась, ориентируйтесь не только на общий минимум, но и на поведение последнего сегмента."
        )
    if best_mae and best_de and best_mae.get("epoch_idx") != best_de.get("epoch_idx"):
        insights.append(
            f"Лучшие эпохи по MAE@16 и ΔE@16 различаются: ep {best_mae.get('epoch_idx')} vs ep {best_de.get('epoch_idx')}. Значит, оптимум зависит от того, что именно вы считаете главным KPI."
        )
    if best_score and best_mae and best_score.get("epoch_idx") != best_mae.get("epoch_idx"):
        insights.append(
            f"Лучшая эпоха по интегральной оценке не совпадает с лучшей по MAE@16: ep {best_score.get('epoch_idx')} vs ep {best_mae.get('epoch_idx')}. Таблица целей ниже привязана к score-best эпохе, а не только к минимуму MAE."
        )
    br = report.get("train_loss_breakdown_last_step")
    if br and br.get("components", {}).get("seam", {}).get("pct_of_weighted_sum", 0) > 80:
        insights.append("`l_seam` доминирует в weighted loss на последнем train-step. Если val-качество перестаёт расти, имеет смысл отдельно проверять баланс seam/low/chroma terms.")

    gap_rows = ""
    if best_score:
        bm = _metric_or_none(best_score, "boundary_mae_16")
        bb = _metric_or_none(best_score, "baseline_boundary_mae_16")
        de = _metric_or_none(best_score, "boundary_ciede2000_16")
        bd = _metric_or_none(best_score, "baseline_boundary_ciede2000_16")
        im_m = (1.0 - bm / bb) * 100.0 if bm is not None and bb and bb > 0 else None
        im_d = (1.0 - de / bd) * 100.0 if de is not None and bd and bd > 0 else None
        gap_rows = f"""
        <tr><td>Integrated score</td><td>{_html_escape(f"{sc:.1f}" if sc else "—")}</td>
            <td class="ok">95 / 95</td>
            <td>{_html_escape(f"до 95 осталось {gap_to_95:.1f}" if score is not None else "недостаточно данных")}</td></tr>
        <tr><td>Гейт к 95: MAE@16</td><td>{_html_escape(f"{bm:.5f}" if bm is not None else "—")}</td>
            <td class="ok">≤ {target_mae}</td>
            <td>{"пропускает к 95" if bm is not None and bm <= target_mae else "блокирует 95"}</td></tr>
        <tr><td>Гейт к 95: ΔE@16</td><td>{_html_escape(f"{de:.3f}" if de is not None else "—")}</td>
            <td class="ok">≤ {target_de}</td>
            <td>{"пропускает к 95" if de is not None and de <= target_de else "блокирует 95"}</td></tr>
        <tr><td>Гейт к 95: Rel. MAE</td><td>{_html_escape(f"{im_m:.1f}%" if im_m is not None else "—")}</td>
            <td class="ok">≥ {target_rel_mae}%</td>
            <td>{"пропускает к 95" if im_m is not None and im_m >= target_rel_mae else "блокирует 95"}</td></tr>
        <tr><td>Гейт к 95: Rel. ΔE</td><td>{_html_escape(f"{im_d:.1f}%" if im_d is not None else "—")}</td>
            <td class="ok">≥ {target_rel_de}%</td>
            <td>{"пропускает к 95" if im_d is not None and im_d >= target_rel_de else "блокирует 95"}</td></tr>
        """

    val_table = ""
    for row in rows:
        ep = row.get("epoch_idx", "")
        st = row.get("epoch_end_step", "")
        mae = row.get("boundary_mae_16")
        bmae = row.get("baseline_boundary_mae_16")
        de = row.get("boundary_ciede2000_16")
        q = row.get("quality_score")
        lt = row.get("loss_total")
        markers: list[str] = []
        if best_mae and ep == best_mae.get("epoch_idx"):
            markers.append("best-MAE")
        if best_de and ep == best_de.get("epoch_idx"):
            markers.append("best-ΔE")
        if best_score and ep == best_score.get("epoch_idx"):
            markers.append("best-score")
        if best_quality and ep == best_quality.get("epoch_idx"):
            markers.append("best-quality")
        note = ", ".join(markers) if markers else "—"
        val_table += f"""<tr>
            <td>{_html_escape(ep)}</td><td>{_html_escape(st)}</td>
            <td>{_html_escape(f"{mae:.5f}" if mae is not None else "—")}</td>
            <td>{_html_escape(f"{bmae:.5f}" if bmae is not None else "—")}</td>
            <td>{_html_escape(f"{de:.3f}" if de is not None else "—")}</td>
            <td>{_html_escape(f"{q:.2f}" if q is not None else "—")}</td>
            <td>{_html_escape(f"{lt:.4f}" if lt is not None else "—")}</td>
            <td>{_html_escape(note)}</td>
        </tr>"""

    train_block = ""
    for tag, s in (report.get("train_summaries") or {}).items():
        train_block += f"""
        <h4>{_html_escape(tag)}</h4>
        <p>first={s['first']:.6f} → last={s['last']:.6f} · min={s['min']:.6f} max={s['max']:.6f} ·
        Δ={s['delta_last_minus_first']:.6f}</p>
        <p>ср. по первым 10% логов: {s['mean_head10pct']:.6f} · по последним 10%: {s['mean_tail10pct']:.6f}</p>
        """

    loss_br = ""
    if br:
        loss_br = f'<div class="card loss-list"><h3>Текстовый разбор loss @ step {br.get("step")}</h3><ul>'
        for name, d in sorted(br.get("components", {}).items(), key=lambda x: -x[1].get("weighted", 0)):
            loss_br += f"<li><code>{_html_escape(name)}</code>: raw={d['raw']:.6f}, w={d['weight']}, "
            loss_br += f"взвеш.={d['weighted']:.6f} ({d['pct_of_weighted_sum']:.1f}%)</li>"
        loss_br += "</ul></div>"

    progress_html = ""
    if best_score:
        bm = _metric_or_none(best_score, "boundary_mae_16")
        bb = _metric_or_none(best_score, "baseline_boundary_mae_16")
        rel_m = (1.0 - bm / bb) * 100.0 if bm is not None and bb and bb > 0 else 0.0
        de = _metric_or_none(best_score, "boundary_ciede2000_16")
        bd = _metric_or_none(best_score, "baseline_boundary_ciede2000_16")
        rel_d = (1.0 - de / bd) * 100.0 if de is not None and bd and bd > 0 else 0.0
        progress_html = (
            "<h2 class='sec-h'>Подметрики на пути к 95</h2><div class=\"card bars-card\">"
            + _bar_pct("Готовность по Rel. MAE", rel_m, target_rel_mae)
            + _bar_pct("Готовность по Rel. ΔE", rel_d, target_rel_de)
            + "</div>"
        )

    kpi_mae = f"{best_mae['boundary_mae_16']:.4f}" if best_mae and best_mae.get("boundary_mae_16") is not None else "—"
    kpi_de = f"{best_de['boundary_ciede2000_16']:.2f}" if best_de and best_de.get("boundary_ciede2000_16") is not None else "—"
    kpi_score = f"{sc:.0f}" if sc else "—"
    n_val = len(rows)
    n_train_pts = len(data.get("train/loss/total") or [])

    segment_rows = ""
    for seg in segments:
        segment_rows += f"""<tr>
          <td>{seg['index']}</td>
          <td>{_html_escape(seg['file_name'])}</td>
          <td>{_html_escape(seg.get('start_step') if seg.get('start_step') is not None else '—')}</td>
          <td>{_html_escape(seg.get('end_step') if seg.get('end_step') is not None else '—')}</td>
          <td>{_html_escape(seg['train_points'])}</td>
          <td>{_html_escape(seg['val_points'])}</td>
          <td>{_html_escape(seg.get('val_start_epoch') if seg.get('val_start_epoch') is not None else '—')}</td>
          <td>{_html_escape(seg.get('val_end_epoch') if seg.get('val_end_epoch') is not None else '—')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Harmonizer — дашборд</title>
<script src="{CHART_CDN}"></script>
<style>
  :root {{
    --bg0:#0a0e14;--bg:#0f1419;--card:#151d2a;--card2:#1a2435;--text:#e8edf4;--muted:#8b9bb4;
    --acc:#4a9fe6;--acc2:#6ec8ff;--ok:#3ecf8e;--bad:#e85d75;--warn:#e8b44c;--line:#2a3a4f;
  }}
  *{{box-sizing:border-box}}
  body{{font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg0);color:var(--text);
    margin:0;line-height:1.55;min-height:100vh;
    background-image:radial-gradient(ellipse 120% 80% at 50% -20%,#1a3048 0%,transparent 55%),var(--bg0);
  }}
  .wrap{{max-width:1200px;margin:0 auto;padding:1.25rem 1.5rem 2.5rem;}}
  .hero{{display:grid;grid-template-columns:1.2fr 1fr;gap:1.25rem;margin-bottom:1.5rem;align-items:stretch;}}
  @media(max-width:900px){{.hero{{grid-template-columns:1fr;}}}}
  .hero-t{{padding:0.5rem 0;}}
  .hero-t h1{{font-size:1.55rem;font-weight:700;margin:0 0 0.35rem;letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff 0%,#8ec8ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
  .hero-t .sub{{color:var(--muted);font-size:0.88rem;margin:0;}}
  .badge{{display:inline-block;background:#1e2d42;color:#9ec5f2;font-size:0.72rem;padding:0.2rem 0.55rem;border-radius:4px;margin-right:0.4rem;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;}}
  @media(max-width:800px){{.kpi-grid{{grid-template-columns:repeat(2,1fr);}}}}
  .kpi{{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:0.7rem 0.85rem;}}
  .kpi .lbl{{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);}}
  .kpi .val{{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;color:var(--acc2);}}
  .kpi .hint{{font-size:0.75rem;color:var(--muted);margin-top:0.2rem;}}
  .funnel{{margin-top:0.8rem;padding:0.9rem 1rem;background:linear-gradient(180deg,#182232,#121a28);
    border:1px solid var(--line);border-radius:10px;}}
  .funnel h4{{margin:0 0 0.6rem;font-size:0.8rem;color:var(--warn);text-transform:uppercase;letter-spacing:0.08em;}}
  .fbar{{display:flex;align-items:center;gap:0.5rem;font-size:0.82rem;margin:0.35rem 0;}}
  .fbar span:first-child{{width:100px;flex-shrink:0;color:var(--muted);font-size:0.78rem;}}
  .ftrack{{flex:1;height:20px;border-radius:8px;background:#243044;position:relative;overflow:hidden;}}
  .ffill{{height:100%;border-radius:8px;transition:width .5s;}}
  .ffill.now{{background:linear-gradient(90deg,#2a5a8a,var(--acc));}}
  .ffill.tgt{{background:linear-gradient(90deg,#1a4d3a,var(--ok));width:100%;opacity:0.35;position:absolute;left:0;top:0;}}
  .fbar .pct{{width:40px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);font-size:0.78rem;}}
  h2.sec-h{{font-size:1.05rem;font-weight:600;color:var(--acc2);margin:2rem 0 0.75rem;padding-bottom:0.35rem;border-bottom:1px solid var(--line);}}
  h2.sec-h span{{color:var(--muted);font-weight:400;font-size:0.85rem;}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;}}
  .card.chart-card{{position:relative;}}
  .chart-canvas-h{{height:300px;position:relative;}}
  .chart-canvas-tall{{height:320px;}}
  .row2{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
  @media(max-width:900px){{.row2{{grid-template-columns:1fr;}}}}
  .row3{{display:grid;grid-template-columns:1.2fr 0.8fr;gap:1rem;align-items:stretch;}}
  @media(max-width:900px){{.row3{{grid-template-columns:1fr;}}}}
  p.meta{{color:var(--muted);font-size:0.82rem;word-break:break-all;margin:0.5rem 0 0;}}
  .callout{{font-size:0.82rem;color:#b8c5d4;background:#1a2333;border-left:3px solid var(--acc);padding:0.6rem 0.8rem;border-radius:0 8px 8px 0;margin:0.5rem 0;}}
  table{{width:100%;border-collapse:collapse;font-size:0.84rem;}}
  th,td{{border:1px solid var(--line);padding:0.4rem 0.5rem;}}
  th{{background:#1e2a3d;color:#b4c2d6;}}
  tr:nth-child(even){{background:#111820;}}
  td.ok{{color:var(--ok);}}
  .scorebox{{text-align:center;padding:1.2rem;}}
  .scorenum{{font-size:2.6rem;font-weight:800;font-variant-numeric:tabular-nums;
    background:linear-gradient(180deg,#fff,#7eb8f0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
  .track{{height:12px;background:#1c2738;border-radius:6px;position:relative;margin:1rem 0;overflow:hidden;}}
  .fill{{height:100%;background:linear-gradient(90deg,#1e4a6e,var(--acc));border-radius:6px;width:{sc_bar:.1f}%;}}
  .mark95{{position:absolute;left:calc(100% - 2px);top:0;bottom:0;width:2px;background:var(--ok);z-index:2;box-shadow:0 0 6px #3ecf8e88;}}
  .grid-tgt{{display:grid;grid-template-columns:1fr 1.2fr;gap:1.2rem;margin-top:0.5rem;}}
  @media(max-width:800px){{.grid-tgt{{grid-template-columns:1fr;}}}}
  .goal-panels{{display:grid;grid-template-columns:1fr;gap:0.6rem;}}
  .gpanel{{background:#182230;border:1px solid #2a3f55;border-radius:8px;padding:0.5rem 0.7rem;}}
  .gpanel .t{{font-size:0.75rem;color:var(--muted);}}
  .gpanel .b{{height:8px;background:#243044;border-radius:4px;margin-top:0.3rem;overflow:hidden;}}
  .gpanel .b i{{display:block;height:100%;background:linear-gradient(90deg,var(--bad),var(--warn));border-radius:4px;}}
  .gpanel.ok .b i{{background:linear-gradient(90deg,#1a4d2e,var(--ok));}}
  .loss-list ul{{margin:0.3rem 0 0 1.1rem;}}
  .note{{font-size:0.78rem;color:var(--muted);margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--line);}}
  code{{background:#1e2d42;padding:0.1em 0.35em;border-radius:4px;font-size:0.9em;}}
  .barrow{{display:flex;align-items:center;gap:0.4rem;font-size:0.82rem;}}
  .blab{{width:170px;color:var(--muted);}}
  .bar{{flex:1;height:7px;background:#243044;border-radius:4px;overflow:hidden;}}
  .bar i{{display:block;height:100%;background:var(--acc);border-radius:4px;}}
  .bval{{width:88px;text-align:right;font-size:0.8rem;color:var(--muted);font-variant-numeric:tabular-nums;}}
  .annot{{font-size:0.75rem;color:var(--warn);margin-top:0.4rem;}}
  ul.ins{{padding-left:1.15rem;color:#c2ccd8;}}
  .legend-d{{display:flex;flex-wrap:wrap;gap:0.6rem;font-size:0.75rem;margin:0.4rem 0 0.6rem;color:var(--muted);}}
  .lg{{display:inline-flex;align-items:center;gap:0.25rem;}}
  .lg i{{width:10px;height:10px;border-radius:2px;display:inline-block;}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="hero-t">
      <h1>Seam Harmonizer</h1>
      <p class="sub">Интерактивный отчёт · все TensorBoard event-файлы run-а → графики, сегменты и путь к целевому качеству 95/95</p>
      <p class="meta"><span class="badge">TB</span><code>{_html_escape(latest.name if latest else "—")}</code>
      <br />event-файлов: {len(source_files)} · шаги: {_html_escape(tmax)} · val точек: {n_val} · train лог-точек: {n_train_pts}
      {f" · ≈{bpe} батчей/эп." if bpe else ""}</p>
    </div>
    <div class="card scorebox" style="margin:0;">
      <div style="font-size:0.8rem;color:var(--muted);">Интегральная оценка</div>
      <div class="scorenum">{_html_escape(score) if score is not None else "—"}<span style="font-size:1.2rem;opacity:0.6">/95</span></div>
      <p style="margin:0.4rem 0 0;font-size:0.78rem;color:var(--muted);">≈{gap_to_95:.0f} пункта до целевого качества «95» на шкале</p>
      <div class="track"><div class="fill"></div><div class="mark95"></div></div>
      <p style="font-size:0.72rem;margin:0;color:var(--muted);">Зелёная отметка показывает финальную цель шкалы: 95/95 соответствует целевому качеству для этого dashboard.</p>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">Best val MAE@16</div><div class="val">{_html_escape(kpi_mae)}</div>
      <div class="hint">ep {best_mae.get("epoch_idx") if best_mae else "—"} · гейт к 95</div></div>
    <div class="kpi"><div class="lbl">Best val ΔE@16</div><div class="val">{_html_escape(kpi_de)}</div>
      <div class="hint">ep {best_de.get("epoch_idx") if best_de else "—"} · гейт к 95</div></div>
    <div class="kpi"><div class="lbl">Val эпох в отчёте</div><div class="val">{_html_escape(n_val)}</div>
      <div class="hint">по всем event-файлам logdir</div></div>
    <div class="kpi"><div class="lbl">Best integrated score</div><div class="val">{_html_escape(kpi_score)}</div>
      <div class="hint">ep {best_score.get("epoch_idx") if best_score else "—"} · цель = 95/95</div></div>
  </div>

  <div class="funnel">
    <h4>Инфографика: сейчас → финальная цель 95/95</h4>
    <div class="fbar">
      <span>Оценка</span>
      <div class="ftrack"><div class="ffill now" style="width:{sc_bar:.1f}%"></div>
        <div class="ffill tgt" style="width:100%"></div></div>
      <span class="pct">{sc_bar:.0f}%</span>
    </div>
    <p class="annot">Верхняя заливка — текущий прогресс по шкале 0–95; зелёная вертикаль справа — финальная цель 95/95.</p>
  </div>

  <h2 class="sec-h">Val · динамика по эпохам <span>— best-by-MAE, best-by-ΔE и best-by-score отмечены отдельно</span></h2>
  <div class="callout">Val-графики строятся по всем эпохам из всех event-файлов. Вертикальные пунктиры показывают место, где начинается следующий event-файл после resume/ротации.</div>
  <div class="row2">
    <div class="card chart-card"><div class="chart-canvas-h"><canvas id="cValMae"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-h"><canvas id="cValDe"></canvas></div></div>
  </div>

  <h2 class="sec-h">Train · кривые по глобальным шагам <span>— с маркерами конца каждого event-сегмента</span></h2>
  <div class="callout">Крупные точки на train-графиках — последнее значение тега внутри конкретного event-файла. Это помогает сравнить, как шёл последний сегмент относительно предыдущих.</div>
  <div class="row2">
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainLoss"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainMae"></canvas></div></div>
  </div>
  <div class="row2" style="margin-top:1rem;">
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainLow"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainGrad"></canvas></div></div>
  </div>

  <h2 class="sec-h">Разложение loss (последний train-step) <span>— доля взвешенного вклада и гейты к 95</span></h2>
  <div class="row3">
    <div class="card chart-card" style="min-height:280px"><div class="chart-canvas-h" style="height:260px"><canvas id="cDoughnut"></canvas></div></div>
    <div class="goal-panels" id="goalPanels"></div>
  </div>

  <h2 class="sec-h">Таблица: score-best эпоха vs цель 95/95</h2>
  <div class="grid-tgt">
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>Гейт / метрика</th><th>Сейчас</th><th>Нужно для 95</th><th>Влияние на 95</th></tr></thead>
      <tbody>
      {gap_rows or "<tr><td colspan=4>—</td></tr>"}
      </tbody>
    </table>
  </div>
  <div>
    {progress_html}
  </div>
  </div>

  <h2 class="sec-h">Val · таблица</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Эп.</th><th>Step</th><th>MAE@16</th><th>Baseline</th>
          <th>ΔE@16</th><th>quality</th><th>val loss</th><th>Метки</th>
        </tr>
      </thead>
      <tbody>{val_table or "<tr><td colspan=8>—</td></tr>"}</tbody>
    </table>
  </div>

  <h2 class="sec-h">Сегменты logdir</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead>
        <tr><th>#</th><th>Event file</th><th>Start step</th><th>End step</th><th>Train pts</th><th>Val pts</th><th>Val start ep</th><th>Val end ep</th></tr>
      </thead>
      <tbody>{segment_rows or "<tr><td colspan=8>—</td></tr>"}</tbody>
    </table>
  </div>

  <h2 class="sec-h">Train · текстовая сводка</h2>
  <div class="card">{train_block or "<p>Нет</p>"}</div>
  {loss_br}

  <h2 class="sec-h">Инсайты</h2>
  <ul class="ins">
    {"".join(f"<li>{_html_escape(s)}</li>" for s in insights)}
  </ul>
  <div class="callout">Если графики пустые: проверьте консоль (F12). При <code>file://</code> CDN может быть заблокирован — выполните <code>python -m http.server 8765</code> в папке с HTML и откройте <code>http://localhost:8765/…</code>.</div>

  <h2 class="sec-h">Мета (best_epochs)</h2>
  <div class="card"><pre style="white-space:pre-wrap;font-size:0.78rem;">{_html_escape(str(report.get("best_epochs")))}</pre></div>

  <p class="note">
    Дашборд интерпретирует 95/95 как финальную цель. Оценка строится как: 35 + 0.45×(rel MAE%) + 0.35×(rel ΔE%) + бонус, cap 0..95.
    <br />Запуск: <code>python scripts/harmonizer_metrics_dashboard.py</code>
  </p>
</div>

<script type="application/json" id="tb-payload">{ch_json}</script>
<script>
(function() {{
  const P = JSON.parse(document.getElementById('tb-payload').textContent);
  const Tm = (P.T_mae != null) ? P.T_mae : 0.02;
  const Td = (P.T_de != null) ? P.T_de : 2.5;
  const common = (title) => ({{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b9bb4', font: {{ size: 10 }} }} }},
      title: {{ display: !!title, text: title || '', color: '#6ec8ff', font: {{ size: 12 }} }}
    }}
  }});
  const segColor = '#5a6a7a';

  const makeVLineDataset = (x, label, yAxisID) => ({{
    type: 'line',
    label,
    data: [{{ x, y: -1e12 }}, {{ x, y: 1e12 }}],
    parsing: false,
    pointRadius: 0,
    borderWidth: 1,
    borderDash: [5, 5],
    borderColor: segColor,
    fill: false,
    yAxisID
  }});

  const valSegments = (P.segments && P.segments.val) || [];
  const trainSegments = (P.segments && P.segments.train) || [];

  const ve = P.val_epochs || [];
  if (ve.length) {{
    const ep = ve.map((r) => r.ep);
    const mae = ve.map((r) => r.mae16);
    const bmae = ve.map((r) => r.baseline_mae);
    const de = ve.map((r) => r.de16);
    const bde = ve.map((r) => r.baseline_de);
    const qu = ve.map((r) => r.quality);
    const hasQ = qu.some((x) => x != null && !isNaN(x));
    const bestMaeIdx = P.best_ep_mae != null ? ve.findIndex((r) => r.ep === P.best_ep_mae) : -1;
    const bestDeIdx = P.best_ep_de != null ? ve.findIndex((r) => r.ep === P.best_ep_de) : -1;
    const bestScoreIdx = P.best_ep_score != null ? ve.findIndex((r) => r.ep === P.best_ep_score) : -1;

    const maeSets = [
      {{ label: 'MAE@16 (полоса)', data: mae, borderColor: '#4a9fe6', backgroundColor: '#4a9fe625', yAxisID: 'y', tension: 0.2, pointRadius: 2, borderWidth: 2 }},
      {{ label: 'Baseline MAE@16 (вход)', data: bmae, borderColor: '#5a6a7a', borderDash: [6,4], fill: false, yAxisID: 'y', pointRadius: 0, borderWidth: 1.2 }},
      {{ label: 'Цель (абс.)', data: ep.map(() => Tm), borderColor: '#3ecf8e', borderDash: [4,4], pointRadius: 0, yAxisID: 'y', borderWidth: 1.5, fill: false }},
      {{ type: 'scatter', label: 'best by MAE', data: bestMaeIdx >= 0 ? [{{ x: ep[bestMaeIdx], y: mae[bestMaeIdx] }}] : [], yAxisID: 'y', pointRadius: 7, pointHoverRadius: 8, showLine: false, backgroundColor: '#6ec8ff' }},
      {{ type: 'scatter', label: 'best by score', data: bestScoreIdx >= 0 ? [{{ x: ep[bestScoreIdx], y: mae[bestScoreIdx] }}] : [], yAxisID: 'y', pointRadius: 7, pointStyle: 'rectRot', showLine: false, backgroundColor: '#e8b44c' }}
    ];
    valSegments.forEach((seg) => {{
      if (seg.boundary_epoch != null) maeSets.push(makeVLineDataset(seg.boundary_epoch, 'граница event-file', 'y'));
    }});
    new Chart(document.getElementById('cValMae'), {{
      type: 'line',
      data: {{ labels: ep, datasets: maeSets }},
      options: {{ ...common('Val · MAE@16'),
        scales: {{
          x: {{ ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          y: {{ position: 'left', title: {{ display: true, text: 'MAE@16', color: '#8b9bb4' }}, ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }}
        }}
      }}
    }});

    const deSets = [
      {{ label: 'ΔE@16 (полоса)', data: de, borderColor: '#3ecf8e', yAxisID: 'y', tension: 0.2, pointRadius: 2, borderWidth: 2 }},
      {{ label: 'Baseline ΔE@16 (вход)', data: bde, borderColor: '#6a8a6a', borderDash: [4,3], yAxisID: 'y', pointRadius: 0, borderWidth: 1.2 }},
      {{ label: 'Цель ΔE (абс.)', data: ep.map(() => Td), borderColor: '#e85d75', borderDash: [4,4], pointRadius: 0, yAxisID: 'y', borderWidth: 1.5, fill: false }},
      {{ type: 'scatter', label: 'best by ΔE', data: bestDeIdx >= 0 ? [{{ x: ep[bestDeIdx], y: de[bestDeIdx] }}] : [], yAxisID: 'y', pointRadius: 7, showLine: false, backgroundColor: '#a5f0c8' }},
      {{ type: 'scatter', label: 'best by score', data: bestScoreIdx >= 0 ? [{{ x: ep[bestScoreIdx], y: de[bestScoreIdx] }}] : [], yAxisID: 'y', pointRadius: 7, pointStyle: 'rectRot', showLine: false, backgroundColor: '#e8b44c' }}
    ];
    if (hasQ) deSets.push({{ label: 'quality (эвр.)', data: qu, borderColor: '#e8b44c', yAxisID: 'y1', tension: 0.2, pointRadius: 2, borderWidth: 1.5 }});
    valSegments.forEach((seg) => {{
      if (seg.boundary_epoch != null) deSets.push(makeVLineDataset(seg.boundary_epoch, 'граница event-file', 'y'));
    }});
    new Chart(document.getElementById('cValDe'), {{
      type: 'line',
      data: {{ labels: ep, datasets: deSets }},
      options: {{ ...common('Val · ΔE@16' + (hasQ ? ' + quality' : '')),
        scales: {{
          x: {{ ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          y: {{ position: 'left', title: {{ display: true, text: 'ΔE / CIEDE', color: '#8b9bb4' }}, ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          ...(hasQ ? {{ y1: {{ position: 'right', title: {{ display: true, text: 'quality', color: '#e8b44c' }}, ticks: {{ color: '#e8b44c' }}, grid: {{ display: false }} }} }} : {{}})
        }}
      }}
    }});
  }}

  const mkLine = (id, data, col, yLabel, markerKey) => {{
    if (!data || !data.length) return;
    const sets = [
      {{
        label: yLabel,
        data,
        borderColor: col,
        pointRadius: 0,
        borderWidth: 1.2,
        fill: true,
        backgroundColor: col + '18',
        tension: 0.15
      }}
    ];
    trainSegments.forEach((seg) => {{
      if (seg.boundary_step != null) sets.push(makeVLineDataset(seg.boundary_step, 'граница event-file', 'y'));
      const m = seg.markers && seg.markers[markerKey];
      if (m) {{
        sets.push({{
          type: 'scatter',
          label: 'конец сегмента #' + seg.index,
          data: [m],
          showLine: false,
          pointRadius: 5,
          pointHoverRadius: 6,
          backgroundColor: col,
          borderColor: '#ffffff'
        }});
      }}
    }});
    new Chart(document.getElementById(id), {{
      type: 'line',
      data: {{ datasets: sets }},
      options: {{
        ...common(yLabel),
        scales: {{
          x: {{ type: 'linear', ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          y: {{ ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }}
        }}
      }}
    }});
  }};
  mkLine('cTrainLoss', P.train_loss, '#4a9fe6', 'train/loss/total', 'train/loss/total');
  mkLine('cTrainMae', P.train_mae16, '#6ec8ff', 'train/metric/boundary_mae_16', 'train/metric/boundary_mae_16');
  mkLine('cTrainLow', P.train_lowfreq, '#3ecf8e', 'train/metric/lowfreq_mae', 'train/metric/lowfreq_mae');
  mkLine('cTrainGrad', P.train_grad, '#e8b44c', 'train/metric/gradient_mae', 'train/metric/gradient_mae');

  const parts = P.loss_doughnut || [];
  if (parts.length) {{
    new Chart(document.getElementById('cDoughnut'), {{
      type: 'doughnut',
      data: {{
        labels: parts.map((p) => p.name + ' (' + p.pct.toFixed(1) + '%)'),
        datasets: [{{ data: parts.map((p) => p.weighted), backgroundColor: ['#4a9fe6','#3ecf8e','#e8b44c','#e85d75','#9b7ed9','#5a6a7a','#6ec8ff','#2a5a8a'] }}]
      }},
      options: {{ ...common('Вклад взвешенного loss'), plugins: {{ ...common().plugins, tooltip: {{ callbacks: {{ label: (c) => c.label + ': ' + c.parsed.toFixed(4) }} }} }} }}
    }});
  }}

  const gp = document.getElementById('goalPanels');
  if (gp && (P.goals || []).length) {{
    P.goals.forEach((g) => {{
      const d = Math.max(0, Math.min(100, g.distance_pct || 0));
      const el = document.createElement('div');
      el.className = 'gpanel' + (g.reached ? ' ok' : '');
      el.innerHTML = '<div class="t">' + g.label + '</div><div class="b"><i style="width:' + d.toFixed(1) + '%"></i></div><div class="t" style="margin-top:0.25rem">дистанция до цели: ~' + d.toFixed(0) + '%</div>';
      gp.appendChild(el);
    }});
  }}
}})();
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=_ROOT / "outputs" / "logs" / "tensorboard_harmonizer")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    files = _find_event_files(args.logdir) if args.logdir.exists() else []
    if not files:
        html_str = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Нет логов</title></head>
        <body style="font-family:sans-serif;padding:2rem;background:#0f1419;color:#e8edf4">
        <p>Нет <code>events.out.tfevents.*</code> в <code>{_html_escape(args.logdir)}</code></p></body></html>"""
        out = args.output or Path(tempfile.mkstemp(suffix="_harmonizer_dashboard.html", prefix="tb_")[1])
        out.write_text(html_str, encoding="utf-8")
        print("No tfevents. Wrote:", out)
        if not args.no_browser:
            webbrowser.open(out.as_uri())
        return

    files = sorted(files, key=lambda p: p.stat().st_mtime)
    data = _merge_runs(files)
    report = analyze(data)
    best_rows = _collect_best_rows(report.get("val_epochs") or [])
    score = _integrated_score_0_95(best_rows.get("score")) if best_rows.get("score") else None
    segments = _build_run_segments(files)
    html_str = build_html(report, data, files, best_rows, score, segments)

    if args.output:
        out: Path = args.output
    else:
        import os

        fd, name = tempfile.mkstemp(suffix="_harmonizer_dashboard.html", prefix="tb_")
        os.close(fd)
        out = Path(name)
    out.write_text(html_str, encoding="utf-8")
    latest = files[-1]
    print("TensorBoard files:", len(files))
    print("Latest event file:", latest)
    print("Dashboard HTML:", out.resolve())
    if not args.no_browser:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
