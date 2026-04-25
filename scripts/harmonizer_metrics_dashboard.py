#!/usr/bin/env python3
"""Open a local HTML report for the latest TensorBoard run (harmonizer).

Picks the newest `events.out.tfevents.*` under the log directory, runs analytics,
renders interactive charts (Chart.js) + static infographic, opens the browser.

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

# Chart.js (CDN) — при блокировке file:// откройте через: python -m http.server
CHART_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def _find_latest_tfevent_file(logdir: Path) -> Path | None:
    if not logdir.is_dir():
        return None
    cands = sorted(logdir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _best_val_row(val_epochs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not val_epochs:
        return None
    with_mae = [r for r in val_epochs if r.get("boundary_mae_16") is not None]
    if not with_mae:
        return val_epochs[-1]
    return min(with_mae, key=lambda r: r["boundary_mae_16"])


def _integrated_score_0_95(r: dict[str, Any]) -> float | None:
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


def _build_chart_payload(
    report: dict[str, Any],
    data: dict[str, list[tuple[int, float]]],
    best: dict[str, Any] | None,
    score: float | None,
) -> dict[str, Any]:
    T_MAE, T_DE = 0.02, 2.5
    T_REL_MAE, T_REL_DE = 50.0, 40.0
    rows = report.get("val_epochs") or []

    val_epochs: list[dict[str, Any]] = []
    best_ep_mae: int | None = None
    if rows:
        min_m = min((r.get("boundary_mae_16") for r in rows if r.get("boundary_mae_16") is not None), default=None)
        for r in rows:
            ep = r.get("epoch_idx")
            mae = r.get("boundary_mae_16")
            bmae = r.get("baseline_boundary_mae_16")
            de = r.get("boundary_ciede2000_16")
            bde = r.get("baseline_boundary_ciede2000_16")
            q = r.get("quality_score")
            if mae is not None and min_m is not None and abs(mae - min_m) < 1e-8:
                best_ep_mae = int(ep) if ep is not None else None
            val_epochs.append(
                {
                    "ep": ep,
                    "mae16": mae,
                    "baseline_mae": bmae,
                    "de16": de,
                    "baseline_de": bde,
                    "quality": q,
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

    # Goal comparison bars (0..1 progress to target, inverted for mae/de lower better)
    goal_bars: list[dict[str, Any]] = []
    if best:
        bm = best.get("boundary_mae_16")
        bbm = best.get("baseline_boundary_mae_16")
        de = best.get("boundary_ciede2000_16")
        bde = best.get("baseline_boundary_ciede2000_16")
        if bm is not None and T_MAE > 0:
            # how close to target line (0 = at target, 1 = as bad as baseline for MAE)
            if bbm and bbm > T_MAE:
                prog = (bm - T_MAE) / max(1e-8, bbm - T_MAE)
            else:
                prog = 0.0 if bm <= T_MAE else 0.5
            goal_bars.append({"id": "mae_abs", "label": "MAE@16 → цель 0.02", "progress": min(1.0, max(0.0, float(prog)))})
        if de is not None and bde and bde > 0 and T_DE > 0:
            if bde > T_DE:
                prog2 = (de - T_DE) / max(1e-8, bde - T_DE)
            else:
                prog2 = 0.0
            goal_bars.append({"id": "de_abs", "label": "ΔE@16 → цель 2.5", "progress": min(1.0, max(0.0, float(prog2)))})
        if bbm and bm is not None:
            imp = (1.0 - float(bm) / float(bbm)) * 100.0
            goal_bars.append(
                {
                    "id": "imp_mae",
                    "label": f"Rel. MAE: {imp:.0f}% / цель {T_REL_MAE:.0f}%",
                    "progress": max(0.0, 1.0 - imp / T_REL_MAE),
                }
            )
        if bde and de is not None:
            imp2 = (1.0 - float(de) / float(bde)) * 100.0
            goal_bars.append(
                {
                    "id": "imp_de",
                    "label": f"Rel. ΔE: {imp2:.0f}% / цель {T_REL_DE:.0f}%",
                    "progress": max(0.0, 1.0 - imp2 / T_REL_DE),
                }
            )

    return {
        "val_epochs": val_epochs,
        "best_ep_mae": best_ep_mae,
        "train_loss": series("train/loss/total"),
        "train_mae16": series("train/metric/boundary_mae_16"),
        "train_lowfreq": series("train/metric/lowfreq_mae"),
        "train_grad": series("train/metric/gradient_mae"),
        "loss_doughnut": loss_parts,
        "goals": goal_bars,
        "T_mae": T_MAE,
        "T_de": T_DE,
        "meta": {
            "score": score,
            "T_MAE": T_MAE,
            "T_DE": T_DE,
            "T_rel_mae": T_REL_MAE,
            "T_rel_de": T_REL_DE,
        },
    }


def build_html(
    report: dict[str, Any],
    data: dict[str, list[tuple[int, float]]],
    source_file: Path,
    best: dict[str, Any] | None,
    score: float | None,
) -> str:
    rows = report.get("val_epochs") or []
    bpe = report.get("inferred_batches_per_epoch")
    tmax = report.get("train_global_step_max")
    T_MAE = 0.02
    T_DE = 2.5
    T_REL_MAE_IMP = 50.0
    T_REL_DE_IMP = 40.0
    ch = _build_chart_payload(report, data, best, score)
    ch_json = _json_for_script(ch)

    insights: list[str] = [
        "Метрики val — в конце каждой эпохи; train — кумулятивное среднее на момент лога.",
        "Оценка 0–95% — эвристика; графики train могут выглядеть гладкими, т.к. это бегущее среднее по эпохе.",
    ]
    if report.get("val_volatility_pstdev"):
        v = report["val_volatility_pstdev"]
        if v.get("boundary_mae_16", 0) and v["boundary_mae_16"] > 0.02:
            insights.append("Высокий разброс val MAE@16 — мало val-сэмплов; смотри доверие к пикам на графике val.")
    if best and best.get("mae16_improvement_pct") is not None:
        if best["mae16_improvement_pct"] < 0:
            insights.append("В лучшей по MAE эпохе выход в полосе ещё хуже baseline-входа — цель: зазор >0 стабильно на val.")
    br = report.get("train_loss_breakdown_last_step")
    if br and br.get("components", {}).get("seam", {}).get("pct_of_weighted_sum", 0) > 80:
        insights.append("l_seam доминирует в weighted loss — кривые rec/seam/balance важны при разборе графиков.")

    gap_rows = ""
    if best:
        bm, bb = best.get("boundary_mae_16"), best.get("baseline_boundary_mae_16")
        de, bd = best.get("boundary_ciede2000_16"), best.get("baseline_boundary_ciede2000_16")
        im_m = (1.0 - bm / bb) * 100.0 if bm is not None and bb else None
        im_d = (1.0 - de / bd) * 100.0 if de is not None and bd else None
        gap_rows = f"""
        <tr><td>MAE@16 (абс.)</td><td>{_html_escape(f"{bm:.5f}" if bm is not None else "—")}</td>
            <td class="ok">≤ {T_MAE}</td>
            <td>{"↓ ниже" if bm and bm > T_MAE else "✓"}</td></tr>
        <tr><td>ΔE@16 (абс.)</td><td>{_html_escape(f"{de:.3f}" if de is not None else "—")}</td>
            <td class="ok">≤ {T_DE}</td>
            <td>{"↓ ниже" if de and de > T_DE else "✓"}</td></tr>
        <tr><td>Улучш. vs baseline (MAE %)</td><td>{_html_escape(f"{im_m:.1f}" if im_m is not None else "—")}</td>
            <td class="ok">≥ {T_REL_MAE_IMP}%</td>
            <td>{"↑" if im_m is not None and im_m < T_REL_MAE_IMP else "✓"}</td></tr>
        <tr><td>Улучш. vs baseline (ΔE %)</td><td>{_html_escape(f"{im_d:.1f}" if im_d is not None else "—")}</td>
            <td class="ok">≥ {T_REL_DE_IMP}%</td>
            <td>{"↑" if im_d is not None and im_d < T_REL_DE_IMP else "✓"}</td></tr>
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
        val_table += f"""<tr>
            <td>{_html_escape(ep)}</td><td>{_html_escape(st)}</td>
            <td>{_html_escape(f"{mae:.5f}" if mae is not None else "—")}</td>
            <td>{_html_escape(f"{bmae:.5f}" if bmae is not None else "—")}</td>
            <td>{_html_escape(f"{de:.3f}" if de is not None else "—")}</td>
            <td>{_html_escape(f"{q:.2f}" if q is not None else "—")}</td>
            <td>{_html_escape(f"{lt:.4f}" if lt is not None else "—")}</td>
        </tr>"""

    train_block = ""
    for tag, s in (report.get("train_summaries") or {}).items():
        train_block += f"""
        <h4>{_html_escape(tag)}</h4>
        <p>first={s['first']:.6f} → last={s['last']:.6f} · min={s['min']:.6f} max={s['max']:.6f} ·
        Δ={s['delta_last_minus_first']:.6f}</p>
        <p>ср. по первым 10% логов: {s['mean_head10pct']:.6f} · по последним 10%: {s['mean_tail10pct']:.6f}</p>
        """

    br = report.get("train_loss_breakdown_last_step")
    loss_br = ""
    if br:
        loss_br = f'<div class="card loss-list"><h3>Текстовый разбор loss @ step {br.get("step")}</h3><ul>'
        for name, d in sorted(br.get("components", {}).items(), key=lambda x: -x[1].get("weighted", 0)):
            loss_br += f"<li><code>{_html_escape(name)}</code>: raw={d['raw']:.6f}, w={d['weight']}, "
            loss_br += f"взвеш.={d['weighted']:.6f} ({d['pct_of_weighted_sum']:.1f}%)</li>"
        loss_br += "</ul></div>"

    progress_html = ""
    if best and best.get("boundary_mae_16") and best.get("baseline_boundary_mae_16"):
        bm, bb = float(best["boundary_mae_16"]), float(best["baseline_boundary_mae_16"])
        rel_m = (1.0 - bm / bb) * 100.0
        rel_d = 0.0
        bde, bd0 = best.get("boundary_ciede2000_16"), best.get("baseline_boundary_ciede2000_16")
        if bde is not None and bd0 and float(bd0) > 0:
            rel_d = (1.0 - float(bde) / float(bd0)) * 100.0
        progress_html = (
            "<h2 class='sec-h'>Прогресс к ориентирам (относит.)</h2><div class=\"card bars-card\">"
            + _bar_pct("Rel. MAE improv. %", rel_m, T_REL_MAE_IMP)
            + _bar_pct("Rel. ΔE improv. %", rel_d, T_REL_DE_IMP)
            + "</div>"
        )

    sc = score if score is not None else 0.0
    sc_bar = min(100.0, (sc / 95.0) * 100.0) if sc else 0.0
    gap_to_80 = max(0.0, 80.0 - sc) if sc else 80.0
    kpi_mae = f"{best['boundary_mae_16']:.4f}" if best and best.get("boundary_mae_16") is not None else "—"
    kpi_de = f"{best['boundary_ciede2000_16']:.2f}" if best and best.get("boundary_ciede2000_16") is not None else "—"
    n_val = len(rows)
    n_train_pts = len(data.get("train/loss/total") or [])

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
  .mark80{{position:absolute;left:80%;top:0;bottom:0;width:2px;background:var(--ok);z-index:2;box-shadow:0 0 6px #3ecf8e88;}}
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
      <p class="sub">Интерактивный отчёт · TensorBoard → графики, инфографика, цели 80–95%</p>
      <p class="meta"><span class="badge">TB</span><code>{_html_escape(source_file.name)}</code>
      <br />Шаги: {_html_escape(tmax)} · val точек: {n_val} · train лог-точек: {n_train_pts}
      {f" · ≈{bpe} батчей/эп." if bpe else ""}</p>
    </div>
    <div class="card scorebox" style="margin:0;">
      <div style="font-size:0.8rem;color:var(--muted);">Интегральная оценка</div>
      <div class="scorenum">{_html_escape(score) if score is not None else "—"}<span style="font-size:1.2rem;opacity:0.6">/95</span></div>
      <p style="margin:0.4rem 0 0;font-size:0.78rem;color:var(--muted);">≈{gap_to_80:.0f} пункта до ориентира «80» на шкале (условно)</p>
      <div class="track"><div class="fill"></div><div class="mark80"></div></div>
      <p style="font-size:0.72rem;margin:0;color:var(--muted);">Пунктир: эвристика «сильный» уровень. Не равен «80% продукта» без визуала.</p>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">Best val MAE@16</div><div class="val">{_html_escape(kpi_mae)}</div>
      <div class="hint">цель &lt; {T_MAE}</div></div>
    <div class="kpi"><div class="lbl">Best val ΔE@16</div><div class="val">{_html_escape(kpi_de)}</div>
      <div class="hint">цель &lt; {T_DE}</div></div>
    <div class="kpi"><div class="lbl">Val эпох в логе</div><div class="val">{_html_escape(n_val)}</div>
      <div class="hint">мало → высокая дисперсия</div></div>
    <div class="kpi"><div class="lbl">Оценка (эвр.)</div><div class="val">{_html_escape(f"{sc:.0f}" if sc else "—")}</div>
      <div class="hint">0–95 см.низ</div></div>
  </div>

  <div class="funnel">
    <h4>Инфографика: сейчас → целевой коридор</h4>
    <div class="fbar">
      <span>Оценка</span>
      <div class="ftrack"><div class="ffill now" style="width:{sc_bar:.1f}%"></div>
        <div class="ffill tgt" style="width:100%"></div></div>
      <span class="pct">{sc_bar:.0f}%</span>
    </div>
    <p class="annot">Нижняя полупрозрачная заливка = «100% пути к заполнению шкалы»; верх — текущий прогресс. Зелёная линия на шкале выше = 80% ширины «идеализации».</p>
  </div>

  <h2 class="sec-h">Val · динамика по эпохам <span>— пики/провалы; лучшая эпоха по MAE@16 — крупные точки слева</span></h2>
  <div class="callout">Слева — MAE@16 (масштаб 10⁻²); справа — ΔE@16 и (если логировалось) quality на отдельной оси. Пунктир — абс. цели: MAE = {T_MAE}, ΔE = {T_DE}. «Baseline» — измерения на входе (до гармона), без нормализации в один график.</div>
  <div class="row2">
    <div class="card chart-card"><div class="chart-canvas-h"><canvas id="cValMae"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-h"><canvas id="cValDe"></canvas></div></div>
  </div>

  <h2 class="sec-h">Train · кривые по глобальным шагам <span>— downsampled для плавности</span></h2>
  <div class="row2">
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainLoss"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainMae"></canvas></div></div>
  </div>
  <div class="row2" style="margin-top:1rem;">
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainLow"></canvas></div></div>
    <div class="card chart-card"><div class="chart-canvas-tall"><canvas id="cTrainGrad"></canvas></div></div>
  </div>

  <h2 class="sec-h">Разложение loss (последний лог-шаг) <span>— доля взвешенного вклада</span></h2>
  <div class="row3">
    <div class="card chart-card" style="min-height:280px"><div class="chart-canvas-h" style="height:260px"><canvas id="cDoughnut"></canvas></div></div>
    <div class="goal-panels" id="goalPanels"></div>
  </div>

  <h2 class="sec-h">Таблица: сейчас vs цель 80–95% (лучшая эп. по MAE@16)</h2>
  <div class="grid-tgt">
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>Метрика</th><th>Сейчас</th><th>Ориентир</th><th>Заметка</th></tr></thead>
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
          <th>ΔE@16</th><th>quality</th><th>val loss</th>
        </tr>
      </thead>
      <tbody>{val_table or "<tr><td colspan=7>—</td></tr>"}</tbody>
    </table>
  </div>

  <h2 class="sec-h">Train · текстовая сводка</h2>
  <div class="card">{train_block or "<p>Нет</p>"}</div>
  {loss_br}

  <h2 class="sec-h">Инсайты</h2>
  <ul class="ins">
    {"".join(f"<li>{_html_escape(s)}</li>" for s in insights)}
  </ul>
  <div class="callout">Если графики пустые: проверьте консоль (F12). При <code>file://</code> CDN может быть заблокирован — выполните <code>python -m http.server 8765</code> в папке с HTML и откройте <code>http://localhost:8765/…</code></div>

  <h2 class="sec-h">Мета (best_epochs)</h2>
  <div class="card"><pre style="white-space:pre-wrap;font-size:0.78rem;">{_html_escape(str(report.get("best_epochs")))}</pre></div>

  <p class="note">
    Оценка: 35 + 0.45×(rel MAE%) + 0.35×(rel ΔE%) + бонус, cap 0..95.
    <br />Запуск: <code>python scripts/harmonizer_metrics_dashboard.py</code>
  </p>
</div>

<script type="application/json" id="tb-payload">{ch_json}</script>
<script>
(function() {{
  const P = JSON.parse(document.getElementById('tb-payload').textContent);
  const Tm = (P.T_mae != null) ? P.T_mae : 0.02;
  const Td = (P.T_de != null) ? P.T_de : 2.5;
  const common = (title) => ({{ responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b9bb4', font: {{ size: 10 }} }} }},
      title: {{ display: !!title, text: title || '', color: '#6ec8ff', font: {{ size: 12 }} }}
    }} }});

  // --- Val: отдельные шкалы (MAE и ΔE несоизмеримы) ---
  const ve = P.val_epochs || [];
  if (ve.length) {{
    const ep = ve.map((r) => r.ep);
    const mae = ve.map((r) => r.mae16);
    const bmae = ve.map((r) => r.baseline_mae);
    const de = ve.map((r) => r.de16);
    const bde = ve.map((r) => r.baseline_de);
    const qu = ve.map((r) => r.quality);
    const hasQ = qu.some((x) => x != null && !isNaN(x));
    const bestIdx = P.best_ep_mae != null ? ve.findIndex((r) => r.ep === P.best_ep_mae) : -1;
    const rBest = mae.map((v, i) => (i === bestIdx ? 7 : 2));
    const tgtM = ep.map(() => Tm);
    const tgtD = ep.map(() => Td);
    new Chart(document.getElementById('cValMae'), {{
      type: 'line',
      data: {{
        labels: ep,
        datasets: [
          {{ label: 'MAE@16 (полоса)', data: mae, borderColor: '#4a9fe6', backgroundColor: '#4a9fe625', yAxisID: 'y', tension: 0.2, pointRadius: rBest, borderWidth: 2 }},
          {{ label: 'Baseline MAE@16 (вход)', data: bmae, borderColor: '#5a6a7a', borderDash: [6,4], fill: false, yAxisID: 'y', pointRadius: 0, borderWidth: 1.2 }},
          {{ label: 'Цель (абс.)', data: tgtM, borderColor: '#3ecf8e', borderDash: [4,4], pointRadius: 0, yAxisID: 'y', borderWidth: 1.5, fill: false }}
        ]
      }},
      options: {{ ...common('Val · MAE@16 (лучшая эп. — крупные точки)'),
        scales: {{ x: {{ ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          y: {{ position: 'left', title: {{ display: true, text: 'MAE@16', color: '#8b9bb4' }}, ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }} }} }}
    }});

    const deM = de.map((v, i) => (i === bestIdx ? 6 : 2));
    const deSets = [
      {{ label: 'ΔE@16 (CIEDE2000, полоса)', data: de, borderColor: '#3ecf8e', yAxisID: 'y', tension: 0.2, pointRadius: deM, borderWidth: 2 }},
      {{ label: 'Baseline ΔE@16 (вход)', data: bde, borderColor: '#6a8a6a', borderDash: [4,3], yAxisID: 'y', pointRadius: 0, borderWidth: 1.2 }},
      {{ label: 'Цель ΔE (абс.)', data: tgtD, borderColor: '#e85d75', borderDash: [4,4], pointRadius: 0, yAxisID: 'y', borderWidth: 1.5, fill: false }}
    ];
    if (hasQ) deSets.push({{ label: 'quality (эвр.)', data: qu, borderColor: '#e8b44c', yAxisID: 'y1', tension: 0.2, pointRadius: 2, borderWidth: 1.5 }});
    new Chart(document.getElementById('cValDe'), {{
      type: 'line',
      data: {{ labels: ep, datasets: deSets }},
      options: {{ ...common('Val · ΔE@16' + (hasQ ? ' + quality' : '')),
        scales: {{
          x: {{ ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          y: {{ position: 'left', title: {{ display: true, text: 'ΔE / CIEDE', color: '#8b9bb4' }}, ticks: {{ color: '#8b9bb4' }}, grid: {{ color: '#2a3a4f' }} }},
          ...(hasQ ? {{ y1: {{ position: 'right', title: {{ display: true, text: 'quality', color: '#e8b44c' }}, ticks: {{ color: '#e8b44c' }}, grid: {{ display: false }} }} }} : {{}})
        }} }}
    }});
  }}

  // Train charts
  const mkLine = (id, data, col, yLabel) => {{
    if (!data || !data.length) return;
    new Chart(document.getElementById(id), {{
      type: 'line',
      data: {{ datasets: [{{ label: yLabel, data, borderColor: col, pointRadius: 0, borderWidth: 1.2, fill: true, backgroundColor: col + '18', tension: 0.15 }}] }},
      options: {{ ...common(yLabel), scales: {{ x: {{ type: 'linear', ticks: {{ color: '#8b9bb4' }} }}, y: {{ ticks: {{ color: '#8b9bb4' }} }} }} }}
    }});
  }};
  mkLine('cTrainLoss', P.train_loss, '#4a9fe6', 'train/loss/total');
  mkLine('cTrainMae', P.train_mae16, '#6ec8ff', 'train/metric/boundary_mae_16');
  mkLine('cTrainLow', P.train_lowfreq, '#3ecf8e', 'train/metric/lowfreq_mae');
  mkLine('cTrainGrad', P.train_grad, '#e8b44c', 'train/metric/gradient_mae');

  // Doughnut
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

  // Goal panels (distance to 80–95% targets)
  const gp = document.getElementById('goalPanels');
  if (gp && (P.goals || []).length) {{
    P.goals.forEach((g) => {{
      const d = 100 * (1 - g.progress);
      const el = document.createElement('div');
      el.className = 'gpanel' + (d < 25 ? ' ok' : '');
      el.innerHTML = '<div class="t">' + g.label + '</div><div class="b"><i style="width:' + Math.min(100, d) + '%"></i></div><div class="t" style="margin-top:0.25rem">дистанция до цели: ~' + d.toFixed(0) + '% (условн.)</div>';
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

    latest = _find_latest_tfevent_file(args.logdir)
    if not latest:
        html_str = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Нет логов</title></head>
        <body style="font-family:sans-serif;padding:2rem;background:#0f1419;color:#e8edf4">
        <p>Нет <code>events.out.tfevents.*</code> в <code>{_html_escape(args.logdir)}</code></p></body></html>"""
        out = args.output or Path(tempfile.mkstemp(suffix="_harmonizer_dashboard.html", prefix="tb_")[1])
        out.write_text(html_str, encoding="utf-8")
        print("No tfevents. Wrote:", out)
        if not args.no_browser:
            webbrowser.open(out.as_uri())
        return

    data = _merge_runs(_find_event_files(latest))
    report = analyze(data)
    best = _best_val_row(report.get("val_epochs") or [])
    score = _integrated_score_0_95(best) if best else None
    html_str = build_html(report, data, latest, best, score)

    if args.output:
        out: Path = args.output
    else:
        import os

        fd, name = tempfile.mkstemp(suffix="_harmonizer_dashboard.html", prefix="tb_")
        os.close(fd)
        out = Path(name)
    out.write_text(html_str, encoding="utf-8")
    print("TensorBoard file:", latest)
    print("Dashboard HTML:", out.resolve())
    if not args.no_browser:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
