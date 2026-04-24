import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import augment_effects as ae
from build_dataset import load_and_square
from seam_canvas_v2 import CENTER

DATASET_ROOT = Path("outputs/dataset_v2")

SPEC_REGIMES = {
    "same_exact": 38, "same_degraded": 17, "mixed_compatible": 15, "mixed_hard": 10,
    "procedural_exact": 8, "procedural_hard": 5, "centerinsert": 7,
}
SPEC_NEIGHBORS = {
    "1": 8, "2": 18, "3": 20, "4": 22, "5": 14, "6": 10, "7": 6, "8": 2,
}
SPEC_DIFFICULTY = {"easy": 27, "medium": 55, "hard": 18}
SPEC_SPLITS = {"train": 80, "val": 12, "benchmark": 8}

_SOURCE_DIRS = {
    "inputs": Path("inputs"),
    "outputs/procedural": Path("outputs/procedural"),
    "output/unsplash": Path("output/unsplash"),
}

# ── cache ────────────────────────────────────────────────────────────────────
_meta_cache: Dict[str, dict] = {}
_list_cache: List[dict] = []
_file_cache: Dict[str, dict] = {}
_report_cache: Optional[dict] = None
_dataset_size_mb: float = 0.0
# Latest mtime among metadata/*.json — when the dataset is rebuilt, refresh cache without server restart.
_metadata_cache_mtime: float = 0.0


def _metadata_dir_max_mtime() -> float:
    meta_dir = DATASET_ROOT / "metadata"
    if not meta_dir.is_dir():
        return 0.0
    mt = 0.0
    for p in meta_dir.glob("*.json"):
        try:
            mt = max(mt, p.stat().st_mtime)
        except OSError:
            pass
    return mt


def _dataset_invalidation_mtime() -> float:
    """Bump when metadata or report changes (typical full rebuild)."""
    m = _metadata_dir_max_mtime()
    rp = DATASET_ROOT / "report.json"
    try:
        if rp.is_file():
            m = max(m, rp.stat().st_mtime)
    except OSError:
        pass
    return m


def _refresh_meta_cache_if_stale() -> None:
    """Reload metadata (and file stats) after build_dataset overwrites JSON/PNGs."""
    global _metadata_cache_mtime
    cur = _dataset_invalidation_mtime()
    if cur <= _metadata_cache_mtime:
        return
    _load_meta_cache()
    _load_report()
    _metadata_cache_mtime = cur


def _load_meta_cache():
    global _meta_cache, _list_cache, _file_cache, _dataset_size_mb
    _meta_cache.clear()
    _file_cache.clear()
    meta_dir = DATASET_ROOT / "metadata"
    if not meta_dir.exists():
        _list_cache = []
        return
    for p in sorted(meta_dir.glob("*.json")):
        with open(p) as f:
            m = json.load(f)
        _meta_cache[p.stem] = m
    _list_cache = [
        {
            "id": sid,
            "regime": m.get("source_regime", ""),
            "split": m.get("split", ""),
            "difficulty": m.get("difficulty", ""),
            "mask_size": m.get("mask_size", 0),
            "caption_family": m.get("caption_family", ""),
            "effects": len(m.get("effects_applied", [])),
            "unique_sources": m.get("unique_source_file_count"),
            "max_unique_cap": m.get("max_unique_sources_configured"),
            "idx_sim_mean": (m.get("index_similarity_summary") or {}).get(
                "index_similarity_mean"
            ),
        }
        for sid, m in _meta_cache.items()
    ]
    total_bytes = 0
    try:
        from PIL import Image as _PILImage
    except ImportError:
        _PILImage = None
    for sid in _meta_cache:
        entry: Dict[str, dict] = {}
        for kind in ("inputs", "targets", "masks"):
            p = DATASET_ROOT / kind / f"{sid}.png"
            sz = None
            dims = None
            cache_ver = None
            try:
                st = p.stat()
                sz = round(st.st_size / 1024, 1)
                total_bytes += st.st_size
                cache_ver = int(st.st_mtime)
            except OSError:
                pass
            if _PILImage:
                try:
                    with _PILImage.open(p) as im:
                        dims = list(im.size)
                except Exception:
                    pass
            entry[kind] = {
                "size_kb": sz,
                "width": dims[0] if dims else None,
                "height": dims[1] if dims else None,
                "cache_ver": cache_ver,
            }
        _file_cache[sid] = entry
    _dataset_size_mb = round(total_bytes / (1024 * 1024), 1)
    for row in _list_cache:
        fc = _file_cache.get(row["id"], {})
        acc = [
            fc.get(k, {}).get("cache_ver")
            for k in ("inputs", "targets", "masks")
        ]
        acc = [v for v in acc if v is not None]
        row["thumb_v"] = max(acc) if acc else 0


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _metadata_cache_mtime
    _load_meta_cache()
    _load_report()
    _metadata_cache_mtime = _dataset_invalidation_mtime()
    yield


app = FastAPI(title="Seam LoRA Studio Viewer", lifespan=lifespan)

if DATASET_ROOT.exists():
    app.mount("/data", StaticFiles(directory=str(DATASET_ROOT)), name="data")
for _mount_path, _dir in _SOURCE_DIRS.items():
    if _dir.exists():
        app.mount(
            f"/src/{_mount_path}",
            StaticFiles(directory=str(_dir)),
            name=f"src-{_mount_path.replace('/', '-')}",
        )


def _load_report() -> dict:
    global _report_cache
    rp = DATASET_ROOT / "report.json"
    if rp.exists():
        with open(rp) as f:
            _report_cache = json.load(f)
    return _report_cache or {}


def _allowed_source_roots() -> List[Path]:
    cwd = Path.cwd().resolve()
    roots: List[Path] = []
    for sub in ("output/unsplash", "outputs/procedural", "inputs"):
        r = (cwd / sub).resolve()
        if r.is_dir():
            roots.append(r)
    return roots


def _resolve_valid_source_file(rel: str) -> Path:
    """Reject traversal; file must live under a known source root (same as static /src/ mounts)."""
    rel_n = rel.replace("\\", "/").lstrip("/")
    p = Path(rel_n)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, "invalid path")
    full = (Path.cwd() / p).resolve()
    allowed = _allowed_source_roots()
    if not allowed:
        raise HTTPException(503, "no source directories mounted")
    ok = False
    for root in allowed:
        try:
            full.relative_to(root)
            ok = True
            break
        except ValueError:
            continue
    if not ok:
        raise HTTPException(403, "path not under allowed source directories")
    if not full.is_file():
        raise HTTPException(404, "file not found")
    return full


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/samples")
async def get_samples():
    _refresh_meta_cache_if_stale()
    return _list_cache


@app.get("/api/sample/{sample_id}")
async def get_sample_detail(sample_id: str):
    _refresh_meta_cache_if_stale()
    m = _meta_cache.get(sample_id)
    if not m:
        raise HTTPException(404, "sample not found")

    cap_path = DATASET_ROOT / "captions" / f"{sample_id}.txt"
    caption = cap_path.read_text() if cap_path.exists() else ""
    files = _file_cache.get(sample_id, {})

    return {"metadata": m, "caption": caption, "files": files}


@app.get("/api/report")
async def get_report():
    _refresh_meta_cache_if_stale()
    r = dict(_report_cache) if _report_cache else {}
    r["dataset_size_mb"] = _dataset_size_mb
    return r


@app.get("/api/dataset-build")
async def get_dataset_build():
    """Orchestrator snapshot from report.json (after running build_dataset.py)."""
    r = _load_report()
    return r.get("dataset_build") or {}


@app.get("/api/stats")
async def get_stats():
    """Compute live stats from metadata cache (fallback if no report.json)."""
    _refresh_meta_cache_if_stale()
    regimes: Counter = Counter()
    neighbors: Counter = Counter()
    effects: Counter = Counter()
    effect_families: Counter = Counter()
    difficulties: Counter = Counter()
    splits: Counter = Counter()
    captions: Counter = Counter()
    origins: Counter = Counter()
    scene_ids: set = set()

    for m in _meta_cache.values():
        regimes[m.get("source_regime", "?")] += 1
        neighbors[str(m.get("neighbor_count", 0))] += 1
        difficulties[m.get("difficulty", "?")] += 1
        splits[m.get("split", "?")] += 1
        captions[m.get("caption_family", "?")] += 1
        for fx in m.get("effects_applied", []):
            effects[fx] += 1
        for fam in m.get("effect_family", []):
            effect_families[fam] += 1
        for o in m.get("source_origins", []):
            origins[o] += 1
        for sid in m.get("source_scene_ids", []):
            scene_ids.add(sid)

    return {
        "total": len(_meta_cache),
        "unique_scenes": len(scene_ids),
        "splits": dict(splits),
        "regimes": dict(regimes),
        "neighbors": dict(neighbors),
        "effects": dict(effects),
        "effect_families": dict(effect_families),
        "difficulties": dict(difficulties),
        "caption_families": dict(captions),
        "origins": dict(origins),
    }


@app.get("/api/config")
async def get_config():
    return {
        "regimes": SPEC_REGIMES,
        "neighbors": SPEC_NEIGHBORS,
        "difficulty": SPEC_DIFFICULTY,
        "splits": SPEC_SPLITS,
        "training_center_size": CENTER,
    }


def _preview_path_allowed_for_sample(rel_norm: str, meta: dict) -> bool:
    sp = meta.get("source_paths") or {}
    paths = {str(p).replace("\\", "/") for p in sp.values()}
    cs = meta.get("center_source")
    if cs:
        paths.add(str(cs).replace("\\", "/"))
    return rel_norm in paths


@app.get("/api/source_preview")
async def source_preview(path: str, sample_id: Optional[str] = None):
    """
    Same center-square crop + resize as build_dataset.load_and_square (what strips are cut from).
    When ``sample_id`` is set and matches metadata, replays ``effects_applied`` like
    ``process_one_sample`` so previews match input/target for degraded samples.
    """
    _refresh_meta_cache_if_stale()
    src = _resolve_valid_source_file(path)
    arr = load_and_square(src, CENTER)
    rel_norm = path.replace("\\", "/").lstrip("/")
    if sample_id:
        meta = _meta_cache.get(sample_id)
        if meta and _preview_path_allowed_for_sample(rel_norm, meta):
            effects = meta.get("effects_applied") or []
            if effects:
                pil = Image.fromarray(arr)
                for name in effects:
                    fn = ae.EFFECT_NAMES.get(name)
                    if fn is not None:
                        pil = fn(pil)
                arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ── HTML ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seam LoRA Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── tokens ─────────────────────────────────────────────── */
:root{
  --bg:#09090b;--panel:#111114;--panel2:#18181b;
  --accent:#6366f1;--accent-dim:#4f46e5;--accent-soft:rgba(99,102,241,.12);
  --green:#10b981;--red:#ef4444;--amber:#f59e0b;--cyan:#06b6d4;
  --text:#e2e8f0;--text2:#a1a1aa;--text3:#52525b;
  --border:#27272a;--border2:#3f3f46;
  --radius:10px;--radius-lg:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:10px}
::-webkit-scrollbar-track{background:transparent}

/* ── sidebar ────────────────────────────────────────────── */
#sidebar{width:300px;flex-shrink:0;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column}
.sb-head{padding:16px 18px;border-bottom:1px solid var(--border)}
.sb-head h1{font-size:1rem;font-weight:700;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sb-head p{font-size:.65rem;color:var(--text3);margin-top:3px;letter-spacing:.06em;text-transform:uppercase}

.sb-tabs{display:flex;gap:4px;padding:8px 10px;border-bottom:1px solid var(--border)}
.tab-btn{flex:1;padding:7px 0;border:none;background:transparent;color:var(--text2);cursor:pointer;border-radius:6px;font-size:.72rem;font-weight:600;letter-spacing:.04em;transition:.15s}
.tab-btn:hover{background:var(--panel2)}
.tab-btn.on{background:var(--accent-soft);color:var(--accent)}

.sb-search{padding:8px 10px;border-bottom:1px solid var(--border)}
.sb-search input{width:100%;padding:7px 10px;border:1px solid var(--border);background:var(--panel2);border-radius:6px;color:var(--text);font-size:.78rem;outline:none;font-family:inherit}
.sb-search input::placeholder{color:var(--text3)}
.sb-search input:focus{border-color:var(--accent)}

.sb-filters{padding:6px 10px;border-bottom:1px solid var(--border);display:flex;gap:4px;flex-wrap:wrap}
.fb{padding:3px 8px;border:1px solid var(--border);border-radius:20px;font-size:.62rem;color:var(--text2);cursor:pointer;background:transparent;transition:.15s;font-weight:500;text-transform:uppercase}
.fb:hover,.fb.on{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}

#sample-list{flex:1;overflow-y:auto;padding:6px}
.si{padding:8px 12px;border-radius:8px;cursor:pointer;margin-bottom:2px;display:flex;align-items:center;gap:10px;transition:.12s;font-size:.8rem}
.si:hover{background:var(--panel2)}
.si.on{background:var(--accent);color:#fff}
.si .sid{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:500;min-width:48px}
.si .sregime{font-size:.58rem;font-weight:600;text-transform:uppercase;opacity:.6;flex:1;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.si .sdiff{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.si .sdiff.easy{background:var(--green)}.si .sdiff.medium{background:var(--amber)}.si .sdiff.hard{background:var(--red)}
.sb-status{padding:8px 12px;border-top:1px solid var(--border);font-size:.65rem;color:var(--text3);text-align:center}

/* ── main ───────────────────────────────────────────────── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.pane{display:none;flex:1;overflow-y:auto;padding:28px 32px}
.pane.on{display:block}

/* ── sample viewer ──────────────────────────────────────── */
.img-strip{display:grid;grid-template-columns:2fr 1fr;grid-template-rows:1fr 1fr 1fr;gap:12px;margin-bottom:24px;height:min(70vh,calc(100vw - 380px))}
.img-strip .cmp-card{grid-column:1;grid-row:1/4;min-height:0}
.img-strip>.img-card:not(.cmp-card){min-height:0}
.img-strip>.img-card:not(.cmp-card) .img-box{aspect-ratio:auto;flex:1;min-height:0;overflow:hidden}
.img-strip>.img-card:not(.cmp-card) .img-box img{width:100%;height:100%;object-fit:contain}
.img-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;display:flex;flex-direction:column}
.img-head{padding:8px 14px;display:flex;justify-content:space-between;align-items:center;background:var(--panel2);border-bottom:1px solid var(--border);font-size:.7rem}
.img-head .lbl{font-weight:600;text-transform:uppercase;color:var(--text2);letter-spacing:.04em}
.img-head .info{color:var(--text3);font-family:'JetBrains Mono',monospace;font-size:.62rem}
.img-box{aspect-ratio:1;background-image:
  linear-gradient(45deg,#18181b 25%,transparent 25%),
  linear-gradient(-45deg,#18181b 25%,transparent 25%),
  linear-gradient(45deg,transparent 75%,#18181b 75%),
  linear-gradient(-45deg,transparent 75%,#18181b 75%);
  background-size:16px 16px;background-position:0 0,0 8px,8px -8px,-8px 0;background-color:#111;
  display:flex;align-items:center;justify-content:center;position:relative;cursor:zoom-in}
.img-box img{max-width:100%;max-height:100%;display:block}
.img-box img.pixel{image-rendering:pixelated}

/* comparison toggles */
.cmp-toggles{display:flex;gap:4px}
.cmp-tog{border:1px solid var(--border);background:transparent;color:var(--text3);padding:3px 10px;border-radius:20px;font-size:.62rem;font-weight:600;text-transform:uppercase;letter-spacing:.03em;cursor:pointer;transition:.15s;font-family:inherit}
.cmp-tog:hover{border-color:var(--accent);color:var(--text2)}
.cmp-tog.on{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}

/* opacity control */
.cmp-opacity{display:flex;align-items:center;gap:5px;margin-left:8px}
.cmp-opa-icon{font-size:.7rem;color:var(--text3)}
.cmp-opa-val{font-size:.58rem;font-family:'JetBrains Mono',monospace;color:var(--text3);min-width:28px}
.cmp-opacity input[type=range]{-webkit-appearance:none;appearance:none;width:60px;height:4px;background:var(--border);border-radius:2px;outline:none;cursor:pointer}
.cmp-opacity input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--accent);border:1px solid #fff;cursor:pointer}
.cmp-opacity input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--accent);border:1px solid #fff;cursor:pointer}

/* comparison slider */
.cmp-card .cmp-wrap{flex:1;min-height:0}
.cmp-wrap{position:relative;overflow:hidden;cursor:col-resize;user-select:none;-webkit-user-select:none;
  background-image:
  linear-gradient(45deg,#18181b 25%,transparent 25%),
  linear-gradient(-45deg,#18181b 25%,transparent 25%),
  linear-gradient(45deg,transparent 75%,#18181b 75%),
  linear-gradient(-45deg,transparent 75%,#18181b 75%);
  background-size:16px 16px;background-position:0 0,0 8px,8px -8px,-8px 0;background-color:#111}
.cmp-wrap img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}
.cmp-wrap .cmp-input{clip-path:inset(0 50% 0 0)}
.cmp-line{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;z-index:5;pointer-events:none;transform:translateX(-1px)}
.cmp-pin{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:28px;height:28px;border-radius:50%;background:var(--accent);border:2px solid #fff;z-index:6;cursor:col-resize;display:flex;align-items:center;justify-content:center;box-shadow:0 0 12px rgba(0,0,0,.5)}
.cmp-pin::before{content:'◀▶';font-size:8px;color:#fff;letter-spacing:2px}
.cmp-labels{position:absolute;bottom:8px;left:0;right:0;display:flex;justify-content:space-between;padding:0 12px;z-index:4;pointer-events:none}
.cmp-labels span{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:rgba(255,255,255,.7);background:rgba(0,0,0,.45);padding:2px 8px;border-radius:10px}

.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:1100px){.detail-grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius-lg);padding:18px;display:flex;flex-direction:column}
.card h3{font-size:.72rem;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;font-weight:600}
.caption-text{line-height:1.65;font-size:.88rem;color:#d1d5db;white-space:pre-wrap;word-break:break-word}

/* metadata table */
.meta-tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.meta-tbl tr{border-bottom:1px solid var(--border)}
.meta-tbl tr:last-child{border:none}
.meta-tbl td{padding:6px 0;vertical-align:top}
.meta-tbl .mk{color:var(--text2);font-weight:500;width:40%;padding-right:10px;white-space:nowrap}
.meta-tbl .mv{font-family:'JetBrains Mono',monospace;font-size:.72rem;word-break:break-all;color:var(--text)}

.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.62rem;font-weight:600;margin:1px 2px;text-transform:uppercase}
.tag.regime{background:var(--accent-soft);color:var(--accent)}
.tag.split-train{background:rgba(16,185,129,.12);color:var(--green)}
.tag.split-val{background:rgba(245,158,11,.12);color:var(--amber)}
.tag.split-benchmark{background:rgba(6,182,212,.12);color:var(--cyan)}
.tag.easy{background:rgba(16,185,129,.12);color:var(--green)}
.tag.medium{background:rgba(245,158,11,.12);color:var(--amber)}
.tag.hard{background:rgba(239,68,68,.12);color:var(--red)}

.empty-s{height:80vh;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--text3);gap:8px}
.empty-s kbd{background:var(--panel2);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-size:.72rem;font-family:'JetBrains Mono',monospace}

/* ── dashboard ──────────────────────────────────────────── */
.dash-hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}
.tile{background:var(--panel);border:1px solid var(--border);padding:16px;border-radius:var(--radius-lg);text-align:center}
.tile-v{font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--accent);line-height:1}
.tile-l{font-size:.6rem;color:var(--text3);text-transform:uppercase;letter-spacing:.06em;margin-top:6px;font-weight:600}
.tile-sub{font-size:.58rem;color:var(--text3);margin-top:4px}
.tile.ok .tile-v{color:var(--green)}.tile.warn .tile-v{color:var(--amber)}.tile.bad .tile-v{color:var(--red)}

.dash-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.wg{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;display:flex;flex-direction:column}
.wg h2{font-size:.85rem;font-weight:600;margin-bottom:4px}
.wg .wg-hint{font-size:.68rem;color:var(--text3);margin-bottom:14px;line-height:1.4}

.bar-row{margin-bottom:10px}
.bar-labels{display:flex;justify-content:space-between;font-size:.72rem;margin-bottom:4px}
.bar-labels .bl-name{color:var(--text2)}.bar-labels .bl-val{font-weight:600;font-family:'JetBrains Mono',monospace}
.bar-track{height:6px;background:var(--panel2);border-radius:3px;overflow:visible;position:relative}
.bar-fill{height:100%;border-radius:3px;background:var(--accent);transition:width .4s ease-out}
.bar-target{position:absolute;top:-3px;height:12px;width:2px;background:var(--red);border-radius:1px;z-index:2}

.kv-row{display:flex;justify-content:space-between;padding:5px 0;font-size:.75rem;border-bottom:1px solid var(--border)}
.kv-row:last-child{border:none}
.kv-row .k{color:var(--text2)}.kv-row .v{font-family:'JetBrains Mono',monospace;font-weight:500}

/* source grid */
.src-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.src-fx-banner{font-size:.65rem;color:var(--amber);margin:0 0 10px;line-height:1.35}
.src-help{font-size:.68rem;color:var(--text2);margin:0 0 12px;line-height:1.45;max-width:900px}
.src-help code{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--cyan)}
.src-help strong{color:var(--text)}
.src-card{background:var(--panel2);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;display:flex;flex-direction:column;position:relative}
.src-card img{width:100%;aspect-ratio:1;object-fit:cover;display:block;cursor:zoom-in}
.src-full-link{font-size:.58rem;color:var(--accent);margin:6px 8px 0 auto;text-decoration:none}
.src-full-link:hover{text-decoration:underline}
.src-crop-hint{display:block;font-size:.58rem;color:var(--text3);padding:0 8px 4px}
.src-label{padding:6px 8px;font-size:.62rem;color:var(--text2);font-family:'JetBrains Mono',monospace;display:flex;justify-content:space-between;align-items:center}
.src-label .pos-tags{display:flex;gap:3px;flex-wrap:wrap}
.src-label .pos-tag{background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:10px;font-size:.55rem;font-weight:600;text-transform:uppercase}

/* gallery */
.gal-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px}
.gal-row .gal-cell{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;display:flex;flex-direction:column}
.gal-row .gal-cell img{width:100%;display:block;cursor:pointer}
.gal-head{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:var(--panel2);border-bottom:1px solid var(--border);font-size:.65rem}
.gal-head .gid{font-family:'JetBrains Mono',monospace;font-weight:500;color:var(--text)}
.gal-head .glabel{color:var(--text3);text-transform:uppercase;font-weight:600;font-size:.58rem}
.gal-tags{padding:5px 10px;display:flex;gap:4px;flex-wrap:wrap;border-top:1px solid var(--border)}

/* dataset build config (dashboard) */
.cfg-mega{margin-bottom:20px}
.cfg-mega .cfg-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.cfg-block h4{font-size:.7rem;color:var(--accent);margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.cfg-tbl{width:100%;border-collapse:collapse;font-size:.72rem}
.cfg-tbl td{padding:5px 0;border-bottom:1px solid var(--border);vertical-align:top}
.cfg-tbl tr:last-child td{border:none}
.cfg-tbl .cfg-k{color:var(--text2);width:44%;padding-right:10px;font-weight:500}
.cfg-tbl .cfg-v{font-family:'JetBrains Mono',monospace;font-size:.68rem;word-break:break-word;color:var(--text);line-height:1.35}
.cfg-note{font-size:.68rem;color:var(--text3);line-height:1.45;margin-bottom:12px}

/* mask grid */
.mask-grid{display:inline-grid;grid-template-columns:repeat(3,20px);grid-template-rows:repeat(3,20px);gap:2px;margin-top:4px}
.mask-cell{border-radius:3px;background:var(--panel2);border:1px solid var(--border)}
.mask-cell.filled{background:var(--accent);border-color:var(--accent-dim)}
.mask-cell.center{background:var(--panel2);border:1px dashed var(--text3)}

/* lightbox */
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:999;display:none;align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.on{display:flex}
.lightbox img{max-width:95vw;max-height:95vh;border-radius:8px;box-shadow:0 0 60px rgba(0,0,0,.6)}
.lb-label{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);color:#fff;font-size:.75rem;background:rgba(0,0,0,.6);padding:4px 14px;border-radius:20px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;z-index:1000;pointer-events:none}
</style>
</head>
<body>

<!-- ── sidebar ──────────────────────────────────────────── -->
<div id="sidebar">
  <div class="sb-head">
    <h1>Seam LoRA Studio</h1>
    <p>Dataset Viewer &amp; Dashboard</p>
  </div>
  <div class="sb-tabs">
    <button class="tab-btn on" data-tab="viewer">Samples</button>
    <button class="tab-btn" data-tab="gallery">Gallery</button>
    <button class="tab-btn" data-tab="dashboard">Dashboard</button>
  </div>
  <div class="sb-search"><input id="search" placeholder="Search id, regime, split…"></div>
  <div class="sb-filters" id="filters"></div>
  <div id="sample-list"></div>
  <div class="sb-status" id="sb-status"></div>
</div>

<!-- ── main ─────────────────────────────────────────────── -->
<div id="main">

  <!-- viewer -->
  <div id="pane-viewer" class="pane on">
    <div id="v-empty" class="empty-s">
      <p style="font-size:.9rem">Select a sample from the list</p>
      <p><kbd>↑</kbd> <kbd>↓</kbd> to navigate &nbsp; <kbd>G</kbd> gallery &nbsp; <kbd>D</kbd> dashboard</p>
    </div>
    <div id="v-content" style="display:none">
      <div class="img-strip">
        <div class="img-card cmp-card">
          <div class="img-head">
            <span class="cmp-toggles">
              <button class="cmp-tog on" data-kind="input">Input</button>
              <button class="cmp-tog on" data-kind="target">Target</button>
              <button class="cmp-tog" data-kind="mask">Mask</button>
            </span>
            <span class="cmp-opacity"><span class="cmp-opa-icon">◐</span><input type="range" id="cmp-opacity" min="0" max="100" value="100"><span class="cmp-opa-val" id="cmp-opa-val">100%</span></span>
            <span class="info" id="cmp-info"></span>
          </div>
          <div class="cmp-wrap" id="cmp-wrap">
            <img id="cmp-bg">
            <img id="cmp-fg" class="cmp-input">
            <div class="cmp-line" id="cmp-line"></div>
            <div class="cmp-pin" id="cmp-pin"></div>
            <div class="cmp-labels"><span id="cmp-lbl-l">Input</span><span id="cmp-lbl-r">Target</span></div>
          </div>
        </div>
        <div class="img-card">
          <div class="img-head"><span class="lbl">Input</span><span class="info" id="fi-input2"></span></div>
          <div class="img-box" onclick="openLightbox(this)"><img id="img-input-solo"></div>
        </div>
        <div class="img-card">
          <div class="img-head"><span class="lbl">Target</span><span class="info" id="fi-target2"></span></div>
          <div class="img-box" onclick="openLightbox(this)"><img id="img-target-solo"></div>
        </div>
        <div class="img-card">
          <div class="img-head"><span class="lbl">Mask</span><span class="info" id="fi-mask2"></span></div>
          <div class="img-box" onclick="openLightbox(this)"><img id="img-mask-solo" class="pixel"></div>
        </div>
      </div>
      <div class="detail-grid">
        <div class="card"><h3>Caption</h3><div id="caption" class="caption-text"></div></div>
        <div class="card"><h3>Metadata</h3><table class="meta-tbl" id="meta-tbl"></table></div>
      </div>
      <div id="sources-section" style="margin-top:24px;display:none">
        <div class="card"><h3>Source Images</h3><div id="sources-grid" class="src-grid"></div></div>
      </div>
    </div>
  </div>

  <!-- gallery -->
  <div id="pane-gallery" class="pane">
    <div id="gallery-content"></div>
    <div id="gallery-load" style="text-align:center;padding:20px;display:none">
      <button onclick="loadMoreGallery()" style="padding:8px 24px;border:1px solid var(--border);background:var(--panel2);color:var(--text);border-radius:8px;cursor:pointer;font-size:.8rem">Load more</button>
    </div>
  </div>

  <!-- dashboard -->
  <div id="pane-dashboard" class="pane">
    <div class="dash-hero" id="dash-hero"></div>
    <div class="dash-grid" id="dash-grid"></div>
  </div>

</div>

<!-- lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()"><img id="lb-img"><div class="lb-label" id="lb-label"></div></div>

<script>
/* ── state ──────────────────────────────────────────────── */
let allSamples=[], filtered=[], idx=-1, spec=null, report=null;
let activeFilter=null, lbIndex=-1;
const LB_KINDS=['input','target','mask'];

/* ── init ───────────────────────────────────────────────── */
(async()=>{
  [allSamples, spec, report] = await Promise.all([
    f('/api/samples'), f('/api/config'), f('/api/report'),
  ]);
  filtered=[...allSamples];
  renderList();
  buildFilters();
  updateStatus();
})();
async function f(u){return(await fetch(u)).json()}
function updateStatus(){
  const pos=idx>=0?`${idx+1} / ${filtered.length}`:`${filtered.length} samples`;
  document.getElementById('sb-status').textContent=pos;
}

/* ── tabs ───────────────────────────────────────────────── */
document.querySelectorAll('.tab-btn').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  document.getElementById('pane-'+b.dataset.tab).classList.add('on');
  if(b.dataset.tab==='dashboard') renderDashboard();
  if(b.dataset.tab==='gallery') renderGalleryTab();
}));

/* ── search & filter ────────────────────────────────────── */
document.getElementById('search').addEventListener('input', e=>{
  const q=e.target.value.toLowerCase();
  filtered=allSamples.filter(s=>{
    const text=`${s.id} ${s.regime} ${s.split} ${s.difficulty} ${s.caption_family} ${s.unique_sources||''} ${s.idx_sim_mean||''} ${s.max_unique_cap||''}`.toLowerCase();
    if(!text.includes(q))return false;
    if(activeFilter&&s.regime!==activeFilter)return false;
    return true;
  });
  renderList(); idx=-1; updateStatus();
});
function buildFilters(){
  const regimes=[...new Set(allSamples.map(s=>s.regime))].sort();
  document.getElementById('filters').innerHTML=regimes.map(r=>`<button class="fb" data-r="${r}">${r.replace(/_/g,' ')}</button>`).join('');
  document.querySelectorAll('.fb').forEach(b=>b.addEventListener('click',()=>{
    if(activeFilter===b.dataset.r){activeFilter=null;b.classList.remove('on')}
    else{document.querySelectorAll('.fb').forEach(x=>x.classList.remove('on'));activeFilter=b.dataset.r;b.classList.add('on')}
    document.getElementById('search').dispatchEvent(new Event('input'));
  }));
}
function renderList(){
  const el=document.getElementById('sample-list');
  el.innerHTML=filtered.map((s,i)=>`<div class="si" data-i="${i}" id="si-${i}">
    <span class="sdiff ${s.difficulty}"></span>
    <span class="sid">${s.id}</span>
    <span class="sregime">${s.regime}</span>
  </div>`).join('');
  el.querySelectorAll('.si').forEach(d=>d.addEventListener('click',()=>selectSample(+d.dataset.i)));
}

/* ── sample selection ───────────────────────────────────── */
let currentSid=null;
async function selectSample(i){
  if(i<0||i>=filtered.length)return;
  if(idx>=0)document.getElementById('si-'+idx)?.classList.remove('on');
  idx=i;
  const el=document.getElementById('si-'+idx);
  el?.classList.add('on');
  el?.scrollIntoView({behavior:'smooth',block:'nearest'});
  updateStatus();

  const sid=filtered[idx].id;
  currentSid=sid;
  const d=await f('/api/sample/'+sid);
  if(currentSid!==sid)return;

  document.getElementById('v-empty').style.display='none';
  document.getElementById('v-content').style.display='block';

  setImg('input',sid,d.files.inputs);
  setImg('target',sid,d.files.targets);
  setImg('mask',sid,d.files.masks);

  document.getElementById('caption').innerText=d.caption||'(no caption)';
  renderMeta(d.metadata);
  renderSources(d.metadata);
  updateCmpSlider();
  if(window._cmpReset)window._cmpReset();
}

const _imgUrls={}, _imgInfo={};
function setImg(kind,sid,info){
  const map={input:'inputs',target:'targets',mask:'masks'};
  let url='/data/'+map[kind]+'/'+sid+'.png';
  if(info&&info.cache_ver!=null)url+='?v='+info.cache_ver;
  _imgUrls[kind]=url;
  const parts=[];
  if(info&&info.width&&info.height)parts.push(info.width+'×'+info.height);
  if(info&&info.size_kb!=null)parts.push(info.size_kb<1024?(info.size_kb+'KB'):(info.size_kb/1024).toFixed(1)+'MB');
  _imgInfo[kind]=parts.join(' · ');
  const solo=document.getElementById('img-'+kind+'-solo');
  if(solo)solo.src=url;
  const fi2=document.getElementById('fi-'+kind+'2');
  if(fi2)fi2.textContent=_imgInfo[kind];
}

/* comparison toggle logic */
let cmpPair=['input','target'];
document.querySelectorAll('.cmp-tog').forEach(b=>b.addEventListener('click',()=>{
  const kind=b.dataset.kind;
  const isOn=b.classList.contains('on');
  if(isOn){
    if(cmpPair.length<=2) return;
  } else {
    if(cmpPair.length>=2) cmpPair.shift();
    cmpPair.push(kind);
  }
  document.querySelectorAll('.cmp-tog').forEach(x=>x.classList.toggle('on',cmpPair.includes(x.dataset.kind)));
  updateCmpSlider();
  if(window._cmpReset)window._cmpReset();
}));

function updateCmpSlider(){
  const [left,right]=cmpPair;
  const fg=document.getElementById('cmp-fg'), bg=document.getElementById('cmp-bg');
  fg.src=_imgUrls[left]||'';
  bg.src=_imgUrls[right]||'';
  fg.classList.toggle('pixel',left==='mask');
  bg.classList.toggle('pixel',right==='mask');
  const opa=document.getElementById('cmp-opacity').value/100;
  fg.style.opacity=opa;
  bg.style.opacity=opa;
  document.getElementById('cmp-lbl-l').textContent=left.charAt(0).toUpperCase()+left.slice(1);
  document.getElementById('cmp-lbl-r').textContent=right.charAt(0).toUpperCase()+right.slice(1);
  document.getElementById('cmp-info').textContent=(_imgInfo[left]||'')+' | '+(_imgInfo[right]||'');
}

document.getElementById('cmp-opacity').addEventListener('input',e=>{
  const v=e.target.value;
  document.getElementById('cmp-opa-val').textContent=v+'%';
  document.getElementById('cmp-fg').style.opacity=v/100;
  document.getElementById('cmp-bg').style.opacity=v/100;
});

const GRID_ORDER=['nw','n','ne','w','center','e','sw','s','se'];
function maskGrid(positions){
  const set=new Set(positions||[]);
  return '<div class="mask-grid">'+GRID_ORDER.map(p=>{
    if(p==='center')return '<div class="mask-cell center"></div>';
    return `<div class="mask-cell${set.has(p)?' filled':''}"></div>`;
  }).join('')+'</div>';
}

function renderMeta(m){
  const rows=[
    ['Sample ID',m.sample_id],
    ['Split',`<span class="tag split-${m.split}">${m.split}</span>`],
    ['Regime',`<span class="tag regime">${m.source_regime}</span>`],
    ['Difficulty',`<span class="tag ${m.difficulty}">${m.difficulty}</span>`],
    ['Center Mode',m.center_mode],
    ['Caption Family',m.caption_family],
    ['Mask',maskGrid(m.neighbors_visible)+' <span style="color:var(--text2);font-size:.7rem;margin-left:8px">'+
      (m.neighbors_visible?.join(', ')||'')+'</span>'],
    ['Neighbor Count',m.neighbor_count],
    ['Rotation',m.rotation_applied+'×90°'],
    ['Flip',m.flip_applied?'Yes':'No'],
    ['Effects',m.effects_applied?.length?m.effects_applied.join(', '):'none'],
    ['Effect Family',m.effect_family?.length?m.effect_family.join(', '):'—'],
    ['Compat Score',m.compatibility_score!=null?m.compatibility_score:'—'],
    ['Procedural Gen',m.procedural_generator||'—'],
    ['Image Size',m.size?m.size[0]+'×'+m.size[1]:'—'],
    ['Export square (px)',m.export_square_px!=null?m.export_square_px+'×'+m.export_square_px:'—'],
    ['Unique source files',m.unique_source_file_count!=null?String(m.unique_source_file_count):'—'],
    ['Max unique sources (cap)',m.max_unique_sources_configured!=null?String(m.max_unique_sources_configured):'— (unlimited)'],
    ['Target center',m.target_center_origin==='inserted'
      ?'<span class="tag regime">inserted</span> (separate center file)'
      :(m.target_center_origin==='synthesized'||!m.target_center_origin
        ?'<span class="tag split-train">synthesized</span> (synthesize_center, not a 6th photo)'
        :String(m.target_center_origin))],
    ['Training square (px)',m.training_square_px!=null?m.training_square_px+'×'+m.training_square_px:'—'],
    ['Sources',m.source_scene_ids?.join(', ')],
  ];
  const idx=m.index_similarity_summary;
  if(idx&&idx.pair_count>0){
    rows.push(['Index sim (mean)',idx.index_similarity_mean!=null?idx.index_similarity_mean:'—']);
    rows.push(['Index sim (min … max)',idx.index_similarity_min!=null&&idx.index_similarity_max!=null?`${idx.index_similarity_min} … ${idx.index_similarity_max}`:'—']);
    rows.push(['Index pairs scored',`${idx.pairs_scored}/${idx.pair_count}`]);
    rows.push(['Index mode',idx.index_mode||'—']);
    if(idx.fingerprint) rows.push(['Index fingerprint',idx.fingerprint.slice(0,24)+'…']);
    if(idx.similarity_definition) rows.push(['Index metric',`<span style="font-size:.68rem;line-height:1.4">${idx.similarity_definition}</span>`]);
    if(idx.pairs&&idx.pairs.length){
      const pairLines=idx.pairs.map(p=>{
        const s=p.similarity!=null?p.similarity:'—';
        return `${p.a_basename||''} ↔ ${p.b_basename||''}: <strong>${s}</strong>`;
      }).join('<br>');
      rows.push(['Index pairwise',`<div style="font-size:.68rem;line-height:1.45">${pairLines}</div>`]);
    }
  }
  document.getElementById('meta-tbl').innerHTML=rows.map(([k,v])=>`<tr><td class="mk">${k}</td><td class="mv">${v}</td></tr>`).join('');
}

/* ── comparison slider ──────────────────────────────────── */
(function(){
  const wrap=document.getElementById('cmp-wrap');
  const line=document.getElementById('cmp-line');
  const pin=document.getElementById('cmp-pin');
  const inp=document.getElementById('cmp-fg');
  let dragging=false;

  function setPos(pct){
    pct=Math.max(0,Math.min(100,pct));
    inp.style.clipPath='inset(0 '+(100-pct)+'% 0 0)';
    line.style.left=pct+'%';
    pin.style.left=pct+'%';
  }
  function pctFromEvent(e){
    const r=wrap.getBoundingClientRect();
    const clientX=e.touches?e.touches[0].clientX:e.clientX;
    return((clientX-r.left)/r.width)*100;
  }
  pin.addEventListener('mousedown',e=>{e.preventDefault();dragging=true});
  pin.addEventListener('touchstart',e=>{dragging=true},{passive:true});
  window.addEventListener('mousemove',e=>{if(dragging)setPos(pctFromEvent(e))});
  window.addEventListener('touchmove',e=>{if(dragging)setPos(pctFromEvent(e))},{passive:true});
  window.addEventListener('mouseup',()=>{dragging=false});
  window.addEventListener('touchend',()=>{dragging=false});
  wrap.addEventListener('click',e=>{if(!dragging)setPos(pctFromEvent(e))});
  wrap.addEventListener('dblclick',e=>{
    e.preventDefault();
    const pct=pctFromEvent(e);
    const kind=pct<50?cmpPair[0]:cmpPair[1];
    lbIndex=LB_KINDS.indexOf(kind);
    showLbImage();
    document.getElementById('lightbox').classList.add('on');
  });

  window._cmpReset=function(){
    setPos(50);
    const opaEl=document.getElementById('cmp-opacity');
    opaEl.value=100;
    document.getElementById('cmp-opa-val').textContent='100%';
    document.getElementById('cmp-fg').style.opacity=1;
    document.getElementById('cmp-bg').style.opacity=1;
  };
})();

/* ── source images on sample page ───────────────────────── */
function sourcesHelpHtml(m){
  const sp=m.source_paths||{};
  const uc=m.unique_source_file_count!=null
    ?m.unique_source_file_count
    :[...new Set(Object.values(sp))].length;
  const tc=m.training_square_px||(spec&&spec.training_center_size)||1024;
  const tco=m.target_center_origin||(m.center_source?'inserted':'synthesized');
  const r=m.source_regime||'';
  if(r==='same_exact'&&uc===1){
    return `<p class="src-help"><strong>same_exact</strong> — one file drives every strip. Strips are edge crops from a scaled view of this <strong>${tc}×${tc}</strong> center square (see <code>load_and_square</code> + <code>scale_to_canvas_fill</code>). They can look unlike the thumbnail middle — same photo, different regions. <strong>Target</strong> center is <code>synthesize_center</code> (not <code>centerinsert</code>); <strong>input</strong> uses a white hole for training.</p>`;
  }
  if(r==='procedural_exact'&&uc===1){
    return `<p class="src-help"><strong>procedural_exact</strong> — same layout as same_exact but procedural tiles: one generator image, strips + synthesized center.</p>`;
  }
  if(m.max_unique_sources_configured!=null){
    return `<p class="src-help"><strong>Patchwork cap</strong> — orchestrator drew at most <strong>N=${m.max_unique_sources_configured}</strong> distinct originals, then assigned sides from that pool. This sample uses <strong>${uc}</strong> unique file(s). Pairwise scores from <code>outputs/similarity_neighbors.json</code> (strip sources only) are in <strong>Metadata → Index …</strong>.</p>`;
  }
  if(uc>1){
    return `<p class="src-help"><strong>${uc} distinct files</strong> — positions map to different paths (see tags on cards). Preview is still the ${tc}×${tc} training square per file.</p>`;
  }
  if(tco==='inserted'&&m.center_source){
    return `<p class="src-help"><strong>centerinsert</strong> — center of <strong>target</strong> is blended with a separate image (<code>center_source</code>), tagged <span class="pos-tag">center</span> on its card.</p>`;
  }
  return '';
}
function renderSources(m){
  const sec=document.getElementById('sources-section');
  const grid=document.getElementById('sources-grid');
  const sp=m.source_paths||{};
  const entries=Object.entries(sp);
  if(entries.length===0&&!m.center_source){sec.style.display='none';return}
  const byFile={};
  entries.forEach(([pos,path])=>{
    if(!byFile[path])byFile[path]=[];
    byFile[path].push(pos);
  });
  if(m.center_source){
    const cp=m.center_source;
    if(!byFile[cp])byFile[cp]=[];
    if(!byFile[cp].includes('center'))byFile[cp].push('center');
  }
  if(m.source_paths_by_file&&Object.keys(m.source_paths_by_file).length===Object.keys(byFile).length){
    const ref=m.source_paths_by_file;
    let match=true;
    for(const [p,ps] of Object.entries(byFile)){
      const a=(ref[p]||[]).slice().sort().join(',');
      const b=ps.slice().sort().join(',');
      if(a!==b){match=false;break}
    }
    if(!match){
      console.warn('source_paths_by_file in metadata does not match grouped source_paths — old JSON?');
    }
  }
  const tc=(spec&&spec.training_center_size)||1024;
  const fxBanner=(m.effects_applied&&m.effects_applied.length)
    ? `<p class="src-fx-banner">Effects recorded for this sample (${m.effects_applied.join(', ')}) — source thumbnails replay them when possible (same as build pipeline for single-file / non–per-tile mixed cases). Mixed patchwork may still differ per strip.</p>`
    : '';
  const helpHtml=sourcesHelpHtml(m);
  const sid=encodeURIComponent(m.sample_id||'');
  const cards=Object.entries(byFile).map(([path,positions])=>{
    const previewUrl='/api/source_preview?path='+encodeURIComponent(path)+(sid?'&sample_id='+sid:'');
    const fullUrl='/src/'+encodeURI(path);
    const fname=path.split('/').pop();
    return `<div class="src-card">
      <a class="src-full-link" href="${fullUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()">full file</a>
      <img src="${previewUrl}" alt="" data-full-src="${fullUrl}" onclick="openSrcLightbox(this)" loading="lazy" title="Center crop ${tc}×${tc} (same as build_dataset.load_and_square)">
      <span class="src-crop-hint">${tc}×${tc} center crop · training input</span>
      <div class="src-label">
        <span title="${path}">${fname}</span>
        <span class="pos-tags">${positions.map(p=>`<span class="pos-tag">${p}</span>`).join('')}</span>
      </div>
    </div>`;
  });
  grid.innerHTML=helpHtml+fxBanner+cards.join('');
  sec.style.display='block';
}
function openSrcLightbox(img){
  document.getElementById('lb-img').src=img.src;
  const full=img.dataset&&img.dataset.fullSrc;
  document.getElementById('lb-label').textContent=full
    ? 'TRAINING CROP (center square) · full file: open link on card'
    : 'SOURCE · ESC to close';
  lbIndex=-1;
  document.getElementById('lightbox').classList.add('on');
}

/* ── gallery tab ────────────────────────────────────────── */
let galPage=0;
const GAL_PAGE_SIZE=30;

function renderGalleryTab(){
  galPage=0;
  document.getElementById('gallery-content').innerHTML='';
  loadMoreGallery();
}
function loadMoreGallery(){
  const start=galPage*GAL_PAGE_SIZE;
  const chunk=filtered.slice(start,start+GAL_PAGE_SIZE);
  if(chunk.length===0)return;
  const container=document.getElementById('gallery-content');
  chunk.forEach(s=>{
    const row=document.createElement('div');
    row.className='gal-row';
    row.style.cursor='pointer';
    row.onclick=()=>{
      document.querySelector('.tab-btn[data-tab="viewer"]').click();
      const fi=filtered.findIndex(x=>x.id===s.id);
      if(fi>=0)selectSample(fi);
    };
    row.innerHTML=`
      <div class="gal-cell"><div class="gal-head"><span class="gid">${s.id}</span><span class="glabel">Input</span></div><img src="/data/inputs/${s.id}.png?v=${s.thumb_v||0}" loading="lazy"></div>
      <div class="gal-cell"><div class="gal-head"><span class="gid">${s.regime}</span><span class="glabel">Target</span></div><img src="/data/targets/${s.id}.png?v=${s.thumb_v||0}" loading="lazy"></div>
      <div class="gal-cell"><div class="gal-head"><span class="gid"><span class="tag ${s.difficulty}" style="font-size:.55rem">${s.difficulty}</span></span><span class="glabel">Mask</span></div><img src="/data/masks/${s.id}.png?v=${s.thumb_v||0}" loading="lazy" style="image-rendering:pixelated"></div>
    `;
    container.appendChild(row);
  });
  galPage++;
  document.getElementById('gallery-load').style.display=start+GAL_PAGE_SIZE<filtered.length?'block':'none';
}

/* ── lightbox ───────────────────────────────────────────── */
function openLightbox(box){
  const img=box.querySelector('img');
  if(!img?.src)return;
  const raw=img.id.replace('img-','').replace('-solo','');
  lbIndex=LB_KINDS.indexOf(raw);
  if(lbIndex<0)lbIndex=0;
  showLbImage();
  document.getElementById('lightbox').classList.add('on');
}
function showLbImage(){
  const kind=LB_KINDS[lbIndex];
  if(_imgUrls[kind]){
    document.getElementById('lb-img').src=_imgUrls[kind];
  }
  document.getElementById('lb-label').textContent=kind.toUpperCase()+' (←→ switch)';
}
function closeLightbox(){
  document.getElementById('lightbox').classList.remove('on');
  lbIndex=-1;
}

/* ── keyboard ───────────────────────────────────────────── */
window.addEventListener('keydown',e=>{
  const lb=document.getElementById('lightbox').classList.contains('on');
  if(lb){
    if(e.key==='Escape'){closeLightbox();return}
    if(e.key==='ArrowLeft'||e.key==='ArrowUp'){e.preventDefault();lbIndex=(lbIndex-1+LB_KINDS.length)%LB_KINDS.length;showLbImage();return}
    if(e.key==='ArrowRight'||e.key==='ArrowDown'){e.preventDefault();lbIndex=(lbIndex+1)%LB_KINDS.length;showLbImage();return}
    return;
  }
  if(document.activeElement?.tagName==='INPUT')return;
  if(e.key==='ArrowDown'||e.key==='ArrowRight'){e.preventDefault();selectSample(idx+1)}
  else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){e.preventDefault();selectSample(idx-1)}
  else if(e.key==='d'||e.key==='D'){document.querySelector('.tab-btn[data-tab="dashboard"]').click()}
  else if(e.key==='s'||e.key==='S'){document.querySelector('.tab-btn[data-tab="viewer"]').click()}
  else if(e.key==='g'||e.key==='G'){document.querySelector('.tab-btn[data-tab="gallery"]').click()}
  else if(e.key==='/'){e.preventDefault();document.getElementById('search').focus()}
  else if(e.key==='Escape'){document.getElementById('search').blur()}
});

/* ── dashboard ──────────────────────────────────────────── */
async function renderDashboard(){
  const rp=report&&report.total_samples?report:await f('/api/stats');
  const isReport=!!report?.total_samples;
  const total=isReport?rp.total_samples:rp.total;

  const sc=isReport?rp.scenes:{};
  const splits=isReport?rp.splits:rp.splits||{};
  const leak=isReport?rp.scene_leakage_pass:null;
  const dsSize=rp.dataset_size_mb;

  const tiles=[
    {v:total,l:'Total Samples',sub:'Generated pairs'},
    {v:sc.unique_scenes||rp.unique_scenes||'?',l:'Unique Scenes',sub:'Source images used'},
    {v:dsSize!=null?(dsSize>=1024?(dsSize/1024).toFixed(1)+' GB':dsSize+' MB'):'?',l:'Dataset Size',sub:'inputs + targets + masks'},
    {v:splits.train||0,l:'Train',sub:`${(100*(splits.train||0)/Math.max(total,1)).toFixed(0)}%`,cls:'ok'},
    {v:splits.val||0,l:'Val',sub:`${(100*(splits.val||0)/Math.max(total,1)).toFixed(0)}%`},
    {v:splits.benchmark||0,l:'Benchmark',sub:`${(100*(splits.benchmark||0)/Math.max(total,1)).toFixed(0)}%`},
    {v:sc.max_samples_per_scene!=null?sc.max_samples_per_scene:'?',l:'Max / Scene',sub:'Limit: 5'},
  ];
  if(isReport){
    tiles.push({v:rp.elapsed_seconds+'s',l:'Gen Time',sub:'Total elapsed'});
    tiles.push({v:leak?'PASS':'FAIL',l:'Leakage Check',cls:leak?'ok':'bad'});
  }
  document.getElementById('dash-hero').innerHTML=tiles.map(t=>`<div class="tile ${t.cls||''}"><span class="tile-v">${t.v}</span><div class="tile-l">${t.l}</div>${t.sub?`<div class="tile-sub">${t.sub}</div>`:''}</div>`).join('');

  const regimes=isReport?rp.regimes:rp.regimes;
  const neighbors=isReport?rp.neighbor_counts:rp.neighbors;
  const diffs=isReport?rp.difficulty:rp.difficulties;
  const capFams=isReport?rp.caption_families:rp.caption_families;
  const fxFams=isReport?rp.effect_families:rp.effect_families;

  let html='';
  html+=widget('Regime Distribution','Target from spec (red line) vs actual (blue bar).',barChart(regimes,total,spec.regimes));
  html+=widget('Difficulty Distribution','Easy / Medium / Hard balance.',barChart(diffs,total,spec.difficulty));
  html+=widget('Neighbor Count Distribution','From 1 (hardest) to 8 (easiest) neighbors.',barChart(neighbors,total,spec.neighbors));
  html+=widget('Splits','Train / Val / Benchmark proportions.',barChart(splits,total,spec.splits));
  html+=widget('Caption Families','Distribution of prompt prefixes.',kvList(capFams,total));
  html+=widget('Effect Families','Augmentation category frequency.',kvList(fxFams));

  if(isReport&&sc.mean_samples_per_scene!=null){
    html+=widget('Scene Usage','How evenly source images are sampled.',kvList({
      'Unique scenes':sc.unique_scenes,'Mean samples/scene':sc.mean_samples_per_scene,
      'Max samples/scene':sc.max_samples_per_scene,'Over-limit':sc.scenes_over_limit}));
  }
  if(isReport){
    html+=widget('Generation Info','Seed and timing from report.json.',kvList({
      Seed:rp.seed,'Elapsed':rp.elapsed_seconds+'s','Leakage check':leak?'✓ PASS':'✗ FAIL'}));
  }

  if(isReport&&rp.dataset_build){
    html+=renderDatasetConfig(rp.dataset_build);
  }
  if(isReport&&rp.similarity_index&&rp.similarity_index.fingerprint){
    const si=rp.similarity_index;
    html+=widget('Similarity index (global)','File used for mixed regimes; per-sample pairwise scores are read from this index.',kvList({
      fingerprint: si.fingerprint,
      mode: (si.config&&si.config.index_mode)||'—',
      path: si.path||'—',
    }));
  }

  document.getElementById('dash-grid').innerHTML=html;
}

function renderDatasetConfig(db){
  if(!db||typeof db!=='object')return '';
  const esc=(t)=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const tbl=(title,obj)=>{
    if(!obj||typeof obj!=='object')return '';
    const body=Object.entries(obj).map(([k,v])=>{
      let disp=v;
      if(Array.isArray(v)) disp=v.join(' … ');
      else if(v!==null&&typeof v==='object') disp=JSON.stringify(v);
      return `<tr><td class="cfg-k">${esc(k)}</td><td class="cfg-v">${esc(disp)}</td></tr>`;
    }).join('');
    return `<div class="cfg-block"><h4>${esc(title)}</h4><table class="cfg-tbl">${body}</table></div>`;
  };
  let inner='';
  inner+=tbl('Regime sampling weights (REGIMES)',db.REGIMES);
  inner+=tbl('Global neighbor-count distribution',db.NEIGHBOR_DIST);
  if(db._REGIME_NEIGHBOR_DIST){
    for(const [reg,dist] of Object.entries(db._REGIME_NEIGHBOR_DIST)){
      const pct=Object.fromEntries(Object.entries(dist).map(([k,v])=>[`${k} neighbors`,`${(Number(v)*100).toFixed(1)}%`]));
      inner+=tbl(`Per-regime neighbor counts · ${reg}`,pct);
    }
  }
  inner+=tbl('Effect count range by regime (min … max)',db._REGIME_EFFECT_RANGE);
  inner+=tbl('Patchwork: max distinct source files',db._REGIME_MAX_UNIQUE_SOURCES);
  inner+=tbl('Similarity / paths (SIMILARITY_NEIGHBORS)',db.SIMILARITY_NEIGHBORS);
  inner+=tbl('Constants',{MAX_SAMPLES_PER_SCENE:db.MAX_SAMPLES_PER_SCENE,CENTER_PX:db.CENTER_PX});
  return `<div class="wg cfg-mega"><h2>Dataset build configuration</h2><p class="cfg-note">Full orchestrator snapshot from the last <code>build_dataset.py</code> run (<code>report.json</code> → <code>dataset_build</code>). Bar charts above still compare to static spec targets from <code>/api/config</code>.</p><div class="cfg-cols">${inner}</div></div>`;
}

function widget(title,hint,body){
  return `<div class="wg"><h2>${title}</h2><div class="wg-hint">${hint}</div>${body}</div>`;
}
function barChart(data,total,targets){
  const keys=targets?Object.keys(targets):Object.keys(data).sort();
  return keys.map(k=>{
    const v=data[k]||0;
    const pct=(v/Math.max(total,1)*100).toFixed(1);
    const tgt=targets?targets[k]:null;
    return `<div class="bar-row"><div class="bar-labels"><span class="bl-name">${k}</span><span class="bl-val">${pct}%<span style="color:var(--text3);font-weight:400"> (${v})</span>${tgt!=null?' <span style="color:var(--text3);font-size:.65rem">target '+tgt+'%</span>':''}</span></div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div>${tgt!=null?`<div class="bar-target" style="left:${tgt}%"></div>`:''}</div></div>`;
  }).join('');
}
function kvList(data,total){
  return Object.entries(data).map(([k,v])=>{
    const extra=total?` <span style="color:var(--text3)">(${(v/total*100).toFixed(1)}%)</span>`:'';
    return `<div class="kv-row"><span class="k">${k}</span><span class="v">${v}${extra}</span></div>`;
  }).join('');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
