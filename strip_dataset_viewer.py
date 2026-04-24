from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "strip_cache"
STATIC_ROOT = ROOT / "static"
_lock = threading.Lock()

app = FastAPI(title="Strip Dataset Viewer")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


def cache_root() -> Path:
    return Path(__import__("os").environ.get("STRIP_CACHE_ROOT", str(DEFAULT_CACHE_ROOT))).resolve()


@lru_cache(maxsize=2048)
def read_meta(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sample_dirs() -> list[Path]:
    root = cache_root()
    if not root.exists():
        return []
    with _lock:
        return sorted(path for path in root.glob("*/*") if path.is_dir())


def _find_sample(sample_id: str) -> Path:
    for sample_dir in _sample_dirs():
        if sample_dir.name == sample_id:
            return sample_dir
    raise HTTPException(404, "sample not found")


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/samples")
def samples(split: str | None = None, tag: str | None = None, orientation: str | None = None, op: str | None = None):
    rows = []
    for sample_dir in _sample_dirs():
        meta = read_meta(str(sample_dir / "meta.json"))
        if split and meta.get("split") != split:
            continue
        if tag and tag not in meta.get("scene_tags", []):
            continue
        if orientation and meta.get("strip", {}).get("axis") != orientation:
            continue
        if op and op not in meta.get("corruption", {}).get("ops", []):
            continue
        rows.append(meta)
    return rows


@app.get("/api/sample/{sample_id}")
def sample(sample_id: str):
    sample_dir = _find_sample(sample_id)
    return read_meta(str(sample_dir / "meta.json"))


def _png(sample_id: str, name: str) -> FileResponse:
    sample_dir = _find_sample(sample_id)
    path = sample_dir / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found")
    return FileResponse(path)


@app.get("/api/strip/{sample_id}/input.png")
def input_png(sample_id: str):
    return _png(sample_id, "input.png")


@app.get("/api/strip/{sample_id}/target.png")
def target_png(sample_id: str):
    return _png(sample_id, "target.png")


@app.get("/api/strip/{sample_id}/residual.png")
def residual_png(sample_id: str):
    return _png(sample_id, "residual.png")


@app.get("/api/strip/{sample_id}/error.png")
def error_png(sample_id: str):
    return _png(sample_id, "error.png")


@app.get("/api/strip/{sample_id}/seam_profile")
def seam_profile(sample_id: str):
    meta = sample(sample_id)
    return {"id": sample_id, "seam_profile": meta.get("metrics_precomputed", {})}


@app.get("/api/strip/{sample_id}/histogram")
def histogram(sample_id: str):
    meta = sample(sample_id)
    return {"id": sample_id, "ops": meta.get("corruption", {}).get("ops", []), "scene_tags": meta.get("scene_tags", [])}


@app.get("/api/stats")
def stats():
    rows = samples()
    return {
        "total": len(rows),
        "splits": sorted({row.get("split") for row in rows}),
        "tags": sorted({tag for row in rows for tag in row.get("scene_tags", [])}),
        "orientations": sorted({row.get("strip", {}).get("axis") for row in rows}),
    }


@app.get("/api/runs")
def runs():
    root = ROOT / "outputs" / "eval_reports"
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        summary = {}
        summary_path = path / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({"run_id": path.name, "summary": summary})
    return rows


@app.get("/api/run/{run_id}/metrics")
def run_metrics(run_id: str):
    run_dir = ROOT / "outputs" / "eval_reports" / run_id
    if not run_dir.exists():
        raise HTTPException(404, "run not found")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    return summary


@app.get("/api/inspect_strip/{sample_id}")
def inspect_strip(sample_id: str):
    meta = sample(sample_id)
    sample_dir = _find_sample(sample_id)
    return {
        "meta": meta,
        "files": sorted(path.name for path in sample_dir.iterdir()),
    }
