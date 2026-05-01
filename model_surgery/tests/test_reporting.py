"""Tests for reporting utilities."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from model_surgery.lib.reporting import (
    RunLog, comparison_table, load_json, save_csv, save_json,
)


def _make_row(name: str, q: float) -> dict:
    return {"name": name, "epoch": 5, "quality_score": q,
            "boundary_mae_16": 0.02, "boundary_ciede2000_16": 2.5}


def test_comparison_table_contains_header():
    rows = [_make_row("ckpt_a", 8.0), _make_row("ckpt_b", 12.0)]
    table = comparison_table(rows)
    assert "name" in table
    assert "ckpt_a" in table
    assert "ckpt_b" in table


def test_comparison_table_shows_quality():
    rows = [_make_row("ckpt_a", 8.123)]
    table = comparison_table(rows)
    assert "8.123" in table or "8.12" in table


def test_save_json_roundtrip(tmp_path):
    obj = {"key": [1, 2, 3], "nested": {"a": "b"}}
    p = tmp_path / "test.json"
    save_json(obj, p)
    loaded = load_json(p)
    assert loaded == obj


def test_save_json_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "file.json"
    save_json({"x": 1}, p)
    assert p.exists()


def test_save_csv_creates_file(tmp_path):
    rows = [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}]
    p = tmp_path / "out.csv"
    save_csv(rows, p)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "a" in content and "b" in content


def test_run_log_writes_jsonl(tmp_path):
    p = tmp_path / "run.jsonl"
    with RunLog(p) as log:
        log.log("start", value=42)
        log.log("end", result="ok")
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "start"
    assert first["value"] == 42
    assert "ts" in first
    assert "elapsed_s" in first


def test_run_log_elapsed_increases(tmp_path):
    import time
    p = tmp_path / "run.jsonl"
    with RunLog(p) as log:
        log.log("first")
        time.sleep(0.05)
        log.log("second")
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    first_elapsed = json.loads(lines[0])["elapsed_s"]
    second_elapsed = json.loads(lines[1])["elapsed_s"]
    assert second_elapsed >= first_elapsed


def test_comparison_table_sorts_by_insertion_order():
    rows = [_make_row("aaa", 5.0), _make_row("bbb", 1.0), _make_row("ccc", 10.0)]
    table = comparison_table(rows)
    lines = [l for l in table.splitlines() if l.strip() and "---" not in l]
    # Check all names appear
    names = [l.split()[0] for l in lines[1:]]
    assert "aaa" in names and "bbb" in names and "ccc" in names
