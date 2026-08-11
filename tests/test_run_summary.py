"""Tests for terminal run_summary persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

from storage import agent_log


def test_build_horizon_projections_scales_linearly():
    projections = agent_log.build_horizon_projections(
        usd_per_sim_day=1.5,
        wall_ms_per_sim_day=3_600_000.0,
    )
    assert projections["30d"]["usd"] == 45.0
    assert projections["30d"]["wall_hours"] == 30.0
    assert projections["365d"]["usd"] == 547.5


def test_write_run_summary_roundtrip(tmp_path: Path):
    runs_root = str(tmp_path / "env" / "runs")
    Path(runs_root).mkdir(parents=True)
    run_id = "run-test-summary"
    path = agent_log.write_run_summary(
        runs_root,
        run_id,
        {
            "status": "finished",
            "sim_days": 7,
            "rates": {"usd_per_sim_day": 0.1},
            "projections": agent_log.build_horizon_projections(
                usd_per_sim_day=0.1,
                wall_ms_per_sim_day=1000.0,
            ),
        },
    )
    assert Path(path).exists()
    loaded = agent_log.read_run_summary(runs_root, run_id)
    assert loaded["run_id"] == run_id
    assert loaded["status"] == "finished"
    assert loaded["projections"]["90d"]["usd"] == 9.0
    # * Atomic writer should leave valid JSON only.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert raw["rates"]["usd_per_sim_day"] == 0.1
    history = agent_log.read_run_history(str(tmp_path))
    assert history["run_count"] == 1
    assert history["runs"][0]["run_id"] == run_id


def test_run_history_upsert_replaces_same_run_id(tmp_path: Path):
    runs_root = str(tmp_path / "env" / "runs")
    Path(runs_root).mkdir(parents=True)
    rid = "run-upsert"
    agent_log.write_run_summary(
        runs_root,
        rid,
        {"status": "finished", "sim_days": 1, "cost_total": {"usd": 0.1}},
    )
    agent_log.write_run_summary(
        runs_root,
        rid,
        {"status": "finished", "sim_days": 1, "cost_total": {"usd": 0.2}},
    )
    history = agent_log.read_run_history(str(tmp_path))
    assert history["run_count"] == 1
    assert history["runs"][0]["usd"] == 0.2
