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
    runs_root = str(tmp_path)
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
