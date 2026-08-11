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


def test_compact_run_history_preserves_public_review_diagnostics():
    entry = agent_log.compact_run_history_entry({
        "run_id": "review-run",
        "result": {
            "reputation_evidence_count": 15,
            "qualified_transaction_count": 120,
            "service_quality_score": 4.1,
            "public_review_rating": 3.8,
            "public_review_count": 15,
            "public_review_eligible_count": 120,
            "public_review_response_rate": 0.125,
            "public_review_full_response_rating": 4.2,
            "public_review_selection_gap": -0.4,
            "public_review_quality_gap": -0.3,
            "public_review_confidence": 15 / 35,
            "public_review_quality_multiplier": 0.9,
            "public_review_reputation_multiplier": 0.885714,
            "public_review_demand_multiplier": 0.797143,
        },
    })

    assert entry["reputation_evidence_count"] == 15
    assert entry["qualified_transaction_count"] == 120
    assert entry["service_quality_score"] == 4.1
    assert entry["public_review_rating"] == 3.8
    assert entry["public_review_count"] == 15
    assert entry["public_review_eligible_count"] == 120
    assert entry["public_review_response_rate"] == 0.125
    assert entry["public_review_full_response_rating"] == 4.2
    assert entry["public_review_selection_gap"] == -0.4
    assert entry["public_review_quality_gap"] == -0.3
    assert entry["public_review_confidence"] == 15 / 35
    assert entry["public_review_quality_multiplier"] == 0.9
    assert entry["public_review_reputation_multiplier"] == 0.885714
    assert entry["public_review_demand_multiplier"] == 0.797143
