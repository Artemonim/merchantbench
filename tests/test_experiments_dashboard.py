import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pytest
import yaml

from core.entities import EventLog, Order, StoreListing
from storage import agent_log
from storage import db as dbm
from web import leaderboard as leaderboard_mod
from web.app import create_app
from web.leaderboard import (
    build_charts,
    build_leaderboard,
    build_run_results,
    merge_chart_payloads,
)
from web.runner import load_default_scenario


@pytest.fixture
def client(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    app.registry._spawn_react_160k_compact_30k = lambda *args, **kwargs: None
    app.registry._spawn_auto_seed = lambda *args, **kwargs: None
    with app.test_client() as c:
        yield c, app


def _tiny_scenario():
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 4
    scenario["run"]["max_hook_seconds"] = 0.01
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 20
    return scenario


def _result_run(
    app,
    *,
    name: str,
    net_assets: float,
    bootstrap_agent: str = "react_160k_compact_30k",
    react_model: Optional[str] = "model-a",
    status: str = "finished",
    seed: int = 42,
    cost_usd: float = 0.0,
    tool_calls: Optional[list[str]] = None,
    write_profit_metric: bool = True,
    virtual_start_date: Optional[str] = None,
) -> str:
    scenario = _tiny_scenario()
    scenario["run"]["master_seed"] = seed
    if virtual_start_date is not None:
        scenario["run"]["virtual_time"] = {
            "enabled": True,
            "start_date": virtual_start_date,
        }
    bootstrap_config = {"react_model": react_model} if react_model else {}
    run_id = app.registry.create_run(
        scenario,
        name=name,
        bootstrap_agent=bootstrap_agent,
        bootstrap_config=bootstrap_config,
        auto_start=False,
    )
    for t, ratio in ((1, 0.5), (4, 1.0)):
        metrics = {
            "net_assets": net_assets * ratio,
            "cum_gmv": 2000.0 * ratio,
            "cum_fine": 40.0 * ratio,
            "shop_rating_score": 0.9,
        }
        if write_profit_metric:
            metrics.update({
                "cum_gross_profit": 300.0 * ratio,
                "cum_net_profit": (net_assets - 4000.0) * ratio,
            })
        dbm.write_metrics(
            app.registry.conn_for(run_id),
            run_id,
            "agent_0",
            t,
            metrics,
        )
        dbm.write_metrics(
            app.registry.conn_for(run_id),
            run_id,
            "_global",
            t,
            {"orders_generated": 2 if t == 1 else 3},
        )
    if cost_usd:
        agent_log.update_cost(
            app.registry.runs_root,
            run_id,
            4,
            [{"token_usage": {"input": 1000, "output": 0, "total": 1000}}],
            pricing={"input_per_million": cost_usd * 1000.0, "output_per_million": 0.0},
        )
    if tool_calls:
        names = list(tool_calls)
        if "end_of_step" not in names:
            names.append("end_of_step")
        agent_log.write_step_index(
            app.registry.runs_root,
            run_id,
            4,
            [{
                "role": "assistant",
                "content": "tools",
                "tool_calls": [
                    {
                        "id": f"call_{i}_{name}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for i, name in enumerate(names)
                ],
            }],
            [],
        )
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 4)
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, status)
    dbm.update_run_finished_at(app.registry.conn_for(run_id), run_id, "2026-06-16T00:00:00")
    return run_id


def test_leaderboard_rows_include_run_metadata_for_links_and_time(client):
    _c, app = client
    run_id = _result_run(
        app,
        name="metadata run",
        net_assets=4200.0,
        seed=123,
        cost_usd=0.123,
    )
    conn = app.registry.conn_for(run_id)
    dbm.update_run_finished_at(conn, run_id, "2026-06-16T00:02:03")
    conn.execute(
        "UPDATE runs SET started_at=? WHERE run_id=?",
        ("2026-06-16T00:00:00", run_id),
    )

    rows = build_leaderboard(build_run_results(app.registry))

    row = next(row for row in rows if row["run_id"] == run_id)
    assert row["open_url"] == f"/dashboard?run_id={run_id}"
    assert row["master_seed"] == 123
    assert row["horizon"] == 4
    assert row["step_hours"] == 1
    assert row["avg_t"] == 4
    assert row["avg_tokens"] == 1000
    assert row["started_at"] == "2026-06-16T00:00:00"
    assert row["elapsed_ms"] == 123000


def test_leaderboard_rows_include_requested_business_reliability_and_activity_metrics(client):
    _c, app = client
    run_id = _result_run(
        app,
        name="complete leaderboard metrics",
        net_assets=5000.0,
        tool_calls=["query_balance", "query_balance", "query_my_orders"],
    )
    conn = app.registry.conn_for(run_id)
    dbm.write_metrics(
        conn,
        run_id,
        "agent_0",
        1,
        {"n_active_listings": 10},
    )
    dbm.write_metrics(
        conn,
        run_id,
        "agent_0",
        3,
        {"n_active_listings": 20},
    )
    # Draining samples after the configured horizon must not bias activity.
    dbm.write_metrics(
        conn,
        run_id,
        "agent_0",
        4,
        {"n_active_listings": 100},
    )
    conn.executemany(
        "INSERT INTO orders("
        " run_id, order_id, agent_id, order_t, preset_anomaly, current_status"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        [
            (run_id, "normal-1", "agent_0", 1, "normal", "settled_normal"),
            (run_id, "refund-1", "agent_0", 2, "refund", "settled_refund"),
            (run_id, "normal-2", "agent_0", 3, "normal", "settled_normal"),
            (
                run_id,
                "bad-review-1",
                "agent_0",
                4,
                "bad_review",
                "settled_bad_review",
            ),
        ],
    )

    run_results = build_run_results(app.registry)
    result = next(row["result"] for row in run_results if row["run_id"] == run_id)
    # The synchronous summary path intentionally avoids scanning full traces.
    assert result["effective_window_rate"] is None
    assert result["total_tool_calls"] is None
    charts = build_charts(app.registry, run_results)
    leaderboard_row = next(
        row for row in build_leaderboard(run_results)
        if row["run_id"] == run_id
    )

    assert result["net_profit_margin"] == 0.5
    assert result["order_anomaly_rate"] == 0.5
    assert result["average_active_listings"] == 15.0
    assert result["effective_window_rate"] == 1.0
    assert result["total_tool_calls"] == 4
    assert leaderboard_row["avg_net_profit_margin"] == 0.5
    assert leaderboard_row["avg_order_anomaly_rate"] == 0.5
    assert leaderboard_row["avg_active_listings"] == 15.0
    assert leaderboard_row["avg_effective_window_rate"] == 1.0
    assert leaderboard_row["avg_total_tool_calls"] == 4
    assert charts["cum_order_anomalies"][0]["data"] == [[1, 0.0], [4, 2.0]]
    assert charts["active_listings"][0]["data"] == [[1, 15.0, 2]]
    assert charts["tool_calls"]["runs"][0]["activity_by_day"] == [{
        "t": 0,
        "available_windows": 1,
        "effective_windows": 1,
        "total_tool_calls": 4,
    }]


def test_order_anomaly_rate_counts_realized_outcomes_not_preset_flags(client):
    _c, app = client
    run_id = _result_run(
        app,
        name="realized order anomaly semantics",
        net_assets=5000.0,
    )
    conn = app.registry.conn_for(run_id)
    conn.executemany(
        "INSERT INTO orders("
        " run_id, order_id, agent_id, order_t, preset_anomaly,"
        " current_status, late_t"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (run_id, "normal", "agent_0", 1, "normal", "settled_normal", None),
            # A hidden scenario assignment is not an observed outcome.
            (run_id, "pending-refund", "agent_0", 2, "refund", "ordered", None),
            # Lateness remains anomalous even when the eventual settlement is normal.
            (run_id, "late-normal", "agent_0", 3, "normal", "settled_normal", 3),
        ],
    )

    run_results = build_run_results(app.registry)
    result = next(row["result"] for row in run_results if row["run_id"] == run_id)

    assert result["order_anomaly_rate"] == 0.333333


def test_dashboard_leaderboard_template_renders_metadata_columns():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert '<th>open</th>' in html
    assert '<th data-sort="master_seed" class="sortable">seed</th>' in html
    assert '<th data-sort="horizon" class="sortable">horizon</th>' in html
    assert '<th data-sort="started_at" class="sortable">started</th>' in html
    assert '<th data-sort="elapsed_ms" class="sortable">total time</th>' in html
    assert "leaderboardT(row)" in html
    assert "leaderboardHorizon(row)" in html
    assert "steps * stepHours / 24" in html
    assert "fmtStarted(row.started_at)" in html
    assert "fmtElapsed(row.elapsed_ms)" in html
    assert html.index("<th>TOKENS</th>") < html.index("<th>USD</th>")
    assert (
        html.index('<th data-sort="avg_tokens" class="sortable">tokens</th>')
        < html.index('<th data-sort="avg_usd" class="sortable">usd</th>')
    )
    for field, label in (
        ("avg_net_profit_margin", "net profit margin"),
        ("avg_order_anomaly_rate", "order anomaly rate"),
        ("avg_active_listings", "average active listings"),
        ("avg_effective_window_rate", "effective window rate"),
        ("avg_total_tool_calls", "total tool calls"),
    ):
        assert f'<th data-sort="{field}" class="sortable">{label}</th>' in html
    for field, label in (
        ("avg_net_profit_margin", "Net Profit Margin"),
        ("avg_order_anomaly_rate", "Order Anomaly Rate"),
        ("avg_active_listings", "Average Active Listings"),
        ("avg_effective_window_rate", "Effective Window Rate"),
        ("avg_total_tool_calls", "Total Tool Calls"),
    ):
        assert f'<option value="{field}">{label} Ranking</option>' in html
    assert "function leaderboardRatioRankingValue" in html
    assert "function leaderboardActivityRankingValue" in html
    assert html.count('direction: "asc"') >= 2
    assert 'metric.direction === "asc"' in html
    theme = Path("env/web/templates/_pixel_theme.html").read_text(encoding="utf-8")
    assert 'content: "DAY (STEPS)"' in theme


def test_dashboard_initial_summary_does_not_parse_full_agent_traces(
    client,
    monkeypatch,
):
    c, app = client
    _result_run(
        app,
        name="lightweight summary",
        net_assets=5000.0,
        tool_calls=["query_balance"],
    )

    def fail_trace_scan(*_args, **_kwargs):
        raise AssertionError("initial dashboard summary parsed full traces")

    monkeypatch.setattr(
        leaderboard_mod,
        "_tool_call_step_counts",
        fail_trace_scan,
    )

    response = c.get("/dashboard")

    assert response.status_code == 200


def test_experiment_groups_api_persists_manual_run_bindings(client):
    c, app = client
    run_id = _result_run(
        app,
        name="grouped run",
        net_assets=4321.0,
        bootstrap_agent="hermes",
        react_model="gpt-5.6-sol",
    )
    payload = c.get("/dashboard/experiment-groups.json").get_json()
    options_payload = c.get(
        "/dashboard/experiment-run-options.json"
    ).get_json()

    assert [row["model"] for row in payload["model_presets"]] == [
        "gpt-5.6-sol",
        "claude-opus-4-8",
        "bailian/glm-5.2",
        "qwen3.7-max",
        "qwen3.7-plus",
        "bailian/deepseek-v4-pro",
        "bailian/deepseek-v4-flash",
        "bailian/kimi-k2.6",
    ]
    model_labels = {
        row["model"]: row["label"] for row in payload["model_presets"]
    }
    assert model_labels["qwen3.7-max"] == "Qwen3.7 Max"
    assert model_labels["qwen3.7-plus"] == "Qwen3.7 Plus"
    run_option = next(
        row for row in payload["run_options"] if row["run_id"] == run_id
    )
    assert next(
        row for row in options_payload["run_options"]
        if row["run_id"] == run_id
    ) == run_option
    assert run_option["framework"] == "Hermes"
    assert run_option["model"] == "gpt-5.6-sol"
    assert run_option["status"] == "finished"
    assert run_option["horizon_days"] == pytest.approx(4 / 24)
    assert "final_net_assets" not in run_option
    assert "rank" not in run_option

    document = {
        "groups": [{
            "id": "group-main",
            "name": "8 Models × 3 Repeats",
            "template": {
                "frameworks": ["hermes"],
                "models": ["qwen3.7-max"],
                "include_human": True,
                "include_rule_based": True,
            },
            "batches": [{
                "id": "batch-1",
                "name": "Batch 1",
                "bindings": {
                    "model::hermes::qwen3.7-max": run_id,
                },
            }],
        }],
    }
    saved = c.put(
        "/dashboard/experiment-groups.json",
        json=document,
    )
    assert saved.status_code == 200
    assert saved.get_json()["groups"][0]["batches"][0]["bindings"] == {
        "model::hermes::qwen3.7-max": run_id,
    }
    stored_path = Path(app.registry.runs_root) / "experiment_groups.json"
    assert stored_path.exists()
    assert c.get("/dashboard/experiment-groups.json").get_json()["groups"][0][
        "name"
    ] == "8 Models × 3 Repeats"


def test_experiment_group_endpoints_do_not_access_chart_cache(
    client,
    monkeypatch,
):
    c, app = client
    _result_run(app, name="summary only", net_assets=4321.0)

    from web import routes_dashboard

    chart_cache_accesses = []
    real_gzip_open = routes_dashboard.gzip.open

    def track_gzip_open(*args, **kwargs):
        chart_cache_accesses.append(args[0] if args else None)
        return real_gzip_open(*args, **kwargs)

    def fail_build_charts(*_args, **_kwargs):
        raise AssertionError("experiment group endpoint built full charts")

    monkeypatch.setattr(routes_dashboard.gzip, "open", track_gzip_open)
    monkeypatch.setattr(routes_dashboard, "build_charts", fail_build_charts)

    assert c.get("/dashboard/experiment-groups.json").status_code == 200
    assert c.get("/dashboard/experiment-run-options.json").status_code == 200
    assert c.put(
        "/dashboard/experiment-groups.json",
        json={"groups": []},
    ).status_code == 200
    assert chart_cache_accesses == []


def test_experiment_groups_api_rejects_duplicate_batch_ids(client):
    c, _app = client
    response = c.put(
        "/dashboard/experiment-groups.json",
        json={
            "groups": [{
                "id": "group-main",
                "name": "duplicates",
                "template": {},
                "batches": [
                    {"id": "same", "name": "Batch 1", "bindings": {}},
                    {"id": "same", "name": "Batch 2", "bindings": {}},
                ],
            }],
        },
    )

    assert response.status_code == 400
    assert "duplicate batch id" in response.get_json()["error"]


def test_experiment_groups_api_does_not_overwrite_corrupt_config(client):
    c, app = client
    stored_path = Path(app.registry.runs_root) / "experiment_groups.json"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt = '{"version": 1, "groups": ['
    stored_path.write_text(corrupt, encoding="utf-8")

    loaded = c.get("/dashboard/experiment-groups.json")
    saved = c.put("/dashboard/experiment-groups.json", json={"groups": []})

    assert loaded.status_code == 500
    assert "cannot read experiment_groups.json" in loaded.get_json()["error"]
    assert saved.status_code == 500
    assert stored_path.read_text(encoding="utf-8") == corrupt


def test_experiment_groups_api_requires_explicit_groups_before_overwrite(client):
    c, app = client
    initial = {
        "groups": [{
            "id": "group-main",
            "name": "Keep me",
            "template": {
                "frameworks": ["hermes"],
                "models": ["gpt-5.6-sol"],
            },
            "batches": [],
        }],
    }
    assert c.put("/dashboard/experiment-groups.json", json=initial).status_code == 200
    stored_path = Path(app.registry.runs_root) / "experiment_groups.json"
    before = stored_path.read_text(encoding="utf-8")

    missing = c.put("/dashboard/experiment-groups.json", json={})
    null_groups = c.put(
        "/dashboard/experiment-groups.json",
        json={"groups": None},
    )

    assert missing.status_code == 400
    assert missing.get_json()["error"] == "groups is required"
    assert null_groups.status_code == 400
    assert null_groups.get_json()["error"] == "groups must be an array"
    assert stored_path.read_text(encoding="utf-8") == before


def test_experiment_groups_api_rejects_wrong_empty_container_types(client):
    c, app = client
    initial = {
        "groups": [{
            "id": "group-main",
            "name": "Keep me",
            "template": {
                "frameworks": ["hermes"],
                "models": ["gpt-5.6-sol"],
            },
            "batches": [{
                "id": "batch-1",
                "name": "Batch 1",
                "bindings": {
                    "model::hermes::gpt-5.6-sol": "run-keep",
                },
            }],
        }],
    }
    assert c.put("/dashboard/experiment-groups.json", json=initial).status_code == 200
    stored_path = Path(app.registry.runs_root) / "experiment_groups.json"
    before = stored_path.read_text(encoding="utf-8")

    invalid_documents = [
        {
            "groups": [{
                "id": "group-main",
                "name": "Keep me",
                "template": [],
                "batches": [],
            }],
        },
        {
            "groups": [{
                "id": "group-main",
                "name": "Keep me",
                "template": {},
                "batches": {},
            }],
        },
        {
            "groups": [{
                "id": "group-main",
                "name": "Keep me",
                "template": {},
                "batches": [{
                    "id": "batch-1",
                    "name": "Batch 1",
                    "bindings": [],
                }],
            }],
        },
    ]
    for document in invalid_documents:
        response = c.put("/dashboard/experiment-groups.json", json=document)
        assert response.status_code == 400
        assert stored_path.read_text(encoding="utf-8") == before


def test_dashboard_has_experiment_group_batch_editor_and_day_labels():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'id="experiment-group-select"' in html
    assert 'id="experiment-batch-toggles"' in html
    assert 'id="experiment-batch-editors"' in html
    assert 'id="experiment-group-export-md"' in html
    assert 'class="experiment-run-picker' in html
    assert 'class="experiment-run-search"' in html
    assert "Search run id, framework, model, status, net assets, or rank…" in html
    assert 'id="batch-average-note"' in html
    assert "experimentRunOptionLabel" in html
    assert 'formatExperimentDays(row?.horizon_days)' in html
    assert "formatExperimentNetAssets(row?.final_net_assets)" in html
    assert "formatExperimentRank(row?.rank)" in html
    assert "Selected ×${entries.length}" in html
    assert 'class="experiment-option-count"' in html
    assert "experimentModelBatchProgress" in html
    assert "experimentRankedBatchSlots" in html
    assert "rank across this batch" in html
    assert "buildExperimentAverageRows" in html
    assert "averageExperimentPointArrays" in html
    assert "sampleStdExperimentValues" in html
    assert "batchSummaryCell" in html
    assert "experimentGroupMarkdown" in html
    assert "mean ± sample SD (n)" in html
    assert "pixelBatchRangeSeries" not in html
    assert "Missing bindings are ignored, never treated as zero." in html
    assert "fullPayloadLoaded: false" in html
    assert "|| !leaderboardViz.fullPayloadLoaded" in html
    assert "leaderboardViz.fullPayloadLoaded = true;" in html
    assert "Complete leaderboard metrics are still loading" in html
    assert "function refreshExperimentSelectionView" in html
    assert html.count("refreshExperimentSelectionView();") >= 6
    assert "newButton.disabled = !experimentGroups.loaded" in html
    assert "if (!experimentGroups.loaded || experimentGroups.saving) return;" in html
    assert "loadExperimentGroupView();\n    renderExperimentGroupControls();" in html
    assert '"/dashboard/experiment-run-options.json"' in html
    assert "refreshExperimentRunOptions();" in html
    assert "const editorDisabled = !group || experimentGroups.saving;" in html
    assert (
        "leaderboardViz.basePayload = payload;\n"
        "      syncExperimentRunOptionMetrics();\n"
        "      if (experimentGroups.loaded) renderExperimentGroupControls();\n"
        "      applyExperimentBatchVisibility();"
    ) in html


def test_experiment_run_day_label_keeps_unknown_duration_unknown():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function formatExperimentDays"):
        html.index("function updateExperimentRunOptions")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const esc = value => String(value ?? "");
const experimentGroups = {
  runOptions: [{
    run_id: "run-searchable",
    framework: "Hermes",
    model: "gpt-5.6-sol",
    status: "finished",
    started_at: "2026-07-25T12:00:00",
    horizon_days: 365,
  }],
};
const leaderboardViz = {
  basePayload: {
    leaderboard: [{
      run_id: "run-searchable",
      avg_final_net_assets: 4321,
      rank: 2,
    }],
  },
};
eval(process.argv[1]);
syncExperimentRunOptionMetrics();
check(formatExperimentDays(null) === "— days",
  "null duration was rendered as zero days");
check(formatExperimentDays(undefined) === "— days",
  "missing duration was rendered as zero days");
check(formatExperimentDays(365).includes("365"),
  "known duration was not rendered");
check(formatExperimentNetAssets(null) === "Net —",
  "missing net assets was rendered as zero");
check(formatExperimentRank(null) === "#—",
  "missing rank was rendered as zero");
const searchLabel = experimentRunOptionLabel(experimentGroups.runOptions[0]);
check(searchLabel.includes("Net 4,321.00"),
  "search label did not expose final net assets");
check(searchLabel.includes("#2"),
  "search label did not expose leaderboard rank");
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_batch_progress_framework_ordering_and_batch_ranking():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function experimentModelBatchProgress"):
        html.index("function experimentRunUsageMap")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const slots = [
  {id: "h-a", kind: "model", model: "a", framework_key: "hermes"},
  {id: "h-b", kind: "model", model: "b", framework_key: "hermes"},
  {id: "h-c", kind: "model", model: "c", framework_key: "hermes"},
  {id: "r-a", kind: "model", model: "a", framework_key: "react"},
  {id: "r-b", kind: "model", model: "b", framework_key: "react"},
];
const runs = new Map([
  ["run-h-a", {final_net_assets: 10}],
  ["run-h-b", {final_net_assets: 30}],
  ["run-h-c", {final_net_assets: 50}],
  ["run-r-a", {final_net_assets: 100}],
  ["run-r-b", {final_net_assets: 90}],
]);
const experimentSlots = () => slots;
const experimentRunById = runId => runs.get(runId) || null;
const selectedExperimentGroup = () => null;
eval(process.argv[1]);

const batch1 = {
  id: "batch-1",
  bindings: {
    "h-a": "run-h-a",
    "h-b": "run-h-b",
    "r-a": "run-r-a",
    "r-b": "run-r-b",
  },
};
const group = {
  batches: [batch1, {id: "batch-2", bindings: {}}],
};
const progress = experimentModelBatchProgress("a", group);
check(progress.linked === 1 && progress.total === 2,
  "model progress did not count linked batches");

let ranked = experimentRankedBatchSlots(group, batch1);
check(ranked.map(row => row.id).join(",") === "h-b,h-a,h-c,r-a,r-b",
  "slots were not ranked inside stable framework groups");
check(ranked.map(row => row.batchRank ?? "-").join(",") === "3,4,-,1,2",
  "cross-framework batch ranks were incorrect");

batch1.bindings["h-c"] = "run-h-c";
ranked = experimentRankedBatchSlots(group, batch1);
check(ranked.slice(0, 3).map(row => row.id).join(",") === "h-c,h-b,h-a",
  "changed binding did not reorder its framework");
check(ranked.map(row => row.batchRank ?? "-").join(",") === "3,4,5,1,2",
  "batch ranks did not update after the binding changed");
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_group_render_preserves_loading_view_and_locks_saving_editor():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function renderExperimentGroupControls"):
        html.index("function experimentSelectedRunIds")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const element = () => ({innerHTML: "", value: "", disabled: false});
const elements = Object.fromEntries([
  "experiment-group-select",
  "experiment-group-new",
  "experiment-batch-toggles",
  "experiment-group-name",
  "experiment-group-delete",
  "experiment-group-save",
  "experiment-batch-add",
  "experiment-framework-options",
  "experiment-model-options",
  "experiment-batch-editors",
].map(id => [id, element()]));
const $ = id => elements[id] || null;
const esc = value => String(value ?? "");
const updateExperimentAverageNote = () => {};
const experimentRunPickerValue = runId => runId;
const frameworkIdentity = (_framework, label) => label;
const modelIdentity = (_model, label) => label;
const experimentSlots = () => [{id: "slot", label: "slot"}];
const experimentRankedBatchSlots = experimentSlots;
const selectedExperimentGroup = () => experimentGroups.groups.find(
  group => group.id === experimentGroups.selectedGroupId
) || null;
const experimentGroups = {
  loaded: false,
  saving: false,
  groups: [],
  selectedGroupId: "group-main",
  activeBatchIds: new Set(["batch-1"]),
  frameworkPresets: [],
  modelPresets: [],
  runOptions: [],
};
eval(process.argv[1]);

renderExperimentGroupControls();
check(experimentGroups.selectedGroupId === "group-main",
  "loading render cleared the persisted group selection");
check(experimentGroups.activeBatchIds.has("batch-1"),
  "loading render cleared the persisted batch selection");
check(elements["experiment-group-new"].disabled === true,
  "new group remained enabled while groups were loading");

experimentGroups.groups = [{
  id: "group-main",
  name: "Main",
  template: {},
  batches: [{id: "batch-1", name: "Batch 1", bindings: {slot: "run-1"}}],
}];
experimentGroups.loaded = true;
experimentGroups.saving = true;
renderExperimentGroupControls();
check(elements["experiment-group-name"].disabled === true,
  "group name remained editable during save");
check(elements["experiment-group-delete"].disabled === true,
  "delete remained enabled during save");
check(elements["experiment-batch-add"].disabled === true,
  "add batch remained enabled during save");
check(elements["experiment-batch-editors"].innerHTML.includes("disabled"),
  "batch binding controls remained editable during save");
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _mark_live_worker(app, run_id: str, state: str = "running") -> None:
    class LiveThread:
        def is_alive(self) -> bool:
            return True

    class DummyWorker:
        pass

    worker = DummyWorker()
    worker.state = state
    worker._thread = LiveThread()
    with app.registry.lock:
        app.registry.workers[run_id] = worker


def test_status_run_without_worker_handles_db_read_error(client, monkeypatch):
    from web import routes_dashboard

    c, app = client
    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    def fail_get_run(*args, **kwargs):
        raise sqlite3.DatabaseError("corrupt page")

    monkeypatch.setattr(routes_dashboard.dbm, "get_run", fail_get_run)

    response = c.get(f"/runs/{run_id}/status")

    assert response.status_code == 404
    assert response.get_json()["error"] == "not found"


def _weekly_result_run(
    app,
    *,
    name: str,
    react_model: str = "model-a",
    write_profit_metric: bool = True,
) -> str:
    scenario = _tiny_scenario()
    scenario["run"]["horizon_steps"] = 14 * 24
    run_id = app.registry.create_run(
        scenario,
        name=name,
        bootstrap_agent="react_160k_compact_30k",
        bootstrap_config={"react_model": react_model},
        auto_start=False,
    )
    for t, net_assets, gmv, profit in (
        (24, 4100.0, 100.0, 10.0),
        (192, 4050.0, 150.0, -5.0),
    ):
        metrics = {
            "net_assets": net_assets,
            "cum_gmv": gmv,
            "cum_fine": 0.0,
            "shop_rating_score": 0.9,
        }
        if write_profit_metric:
            metrics["cum_net_profit"] = profit
        dbm.write_metrics(
            app.registry.conn_for(run_id),
            run_id,
            "agent_0",
            t,
            metrics,
        )
    for t, calls in (
        (0, ["end_of_step"]),
        (12, ["query_balance", "end_of_step"]),
        (192, [
            "list_product",
            "adjust_price",
            "delist_product",
            "end_of_step",
        ]),
    ):
        agent_log.write_step_index(
            app.registry.runs_root,
            run_id,
            t,
            [{
                "role": "assistant",
                "content": "tools",
                "tool_calls": [
                    {
                        "id": f"call_{i}_{name}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for i, name in enumerate(calls)
                ],
            }],
            [],
        )
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, scenario["run"]["horizon_steps"])
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "finished")
    dbm.update_run_finished_at(app.registry.conn_for(run_id), run_id, "2026-06-16T00:00:00")
    return run_id


def test_dashboard_renders_leaderboard_without_run_results_or_queue_ui(client):
    c, _ = client

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert '<div class="title">Leaderboard</div>' in html
    assert "Run Results" not in html
    assert 'id="tbl-run-results"' not in html
    assert "Queue" not in html
    assert "Batch Experiments" not in html
    assert "New Batch" not in html
    assert "Experiments</h2>" not in html
    assert 'id="dashboard-leaderboard"' in html
    assert "Final Net Assets Ranking" in html
    assert 'id="ch-exp-net-assets-ranking"' in html
    assert 'id="ranking-metric-mode"' in html
    assert '<option value="avg_tokens">Tokens Ranking</option>' in html
    assert "function renderLeaderboardRankingChart" in html
    assert "function leaderboardRankingAxisRows" in html
    assert "function leaderboardRankingCategoryAxis" in html
    assert "function leaderboardRankingBarLabelRich" in html
    assert "renderLeaderboardRankingChart();" in html
    assert html.index('<div class="title" id="all-runs-title">All Runs</div>') < \
        html.index('<div class="title">Leaderboard</div>')


def test_leaderboard_payload_does_not_cache_historical_run_connections(client):
    c, app = client
    run_a = _result_run(app, name="A", net_assets=5000.0)
    run_b = _result_run(app, name="B", net_assets=4500.0)
    with app.registry.lock:
        app.registry.envs.clear()
        app.registry.workers.clear()
    app.registry.close_run_conn(run_a)
    app.registry.close_run_conn(run_b)

    assert app.registry._conns == {}

    resp = c.get("/dashboard/leaderboard.json")

    assert resp.status_code == 200
    assert {row["run_id"] for row in resp.get_json()["run_results"]} == {run_a, run_b}
    assert app.registry._conns == {}


def test_terminal_chart_cache_survives_app_restart(client, monkeypatch):
    c, app = client
    run_id = _result_run(app, name="cached", net_assets=5000.0)
    first = c.get("/dashboard/leaderboard.json")
    assert first.status_code == 200
    assert run_id in {
        row["run_id"] for row in first.get_json()["charts"]["runs"]
    }

    from web import routes_dashboard

    cache_path = os.path.join(
        app.registry.runs_root,
        routes_dashboard._TERMINAL_CHART_CACHE_FILENAME,
    )
    assert os.path.exists(cache_path)

    def fail_build_charts(*_args, **_kwargs):
        raise AssertionError("terminal charts should load from persistent cache")

    monkeypatch.setattr(routes_dashboard, "build_charts", fail_build_charts)
    second_app = create_app(
        db_path=os.path.join(
            os.path.dirname(app.registry.runs_root),
            "second-app.db",
        ),
        runs_root=app.registry.runs_root,
    )
    try:
        with second_app.test_client() as second_client:
            second = second_client.get("/dashboard/leaderboard.json")
        assert second.status_code == 200
        assert run_id in {
            row["run_id"] for row in second.get_json()["charts"]["runs"]
        }
    finally:
        second_app.registry.shutdown()


def test_dashboard_summary_does_not_wait_for_cold_chart_build(
    client,
    monkeypatch,
):
    c, app = client
    _result_run(app, name="cold chart", net_assets=5000.0)
    assert c.get("/dashboard").status_code == 200

    from web import routes_dashboard

    real_build_charts = routes_dashboard.build_charts
    build_started = threading.Event()
    release_build = threading.Event()
    summary_done = threading.Event()
    responses = {}
    errors = []

    def blocking_build_charts(*args, **kwargs):
        build_started.set()
        if not release_build.wait(timeout=5):
            raise AssertionError("timed out waiting to release chart build")
        return real_build_charts(*args, **kwargs)

    def request_full_payload():
        try:
            with app.test_client() as thread_client:
                responses["full"] = thread_client.get(
                    "/dashboard/leaderboard.json"
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def request_summary():
        try:
            with app.test_client() as thread_client:
                responses["summary"] = thread_client.get("/dashboard")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            summary_done.set()

    monkeypatch.setattr(
        routes_dashboard,
        "build_charts",
        blocking_build_charts,
    )
    full_thread = threading.Thread(target=request_full_payload)
    summary_thread = threading.Thread(target=request_summary)
    full_thread.start()
    assert build_started.wait(timeout=5)
    summary_thread.start()
    try:
        assert summary_done.wait(timeout=1)
    finally:
        release_build.set()
    full_thread.join(timeout=10)
    summary_thread.join(timeout=10)

    assert not errors
    assert responses["summary"].status_code == 200
    assert responses["full"].status_code == 200


def test_terminal_cache_does_not_freeze_elapsed_without_finished_at(
    client,
    monkeypatch,
):
    c, app = client
    run_id = _result_run(
        app,
        name="unfinished error",
        net_assets=5000.0,
        status="error",
    )
    app.registry.conn_for(run_id).execute(
        "UPDATE runs SET started_at=?, finished_at=NULL WHERE run_id=?",
        ("2026-06-21T12:00:00", run_id),
    )

    class OnePM(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 21, 13, 0, 0, tzinfo=tz)

    class TwoPM(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 21, 14, 0, 0, tzinfo=tz)

    monkeypatch.setattr(leaderboard_mod, "datetime", OnePM)
    first = c.get("/dashboard/leaderboard.json").get_json()
    first_result = next(
        row["result"]
        for row in first["run_results"]
        if row["run_id"] == run_id
    )
    assert first_result["elapsed_ms"] == 60 * 60 * 1000

    monkeypatch.setattr(leaderboard_mod, "datetime", TwoPM)
    second_app = create_app(
        db_path=os.path.join(
            os.path.dirname(app.registry.runs_root),
            "elapsed-second-app.db",
        ),
        runs_root=app.registry.runs_root,
    )
    try:
        with second_app.test_client() as second_client:
            second = second_client.get("/dashboard/leaderboard.json").get_json()
        second_result = next(
            row["result"]
            for row in second["run_results"]
            if row["run_id"] == run_id
        )
        assert second_result["elapsed_ms"] == 2 * 60 * 60 * 1000
    finally:
        second_app.registry.shutdown()


def test_default_scenario_yaml_uses_365_day_horizon_and_activation_12():
    scenario = load_default_scenario()

    assert scenario["run"]["horizon_steps"] == 365 * 24
    assert scenario["agent"]["activation_period"] == 12
    assert scenario["run"]["virtual_time"] == {
        "enabled": True,
        "start_date": "2025-06-01",
    }


def test_default_scenario_yaml_uses_unscaled_event_rates():
    scenario = load_default_scenario()

    assert scenario["data"]["num_products"] == 1000
    assert scenario["data"]["num_suppliers"] == 200
    assert scenario["data"]["small_share"] == 1.0
    assert scenario["difficulty_rate"] == {
        "cancel_rate": 1.0,
        "refund_rate": 1.0,
        "only_refund_rate": 1.0,
        "bad_review_rate": 1.0,
        "timeout_rate": 1.0,
        "price_change_rate": 1.0,
        "supplier_delist_rate": 1.0,
    }
    assert scenario["lifecycle"] == {
        "start": 0.2,
        "ramp_days": 14,
        "decay_rate": 0.0092,
        "floor": 0.10,
    }


def test_default_scenario_yaml_uses_merchant_listing_rating_defaults():
    scenario = load_default_scenario()

    assert scenario["rating_outcomes"] == {
        "normal_score": 4.5,
        "late_score": 3.0,
        "refund_score": 2.0,
        "only_refund_score": 1.5,
        "bad_review_score": 1.0,
        "stockout_score": 1.0,
        "normal_weight": 1.0,
        "late_weight": 1.0,
        "refund_weight": 1.0,
        "only_refund_weight": 2.0,
        "bad_review_weight": 2.0,
        "stockout_weight": 3.0,
    }
    assert scenario["shop_rating"] == {
        "enabled": True,
        "model": "order_outcome_v4",
        "initial_rating": 4.0,
        "prior_weight": 0,
        "half_life_days": 180,
        "bucket_thresholds": [2.50, 3.30, 3.80, 4.20],
        "star_multipliers": [0.10, 0.35, 0.80, 1.00, 1.12],
    }
    assert scenario["public_reviews"] == {
        "enabled": True,
        "model": "self_selection_v1",
        "probability_by_star": [0.30, 0.18, 0.08, 0.06, 0.12],
        "demand": {
            "min_trust_multiplier": 0.80,
            "max_trust_multiplier": 1.00,
            "half_saturation_reviews": 20,
        },
    }
    assert scenario["listing_rating"] == {
        "initial_rating": 4.0,
        "prior_weight": 20,
        "half_life_days": 90,
    }


def test_legacy_experiment_pages_redirect_to_dashboard(client):
    c, _ = client

    experiments = c.get("/experiments")
    leaderboard = c.get("/leaderboard")

    assert experiments.status_code in (301, 302)
    assert experiments.headers["Location"].endswith("/dashboard")
    assert leaderboard.status_code in (301, 302)
    assert leaderboard.headers["Location"].endswith("/dashboard")


def test_new_run_page_renders_model_virtual_time_and_pricing_controls(client):
    c, _ = client

    resp = c.get("/new_run")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'name="bootstrap_agent" value="human"' in html
    assert "<strong>Human</strong>" in html
    assert 'name="bootstrap_agent" value="rule_based"' in html
    assert 'name="rule_based_mode"' in html
    assert '<option value="daily_report">' in html
    assert '<option value="random">' in html
    assert 'name="human_model"' in html
    assert 'list="human-model-options"' in html
    for name in (
        "Beethoven", "Mozart", "Chopin", "Bach", "Liszt", "Schubert",
        "Tchaikovsky", "Vivaldi", "Rachmaninoff", "Debussy",
    ):
        assert f'value="{name}"' in html
    assert 'name="bootstrap_agent" value="react_160k_compact_30k"' in html
    assert "<strong>React</strong>" in html
    assert 'name="bootstrap_agent" value="hermes"' in html
    assert "<strong>Hermes</strong>" in html
    assert 'name="react_model"' in html
    assert 'list="react-model-options"' in html
    assert 'name="virtual_time_enabled"' in html
    assert 'name="virtual_time_enabled" id="virtual-time-enabled" checked' in html
    assert 'name="virtual_start_date"' in html
    assert 'name="virtual_start_date" value="2025-06-01"' in html
    assert 'name="data_anchor_date" placeholder="2025-06-01"' in html
    assert "private_real defaults to 2025-06-01" in html
    assert 'value="qwen3.7-max"' in html
    assert 'value="gpt-5.5-0424-global"' in html
    for model in (
        "gpt-5.6-sol",
        "bailian/deepseek-v4-flash",
        "qwen3.7-plus",
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "bailian/kimi-k2.6",
        "moonshot/kimi-k3",
    ):
        assert f'value="{model}"' in html
    assert 'name="cost_input_per_million"' in html
    assert 'name="cost_output_per_million"' in html
    assert 'name="cost_cached_input_per_million"' in html
    assert 'const HERMES_SCENARIO_NAME = "agents/hermes";' in html
    assert 'loadScenarioByName(HERMES_SCENARIO_NAME)' in html
    assert 'name="react_model_preset"' not in html
    assert "ReAct model preset" not in html
    assert "ReAct model name" not in html
    assert 'name="cost_input_per_1k"' not in html
    assert "USD / 1M tokens" in html
    assert 'name="interval_ms"' not in html
    assert "Experiment Settings" not in html
    assert "Token Pricing" not in html
    assert "Step interval ms" not in html
    assert "queue" not in html.lower()


def test_new_run_model_input_clears_pricing_for_custom_model():
    html = Path("env/web/templates/new_run.html").read_text(encoding="utf-8")

    assert re.search(
        r"function applyModelPricing\(\) \{\s*"
        r"const preset = pricingByModel\[reactModel\.value\];\s*"
        r"if \(preset\) \{\s*"
        r"setPricingInputs\(preset\);\s*"
        r"\} else \{\s*"
        r"setPricingInputs\(null\);\s*"
        r"\}\s*"
        r"\}",
        html,
    )


def test_dashboard_template_renders_reasoning_content_blocks():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "reasoning_content" in html
    assert "renderReasoningBlock" in html
    assert "Model reasoning" in html


def test_frontend_templates_share_pixel_minimal_theme():
    dashboard_html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    new_run_html = Path("env/web/templates/new_run.html").read_text(encoding="utf-8")
    theme_html = Path("env/web/templates/_pixel_theme.html").read_text(encoding="utf-8")

    assert '{% include "_pixel_theme.html" %}' in dashboard_html
    assert '{% include "_pixel_theme.html" %}' in new_run_html
    assert "MerchantBench pixel minimal theme" in theme_html
    assert "--pixel-ink: #0F1720" in theme_html
    assert "--pixel-shadow: #CBD5E1" in theme_html
    assert "--pixel-grid: #E5E7EB" in theme_html
    assert "--pixel-chart-bg: var(--panel)" in theme_html
    assert ".topbar" in theme_html
    assert ".section, .card" in theme_html
    assert "button, input, select, textarea" in theme_html
    assert ".chart-wrap" in theme_html
    assert "border-radius: 0" in theme_html
    assert "box-shadow: 3px 3px 0 var(--pixel-shadow)" in theme_html
    for off_theme_color in ("#fffdf2", "#f7f0d5", "#d8cfaa", "#d0c7a2", "#e0d8b7"):
        assert off_theme_color not in theme_html


def test_full_page_templates_share_favicon():
    templates = Path("env/web/templates")
    favicon = (templates / "_favicon.html").read_text(encoding="utf-8")

    assert 'rel="icon"' in favicon
    assert "data:image/svg+xml" in favicon

    for name in ("dashboard.html", "new_run.html", "human_playground.html"):
        html = (templates / name).read_text(encoding="utf-8")
        assert '{% include "_favicon.html" %}' in html


def test_new_run_submit_creates_and_starts_run_with_options(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    scenario["run"]["interval_ms"] = 0
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "scenario": scenario_arg,
            "master_seed": master_seed,
            "name": name,
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_base_url": bootstrap_base_url,
            "auto_start": auto_start,
            "interval_ms": interval_ms,
            "bootstrap_config": bootstrap_config,
        })
        return "run-direct"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "direct react160k",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "react_160k_compact_30k",
        "react_model": "model-a",
        "virtual_time_enabled": "on",
        "virtual_start_date": "2025-06-15",
        "data_anchor_date": "2025-06-10",
        "cost_input_per_million": "250",
        "cost_output_per_million": "750",
        "cost_cached_input_per_million": "25",
    })

    assert resp.status_code in (301, 302)
    assert resp.headers["Location"].endswith("/dashboard?run_id=run-direct")
    assert captured["name"] == "direct react160k"
    assert captured["bootstrap_agent"] == "react_160k_compact_30k"
    assert captured["bootstrap_config"] == {"react_model": "model-a"}
    assert captured["auto_start"] is True
    assert captured["interval_ms"] == 0
    assert captured["scenario"]["run"]["virtual_time"] == {
        "enabled": True,
        "start_date": "2025-06-15",
    }
    assert captured["scenario"]["data"]["calendar_anchor_date"] == "2025-06-10"
    assert captured["scenario"]["agent"]["cost_pricing"] == {
        "input_per_million": 250.0,
        "output_per_million": 750.0,
        "cached_input_per_million": 25.0,
    }


def test_new_run_submit_supports_compact_react_bootstrap(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "scenario": scenario_arg,
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_config": bootstrap_config,
        })
        return "run-compact"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "compact react",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "react_160k_compact_30k",
        "react_model": "model-a",
        "virtual_time_enabled": "on",
        "virtual_start_date": "2025-06-15",
    })

    assert resp.status_code in (301, 302)
    assert captured["bootstrap_agent"] == "react_160k_compact_30k"
    assert captured["bootstrap_config"] == {"react_model": "model-a"}


def test_new_run_submit_supports_rule_based_random_mode(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_config": bootstrap_config,
        })
        return "run-rule-based"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "random baseline",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "rule_based",
        "rule_based_mode": "random",
    })

    assert resp.status_code in (301, 302)
    assert captured["bootstrap_agent"] == "rule_based"
    assert captured["bootstrap_config"] == {"selection_mode": "random"}


def test_new_run_submit_supports_hermes_bootstrap(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "scenario": scenario_arg,
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_config": bootstrap_config,
        })
        return "run-hermes"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "hermes",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "hermes",
        "react_model": "model-a",
        "virtual_time_enabled": "on",
        "virtual_start_date": "2025-06-15",
    })

    assert resp.status_code in (301, 302)
    assert captured["bootstrap_agent"] == "hermes"
    assert captured["bootstrap_config"] == {"react_model": "model-a"}


def test_new_run_submit_uses_builtin_model_preset_pricing(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "scenario": scenario_arg,
            "name": name,
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_config": bootstrap_config,
        })
        return "run-preset"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "preset react160k",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "react_160k_compact_30k",
        "react_model": "bailian/deepseek-v4-flash",
        "virtual_time_enabled": "on",
        "virtual_start_date": "2025-06-15",
    })

    assert resp.status_code in (301, 302)
    assert captured["bootstrap_agent"] == "react_160k_compact_30k"
    assert captured["bootstrap_config"] == {"react_model": "bailian/deepseek-v4-flash"}
    assert captured["scenario"]["agent"]["cost_pricing"] == pytest.approx({
        "input_per_million": 0.14,
        "output_per_million": 0.28,
        "cached_input_per_million": 0.03,
    })


def test_new_run_submit_human_redirects_to_playground(client, monkeypatch):
    c, app = client
    scenario = _tiny_scenario()
    captured = {}

    def fake_create_run(
        scenario_arg,
        master_seed=None,
        name=None,
        bootstrap_agent="none",
        bootstrap_base_url=None,
        auto_start=False,
        interval_ms=500,
        bootstrap_config=None,
    ):
        captured.update({
            "scenario": scenario_arg,
            "name": name,
            "bootstrap_agent": bootstrap_agent,
            "bootstrap_base_url": bootstrap_base_url,
            "auto_start": auto_start,
            "interval_ms": interval_ms,
            "bootstrap_config": bootstrap_config,
        })
        return "run-human"

    monkeypatch.setattr(app.registry, "create_run", fake_create_run)

    resp = c.post("/new_run", data={
        "name": "human player",
        "scenario_yaml": yaml.safe_dump(scenario),
        "bootstrap_agent": "human",
        "human_model": "Session Player",
    })

    assert resp.status_code in (301, 302)
    assert resp.headers["Location"].endswith(
        "/runs/run-human/playground?agent_id=agent_0"
    )
    assert captured["name"] == "human player"
    assert captured["bootstrap_agent"] == "human"
    assert captured["auto_start"] is True
    assert captured["bootstrap_config"] == {"human_model": "Session Player"}


def test_registry_accepts_human_without_spawning_baseline(client, monkeypatch):
    _, app = client
    scenario = _tiny_scenario()

    monkeypatch.setattr(
        app.registry,
        "_spawn_auto_seed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("human must not spawn auto_seed")
        ),
    )
    monkeypatch.setattr(
        app.registry,
        "_spawn_react_160k_compact_30k",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("human must not spawn react160k")
        ),
    )

    run_id = app.registry.create_run(
        scenario,
        name="human direct",
        bootstrap_agent="human",
        auto_start=False,
    )

    row = dbm.get_run(app.registry.conn_for(run_id), run_id)
    assert row["bootstrap_agent"] == "human"
    assert app.registry.bootstrap_procs == {}


def test_dashboard_all_runs_shows_framework_and_model(client):
    c, app = client
    _result_run(app, name="finished react", net_assets=5000.0)
    _result_run(
        app,
        name="stopped auto_seed",
        net_assets=4500.0,
        bootstrap_agent="auto_seed",
        react_model=None,
        status="stopped",
    )
    human_id = _result_run(
        app,
        name="human stopped",
        net_assets=4700.0,
        bootstrap_agent="human",
        react_model=None,
        status="stopped",
    )

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    all_runs_html = html.split(
        '<div class="title" id="all-runs-title">All Runs</div>', 1
    )[1].split(
        '<div class="title">Leaderboard</div>', 1
    )[0]
    assert "<th>FRAMEWORK</th>" in all_runs_html
    assert "<th>MODEL</th>" in all_runs_html
    assert "all-runs-table" in all_runs_html
    assert all_runs_html.index("<th>TOKENS</th>") < all_runs_html.index("<th>USD</th>")
    assert re.search(
        r'<td><span class="lb-framework-badge[^"]*">React</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="model-a">.*?'
        r'<span class="model-name">model-a</span>\s*</span>',
        all_runs_html,
        re.S,
    )
    assert re.search(
        r'<td><span class="lb-framework-badge[^"]*">Auto Seed</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="—">.*?'
        r'<span class="model-name">—</span>\s*</span>',
        all_runs_html,
        re.S,
    )
    assert re.search(
        r'<td><span class="lb-framework-badge[^"]*">Human</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="—">.*?'
        r'<span class="model-name">—</span>\s*</span>',
        all_runs_html,
        re.S,
    )
    assert f"/dashboard?run_id={human_id}" in all_runs_html


def test_dashboard_all_runs_includes_human_run_without_metrics(client):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="human paused at start",
        bootstrap_agent="human",
        bootstrap_config={"human_model": "Mozart"},
        auto_start=False,
    )
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "paused")

    html = c.get("/dashboard").data.decode("utf-8")
    all_runs_html = html.split(
        '<div class="title" id="all-runs-title">All Runs</div>', 1
    )[1].split(
        '<div class="title">Leaderboard</div>', 1
    )[0]

    assert run_id in all_runs_html
    assert "human paused at start" in all_runs_html
    assert "Mozart" in all_runs_html


def test_dashboard_all_runs_status_controls_stopped_paused_and_running(client):
    c, app = client
    stopped_id = _result_run(
        app, name="stopped run", net_assets=4100.0, status="stopped"
    )
    running_id = _result_run(
        app, name="running run", net_assets=4200.0, status="running"
    )
    paused_id = _result_run(
        app, name="paused run", net_assets=4150.0, status="paused"
    )
    _mark_live_worker(app, running_id)
    _mark_live_worker(app, paused_id, state="paused")

    html = c.get("/dashboard").data.decode("utf-8")
    all_runs_html = html.split(
        '<div class="title" id="all-runs-title">All Runs</div>', 1
    )[1].split(
        '<div class="title">Leaderboard</div>', 1
    )[0]

    assert f'data-run-id="{stopped_id}" data-run-state="stopped"' in all_runs_html
    assert f'data-run-id="{paused_id}" data-run-state="paused"' in all_runs_html
    assert f'data-run-id="{running_id}" data-run-state="running"' in all_runs_html
    assert 'title="Continue run">stopped</button>' in all_runs_html
    assert 'title="Continue run">paused</button>' in all_runs_html
    assert 'title="Pause run">running</button>' in all_runs_html


def test_dashboard_leaderboard_uses_terminal_and_live_runs_with_result_metrics(client):
    c, app = client
    finished_id = _result_run(
        app,
        name="finished react",
        net_assets=5000.0,
        cost_usd=12.345,
    )
    _result_run(
        app,
        name="stopped auto_seed",
        net_assets=4500.0,
        bootstrap_agent="auto_seed",
        react_model=None,
        status="stopped",
    )
    running_id = _result_run(
        app,
        name="running react",
        net_assets=9000.0,
        status="running",
    )
    _mark_live_worker(app, running_id)

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    leaderboard_html = html.split('id="tbl-leaderboard"', 1)[1].split(
        '<h3 style="margin-top:14px;">Analysis</h3>', 1
    )[0]
    assert finished_id in leaderboard_html
    assert "finished react" in leaderboard_html
    assert "model-a" in html
    assert "Auto Seed" in html
    assert '<th>show</th><th data-sort="rank" class="sortable">rank</th>' in leaderboard_html
    assert '<th data-sort="name" class="sortable">name</th>' in leaderboard_html
    assert '<th data-sort="framework" class="sortable">framework</th>' in leaderboard_html
    assert '<th data-sort="model" class="sortable">model</th>' in leaderboard_html
    for col in (
        '<th data-sort="avg_cum_gmv" class="sortable">gmv</th>',
        '<th data-sort="avg_net_profit" class="sortable">profit</th>',
        '<th data-sort="avg_cum_fine" class="sortable">total fines</th>',
        '<th data-sort="avg_orders" class="sortable">orders</th>',
    ):
        assert col in leaderboard_html
    assert "mean score" not in leaderboard_html
    assert "<th>min</th>" not in leaderboard_html
    assert "<th>std</th>" not in leaderboard_html
    assert '<th data-sort="avg_tokens" class="sortable">tokens</th>' in leaderboard_html
    assert leaderboard_html.index(
        '<th data-sort="avg_tokens" class="sortable">tokens</th>'
    ) < leaderboard_html.index(
        '<th data-sort="avg_usd" class="sortable">usd</th>'
    )
    assert "<th>survival</th>" not in leaderboard_html
    assert '<th data-sort="avg_t" class="sortable">t</th>' in leaderboard_html
    assert "<td>1k</td>" in leaderboard_html
    assert "const fmtTokens = v =>" in html
    assert 'axisFormat: value => fmtTokens(value)' in html
    assert "12.35" in leaderboard_html
    assert running_id in leaderboard_html
    assert "running react" in leaderboard_html


def test_dashboard_leaderboard_json_includes_live_runs_for_realtime_updates(client):
    c, app = client
    running_id = _result_run(
        app,
        name="running react",
        net_assets=9000.0,
        status="running",
    )
    _mark_live_worker(app, running_id)

    resp = c.get("/dashboard/leaderboard.json")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert running_id in {row["run_id"] for row in payload["run_results"]}
    assert running_id in {row["run_id"] for row in payload["leaderboard"]}
    net_assets = next(row for row in payload["charts"]["net_assets"] if row["run_id"] == running_id)
    assert net_assets["data"][-1] == [4, 9000.0]


def test_dashboard_leaderboard_excludes_stale_live_db_state_without_worker(client):
    _, app = client
    running_id = _result_run(
        app,
        name="stale running react",
        net_assets=9000.0,
        status="running",
    )

    assert running_id not in {row["run_id"] for row in build_run_results(app.registry)}


def test_dashboard_leaderboard_excludes_live_state_worker_without_thread(client):
    _, app = client
    running_id = _result_run(
        app,
        name="threadless running react",
        net_assets=9000.0,
        status="running",
    )

    class DummyWorker:
        state = "running"

    with app.registry.lock:
        app.registry.workers[running_id] = DummyWorker()

    assert running_id not in {row["run_id"] for row in build_run_results(app.registry)}


def test_dashboard_overview_polls_full_leaderboard_payload_slowly():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "async function refreshLeaderboardPayload" in html
    assert 'fetch("/dashboard/leaderboard.json"' in html
    assert "leaderboard_summary" not in html
    assert "renderCharts" not in html
    assert "let leaderboardRefreshInFlight = false;" in html
    assert "if (leaderboardRefreshInFlight) return;" in html
    assert "leaderboardRefreshInFlight = false;" in html
    assert "setInterval(() => {" in html
    assert "refreshLeaderboardPayload();" in html
    assert "refreshExperimentRunOptions();" in html
    assert "}, 120000);" in html


def test_dashboard_initial_html_keeps_full_chart_history_out_of_first_paint(client):
    c, app = client
    run_id = _result_run(app, name="summary first", net_assets=5000.0)

    html = c.get("/dashboard").data.decode("utf-8")
    match = re.search(
        r"const INITIAL_LEADERBOARD_PAYLOAD = (.*?);\n  const STEP_HOURS",
        html,
        re.S,
    )

    assert match is not None
    payload = json.loads(match.group(1))
    assert run_id in {row["run_id"] for row in payload["leaderboard"]}
    assert payload["charts"]["net_assets"] == []
    assert "refreshLeaderboardPayload();" in html


def test_single_run_dashboard_does_not_embed_global_leaderboard_payload(client):
    c, app = client
    run_id = _result_run(app, name="detail only", net_assets=5000.0)

    html = c.get(f"/dashboard?run_id={run_id}").data.decode("utf-8")

    assert "const INITIAL_LEADERBOARD_PAYLOAD = null;" in html


def test_dashboard_leaderboard_lists_each_run_without_grouping(client):
    c, app = client
    first_id = _result_run(app, name="react one", net_assets=5000.0, react_model="model-a")
    second_id = _result_run(app, name="react two", net_assets=7000.0, react_model="model-a", seed=43)
    auto_id = _result_run(
        app,
        name="auto seed one",
        net_assets=4500.0,
        bootstrap_agent="auto_seed",
        react_model=None,
    )
    human_id = _result_run(
        app,
        name="human one",
        net_assets=4600.0,
        bootstrap_agent="human",
        react_model=None,
    )

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    rows = build_leaderboard(build_run_results(app.registry))
    assert [row["run_id"] for row in rows] == [second_id, first_id, human_id, auto_id]
    assert all(row["runs"] == 1 for row in rows)
    assert "react one" in html
    assert "react two" in html
    assert re.search(
        r'<span class="lb-framework-badge[^"]*">React</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="model-a">.*?'
        r'<span class="model-name">model-a</span>\s*</span>',
        html,
        re.S,
    )
    assert re.search(
        r'<span class="lb-framework-badge[^"]*">Auto Seed</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="—">.*?'
        r'<span class="model-name">—</span>\s*</span>',
        html,
        re.S,
    )
    assert re.search(
        r'<span class="lb-framework-badge[^"]*">Human</span></td>\s*'
        r'<td>\s*<span class="model-identity">\s*<span class="model-only-icon[^"]*" title="—">.*?'
        r'<span class="model-name">—</span>\s*</span>',
        html,
        re.S,
    )


def test_dashboard_leaderboard_headers_are_click_sortable_like_run_tables(client):
    c, app = client
    _result_run(app, name="react one", net_assets=5000.0)

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    leaderboard_html = html.split('id="tbl-leaderboard"', 1)[1].split(
        '<h3 style="margin-top:14px;">Analysis</h3>', 1
    )[0]
    for field, label in (
        ("rank", "rank"),
            ("name", "name"),
        ("framework", "framework"),
        ("model", "model"),
        ("avg_final_net_assets", "final net assets"),
        ("avg_cum_gmv", "gmv"),
        ("avg_net_profit", "profit"),
        ("avg_cum_fine", "total fines"),
        ("avg_orders", "orders"),
        ("avg_shop_rating_score", "average store rating"),
        ("avg_tokens", "tokens"),
        ("avg_usd", "usd"),
        ("avg_t", "t"),
    ):
        assert f'<th data-sort="{field}" class="sortable">{label}</th>' in leaderboard_html
    assert "function sortLeaderboardRows" in html
    assert 'document.querySelector("#tbl-leaderboard thead").addEventListener("click"' in html


def test_dashboard_chart_run_rows_follow_leaderboard_rank_order(client):
    _, app = client
    low_id = _result_run(
        app,
        name="low",
        net_assets=4200.0,
        react_model="low-model",
        tool_calls=["query_my_orders"],
        cost_usd=0.03,
    )
    high_id = _result_run(
        app,
        name="high",
        net_assets=7200.0,
        react_model="high-model",
        tool_calls=["query_balance"],
        cost_usd=0.01,
    )
    mid_id = _result_run(
        app,
        name="mid",
        net_assets=5600.0,
        react_model="mid-model",
        tool_calls=["query_platform_rules"],
        cost_usd=0.02,
    )

    run_results = build_run_results(app.registry)
    leaderboard_order = [row["run_id"] for row in build_leaderboard(run_results)]
    charts = build_charts(app.registry, run_results)

    assert leaderboard_order == [high_id, mid_id, low_id]
    assert [row["run_id"] for row in charts["runs"]] == leaderboard_order
    assert [row["run_id"] for row in charts["net_assets"]] == leaderboard_order
    assert [row["run_id"] for row in charts["net_assets_cost"]] == leaderboard_order
    assert [row["run_id"] for row in charts["tool_calls"]["runs"]] == leaderboard_order
    assert [
        row["run_id"]
        for row in charts["weekly"]["metrics"]["total_tool_calls"]
    ] == leaderboard_order


def test_split_chart_payload_merge_matches_full_build(client):
    _, app = client
    first_id = _result_run(
        app,
        name="first",
        net_assets=4200.0,
        react_model="first-model",
        tool_calls=["query_balance"],
    )
    second_id = _result_run(
        app,
        name="second",
        net_assets=7200.0,
        react_model="second-model",
        tool_calls=["query_my_orders"],
    )
    run_results = build_run_results(app.registry)
    by_id = {row["run_id"]: row for row in run_results}

    full = build_charts(app.registry, run_results)
    merged = merge_chart_payloads(
        [
            build_charts(app.registry, [by_id[first_id]]),
            build_charts(app.registry, [by_id[second_id]]),
        ],
        run_results,
    )

    assert merged == full


def test_split_chart_payload_merge_matches_full_with_mixed_start_dates(client):
    _, app = client
    first_id = _result_run(
        app,
        name="january",
        net_assets=4200.0,
        virtual_start_date="2025-01-15",
    )
    second_id = _result_run(
        app,
        name="june",
        net_assets=7200.0,
        virtual_start_date="2025-06-01",
    )
    run_results = build_run_results(app.registry)
    by_id = {row["run_id"]: row for row in run_results}

    full = build_charts(app.registry, run_results)
    merged = merge_chart_payloads(
        [
            build_charts(app.registry, [by_id[first_id]]),
            build_charts(app.registry, [by_id[second_id]]),
        ],
        run_results,
    )

    assert merged == full
    assert merged["monthly"]["start_date"] is None
    assert merged["monthly"]["periods"] == [{
        "index": 1,
        "start_day": 0,
        "end_day": 30,
    }]


def test_leaderboard_order_totals_and_series_are_agent_scoped(client):
    _, app = client
    run_id = _result_run(
        app,
        name="multi-agent orders",
        net_assets=5000.0,
    )
    conn = app.registry.conn_for(run_id)
    conn.executemany(
        "INSERT INTO orders(run_id, order_id, agent_id, order_t)"
        " VALUES (?, ?, ?, ?)",
        [
            (run_id, "agent-0-order", "agent_0", 1),
            (run_id, "agent-1-order-a", "agent_1", 1),
            (run_id, "agent-1-order-b", "agent_1", 4),
        ],
    )

    assert leaderboard_mod._orders_generated_total(
        conn,
        run_id,
        "agent_0",
    ) == 1.0
    assert leaderboard_mod._orders_generated_total(
        conn,
        run_id,
        "agent_2",
    ) == 0.0
    assert leaderboard_mod._cum_orders_payload(
        conn,
        run_id,
        "agent_0",
        step_hours=1,
    ) == [[1, 1.0], [4, 1.0]]


def test_tool_step_counts_cache_is_bounded(client, monkeypatch):
    _, app = client
    monkeypatch.setattr(
        leaderboard_mod,
        "_TOOL_STEP_COUNTS_CACHE_MAX_ENTRIES",
        2,
    )
    with leaderboard_mod._TOOL_STEP_COUNTS_CACHE_LOCK:
        leaderboard_mod._TOOL_STEP_COUNTS_CACHE.clear()
    try:
        run_ids = [
            _result_run(
                app,
                name=f"cache-{index}",
                net_assets=5000.0 + index,
                tool_calls=["query_balance"],
            )
            for index in range(3)
        ]
        for run_id in run_ids:
            leaderboard_mod._tool_call_step_counts(app.registry, run_id)

        with leaderboard_mod._TOOL_STEP_COUNTS_CACHE_LOCK:
            cached_run_ids = [
                cache_key[1]
                for cache_key in leaderboard_mod._TOOL_STEP_COUNTS_CACHE
            ]
        assert cached_run_ids == run_ids[-2:]
    finally:
        with leaderboard_mod._TOOL_STEP_COUNTS_CACHE_LOCK:
            leaderboard_mod._TOOL_STEP_COUNTS_CACHE.clear()


def test_compact_react_run_results_keep_framework_and_model_labels(client):
    c, app = client
    _result_run(
        app,
        name="compact react",
        net_assets=5000.0,
        bootstrap_agent="react_160k_compact_30k",
        react_model="model-a",
    )

    row = build_run_results(app.registry)[0]

    assert row["bootstrap_agent"] == "react_160k_compact_30k"
    assert row["framework"] == "React"
    assert row["model"] == "model-a"


def test_human_run_identity_uses_selected_participant_model():
    identity = leaderboard_mod.run_identity({
        "bootstrap_agent": "human",
        "bootstrap_config": {"human_model": "Beethoven"},
    })

    assert identity == {
        "framework_key": "human",
        "framework": "Human",
        "model": "Beethoven",
        "display_label": "Human (Beethoven)",
    }


def test_rule_based_run_identity_uses_selection_mode():
    identity = leaderboard_mod.run_identity({
        "bootstrap_agent": "rule_based",
        "bootstrap_config": {"selection_mode": "random"},
    })

    assert identity == {
        "framework_key": "rule_based",
        "framework": "Rule-based",
        "model": "random",
        "display_label": "Rule-based (random)",
    }


def test_hermes_run_results_keep_registered_model_label(client):
    c, app = client
    run_id = _result_run(
        app,
        name="hermes",
        net_assets=5000.0,
        bootstrap_agent="hermes",
        react_model=None,
    )
    agent_log.write_meta(
        app.registry.runs_root,
        run_id,
        {
            "agent_id": "agent_0",
            "framework": "hermes",
            "model": "bailian/glm-5.2",
            "version": "test",
        },
    )

    row = build_run_results(app.registry)[0]

    assert row["bootstrap_agent"] == "hermes"
    assert row["framework"] == "Hermes"
    assert row["model"] == "bailian/glm-5.2"
    assert row["display_label"] == "Hermes (bailian/glm-5.2)"

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    assert "bailian/glm-5.2" in resp.data.decode("utf-8")


def test_human_playground_route_renders_protocol_config(client):
    c, app = client
    scenario = _tiny_scenario()
    scenario["run"]["max_hook_seconds"] = 600
    scenario["agent"]["activation_period"] = 12
    scenario["agent"]["max_turns_per_step"] = 30
    run_id = app.registry.create_run(
        scenario,
        name="human playground",
        bootstrap_agent="human",
        bootstrap_config={"human_model": "Chopin"},
        auto_start=False,
    )

    resp = c.get(f"/runs/{run_id}/playground?agent_id=agent_0")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "人工经营工作台" in html
    assert '"maxHookSeconds": 600' in html
    assert '"activationPeriod": 12' in html
    assert '"maxTurnsPerStep": 30' in html
    assert '"modelName": "Chopin"' in html
    assert "Human - Chopin" in html
    assert 'model: PLAYGROUND_CONFIG.modelName || "human-playground"' in html
    assert 'provider_api_failed_attempts: "not_applicable"' in html
    assert 'skills_evolutions: "not_applicable"' in html
    assert f'"/runs/{run_id}/tools/schema"' in html
    assert f'"/runs/{run_id}/pause"' in html
    assert f'"/runs/{run_id}/resume"' in html
    assert f'"/runs/{run_id}/agents/agent_0/trace_index"' in html
    assert f'"/runs/{run_id}/agents/agent_0/observation"' in html
    assert f'"/runs/{run_id}/agents/agent_0/act"' in html
    assert f'"/runs/{run_id}/agent/all_traces"' not in html
    assert f'"/runs/{run_id}/agent/all_traces_index"' not in html
    assert f'"/runs/{run_id}/agent/trace"' not in html
    assert f'"/runs/{run_id}/agents/agent_0/playground/dashboard-data"' in html
    assert '"platformRules": {' in html
    assert "query_product_sales_trend" not in html
    assert "query_my_listings" not in html
    assert "/sections/supplier" not in html
    assert "Run Dashboard" not in html
    assert '"dashboardUrl"' not in html
    assert html.index('class="app-shell"') < html.index(
        'id="platform-rules-summary"'
    )


def test_human_playground_dashboard_data_is_safe_and_tool_schema_is_unchanged(client):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="human dashboard data",
        bootstrap_agent="human",
        auto_start=False,
    )
    env = app.registry.get_env(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        sale_price=float(product.price) * 2,
        listed_at=0,
        first_listed_at=0,
    )
    env.agents["agent_0"].listings[product.product_id] = listing
    dbm.upsert_listing(
        app.registry.conn_for(run_id), run_id, "agent_0", listing,
    )
    dbm.insert_orders(app.registry.conn_for(run_id), run_id, [
        Order(
            order_id="future-order",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=2,
            promised_delivery_t=30,
            sale_price=100.0,
            purchase_price=70.0,
            current_status="ordered",
        ),
    ])

    response = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == {
        "t",
        "agent_id",
        "cash",
        "listings",
        "series",
        "listing_ops",
        "daily_sales_by_product",
        "shop_rating",
        "product_names",
    }
    assert set(payload["series"]) == {
        "balance",
        "deposit_pool",
        "in_transit",
        "receivable",
        "net_assets",
        "n_active_listings",
        "cum_gmv",
        "cum_cost",
        "cum_gross_profit",
        "cum_net_profit",
        "cum_fine",
        "cum_fee",
        "shop_rating_mean",
        "shop_rating_score",
    }
    assert payload["shop_rating"] == {
        "enabled": True,
        "model": "order_outcome_v4",
        "score": None,
        "stars": None,
        "rated_order_count": 0,
        "qualified_transaction_count": 0,
        "reputation_evidence_count": 0,
        "quality_multiplier": 1.0,
        "reputation_multiplier": 0.8,
        "demand_multiplier": 0.8,
        "service_quality_score": 4.0,
        "service_quality_stars": 4,
        "service_quality_multiplier": 1.0,
        "rating_available": False,
        "demand_source": "public_reviews",
        "public_reviews": {
            "model": "self_selection_v1",
            "rating": None,
            "count": 0,
            "eligible_count": 0,
            "response_rate": 0.0,
            "full_response_rating": None,
            "selection_gap": None,
            "quality_gap": None,
            "affects_demand": True,
            "stars": None,
            "confidence": 0.0,
            "raw_quality_multiplier": 1.0,
            "quality_multiplier": 1.0,
            "reputation_multiplier": 0.8,
            "demand_multiplier": 0.8,
        },
    }
    assert set(payload["listing_ops"]) == {"grain", "days", "buckets", "series"}
    assert set(payload["listing_ops"]["series"]) == {
        "ops", "list", "delist", "price", "promise",
    }
    assert all(
        set(bucket) == {"key", "label", "start_day", "end_day"}
        for bucket in payload["listing_ops"]["buckets"]
    )
    assert set(payload["daily_sales_by_product"]) == {
        "grain", "days", "buckets", "series",
    }
    assert all(
        set(bucket) == {"key", "label", "start_day", "end_day"}
        for bucket in payload["daily_sales_by_product"]["buckets"]
    )
    assert all(
        set(product_row) == {"product_id", "name", "category", "data"}
        for product_row in payload["daily_sales_by_product"]["series"]
    )
    assert all(
        set(point) == {
            "bucket", "label", "start_day", "end_day", "day", "orders",
            "value", "gmv", "gross_profit", "net_profit",
            "supply_chain_anomalies", "order_anomalies",
        }
        for product_row in payload["daily_sales_by_product"]["series"]
        for point in product_row["data"]
    )
    assert len(payload["listings"]) == 1
    assert product.product_id not in payload["product_names"]
    assert set(payload["listings"][0]) == {
        "product_id",
        "name",
        "category",
        "quantity",
        "sale_price",
        "supplier_price",
        "price_ratio",
        "supplier_id",
        "supplier_name",
        "supplier_ship_hours",
        "supplier_logistics_hours",
        "historical_avg_rating",
        "shop_rating",
        "supplier_age_years",
        "procured_orders",
        "cum_gross_profit",
        "cum_net_profit",
        "cum_fine",
        "listing_rating",
        "first_listed_day",
        "last_sale_day",
        "days_without_sales",
    }
    encoded = response.get_data(as_text=True)
    for hidden_field in (
        "elasticity",
        "cancel_rate",
        "refund_rate",
        "timeout_rate",
        "supplier_delist_rate",
        "future_demand",
    ):
        assert hidden_field not in encoded

    weekly = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data?level=week"
    )
    assert weekly.status_code == 200
    assert weekly.get_json()["daily_sales_by_product"]["grain"] == "week"
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 3)
    as_of_zero = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data"
        "?as_of=0&level=day"
    )
    assert as_of_zero.status_code == 400
    assert as_of_zero.get_json()["error"] == (
        "as_of is not supported by this endpoint"
    )
    invalid = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data?level=month"
    )
    assert invalid.status_code == 400

    schema = c.get(f"/runs/{run_id}/tools/schema").get_json()
    specs = {tool["name"]: tool for tool in schema["tools"]}
    assert "query_product_sales_trend" not in specs
    assert specs["query_my_listings"]["parameters"]["properties"] == {}
    assert specs["query_platform_rules"]["parameters"]["properties"] == {}


def test_merchant_dashboard_exposes_v4_public_review_demand_contract(
    client,
):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="public review dashboard",
        bootstrap_agent="none",
        auto_start=False,
    )

    response = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant")

    assert response.status_code == 200
    rating = response.get_json()["shop_rating"]
    assert rating["score"] is None
    assert rating["stars"] is None
    assert rating["rating_available"] is False
    assert rating["reputation_evidence_count"] == 0
    assert rating["public_reviews"] == {
        "model": "self_selection_v1",
        "rating": None,
        "count": 0,
        "eligible_count": 0,
        "response_rate": 0.0,
        "full_response_rating": None,
        "selection_gap": None,
        "quality_gap": None,
        "affects_demand": True,
        "confidence": 0.0,
        "raw_quality_multiplier": 1.0,
        "quality_multiplier": 1.0,
        "reputation_multiplier": 0.8,
        "demand_multiplier": 0.8,
    }


def test_human_playground_dashboard_data_limits_weekly_aggregates_to_requested_range(client):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="human dashboard range",
        bootstrap_agent="human",
        auto_start=False,
    )
    env = app.registry.get_env(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        sale_price=float(product.price) * 2,
        listed_at=0,
        first_listed_at=0,
    )
    env.agents["agent_0"].listings[product.product_id] = listing
    conn = app.registry.conn_for(run_id)
    dbm.upsert_listing(conn, run_id, "agent_0", listing)
    dbm.insert_orders(conn, run_id, [
        Order(
            order_id="before-range",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=24,
            promised_delivery_t=60,
            sale_price=50.0,
            purchase_price=20.0,
            current_status="ordered",
        ),
        Order(
            order_id="inside-range",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=120,
            promised_delivery_t=160,
            sale_price=80.0,
            purchase_price=30.0,
            current_status="ordered",
        ),
    ])
    dbm.write_events(conn, run_id, [
        EventLog(
            t=24,
            event_type="agent_list_product",
            entity_id=product.product_id,
            agent_id="agent_0",
            payload={},
        ),
        EventLog(
            t=144,
            event_type="agent_adjust_price",
            entity_id=product.product_id,
            agent_id="agent_0",
            payload={},
        ),
    ])
    with env.lock:
        env.t = 240
    dbm.update_run_t(conn, run_id, 240)

    response = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data"
        "?t_from=72&t_to=239&level=week"
    )

    assert response.status_code == 200
    payload = response.get_json()
    sales = payload["daily_sales_by_product"]
    assert sales["buckets"] == [{
        "key": "W1",
        "label": "W1",
        "start_day": 4,
        "end_day": 10,
    }]
    product_row = next(
        row for row in sales["series"]
        if row["product_id"] == product.product_id
    )
    assert product_row["data"][0]["orders"] == 1
    assert product_row["data"][0]["gmv"] == 80.0
    listing_ops = payload["listing_ops"]
    assert listing_ops["buckets"] == sales["buckets"]
    assert listing_ops["series"]["ops"] == [[4, 1]]
    assert listing_ops["series"]["price"] == [[4, 1]]
    assert listing_ops["series"]["list"] == [[4, 0]]

    invalid = c.get(
        f"/runs/{run_id}/agents/agent_0/playground/dashboard-data"
        "?t_from=not-an-integer&t_to=239&level=week"
    )
    assert invalid.status_code == 400
    assert invalid.get_json()["error"] == "t_from and t_to must be integers"


def test_human_playground_config_is_script_safe_for_agent_names(client):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="human playground",
        bootstrap_agent="human",
        auto_start=False,
    )
    malicious_name = "</script><script>window.__merchantbench_xss=1</script>"
    app.registry.add_agent(run_id, "agent_xss", malicious_name)

    resp = c.get(f"/runs/{run_id}/playground?agent_id=agent_xss")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert malicious_name not in html
    assert "\\u003c/script\\u003e" in html
    assert "\\u003cscript\\u003ewindow.__merchantbench_xss=1\\u003c/script\\u003e" in html


def test_human_playground_order_history_form_uses_paginated_parameters():
    html = Path("env/web/templates/human_playground.html").read_text(encoding="utf-8")
    order_form = html.split('data-tool="query_my_orders"', 1)[1].split(
        "</form>", 1
    )[0]

    assert 'name="page"' in order_form
    assert 'name="page_size"' in order_form
    assert 'name="supplier_id"' in order_form
    assert 'name="limit"' not in order_form


def test_human_playground_catalog_form_matches_search_products_contract():
    html = Path("env/web/templates/human_playground.html").read_text(
        encoding="utf-8"
    )
    catalog_form = html.split('data-tool="search_products"', 1)[1].split(
        "</form>", 1
    )[0]

    for field in (
        "query",
        "price_min",
        "price_max",
        "supplier_rating_min",
        "historical_rating_min",
        "logistics_hours_max",
        "supplier_ship_hours_max",
        "delivery_hours_max",
        "quantity_min",
        "sort_by",
        "page",
        "page_size",
    ):
        assert f'name="{field}"' in catalog_form
    for sort_value in (
        "relevance",
        "price_asc",
        "price_desc",
        "rating",
        "supplier_rating",
        "logistics_speed",
    ):
        assert f'value="{sort_value}"' in catalog_form
    assert 'id="catalog-page-size"' in catalog_form
    assert '<option value="20" selected>20 条</option>' in catalog_form


def test_human_playground_form_args_omit_blank_optional_numbers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for form serialization validation")
    source = Path("env/web/templates/_human_playground_script.html").read_text(
        encoding="utf-8"
    )
    source = source.strip().removeprefix("<script>\n").removesuffix("\n</script>")
    marker = "  init();\n})();"
    assert marker in source
    source = source.replace(
        marker,
        "  globalThis.__hpFormTest = {formArgs, orderTabToolCall, overviewAssetData, formatTime, formatSimulationDateTime, orderDetailHtml, priceRatioVisual, salesHeatTier, catalogListingPlan, mutationSummary, currentRiskDetails, eventRiskDetails, syncActiveListingIdsFromMutation, attentionProductIsActive, state};\n})();",
        1,
    )
    harness = r"""
globalThis.document = {getElementById() {}, querySelectorAll() { return []; }};
globalThis.window = {MerchantBenchMerchantCharts: null, addEventListener() {}};
globalThis.PLAYGROUND_CONFIG = {};
eval(require("fs").readFileSync(0, "utf8"));
const h = globalThis.__hpFormTest;
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
h.state.toolByName = new Map([["search_products", {
  parameters: {properties: {
    query: {type: "string"},
    price_min: {type: "number"},
    price_max: {type: "number"},
    quantity_min: {type: "integer"},
    sort_by: {type: "string"},
    page: {type: "integer"},
    page_size: {type: "integer"},
  }},
}]]);
const control = (name, value, type = "text") => ({
  name, value, type, disabled: false, checked: true,
});
const blankForm = {
  dataset: {tool: "search_products"},
  elements: [
    control("query", "风扇"),
    control("price_min", "", "number"),
    control("price_max", "", "number"),
    control("quantity_min", "", "number"),
    control("sort_by", "price_asc", "select-one"),
    control("page", "1", "hidden"),
    control("page_size", "20", "select-one"),
  ],
};
const blankArgs = h.formArgs(blankForm);
check(blankArgs.query === "风扇", "query was lost");
check(blankArgs.sort_by === "price_asc", "sort was lost");
check(blankArgs.page === 1 && blankArgs.page_size === 20, "pagination was not numeric");
check(!("price_min" in blankArgs), "blank price_min was serialized");
check(!("price_max" in blankArgs), "blank price_max was serialized as zero");
check(!("quantity_min" in blankArgs), "blank quantity_min was serialized");

blankForm.elements[2].value = "19.50";
blankForm.elements[4].value = "price_desc";
const explicitArgs = h.formArgs(blankForm);
check(explicitArgs.price_max === 19.5, "explicit price_max was not numeric");
check(explicitArgs.sort_by === "price_desc", "descending sort was not preserved");

const updatesCall = h.orderTabToolCall("updates");
const openCall = h.orderTabToolCall("open");
const historyCall = h.orderTabToolCall("history", {status: "delivered", page: 1});
check(updatesCall[0] === "query_order_updates", "updates tab uses the wrong tool");
check(updatesCall[1].include_ordered === true, "updates tab omitted ordered events");
check(updatesCall[1].page === 1 && updatesCall[1].page_size === 100,
  "updates tab omitted pagination");
check(openCall[0] === "query_open_orders", "open tab uses the wrong tool");
check(openCall[1].page === 1 && openCall[1].page_size === 50,
  "open tab omitted pagination");
check(historyCall[0] === "query_my_orders", "history tab uses the wrong tool");
check(historyCall[1].status === "delivered", "history filters were lost");

const assets = h.overviewAssetData({
  net_assets: 3000,
  balance: 1900,
  deposit_pool: 1000,
  in_transit: 60,
  receivable: 40,
});
check(assets.components.length === 4, "asset composition omitted a component");
check(assets.bars.length === 5, "asset comparison omitted net assets");
check(assets.bars[0].name === "净资产" && assets.bars[0].value === 3000,
  "net assets was not the first comparison bar");
check(
  h.formatSimulationDateTime({datetime: "2025-06-04T12:00:00"}) === "2025年6月4日 12:00",
  "simulation datetime was not localized",
);
check(
  h.formatSimulationDateTime({day: 4, hour: 12}) === "第 4 天 12 时",
  "simulation datetime fallback lost day/hour",
);
check(
  h.formatTime({day: 4, hour: 12, datetime: "2025-06-04T12:00:00"}) ===
    "第 4 天 12 时 · 2025年6月4日 12:00",
  "overview tick summary did not localize the simulation datetime",
);
const orderDetail = h.orderDetailHtml({
  order_id: "order-1",
  product_id: "product-1",
  sale_price: 20,
  purchase_price: 8,
  total_penalty: 3,
  net_profit: 9,
  profit_finalized: true,
  current_status: "settled_normal",
  status_log: [
    {status: "ordered", time: {day: 1, hour: 2}},
    {status: "shipped", time: {day: 1, hour: 6}},
    {status: "late", time: {day: 2, hour: 3}},
    {status: "settled_normal", time: {day: 3, hour: 4}},
  ],
});
check(orderDetail.includes("订单生命周期"), "order detail omitted lifecycle heading");
check(orderDetail.includes("已下单") && orderDetail.includes("已发货"),
  "order lifecycle statuses were not localized");
check(orderDetail.includes("超时") && orderDetail.includes("正常结算"),
  "order lifecycle risk/completion statuses were not localized");
check(orderDetail.includes("第 3 天 4 时"), "order lifecycle time was not formatted");
check(!orderDetail.includes("status_log"), "order detail leaked raw status JSON");
const lossRatio = h.priceRatioVisual(0.8);
check(lossRatio.label === "0.80×" && lossRatio.tone === "loss",
  "below-cost price ratio was not rendered as a loss");
const deepRatio = h.priceRatioVisual(2.5);
check(deepRatio.label === "2.50×" && deepRatio.tone === "strong",
  "high price ratio did not use the strongest green tier");
const salesTiers = [0, 1, 5, 25, 100].map(value => h.salesHeatTier(value, 100));
check(salesTiers[0] === 0, "zero-sales listing received a sales heat color");
check(salesTiers.at(-1) === 5, "top-selling listing did not receive the strongest heat color");
check(salesTiers.every((tier, index) => index === 0 || tier >= salesTiers[index - 1]),
  "sales heat tiers were not monotonic");

const capacityPlan = h.catalogListingPlan(
  [{product_id: "existing"}, {product_id: "new-1"}, {product_id: "new-2"}],
  new Set(["existing"]),
  1,
);
check(
  capacityPlan.items.map(row => row.product_id).join(",") === "existing,new-1",
  "listing plan did not preserve an existing listing and one free slot",
);
check(
  capacityPlan.blockedItems.map(row => row.product_id).join(",") === "new-2",
  "listing plan did not retain the over-capacity item",
);
const fullPlan = h.catalogListingPlan(
  [{product_id: "new-only"}],
  new Set(["existing"]),
  0,
);
check(fullPlan.items.length === 0, "full shelf still submitted a new listing");

const partialSummary = h.mutationSummary({
  ok: false,
  items: {
    columns: ["product_id", "ok", "error"],
    rows: [
      ["p-1", true, null],
      ["p-2", false, "max_active_listings=50 reached"],
    ],
  },
});
check(partialSummary.successCount === 1, "partial success was lost");
check(partialSummary.failureCount === 1, "partial failure was lost");
check(
  partialSummary.message.includes("货架上限 50") && partialSummary.message.includes("1 成功"),
  "capacity failure did not get a specific Chinese explanation",
);

const timeoutRisk = h.currentRiskDetails({
  supplier_listed: true,
  sale_price: 46,
  supplier_price: 23,
  supplier_ship_hours: 104,
  supplier_logistics_hours: 72,
}, 48);
check(timeoutRisk.includes("当前 104h"), "timeout risk omitted the current value");
check(timeoutRisk.includes("承诺 48h"), "timeout risk omitted the rule threshold");
check(timeoutRisk.includes("超出 56h"), "timeout risk omitted the delta");
const priceRisk = h.currentRiskDetails({
  supplier_listed: true,
  sale_price: 40,
  supplier_price: 46,
  supplier_ship_hours: 10,
}, 48);
check(priceRisk.includes("供货价 ¥46.00 > 售价 ¥40.00"), "price risk omitted values");
check(priceRisk.includes("倒挂 ¥6.00"), "price risk omitted the loss delta");
const timeoutEvent = h.eventRiskDetails({
  event_type: "supplier_timeout",
  before: {supplier_ship_hours: 12},
  after: {supplier_ship_hours: 60},
});
check(timeoutEvent.includes("12h → 60h"), "event omitted before/after values");
check(timeoutEvent.includes("增加 48h"), "event omitted the numeric delta");

h.state.activeListingIds = new Set(["p1", "p2"]);
h.state.activeListingIdsReady = true;
h.syncActiveListingIdsFromMutation("list_product", {
  items: {columns: ["product_id", "ok"], rows: [["p3", true], ["p4", false]]},
});
check(h.state.activeListingIds.has("p3"), "successful listing did not update authoritative IDs");
check(!h.state.activeListingIds.has("p4"), "failed listing polluted authoritative IDs");
h.syncActiveListingIdsFromMutation("delist_product", {
  items: {columns: ["product_id", "ok"], rows: [["p2", true]]},
});
check(!h.state.activeListingIds.has("p2"), "successful delist did not update authoritative IDs");
check(!h.attentionProductIsActive("p2"), "resolved risk product remained eligible for attention");
check(h.attentionProductIsActive("p1"), "active listing disappeared from attention eligibility");
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_human_playground_template_is_a_compact_pixel_workbench():
    page = Path("env/web/templates/human_playground.html").read_text(encoding="utf-8")
    script = Path("env/web/templates/_human_playground_script.html").read_text(
        encoding="utf-8"
    )
    style = Path("env/web/templates/_human_playground_style.html").read_text(
        encoding="utf-8"
    )
    html = page + script + style

    assert "const PLAYGROUND_CONFIG = {{ playground_config|tojson }};" in page
    assert "playground_config_json|safe" not in page
    for label in ("经营总览", "本轮概览", "商品与货架", "订单", "经营统计", "市场"):
        assert label in page
    assert page.count('class="nav-item') == 4
    assert 'data-scroll-panel="analytics"' in page
    assert 'data-workspace-group="analytics"' not in page
    assert page.count('data-workspace-group="operations"') == 4
    assert page.index('data-view-panel="market"') < page.index(
        'data-view-panel="overview"'
    )
    assert page.index('data-view-panel="overview"') < page.index(
        'data-view-panel="products"'
    )
    assert page.index('data-view-panel="products"') < page.index(
        'data-view-panel="orders"'
    )
    assert page.index('data-view-panel="orders"') < page.index(
        'data-view-panel="analytics"'
    )
    assert '{% include "_pixel_theme.html" %}' in page
    assert "平台经营红线" in page
    assert "先看红线，再做经营决策" in page
    assert 'id="platform-rules-summary"' in page
    assert 'id="platform-refund-summary"' in page
    assert 'id="platform-penalties-summary"' in page
    assert 'id="ch-hp-asset-composition"' not in page
    assert 'id="ch-hp-asset-distribution"' in page
    for chart_id in (
        "ch-hp-active-listings",
        "ch-hp-shop-rating",
        "ch-hp-order-total",
        "ch-hp-assets",
        "ch-hp-listings",
        "ch-hp-pnl",
        "ch-hp-rating",
        "ch-hp-products",
        "ch-hp-rank",
    ):
        assert f'id="{chart_id}"' in page
    assert 'id="overview-kpis"' not in page
    for memory_id in (
        "memory-doc-input",
        "memory-read-btn",
        "memory-save-btn",
        "memory-status",
    ):
        assert f'id="{memory_id}"' in page
    assert page.index('id="memory-doc-input"') < page.index("ANALYTICS")
    assert 'id="price-multiplier"' in page
    assert 'id="listing-price-multiplier"' in page
    assert 'id="apply-listing-price-multiplier"' in page
    assert page.index('class="app-shell"') < page.index('class="global-rules-wrap"')
    assert page.index('class="side-nav"') < page.index('class="global-rules-wrap"')
    assert 'id="listing-search-form"' in page
    assert 'class="sales-heat-legend"' in page
    assert 'data-listing-sort="supplier_price"' in page
    assert 'data-listing-sort="price_ratio"' in page
    assert "售价 ÷ 供货价" in page
    assert 'data-listing-sort="supplier_ship_hours"' in page
    assert 'value="1.5"' in page
    assert 'id="total-elapsed"' in page
    assert 'id="run-datetime"' in page
    assert 'id="run-transition-overlay"' in page
    assert 'id="transition-simulation-time"' in page
    assert 'id="transition-net-assets"' in page
    assert 'id="transition-total-cash"' in page
    assert 'id="delist-no-sales-btn"' in page
    for control_id in (
        "catalog-prev",
        "catalog-next",
        "order-history-prev",
        "order-history-next",
    ):
        assert f'id="{control_id}"' in page
    assert 'id="advanced-drawer"' in page
    assert "原始工具" in page
    assert 'id="trace-view"' in page
    assert "--nav-width: 154px" in style
    assert ".app-shell" in style
    assert ".app-content { min-width: 0; }" in style
    assert ".global-rules-wrap { margin-left:" not in style
    assert ".tool-card" not in html
    assert "font-family: ui-monospace" in style
    assert "border-radius: 0" in style
    assert "var(--pixel-shadow" in style
    assert ".rule-detail-block" in style
    assert ".penalty-strip" in style and "font-size: 13px" in style
    assert ".overview-metric-grid" in style
    assert ".order-lifecycle" in style
    assert ".order-detail-hero" in style
    assert ".public-detail-hero" in style
    assert ".table-detail-link" in style
    assert ".price-ratio-meter" in style
    assert ".price-ratio-meter.loss" in style
    assert ".price-ratio-meter.strong" in style
    assert "#listings-table tbody tr.sales-heat-1" in style
    assert "#listings-table tbody tr.sales-heat-5" in style
    assert ".run-transition-overlay" in style
    assert ".run-transition-overlay.waiting" in style
    assert ".run-transition-overlay.ready" in style
    assert ".run-transition-overlay.finishing" in style
    assert "@keyframes round-finishing-card" in style
    assert ".report-content" in style and "font-size: 12px" in style
    for status in (
        "cancelled",
        "settled_refund",
        "settled_only_refund",
        "settled_bad_review",
        "settled_normal",
    ):
        assert f".status-tag.{status}" in style
    assert "analytics-active" not in style
    assert ".chart.chart-overview-asset { height: 245px; }" in style
    assert ".chart.chart-overview-metric { height: 225px; }" in style
    assert ".nav-item { width: auto; min-width: 120px; flex: 1 0 120px;" in style
    assert 'classList.toggle("analytics-active", showAnalytics)' not in script

    assert 'content: options.auto ? "[human:auto] refresh operational state"' in script
    assert "const AUTO_TOOLS" in script
    for tool_name in (
        "get_store_snapshot",
        "query_order_updates",
        "query_open_orders",
        "query_supply_chain_anomalies",
        "read_memory_doc",
        "write_memory_doc",
    ):
        assert tool_name in script
    for tool_name in (
        "query_product_sales_trend",
        "query_my_listings",
        "query_store_performance",
        "query_platform_rules",
        "review_my_listings",
    ):
        assert tool_name not in script
    assert "refreshDashboardData" in script
    assert "fromDashboardPayload" in script
    assert "auto: automatic" in script
    assert "runAutoRefresh({automatic: false})" in script
    assert "function canUseReadTurn" in script
    assert "if (state.calling)" in script
    assert 'error: "client_busy"' in script
    assert "body._clientToolResults = clientToolResults" in script
    assert 'dataFreshness: "empty"' in script
    assert "datasetFreshness" in script
    assert "datasetIsFresh" in script
    assert "finalizeLoadingDatasets" in script
    assert "preserveMutationInputsForRetry" in script
    assert "operationalDataStep: null" in script
    assert "function invalidateOperationalData" in script
    assert "function syncAnalyticsControls" in script
    assert 'rankMetric: "orders"' in script
    assert "updatesMeta" in script
    assert 'return "—"' in script
    assert "observation.turn_count" in script
    assert "body.turn_idx" in script
    assert "return isNewStep" in script
    assert "if (isNewStep) {" in script
    assert "await runAutoRefresh();" in script
    assert "showRoundReadyOverlay();" in script
    assert "isNewStep && state.turnCount === 0" not in script
    assert "remainingTurns() > 1" in script
    assert "Number(observation.tick?.step)" in script
    assert "response.status === 425" in script
    assert "response.status === 408" in script
    assert "response.status === 410" in script
    assert "pauseCurrentHookUi" in script
    assert "startCountdown(remaining)" in script

    for tool_name, argument_name in (
        ("list_product", "sale_price"),
        ("adjust_price", "new_price"),
        ("delist_product", "product_id"),
    ):
        if tool_name == "list_product":
            assert '"list_product", {items: plan.items}' in script
        else:
            assert f'"{tool_name}", {{items}}' in script
        assert argument_name in script
    assert 'listingSortBy: "procured_orders"' in script
    assert 'listingSortOrder: "desc"' in script
    assert 'currentOrderTab: "open"' in script
    assert 'class="active" data-order-tab="open"' in page
    assert '<div id="orders-caption" class="table-caption">进行中</div>' in page
    assert "applyListingView(next)" in script
    assert "不消耗 turn" in page
    assert "state.catalogSelection" in script
    assert "state.listingSelection" in script
    assert "priceMultiplier: 1.5" in script
    assert "listingPriceMultiplier: 1.5" in script
    assert "state.priceMultiplier" in script
    assert "function applyListingPriceMultiplier" in script
    assert "supplierPrice * state.listingPriceMultiplier" in script
    assert "next < 0.1" in script
    assert 'dialog.returnValue = ""' in script
    assert "apply-price-multiplier" in page
    assert "state.platformRules" in script
    assert "status.elapsed_ms" in script
    assert "formatElapsed" in script
    assert "catalogMeta" in script
    assert "historyOrderMeta" in script
    assert "function searchCatalog" in script
    assert script.count('data-catalog-detail-tool="get_product_detail"') == 2
    assert script.count('data-catalog-detail-tool="get_supplier_profile"') == 2
    listings_renderer = script[
        script.index("function renderListings"):
        script.index("function renderCatalog")
    ]
    assert "查看商品详情" in listings_renderer
    assert "查看供应商详情" in listings_renderer
    assert "function catalogDetailToolCall" in script
    assert "return callTools([call]" in script
    assert "function publicDetailHtml" in script
    assert 'get_product_detail: ["PRODUCT DETAIL", "商品详情"]' in script
    assert 'get_supplier_profile: ["SUPPLIER DETAIL", "供应商详情"]' in script
    assert 'catalogMeta: {page: 1, page_size: 20, has_next: false}' in script
    assert '$("catalog-page-size").addEventListener("change", () => searchCatalog(1));' in script
    assert "function refreshListings" in script
    assert "function listingQueryArgs" not in script
    assert "function loadOrderHistory" in script
    assert "function successfulMutationIds" in script
    assert "function syncActiveListingIdsFromMutation" in script
    assert "activeListingIdsReady" in script
    assert "function retainFailedSelection" in script
    assert "function changeProductFocus" in script
    assert "function selectProductFocus" in script
    assert "function minimumStepWaitSeconds" in script
    assert "function showRoundWaitingOverlay" in script
    assert "function showRoundEndingOverlay" in script
    assert "function showRoundReadyOverlay" in script
    assert "const MIN_STEP_DURATION_MS = 15000" in script
    assert "showRoundEndingOverlay();" in script
    assert "function staleListingCandidates" in script
    assert "function reviewAndConfirmNoSalesDelist" in script
    assert "function openOrderStatusCounts" in script
    assert "function currentRiskDetails" in script
    assert "function eventRiskDetails" in script
    assert "function dedupeAttention" in script
    assert "const attention = dedupeAttention([" in script
    assert "function confirmRiskDelist" in script
    assert 'data-attention-action="delist"' in script
    assert script.count("canDelist: true") >= 2
    assert "function overviewAssetData" in script
    assert "function renderOverviewAssetCharts" in script
    assert "function renderOverviewMetricCharts" in script
    assert "function formatSimulationDateTime" in script
    assert '$("run-datetime").textContent = formatSimulationDateTime' in script
    assert 'id="shop-rating-badge"' not in page
    assert "function renderShopRatingBadge" not in script
    assert "function ratingKpi" not in script
    assert "renderOverviewMetricCharts(snapshot)" in script
    assert "button.disabled = !hasWritableTurn" in script
    assert "function resizeVisibleCharts" in script
    assert "function afterWorkspaceLayout" in script
    assert "element.getClientRects().length === 0" in script
    assert 'window.addEventListener("resize", resizeVisibleCharts)' in script
    listings_render = script.split("const listingsOption = builders.listings", 1)[1].split(
        "const pnlOption", 1
    )[0]
    assert 'mode: "active"' in listings_render
    assert 'labels: {active: "活跃商品数"}' in listings_render
    assert "list:" not in listings_render
    assert "delist:" not in listings_render
    assert "price:" not in listings_render
    assert "活跃商品数，与 Merchant Dashboard 保持一致" in page
    assert 'kpi("净资产"' not in script
    assert 'kpi("可用余额"' not in script
    assert 'kpi("在途 / 应收"' not in script
    assert "function waitForPendingCall" in script
    assert "function cancelPendingConfirm" in script
    assert "function orderTabToolCall" in script
    assert "function refreshOrderTab" in script
    order_tab_handler = script.split(
        'qsa("[data-order-tab]").forEach(button => button.addEventListener("click",',
        1,
    )[1].split("}));", 1)[0]
    assert "refreshOrderTab(state.currentOrderTab)" in order_tab_handler
    assert "function enterTerminalState" in script
    assert 'dailyChart?.off("click")' in script
    assert 'rankChart?.off("click")' in script
    assert "datasetIsFresh(\"productTrendFocus\")" not in script
    assert "staleListingCandidates(state.allListings, 7)" in script
    assert "Number(salePrice || 0) - Number(purchasePrice || 0)" not in script
    assert "row.final_net_profit ?? row.realized_net_profit" not in script
    assert "row.order_id ?? row.id" not in script
    assert "formatRuleThreshold" not in script
    assert "履约保证金归零" in script
    assert "所有罚款先扣余额" in script

    init_block = script.split("async function init()", 1)[1].split(
        "\n  init();", 1
    )[0]
    assert init_block.index("await refreshStatus()") < init_block.index(
        "await activateRuntimeSession()"
    )
    assert 'if (state.runState !== "paused") await activateRuntimeSession();' in init_block
    assert "async function activateRuntimeSession" in script
    assert "if (!state.toolsLoaded) await fetchTools();" in script
    assert "await registerHuman();" in script
    assert "void observeLoop().finally" in script
    assert "if (state.finished) {" in init_block
    assert "await hydrateTerminalDashboard();" in init_block
    assert "await refreshStatus()" in init_block.split("catch (error)", 1)[1]

    auto_refresh = script.split(
        "async function runAutoRefresh", 1
    )[1].split("const PENALTY_LABELS", 1)[0]
    assert "query_product_sales_trend" not in auto_refresh
    assert "query_store_performance" not in auto_refresh
    assert "await refreshDashboardData()" in auto_refresh
    assert "markOperationalDataFresh" in auto_refresh

    refresh_analytics = script.split(
        "async function refreshAnalytics", 1
    )[1].split("async function loadTraceIndex", 1)[0]
    assert "next.range ?? state.analyticsRange" in refresh_analytics
    assert "refreshDashboardData(candidate)" in refresh_analytics
    assert "state.analyticsRange = candidate.range" in refresh_analytics
    assert "syncAnalyticsControls()" in refresh_analytics

    assert (
        'qsa("[data-range]").forEach(item => item.classList.toggle("active", item === button));'
        not in script
    )

    for status in (
        "settled_normal",
        "settled_refund",
        "settled_only_refund",
        "settled_bad_review",
    ):
        assert f'<option value="{status}">' in page
    for invalid_status in ("settled", "refunded", "bad_review"):
        assert f'<option value="{invalid_status}">' not in page

    forbidden_paths = ("/sections/merchant", "/sections/orders", "/product_diagnostics")
    for path in forbidden_paths:
        assert path not in html
    for path in ("/agent/all_traces", "/agent/all_traces_index", "/agent/trace"):
        assert path not in html
    for hidden_field in (
        "elasticity",
        "market_curve",
        "future_demand",
        "supplier_delist_rate",
    ):
        assert hidden_field not in html


def test_human_playground_tables_and_charts_have_explicit_safe_models():
    page = Path("env/web/templates/human_playground.html").read_text(encoding="utf-8")
    script = Path("env/web/templates/_human_playground_script.html").read_text(
        encoding="utf-8"
    )
    shared = Path("env/web/templates/_merchant_analytics_charts.html").read_text(
        encoding="utf-8"
    )

    assert 'id="listings-table"' in page
    assert 'id="catalog-table"' in page
    assert 'id="orders-table"' in page
    assert 'class="order-product-name"' in script
    assert "function tableToRecords(value)" in script
    assert "Object.keys(rows[0]).slice" not in script
    for chart_id in (
        "ch-hp-assets",
        "ch-hp-listings",
        "ch-hp-pnl",
        "ch-hp-products",
        "ch-hp-rank",
        "ch-hp-rating",
    ):
        assert f'id="{chart_id}"' in page
    assert "MerchantBenchMerchantCharts" in script
    assert "fromDashboardPayload" in shared
    assert "fromToolResults" not in shared
    assert "productTotals" in shared
    assert "optionBuilders" in shared
    for builder in (
        "buildAssetsOption",
        "buildListingsOption",
        "buildPnlOption",
        "buildRatingOption",
        "buildProductTrendOption",
        "buildProductDailyRankedOption",
        "buildProductRankOption",
    ):
        assert builder in shared
    assert "function ratingAxisBounds(data)" in shared
    assert "yAxis: ratingAxisBounds(model?.rating)" in shared
    assert "...ratingOption.yAxis" in script
    assert "min: 1, max: 5, interval: 1" not in script
    for call in (
        "builders.assets",
        "builders.listings",
        "builders.pnl",
        "builders.rating",
        "builders.productDailyRanked",
        "builders.productRank",
    ):
        assert call in script

    dashboard = Path("env/web/templates/dashboard.html").read_text(
        encoding="utf-8"
    )
    for call in (
        "builders?.assets",
        "builders?.listings",
        "builders?.pnl",
        "optionBuilders?.productRank",
    ):
        assert call in dashboard


def test_shared_merchant_chart_option_builders_run_for_dashboard_payload():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for shared chart builder validation")
    source = Path(
        "env/web/templates/_merchant_analytics_charts.html"
    ).read_text(encoding="utf-8")
    source = source.strip().removeprefix("<script>\n").removesuffix("\n</script>")
    harness = r"""
globalThis.window = {};
eval(require("fs").readFileSync(0, "utf8"));
const api = window.MerchantBenchMerchantCharts;
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const dashboardModel = api.fromDashboardPayload({
  series: {
    balance: [[0, 100]], deposit_pool: [[0, 50]],
    in_transit: [[0, 10]], receivable: [[0, 5]], net_assets: [[0, 165]],
    n_active_listings: [[0, 1]], cum_gmv: [[0, 20]], cum_cost: [[0, 10]],
    cum_gross_profit: [[0, 10]], cum_net_profit: [[0, 8]], cum_fine: [[0, 2]],
    shop_rating_mean: [[0, 4]],
  },
  listing_ops: {series: {ops: [[0, 1]], list: [[0, 1]], delist: [], price: []}},
  daily_sales_by_product: {
    buckets: [{key: "D1", label: "D1", start_day: 1, end_day: 1}],
    series: [{
      product_id: "p1", name: "one", category: "office",
      data: [{orders: 1, gmv: 20, gross_profit: 10, net_profit: 8,
        supply_chain_anomalies: 2, order_anomalies: 1}],
    }],
  },
});
const assets = api.optionBuilders.assets(dashboardModel);
check(assets.series.length === 5, "asset option did not include five series");
const listings = api.optionBuilders.listings(dashboardModel);
check(listings.series.map(row => row.type).join(",") === "line,bar,bar,bar", "listing presentation drifted");
const products = api.optionBuilders.productTrend(dashboardModel, {selectedProductId: "p1"});
check(products.series.length === 9, "focused product option omitted a metric");
const rank = api.optionBuilders.productRank(dashboardModel, "orders", {reverse: false});
check(rank.totals[0].product_id === "p1" && rank.totals[0].value === 1, "product rank option is wrong");
const signedRank = api.optionBuilders.productRank({products: [
  {product_id: "positive", name: "positive", net_profit: [10]},
  {product_id: "zero", name: "zero", net_profit: [0]},
  {product_id: "negative", name: "negative", net_profit: [-5]},
]}, "net_profit", {reverse: false, limit: 2});
check(signedRank.totals.map(row => row.product_id).join(",") === "positive,negative",
  "rank did not filter zero values before applying its limit");
const dashboardAssets = api.optionBuilders.assets(
  dashboardModel,
  {keepPoints: true},
);
check(
  JSON.stringify(dashboardAssets.series[0].data) === JSON.stringify([[0, 100]]),
  "dashboard points were not preserved",
);
const dashboardDaily = api.optionBuilders.productDailyRanked(
  dashboardModel,
  "orders",
  {itemStyleForSegment: segment => ({opacity: .4, color: segment.color})},
);
check(dashboardDaily.series[0].data[0].start_day === 1,
  "dashboard bucket metadata was dropped by the shared daily builder");
check(dashboardDaily.series[0].data[0].supply_chain_anomalies === 2,
  "dashboard anomaly details were dropped by the shared daily builder");
check(dashboardDaily.series[0].data[0].itemStyle.opacity === .4,
  "dashboard daily item styling callback was ignored");
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_human_playground_state_transitions_preserve_stale_mutations_and_track_freshness():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser state-machine test")
    source = Path("env/web/templates/_human_playground_script.html").read_text(
        encoding="utf-8"
    )
    source = source.strip().removeprefix("<script>\n").removesuffix("\n</script>")
    marker = "  init();\n})();"
    assert marker in source
    source = source.replace(
        marker,
        """  globalThis.__hpTest = {
    callTools,
    catalogMutationItems,
    state,
    datasetIsFresh,
    finalizeLoadingDatasets,
    ingestResult,
    init,
    connectSSE,
    observeLoop,
    invalidateOperationalData,
    preserveMutationInputsForRetry,
    changeProductFocus,
    applyPriceMultiplier,
    applyListingPriceMultiplier,
    confirmAction,
    openOrderStatusCounts,
    orderTabToolCall,
    enterTerminalState,
    waitForPendingCall,
    refreshAnalytics,
    refreshDashboardData,
    runAutoRefresh,
    retainFailedSelection,
    dedupeAttention,
    catalogDetailToolCall,
	    canLoadCatalogDetail,
	    syncControls,
	    minimumStepWaitSeconds,
	    beginStepMinimumDuration,
	    syncFinishStepControl,
	    showRoundWaitingOverlay,
	    showRoundEndingOverlay,
	    showRoundReadyOverlay,
    updateRunState,
    staleListingCandidates,
    applyListingView,
    refreshListings,
  };
})();""",
        1,
    )
    harness = r"""
const fs = require("fs");
const source = fs.readFileSync(0, "utf8");
const nativeSetTimeout = globalThis.setTimeout;
const elements = new Map();
const selectorResults = new Map();
const classList = {toggle() {}, add() {}, remove() {}};
const elementFor = id => {
  if (!elements.has(id)) {
    const target = {
      classList,
      elements: new Proxy({}, {get: (_obj, key) => elementFor(`${id}.${String(key)}`)}),
      querySelector: selector => elementFor(`${id}.${selector}`),
      addEventListener() {},
      setAttribute() {},
      showModal() { this.open = true; },
      close(value) {
        if (value !== undefined) this.returnValue = value;
        this.open = false;
      },
    };
    elements.set(id, new Proxy(target, {
      get: (obj, key) => key in obj ? obj[key] : undefined,
    }));
  }
  return elements.get(id);
};
globalThis.document = {
  getElementById: id => elementFor(id),
  querySelector: selector => elementFor(selector),
  querySelectorAll: selector => selectorResults.get(selector) || [],
};
globalThis.window = {
  MerchantBenchMerchantCharts: null,
  addEventListener() {},
  location: {origin: "http://localhost"},
};
globalThis.PLAYGROUND_CONFIG = {
  actUrl: "/act",
  observationUrl: "/observation",
  statusUrl: "/status",
  toolsSchemaUrl: "/tools",
  dashboardDataUrl: "/dashboard-data",
  stepHours: 1,
  maxTurnsPerStep: 30,
  platformRules: {important_rules: {horizon_days: 365}},
};
globalThis.setTimeout = () => 0;
globalThis.clearTimeout = () => {};
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
eval(source);
	const h = globalThis.__hpTest;
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};

	(async () => {
	let fakeNow = 100000;
	Date.now = () => fakeNow;
	h.state.stepActive = true;
	h.state.finished = false;
	h.state.calling = false;
	h.state.turnCount = 0;
	h.state.toolByName = new Map([["end_of_step", {mutating: true}]]);
	h.beginStepMinimumDuration(true);
	check(h.minimumStepWaitSeconds() === 15, "new hook did not start the 15 second gate");
	h.syncFinishStepControl();
	check(elementFor("finish-step-btn").disabled, "finish button was enabled before 15 seconds");
	let earlyEndFetchCount = 0;
	globalThis.fetch = async () => {
	  earlyEndFetchCount += 1;
	  throw new Error("early end_of_step reached fetch");
	};
	const earlyEnd = await h.callTools([["end_of_step", {}]], {raw: true});
	check(earlyEnd.error === "minimum_step_duration",
	  "raw end_of_step bypassed the 15 second gate");
	check(earlyEndFetchCount === 0, "blocked end_of_step still reached fetch");
	fakeNow += 15000;
	h.syncFinishStepControl();
	check(h.minimumStepWaitSeconds() === 0, "15 second gate did not expire");
	check(!elementFor("finish-step-btn").disabled, "finish button stayed disabled after 15 seconds");

	h.showRoundEndingOverlay();
	check(elementFor("run-transition-overlay").className.includes("finishing"),
	  "successful round completion did not start the finishing animation");
	h.showRoundWaitingOverlay();
	check(elementFor("run-transition-overlay").className.includes("finishing"),
	  "waiting state interrupted the finishing animation");
	elementFor("run-transition-overlay").className = "run-transition-overlay";
	h.showRoundWaitingOverlay();
	check(!elementFor("run-transition-overlay").hidden, "waiting overlay was not shown");
	check(elementFor("run-transition-overlay").className.includes("waiting"),
	  "waiting overlay did not use the waiting state");
	h.state.countdownRemainingSeconds = 42;
	h.updateRunState("paused");
	check(elementFor("run-transition-overlay").hidden,
	  "paused run left the blocking waiting overlay visible");
	h.updateRunState("running");
	check(elementFor("run-transition-overlay").hidden,
	  "resuming a paused active hook showed a blocking waiting overlay");
	h.state.countdownRemainingSeconds = null;
	h.updateRunState("running");
	check(!elementFor("run-transition-overlay").hidden,
	  "running without an open hook did not restore the waiting overlay");
	h.state.stepActive = true;
	h.state.latestTick = {day: 3, hour: 12, datetime: "2025-06-03T12:00:00"};
	h.state.snapshot = {cash: {balance: 120, deposit_pool: 30, in_transit: 10, receivable: 5, net_assets: 165}};
	h.state.datasetFreshness.snapshot = "fresh";
	h.state.operationalDataStep = h.state.latestEnvT;
	h.showRoundReadyOverlay();
	check(elementFor("transition-net-assets").textContent.includes("165.00"),
	  "ready overlay did not show net assets");
	check(elementFor("transition-total-cash").textContent.includes("150.00"),
	  "ready overlay did not show balance plus deposit cash");

	const staleRows = h.staleListingCandidates([
	  {product_id: "stale", days_without_sales: 7, last_sale_day: 0},
	  {product_id: "recent", days_without_sales: 6, last_sale_day: 0},
	  {product_id: "failed-only", days_without_sales: 9, last_sale_day: 0},
	  {product_id: "sold-before", days_without_sales: 8, last_sale_day: 4},
	]);
check(JSON.stringify(staleRows.map(row => row.product_id)) === JSON.stringify(["failed-only", "stale"]),
  "seven-day never-sold candidates were filtered or sorted incorrectly");

h.state.allListings = [
  {product_id: "p-low", name: "冬季热水袋", category: "home", supplier_name: "A", procured_orders: 2},
  {product_id: "p-high", name: "夏季风扇", category: "appliances", supplier_name: "B", procured_orders: 12},
];
h.state.listingQuery = "";
h.state.listingSortBy = "procured_orders";
h.state.listingSortOrder = "desc";
h.applyListingView();
check(h.state.listings.map(row => row.product_id).join(",") === "p-high,p-low",
  "local shelf sort did not use cumulative sales descending");
h.refreshListings({query: "热水袋"});
check(h.state.listings.map(row => row.product_id).join(",") === "p-low",
  "local shelf search did not filter public listing fields");

	const memoryInput = elementFor("memory-doc-input");
	memoryInput.value = "new edit while saving";
	h.state.memoryDirty = true;
	h.ingestResult("write_memory_doc", {ok: true, bytes: 12}, {content: "submitted version"});
	check(memoryInput.value === "new edit while saving", "memory save response overwrote a newer edit");
	check(h.state.memoryDirty, "newer memory edit was incorrectly marked clean");
	check(h.state.memoryContent === "submitted version", "saved memory baseline was not updated");
	h.state.stepActive = false;

	check(
  JSON.stringify(h.catalogDetailToolCall("get_product_detail", "p-1", "s-1"))
    === JSON.stringify(["get_product_detail", {product_id: "p-1"}]),
  "product detail did not map to the public tool contract",
);
check(
  JSON.stringify(h.catalogDetailToolCall("get_supplier_profile", "p-1", "s-1"))
    === JSON.stringify(["get_supplier_profile", {supplier_id: "s-1"}]),
  "supplier detail did not map to the public tool contract",
);
check(
  h.catalogDetailToolCall("get_supplier_profile", "p-1", "") === null,
  "supplier detail accepted a missing supplier id",
);
check(
  h.catalogDetailToolCall("get_product_detail", null, "s-1") === null,
  "product detail accepted a null product id",
);
check(
  h.catalogDetailToolCall("get_product_detail", "   ", "s-1") === null,
  "product detail accepted a blank product id",
);
h.state.toolByName = new Map([
  ["get_product_detail", {mutating: false}],
  ["get_supplier_profile", {mutating: false}],
]);
check(!h.canLoadCatalogDetail("get_product_detail", "p-1"),
  "product detail was enabled outside an active hook");
h.state.stepActive = true;
h.state.finished = false;
h.state.turnCount = 0;
h.state.calling = false;
check(h.canLoadCatalogDetail("get_product_detail", "p-1"),
  "product detail stayed disabled in an active hook");
check(!h.canLoadCatalogDetail("get_supplier_profile", ""),
  "supplier detail was enabled without a supplier id");

const attention = h.dedupeAttention([
  {productId: "p-1", text: "供应商状态：可供货 → 已下架", canDelist: true},
  {productId: "p-1", text: "供应商状态：可供货 → 已下架", canDelist: true},
  {productId: "p-1", text: "供货价发生变化", canDelist: true},
  {title: "订单 o-1", text: "当前状态：超时", canDelist: false},
  {title: "订单 o-2", text: "当前状态：超时", canDelist: false},
]);
check(attention.length === 4, "duplicate attention entries were not merged safely");
check(attention.some(item => item.text === "供货价发生变化"), "distinct product risks were collapsed");
check(attention.filter(item => item.title?.startsWith("订单 ")).length === 2,
  "different late orders were collapsed together");

h.state.calling = true;
let fetchCount = 0;
globalThis.fetch = async () => {
  fetchCount += 1;
  throw new Error("unexpected fetch");
};
const busy = await h.callTools([["query_my_listings", {}]]);
check(busy.error === "client_busy", "concurrent call did not return client_busy");
check(fetchCount === 0, "concurrent call reached fetch");
h.state.calling = false;

const riskButton = {disabled: true};
selectorResults.set('[data-attention-action="delist"]', [riskButton]);
h.state.stepActive = true;
h.state.finished = false;
h.state.turnCount = 0;
h.state.toolByName = new Map([["delist_product", {mutating: true}]]);
h.syncControls();
check(riskButton.disabled === false, "risk action stayed disabled after a tool call settled");
h.state.calling = true;
h.syncControls();
check(riskButton.disabled === true, "risk action stayed enabled during a tool call");
h.state.calling = false;
h.syncControls();
check(riskButton.disabled === false, "risk action was not re-enabled after a tool call");

const silentTimeout = globalThis.setTimeout;
globalThis.setTimeout = callback => nativeSetTimeout(callback, 0);
h.state.calling = true;
let pendingCallReleased = false;
const pendingCallWait = h.waitForPendingCall().then(() => {
  pendingCallReleased = true;
});
await new Promise(resolve => nativeSetTimeout(resolve, 0));
check(!pendingCallReleased, "new observation was not held behind the pending call");
h.state.calling = false;
await pendingCallWait;
check(pendingCallReleased, "new observation did not resume after the call settled");
globalThis.setTimeout = silentTimeout;

h.state.openOrderMeta = null;
h.state.snapshot = {
  orders: {
    totals: {
      total: 9,
      ordered: 2,
      late: 1,
      shipped: 1,
      delivered: 1,
      settled_normal: 4,
    },
  },
};
const openStatusFallback = h.openOrderStatusCounts();
check(openStatusFallback.ordered === 2, "snapshot open-order fallback lost active orders");
check(openStatusFallback.total === undefined, "snapshot total leaked into open-order chart");
check(
  openStatusFallback.settled_normal === undefined,
  "terminal status leaked into open-order chart",
);

h.state.latestEnvT = 12;
h.state.listingSelection = new Set(["p-1", "p-2"]);
h.state.catalogSelection = new Set(["p-3"]);
h.state.listingDrafts = new Map([["p-1", "33"]]);
h.state.catalog = [{product_id: "p-3", price: 10}];
h.state.catalogDrafts = new Map();
h.state.datasetFreshness.catalog = "fresh";
h.state.allListings = [
  {product_id: "p-1", supplier_price: 10, sale_price: 12},
  {product_id: "p-2", supplier_price: 20, sale_price: 24},
];
h.state.listings = [...h.state.allListings];
h.state.datasetFreshness.listings = "fresh";
const listingMultiplierInput = elementFor("listing-price-multiplier");
listingMultiplierInput.value = "1.5";
h.applyListingPriceMultiplier();
check(h.state.listingPriceMultiplier === 1.5, "listing multiplier was not saved");
check(h.state.listingDrafts.get("p-1") === "15.00", "first listing multiplier price is wrong");
check(h.state.listingDrafts.get("p-2") === "30.00", "second listing multiplier price is wrong");
const listingDraftBeforeStale = h.state.listingDrafts.get("p-1");
listingMultiplierInput.value = "0.05";
h.applyListingPriceMultiplier();
check(listingMultiplierInput.value === "1.5", "invalid listing multiplier was not restored");
h.state.priceMultiplier = 2;
const multiplierInput = elementFor("price-multiplier");
multiplierInput.value = "0.05";
h.applyPriceMultiplier();
check(h.state.priceMultiplier === 2, "multiplier below 0.1 was accepted");
check(multiplierInput.value === "2", "invalid multiplier input was not restored");
check(h.state.catalogDrafts.size === 0, "invalid multiplier changed draft prices");
multiplierInput.value = "2";
h.applyPriceMultiplier();

const confirmDialog = elementFor("confirm-dialog");
confirmDialog.returnValue = "default";
h.confirmAction("confirm", "summary", () => {});
check(confirmDialog.returnValue === "", "reopened confirmation kept the previous result");
confirmDialog.close();
check(confirmDialog.returnValue === "", "cancel-like close looked like confirmation");
h.state.pendingConfirm = null;

const initialCatalogItems = h.catalogMutationItems();
check(initialCatalogItems[0].sale_price === 20, "default suggested price was not submitted");
check(h.state.catalogDrafts.get("p-3") === "20", "submitted default price was not persisted");
h.state.stepActive = true;
h.state.finished = false;
h.state.turnCount = 0;
h.state.toolByName = new Map([["list_product", {mutating: true}]]);
globalThis.fetch = async () => ({
  ok: false,
  status: 425,
  json: async () => ({ok: false, error: "stale_step"}),
});
const stale = await h.callTools([
  ["list_product", {items: initialCatalogItems}],
]);
check(stale.error === "stale_step", "425 response was not returned");
check(h.state.preserveMutationInputsOnNextStep, "425 mutation was not marked for retry");
h.invalidateOperationalData();
check(h.state.listingSelection.size === 2, "stale listing selection was cleared");
check(h.state.catalogSelection.has("p-3"), "stale catalog selection was cleared");
check(h.state.listingDrafts.get("p-1") === listingDraftBeforeStale, "stale listing price draft was cleared");
check(h.state.catalogDrafts.get("p-3") === "20", "stale price draft was cleared");
check(h.catalogMutationItems()[0].sale_price === 20, "retry changed the submitted default price");

h.ingestResult("get_store_snapshot", {cash: {balance: 100}});
h.finalizeLoadingDatasets([
  ["get_store_snapshot", {}],
  ["query_order_updates", {}],
]);
check(h.datasetIsFresh("snapshot"), "snapshot result was not marked fresh");
check(h.state.datasetFreshness.listings === "loading", "Dashboard listings were finalized as a missing tool result");
check(h.state.datasetFreshness.updates === "unavailable", "missing updates looked fresh");

const partial = {
  _clientToolResults: [{
    name: "adjust_price",
    payload: {
      ok: false,
      items: {
        columns: ["product_id", "ok", "error"],
        rows: [["p-1", true, null], ["p-2", false, "rejected"]],
      },
    },
  }],
};
const retained = h.retainFailedSelection(
  new Set(["p-1", "p-2"]), partial, "adjust_price",
);
check(!retained.has("p-1") && retained.has("p-2"), "partial failure retry set is wrong");

let staleConfirmRan = false;
h.state.pendingConfirm = () => { staleConfirmRan = true; };
h.invalidateOperationalData();
check(h.state.pendingConfirm === null, "normal next step kept a stale confirmation");
check(!staleConfirmRan, "normal next step executed a stale confirmation");
check(h.state.listingSelection.size === 0, "non-stale next step kept old selection");
check(h.state.catalogSelection.size === 0, "non-stale next step kept old catalog selection");
check(h.state.listingDrafts.size === 0, "non-stale next step kept old listing drafts");

h.state.calling = true;
h.state.selectedProductId = null;
check(h.changeProductFocus("p-race") === false, "focus changed during an in-flight call");
check(h.state.selectedProductId === null, "blocked focus change mutated selection");
h.state.calling = false;

const oldMerchantData = {t: 1, listings: [], series: {}, daily_sales_by_product: {series: []}};
h.state.merchantData = oldMerchantData;
h.state.datasetFreshness.performance = "fresh";
h.state.datasetFreshness.productTrend = "fresh";
h.state.analyticsRange = "30";
h.state.analyticsLevel = "day";
h.state.rankMetric = "orders";
h.state.stepActive = true;
h.state.turnCount = 0;
const deferredDashboard = [];
globalThis.fetch = url => new Promise(resolve => {
  deferredDashboard.push({url: String(url), resolve});
});
const olderRefresh = h.refreshAnalytics({range: "7", level: "day", sortBy: "orders"});
const newerRefresh = h.refreshAnalytics({range: "all", level: "week", sortBy: "gmv"});
check(deferredDashboard.length === 2, "Dashboard refreshes were not started concurrently");
deferredDashboard[1].resolve({
  ok: true,
  status: 200,
  json: async () => ({t: 2, listings: [], series: {}, daily_sales_by_product: {series: []}}),
});
const newerResult = await newerRefresh;
deferredDashboard[0].resolve({
  ok: true,
  status: 200,
  json: async () => ({t: 1, listings: [], series: {}, daily_sales_by_product: {series: []}}),
});
const olderResult = await olderRefresh;
check(newerResult?.t === 2, "latest Dashboard response was not committed");
check(olderResult === null, "stale Dashboard response was committed");
check(h.state.merchantData?.t === 2, "stale Dashboard response overwrote current data");
check(
  h.state.analyticsRange === "all" && h.state.analyticsLevel === "week"
    && h.state.rankMetric === "gmv",
  "stale Dashboard response overwrote the latest controls",
);

h.state.stepActive = true;
h.state.finished = false;
h.state.calling = false;
h.state.turnCount = 0;
h.state.toolByName = new Map([
  ["get_store_snapshot", {}],
  ["query_order_updates", {}],
  ["query_open_orders", {}],
  ["query_supply_chain_anomalies", {}],
  ["read_memory_doc", {}],
]);
const automaticRaceRequests = [];
globalThis.fetch = async (url, options = {}) => {
  if (url === "/act") {
    const request = JSON.parse(options.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        ok: true,
        turn_idx: 0,
        tool_results: request.messages[0].tool_calls.map(call => ({
          tool_call_id: call.id,
          name: call.function.name,
          content: JSON.stringify(
            call.function.name === "get_store_snapshot" ? {cash: {balance: 100}} : {},
          ),
        })),
      }),
    };
  }
  return new Promise(resolve => automaticRaceRequests.push({url: String(url), resolve}));
};
const automaticRefresh = h.runAutoRefresh();
for (let i = 0; i < 20 && automaticRaceRequests.length < 1; i += 1) {
  await Promise.resolve();
}
check(automaticRaceRequests.length === 1, "automatic Dashboard refresh did not start");
const userRefresh = h.refreshAnalytics({range: "30", level: "day", sortBy: "orders"});
check(automaticRaceRequests.length === 2, "user Dashboard refresh did not supersede automatic refresh");
automaticRaceRequests[1].resolve({
  ok: true,
  status: 200,
  json: async () => ({t: 3, listings: [], series: {}, daily_sales_by_product: {series: []}}),
});
await userRefresh;
automaticRaceRequests[0].resolve({
  ok: true,
  status: 200,
  json: async () => ({t: 2, listings: [], series: {}, daily_sales_by_product: {series: []}}),
});
await automaticRefresh;
check(h.state.merchantData?.t === 3, "automatic refresh overwrote newer user data");
check(elementFor("analytics-unavailable").hidden,
  "superseded automatic refresh displayed a false Dashboard failure");

h.state.analyticsRange = "30";
h.state.analyticsLevel = "day";
h.state.rankMetric = "orders";
oldMerchantData.t = 1;
h.state.merchantData = oldMerchantData;
globalThis.fetch = async () => ({ok: false, status: 503});
const analyticsOk = await h.refreshAnalytics({
  range: "7",
  level: "week",
  sortBy: "gmv",
});
check(analyticsOk === null, "failed Dashboard analytics refresh was committed");
check(
  h.state.analyticsRange === "30" && h.state.analyticsLevel === "day",
  "analytics filters did not roll back",
);
check(h.state.rankMetric === "orders", "rank metric did not roll back");
check(h.state.merchantData === oldMerchantData, "failed Dashboard refresh replaced the previous dataset");
check(
  h.state.datasetFreshness.performance === "unavailable"
    && h.state.datasetFreshness.productTrend === "unavailable",
  "failed Dashboard refresh was still marked fresh",
);

const terminalRequests = [];
h.state.finished = false;
h.state.runState = "loading";
globalThis.fetch = async url => {
  terminalRequests.push(url);
  if (url !== "/status") {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        t: 4,
        listings: [],
        series: {},
        listing_ops: {series: {}},
        daily_sales_by_product: {series: []},
      }),
    };
  }
  return {
    ok: true,
    status: 200,
    json: async () => ({state: "finished", elapsed_ms: 12345, t: 8760}),
  };
};
await h.init();
check(
  terminalRequests.length === 2
    && terminalRequests[0] === "/status"
    && String(terminalRequests[1]).startsWith("http://localhost/dashboard-data?"),
  "terminal init did not hydrate the safe Dashboard endpoint exactly once",
);
const terminalDashboardUrl = new URL(String(terminalRequests[1]));
check(
  terminalDashboardUrl.searchParams.get("t_from") === "8040"
    && terminalDashboardUrl.searchParams.get("t_to") === "8759",
  "terminal Dashboard range was not anchored to the completed 365-day run",
);
check(h.state.finished, "terminal init did not enter read-only mode");
check(h.state.merchantData?.t === 4, "terminal init did not hydrate Dashboard data");
check(
  h.state.elapsedMs >= 12345 && h.state.elapsedMs < 12500,
  "terminal elapsed time was not hydrated",
);

let liveSource = null;
globalThis.EventSource = class {
  constructor() {
    this.listeners = new Map();
    this.closed = false;
    liveSource = this;
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  close() { this.closed = true; }
  emit(name, data) {
    this.listeners.get(name)?.({data: JSON.stringify(data)});
  }
};
h.state.finished = false;
h.state.runState = "running";
h.connectSSE();
liveSource.emit("stopped", {type: "stopped", phase: "stopped"});
check(h.state.finished, "stopped SSE event did not enter read-only mode");
check(h.state.runState === "stopped", "stopped SSE event was labelled finished");
check(liveSource.closed, "stopped SSE event left the stream connected");

PLAYGROUND_CONFIG.observationUrl = "/observation";
h.state.finished = false;
h.state.runState = "running";
globalThis.fetch = async url => {
  if (url === "/observation?timeout=30") {
    return {
      ok: false,
      status: 410,
      json: async () => ({ok: false, error: "agent_dead"}),
    };
  }
  if (url === "/status") {
    return {
      ok: true,
      status: 200,
      json: async () => ({state: "running", elapsed_ms: 12500}),
    };
  }
  throw new Error(`unexpected dead-agent request ${url}`);
};
await h.observeLoop();
check(h.state.finished, "dead agent remained interactive");
check(
  h.state.runState === "running",
  "dead-agent transition falsely labelled the still-transitioning run finished",
);
check(
  elementFor("step-bar-title").textContent === "店铺已关闭",
  "dead-agent transition did not show the closure state",
);
check(
  elementFor("step-bar-detail").textContent.includes("当前经营者已退出")
    && !elementFor("step-bar-detail").textContent.includes("全部经营者已退出"),
  "dead-agent transition incorrectly claimed that every agent exited",
);

const raceRequests = [];
h.state.finished = false;
h.state.runState = "loading";
h.state.registered = false;
globalThis.fetch = async url => {
  raceRequests.push(url);
  if (url === "/status") {
    const terminal = raceRequests.filter(item => item === "/status").length > 1;
    return {
      ok: true,
      status: 200,
      json: async () => terminal
        ? {state: "stopped", elapsed_ms: 13000}
        : {state: "running", elapsed_ms: 12000},
    };
  }
  return {ok: false, status: 410, json: async () => ({})};
};
await h.init();
check(
  JSON.stringify(raceRequests) === JSON.stringify(["/status", "/tools", "/status"]),
  "terminal initialization race did not re-check status",
);
check(h.state.finished, "terminal initialization race stayed interactive");
check(h.state.runState === "stopped", "terminal initialization race lost stopped state");
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_product_tables_use_grouped_sticky_record_layout():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert '<table id="tbl-merchant" class="record-table wide-record-table">' in html
    assert '<table id="tbl-supplier" class="record-table wide-record-table">' in html
    assert '<col class="col-name">' in html
    assert '<tr class="field-group-row">' in html
    assert '<tr class="field-header-row">' in html
    assert '<th colspan="3">Identity</th>' in html
    assert '<th colspan="6">Pricing</th>' in html
    assert '<th colspan="6">Supplier</th>' in html
    assert '<th colspan="5">Ratings &amp; Trust</th>' in html
    assert 'data-sort="downstream_rating"' in html
    assert "this shop's downstream order rating; does not affect demand" in html
    assert "#tbl-merchant tbody td:nth-child(1)" in html
    assert "#tbl-supplier tbody td:nth-child(2)" in html
    assert "left: var(--record-sticky-first-width)" in html


def test_dashboard_all_messages_panel_uses_double_height():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert re.search(
        r"#agent-all-messages\s*\{\s*max-height:\s*960px;",
        html,
    )


def test_record_table_headers_use_pixel_palette_tokens():
    dashboard_html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    playground_html = Path("env/web/templates/human_playground.html").read_text(encoding="utf-8")
    theme_html = Path("env/web/templates/_pixel_theme.html").read_text(encoding="utf-8")

    assert "--pixel-table-group-bg" in theme_html
    assert "--pixel-table-header-bg" in theme_html
    assert "--pixel-table-sticky-bg" in theme_html
    assert "--pixel-table-group-separator" in theme_html
    assert "background: var(--pixel-table-group-bg)" in theme_html
    assert "background: var(--pixel-table-sticky-bg)" in theme_html
    assert ".record-table .field-group-row th + th" in theme_html
    assert "border-left-color: var(--pixel-table-group-separator)" in theme_html

    for html in (dashboard_html, playground_html):
        record_css = "\n".join(
            match.group(0)
            for match in re.finditer(r"\.record-table[^{}]*\{[^{}]*\}", html)
        )
        assert "#e0f2fe" not in record_css
        assert "#e2e8f0" not in record_css


def test_dashboard_daily_sales_uses_shared_builder_with_cumulative_pl_palette():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    daily_block = html[
        html.index("function renderMerchantDailySales"):
        html.index("function renderMerchantSelectedProduct")
    ]

    assert "const DAILY_SALES_COLOR_BY_CATEGORY" in html
    assert "function dailySalesColor" in html
    assert '"office": "#2F9E44"' in html
    assert '"womenswear": "#C2255C"' in html
    assert '"pet_garden": "#5C940D"' in html
    assert '"appliances": "#168AAD"' in html
    assert '"#dbeafe", "#dcfce7", "#fef3c7"' not in html
    assert "optionBuilders?.productDailyRanked" in daily_block
    assert "colorForProduct: product => dailySalesColor" in daily_block
    assert "itemStyleForSegment: segment =>" in daily_block
    assert "catColor(product.category || product.product_id)" not in daily_block


def test_dashboard_category_colors_are_stable_not_discovery_ordered():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "const CATEGORY_COLOR_BY_NAME" in html
    assert "function stablePaletteIndex" in html
    assert "Object.keys(CAT_COLORS).length" not in html
    for category in (
        "office",
        "womenswear",
        "pet_garden",
        "appliances",
        "home_decor",
        "home_goods",
        "cleaning",
        "toys",
        "bags",
        "sports",
    ):
        assert f'"{category}":' in html


def test_dashboard_live_charts_use_shared_pixel_chart_style():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    live_chart_helpers = html[html.index("function lineOpt"):html.index("function resizeAll")]

    assert "pixelChartBase" in live_chart_helpers
    assert "pixelTooltip" in live_chart_helpers
    assert "PIXEL_CHART_GRID" in live_chart_helpers
    assert "PIXEL_CHART_INK" in live_chart_helpers
    assert "borderWidth: 1" in live_chart_helpers
    assert "#6b7280" not in live_chart_helpers


def test_dashboard_leaderboard_line_charts_share_hover_position():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'const EXP_T_LINE_HOVER_GROUP = "dashboard-experiment-t-line-hover";' in html
    assert "function connectLineHover(chart, groupName)" in html
    assert "echarts.connect(groupName)" in html

    assert 'expChart("ch-exp-net-assets"' in html
    assert "{lineHoverGroup: EXP_T_LINE_HOVER_GROUP}" in html
    assert 'lineChart("ch-exp-cum-gmv", charts.cum_gmv, "GMV");' in html
    assert 'lineChart("ch-exp-cum-profit", charts.cum_net_profit, "net_profit");' in html
    assert 'lineChart("ch-exp-cum-fine", charts.cum_fine, "fine");' in html
    assert 'lineChart("ch-exp-cum-orders", charts.cum_orders, "orders");' in html
    assert "LIVE_T_LINE_HOVER_GROUP" not in html
    assert "{lineHoverGroup: LIVE_T_LINE_HOVER_GROUP}" not in html


def test_human_playground_unknown_agent_returns_404(client):
    c, app = client
    run_id = app.registry.create_run(
        _tiny_scenario(),
        name="human playground",
        bootstrap_agent="human",
        auto_start=False,
    )

    resp = c.get(f"/runs/{run_id}/playground?agent_id=ghost")

    assert resp.status_code == 404


def test_dashboard_analysis_uses_framework_model_labels_and_requested_charts(client):
    c, app = client
    _result_run(
        app,
        name="react one",
        net_assets=5000.0,
        react_model="model-a",
        tool_calls=["query_balance", "query_balance", "query_my_orders"],
    )

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "Tokens and USD" not in html
    assert "Tool-call Mix" not in html
    assert "Status Breakdown" not in html
    assert "Cumulative GMV" in html
    assert "Cumulative Net Profit" in html
    assert "Cumulative Fine" in html
    assert "Cumulative Orders" in html
    assert "Shop Rating Score" in html
    assert "Average Product Price" in html
    assert "Average Product Margin" in html
    assert "Average Product Rating" in html
    assert "Tool Calls by Function Type" in html
    assert "Tool Calls Detail" in html
    assert "Shelf Product Count" in html
    assert "Shelf &amp; Sell-through" in html
    assert "New Products Tried" in html
    assert "Effective Window Rate" in html
    assert "Sourcing Activity" not in html
    assert "Agent Runtime Health" in html
    assert "Listing Action Calls" not in html
    assert "Listing Tool Calls" not in html
    assert "Weekly GMV" in html
    assert "Weekly Profit" in html
    assert "React (model-a)" in html
    assert "react / model-a" not in html
    assert "framework" in html and "model" in html

    assert '<option value="avg_final_net_assets">Final Net Assets Ranking</option>' in html
    assert '<option value="avg_cum_gmv">GMV Ranking</option>' in html
    assert '<option value="elapsed_ms">Total Time Ranking</option>' in html
    chart_titles = re.findall(r'<div class="ctitle">([^<]+)</div>', html)
    analysis_titles = chart_titles[:11]
    assert analysis_titles == [
        "Net Assets",
        "Cumulative GMV",
        "Cumulative Net Profit",
        "Cumulative Fine",
        "Cumulative Orders",
        "Tool Calls by Function Type",
        "Tool Calls Detail",
        "Shop Rating Score",
        "Average Product Rating",
        "Average Product Price",
        "Average Product Margin",
    ]
    assert chart_titles[11:19] == [
        "Net Assets vs Cost",
        "Cumulative Orders vs Cumulative Net Profit",
        "Shelf &amp; Sell-through",
        "GMV / Profit by Period",
        "Effective Window Rate",
        "Total Tool Calls",
        "New Products Tried",
        "Agent Runtime Health",
    ]
    assert html.index('id="ch-exp-tool-calls-detail"') < html.index('id="ch-exp-shop-rating-score"')
    assert html.index('id="ch-exp-tool-calls-detail"') < html.index('id="ch-exp-average-product-price"')
    assert html.index('id="ch-exp-shop-rating-score"') < html.index('id="ch-exp-average-product-rating"')
    assert html.index('id="ch-exp-average-product-rating"') < html.index('id="ch-exp-average-product-price"')
    assert html.index('id="ch-exp-average-product-price"') < html.index('id="ch-exp-average-product-margin"')
    assert 'class="leaderboard-rating-grid"' in html
    assert "#dashboard-leaderboard .leaderboard-rating-grid" in html
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in html
    assert (
        '<div class="chart-wrap"><div class="ctitle">Tool Calls by Function Type</div>'
        '<div class="chart-box" id="ch-exp-tool-calls-category"></div></div>\n'
        '        <div class="chart-wrap"><div class="ctitle">Tool Calls Detail</div>'
        '<div class="chart-box" id="ch-exp-tool-calls-detail"></div></div>\n'
        '        <div class="leaderboard-rating-grid">\n'
        '        <div class="chart-wrap"><div class="ctitle">Shop Rating Score</div>'
        '<div class="chart-box line-chart" id="ch-exp-shop-rating-score"></div></div>\n'
        '        <div class="chart-wrap"><div class="ctitle">Average Product Rating</div>'
        '<div class="chart-box line-chart" id="ch-exp-average-product-rating"></div></div>\n'
        '        <div class="chart-wrap"><div class="ctitle">Average Product Price</div>'
        '<div class="chart-box line-chart" id="ch-exp-average-product-price"></div></div>\n'
        '        <div class="chart-wrap"><div class="ctitle">Average Product Margin</div>'
        '<div class="chart-box line-chart" id="ch-exp-average-product-margin"></div></div>\n'
        '        </div>'
    ) in html
    assert "SHOP_RATING_BAND_COLORS" in html
    assert "function shopRatingMarkIcon(stars)" not in html
    assert "symbol: shopRatingMarkIcon(stars)" not in html
    assert "symbolSize: 24" not in html
    assert 'width="24" height="24"' not in html
    assert 'width="70" height="24"' not in html
    assert "Array.from({length: 5}" not in html
    assert 'symbol: "pin"' not in html
    assert "function shopRatingYAxis(rows, scale=null)" in html
    assert "function shopRatingYAxes(rows)" in html
    assert "function shopRatingAxisBounds(values, lowerBound, upperBound)" in html
    assert "yAxis: shopRatingYAxes(rows)" in html
    assert 'return {...pixelValueAxis("score", false), ...bounds};' in html
    assert 'series.yAxisIndex = shopRatingScale(row) === "five" ? 0 : 1' in html
    assert "fivePointScale ? 5 : 1" in html
    assert 'min: 1, max: 5, interval: 1' not in html
    assert 'yAxis: {...pixelValueAxis("score", false), min: 0, max: 1}' not in html
    assert "function shopRatingTransitionMarks" not in html
    assert "series.markArea = shopRatingBandArea" in html
    assert "series.markPoint" not in html
    assert "function renderAverageProductPriceChart" in html
    assert "function renderAverageProductMarginChart" in html
    assert "function formatAverageProductMarginTooltip" in html
    assert 'pixelValueAxis("avg margin (%)")' in html
    assert "function renderAverageProductRatingChart" in html
    assert 'id="ch-exp-tool-calls-category"' in html
    assert 'id="ch-exp-tool-calls-detail"' in html
    assert 'id="viz-framework-filter"' in html
    assert 'id="viz-model-filter"' in html
    assert 'id="viz-day-from"' in html
    assert 'id="viz-day-to"' in html
    assert 'id="viz-show-all"' in html
    assert 'aria-label="Show or hide all filtered leaderboard runs"' in html
    assert 'class="show-all-slot"' in html
    assert "#dashboard-leaderboard .leaderboard-controls .show-all-slot" in html
    assert "#dashboard-leaderboard .leaderboard-controls .show-all-toggle" in html
    assert "height: 18px;" in html
    assert "function filteredLeaderboardRows" in html
    assert "function syncVizShowAll" in html
    assert '$("viz-show-all")?.addEventListener("change"' in html
    assert "if (show) leaderboardViz.hiddenLeaderKeys.delete(key);" in html
    assert "else leaderboardViz.hiddenLeaderKeys.add(key);" in html
    assert "syncVizShowAll(rows);" in html
    assert 'id="total-tool-heatmap-mode"' in html
    assert 'value="total_tool_calls"' in html
    assert 'value="sourcing_calls"' in html
    assert 'value="listing_action_ui_calls"' in html
    assert 'value="listing_action_list_calls"' in html
    assert 'value="listing_action_delist_calls"' in html
    assert 'value="listing_action_price_calls"' in html
    assert 'value="search_products_calls"' in html
    assert 'value="get_daily_report_calls"' in html
    assert 'id="shelf-heatmap-mode"' in html
    assert 'value="shelf_product_count"' in html
    assert 'value="shelf_utilization_rate"' in html
    assert 'value="sell_through_capacity_rate"' in html
    assert 'value="sell_through_active_shelf_rate"' in html
    assert 'id="weekly-financial-mode"' in html
    assert 'id="weekly-chart-mode"' in html
    assert 'id="weekly-chart-grain"' in html
    assert '<option value="week">Week</option>' in html
    assert '<option value="month">Month</option>' in html
    assert 'id="sourcing-heatmap-mode"' not in html
    assert html.index('id="ch-exp-heat-effective-window-rate"') < html.index(
        'id="ch-exp-heat-total-tool-calls"'
    )
    assert 'id="runtime-health-mode"' in html
    assert '<option value="abnormal_ended_windows">Abnormal Ended Windows</option>' in html
    assert 'label: "Abnormal Ended Windows"' in html
    assert 'if ($("shelf-heatmap-mode")) $("shelf-heatmap-mode").value = "shelf_product_count";' in html
    assert 'if ($("runtime-health-mode")) $("runtime-health-mode").value = "abnormal_ended_windows";' in html
    assert '$("runtime-health-mode")?.addEventListener("change"' in html
    assert "function renderWeeklyMetricChart" in html
    assert "function renderWeeklyLineChart" in html
    assert "function weeklyChartGrain" in html
    assert "function periodicMetricData" in html
    assert "function virtualDateLabel" in html
    assert "`${month}/${pad2(date.getUTCDate())}/${year}`" in html
    assert 'label: `W${week}`' in html
    assert 'label: `M${month}`' in html
    assert 'const source = weeklyChartGrain() === "month" ? charts.monthly : charts.weekly;' in html
    assert 'const axis = pixelValueAxis("days", false);' in html
    assert 'rotate: hasVirtualDates ? 25 : 0' not in html
    assert "renderShelfHeatmap(charts)" in html
    assert "renderWeeklyFinancialHeatmap(charts)" in html
    assert "renderEffectiveWindowRate(charts)" in html
    assert "renderRuntimeHealthHeatmap(charts)" in html
    assert "function runtimeMetricSummary(charts, metric)" in html
    assert "const {rows} = periodicMetricData(charts, metric);" in html
    assert "Partial ${summary.completeRuns}/${summary.totalRuns} complete" in html
    assert 'cellCoverage === "partial"' in html
    assert "if (!hasActiveWeek) return;" in html
    assert "let totalRuns = 0;" in html
    assert '<th>show</th><th data-sort="rank" class="sortable">rank</th>' in html
    assert '<th data-sort="name" class="sortable">name</th>' in html
    assert '<th data-sort="run_id" class="sortable">run id</th>' in html
    assert '<th data-sort="framework" class="sortable">framework</th>' in html
    assert '<th data-sort="model" class="sortable">model</th>' in html
    assert '<th data-sort="avg_cum_gmv" class="sortable">gmv</th>' in html
    assert '<th data-sort="avg_net_profit" class="sortable">profit</th>' in html
    assert '<th data-sort="avg_cum_fine" class="sortable">total fines</th>' in html
    assert '<th data-sort="avg_orders" class="sortable">orders</th>' in html
    assert "formatSeriesTooltip" in html
    assert "Shop rating: terminal order weighted score → daily stars → demand multiplier" in html
    assert "fmtExpFixed(value, 2)" in html
    assert "showSymbol: (s.data || []).length <= 1" in html

    run_results = build_run_results(app.registry)
    charts = build_charts(app.registry, run_results)
    assert charts["net_assets"][0]["label"] == "React (model-a)"
    assert charts["net_assets_cost"][0]["framework"] == "React"
    assert charts["net_assets_cost"][0]["model"] == "model-a"
    assert charts["net_assets_cost"][0]["cumulative_orders"] == 5
    assert charts["net_assets_cost"][0]["cumulative_net_profit"] == 1000.0
    assert '<div class="chart-box line-chart" id="ch-exp-net-assets-cost"></div>' in html
    assert '<div class="chart-box line-chart" id="ch-exp-orders-profit"></div>' in html
    assert "function modelScatterSymbol(model)" in html
    assert html.count("symbol: modelScatterSymbol(d.model)") == 2
    assert charts["cum_gmv"][0]["label"] == "React (model-a)"
    assert charts["cum_net_profit"][0]["label"] == "React (model-a)"
    assert charts["cum_fine"][0]["label"] == "React (model-a)"
    assert charts["cum_orders"][0]["label"] == "React (model-a)"
    assert charts["cum_orders"][0]["data"] == [[1, 2.0], [4, 5.0]]
    assert charts["shop_rating_score"][0]["label"] == "React (model-a)"
    assert charts["shop_rating_score"][0]["data"] == [[1, 0.9], [4, 0.9]]
    assert charts["shop_rating_score"][0]["rating_scale"] == "0-1"
    assert "average_product_margin" in charts
    assert "average_product_rating" in charts
    assert charts["tool_calls"]["tools"] == ["query_balance", "query_my_orders"]
    assert charts["tool_calls"]["runs"][0]["label"] == "React (model-a)"
    assert charts["tool_calls"]["runs"][0]["counts"] == {
        "query_balance": 2,
        "query_my_orders": 1,
    }


def test_dashboard_rating_chart_payload_includes_stars_thresholds_price_and_product_rating(client):
    _, app = client
    run_id = _result_run(
        app,
        name="rating run",
        net_assets=5000.0,
        react_model="rating-model",
    )
    conn = app.registry.conn_for(run_id)
    dbm.write_metrics(conn, run_id, "agent_0", 1, {
        "shop_rating_mean": 3.6,
        "shop_rating_stars": 3,
        "avg_listing_sale_price": 20.0,
        "avg_listing_sale_price_count": 2,
        "avg_listing_margin": 7.5,
        "avg_listing_margin_count": 2,
        "avg_listing_margin_ratio": 0.375,
        "avg_listing_margin_ratio_count": 2,
        "avg_listing_rating": 3.8,
        "avg_listing_rating_count": 2,
    })
    dbm.write_metrics(conn, run_id, "agent_0", 4, {
        "shop_rating_mean": 3.9,
        "shop_rating_stars": 4,
        "avg_listing_sale_price": 30.0,
        "avg_listing_sale_price_count": 2,
        "avg_listing_margin": 12.5,
        "avg_listing_margin_count": 2,
        "avg_listing_margin_ratio": 0.4167,
        "avg_listing_margin_ratio_count": 2,
        "avg_listing_rating": 4.1,
        "avg_listing_rating_count": 2,
    })

    charts = build_charts(app.registry, build_run_results(app.registry))

    shop = charts["shop_rating_score"][0]
    assert shop["run_id"] == run_id
    assert shop["data"] == [[1, 3.6], [4, 3.9]]
    assert shop["rating_scale"] == "1-5"
    assert shop["stars"] == [[1, 3.0], [4, 4.0]]
    assert shop["thresholds"] == [2.5, 3.3, 3.8, 4.2]
    assert shop["star_multipliers"] == [0.1, 0.35, 0.8, 1.0, 1.12]

    price = charts["average_product_price"][0]
    assert price["run_id"] == run_id
    assert price["data"] == [[1, 20.0], [4, 30.0]]
    assert price["counts"] == [[1, 2.0], [4, 2.0]]

    margin = charts["average_product_margin"][0]
    assert margin["run_id"] == run_id
    assert margin["data"] == [[1, 0.375], [4, 0.4167]]
    assert margin["counts"] == [[1, 2.0], [4, 2.0]]

    product_rating = charts["average_product_rating"][0]
    assert product_rating["run_id"] == run_id
    assert product_rating["data"] == [[1, 3.8], [4, 4.1]]
    assert product_rating["counts"] == [[1, 2.0], [4, 2.0]]


def test_dashboard_average_product_margin_converts_legacy_amount_metrics(client):
    _, app = client
    run_id = _result_run(
        app,
        name="legacy margin run",
        net_assets=5000.0,
        react_model="legacy-margin-model",
    )
    conn = app.registry.conn_for(run_id)
    dbm.write_metrics(conn, run_id, "agent_0", 1, {
        "avg_listing_sale_price": 20.0,
        "avg_listing_sale_price_count": 2,
        "avg_listing_margin": 7.5,
        "avg_listing_margin_count": 2,
    })

    charts = build_charts(app.registry, build_run_results(app.registry))

    margin = charts["average_product_margin"][0]
    assert margin["data"] == [[1, 0.375]]
    assert margin["counts"] == [[1, 2.0]]


def test_dashboard_average_product_price_falls_back_to_snapshots(client):
    _, app = client
    run_id = _result_run(
        app,
        name="snapshot rating run",
        net_assets=5000.0,
        react_model="snapshot-rating-model",
    )
    snap_dir = Path(app.registry.runs_root) / run_id / "env_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    products_by_t = {
        1: {
            "p1": {"price": 6.0},
            "p2": {"price": 20.0},
        },
        4: {
            "p1": {"price": 6.0},
        },
    }
    for t, listings in (
        (1, [
            {"product_id": "p1", "sale_price": 10.0, "rating_sum": 5.0, "rating_count": 1},
            {"product_id": "p2", "sale_price": 30.0, "rating_sum": 3.0, "rating_count": 1},
        ]),
        (4, [
            {"product_id": "p1", "sale_price": 20.0, "rating_sum": 10.0, "rating_count": 2},
            {"product_id": "p2", "sale_price": 40.0, "rating_sum": 8.0, "rating_count": 2},
        ]),
    ):
        (snap_dir / f"t_{t:05d}.json").write_text(json.dumps({
            "t": t,
            "products": products_by_t[t],
            "agents": [{
                "agent_id": "agent_0",
                "store_listings": listings,
            }],
        }), encoding="utf-8")

    charts = build_charts(app.registry, build_run_results(app.registry))

    price = charts["average_product_price"][0]
    assert price["run_id"] == run_id
    assert price["data"] == [[1, 20.0], [4, 30.0]]
    assert price["counts"] == [[1, 2.0], [4, 2.0]]

    assert charts["average_product_margin"] == []

    product_rating = charts["average_product_rating"][0]
    assert product_rating["run_id"] == run_id
    assert product_rating["data"] == [[1, 4.0], [4, 4.0455]]
    assert product_rating["counts"] == [[1, 2.0], [4, 2.0]]


def test_dashboard_analysis_keeps_zero_tool_call_runs_in_tool_chart_payload(client):
    _, app = client
    zero_id = _result_run(
        app,
        name="end only",
        net_assets=4200.0,
        react_model="zero-tools",
        tool_calls=["end_of_step"],
    )
    active_id = _result_run(
        app,
        name="active tools",
        net_assets=5000.0,
        react_model="active-tools",
        tool_calls=["query_balance"],
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    tool_runs = {row["run_id"]: row for row in charts["tool_calls"]["runs"]}

    assert zero_id in tool_runs
    assert active_id in tool_runs
    assert tool_runs[zero_id]["counts"] == {}
    assert tool_runs[zero_id]["total"] == 1
    assert tool_runs[active_id]["total"] == 2
    assert tool_runs[zero_id]["activity_by_day"][0] == {
        "t": 0,
        "available_windows": 1,
        "effective_windows": 0,
        "total_tool_calls": 1,
    }
    assert charts["tool_calls"]["tools"] == ["query_balance"]


def test_dashboard_analysis_counts_only_merchantbench_env_tool_calls(client):
    _, app = client
    run_id = _result_run(
        app,
        name="mixed origins",
        net_assets=5000.0,
        react_model="mixed-tools",
    )
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        4,
        [{
            "role": "assistant",
            "content": "mixed tools",
            "tool_origin": "mixed",
            "tool_calls": [
                {
                    "id": "call_env_0",
                    "type": "function",
                    "tool_origin": "merchantbench_env",
                    "function": {"name": "query_balance", "arguments": "{}"},
                },
                {
                    "id": "call_native_0",
                    "type": "function",
                    "tool_origin": "hermes_native",
                    "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
                },
                {
                    "id": "call_end_0",
                    "type": "function",
                    "tool_origin": "merchantbench_env",
                    "function": {"name": "end_of_step", "arguments": "{}"},
                },
            ],
        }, {
            "role": "assistant",
            "content": "legacy env tool",
            "tool_calls": [
                {
                    "id": "call_legacy_env_0",
                    "type": "function",
                    "function": {"name": "query_my_orders", "arguments": "{}"},
                },
            ],
        }],
        [],
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    tool_runs = {row["run_id"]: row for row in charts["tool_calls"]["runs"]}

    assert charts["tool_calls"]["tools"] == ["query_balance", "query_my_orders"]
    assert tool_runs[run_id]["counts"] == {
        "query_balance": 1,
        "query_my_orders": 1,
    }
    assert tool_runs[run_id]["total"] == 3


def test_dashboard_analysis_builds_tool_call_categories_weekly_heatmaps_and_stable_colors(client):
    _, app = client
    first_id = _weekly_result_run(app, name="react one", react_model="qwen3.7-max")
    second_id = _weekly_result_run(app, name="react two", react_model="gpt-5.5-0424-global")

    run_results = build_run_results(app.registry)
    charts = build_charts(app.registry, run_results)

    color_by_run = {row["run_id"]: row["color"] for row in charts["runs"]}
    charts_without_first = build_charts(
        app.registry,
        [row for row in run_results if row["run_id"] != first_id],
    )
    assert {row["run_id"]: row["color"] for row in charts_without_first["runs"]}[second_id] == color_by_run[second_id]

    tool_categories = {row["key"]: row for row in charts["tool_calls"]["categories"]}
    assert tool_categories["cash_flow"]["count"] == 2
    assert tool_categories["listing_pricing"]["count"] == 6
    assert tool_categories["listing_pricing"]["tools"][0] == {"name": "list_product", "count": 2}
    assert charts["tool_calls"]["category_specs"] == [
        {"key": "sourcing", "label": "Sourcing", "color": "#168AAD"},
        {"key": "listing_pricing", "label": "Listing and Pricing", "color": "#2F9E44"},
        {"key": "cash_flow", "label": "Cash-Flow", "color": "#F08C00"},
        {"key": "store_state", "label": "Store State", "color": "#1971C2"},
        {"key": "memory", "label": "Memory", "color": "#7048E8"},
    ]

    weekly = charts["weekly"]
    assert weekly["weeks"] == [1, 2]
    metric_rows = {
        metric: {row["run_id"]: row["values"] for row in rows}
        for metric, rows in weekly["metrics"].items()
    }
    assert metric_rows["effective_window_rate"][first_id] == [0.0714, 0.0714]
    assert metric_rows["total_tool_calls"][first_id] == [3, 4]
    assert metric_rows["listing_action_calls"][first_id] == [0, 3]
    assert metric_rows["listing_action_ui_calls"][first_id] == [0, 3]
    assert metric_rows["listing_action_list_calls"][first_id] == [0, 1]
    assert metric_rows["listing_action_delist_calls"][first_id] == [0, 1]
    assert metric_rows["listing_action_price_calls"][first_id] == [0, 1]
    assert metric_rows["sourcing_calls"][first_id] == [0, 0]
    assert metric_rows["market_brief_calls"][first_id] == [0, 0]
    assert metric_rows["hot_search_terms_calls"][first_id] == [0, 0]
    assert metric_rows["search_products_calls"][first_id] == [0, 0]
    assert metric_rows["get_daily_report_calls"][first_id] == [0, 0]
    assert metric_rows["get_product_detail_calls"][first_id] == [0, 0]
    assert metric_rows["get_supplier_profile_calls"][first_id] == [0, 0]
    assert metric_rows["list_supplier_products_calls"][first_id] == [0, 0]
    assert weekly["start_date"] == "2025-06-01"
    assert charts["weekly"]["listing_action_tools"] == [
        {"key": "all", "label": "All Actions", "metric": "listing_action_calls"},
        {"key": "list", "label": "List", "metric": "listing_action_list_calls"},
        {"key": "delist", "label": "Delist", "metric": "listing_action_delist_calls"},
        {"key": "price", "label": "Price", "metric": "listing_action_price_calls"},
    ]
    assert metric_rows["weekly_gmv"][first_id] == [100.0, 50.0]
    assert metric_rows["weekly_profit"][first_id] == [10.0, -15.0]

    monthly = charts["monthly"]
    assert monthly["months"] == [1]
    assert monthly["periods"] == [{
        "index": 1,
        "start_day": 0,
        "end_day": 30,
        "start_date": "2025-06-01",
        "end_date": "2025-06-30",
    }]
    monthly_rows = {
        metric: {row["run_id"]: row["values"] for row in rows}
        for metric, rows in monthly["metrics"].items()
    }
    assert monthly_rows["effective_window_rate"][first_id] == [0.0714]
    assert monthly_rows["total_tool_calls"][first_id] == [7]
    assert monthly_rows["weekly_gmv"][first_id] == [150.0]
    assert monthly_rows["weekly_profit"][first_id] == [-5.0]


def test_dashboard_tool_category_taxonomy_places_supplier_profile_and_business_tools():
    rows = leaderboard_mod._tool_category_rows({
        "get_supplier_profile": 2,
        "query_my_listings": 3,
        "query_balance": 4,
        "query_my_orders": 5,
        "read_memory_doc": 6,
    })

    by_key = {row["key"]: row for row in rows}
    assert list(by_key) == [
        "sourcing",
        "listing_pricing",
        "cash_flow",
        "store_state",
        "memory",
    ]
    assert by_key["sourcing"]["tools"] == [
        {"name": "get_supplier_profile", "count": 2},
    ]
    assert by_key["listing_pricing"]["tools"] == [
        {"name": "query_my_listings", "count": 3},
    ]
    assert by_key["cash_flow"]["tools"] == [
        {"name": "query_balance", "count": 4},
    ]
    assert by_key["store_state"]["tools"] == [
        {"name": "query_my_orders", "count": 5},
    ]
    assert by_key["memory"]["tools"] == [
        {"name": "read_memory_doc", "count": 6},
    ]


def test_weekly_tool_metrics_count_all_sourcing_tools_and_aggregate():
    metrics = leaderboard_mod._weekly_tool_metrics(
        [
            {
                "t": 0,
                "counts": {
                    "market_brief": 1,
                    "hot_search_terms": 2,
                    "search_products": 2,
                    "get_daily_report": 1,
                    "get_product_detail": 2,
                    "get_supplier_profile": 1,
                    "list_supplier_products": 1,
                },
            },
            {
                "t": 7 * 24,
                "counts": {"get_daily_report": 3},
            },
        ],
        step_hours=1.0,
        window_counts={1: 2, 2: 1},
    )

    assert metrics["search_products_calls"] == {1: 2, 2: 0}
    assert metrics["get_daily_report_calls"] == {1: 1, 2: 3}
    assert metrics["market_brief_calls"] == {1: 1, 2: 0}
    assert metrics["hot_search_terms_calls"] == {1: 2, 2: 0}
    assert metrics["get_product_detail_calls"] == {1: 2, 2: 0}
    assert metrics["get_supplier_profile_calls"] == {1: 1, 2: 0}
    assert metrics["list_supplier_products_calls"] == {1: 1, 2: 0}
    assert metrics["sourcing_calls"] == {1: 10, 2: 3}


def test_monthly_tool_metrics_use_calendar_boundaries_and_raw_window_counts():
    virtual_start = date(2025, 6, 1)

    def month_bucket(t, step_hours):
        return leaderboard_mod._month_for_t(t, step_hours, virtual_start)

    metrics = leaderboard_mod._weekly_tool_metrics(
        [
            {"t": 29 * 24, "counts": {"query_balance": 1}},
            {"t": 30 * 24, "counts": {"query_balance": 2}},
        ],
        step_hours=1.0,
        window_counts={1: 14, 2: 1},
        bucket_for_t=month_bucket,
    )

    assert leaderboard_mod._month_for_t(29 * 24, 1.0, virtual_start) == 1
    assert leaderboard_mod._month_for_t(30 * 24, 1.0, virtual_start) == 2
    assert leaderboard_mod._month_period_descriptor(1, virtual_start) == {
        "index": 1,
        "start_day": 0,
        "end_day": 30,
        "start_date": "2025-06-01",
        "end_date": "2025-06-30",
    }
    assert metrics["total_tool_calls"] == {1: 1, 2: 2}
    assert metrics["effective_window_rate"] == {
        1: pytest.approx(1 / 14),
        2: 1.0,
    }


def test_dashboard_builds_runtime_health_metrics_including_abnormal_windows(client):
    _, app = client
    run_id = _weekly_result_run(app, name="runtime health", react_model="model-a")
    agent_log.write_meta(
        app.registry.runs_root,
        run_id,
        {
            "agent_id": "agent_0",
            "framework": "react_160k_compact_30k",
            "extra": {"runtime_health_version": 1},
        },
    )
    agent_log.init_runtime_events(app.registry.runs_root, run_id)
    agent_log.record_runtime_event(
        app.registry.runs_root,
        run_id,
        agent_id="agent_0",
        t=12,
        event_type="merchantbench_api_failed_attempt",
        payload={"status": 425, "error": "stale_step"},
    )
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        12,
        [
            {
                "role": "assistant",
                "content": "query",
                "tool_calls": [{
                    "id": "call_failed",
                    "type": "function",
                    "tool_origin": "merchantbench_env",
                    "function": {"name": "query_balance", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_failed",
                "name": "query_balance",
                "tool_origin": "merchantbench_env",
                "content": json.dumps({"ok": False, "error": "temporary"}),
            },
        ],
        [{
            "turn_idx": 0,
            "context": {
                "compacted": True,
                "provider_api_failed_attempts": 2,
                "retry_exhausted": 1,
                "skills_evolutions": 1,
            },
        }],
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    metric_rows = {
        metric: {row["run_id"]: row["values"] for row in rows}
        for metric, rows in charts["weekly"]["metrics"].items()
    }
    coverage_rows = {
        metric: {row["run_id"]: row.get("coverage") for row in rows}
        for metric, rows in charts["weekly"]["metrics"].items()
    }

    assert metric_rows["api_failed_attempts"][run_id] == [3, 0]
    # Week 1: 12 absent hooks + the t=12 window (retry exhausted and 425 are
    # deduplicated). Week 2: 13 absent hooks.
    assert metric_rows["abnormal_ended_windows"][run_id] == [13, 13]
    assert metric_rows["tool_call_failures"][run_id] == [1, 0]
    assert metric_rows["retry_exhausted"][run_id] == [1, 0]
    assert metric_rows["memory_compactions"][run_id] == [1, 0]
    assert metric_rows["skills_evolutions"][run_id] == [1, 0]
    assert all(coverage_rows[metric][run_id] == "partial"
               for metric in leaderboard_mod.RUNTIME_HEALTH_METRICS)


def test_token_only_context_keeps_hermes_runtime_log_backfill_enabled(client):
    _, app = client
    run_id = _weekly_result_run(app, name="hermes runtime", react_model="model-a")
    wall_ms = 1_700_000_000_000
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        12,
        [{"role": "assistant", "content": "observe", "tool_calls": []}],
        [{"turn_idx": 0, "context": {"tokens": 1000}}],
        hook_open_wall_ms=wall_ms,
        hook_close_wall_ms=wall_ms + 2_000,
    )
    log_dir = Path(agent_log.agent_dir(app.registry.runs_root, run_id)) / "hermes_home" / "logs"
    log_dir.mkdir(parents=True)
    stamp = datetime.fromtimestamp((wall_ms + 1_000) / 1000).strftime(
        "%Y-%m-%d %H:%M:%S,%f"
    )[:-3]
    (log_dir / "agent.log").write_text(
        "\n".join([
            f"{stamp} WARNING agent.conversation_loop: API call failed (attempt 2/2)",
            f"{stamp} INFO agent.tool_executor: tool skill_manage completed (0.01s, 400 chars)",
            f"{stamp} INFO run_agent: OpenAI client closed (agent_close, shared=True) "
            "thread=merchantbench-checkpoint-review:123",
        ]) + "\n",
        encoding="utf-8",
    )

    step = next(
        row for row in leaderboard_mod._tool_call_step_counts(app.registry, run_id)
        if row["t"] == 12
    )
    assert "provider_api_failed_attempts" not in step["runtime"]
    assert "retry_exhausted" not in step["runtime"]
    assert "skills_evolutions" not in step["runtime"]
    assert step["runtime_telemetry"] == []

    charts = build_charts(app.registry, build_run_results(app.registry))
    rows = {
        metric: next(row for row in metric_rows if row["run_id"] == run_id)
        for metric, metric_rows in charts["weekly"]["metrics"].items()
    }
    assert rows["api_failed_attempts"]["values"] == [1, None]
    assert rows["api_failed_attempts"]["coverage"] == "partial"
    assert rows["api_failed_attempts"]["available"] is False
    assert rows["api_failed_attempts"]["coverage_values"] == [
        "partial", "unavailable",
    ]
    assert rows["retry_exhausted"]["values"] == [1, None]
    assert rows["retry_exhausted"]["coverage"] == "partial"
    assert rows["skills_evolutions"]["values"] == [1, None]
    assert rows["skills_evolutions"]["coverage"] == "partial"
    assert rows["memory_compactions"]["values"] == [None, None]
    assert rows["memory_compactions"]["coverage"] == "unavailable"


def test_abnormal_windows_count_foreground_nonretryable_but_not_review_thread(client):
    _, app = client
    run_id = _weekly_result_run(app, name="foreground terminal only")
    wall_ms = 1_700_000_000_000
    for t, offset in ((12, 0), (192, 10_000)):
        agent_log.write_step_index(
            app.registry.runs_root,
            run_id,
            t,
            [{"role": "assistant", "content": "observe", "tool_calls": []}],
            [],
            hook_open_wall_ms=wall_ms + offset,
            hook_close_wall_ms=wall_ms + offset + 2_000,
        )

    def stamp(offset_ms: int) -> str:
        return datetime.fromtimestamp((wall_ms + offset_ms) / 1000).strftime(
            "%Y-%m-%d %H:%M:%S,%f"
        )[:-3]

    log_dir = (
        Path(agent_log.agent_dir(app.registry.runs_root, run_id))
        / "hermes_home" / "logs"
    )
    log_dir.mkdir(parents=True)
    (log_dir / "agent.log").write_text(
        "\n".join([
            f"{stamp(1_000)} WARNING agent.conversation_loop: "
            "API call failed (attempt 1/6) thread=MainThread:1",
            f"{stamp(1_100)} ERROR agent.conversation_loop: "
            "Non-retryable client error: MPE-001",
            f"{stamp(11_000)} WARNING agent.conversation_loop: "
            "API call failed (attempt 1/6) thread=merchantbench-checkpoint-review:2",
            f"{stamp(11_100)} ERROR agent.conversation_loop: "
            "Non-retryable client error: PRE-004",
        ]) + "\n",
        encoding="utf-8",
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    row = next(
        item for item in charts["weekly"]["metrics"]["abnormal_ended_windows"]
        if item["run_id"] == run_id
    )
    # Missing hooks contribute [12, 13]. Only the foreground t=12 terminal
    # adds one; the t=192 checkpoint-review failure is not a business window.
    assert row["values"] == [13, 13]


def test_checkpoint_review_usage_is_not_counted_as_skill_evolution(client):
    _, app = client
    run_id = _weekly_result_run(app, name="review without skill mutation")
    result = agent_log.record_auxiliary_usage(
        app.registry.runs_root,
        run_id,
        "agent_0",
        "checkpoint-review-only",
        {"input": 100, "output": 20, "total": 120},
        t=12,
        source="checkpoint_review",
    )
    assert result["recorded"] is True

    charts = build_charts(app.registry, build_run_results(app.registry))
    row = next(
        item for item in charts["weekly"]["metrics"]["skills_evolutions"]
        if item["run_id"] == run_id
    )
    assert row["values"] == [None, None]
    assert row["coverage"] == "unavailable"


def test_runtime_health_ignores_other_agents_messages_and_context(client):
    _, app = client
    run_id = _weekly_result_run(app, name="multi agent runtime")
    messages = [
        {
            "role": "assistant",
            "content": "agent 1 call",
            "tool_calls": [{
                "id": "agent1_call",
                "type": "function",
                "tool_origin": "merchantbench_env",
                "function": {"name": "query_balance", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "agent1_call",
            "name": "query_balance",
            "tool_origin": "merchantbench_env",
            "content": json.dumps({"ok": False, "error": "agent 1 failed"}),
        },
        {
            "role": "assistant",
            "content": "agent 0 call",
            "tool_calls": [{
                "id": "agent0_call",
                "type": "function",
                "tool_origin": "merchantbench_env",
                "function": {"name": "query_balance", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "agent0_call",
            "name": "query_balance",
            "tool_origin": "merchantbench_env",
            "content": json.dumps({"ok": True}),
        },
    ]
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        12,
        messages,
        [
            {
                "turn_idx": 0,
                "agent_id": "agent_1",
                "context": {
                    "compacted": True,
                    "provider_api_failed_attempts": 9,
                    "retry_exhausted": 4,
                },
            },
            {
                "turn_idx": 0,
                "agent_id": "agent_0",
                "context": {"provider_api_failed_attempts": 1},
            },
        ],
        message_agents=["agent_1", "agent_1", "agent_0", "agent_0"],
    )

    step = next(
        row for row in leaderboard_mod._tool_call_step_counts(app.registry, run_id)
        if row["t"] == 12
    )
    assert step["counts"] == {"query_balance": 1}
    assert step["runtime"] == {"provider_api_failed_attempts": 1}
    assert step["runtime_telemetry"] == ["provider_api_failed_attempts"]


def test_runtime_health_deduplicates_executions_and_skips_historical_results(client):
    _, app = client
    run_id = _weekly_result_run(app, name="deduplicated failures")
    failure = {
        "role": "tool",
        "tool_call_id": "failed_call",
        "name": "query_balance",
        "tool_origin": "merchantbench_env",
        "runtime_execution_id": "execution-1",
        "content": json.dumps({"ok": False, "error": "failed"}),
    }
    historical = {
        **failure,
        "tool_call_id": "historical_call",
        "runtime_execution_id": "",
        "runtime_historical": True,
    }
    messages = [
        {"role": "assistant", "content": "runtime failures", "tool_calls": []},
        failure,
        dict(failure),
        historical,
    ]
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        12,
        messages,
        [],
        message_agents=["agent_0"] * len(messages),
    )

    step = next(
        row for row in leaderboard_mod._tool_call_step_counts(app.registry, run_id)
        if row["t"] == 12
    )
    assert step["runtime"]["tool_call_failures"] == 1


def test_tool_failure_coverage_is_unavailable_without_trace(client):
    _, app = client
    run_id = _weekly_result_run(app, name="missing runtime trace")
    shutil.rmtree(
        Path(agent_log.agent_dir(app.registry.runs_root, run_id)) / "by_step"
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    row = next(
        item for item in charts["weekly"]["metrics"]["tool_call_failures"]
        if item["run_id"] == run_id
    )
    assert row["values"] == [None, None]
    assert row["coverage_values"] == ["unavailable", "unavailable"]
    assert row["coverage"] == "unavailable"


def test_empty_hermes_log_does_not_claim_complete_coverage(client):
    _, app = client
    run_id = _weekly_result_run(app, name="empty hermes log")
    wall_ms = 1_700_000_000_000
    agent_log.write_step_index(
        app.registry.runs_root,
        run_id,
        12,
        [{"role": "assistant", "content": "observe", "tool_calls": []}],
        [{"turn_idx": 0, "context": {"tokens": 1000}}],
        hook_open_wall_ms=wall_ms,
        hook_close_wall_ms=wall_ms + 2_000,
    )
    log_dir = (
        Path(agent_log.agent_dir(app.registry.runs_root, run_id))
        / "hermes_home" / "logs"
    )
    log_dir.mkdir(parents=True)
    (log_dir / "agent.log").write_text("", encoding="utf-8")

    charts = build_charts(app.registry, build_run_results(app.registry))
    retry_row = next(
        item for item in charts["weekly"]["metrics"]["retry_exhausted"]
        if item["run_id"] == run_id
    )
    skills_row = next(
        item for item in charts["weekly"]["metrics"]["skills_evolutions"]
        if item["run_id"] == run_id
    )
    assert retry_row["values"] == [None, None]
    assert retry_row["coverage"] == "unavailable"
    assert skills_row["values"] == [None, None]
    assert skills_row["coverage"] == "unavailable"


def test_runtime_health_marks_known_server_api_counts_partial_and_unknown_metrics_unavailable(client):
    _, app = client
    run_id = _weekly_result_run(app, name="unknown telemetry", react_model="model-a")
    agent_log.init_runtime_events(app.registry.runs_root, run_id)

    charts = build_charts(app.registry, build_run_results(app.registry))
    api_row = next(
        row for row in charts["weekly"]["metrics"]["api_failed_attempts"]
        if row["run_id"] == run_id
    )
    retry_row = next(
        row for row in charts["weekly"]["metrics"]["retry_exhausted"]
        if row["run_id"] == run_id
    )
    memory_row = next(
        row for row in charts["weekly"]["metrics"]["memory_compactions"]
        if row["run_id"] == run_id
    )
    skills_row = next(
        row for row in charts["weekly"]["metrics"]["skills_evolutions"]
        if row["run_id"] == run_id
    )
    assert api_row["values"] == [0, 0]
    assert api_row["coverage"] == "partial"
    assert api_row["available"] is False
    assert retry_row["values"] == [None, None]
    assert retry_row["coverage"] == "unavailable"
    assert retry_row["available"] is False
    assert memory_row["values"] == [None, None]
    assert memory_row["coverage"] == "unavailable"
    assert skills_row["values"] == [None, None]
    assert skills_row["coverage"] == "unavailable"


def test_runtime_health_coverage_starts_at_registration_step(client):
    _, app = client
    run_id = _weekly_result_run(app, name="late telemetry", react_model="model-a")
    agent_log.write_meta(
        app.registry.runs_root,
        run_id,
        {
            "agent_id": "agent_0",
            "framework": "late-agent",
            "extra": {
                "runtime_health_version": 1,
                "runtime_health_capabilities": {
                    "provider_api_failed_attempts": "not_applicable",
                    "retry_exhausted": "not_applicable",
                    "memory_compactions": "not_applicable",
                    "skills_evolutions": "not_applicable",
                },
            },
        },
    )
    agent_log.record_runtime_event(
        app.registry.runs_root,
        run_id,
        agent_id="agent_0",
        t=12,
        event_type="merchantbench_api_failed_attempt",
        payload={"status": 425},
    )
    agent_log.init_runtime_events(
        app.registry.runs_root,
        run_id,
        agent_id="agent_0",
        t=200,
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    api_row = next(
        row for row in charts["weekly"]["metrics"]["api_failed_attempts"]
        if row["run_id"] == run_id
    )
    retry_row = next(
        row for row in charts["weekly"]["metrics"]["retry_exhausted"]
        if row["run_id"] == run_id
    )
    assert api_row["values"] == [1, 0]
    assert api_row["coverage_values"] == ["partial", "partial"]
    assert api_row["active_values"] == [True, True]
    assert api_row["coverage"] == "partial"
    assert retry_row["values"] == [None, 0]
    assert retry_row["coverage_values"] == ["unavailable", "partial"]


def test_dashboard_analysis_builds_weekly_shelf_count_and_sell_through(client):
    _, app = client
    run_id = _weekly_result_run(app, name="weekly shelf", react_model="model-a")
    conn = app.registry.conn_for(run_id)
    dbm.write_events(conn, run_id, [
        EventLog(t=0, event_type="agent_list_product", entity_id="p1", agent_id="agent_0", payload={}),
        EventLog(t=24, event_type="agent_list_product", entity_id="p2", agent_id="agent_0", payload={}),
        EventLog(t=192, event_type="agent_delist_product", entity_id="p1", agent_id="agent_0", payload={}),
        EventLog(t=200, event_type="agent_list_product", entity_id="p3", agent_id="agent_0", payload={}),
    ])
    dbm.insert_orders(conn, run_id, [
        Order(
            order_id="o1",
            product_id="p1",
            supplier_id="s1",
            order_t=36,
            promised_delivery_t=72,
            sale_price=10.0,
            purchase_price=4.0,
            agent_id="agent_0",
        ),
        Order(
            order_id="o2",
            product_id="p3",
            supplier_id="s1",
            order_t=220,
            promised_delivery_t=260,
            sale_price=12.0,
            purchase_price=5.0,
            agent_id="agent_0",
        ),
    ])

    charts = build_charts(app.registry, build_run_results(app.registry))
    metric_rows = {
        metric: {row["run_id"]: row["values"] for row in rows}
        for metric, rows in charts["weekly"]["metrics"].items()
    }

    assert metric_rows["shelf_product_count"][run_id] == [2, 2]
    assert metric_rows["shelf_utilization_rate"][run_id] == [0.04, 0.04]
    assert metric_rows["weekly_new_unique_products"][run_id] == [2, 1]
    # Default max_active_listings is 50, so selling one distinct product uses
    # 1/50 of total shelf capacity in each week.
    assert metric_rows["sell_through_capacity_rate"][run_id] == [0.02, 0.02]
    assert metric_rows["sell_through_active_shelf_rate"][run_id] == [0.5, 0.5]


def test_dashboard_weekly_shelf_metrics_preserve_same_tick_listing_event_order(client):
    _, app = client
    run_id = _weekly_result_run(app, name="same tick shelf", react_model="model-a")
    conn = app.registry.conn_for(run_id)
    dbm.write_events(conn, run_id, [
        EventLog(t=24, event_type="agent_list_product", entity_id="p1", agent_id="agent_0", payload={}),
        EventLog(t=24, event_type="agent_delist_product", entity_id="p1", agent_id="agent_0", payload={}),
    ])

    charts = build_charts(app.registry, build_run_results(app.registry))
    metric_rows = {
        metric: {row["run_id"]: row["values"] for row in rows}
        for metric, rows in charts["weekly"]["metrics"].items()
    }

    assert metric_rows["shelf_product_count"][run_id] == [0, 0]
    assert metric_rows["sell_through_capacity_rate"][run_id] == [0.0, 0.0]
    assert metric_rows["sell_through_active_shelf_rate"][run_id] == [None, None]


def test_dashboard_counts_legacy_set_promised_ship_hours_as_listing_pricing(client):
    """Historical set_promised_ship_hours traces keep listing/pricing classification
    and contribute to listing_action_calls aggregate, but are not exposed in the
    current listing-action UI.
    """
    _, app = client
    run_id = _result_run(
        app,
        name="legacy promise",
        net_assets=5000.0,
        react_model="model-a",
        tool_calls=["set_promised_ship_hours"],
    )
    charts = build_charts(app.registry, build_run_results(app.registry))

    categories = {row["key"]: row for row in charts["tool_calls"]["categories"]}
    assert categories["listing_pricing"]["count"] == 1
    assert categories["listing_pricing"]["tools"] == [
        {"name": "set_promised_ship_hours", "count": 1},
    ]

    listing_action_calls = next(
        row["values"]
        for row in charts["weekly"]["metrics"]["listing_action_calls"]
        if row["run_id"] == run_id
    )
    assert listing_action_calls == [1]

    assert charts["weekly"]["listing_action_tools"] == [
        {"key": "all", "label": "All Actions", "metric": "listing_action_calls"},
        {"key": "list", "label": "List", "metric": "listing_action_list_calls"},
        {"key": "delist", "label": "Delist", "metric": "listing_action_delist_calls"},
        {"key": "price", "label": "Price", "metric": "listing_action_price_calls"},
    ]


def test_dashboard_analysis_colors_are_stable_by_framework(client):
    _, app = client
    react_id = _result_run(app, name="react one", net_assets=5000.0, react_model="model-a")
    auto_id = _result_run(
        app,
        name="auto one",
        net_assets=4500.0,
        bootstrap_agent="auto_seed",
        react_model=None,
    )

    charts = build_charts(app.registry, build_run_results(app.registry))
    run_meta = {row["run_id"]: row for row in charts["runs"]}

    assert run_meta[react_id]["framework_color"] != run_meta[auto_id]["framework_color"]
    assert run_meta[react_id]["color"] == run_meta[react_id]["framework_color"]
    assert run_meta[auto_id]["color"] == run_meta[auto_id]["framework_color"]


def test_dashboard_analysis_falls_back_to_net_assets_profit_when_profit_series_missing(client):
    _, app = client
    run_id = _weekly_result_run(
        app,
        name="legacy run",
        react_model="legacy-model",
        write_profit_metric=False,
    )

    run_results = build_run_results(app.registry)
    result = next(row["result"] for row in run_results if row["run_id"] == run_id)
    charts = build_charts(app.registry, run_results)
    weekly_profit = {
        row["run_id"]: row["values"]
        for row in charts["weekly"]["metrics"]["weekly_profit"]
    }

    assert result["net_profit"] == 1050.0
    assert weekly_profit[run_id] == [1100.0, -50.0]


def test_dashboard_analysis_uses_pixel_minimal_chart_style(client):
    c, app = client
    _result_run(
        app,
        name="react one",
        net_assets=5000.0,
        react_model="model-a",
        tool_calls=["query_balance", "query_my_orders"],
    )

    resp = c.get("/dashboard")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="run-select"' not in html
    assert "(select a run)" not in html
    assert "--pixel-chart-bg: var(--panel)" in html
    assert "--pixel-chart-paper: #EEF2F7" in html
    assert "--pixel-chart-grid: #E5E7EB" in html
    assert "#dashboard-leaderboard .chart-wrap" in html
    assert "PIXEL_CHART_COLORS" in html
    assert "const PIXEL_CHART_BG = PIXEL_CHART_SURFACE;" in html
    assert '"#2F9E44", "#168AAD", "#F08C00", "#1971C2", "#E03131"' in html
    assert "pixelChartBase" in html
    assert "pixelLineSeries" in html
    assert "lineStyle: {width: 3" in html
    assert "triggerLineEvent: true" in html
    assert "blur: {lineStyle: {opacity: 0.14}" in html
    assert 'blurScope: "coordinateSystem"' in html
    assert "backgroundColor: \"#FFFFFF\"" in html
    assert "borderColor: \"#0F1720\"" in html
    for off_theme_color in ("#fffdf2", "#f7f0d5", "#d8cfaa", "#d0c7a2", "#e0d8b7"):
        assert off_theme_color not in html
    assert "renderToolCategoryChart" in html
    assert "renderToolDetailChart" in html
    assert "renderTotalToolHeatmap" in html
    assert "renderShelfHeatmap" in html
    assert "renderWeeklyFinancialHeatmap" in html
    assert "renderEffectiveWindowRate" in html
    assert "renderRuntimeHealthHeatmap" in html
    assert "renderEChartsHeatmap" in html
    assert "total-tool-heatmap-mode" in html
    assert "shelf-heatmap-mode" in html
    assert "weekly-chart-mode" in html
    assert "weekly-chart-grain" in html
    assert "runtime-health-mode" in html
    assert "listing_action_calls" in html
    assert ".lb-metric-net" in html
    assert ".lb-metric-gmv" in html
    assert ".lb-metric-profit" in html
    assert ".lb-metric-fine" in html
    assert "#dashboard-leaderboard .leaderboard-scroll { overflow-x: auto; }" in html
    assert "#dashboard-leaderboard #tbl-leaderboard { table-layout: auto; min-width: 1700px; }" in html
    assert "#dashboard-leaderboard #tbl-leaderboard td:nth-child(4) code" in html
    assert "#dashboard-leaderboard .viz-toggle::after" in html
    assert "const MODEL_FAMILY_STYLES" in html
    assert "const MODEL_COLOR_BY_NAME" in html
    assert "function runSeriesStyle" in html
    assert "function modelColor" in html
    assert "function modelIcon" in html
    assert "function modelIconUrl" in html
    assert "function chartRunLabel" in html
    assert "function runSeriesColor" in html
    assert "model-logo" in html
    assert "model-only-icon" in html
    assert "model-identity" in html
    assert "lb-framework-badge" in html
    assert "function frameworkBadge" in html
    assert "function modelOnlyIcon" in html
    assert "function modelIdentity" in html
    assert ".lb-tag {" not in html
    assert ".framework-tag," not in html
    assert ".model-tag," not in html
    assert "function frameworkTag" not in html
    assert "function modelTag" not in html
    assert "function tagStyle" not in html
    assert "function modelIconHtml" not in html
    assert ".model-icon {" not in html
    assert "domain_url=https://" in html
    assert 'return "gemini.google.com"' in html
    assert 'return "kimi.com"' in html
    template_html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    assert "'gemini.google.com'" in template_html
    assert "'kimi.com'" in template_html
    assert "name: chartRunLabel(s)" in html
    assert 'type: "solid"' in html
    assert "FRAMEWORK_MARKER_STYLES" not in html
    assert "function frameworkMarkerPoints" not in html
    assert "showFrameworkMarkers" not in html
    assert "markPoint: markerPoints.length" not in html
    assert "function connectSeriesHover" in html
    assert "function tooltipSeriesRow" in html
    assert "opacity:${opacity}" in html
    assert 'id="framework-style-key"' not in html
    assert "modelIdentity(row.model)" in html
    assert "backgroundColor: iconUrl ? {image: iconUrl} : PIXEL_CHART_BG" in html
    assert "formatter: (value, idx) => `{icon${idx}|${modelIcon((rows || [])[idx]?.model)}} {name|${value}}`" in html
    assert "const model = row.model && row.model !== \"—\" ? row.model : \"—\";" in html
    assert "axis_label: `${row.framework || \"—\"} · ${model}`" in html
    assert "yAxis: leaderboardRankingCategoryAxis(leaderboardRankingAxisRows(rows))" in html
    assert "formatter: value => value" in html
    assert "position: \"right\"" in html
    assert "formatter: p => `${metric.format(p.value, rows[p.dataIndex])} {barIcon${p.dataIndex}|${modelIcon((rows || [])[p.dataIndex]?.model)}}`" in html
    assert "position: \"insideLeft\"" not in html
    assert "axis.axisLabel.margin = 12" in html
    assert "width: 210" in html
    assert "height: 18" in html
    assert 'borderColor: iconUrl ? "#D0D5DD" : PIXEL_CHART_INK' in html
    assert ".lb-framework-badge.react-160k-compact-30k {" in html
    assert "background: #EAF2FF;" in html
    assert "background: #EAF8F1;" in html
    assert "background: #FFF2E2;" in html
    assert "td class=\"lb-metric lb-metric-net\"" in html
    assert "td class=\"lb-metric lb-metric-gmv\"" in html
    assert "Listing Tool Calls" not in html
    assert "ResizeObserver" in html
    assert "resizeExperimentCharts" in html
    assert "resizeExperimentChart(id)" in html
    assert "chart.resize({width: el.clientWidth, height: el.clientHeight})" in html
    assert "day < filters.dayTo" in html
    assert 'seriesKey: "net_assets", windowMode: "last"' in html
    assert 'seriesKey: "cum_gmv", windowMode: "delta"' in html
    assert 'seriesKey: "cum_net_profit", windowMode: "delta"' in html
    assert 'seriesKey: "cum_fine", windowMode: "delta"' in html
    assert 'seriesKey: "cum_orders", windowMode: "delta"' in html
    assert 'seriesKey: "shop_rating_score", windowMode: "last"' in html
    assert "function leaderboardWindowSeriesValue" in html
    assert 'return mode === "delta" ? last - (baseline ?? 0) : last;' in html
    assert "ranking_value: leaderboardRankingValue(row, metric)" in html
    assert "value: row.ranking_value" in html
    assert 'stack: "tool-category"' in html
    assert 'stack: "tool-detail"' in html
    assert "experimentCategoryAxis(toolRuns)" in html
    assert "experimentAxisLabel" in html
    assert 'align: "left"' in html
    assert "#dashboard-leaderboard .grid-2 > .chart-wrap { min-width: 0; }" in html
    assert "#dashboard-leaderboard .chart-box.line-chart { height: 340px; }" in html
    assert "#dashboard-leaderboard .chart-box.line-chart.tall { height: clamp(580px, 68vh, 760px); }" in html
    assert "zeroCenter" in html
    assert "clipQuantile" in html
    assert 'colors: ["#B2182B", "#FFFFFF", "#1A9850"], zeroCenter: true, clipQuantile: 0.9' in html
    assert "{tag${idx}| ${value} }" not in html
    assert "backgroundColor: runSeriesColor(row)" not in html
    assert "no-data" in html
    assert "visualMapSeriesIndex" in html
    assert "seriesIndex: visualMapSeriesIndex" in html
    assert "ch-exp-heat-shelf" in html
    assert "ch-exp-heat-weekly-financial" in html
    assert "ch-exp-heat-effective-window-rate" in html
    assert "ch-exp-heat-sourcing" not in html
    assert "ch-exp-heat-new-products" in html
    assert "ch-exp-heat-runtime-health" in html
    assert "sell_through_capacity_rate" in html
    assert "sell_through_active_shelf_rate" in html
    assert 'colors: ["#E0F2FE", "#7DD3FC", "#38BDF8", "#0369A1"]' in html
    assert 'colors: ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45"]' in html
    assert "renderTotalToolHeatmap(charts)" in html
    assert "const rowHeight = opts.rowHeight || 28" in html
    assert "opts.minHeight || 230" in html
    assert "xAxis: pixelValueAxis(\"calls\")" in html
    assert "barWidth: 18" in html
    assert 'axisLabel: {interval: 0, rotate: 20}' not in html


def test_dashboard_human_colors_and_net_assets_framework_controls(client):
    c, app = client
    _result_run(
        app,
        name="human result",
        net_assets=5000.0,
        bootstrap_agent="human",
        react_model=None,
    )

    html = c.get("/dashboard").data.decode("utf-8")

    assert 'id="net-assets-framework-controls"' in html
    assert 'aria-label="Net Assets frameworks"' in html
    assert 'const HUMAN_FRAMEWORK_COLOR = "#E76F51";' in html
    assert "const HUMAN_COLOR_BY_NAME" in html
    assert "beethoven: HUMAN_FRAMEWORK_COLOR" in html
    assert 'mozart: "#F4A261"' in html
    assert 'debussy: "#4F8A5B"' in html
    assert 'if (netAssetsFrameworkKey(row).toLowerCase() === "human") return HUMAN_FRAMEWORK_COLOR;' in html
    assert 'if (framework === "human") return humanColor(model);' in html
    assert "function renderNetAssetsFrameworkControls(charts, rows)" in html
    assert "hiddenNetAssetsFrameworks: new Set()" in html
    assert 'button.addEventListener("mouseenter", showFrameworkFocus)' in html
    assert 'button.addEventListener("click", () =>' in html
    assert "renderNetAssetsChart(charts);" in html
    assert 'lineStyle: {width: 3, opacity: 0.96, cap: "square", join: "miter", color, type: "solid"}' in html


def test_dashboard_tooltips_receive_virtual_time_config_with_weekday(client):
    c, app = client
    scenario = _tiny_scenario()
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-15",
    }
    run_id = app.registry.create_run(
        scenario,
        name="virtual dashboard",
        bootstrap_agent="none",
        auto_start=False,
    )

    resp = c.get(f"/dashboard?run_id={run_id}")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "const VIRTUAL_TIME" in html
    assert '"startDate": "2025-06-15"' in html
    assert "WEEKDAY_SHORT" in html
    assert "formatSimTimeHeader" in html
    assert "Day ${time.day} · Hour ${time.hour} · t=${time.t}" in html
    assert "`${WEEKDAY_SHORT[dt.getUTCDay()]}, ${pad2(month)}/${pad2(dayOfMonth)}/${yyyy} ${hh}:00`" in html


def test_dashboard_time_tooltip_only_uses_tuple_series_values():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function tooltipTFromRows"):
        html.index("const fmtElapsed")
    ]

    assert "Array.isArray(value)" in helper
    assert "axisValue" not in helper
    assert "axisValueLabel" not in helper


def test_dashboard_365d_curve_tooltips_map_day_index_to_sim_time():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'mode === "dayIndex"' in html
    assert "Number(value[0]) * 24 / Number(STEP_HOURS || 1)" in html
    assert 'ch-cat-market365", catalogMarketLineOpt' in html
    assert 'lineOpt(series, {dualY: true, xName: "day", timeTooltip: "dayIndex"})' in html
    assert '], {legend: true, scale: true, xName: "day", timeTooltip: "dayIndex"}' in html


def test_leaderboard_windowed_ranking_and_show_all_helpers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for leaderboard helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    row_helpers = html[
        html.index("function filteredLeaderboardRows"):
        html.index("function setupLeaderboardControls")
    ]
    window_helper = html[
        html.index("function leaderboardWindowSeriesValue"):
        html.index("function leaderboardRankingValue")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
let filters = {framework: "Hermes", model: "", dayFrom: 2, dayTo: 4};
const getVizFilters = () => filters;
const pointDay = (t, stepHours) => Number(t || 0) * Number(stepHours || 1) / 24;
const leaderRowKey = row => String(row.run_id);
const toggle = {};
const $ = id => id === "viz-show-all" ? toggle : null;
const leaderboardViz = {
  payload: {leaderboard: [
    {run_id: "a", framework: "Hermes", model: "m1"},
    {run_id: "b", framework: "Hermes", model: "m2"},
    {run_id: "c", framework: "React", model: "m1"},
  ]},
  hiddenLeaderKeys: new Set(["b"]),
};
eval(process.argv[1]);
eval(process.argv[2]);

const filtered = filteredLeaderboardRows();
check(filtered.map(row => row.run_id).join(",") === "a,b",
  "show-all scope did not follow framework/model filters");
syncVizShowAll(filtered);
check(toggle.checked === false && toggle.indeterminate === true && toggle.disabled === false,
  "show-all toggle did not expose a mixed state");
leaderboardViz.hiddenLeaderKeys.add("a");
syncVizShowAll(filtered);
check(toggle.checked === false && toggle.indeterminate === false,
  "show-all toggle did not expose the all-hidden state");
leaderboardViz.hiddenLeaderKeys.clear();
syncVizShowAll(filtered);
check(toggle.checked === true && toggle.indeterminate === false,
  "show-all toggle did not expose the all-shown state");

const series = {
  step_hours: 24,
  data: [[4, 34], [1, 10], [3, 21], [0, 5], [2, 15]],
};
check(leaderboardWindowSeriesValue(series, "last") === 21,
  "state ranking did not take the last point inside [from, to)");
check(leaderboardWindowSeriesValue(series, "delta") === 11,
  "cumulative ranking did not subtract the pre-window baseline");
const weightedSeries = {
  step_hours: 24,
  data: [[2, 10, 2], [3, 20, 1]],
};
check(Math.abs(leaderboardWindowSeriesValue(weightedSeries, "weightedMean") - (40 / 3)) < 1e-9,
  "weighted ranking did not preserve the raw sample mean");
filters = {...filters, dayFrom: 0};
check(leaderboardWindowSeriesValue(series, "delta") === 21,
  "a window starting at day zero did not use a zero cumulative baseline");
filters = {...filters, dayFrom: 6, dayTo: 7};
check(leaderboardWindowSeriesValue(series, "last") === null,
  "an empty ranking window should omit the run");
"""
    result = subprocess.run(
        [node, "-e", harness, row_helpers, window_helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_average_ranking_matches_summary_and_rejects_partial_windows():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for leaderboard helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helpers = html[
        html.index("function leaderboardRankingSeriesRow"):
        html.index("function leaderboardWindowLabel")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const meanExperimentValues = values => {
  const finite = values.filter(value => value !== null && value !== undefined)
    .map(Number).filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null;
};
const leaderRowKey = row => String(row.run_id);
const pointDay = (t, stepHours) => Number(t || 0) * Number(stepHours || 1) / 24;
let filters = {dayFrom: 0, dayTo: null};
const getVizFilters = () => filters;
const leaderboardViz = {payload: {
  leaderboard: [
    {run_id: "a", avg_final_net_assets: 92051.51},
    {run_id: "b", avg_final_net_assets: 77267.68},
    {
      run_id: "avg",
      is_batch_average: true,
      source_run_ids: ["a", "b"],
      avg_final_net_assets: 84659.595,
      avg_net_profit_margin: 0.3,
    },
  ],
    charts: {
      runs: [{run_id: "a"}, {run_id: "b"}],
    // Reproduce an early-ended source whose chart is absent. The complete-run
    // bar must still use the same authoritative average as the table.
    net_assets: [
      {run_id: "b", step_hours: 1, data: [[0, 77267.68]]},
      {run_id: "avg", step_hours: 1, data: [[0, 77267.68]]},
    ],
    cum_net_profit: [
      {run_id: "a", step_hours: 1, data: [[0, 10]]},
      {run_id: "b", step_hours: 1, data: [[0, 100]]},
    ],
    cum_gmv: [
      {run_id: "a", step_hours: 1, data: [[0, 20]]},
      {run_id: "b", step_hours: 1, data: [[0, 1000]]},
    ],
    tool_calls: {runs: [
      {
        run_id: "a",
        step_hours: 24,
        activity_by_day: [
          {t: 2, available_windows: 1, effective_windows: 1, total_tool_calls: 4},
        ],
      },
      {
        run_id: "b",
        step_hours: 24,
        activity_by_day: [
          {t: 2, available_windows: 1, effective_windows: 1, total_tool_calls: 6},
        ],
      },
    ]},
  },
}};
eval(process.argv[1]);
const averageRow = leaderboardViz.payload.leaderboard[2];
const netAssetsMetric = {
  field: "avg_final_net_assets",
  seriesKey: "net_assets",
  windowMode: "last",
};
check(
  Math.abs(leaderboardRankingValue(averageRow, netAssetsMetric) - 84659.595) < 1e-9,
  "complete-run ranking disagreed with the summary-table average"
);

filters = {dayFrom: 0, dayTo: 1};
const metric = {
  field: "avg_net_profit_margin",
  value: row => leaderboardRatioRankingValue(row, "cum_net_profit", "cum_gmv"),
};
const value = leaderboardRankingValue(averageRow, metric);
check(Math.abs(value - 0.3) < 1e-9,
  "virtual average used ratio-of-sums instead of mean per-run margin");
check(leaderboardRankingValue(averageRow, netAssetsMetric) === null,
  "a custom window silently averaged only the source curve that was available");
const totalCallsMetric = {
  field: "avg_total_tool_calls",
  value: row => leaderboardActivityRankingValue(row, "total_tool_calls"),
};
check(leaderboardRankingValue(averageRow, totalCallsMetric) === null,
  "an uncovered activity window was treated as zero tool calls");
"""
    result = subprocess.run(
        [node, "-e", harness, helpers],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_group_average_helpers_ignore_missing_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    numeric_helpers = html[
        html.index("function finiteExperimentValues"):
        html.index("function experimentSlotSources")
    ]
    point_helper = html[
        html.index("function averageExperimentPointArrays"):
        html.index("function averageExperimentLineRows")
    ]
    tool_helpers = html[
        html.index("function averageExperimentCountMaps"):
        html.index("function averageExperimentPeriod")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
function experimentChartMeta(row) {
  return {run_id: row.run_id};
}
eval(process.argv[1]);
eval(process.argv[2]);
eval(process.argv[3]);
check(meanExperimentValues([10, null, undefined, 30]) === 20,
  "missing scalar values changed the denominator");
check(Math.abs(sampleStdExperimentValues([10, 30]) - Math.sqrt(200)) < 1e-9,
  "sample standard deviation was incorrect");
const points = averageExperimentPointArrays([
  {data: [[0, 10], [1, 20]]},
  {data: [[0, 30]]},
]);
check(JSON.stringify(points) === JSON.stringify([[0, 20], [1, 20]]),
  "missing time-series points should be ignored instead of treated as zero");
const averagedTools = averageExperimentToolRuns({
  runs: [
    {run_id: "a", counts: {search: 10}, by_step: [{t: 0, counts: {search: 10}}]},
    {run_id: "b", counts: {}, by_step: []},
  ],
}, [{run_id: "avg", source_run_ids: ["a", "b"]}]).runs[2];
check(averagedTools.counts.search === 5,
  "a run with zero calls was omitted from the total-tool denominator");
check(averagedTools.by_step[0].counts.search === 5,
  "a run with zero calls was omitted from the per-step denominator");
"""
    result = subprocess.run(
        [node, "-e", harness, numeric_helpers, point_helper, tool_helpers],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_batch_summary_cell_uses_paper_style_mean_sd_and_sample_size():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function batchSummaryCell"):
        html.index("function sortLeaderboardRows")
    ]
    harness = r"""
const fmtExpFixed = (value, digits=2) => Number(value).toLocaleString(
  "en-US",
  {minimumFractionDigits: digits, maximumFractionDigits: digits}
);
const fmtElapsed = value => `${value} ms`;
const esc = value => String(value);
eval(process.argv[1]);
const rendered = batchSummaryCell(
  {is_batch_average: true, std_score: 19330.13, n_score: 3},
  "score",
  "48,704.81"
);
if (!rendered.includes("48,704.81 ± 19,330.13")) {
  throw new Error(`paper-style mean ± SD is missing: ${rendered}`);
}
if (!rendered.includes("(n=3)")) {
  throw new Error(`per-metric sample size is missing: ${rendered}`);
}
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_group_markdown_export_distinguishes_slot_and_run_models():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("const EXPERIMENT_EXPORT_METRICS"):
        html.index("async function saveExperimentGroups")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const finiteExperimentValues = values => (
  values.filter(value => value !== null && value !== undefined && value !== "")
    .map(Number).filter(Number.isFinite)
);
const meanExperimentValues = values => {
  const finite = finiteExperimentValues(values);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null;
};
const sampleStdExperimentValues = values => {
  const finite = finiteExperimentValues(values);
  if (finite.length < 2) return null;
  const mean = meanExperimentValues(finite);
  return Math.sqrt(
    finite.reduce((sum, value) => sum + ((value - mean) ** 2), 0)
      / (finite.length - 1)
  );
};
const slots = [{
  id: "model::react::slot-model",
  framework: "React",
  model: "slot-model",
}];
const experimentSlots = () => slots;
const leaderboardViz = {basePayload: null, fullPayloadLoaded: false};
const experimentGroups = {runOptions: [
  {run_id: "run-a", framework: "React", model: "run-model-a", status: "finished"},
  {run_id: "run-b", framework: "Hermes", model: "run-model-b", status: "finished"},
]};
let exportStatus = null;
const selectedExperimentGroup = () => group;
const setExperimentStatus = (message, isError=false) => {
  exportStatus = {message, isError};
};
eval(process.argv[1]);
const group = {
  id: "group-main",
  name: "Main | Group",
  batches: [
    {id: "batch-1", name: "Repeat 1", bindings: {"model::react::slot-model": "run-a"}},
    {id: "batch-2", name: "Repeat 2", bindings: {"model::react::slot-model": "run-b"}},
  ],
};
const basePayload = {leaderboard: [
  {run_id: "run-a", model: "run-model-a", framework: "React", avg_final_net_assets: 10},
  {run_id: "run-b", model: "run-model-b", framework: "Hermes", avg_final_net_assets: 30},
]};
const markdown = experimentGroupMarkdown(group, basePayload);
check(markdown.includes("| model | run-id | run-framework | run-model |"),
  "export did not separate configured model, run id, and actual run model");
check(markdown.includes("slot-model") && markdown.includes("run-model-a")
    && markdown.includes("run-model-b"),
  "export lost slot or run model identities");
check(markdown.includes("| 20 ± 14.142136 (n=2) |"),
  "export did not append mean ± sample SD with sample size");
check(markdown.includes("## Final slot summaries (mean ± sample SD)"),
  "export did not label the final summary convention");
check(markdown.includes("# Experiment group: Main \\| Group"),
  "export did not escape markdown table/header text");
downloadSelectedExperimentGroupMarkdown();
check(exportStatus?.isError === true
    && exportStatus.message.includes("still loading"),
  "export did not reject the lightweight leaderboard payload");
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_experiment_batch_visibility_restores_manual_hidden_rows():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for experiment-group helper validation")
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    helper = html[
        html.index("function applyExperimentBatchVisibility"):
        html.index("function refreshExperimentSelectionView")
    ]
    harness = r"""
const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
let batches = [];
const selectedExperimentBatches = () => batches;
const experimentSelectedRunIds = () => new Set(["a"]);
const leaderRowKey = row => row.run_id;
const saveHiddenLeaderKeys = () => {};
const leaderboardViz = {
  basePayload: {leaderboard: [{run_id: "a"}, {run_id: "b"}]},
  hiddenLeaderKeys: new Set(["a"]),
};
const experimentGroups = {
  selectedRunIds: new Set(),
  manualHiddenLeaderKeys: null,
};
eval(process.argv[1]);

applyExperimentBatchVisibility();
check([...leaderboardViz.hiddenLeaderKeys].join(",") === "a",
  "loading with no active batch erased manual visibility");
batches = [{id: "batch-1"}];
applyExperimentBatchVisibility();
check([...leaderboardViz.hiddenLeaderKeys].join(",") === "b",
  "batch selection did not show only its bound run");
leaderboardViz.basePayload = {
  leaderboard: [{run_id: "a"}, {run_id: "b"}, {run_id: "c"}],
};
applyExperimentBatchVisibility();
check([...leaderboardViz.hiddenLeaderKeys].join(",") === "b,c",
  "refreshed unbound runs were not hidden by the active batch");
batches = [];
applyExperimentBatchVisibility();
check([...leaderboardViz.hiddenLeaderKeys].join(",") === "a",
  "clearing batches did not restore manual visibility");
"""
    result = subprocess.run(
        [node, "-e", harness, helper],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_leaderboard_rows_do_not_expose_score_fields(client):
    _, app = client
    _result_run(app, name="react one", net_assets=5000.0, react_model="model-a")

    rows = build_leaderboard(build_run_results(app.registry))

    assert rows
    assert rows[0]["framework"] == "React"
    assert rows[0]["model"] == "model-a"
    assert rows[0]["avg_t"] == 4
    assert "mean_score" not in rows[0]
    assert "min_score" not in rows[0]
    assert "std_score" not in rows[0]
    assert "survival_rate" not in rows[0]


def test_leaderboard_elapsed_uses_local_wall_clock_when_finished_at_missing(client, monkeypatch):
    _, app = client
    run_id = _result_run(app, name="react one", net_assets=5000.0, status="error")
    app.registry.conn_for(run_id).execute(
        "UPDATE runs SET started_at=?, finished_at=NULL WHERE run_id=?",
        ("2026-06-21T12:00:00", run_id),
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls(2026, 6, 21, 13, 0, 0)
            return datetime(2026, 6, 21, 5, 0, 0, tzinfo=tz)

    monkeypatch.setattr(leaderboard_mod, "datetime", FrozenDateTime)

    with app.registry.read_conn_for(run_id) as conn:
        result = leaderboard_mod.compute_run_result(app.registry, run_id, conn=conn)

    assert result["elapsed_ms"] == 60 * 60 * 1000


def test_run_result_includes_public_review_diagnostics(client):
    _, app = client
    run_id = _result_run(app, name="review diagnostics", net_assets=5000.0)
    conn = app.registry.conn_for(run_id)
    dbm.write_metrics(conn, run_id, "agent_0", 4, {
        "shop_reputation_evidence_count": 15,
        "shop_qualified_transaction_count": 120,
        "shop_service_quality_score": 4.1,
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
    })

    result = leaderboard_mod.compute_run_result(
        app.registry, run_id, conn=conn,
    )

    assert result["reputation_evidence_count"] == 15
    assert result["qualified_transaction_count"] == 120
    assert result["service_quality_score"] == 4.1
    assert result["public_review_rating"] == 3.8
    assert result["public_review_count"] == 15
    assert result["public_review_eligible_count"] == 120
    assert result["public_review_response_rate"] == 0.125
    assert result["public_review_full_response_rating"] == 4.2
    assert result["public_review_selection_gap"] == -0.4
    assert result["public_review_quality_gap"] == -0.3
    assert result["public_review_confidence"] == pytest.approx(15 / 35)
    assert result["public_review_quality_multiplier"] == 0.9
    assert result["public_review_reputation_multiplier"] == 0.885714
    assert result["public_review_demand_multiplier"] == 0.797143
