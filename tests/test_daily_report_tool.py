import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from storage import agent_log
from tools import registry
from tools import tools as tools_mod
from tools.dispatch import dispatch_tool
from web.app import create_app
from web.runner import load_default_scenario


def _scenario(report_dir):
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 48
    scenario["run"]["max_hook_seconds"] = 0.01
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-01",
    }
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 20
    scenario["data"]["daily_report_dir"] = str(report_dir)
    scenario.setdefault("agent", {})["tool_denylist"] = []
    return scenario


def _env_for(tmp_path, report_dir):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()
    run_id = client.post("/runs", json={"scenario": _scenario(report_dir)}).get_json()["run_id"]
    return app.registry._require(run_id)


def test_get_daily_report_schema_has_no_parameters():
    spec = registry.get("get_daily_report")

    assert spec is not None
    assert spec.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert spec.handler is tools_mod.get_daily_report
    assert spec.mutating is False
    assert spec.description == (
        "Return the merchant business opportunity daily report published for the current "
        "simulation date. The response includes report_date, data_as_of (the previous "
        "simulation date), and Markdown content with recent market news, category momentum, "
        "and keyword opportunity signals. "
    )


def test_get_daily_report_returns_current_simulation_date_content(tmp_path):
    report_dir = tmp_path / "daily_reports"
    report_dir.mkdir()
    (report_dir / "20250602.md").write_text("# second report\nbody", encoding="utf-8")
    env = _env_for(tmp_path, report_dir)
    env.t = 24

    result = tools_mod.get_daily_report(env)

    assert result == {
        "ok": True,
        "report_date": "2025-06-02",
        "data_as_of": "2025-06-01",
        "content": "# second report\nbody",
    }


def test_successful_agent_read_suppresses_and_persists_same_day_notice(tmp_path):
    report_dir = tmp_path / "daily_reports"
    report_dir.mkdir()
    (report_dir / "20250602.md").write_text("# second report\nbody", encoding="utf-8")
    env = _env_for(tmp_path, report_dir)
    env.t = 24

    assert tools_mod.daily_report_notice_available(env, "agent_0") is True
    result = dispatch_tool(env, "agent_0", "get_daily_report", {})

    assert result["ok"] is True
    assert tools_mod.daily_report_notice_available(env, "agent_0") is False
    assert agent_log.load_daily_report_read_dates(env.runs_root, env.run_id) == {
        "agent_0": "2025-06-02",
    }


def test_get_daily_report_returns_error_when_current_date_is_missing(tmp_path):
    report_dir = tmp_path / "daily_reports"
    report_dir.mkdir()
    env = _env_for(tmp_path, report_dir)
    env.t = 24

    result = tools_mod.get_daily_report(env)

    assert result["ok"] is False
    assert result["error"] == "daily report not found for 2025-06-02"


def test_bundled_daily_report_headers_match_publication_date_and_data_cutoff():
    report_dir = Path(tools_mod.daily_reports.DEFAULT_DAILY_REPORT_DIR)
    paths = sorted(report_dir.glob("*.md"))

    if not paths:
        pytest.skip("non-redistributable bundled daily reports are not in the artifact")
    assert len(paths) == 365
    for path in paths:
        report_date = datetime.strptime(path.stem, "%Y%m%d").date()
        data_as_of = report_date - timedelta(days=1)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == f"# 【{report_date.month}月{report_date.day}日 市场商机速递】"
        assert lines[2] == (
            f"> 日报日期：{report_date.year}年{report_date.month}月{report_date.day}日"
            f"｜数据截至：{data_as_of.year}年{data_as_of.month}月{data_as_of.day}日"
        )
