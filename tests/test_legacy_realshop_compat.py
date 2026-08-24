"""Compatibility contract for submissions and runs using the former name."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

from agent.sdk.merchantbench_tool_client import (
    MerchantBenchToolClient,
    RealShopToolClient,
)
from agent.sdk.realshop_tool_client import RealShopToolClient as ShimClient
from compat import (
    ENV_TOOL_ORIGIN,
    canonical_tool_origin,
    is_api_failed_event,
    is_env_tool_origin,
)
from data import daily_reports, private_real
from env import run as env_run
from storage import agent_log
from tools import tools as tool_impl
from web.app import create_app
from web.leaderboard import _is_merchantbench_env_tool_call
from web.runner import load_default_scenario


def _scenario():
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 2
    scenario["run"]["max_hook_seconds"] = 2
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 20
    scenario.setdefault("agent", {})["tool_denylist"] = []
    return scenario


def test_legacy_names_canonicalize_only_at_the_boundary():
    assert canonical_tool_origin("realshop_env") == ENV_TOOL_ORIGIN
    assert is_env_tool_origin("realshop_env")
    assert is_env_tool_origin("merchantbench_env")
    assert is_api_failed_event("realshop_api_failed_attempt")
    assert is_api_failed_event("merchantbench_api_failed_attempt")


def test_legacy_sdk_imports_and_token_env(monkeypatch):
    monkeypatch.setattr(MerchantBenchToolClient, "refresh_schema", lambda self: None)
    monkeypatch.delenv("MERCHANTBENCH_AGENT_TOKEN", raising=False)
    monkeypatch.setenv("REALSHOP_AGENT_TOKEN", "legacy-token")

    client = ShimClient("http://example.test", "run-1", "agent_0")

    assert RealShopToolClient is MerchantBenchToolClient
    assert ShimClient is MerchantBenchToolClient
    assert client._session.headers["Authorization"] == "Bearer legacy-token"


def test_legacy_origin_executes_and_new_trace_is_canonical(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "legacy.db"),
        runs_root=str(tmp_path / "runs"),
    )
    with app.test_client() as client:
        run_id = client.post("/runs", json={"scenario": _scenario()}).get_json()["run_id"]
        env = app.registry._require(run_id)
        with env.hook_cond:
            env.hook_open = True
            env.hook_cond.notify_all()
        response = client.post(
            f"/runs/{run_id}/agents/agent_0/act",
            json={"messages": [{
                "role": "assistant",
                "content": "legacy request",
                "tool_calls": [{
                    "id": "legacy-balance",
                    "type": "function",
                    "tool_origin": "realshop_env",
                    "function": {"name": "query_balance", "arguments": "{}"},
                }],
            }]},
        )
        assert response.status_code == 200
        assert response.get_json()["tool_results"][0]["tool_origin"] == ENV_TOOL_ORIGIN
        trace = agent_log.read_step_index(app.registry.runs_root, run_id, 0)
        assert trace["messages"][0]["tool_calls"][0]["tool_origin"] == ENV_TOOL_ORIGIN


def test_schema_exposes_protocol_and_run_fingerprints(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "schema.db"),
        runs_root=str(tmp_path / "runs"),
    )
    with app.test_client() as client:
        run_id = client.post("/runs", json={"scenario": _scenario()}).get_json()["run_id"]
        schema = client.get(f"/runs/{run_id}/tools/schema").get_json()
        assert schema["protocol"] == {
            "name": "merchantbench",
            "version": 2,
            "legacy_input_names": ["realshop"],
        }
        assert len(schema["tool_schema_sha256"]) == 64
        assert schema["scenario_id"] == _scenario()["scenario_id"]
        assert schema["dataset"]["id"] == "synthetic-v1"
        meta = json.loads((tmp_path / "runs" / run_id / "meta.json").read_text())
        assert meta["protocol_version"] == 2
        assert meta["tool_schema_sha256"] == schema["tool_schema_sha256"]


def test_legacy_auth_header_is_accepted(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "auth.db"),
        runs_root=str(tmp_path / "runs"),
    )
    app.config["MERCHANTBENCH_REQUIRE_TOKENS"] = True
    app.config["MERCHANTBENCH_ADMIN_TOKEN"] = "admin"
    with app.test_client() as client:
        created = client.post(
            "/runs",
            json={"scenario": _scenario()},
            headers={"Authorization": "Bearer admin"},
        ).get_json()
        response = client.get(
            f"/runs/{created['run_id']}/tools/schema",
            headers={"X-RealShop-Token": created["agent_token"]},
        )
        assert response.status_code == 200


def test_legacy_server_auth_environment_is_accepted(monkeypatch, tmp_path):
    monkeypatch.delenv("MERCHANTBENCH_REQUIRE_TOKENS", raising=False)
    monkeypatch.delenv("MERCHANTBENCH_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("REALSHOP_REQUIRE_TOKENS", "true")
    monkeypatch.setenv("REALSHOP_ADMIN_TOKEN", "legacy-admin")

    app = create_app(
        db_path=str(tmp_path / "legacy-auth.db"),
        runs_root=str(tmp_path / "runs"),
    )

    assert app.config["MERCHANTBENCH_REQUIRE_TOKENS"] is True
    assert app.config["MERCHANTBENCH_ADMIN_TOKEN"] == "legacy-admin"


def test_legacy_flask_config_keys_still_enforce_auth(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "legacy-config-auth.db"),
        runs_root=str(tmp_path / "runs"),
    )
    app.config["REALSHOP_REQUIRE_TOKENS"] = True
    app.config["REALSHOP_ADMIN_TOKEN"] = "legacy-admin"

    with app.test_client() as client:
        assert client.get("/runs").status_code == 401
        assert client.get(
            "/runs",
            headers={"Authorization": "Bearer legacy-admin"},
        ).status_code == 200


def test_legacy_trace_and_event_names_are_read_as_environment_activity():
    message = {"tool_origin": "realshop_env"}
    call = {"function": {"name": "query_balance"}}
    assert _is_merchantbench_env_tool_call(message, call)


def test_memory_version_continues_after_legacy_marker(tmp_path):
    env = SimpleNamespace(runs_root=str(tmp_path), run_id="run-memory", t=12)
    history_dir = tmp_path / "run-memory" / "agent" / "memory"
    history_dir.mkdir(parents=True)
    history = history_dir / "agent_0.history.md"
    history.write_text(
        "<!-- realshop-memory-version 1 -->\n## Memory version 1\n",
        encoding="utf-8",
    )

    tool_impl._append_memory_history(env, "agent_0", "new content")

    text = history.read_text(encoding="utf-8")
    assert "<!-- merchantbench-memory-version 2 -->" in text


def test_missing_legacy_private_paths_remap_to_configured_root(monkeypatch, tmp_path):
    private_root = tmp_path / "private_data"
    private_root.mkdir()
    monkeypatch.setenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", str(private_root))

    dataset = private_real.resolve_dataset_path(
        "/old/realshop-dev/env/data/private_data/catalog.sqlite"
    )
    reports = daily_reports.resolve_report_dir(
        "/old/realshop-dev/env/data/private_data/daily_reports"
    )

    assert dataset == str(private_root / "catalog.sqlite")
    assert reports == str(private_root / "daily_reports")


def test_empty_canonical_environment_value_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", "")
    monkeypatch.setenv("REALSHOP_PRIVATE_DATA_ROOT", "/legacy/private-data")

    # * os.path.join keeps the assertion platform-correct (Windows uses "\\").
    assert private_real.resolve_dataset_path() == os.path.join(
        "/legacy/private-data", "private_real_1k.sqlite"
    )


def test_private_root_overrides_repo_relative_data_paths(monkeypatch, tmp_path):
    private_root = tmp_path / "overlay"
    private_root.mkdir()
    monkeypatch.setenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", str(private_root))

    assert private_real.resolve_dataset_path(
        "data/private_data/catalog.sqlite"
    ) == str(private_root / "catalog.sqlite")
    assert daily_reports.resolve_report_dir(
        "data/private_data/daily_reports"
    ) == str(private_root / "daily_reports")


def test_env_entrypoint_loads_repo_dotenv_without_overriding_exports(
    monkeypatch, tmp_path
):
    env_dir = tmp_path / "checkout/env"
    env_dir.mkdir(parents=True)
    dotenv_path = tmp_path / "checkout/.env"
    dotenv_path.write_text(
        "MERCHANTBENCH_PRIVATE_DATA_ROOT=/from-dotenv\n"
        "MERCHANTBENCH_ADMIN_TOKEN=dotenv-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(env_run, "__file__", str(env_dir / "run.py"))
    monkeypatch.delenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", raising=False)
    monkeypatch.setenv("MERCHANTBENCH_ADMIN_TOKEN", "exported-token")

    loaded = env_run._load_local_dotenv()

    assert loaded == str(dotenv_path)
    assert os.environ["MERCHANTBENCH_PRIVATE_DATA_ROOT"] == "/from-dotenv"
    assert os.environ["MERCHANTBENCH_ADMIN_TOKEN"] == "exported-token"
