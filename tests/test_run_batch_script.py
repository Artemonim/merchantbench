import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/run_batch.py")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_batch", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_queue_path_is_batch_queue():
    mod = _load_script()

    assert mod.DEFAULT_QUEUE_PATH.name == "batch_queue.yaml"


def test_load_queue_config_reads_yaml_defaults_and_enabled_models(tmp_path):
    mod = _load_script()
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        """
scenario_path: env/scenarios/default.yaml
bootstrap_agent: react_160k_compact_30k
detailed: true
days: 14
poll_seconds: 9
queue:
  - model: qwen3.7-max
  - model: bailian/deepseek-v4-pro
    detailed: false
# - model: qwen3.7-plus
""",
        encoding="utf-8",
    )

    config = mod.load_queue_config(queue_path)
    jobs = mod.jobs_from_config(config)

    assert config["scenario_path"] == "env/scenarios/default.yaml"
    assert config["bootstrap_agent"] == "react_160k_compact_30k"
    assert config["days"] == 14
    assert config["seed"] == 42
    assert config["poll_seconds"] == 9
    assert [job["model"] for job in jobs] == [
        "qwen3.7-max",
        "bailian/deepseek-v4-pro",
    ]
    assert [job["detailed"] for job in jobs] == [True, False]
    assert all(
        job["scenario_path"] == "env/scenarios/default.yaml"
        for job in jobs
    )
    assert all(job["days"] == 14 for job in jobs)
    assert all(job["seed"] == 42 for job in jobs)


def test_jobs_from_config_defaults_batch_days_to_90():
    mod = _load_script()
    jobs = mod.jobs_from_config({
        "scenario_path": "env/scenarios/default.yaml",
        "queue": [{"model": "qwen3.7-max"}],
    })

    assert jobs[0]["days"] == 90


def test_jobs_from_config_allows_cli_days_override():
    mod = _load_script()
    jobs = mod.jobs_from_config(
        {
            "scenario_path": "env/scenarios/default.yaml",
            "days": 14,
            "queue": [{"model": "qwen3.7-max", "days": 7}],
        },
        days_override=30,
    )

    assert jobs[0]["days"] == 30


def test_jobs_from_config_seed_precedence_and_default():
    mod = _load_script()
    config = {
        "scenario_path": "env/scenarios/default.yaml",
        "seed": 123,
        "queue": [
            {"model": "qwen3.7-max"},
            {"model": "bailian/deepseek-v4-pro", "seed": 1337},
        ],
    }

    jobs = mod.jobs_from_config(config)
    overridden = mod.jobs_from_config(config, seed_override=7)
    defaults = mod.jobs_from_config({
        "scenario_path": "env/scenarios/default.yaml",
        "queue": [{"model": "qwen3.7-max"}],
    })

    assert [job["seed"] for job in jobs] == [123, 1337]
    assert [job["seed"] for job in overridden] == [7, 7]
    assert defaults[0]["seed"] == 42


def test_jobs_from_config_rejects_non_integer_seed():
    mod = _load_script()

    with pytest.raises(ValueError, match="seed must be an integer"):
        mod.jobs_from_config({
            "scenario_path": "env/scenarios/default.yaml",
            "seed": "not-a-seed",
            "queue": [{"model": "qwen3.7-max"}],
        })


def test_jobs_from_config_supports_rule_based_random_seed_matrix():
    mod = _load_script()
    jobs = mod.jobs_from_config({
        "scenario_path": "env/scenarios/default.yaml",
        "bootstrap_agent": "rule_based",
        "selection_mode": "random",
        "days": 365,
        "queue": [
            {"seed": 0},
            {"seed": 42},
            {"seed": 123},
        ],
    })

    assert [job["model"] for job in jobs] == ["random", "random", "random"]
    assert [job["bootstrap_agent"] for job in jobs] == [
        "rule_based",
        "rule_based",
        "rule_based",
    ]
    assert [job["seed"] for job in jobs] == [0, 42, 123]
    assert [job["selection_seed"] for job in jobs] == [0, 42, 123]
    assert all(job["days"] == 365 for job in jobs)


def test_jobs_from_config_preserves_per_job_scenario_path():
    mod = _load_script()
    jobs = mod.jobs_from_config({
        "scenario_path": "env/scenarios/default.yaml",
        "bootstrap_agent": "rule_based",
        "selection_mode": "random",
        "days": 7,
        "queue": [
            {"scenario_path": "env/scenarios/ablations/pricing_only.yaml"},
            {"scenario_path": "env/scenarios/ablations/demand_only.yaml"},
            {"scenario_path": "env/scenarios/ablations/both.yaml"},
        ],
    })

    assert [job["scenario_path"] for job in jobs] == [
        "env/scenarios/ablations/pricing_only.yaml",
        "env/scenarios/ablations/demand_only.yaml",
        "env/scenarios/ablations/both.yaml",
    ]
    assert all(job["days"] == 7 for job in jobs)
    assert all(job["seed"] == 42 for job in jobs)


def test_load_queue_config_reads_rule_ablations_yaml():
    mod = _load_script()
    config = mod.load_queue_config(
        mod.ROOT / "scripts" / "batch_queue_rule_ablations.yaml"
    )
    jobs = mod.jobs_from_config(config)

    assert config["bootstrap_agent"] == "rule_based"
    assert config["selection_mode"] == "random"
    assert config["days"] == 7
    assert len(jobs) == 3
    assert [job["scenario_path"] for job in jobs] == [
        "env/scenarios/ablations/pricing_only.yaml",
        "env/scenarios/ablations/demand_only.yaml",
        "env/scenarios/ablations/both.yaml",
    ]
    assert all(job["bootstrap_agent"] == "rule_based" for job in jobs)
    assert all(job["selection_mode"] == "random" for job in jobs)
    assert all(job["days"] == 7 for job in jobs)


def test_load_queue_config_reads_economy_v6_ablations_yaml():
    mod = _load_script()
    config = mod.load_queue_config(
        mod.ROOT / "scripts" / "batch_queue_economy_v6_ablations.yaml"
    )
    jobs = mod.jobs_from_config(config)

    assert config["bootstrap_agent"] == "rule_based"
    assert config["selection_mode"] == "random"
    assert config["days"] == 7
    assert len(jobs) == 3
    assert [job["scenario_path"] for job in jobs] == [
        "env/scenarios/ablations/economy_v6_fees_only.yaml",
        "env/scenarios/ablations/economy_v6_refund_only.yaml",
        "env/scenarios/ablations/economy_v6_both.yaml",
    ]
    assert all(job["bootstrap_agent"] == "rule_based" for job in jobs)
    assert all(job["selection_mode"] == "random" for job in jobs)
    assert all(job["days"] == 7 for job in jobs)


def test_load_queue_config_reads_v5_model_goal_yaml():
    mod = _load_script()
    config = mod.load_queue_config(
        mod.ROOT / "scripts" / "batch_queue_hermes_v5_30d_x4_model_goal.yaml"
    )
    jobs = mod.jobs_from_config(config)

    assert config["bootstrap_agent"] == "hermes"
    assert config["days"] == 30
    assert config["max_parallel"] == 4
    assert len(jobs) == 4
    assert [job["model"] for job in jobs] == [
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-flash-0731",
        "google/gemini-3.7-flash",
        "google/gemini-3.7-flash",
    ]
    assert [job["scenario_path"] for job in jobs] == [
        "env/scenarios/agents/hermes.yaml",
        "env/scenarios/agents/hermes_bankrupt.yaml",
        "env/scenarios/agents/hermes_gemini.yaml",
        "env/scenarios/agents/hermes_gemini_bankrupt.yaml",
    ]
    assert all(job["days"] == 30 for job in jobs)
    assert all(job["seed"] == 42 for job in jobs)
    assert all(job["bootstrap_agent"] == "hermes" for job in jobs)


def test_load_queue_config_reads_hermes_v6_red_7d_yaml():
    mod = _load_script()
    config = mod.load_queue_config(
        mod.ROOT / "scripts" / "batch_queue_hermes_v6_red_7d_x3.yaml"
    )
    jobs = mod.jobs_from_config(config)

    assert config["bootstrap_agent"] == "hermes"
    assert config["days"] == 7
    assert config["max_parallel"] == 3
    assert len(jobs) == 3
    assert [job["model"] for job in jobs] == ["stealth/ox-alpha"] * 3
    assert [job["scenario_path"] for job in jobs] == [
        "env/scenarios/agents/hermes_v6_red_unrestricted.yaml",
        "env/scenarios/agents/hermes_v6_red_bad_merchant.yaml",
        "env/scenarios/agents/hermes_v6_red_bad_economics.yaml",
    ]
    assert all(job["days"] == 7 for job in jobs)
    assert all(job["seed"] == 42 for job in jobs)
    assert all(job["bootstrap_agent"] == "hermes" for job in jobs)


def test_create_run_payload_wires_oxalpha_red_scenario(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(_method, _base_url, _path, body=None):
        captured["body"] = body
        return {"run_id": "run-oxalpha-red"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    run_id = mod.create_run(
        "http://env.test",
        {
            "model": "stealth/ox-alpha",
            "scenario_path": "env/scenarios/agents/hermes_v6_red_unrestricted.yaml",
            "bootstrap_agent": "hermes",
            "days": 7,
            "seed": 42,
            "name": "hermes-oxalpha-7d-v6-red-unrestricted-seed-42",
        },
    )

    assert run_id == "run-oxalpha-red"
    body = captured["body"]
    assert body["bootstrap_agent"] == "hermes"
    assert body["bootstrap_config"] == {"react_model": "stealth/ox-alpha"}
    assert body["scenario"]["run"]["horizon_steps"] == 7 * 24
    assert body["scenario"]["economy_v6"]["enabled"] is True
    assert body["scenario"]["data"]["source"] == "private_real"
    assert body["scenario"]["agent"]["hermes"]["reasoning_effort"] == "xhigh"
    assert body["scenario"]["agent"]["hermes"]["provider_routing"] == {}
    assert body["scenario"]["agent"]["cost_pricing"] == pytest.approx({
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "cached_input_per_million": 0.0,
    })
    assert any(
        "bankruptcy" in goal.lower()
        for goal in body["scenario"]["agent"]["goals"]["en"]
    )


def test_create_run_payload_uses_queue_job_and_builtin_pricing(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(method, base_url, path, body=None):
        captured.update({
            "method": method,
            "base_url": base_url,
            "path": path,
            "body": body,
        })
        return {"run_id": "run-qwen"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    run_id = mod.create_run(
        "http://env.test",
        {
            "model": "qwen3.7-max",
            "scenario_path": "env/scenarios/default.yaml",
            "bootstrap_agent": "react_160k_compact_30k",
            "detailed": True,
            "seed": 123,
        },
    )

    assert run_id == "run-qwen"
    assert captured["method"] == "POST"
    assert captured["base_url"] == "http://env.test"
    assert captured["path"] == "/runs"
    body = captured["body"]
    assert body["bootstrap_agent"] == "react_160k_compact_30k"
    assert body["bootstrap_config"] == {"react_model": "qwen3.7-max"}
    assert body["auto_start"] is True
    assert body["master_seed"] == 123
    assert body["scenario"]["run"]["master_seed"] == 123
    assert body["scenario"]["run"]["horizon_steps"] == 90 * 24
    assert body["scenario"]["agent"]["detailed"] is True
    assert body["scenario"]["agent"]["cost_pricing"] == pytest.approx({
        "input_per_million": 1.66,
        "output_per_million": 4.97,
        "cached_input_per_million": 0.17,
    })


def test_create_run_payload_allows_hermes_bootstrap_agent(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(method, base_url, path, body=None):
        captured.update({
            "method": method,
            "base_url": base_url,
            "path": path,
            "body": body,
        })
        return {"run_id": "run-hermes"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    run_id = mod.create_run(
        "http://env.test",
        {
            "model": "qwen3.7-max",
            "scenario_path": "env/scenarios/default.yaml",
            "bootstrap_agent": "hermes",
        },
    )

    assert run_id == "run-hermes"
    assert captured["method"] == "POST"
    assert captured["path"] == "/runs"
    body = captured["body"]
    assert body["bootstrap_agent"] == "hermes"
    assert body["bootstrap_config"] == {"react_model": "qwen3.7-max"}
    assert body["auto_start"] is True
    assert body["name"] == "hermes-qwen3.7-max"
    assert body["master_seed"] == 42
    assert body["scenario"]["run"]["master_seed"] == 42
    assert body["scenario"]["agent"]["tool_denylist"] == [
        "market_brief",
        "hot_search_terms",
        "read_memory_doc",
        "write_memory_doc",
    ]


def test_create_run_payload_keeps_bankrupt_goals_and_gemini_pricing(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(method, base_url, path, body=None):
        captured["body"] = body
        return {"run_id": "run-gemini-bankrupt"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    run_id = mod.create_run(
        "http://env.test",
        {
            "model": "google/gemini-3.7-flash",
            "scenario_path": "env/scenarios/agents/hermes_gemini_bankrupt.yaml",
            "bootstrap_agent": "hermes",
            "days": 30,
            "seed": 42,
            "name": "hermes-gemini37-30d-v5-bankrupt-seed-42",
        },
    )

    assert run_id == "run-gemini-bankrupt"
    body = captured["body"]
    assert body["name"] == "hermes-gemini37-30d-v5-bankrupt-seed-42"
    assert body["bootstrap_agent"] == "hermes"
    assert body["bootstrap_config"] == {"react_model": "google/gemini-3.7-flash"}
    assert body["scenario"]["run"]["horizon_steps"] == 30 * 24
    assert any(
        "bankruptcy" in goal.lower()
        for goal in body["scenario"]["agent"]["goals"]["en"]
    )
    assert body["scenario"]["agent"]["hermes"]["provider_routing"]["only"] == [
        "google-vertex/global",
    ]
    assert body["scenario"]["agent"]["cost_pricing"] == pytest.approx({
        "input_per_million": 0.375,
        "output_per_million": 1.875,
        "cached_input_per_million": 0.0375,
    })


def test_create_run_payload_supports_rule_based_random(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(method, base_url, path, body=None):
        captured.update({
            "method": method,
            "base_url": base_url,
            "path": path,
            "body": body,
        })
        return {"run_id": "run-rule-random"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    run_id = mod.create_run(
        "http://env.test",
        {
            "scenario_path": "env/scenarios/default.yaml",
            "bootstrap_agent": "rule_based",
            "selection_mode": "random",
            "seed": 123,
            "selection_seed": 123,
            "days": 30,
        },
    )

    assert run_id == "run-rule-random"
    body = captured["body"]
    assert body["name"] == "rule_based-random-seed-123"
    assert body["master_seed"] == 123
    assert body["scenario"]["run"]["master_seed"] == 123
    assert body["scenario"]["run"]["horizon_steps"] == 30 * 24
    assert body["bootstrap_agent"] == "rule_based"
    assert body["bootstrap_config"] == {
        "selection_mode": "random",
        "selection_seed": 123,
    }


def test_hermes_bootstrap_preserves_custom_scenario_path(monkeypatch, tmp_path):
    mod = _load_script()
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        f"""
extends: {mod.ROOT / "env/scenarios/default.yaml"}
agent:
  tool_denylist:
    - get_daily_report
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_request_json(_method, _base_url, _path, body=None):
        captured["body"] = body
        return {"run_id": "run-custom-hermes"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    mod.create_run(
        "http://env.test",
        {
            "model": "qwen3.7-max",
            "scenario_path": str(custom),
            "bootstrap_agent": "hermes",
        },
    )

    assert captured["body"]["scenario"]["agent"]["tool_denylist"] == [
        "get_daily_report",
    ]


def test_create_run_payload_converts_days_to_scenario_steps(monkeypatch):
    mod = _load_script()
    captured = {}

    def fake_request_json(_method, _base_url, _path, body=None):
        captured["body"] = body
        return {"run_id": "run-14d"}

    monkeypatch.setattr(mod, "request_json", fake_request_json)

    mod.create_run(
        "http://env.test",
        {
            "model": "qwen3.7-max",
            "scenario_path": "env/scenarios/default.yaml",
            "bootstrap_agent": "react_160k_compact_30k",
            "days": 14,
        },
    )

    assert captured["body"]["scenario"]["run"]["horizon_steps"] == 14 * 24


def test_react_model_pricing_includes_current_model_presets():
    mod = _load_script()

    expected = {
        "gpt-5.6-sol": (5.00, 30.00, 0.50),
        "bailian/deepseek-v4-flash": (0.14, 0.28, 0.03),
        "qwen3.7-plus": (0.28, 1.10, 0.06),
        "gemini-3.5-flash": (1.50, 9.00, 0.15),
        "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
        "bailian/kimi-k2.6": (0.90, 3.75, 0.15),
        "moonshot/kimi-k3": (3.00, 15.00, 0.30),
        "bailian/glm-5.2": (1.10, 3.87, 0.22),
        "claude-sonnet-4-6": (3.00, 15.00, 0.30),
        "claude-sonnet-5": (2.00, 10.00, 0.20),
        "claude-opus-4-7": (5.00, 25.00, 0.50),
        "claude-opus-4-8": (5.00, 25.00, 0.50),
        "deepseek/deepseek-v4-flash-0731": (0.13, 0.28, 0.07),
        "google/gemini-3.7-flash": (0.375, 1.875, 0.0375),
        "stealth/ox-alpha": (0.0, 0.0, 0.0),
    }

    for model, (input_price, output_price, cached_input_price) in expected.items():
        pricing = mod.REACT_MODEL_PRICING_BY_MODEL[model]
        assert {
            "input": pricing["input"],
            "output": pricing["output"],
            "cached_input": pricing["cached_input"],
        } == pytest.approx({
            "input": input_price,
            "output": output_price,
            "cached_input": cached_input_price,
        })


def test_run_models_bounds_parallelism_and_keeps_result_order(monkeypatch):
    mod = _load_script()
    events = []
    polls = {}

    def fake_create_run(base_url, job):
        events.append(("create", job["model"], job["bootstrap_agent"]))
        return f"run-{job['model']}"

    def fake_get_run_status(base_url, run_id):
        polls[run_id] = polls.get(run_id, 0) + 1
        events.append(("poll", run_id))
        if run_id.endswith("qwen3.7-max") and polls[run_id] < 2:
            return {"state": "running", "phase": "running", "t": 1}
        return {"state": "finished", "phase": "finished", "t": 2}

    monkeypatch.setattr(mod, "create_run", fake_create_run)
    monkeypatch.setattr(mod, "get_run_status", fake_get_run_status)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    jobs = [
        {"model": "qwen3.7-max", "bootstrap_agent": "react_160k_compact_30k"},
        {"model": "bailian/deepseek-v4-pro", "bootstrap_agent": "react_160k_compact_30k"},
        {"model": "bailian/glm-5.2", "bootstrap_agent": "react_160k_compact_30k"},
    ]
    results = mod.run_models(
        jobs, base_url="http://env.test", poll_seconds=0, max_parallel=2
    )

    third_create = events.index(("create", "bailian/glm-5.2", "react_160k_compact_30k"))
    assert ("poll", "run-bailian/deepseek-v4-pro") in events[:third_create]
    assert [model for model, _run_id, _status in results] == [
        "qwen3.7-max",
        "bailian/deepseek-v4-pro",
        "bailian/glm-5.2",
    ]


def test_resolve_max_parallel_precedence(monkeypatch):
    mod = _load_script()
    monkeypatch.setenv("MAX_PARALLEL", "3")

    assert mod.resolve_max_parallel(4, {"max_parallel": 2}) == 4
    assert mod.resolve_max_parallel(None, {"max_parallel": 2}) == 3
    monkeypatch.delenv("MAX_PARALLEL")
    assert mod.resolve_max_parallel(None, {"max_parallel": 2}) == 2
    assert mod.resolve_max_parallel(None, {}) == 1


def test_stopped_run_is_failure_even_with_finished_phase_and_queue_continues(monkeypatch):
    mod = _load_script()
    created = []

    def fake_create_run(_base_url, job):
        created.append(job["model"])
        return f"run-{job['model']}"

    def fake_status(_base_url, run_id):
        if run_id == "run-first":
            return {"state": "stopped", "phase": "finished", "t": 5}
        return {"state": "finished", "phase": "finished", "t": 6}

    monkeypatch.setattr(mod, "create_run", fake_create_run)
    monkeypatch.setattr(mod, "get_run_status", fake_status)
    jobs = [
        {"model": "first", "bootstrap_agent": "react_160k_compact_30k"},
        {"model": "second", "bootstrap_agent": "react_160k_compact_30k"},
        {"model": "third", "bootstrap_agent": "react_160k_compact_30k"},
    ]

    with pytest.raises(mod.BatchRunError) as exc_info:
        mod.run_models(jobs, base_url="http://env.test", poll_seconds=0, max_parallel=2)

    assert created == ["first", "second", "third"]
    assert exc_info.value.failures[0]["model"] == "first"
    assert [result[0] for result in exc_info.value.results] == ["second", "third"]


def test_create_run_rejects_legacy_react_bootstrap_agent():
    mod = _load_script()

    with pytest.raises(ValueError, match="unsupported bootstrap_agent"):
        mod.create_run(
            "http://env.test",
            {
                "model": "qwen3.7-max",
                "scenario_path": "env/scenarios/default.yaml",
                "bootstrap_agent": "react",
            },
        )
