import importlib
import os
import sys
import types


def _import_run_eval(monkeypatch):
    docker_mod = types.ModuleType("docker")
    docker_errors = types.ModuleType("docker.errors")

    class APIError(Exception):
        pass

    class ImageNotFound(Exception):
        pass

    class NotFound(Exception):
        pass

    docker_errors.APIError = APIError
    docker_errors.ImageNotFound = ImageNotFound
    docker_errors.NotFound = NotFound
    docker_mod.errors = docker_errors
    docker_mod.from_env = lambda: None
    monkeypatch.setitem(sys.modules, "docker", docker_mod)
    monkeypatch.setitem(sys.modules, "docker.errors", docker_errors)
    sys.modules.pop("eval.run_eval", None)
    return importlib.import_module("eval.run_eval")


def test_agent_env_ignores_merchantbench_values_from_env_file(monkeypatch):
    run_eval = _import_run_eval(monkeypatch)

    env = run_eval._build_agent_env(
        env_name="env-container",
        run_id="run-fresh",
        agent_id="agent_0",
        agent_token="fresh-token",
        creds={
            "OPENAI_API_KEY": "sk-test",
            "MODEL_NAME": "test-model",
            "MERCHANTBENCH_BASE_URL": "http://wrong",
            "MERCHANTBENCH_RUN_ID": "old-run",
            "MERCHANTBENCH_AGENT_ID": "agent_9",
            "MERCHANTBENCH_AGENT_TOKEN": "stale-token",
            "REALSHOP_RUN_ID": "older-run",
            "RSH_SCENARIO": "wrong-scenario",
        },
    )

    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["MODEL_NAME"] == "test-model"
    assert env["MERCHANTBENCH_BASE_URL"] == "http://env-container:5000"
    assert env["MERCHANTBENCH_RUN_ID"] == "run-fresh"
    assert env["MERCHANTBENCH_AGENT_ID"] == "agent_0"
    assert env["MERCHANTBENCH_AGENT_TOKEN"] == "fresh-token"
    assert env["REALSHOP_BASE_URL"] == "http://env-container:5000"
    assert env["REALSHOP_RUN_ID"] == "run-fresh"
    assert env["REALSHOP_AGENT_ID"] == "agent_0"
    assert env["REALSHOP_AGENT_TOKEN"] == "fresh-token"
    assert "RSH_SCENARIO" not in env


def test_private_data_root_is_mounted_read_only_for_env_container(monkeypatch, tmp_path):
    run_eval = _import_run_eval(monkeypatch)
    monkeypatch.delenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", raising=False)
    monkeypatch.delenv("REALSHOP_PRIVATE_DATA_ROOT", raising=False)

    root = run_eval._resolve_private_data_root(
        {
            "MERCHANTBENCH_PRIVATE_DATA_ROOT": str(tmp_path),
        }
    )
    environment, volumes = run_eval._build_env_container_config(
        admin_token="admin-token",
        private_data_root=root,
    )

    assert root == os.path.realpath(tmp_path)
    assert environment["MERCHANTBENCH_PRIVATE_DATA_ROOT"] == "/merchantbench-private-data"
    assert environment["REALSHOP_PRIVATE_DATA_ROOT"] == "/merchantbench-private-data"
    assert volumes == {
        root: {"bind": "/merchantbench-private-data", "mode": "ro"},
    }


def test_private_data_root_rejects_missing_directory(monkeypatch, tmp_path):
    run_eval = _import_run_eval(monkeypatch)
    monkeypatch.delenv("MERCHANTBENCH_PRIVATE_DATA_ROOT", raising=False)
    monkeypatch.delenv("REALSHOP_PRIVATE_DATA_ROOT", raising=False)

    missing = tmp_path / "missing"
    try:
        run_eval._resolve_private_data_root(
            {
                "MERCHANTBENCH_PRIVATE_DATA_ROOT": str(missing),
            }
        )
    except SystemExit as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing private data root should fail before Docker startup")


def test_eval_scenario_loader_resolves_relative_extends_and_deep_merges(monkeypatch, tmp_path):
    run_eval = _import_run_eval(monkeypatch)
    scenario_dir = tmp_path / "env/scenarios"
    profile_dir = scenario_dir / "profiles"
    profile_dir.mkdir(parents=True)
    (scenario_dir / "default.yaml").write_text(
        "run:\n  horizon_steps: 24\n  step_hours: 1\n"
        "agent:\n  language: en\n  cost_pricing:\n    input: 5\n    output: 30\n",
        encoding="utf-8",
    )
    (profile_dir / "child.yaml").write_text(
        "extends: ../default.yaml\nagent:\n  language: zh\n  cost_pricing:\n    output: 12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_eval, "_REPO_ROOT", str(tmp_path))

    scenario = run_eval._load_scenario("profiles/child")

    assert "extends" not in scenario
    assert scenario["run"] == {"horizon_steps": 24, "step_hours": 1}
    assert scenario["agent"] == {
        "language": "zh",
        "cost_pricing": {"input": 5, "output": 12},
    }
