import importlib
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
        },
    )

    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["MODEL_NAME"] == "test-model"
    assert env["MERCHANTBENCH_BASE_URL"] == "http://env-container:5000"
    assert env["MERCHANTBENCH_RUN_ID"] == "run-fresh"
    assert env["MERCHANTBENCH_AGENT_ID"] == "agent_0"
    assert env["MERCHANTBENCH_AGENT_TOKEN"] == "fresh-token"
