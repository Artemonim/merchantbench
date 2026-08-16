import os
import sys
from pathlib import Path

import pytest
import yaml

from web import runner as runner_mod
from web.runner import RunRegistry


def test_runner_rejects_python_below_project_minimum():
    with pytest.raises(RuntimeError, match="Python 3.10\\+"):
        runner_mod._ensure_supported_python((3, 9, 6))


def test_spawn_compact_react_uses_dedicated_baseline_script(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    baselines_dir = tmp_path / "agent" / "baselines"
    baselines_dir.mkdir(parents=True)
    (baselines_dir / "react_160k_compact_30k.py").write_text("# fake baseline\n")
    monkeypatch.setattr(registry, "_agent_baselines_dir", lambda: str(baselines_dir))

    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_react_160k_compact_30k(
        "run-1",
        "http://127.0.0.1:5050",
        model="qwen-max",
        max_steps=4320,
    )

    cmd = captured["cmd"]
    assert cmd[1].endswith("react_160k_compact_30k.py")
    assert cmd[cmd.index("--model") + 1] == "qwen-max"
    assert cmd[cmd.index("--max-steps") + 1] == "4320"
    assert cmd[cmd.index("--context-window-tokens") + 1] == "160000"
    assert cmd[cmd.index("--compact-trigger-tokens") + 1] == "160000"
    assert cmd[cmd.index("--compact-keep-tokens") + 1] == "30000"
    assert captured["kwargs"]["env"]["MODEL_NAME"] == "qwen-max"


def test_spawn_baseline_passes_agent_token(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    baselines_dir = tmp_path / "agent" / "baselines"
    baselines_dir.mkdir(parents=True)
    (baselines_dir / "auto_seed.py").write_text("# fake baseline\n")
    monkeypatch.setattr(registry, "_agent_baselines_dir", lambda: str(baselines_dir))
    registry._write_auth_for_run("run-1", {
        "agent_token": "agent-0-token",
        "agent_tokens": {"agent_0": "agent-0-token"},
    })
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_auto_seed("run-1", "http://127.0.0.1:5050", {
        "run": {"auto_seed_count": 1, "horizon_steps": 10}
    })

    assert captured["kwargs"]["env"]["MERCHANTBENCH_AGENT_TOKEN"] == "agent-0-token"


def test_spawn_rule_based_passes_mode_seed_and_count(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    baselines_dir = tmp_path / "agent" / "baselines"
    baselines_dir.mkdir(parents=True)
    (baselines_dir / "rule_based.py").write_text("# fake baseline\n")
    monkeypatch.setattr(registry, "_agent_baselines_dir", lambda: str(baselines_dir))
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_rule_based(
        "run-1",
        "http://127.0.0.1:5050",
        {"run": {"rule_based_count": 7, "horizon_steps": 10}},
        selection_mode="random",
        selection_seed=42,
    )

    cmd = captured["cmd"]
    assert cmd[1].endswith("rule_based.py")
    assert cmd[cmd.index("--selection-mode") + 1] == "random"
    assert cmd[cmd.index("--selection-seed") + 1] == "42"
    assert cmd[cmd.index("--seed-count") + 1] == "7"


def test_spawn_hermes_uses_external_adapter_repo(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    registry._write_auth_for_run("run-1", {
        "agent_token": "agent-0-token",
        "agent_tokens": {"agent_0": "agent-0-token"},
    })
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_hermes(
        "run-1",
        "http://127.0.0.1:5050",
        model="qwen-max",
        max_steps=4320,
    )

    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "merchantbench_adapter"]
    assert cmd[cmd.index("--model") + 1] == "qwen-max"
    assert cmd[cmd.index("--max-steps") + 1] == "4320"
    assert cmd[cmd.index("--max-hops-per-step") + 1] == "30"
    assert captured["kwargs"]["cwd"] == str(hermes_root)
    env = captured["kwargs"]["env"]
    assert env["MODEL_NAME"] == "qwen-max"
    assert env["MERCHANTBENCH_AGENT_TOKEN"] == "agent-0-token"
    assert env["MERCHANTBENCH_AGENT_SDK_ROOT"] == str(
        Path(__file__).resolve().parents[1] / "agent"
    )
    assert env["MERCHANTBENCH_AGENT_SDK_ROOT"] in env["PYTHONPATH"].split(os.pathsep)
    log_path = captured["kwargs"]["stderr"].name
    assert log_path.endswith(os.path.join("runs", "run-1", "agent", "bootstrap_hermes.log"))


def test_spawn_hermes_uses_run_local_home_and_copies_official_skills(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    skill_dir = hermes_root / "skills" / "software-development" / "plan"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: plan\ndescription: planning\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://idealab.example/v1")
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_hermes("run-1", "http://127.0.0.1:5050")

    agent_dir = tmp_path / "runs" / "run-1" / "agent"
    hermes_home = agent_dir / "hermes_home"
    hermes_workspace = agent_dir / "hermes_workspace"
    assert captured["kwargs"]["env"]["HERMES_HOME"] == str(hermes_home)
    assert captured["kwargs"]["env"]["TERMINAL_CWD"] == str(hermes_workspace)
    assert (hermes_home / "skills" / "software-development" / "plan" / "SKILL.md").exists()
    assert not (hermes_home / "AGENTS.md").exists()
    assert hermes_workspace.is_dir()
    config = yaml.safe_load(
        (hermes_home / "config.yaml").read_text(encoding="utf-8")
    )
    assert config["model"] == {
        "context_length": 262144,
        "max_tokens": 16384,
    }
    assert config["context_file_max_chars"] == 80000
    assert config["auxiliary"]["compression"] == {
        "provider": "auto",
    }
    assert config["compression"] == {
        "threshold": 0.85,
        "abort_on_summary_failure": False,
    }
    manifest = agent_dir / "hermes_profile_manifest.json"
    assert manifest.exists()
    assert str(hermes_root) in manifest.read_text(encoding="utf-8")


def test_spawn_hermes_applies_scenario_context_overrides(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://idealab.example/v1")

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            pass

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    scenario = {
        "agent": {
            "hermes": {
                "context_length": 350000,
                "compression_threshold": 0.85,
            },
        },
        "run": {"horizon_steps": 24},
    }
    registry._spawn_hermes(
        "run-ctx",
        "http://127.0.0.1:5050",
        scenario=scenario,
    )

    config = yaml.safe_load(
        (tmp_path / "runs" / "run-ctx" / "agent" / "hermes_home" / "config.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["model"]["context_length"] == 350000
    assert config["compression"]["threshold"] == 0.85


def test_spawn_hermes_applies_provider_routing_and_reasoning_overrides(
    monkeypatch, tmp_path,
):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            pass

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    scenario = {
        "agent": {
            "hermes": {
                "context_length": 262144,
                "compression_threshold": 0.85,
                "provider_routing": {
                    "only": ["google-vertex/global"],
                    "require_parameters": True,
                },
                "reasoning_effort": "high",
            },
        },
        "run": {"horizon_steps": 24},
    }
    registry._spawn_hermes(
        "run-gemini",
        "http://127.0.0.1:5050",
        scenario=scenario,
    )

    config = yaml.safe_load(
        (tmp_path / "runs" / "run-gemini" / "agent" / "hermes_home" / "config.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["provider_routing"] == {
        "only": ["google-vertex/global"],
        "require_parameters": True,
    }
    assert config["agent"]["reasoning_effort"] == "high"


def test_spawn_hermes_reuses_existing_run_local_home_without_overwriting(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    source_skill_dir = hermes_root / "skills" / "official"
    source_skill_dir.mkdir(parents=True)
    (source_skill_dir / "SKILL.md").write_text(
        "---\nname: official\ndescription: official\n---\n",
        encoding="utf-8",
    )
    existing_skill = (
        tmp_path / "runs" / "run-1" / "agent" / "hermes_home"
        / "skills" / "existing" / "SKILL.md"
    )
    existing_skill.parent.mkdir(parents=True)
    existing_skill.write_text(
        "---\nname: existing\ndescription: existing\n---\n",
        encoding="utf-8",
    )
    existing_config = existing_skill.parents[2] / "config.yaml"
    existing_config.write_text(
        yaml.safe_dump(
            {
                "model": {"context_length": 131072, "max_tokens": 65536},
                "context_file_max_chars": 20000,
                "compression": {
                    "threshold": 0.50,
                    "protect_last_n": 24,
                },
                "auxiliary": {
                    "compression": {
                        "provider": "auto",
                        "model": "old-summary-model",
                        "base_url": "https://old-summary.example/v1",
                        "api_key": "old-summary-key",
                        "context_length": 131072,
                    },
                },
                "display": {"tool_progress": "off"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://idealab.example/v1")

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            pass

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_hermes("run-1", "http://127.0.0.1:5050")

    assert existing_skill.read_text(encoding="utf-8").startswith("---\nname: existing")
    assert not (
        tmp_path / "runs" / "run-1" / "agent" / "hermes_home"
        / "skills" / "official" / "SKILL.md"
    ).exists()
    config = yaml.safe_load(existing_config.read_text(encoding="utf-8"))
    assert config["model"] == {
        "context_length": 262144,
        "max_tokens": 16384,
    }
    assert config["context_file_max_chars"] == 80000
    assert config["auxiliary"]["compression"] == {
        "provider": "auto",
    }
    assert config["display"] == {"tool_progress": "off"}
    assert config["compression"] == {
        "threshold": 0.85,
        "protect_last_n": 24,
        "abort_on_summary_failure": False,
    }


def test_spawn_hermes_does_not_rewrite_config_while_existing_process_is_alive(
    monkeypatch,
    tmp_path,
):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    hermes_home = tmp_path / "runs" / "run-1" / "agent" / "hermes_home"
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    original_config = yaml.safe_dump(
        {
            "model": {"context_length": 131072, "max_tokens": 65536},
            "marker": "active-run-config",
        },
        sort_keys=False,
    )
    config_path.write_text(original_config, encoding="utf-8")
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://idealab.example/v1")

    class ExistingProc:
        pid = 456

        def poll(self):
            return None

    registry.bootstrap_procs["run-1"] = ExistingProc()

    def fail_popen(*args, **kwargs):
        raise AssertionError("should not spawn a duplicate Hermes process")

    monkeypatch.setattr("subprocess.Popen", fail_popen)

    registry._spawn_hermes("run-1", "http://127.0.0.1:5050")

    assert config_path.read_text(encoding="utf-8") == original_config


def test_spawn_hermes_uses_configured_python(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    configured_python = tmp_path / "python-hermes"
    configured_python.write_text("# fake python\n")
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.setenv("MERCHANTBENCH_HERMES_PYTHON", str(configured_python))
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_hermes("run-1", "http://127.0.0.1:5050")

    assert captured["cmd"][0] == str(configured_python)


def test_hermes_python_defaults_to_current_project_runtime(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    monkeypatch.delenv("MERCHANTBENCH_HERMES_PYTHON", raising=False)

    assert registry._hermes_python_executable() == sys.executable


def test_hermes_python_prefers_checkout_virtualenv(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    hermes_root = tmp_path / "hermes-agent"
    hermes_python = hermes_root / ".venv" / "bin" / "python"
    hermes_python.parent.mkdir(parents=True)
    hermes_python.write_text("# fake python\n")
    hermes_python.chmod(0o755)
    monkeypatch.delenv("MERCHANTBENCH_HERMES_PYTHON", raising=False)

    assert registry._hermes_python_executable(str(hermes_root)) == str(hermes_python)


def test_spawn_hermes_loads_repo_dotenv_for_llm_env(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    repo_root = tmp_path / "merchantbench-dev"
    repo_root.mkdir()
    (repo_root / ".env").write_text(
        "OPENAI_API_KEY=repo-key\n"
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
        "MODEL_NAME=repo-model\n"
    )
    hermes_root = tmp_path / "hermes-agent"
    adapter_dir = hermes_root / "merchantbench_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__main__.py").write_text("# fake adapter\n")
    monkeypatch.setattr(registry, "_repo_root", lambda: str(repo_root))
    monkeypatch.setenv("MERCHANTBENCH_HERMES_AGENT_ROOT", str(hermes_root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    captured = {}

    class FakePopen:
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    registry._spawn_hermes("run-1", "http://127.0.0.1:5050")

    env = captured["kwargs"]["env"]
    assert env["OPENAI_API_KEY"] == "repo-key"
    assert env["OPENAI_BASE_URL"] == "https://example.invalid/v1"
    assert env["MODEL_NAME"] == "repo-model"
    config_path = (
        tmp_path / "runs" / "run-1" / "agent" / "hermes_home" / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["auxiliary"]["compression"] == {"provider": "auto"}
    assert config["compression"] == {
        "threshold": 0.85,
        "abort_on_summary_failure": False,
    }


def test_spawn_compact_react_skips_when_existing_bootstrap_process_is_alive(monkeypatch, tmp_path):
    registry = RunRegistry(
        db_path=str(tmp_path / "test.db"),
        runs_root=str(tmp_path / "runs"),
    )
    baselines_dir = tmp_path / "agent" / "baselines"
    baselines_dir.mkdir(parents=True)
    (baselines_dir / "react_160k_compact_30k.py").write_text("# fake baseline\n")
    monkeypatch.setattr(registry, "_agent_baselines_dir", lambda: str(baselines_dir))

    class ExistingProc:
        pid = 456

        def poll(self):
            return None

    registry.bootstrap_procs["run-1"] = ExistingProc()

    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        raise AssertionError("should not spawn a duplicate bootstrap agent")

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    registry._spawn_react_160k_compact_30k(
        "run-1",
        "http://127.0.0.1:5050",
        model="qwen-max",
        max_steps=4320,
    )

    assert popen_calls == []
    assert registry.bootstrap_procs["run-1"].pid == 456
