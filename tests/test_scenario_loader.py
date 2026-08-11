from pathlib import Path

from web.runner import load_default_scenario, load_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_scenario_extends_relative_base_and_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """
run:
  horizon_steps: 24
  step_hours: 1
agent:
  language: en
  activation_period: 12
  tool_denylist:
    - market_brief
    - hot_search_terms
  cost_pricing:
    input_per_million: 5.0
    output_per_million: 30.0
    cached_input_per_million: 0.5
""",
        encoding="utf-8",
    )
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    child = profile_dir / "hermes.yaml"
    child.write_text(
        """
extends: ../base.yaml
agent:
  language: zh
  tool_denylist:
    - read_memory_doc
    - write_memory_doc
  cost_pricing:
    output_per_million: 12.0
""",
        encoding="utf-8",
    )

    scenario = load_scenario(str(child))

    assert "extends" not in scenario
    assert scenario["run"] == {"horizon_steps": 24, "step_hours": 1}
    assert scenario["agent"]["language"] == "zh"
    assert scenario["agent"]["activation_period"] == 12
    assert scenario["agent"]["tool_denylist"] == [
        "read_memory_doc",
        "write_memory_doc",
    ]
    assert scenario["agent"]["cost_pricing"] == {
        "input_per_million": 5.0,
        "output_per_million": 12.0,
        "cached_input_per_million": 0.5,
    }


def test_hermes_scenario_extends_default_and_denies_market_and_memory_tools():
    default = load_default_scenario()
    hermes = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes.yaml"))

    assert default["agent"]["detailed"] is False
    assert default["agent"]["tool_denylist"] == [
        "market_brief",
        "hot_search_terms",
    ]
    assert hermes["run"]["step_hours"] == default["run"]["step_hours"]
    assert hermes["run"]["horizon_steps"] == default["run"]["horizon_steps"]
    assert hermes["data"]["source"] == default["data"]["source"]
    assert hermes["agent"]["tool_denylist"] == [
        "market_brief",
        "hot_search_terms",
        "read_memory_doc",
        "write_memory_doc",
    ]
    assert hermes["agent"]["detailed"] is False
    assert hermes["agent"]["cost_pricing"] == default["agent"]["cost_pricing"]


def test_zero_prior_scenario_preserves_legacy_v2_experiment_policy():
    scenario = load_scenario(
        str(REPO_ROOT / "env/scenarios/agents/hermes_zero_prior.yaml")
    )

    assert scenario["shop_rating"]["model"] == "order_outcome_v2"
    assert scenario["shop_rating"]["prior_weight"] == 0
    assert scenario["shop_rating"]["half_life_days"] == 30
    assert scenario["shop_rating"]["star_multipliers"] == [
        0.10, 0.35, 0.80, 1.00, 1.20,
    ]
