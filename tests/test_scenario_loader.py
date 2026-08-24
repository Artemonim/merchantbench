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


def test_default_economy_v6_flags_are_off():
    default = load_default_scenario()
    block = default["economy_v6"]
    assert block["enabled"] is False
    assert block["take_rate"]["enabled"] is False
    assert block["fulfillment"]["enabled"] is False
    assert block["refund"]["enabled"] is False
    assert default["generation_params"]["risk_trust_coupling"] is False
    assert set(block["take_rate"]["by_category"]) == set(default["data"]["category_pool"])
    assert set(block["fulfillment"]["by_category"]) == set(default["data"]["category_pool"])


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
    assert hermes["agent"]["cost_pricing"] == {
        "input_per_million": 0.13,
        "output_per_million": 0.28,
        "cached_input_per_million": 0.07,
    }
    assert hermes["shop_rating"]["model"] == "order_outcome_v4"
    assert hermes["public_reviews"]["enabled"] is True
    assert hermes["public_reviews"]["demand"] == {
        "min_trust_multiplier": 0.8,
        "max_trust_multiplier": 1.0,
        "half_saturation_reviews": 20,
    }


def test_zero_prior_scenario_preserves_legacy_v2_experiment_policy():
    scenario = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_zero_prior.yaml"))

    assert scenario["shop_rating"]["model"] == "order_outcome_v2"
    assert scenario["shop_rating"]["prior_weight"] == 0
    assert scenario["shop_rating"]["half_life_days"] == 30
    assert scenario["shop_rating"]["star_multipliers"] == [
        0.10,
        0.35,
        0.80,
        1.00,
        1.20,
    ]
    assert scenario["public_reviews"]["enabled"] is False


def test_v3_scenario_preserves_pre_public_review_economics():
    scenario = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_v3.yaml"))

    assert scenario["shop_rating"]["model"] == "order_outcome_v3"
    assert scenario["shop_rating"]["reputation_volume"] == {
        "min_multiplier": 0.8,
        "max_multiplier": 1.0,
        "half_saturation_orders": 20,
    }
    assert scenario["public_reviews"]["enabled"] is False


def test_bankrupt_scenario_overrides_role_and_goals_on_v5_catalog():
    default = load_default_scenario()
    scenario = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_bankrupt.yaml"))

    assert scenario["agent"]["role"]["en"].startswith(
        "You are an operating agent of a small store inside MerchantBench"
    )
    assert "simulated e-commerce economy" in scenario["agent"]["role"]["en"]
    assert any("bankruptcy" in goal.lower() for goal in scenario["agent"]["goals"]["en"])
    assert any("financial suicide" in goal.lower() for goal in scenario["agent"]["goals"]["en"])
    assert scenario["generation_params"]["pricing_model"] == (default["generation_params"]["pricing_model"])
    assert scenario["generation_params"]["base_demand"] == (default["generation_params"]["base_demand"])
    assert scenario["shop_rating"]["model"] == "order_outcome_v4"


def test_hermes_v6_base_enables_v6_economy_on_olist_catalog():
    scenario = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_v6.yaml"))

    assert scenario["economy_v6"]["enabled"] is True
    assert scenario["economy_v6"]["take_rate"]["enabled"] is True
    assert scenario["economy_v6"]["fulfillment"]["enabled"] is True
    assert scenario["economy_v6"]["refund"]["enabled"] is True
    assert scenario["generation_params"]["risk_trust_coupling"] is True
    assert scenario["data"]["source"] == "private_real"
    assert scenario["data"]["num_products"] == 1000
    assert scenario["agent"]["tool_denylist"] == [
        "market_brief",
        "hot_search_terms",
        "read_memory_doc",
        "write_memory_doc",
    ]
    assert scenario["agent"]["hermes"]["context_length"] == 262144
    assert scenario["agent"]["hermes"]["compression_threshold"] == 0.85


def test_hermes_v6_red_variants_carry_distinct_modes_and_oxalpha_knobs():
    base = REPO_ROOT / "env/scenarios/agents"
    unrestricted = load_scenario(str(base / "hermes_v6_red_unrestricted.yaml"))
    bad_merchant = load_scenario(str(base / "hermes_v6_red_bad_merchant.yaml"))
    bad_economics = load_scenario(str(base / "hermes_v6_red_bad_economics.yaml"))

    for scenario in (unrestricted, bad_merchant, bad_economics):
        # * All red modes run on the v6 economy with the same Hermes knobs.
        assert scenario["economy_v6"]["enabled"] is True
        assert scenario["data"]["source"] == "private_real"
        assert scenario["agent"]["hermes"]["reasoning_effort"] == "xhigh"
        assert scenario["agent"]["hermes"]["provider_routing"] == {}
        assert scenario["agent"]["cost_pricing"] == {
            "input_per_million": 0.0,
            "output_per_million": 0.0,
            "cached_input_per_million": 0.0,
        }
        assert "simulated e-commerce economy" in scenario["agent"]["role"]["en"]
        assert any("deposit_pool" in goal for goal in scenario["agent"]["goals"]["en"])

    # * Mode identity: each overlay encodes a distinct failure policy.
    assert any(
        "red-team" in goal.lower() or "red-team" in unrestricted["agent"]["role"]["en"].lower()
        for goal in [*unrestricted["agent"]["goals"]["en"], ""]
    )
    assert any("penalty-farming" in goal and "permitted" in goal for goal in unrestricted["agent"]["goals"]["en"])
    assert any("plausible merchant" in goal for goal in bad_merchant["agent"]["goals"]["en"])
    assert any("only assortment and pricing" in goal for goal in bad_economics["agent"]["goals"]["en"])


def test_gemini_scenarios_pin_vertex_global_routing():
    gemini = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_gemini.yaml"))
    bankrupt = load_scenario(str(REPO_ROOT / "env/scenarios/agents/hermes_gemini_bankrupt.yaml"))

    for scenario in (gemini, bankrupt):
        assert scenario["agent"]["hermes"]["provider_routing"] == {
            "only": ["google-vertex/global"],
            "require_parameters": True,
        }
        assert scenario["agent"]["hermes"]["reasoning_effort"] == "high"
        assert scenario["agent"]["hermes"]["context_length"] == 262144
        assert scenario["agent"]["cost_pricing"] == {
            "input_per_million": 0.375,
            "output_per_million": 1.875,
            "cached_input_per_million": 0.0375,
        }

    assert "Maximize total assets" not in str(gemini.get("agent", {}).get("goals", ""))
    assert any("bankruptcy" in goal.lower() for goal in bankrupt["agent"]["goals"]["en"])
