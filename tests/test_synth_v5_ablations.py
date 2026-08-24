"""Offline v5 synthetic-economy ablations (no Flask, no agent subprocess)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from data.economy_diagnostics import (
    APPLIANCES_CATEGORY,
    RULE_BASED_DEFAULT_MARKUP,
    catalog_economy_aggregates,
    listing_day_demand_at_sale,
    listing_day_gross_at_sale,
)
from data.generation_profiles import (
    CALIBRATED_BASE_DEMAND_RANGE,
    ELASTICITY_CLIP_MIN,
    LEGACY_BASE_DEMAND_RANGE,
    PRICING_MODEL_LEGACY_ANCHOR_AT_COST,
    PRICING_MODEL_MARGIN_CONSISTENT_V1,
)
from data.synth import generate
from web.runner import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "env" / "scenarios"
ABLATIONS_DIR = SCENARIOS_DIR / "ablations"
MASTER_SEED = 42


def _load_catalog(path: Path):
    scenario = load_scenario(str(path))
    assert scenario["run"]["master_seed"] == MASTER_SEED
    products, _hourly = generate(scenario)
    small_share = float(scenario["data"]["small_share"])
    return scenario, products, small_share


@pytest.fixture(scope="module")
def default_catalog():
    return _load_catalog(SCENARIOS_DIR / "default.yaml")


@pytest.fixture(scope="module")
def both_catalog():
    return _load_catalog(ABLATIONS_DIR / "both.yaml")


@pytest.fixture(scope="module")
def pricing_only_catalog():
    return _load_catalog(ABLATIONS_DIR / "pricing_only.yaml")


@pytest.fixture(scope="module")
def demand_only_catalog():
    return _load_catalog(ABLATIONS_DIR / "demand_only.yaml")


@pytest.fixture(scope="module")
def legacy_v4_catalog():
    return _load_catalog(ABLATIONS_DIR / "legacy_v4.yaml")


def test_listing_day_ces_matches_mean_curve_times_price_ratio():
    product = SimpleNamespace(
        market_curve=[2.0, 4.0],
        ref_price=100.0,
        elasticity=2.0,
        price=50.0,
        category="office",
    )
    assert listing_day_demand_at_sale(product, 100.0, 1.0) == pytest.approx(3.0)
    assert listing_day_demand_at_sale(product, 200.0, 1.0) == pytest.approx(0.75)
    assert listing_day_gross_at_sale(product, 100.0, 1.0) == pytest.approx(150.0)
    assert listing_day_demand_at_sale(product, 0.0, 1.0) == 0.0


def test_both_default_and_overlay_are_margin_consistent_calibrated(default_catalog, both_catalog):
    for _scenario, products, small_share in (default_catalog, both_catalog):
        assert len(products) == 1000
        assert all(product.price < product.ref_price for product in products)
        assert all(product.elasticity >= ELASTICITY_CLIP_MIN for product in products)
        metrics = catalog_economy_aggregates(products, small_share)
        assert metrics["share_eps_lt_1"] == 0.0
        assert 0.40 <= metrics["mean_q_day_at_ref"] <= 0.65
        assert 0.20 <= metrics["mean_margin_at_ref"] <= 0.50
        assert metrics["mean_gross_day_at_ref"] > 0.0


def test_pricing_only_keeps_v5_margins_and_legacy_demand_scale(pricing_only_catalog, both_catalog):
    _scenario, products, small_share = pricing_only_catalog
    assert all(product.price < product.ref_price for product in products)
    metrics = catalog_economy_aggregates(products, small_share)
    assert metrics["share_eps_lt_1"] == 0.0
    assert 20.0 <= metrics["mean_q_day_at_ref"] <= 32.0

    both_metrics = catalog_economy_aggregates(both_catalog[1], both_catalog[2])
    assert metrics["mean_q_day_at_ref"] > 10.0 * both_metrics["mean_q_day_at_ref"]


def test_demand_only_is_legacy_anchor_with_calibrated_volume(demand_only_catalog):
    _scenario, products, small_share = demand_only_catalog
    assert all(product.price == product.ref_price for product in products)
    appliances = [product for product in products if product.category == APPLIANCES_CATEGORY]
    assert appliances
    assert all(product.elasticity < 1.0 for product in appliances)

    metrics = catalog_economy_aggregates(products, small_share)
    assert metrics["share_eps_lt_1"] > 0.0
    assert 0.40 <= metrics["mean_q_day_at_ref"] <= 0.65
    assert metrics["mean_margin_at_ref"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mean_gross_day_at_ref"] == pytest.approx(0.0, abs=1e-9)


def test_exploit_probe_uses_appliances_only(demand_only_catalog, both_catalog):
    _demand_scenario, demand_products, demand_share = demand_only_catalog
    demand_appliances = catalog_economy_aggregates(
        demand_products,
        demand_share,
        category=APPLIANCES_CATEGORY,
        rule_markup=RULE_BASED_DEFAULT_MARKUP,
    )
    assert all(product.elasticity < 1.0 for product in demand_products if product.category == APPLIANCES_CATEGORY)
    assert demand_appliances["mean_gross_day_at_10x_cost"] > demand_appliances["mean_gross_day_at_ref"]

    _both_scenario, both_products, both_share = both_catalog
    both_appliances = catalog_economy_aggregates(
        both_products,
        both_share,
        category=APPLIANCES_CATEGORY,
        rule_markup=RULE_BASED_DEFAULT_MARKUP,
    )
    assert all(product.elasticity >= 2.0 for product in both_products if product.category == APPLIANCES_CATEGORY)
    assert both_appliances["mean_gross_day_at_10x_cost"] < both_appliances["mean_gross_day_at_rule_markup"]


def test_legacy_v4_is_price_equals_ref_and_old_demand_scale(legacy_v4_catalog):
    _scenario, products, small_share = legacy_v4_catalog
    assert all(product.price == product.ref_price for product in products)
    metrics = catalog_economy_aggregates(products, small_share)
    assert 20.0 <= metrics["mean_q_day_at_ref"] <= 32.0


def test_ablation_overlays_change_only_intended_generation_keys():
    default = load_scenario(str(SCENARIOS_DIR / "default.yaml"))
    both = load_scenario(str(ABLATIONS_DIR / "both.yaml"))
    pricing_only = load_scenario(str(ABLATIONS_DIR / "pricing_only.yaml"))
    demand_only = load_scenario(str(ABLATIONS_DIR / "demand_only.yaml"))
    legacy_v4 = load_scenario(str(ABLATIONS_DIR / "legacy_v4.yaml"))

    default_gp = default["generation_params"]
    assert both["generation_params"]["pricing_model"] == default_gp["pricing_model"]
    assert both["generation_params"]["base_demand"] == default_gp["base_demand"]
    assert both["generation_params"]["pricing_model"] == PRICING_MODEL_MARGIN_CONSISTENT_V1
    assert list(both["generation_params"]["base_demand"]) == list(CALIBRATED_BASE_DEMAND_RANGE)
    assert _generation_params_without(both, ()) == _generation_params_without(default, ())

    assert pricing_only["generation_params"]["pricing_model"] == default_gp["pricing_model"]
    assert list(pricing_only["generation_params"]["base_demand"]) == list(LEGACY_BASE_DEMAND_RANGE)
    assert _generation_params_without(pricing_only, ("base_demand",)) == (
        _generation_params_without(default, ("base_demand",))
    )

    assert demand_only["generation_params"]["pricing_model"] == PRICING_MODEL_LEGACY_ANCHOR_AT_COST
    assert demand_only["generation_params"]["base_demand"] == default_gp["base_demand"]
    assert _generation_params_without(demand_only, ("pricing_model",)) == (
        _generation_params_without(default, ("pricing_model",))
    )

    assert legacy_v4["generation_params"]["pricing_model"] == PRICING_MODEL_LEGACY_ANCHOR_AT_COST
    assert list(legacy_v4["generation_params"]["base_demand"]) == list(LEGACY_BASE_DEMAND_RANGE)
    assert _generation_params_without(legacy_v4, ("pricing_model", "base_demand")) == _generation_params_without(
        default, ("pricing_model", "base_demand")
    )


def test_economy_v6_ablation_overlays_change_only_intended_enabled_flags():
    default = load_scenario(str(SCENARIOS_DIR / "default.yaml"))
    fees_only = load_scenario(str(ABLATIONS_DIR / "economy_v6_fees_only.yaml"))
    refund_only = load_scenario(str(ABLATIONS_DIR / "economy_v6_refund_only.yaml"))
    v6_both = load_scenario(str(ABLATIONS_DIR / "economy_v6_both.yaml"))

    assert default["economy_v6"]["enabled"] is False
    assert default["economy_v6"]["take_rate"]["enabled"] is False
    assert default["economy_v6"]["fulfillment"]["enabled"] is False
    assert default["economy_v6"]["refund"]["enabled"] is False
    assert default["generation_params"]["risk_trust_coupling"] is False

    assert fees_only["economy_v6"]["enabled"] is True
    assert fees_only["economy_v6"]["take_rate"]["enabled"] is True
    assert fees_only["economy_v6"]["fulfillment"]["enabled"] is True
    assert fees_only["economy_v6"]["refund"]["enabled"] is False
    assert fees_only["generation_params"]["risk_trust_coupling"] is True

    assert refund_only["economy_v6"]["enabled"] is True
    assert refund_only["economy_v6"]["take_rate"]["enabled"] is False
    assert refund_only["economy_v6"]["fulfillment"]["enabled"] is False
    assert refund_only["economy_v6"]["refund"]["enabled"] is True
    assert refund_only["generation_params"]["risk_trust_coupling"] is True

    assert v6_both["economy_v6"]["enabled"] is True
    assert v6_both["economy_v6"]["take_rate"]["enabled"] is True
    assert v6_both["economy_v6"]["fulfillment"]["enabled"] is True
    assert v6_both["economy_v6"]["refund"]["enabled"] is True
    assert v6_both["generation_params"]["risk_trust_coupling"] is True

    for overlay in (fees_only, refund_only, v6_both):
        assert _economy_v6_without_enabled(overlay) == _economy_v6_without_enabled(default)
        assert _generation_params_without(overlay, ("risk_trust_coupling",)) == (
            _generation_params_without(default, ("risk_trust_coupling",))
        )


def _generation_params_without(scenario: dict, keys: tuple[str, ...]) -> dict:
    params = dict(scenario["generation_params"])
    for key in keys:
        params.pop(key, None)
    return params


def _economy_v6_without_enabled(scenario: dict) -> dict:
    """Return economy_v6 with master and nested enabled flags removed."""
    block = dict(scenario["economy_v6"])
    block.pop("enabled", None)
    stripped: dict = {}
    for key, value in block.items():
        if isinstance(value, dict):
            nested = dict(value)
            nested.pop("enabled", None)
            stripped[key] = nested
        else:
            stripped[key] = value
    return stripped
