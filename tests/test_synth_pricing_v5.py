import copy

import pytest

from data.generation_profiles import (
    ELASTICITY_CLIP_MAX,
    ELASTICITY_CLIP_MIN,
    MAX_RETAIL_MARGIN,
    PRICING_MODEL_LEGACY_ANCHOR_AT_COST,
)
from data.synth import generate
from web.runner import load_default_scenario


def test_default_generate_is_margin_consistent_v1():
    scenario = load_default_scenario()
    assert scenario["run"]["master_seed"] == 42
    assert scenario["data"]["num_products"] == 1000
    products, _ = generate(scenario)

    assert len(products) == 1000
    min_elasticity = 1.0 / MAX_RETAIL_MARGIN
    for product in products:
        assert product.price < product.ref_price
        assert product.elasticity >= ELASTICITY_CLIP_MIN
        assert product.elasticity >= min_elasticity
        assert product.elasticity <= ELASTICITY_CLIP_MAX
        implied = product.ref_price / (product.ref_price - product.price)
        assert product.elasticity == pytest.approx(implied, rel=1e-9, abs=1e-12)
        p_star = (
            product.elasticity / (product.elasticity - 1.0) * product.price
        )
        assert p_star == pytest.approx(product.ref_price, rel=1e-9, abs=1e-9)
        if product.category == "appliances":
            assert product.elasticity >= 1.0


def test_legacy_anchor_at_cost_restores_price_equals_ref_and_inelastic_appliances():
    scenario = load_default_scenario()
    scenario["generation_params"]["pricing_model"] = PRICING_MODEL_LEGACY_ANCHOR_AT_COST
    products, _ = generate(scenario)

    assert all(product.price == product.ref_price for product in products)
    appliances = [p for p in products if p.category == "appliances"]
    assert appliances
    assert all(p.elasticity < 1.0 for p in appliances)


def test_v5_and_legacy_share_operational_rng_prefix():
    scenario = load_default_scenario()
    v5_products, _ = generate(copy.deepcopy(scenario))
    scenario["generation_params"]["pricing_model"] = PRICING_MODEL_LEGACY_ANCHOR_AT_COST
    legacy_products, _ = generate(scenario)

    assert len(v5_products) == len(legacy_products)
    for v5, legacy in zip(v5_products, legacy_products):
        assert v5.product_id == legacy.product_id
        assert v5.category == legacy.category
        assert v5.max_quantity == legacy.max_quantity
        assert v5.cancel_rate == legacy.cancel_rate
        assert v5.price != legacy.price
        assert v5.elasticity != legacy.elasticity


def test_unknown_pricing_model_raises_value_error():
    scenario = load_default_scenario()
    scenario["generation_params"]["pricing_model"] = "not_a_model"
    with pytest.raises(ValueError, match="unknown pricing_model"):
        generate(scenario)


def test_missing_pricing_model_defaults_to_margin_consistent_v1():
    scenario = load_default_scenario()
    del scenario["generation_params"]["pricing_model"]
    products, _ = generate(scenario)
    assert all(product.price < product.ref_price for product in products)
