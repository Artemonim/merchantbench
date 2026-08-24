import pytest
from data.generation_profiles import (
    CALIBRATED_BASE_DEMAND_RANGE,
    LEGACY_BASE_DEMAND_RANGE,
    TARGET_ACTIVE_LISTINGS,
    expected_shop_day_orders_at_ref,
    mean_listing_day_demand_at_ref,
    resolve_base_demand_range,
)
from data.synth import generate
from web.runner import load_default_scenario


def test_default_generate_listing_day_demand_near_paper_target():
    scenario = load_default_scenario()
    assert scenario["run"]["master_seed"] == 42
    assert scenario["data"]["num_products"] == 1000
    assert scenario["data"]["small_share"] == 1.0
    products, _ = generate(scenario)

    mean_q = mean_listing_day_demand_at_ref(products, small_share=1.0)
    assert 0.40 <= mean_q <= 0.65
    assert mean_q == pytest.approx(0.528308, rel=0, abs=0.01)


def test_default_generate_shop_day_orders_near_human_baseline():
    scenario = load_default_scenario()
    products, _ = generate(scenario)

    shop_q = expected_shop_day_orders_at_ref(
        products,
        small_share=1.0,
        n_listings=TARGET_ACTIVE_LISTINGS,
    )
    assert 20.0 <= shop_q <= 33.0


def test_explicit_legacy_base_demand_restores_old_scale():
    scenario = load_default_scenario()
    scenario["generation_params"]["base_demand"] = list(LEGACY_BASE_DEMAND_RANGE)
    products, _ = generate(scenario)

    mean_q = mean_listing_day_demand_at_ref(products, small_share=1.0)
    assert 20.0 <= mean_q <= 32.0


def test_invalid_base_demand_range_raises():
    scenario = load_default_scenario()
    invalid_ranges = (
        [0.0, 1.0],
        [-0.1, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [float("nan"), 1.0],
        [0.02, float("inf")],
        [0.02],
        "0.02,1.02",
    )
    for bad in invalid_ranges:
        scenario["generation_params"]["base_demand"] = bad
        with pytest.raises(ValueError, match="base_demand"):
            generate(scenario)


def test_missing_base_demand_uses_calibrated_default_and_keeps_v5_pricing():
    default_scenario = load_default_scenario()
    missing_scenario = load_default_scenario()
    del missing_scenario["generation_params"]["base_demand"]

    assert resolve_base_demand_range({}) == CALIBRATED_BASE_DEMAND_RANGE
    assert resolve_base_demand_range(missing_scenario) == CALIBRATED_BASE_DEMAND_RANGE

    default_products, _ = generate(default_scenario)
    missing_products, _ = generate(missing_scenario)

    assert [p.price for p in missing_products] == [p.price for p in default_products]
    assert [p.ref_price for p in missing_products] == [p.ref_price for p in default_products]
    assert [p.elasticity for p in missing_products] == [p.elasticity for p in default_products]
    assert all(p.price < p.ref_price for p in missing_products)

    mean_q = mean_listing_day_demand_at_ref(missing_products, small_share=1.0)
    assert 0.40 <= mean_q <= 0.65
