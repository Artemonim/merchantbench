import numpy as np
import pytest


def test_default_generation_params_translate_categories_and_build_hourly_weights():
    from data.generation_profiles import (
        build_hourly_dist_for_categories,
        load_default_generation_params,
        translate_category,
    )

    params = load_default_generation_params()

    assert translate_category("办公、文化", params) == "office"
    assert translate_category("女装", params) == "womenswear"

    hourly = build_hourly_dist_for_categories(
        ["office", "womenswear"],
        seed=42,
        params=params,
    )

    assert set(hourly) == {"office", "womenswear"}
    assert hourly["office"].shape == (24,)
    assert hourly["womenswear"].shape == (24,)
    assert np.isclose(hourly["office"].sum(), 1.0)
    assert np.isclose(hourly["womenswear"].sum(), 1.0)
    assert not np.allclose(hourly["office"], hourly["womenswear"])


def test_periodic_resample_interpolates_without_total_rescaling():
    from data.generation_profiles import resample_periodic_curve

    out = resample_periodic_curve([1.0, 2.0, 4.0], target_len=5)

    assert np.allclose(out, [1.0, 1.6, 2.4, 3.6, 2.8])


def test_default_scenario_is_single_source_for_generation_ranges():
    from data.generation_profiles import generation_params_from_scenario
    from web.runner import load_default_scenario

    scenario = load_default_scenario()
    params = generation_params_from_scenario(scenario)

    assert params["risk_ranges"] == scenario["risk_ranges"]
    for field in (
        "max_quantity",
        "hourly_increment",
        "timeout_rate",
        "price_change_rate",
        "supplier_delist_rate",
        "ship_hours",
        "logistics_hours",
    ):
        assert params["supplier_ranges"][field] == scenario["supplier_ranges"][field]
    assert params["category_mapping"]["办公、文化"] == "office"
    assert params["categories"]["office"]["hour_shape"][9] == 1.05
    assert "dow_shape" not in params["categories"]["office"]


def test_shared_generation_helpers_respect_ranges_and_category_elasticity():
    from data.generation_profiles import (
        load_default_generation_params,
        normalize_supplier_ranges,
        sample_elasticity,
        sample_operational_fields,
        sample_risk_event_fields,
    )

    params = load_default_generation_params()
    supplier_ranges = normalize_supplier_ranges(
        params["supplier_ranges"],
        {"default_promised_ship_hours": 48},
    )
    rng = np.random.default_rng(123)

    operational = sample_operational_fields(rng, supplier_ranges)
    risk = sample_risk_event_fields(rng, params["risk_ranges"], supplier_ranges)
    elasticity = sample_elasticity(rng, "appliances", params)

    assert 0 <= operational["quantity"] <= operational["max_quantity"]
    assert supplier_ranges["max_quantity"][0] <= operational["max_quantity"] < supplier_ranges["max_quantity"][1]
    assert (
        supplier_ranges["hourly_increment"][0]
        <= operational["hourly_increment"]
        < supplier_ranges["hourly_increment"][1]
    )
    assert supplier_ranges["ship_hours"][0] <= operational["ship_hours"] < supplier_ranges["ship_hours"][1]
    assert 1 <= operational["logistics_hours"] <= 72
    for field in ("cancel_rate", "refund_rate", "only_refund_rate", "bad_review_rate"):
        lo, hi = params["risk_ranges"][field]
        assert lo <= risk[field] <= hi
    for field in ("timeout_rate", "price_change_rate", "supplier_delist_rate"):
        lo, hi = supplier_ranges[field]
        assert lo <= risk[field] <= hi
    assert 0.65 <= elasticity <= 1.05


def test_cost_and_elasticity_from_margin_matches_ces_optimum_and_clips():
    from data.generation_profiles import (
        ELASTICITY_CLIP_MAX,
        ELASTICITY_CLIP_MIN,
        MAX_RETAIL_MARGIN,
        MIN_RETAIL_MARGIN,
        cost_and_elasticity_from_margin,
    )

    ref = 100.0
    margin = 0.25
    cost, elasticity = cost_and_elasticity_from_margin(ref, margin)

    assert elasticity == pytest.approx(1.0 / margin)
    assert cost == pytest.approx(ref * (1.0 - margin))
    p_star = elasticity / (elasticity - 1.0) * cost
    assert p_star == pytest.approx(ref)
    assert 0.0 < cost < ref

    cost_hi, eps_hi = cost_and_elasticity_from_margin(ref, 0.01)
    assert eps_hi == pytest.approx(1.0 / MIN_RETAIL_MARGIN)
    assert eps_hi == pytest.approx(ELASTICITY_CLIP_MAX)
    assert cost_hi == pytest.approx(ref * (1.0 - MIN_RETAIL_MARGIN))
    assert (eps_hi / (eps_hi - 1.0) * cost_hi) == pytest.approx(ref)
    assert 0.0 < cost_hi < ref

    cost_lo, eps_lo = cost_and_elasticity_from_margin(ref, 0.99)
    assert eps_lo == pytest.approx(1.0 / MAX_RETAIL_MARGIN)
    assert eps_lo != pytest.approx(ELASTICITY_CLIP_MIN)
    assert cost_lo == pytest.approx(ref * (1.0 - MAX_RETAIL_MARGIN))
    assert (eps_lo / (eps_lo - 1.0) * cost_lo) == pytest.approx(ref)
    assert 0.0 < cost_lo < ref


def test_sample_retail_margin_clamps_to_profile_and_global_bounds():
    from data.generation_profiles import (
        MAX_RETAIL_MARGIN,
        MIN_RETAIL_MARGIN,
        load_default_generation_params,
        sample_retail_margin,
    )

    params = load_default_generation_params()
    rng = np.random.default_rng(123)
    for _ in range(50):
        margin = sample_retail_margin(rng, "appliances", params)
        assert 0.32 <= margin <= 0.48
        assert MIN_RETAIL_MARGIN <= margin <= MAX_RETAIL_MARGIN

    wild = {
        "categories": {
            "x": {
                "retail_margin": {
                    "mean": 0.9,
                    "jitter": 0.0,
                    "min": 0.9,
                    "max": 0.9,
                }
            }
        }
    }
    assert sample_retail_margin(np.random.default_rng(0), "x", wild) == pytest.approx(MAX_RETAIL_MARGIN)
    low = {
        "categories": {
            "x": {
                "retail_margin": {
                    "mean": 0.01,
                    "jitter": 0.0,
                    "min": 0.01,
                    "max": 0.01,
                }
            }
        }
    }
    assert sample_retail_margin(np.random.default_rng(0), "x", low) == pytest.approx(MIN_RETAIL_MARGIN)


def test_resolve_base_demand_range_defaults_to_calibrated_and_rejects_invalid():
    from data.generation_profiles import (
        CALIBRATED_BASE_DEMAND_RANGE,
        LEGACY_BASE_DEMAND_RANGE,
        load_default_generation_params,
        resolve_base_demand_range,
    )

    params = load_default_generation_params()
    assert list(params["base_demand"]) == list(CALIBRATED_BASE_DEMAND_RANGE)
    assert resolve_base_demand_range(params) == CALIBRATED_BASE_DEMAND_RANGE
    assert resolve_base_demand_range(None) == CALIBRATED_BASE_DEMAND_RANGE
    assert resolve_base_demand_range({}) == CALIBRATED_BASE_DEMAND_RANGE
    assert resolve_base_demand_range({"base_demand": list(LEGACY_BASE_DEMAND_RANGE)}) == LEGACY_BASE_DEMAND_RANGE

    with pytest.raises(ValueError, match="base_demand"):
        resolve_base_demand_range({"base_demand": [0.0, 1.02]})
    with pytest.raises(ValueError, match="base_demand"):
        resolve_base_demand_range({"base_demand": [1.0, 1.0]})


def test_mean_listing_day_demand_at_ref_scales_curve_mean_by_share():
    from types import SimpleNamespace

    from data.generation_profiles import (
        expected_shop_day_orders_at_ref,
        mean_listing_day_demand_at_ref,
    )

    products = [
        SimpleNamespace(market_curve=[1.0, 3.0]),
        SimpleNamespace(market_curve=[2.0, 2.0]),
    ]
    assert mean_listing_day_demand_at_ref(products, small_share=0.5) == pytest.approx(1.0)
    assert expected_shop_day_orders_at_ref(products, small_share=0.5, n_listings=50) == pytest.approx(50.0)
