import numpy as np


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
    assert supplier_ranges["hourly_increment"][0] <= operational["hourly_increment"] < supplier_ranges["hourly_increment"][1]
    assert supplier_ranges["ship_hours"][0] <= operational["ship_hours"] < supplier_ranges["ship_hours"][1]
    assert 1 <= operational["logistics_hours"] <= 72
    for field in ("cancel_rate", "refund_rate", "only_refund_rate", "bad_review_rate"):
        lo, hi = params["risk_ranges"][field]
        assert lo <= risk[field] <= hi
    for field in ("timeout_rate", "price_change_rate", "supplier_delist_rate"):
        lo, hi = supplier_ranges[field]
        assert lo <= risk[field] <= hi
    assert 0.65 <= elasticity <= 1.05
