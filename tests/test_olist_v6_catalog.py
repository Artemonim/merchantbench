"""CI-safe tests for the Olist v6 catalog and seed subsample."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.build_olist_v6 import (
    EMPTY_CURVE_FLOOR,
    FALLBACK_CATEGORY,
    map_olist_category,
    market_curve_from_timestamps,
    prepare_olist_v6_from_tables,
)
from data.generation_profiles import load_default_generation_params
from data.private_real import PrivateRealDataError, load_dataset, subsample_catalog
from data.synth import generate
from tests.olist_v6_fixture import (
    N_VALID_SKUS,
    SELLER_A,
    SELLER_B,
    inline_olist_tables,
    write_olist_v6_fixture_db,
)
from web.app import create_app
from web.runner import load_default_scenario, load_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "env" / "scenarios"


def _prepared(n_valid=N_VALID_SKUS):
    params = load_default_generation_params()
    return prepare_olist_v6_from_tables(
        inline_olist_tables(n_valid=n_valid),
        seed=42,
        params=params,
        source_label="test",
    )


def test_map_olist_category_uses_pool_and_home_goods_fallback():
    assert map_olist_category("housewares") == "home_goods"
    assert map_olist_category("sports_leisure") == "sports"
    assert map_olist_category("office_furniture") == "office"
    assert map_olist_category("unknown_widget_line") == FALLBACK_CATEGORY
    with pytest.raises(ValueError):
        map_olist_category("   ")


def test_market_curve_tiles_short_history_and_floors_empty():
    from datetime import datetime, timedelta

    empty = market_curve_from_timestamps([], None)
    assert empty == [EMPTY_CURVE_FLOOR] * 365
    start = datetime(2017, 10, 1)
    stamps = [datetime(2017, 10, 1, 12, 0, 0), datetime(2017, 10, 3, 9, 0, 0)]
    curve = market_curve_from_timestamps(stamps, (start, datetime(2017, 10, 10)))
    assert len(curve) == 365
    assert all(value >= 0.0 for value in curve)
    assert sum(curve) > 0.0
    # * Tiled observed counts keep a zero/nonzero pattern, not a sine wave.
    assert curve.count(0.0) != 0 or max(curve) != min(curve)

    # * A single spike on a 700-day calendar must not crash resample.
    sparse_start = datetime(2016, 1, 1)
    sparse_end = sparse_start + timedelta(days=699)
    sparse = market_curve_from_timestamps(
        [datetime(2016, 6, 15, 8, 0, 0)],
        (sparse_start, sparse_end),
    )
    assert len(sparse) == 365
    assert all(value >= 0.0 for value in sparse)
    assert sum(sparse) > 0.0


def test_builder_filters_junk_and_keeps_seller_identity():
    products, hourly_dist, meta = _prepared()
    assert len(products) == N_VALID_SKUS
    assert {row["supplier_id"] for row in products} == {SELLER_A, SELLER_B}
    assert {row["category"] for row in products} <= {"home_goods", "sports"}
    assert set(hourly_dist) == {row["category"] for row in products}
    assert meta["ref_price_policy"] == "observed_listing_price"

    by_seller: dict[str, list[dict]] = {}
    for row in products:
        by_seller.setdefault(row["supplier_id"], []).append(row)
        curve = json.loads(row["market_curve"])
        assert len(curve) == 365
        assert all(value >= 0.0 for value in curve)
        assert float(row["ref_price"]) > 0.0
        assert 0.0 < float(row["price"]) < float(row["ref_price"])
        assert 1.0 <= float(row["historical_avg_rating"]) <= 5.0
        assert 0.0 <= float(row["refund_rate"]) <= 1.0
        assert 1 <= int(row["ship_hours"])
        assert 1 <= int(row["logistics_hours"]) <= 72
    for seller_id, group in by_seller.items():
        ratings = {item["shop_rating"] for item in group}
        buyers = {item["return_buyer_rate"] for item in group}
        ages = {item["supplier_age_years"] for item in group}
        assert len(ratings) == 1, seller_id
        assert len(buyers) == 1, seller_id
        assert len(ages) == 1, seller_id


def test_load_dataset_and_subsample_prefix_is_seed_stable(tmp_path):
    db_path = tmp_path / "olist_v6_fixture.sqlite"
    write_olist_v6_fixture_db(str(db_path))
    products, hourly, _meta = load_dataset(str(db_path))
    assert len(products) == N_VALID_SKUS

    ten, hourly_ten = subsample_catalog(products, hourly, 10, 42)
    twenty, hourly_twenty = subsample_catalog(products, hourly, 20, 42)
    ten_again, _ = subsample_catalog(products, hourly, 10, 42)
    other_seed, _ = subsample_catalog(products, hourly, 10, 99)

    assert [item.product_id for item in ten] == [item.product_id for item in twenty][:10]
    assert [item.product_id for item in ten] == [item.product_id for item in ten_again]
    assert [item.product_id for item in ten] != [item.product_id for item in other_seed]
    assert set(hourly_ten) == {item.category for item in ten}
    assert set(hourly_twenty) == {item.category for item in twenty}


def test_subsample_skips_when_pool_not_larger_than_n(tmp_path):
    db_path = tmp_path / "olist_v6_fixture.sqlite"
    write_olist_v6_fixture_db(str(db_path))
    products, hourly, _meta = load_dataset(str(db_path))
    same, same_hourly = subsample_catalog(products, hourly, len(products), 42)
    assert [item.product_id for item in same] == [item.product_id for item in products]
    assert set(same_hourly) == set(hourly)


def test_missing_private_real_db_raises(tmp_path):
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(PrivateRealDataError, match="not found"):
        load_dataset(str(missing))


def test_runner_subsamples_and_fails_closed_on_missing_db(tmp_path):
    db_path = tmp_path / "olist_v6_fixture.sqlite"
    write_olist_v6_fixture_db(str(db_path))
    app = create_app(
        db_path=str(tmp_path / "legacy.db"),
        runs_root=str(tmp_path / "runs"),
    )
    scenario = load_default_scenario()
    scenario["data"]["source"] = "private_real"
    scenario["data"]["private_real_db_path"] = str(db_path)
    scenario["data"]["num_products"] = 10
    scenario["run"]["master_seed"] = 42
    products, hourly, meta = app.registry._load_catalog_for_scenario(scenario)
    assert len(products) == 10
    assert scenario["data"]["num_products"] == 10
    assert scenario["data"]["num_suppliers"] == len({item.supplier_id for item in products})
    assert set(hourly) == set(scenario["data"]["category_pool"])
    assert meta["data_source"] == "private_real"
    assert scenario["data"]["dataset_id"].startswith("olist_v6_")
    assert scenario["data"]["source_label"] == "test_fixture"
    assert meta["dataset_id"] == scenario["data"]["dataset_id"]
    assert meta["source_label"] == "test_fixture"

    missing = load_default_scenario()
    missing["data"]["source"] = "private_real"
    missing["data"]["catalog_pool_path"] = str(tmp_path / "absent.sqlite")
    missing["data"]["num_products"] = 10
    with pytest.raises(PrivateRealDataError, match="not found"):
        app.registry._load_catalog_for_scenario(missing)


def test_synthetic_generate_is_not_double_sampled():
    scenario = load_default_scenario()
    scenario["data"]["num_products"] = 30
    scenario["data"]["source"] = "synthetic"
    products, _hourly = generate(scenario)
    assert len(products) == 30


def test_named_v6_scenario_is_real_catalog_and_default_stays_synthetic():
    default = load_default_scenario()
    named = load_scenario(str(SCENARIOS_DIR / "economy_v6.yaml"))
    both = load_scenario(str(SCENARIOS_DIR / "ablations" / "economy_v6_both.yaml"))

    assert default["data"]["source"] == "synthetic"
    assert default["economy_v6"]["enabled"] is False
    assert named["data"]["source"] == "private_real"
    assert named["data"]["num_products"] == 1000
    assert named["data"]["private_real_db_path"].endswith("olist_v6.sqlite")
    assert named["economy_v6"]["enabled"] is True
    assert named["economy_v6"]["take_rate"]["enabled"] is True
    assert named["economy_v6"]["fulfillment"]["enabled"] is True
    assert named["economy_v6"]["refund"]["enabled"] is True
    assert named["generation_params"]["risk_trust_coupling"] is True
    assert both["data"]["source"] == "synthetic"
    assert both["economy_v6"]["enabled"] is True
