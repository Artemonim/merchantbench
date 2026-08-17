"""Economy v6 overlays: risk↔trust coupling and fee-contribution diagnostics."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.economy_v6 import EconomyV6, public_return_rate
from core.entities import Product
from data.economy_diagnostics import (
    expected_contribution_at_ref,
    expected_contribution_at_sale,
    share_negative_contribution_at_ref,
)
from data.generation_profiles import apply_risk_trust_coupling
from data.synth import generate
from web.runner import load_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "env" / "scenarios"
ABLATIONS_DIR = SCENARIOS_DIR / "ablations"
MASTER_SEED = 42

_PREFIX_FIELDS = (
    "product_id",
    "name",
    "quantity",
    "price",
    "ref_price",
    "supplier_id",
    "ship_hours",
    "category",
    "historical_avg_rating",
    "shop_rating",
    "return_buyer_rate",
    "supplier_age_years",
    "cancel_rate",
    "max_quantity",
    "hourly_increment",
    "timeout_rate",
    "price_change_rate",
    "supplier_delist_rate",
    "elasticity",
)
_COUPLED_FIELDS = (
    "refund_rate",
    "only_refund_rate",
    "bad_review_rate",
    "logistics_hours",
)
_ALL_SNAPSHOT_FIELDS = _PREFIX_FIELDS + _COUPLED_FIELDS


def _snapshot(products, fields=_ALL_SNAPSHOT_FIELDS):
    rows = []
    for product in products:
        row = tuple(getattr(product, field) for field in fields)
        curve = tuple(product.market_curve)
        rows.append(row + (curve,))
    return rows


def _load_generated(path, **generation_overrides):
    scenario = load_scenario(str(path))
    scenario["run"]["master_seed"] = MASTER_SEED
    for key, value in generation_overrides.items():
        scenario.setdefault("generation_params", {})[key] = value
    products, _hourly = generate(scenario)
    return scenario, products


def _risk_ranges(scenario):
    return (
        scenario["risk_ranges"],
        scenario["supplier_ranges"],
        scenario["supplier_profile_ranges"],
        scenario["product_profile_ranges"],
    )


def _mkproduct(**overrides) -> Product:
    fields = dict(
        product_id="P0",
        name="x",
        quantity=10,
        price=70.0,
        ref_price=100.0,
        supplier_id="s",
        supplier_name="S",
        ship_hours=12,
        logistics_hours=36,
        category="office",
        historical_avg_rating=4.25,
        shop_rating=4.25,
        return_buyer_rate=0.18,
        supplier_age_years=3.5,
        cancel_rate=0.04,
        refund_rate=0.06,
        only_refund_rate=0.0175,
        bad_review_rate=0.055,
        max_quantity=100,
        hourly_increment=5,
        timeout_rate=0.001,
        price_change_rate=0.002,
        supplier_delist_rate=0.001,
        elasticity=1.4,
        market_curve=[1.0] * 365,
    )
    fields.update(overrides)
    return Product(**fields)


def test_default_catalog_is_bit_identical_when_coupling_is_off():
    default_scenario, default_products = _load_generated(SCENARIOS_DIR / "default.yaml")
    assert default_scenario["generation_params"]["risk_trust_coupling"] is False
    _, explicit_off = _load_generated(
        SCENARIOS_DIR / "default.yaml", risk_trust_coupling=False
    )
    missing = load_scenario(str(SCENARIOS_DIR / "default.yaml"))
    del missing["generation_params"]["risk_trust_coupling"]
    missing["run"]["master_seed"] = MASTER_SEED
    missing_products, _ = generate(missing)

    assert _snapshot(default_products) == _snapshot(explicit_off)
    assert _snapshot(default_products) == _snapshot(missing_products)


def test_coupling_preserves_v5_prefix_and_supplier_profiles():
    _, off_products = _load_generated(
        SCENARIOS_DIR / "default.yaml", risk_trust_coupling=False
    )
    _, on_products = _load_generated(
        SCENARIOS_DIR / "default.yaml", risk_trust_coupling=True
    )

    assert _snapshot(off_products, _PREFIX_FIELDS) == _snapshot(
        on_products, _PREFIX_FIELDS
    )
    assert _snapshot(off_products, _COUPLED_FIELDS) != _snapshot(
        on_products, _COUPLED_FIELDS
    )

    by_sup: dict[str, list[Product]] = {}
    for product in on_products:
        by_sup.setdefault(product.supplier_id, []).append(product)
    for supplier_id, group in by_sup.items():
        ratings = {item.shop_rating for item in group}
        buyers = {item.return_buyer_rate for item in group}
        ages = {item.supplier_age_years for item in group}
        assert len(ratings) == 1, supplier_id
        assert len(buyers) == 1, supplier_id
        assert len(ages) == 1, supplier_id


def test_coupling_makes_shop_rating_predict_return_rate_and_logistics():
    _, products = _load_generated(
        ABLATIONS_DIR / "economy_v6_both.yaml"
    )
    assert products
    ratings = sorted(product.shop_rating for product in products)
    mid = ratings[len(ratings) // 2]
    low = [p for p in products if p.shop_rating <= mid]
    high = [p for p in products if p.shop_rating > mid]
    assert low and high

    def _mean_return(group):
        return sum(
            public_return_rate(item.refund_rate, item.only_refund_rate)
            for item in group
        ) / float(len(group))

    def _mean_hours(group):
        return sum(item.logistics_hours for item in group) / float(len(group))

    def _mean_bad(group):
        return sum(item.bad_review_rate for item in group) / float(len(group))

    assert _mean_return(low) > _mean_return(high)
    assert _mean_hours(low) > _mean_hours(high)
    assert _mean_bad(low) > _mean_bad(high)

    hist = sorted(product.historical_avg_rating for product in products)
    hist_mid = hist[len(hist) // 2]
    low_hist = [p for p in products if p.historical_avg_rating <= hist_mid]
    high_hist = [p for p in products if p.historical_avg_rating > hist_mid]
    assert _mean_return(low_hist) > _mean_return(high_hist)

    for product in products:
        assert 0.0 <= product.refund_rate <= 1.0
        assert 0.0 <= product.only_refund_rate <= 1.0
        assert 0.0 <= product.bad_review_rate <= 1.0
        assert 12 <= product.logistics_hours <= 72


def test_apply_risk_trust_coupling_is_deterministic_and_clamped():
    scenario = load_scenario(str(SCENARIOS_DIR / "default.yaml"))
    risk_ranges, supplier_ranges, supplier_profile, product_profile = _risk_ranges(
        scenario
    )
    low = _mkproduct(shop_rating=3.5, historical_avg_rating=3.5, refund_rate=0.06)
    high = _mkproduct(shop_rating=5.0, historical_avg_rating=5.0, refund_rate=0.06)
    apply_risk_trust_coupling(
        low,
        risk_ranges=risk_ranges,
        supplier_ranges=supplier_ranges,
        supplier_profile_ranges=supplier_profile,
        product_profile_ranges=product_profile,
    )
    apply_risk_trust_coupling(
        high,
        risk_ranges=risk_ranges,
        supplier_ranges=supplier_ranges,
        supplier_profile_ranges=supplier_profile,
        product_profile_ranges=product_profile,
    )
    assert low.shop_rating == 3.5
    assert high.shop_rating == 5.0
    assert low.return_buyer_rate == high.return_buyer_rate
    assert low.refund_rate > high.refund_rate
    assert low.only_refund_rate > high.only_refund_rate
    assert low.bad_review_rate > high.bad_review_rate
    assert low.logistics_hours > high.logistics_hours
    assert 0.02 <= low.refund_rate <= 0.10
    assert 0.02 <= high.refund_rate <= 0.10
    assert 12 <= low.logistics_hours <= 72
    assert 12 <= high.logistics_hours <= 72


def test_v6_fees_make_some_ref_contributions_negative():
    scenario, products = _load_generated(SCENARIOS_DIR / "default.yaml")
    v5 = EconomyV6.from_scenario(scenario)
    v6 = EconomyV6.from_scenario(
        load_scenario(str(ABLATIONS_DIR / "economy_v6_fees_only.yaml"))
    )
    assert share_negative_contribution_at_ref(products, v5) == pytest.approx(0.0)
    assert share_negative_contribution_at_ref(products, v6) > 0.0

    office = SimpleNamespace(price=70.0, ref_price=100.0, category="office")
    assert expected_contribution_at_ref(office, v5) == pytest.approx(30.0)
    # * office τ=0.07, F=7 → 100*(1-0.07) - 70 - 7 = 16.
    assert expected_contribution_at_sale(office, 100.0, v6) == pytest.approx(16.0)
