import numpy as np

from core.demand import (
    MAX_EXPECTED_DEMAND_PER_LISTING_STEP,
    expected_demand,
    generate_orders_for_step,
)
from core.entities import StoreListing
from data.synth import generate
from web.runner import load_default_scenario


def test_expected_demand_scales_with_price():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    p = products[0]
    listing_at_ref = StoreListing(product_id=p.product_id, sale_price=p.ref_price)
    listing_higher = StoreListing(product_id=p.product_id, sale_price=p.ref_price * 1.5)
    w = hourly_dist[p.category]
    q_ref = expected_demand(p, listing_at_ref, w, t=0, step_hours=1, small_share=0.0005)
    q_high = expected_demand(p, listing_higher, w, t=0, step_hours=1, small_share=0.0005)
    # higher price => lower demand
    assert q_high < q_ref


def test_expected_demand_fails_closed_for_sub_currency_price():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    product = products[0]
    listing = StoreListing(product_id=product.product_id, sale_price=5e-324)

    assert expected_demand(
        product,
        listing,
        hourly_dist[product.category],
        t=0,
        step_hours=1,
        small_share=0.0005,
    ) == 0.0


def test_expected_demand_saturates_extreme_valid_discount():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    product = products[0]
    product.ref_price = 12_798.6813
    product.elasticity = 6.0
    product.market_curve = [1e12] * 365
    listing = StoreListing(product_id=product.product_id, sale_price=0.01)

    q = expected_demand(
        product,
        listing,
        hourly_dist[product.category],
        t=0,
        step_hours=1,
        small_share=1.0,
    )

    assert q == MAX_EXPECTED_DEMAND_PER_LISTING_STEP
    assert np.isfinite(q)


def test_expected_demand_preserves_daily_curve_after_hour_split():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    p = products[0]
    listing = StoreListing(product_id=p.product_id, sale_price=p.ref_price)
    small_share = 0.01
    w = hourly_dist[p.category]

    q_day = sum(
        expected_demand(p, listing, w, t=hour, step_hours=1, small_share=small_share)
        for hour in range(24)
    )

    assert np.isclose(q_day, p.market_curve[0] * small_share)


def test_arrivals_deterministic():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    for p in products[:5]:
        p.market_curve = [v * 1000 for v in p.market_curve]
    listings = [StoreListing(product_id=p.product_id, sale_price=p.ref_price) for p in products[:5]]
    triples = [(p, l, "agent_0") for p, l in zip(products[:5], listings)]
    o1 = generate_orders_for_step(triples, hourly_dist, t=10, step_hours=1,
                                   small_share=0.0005, master_seed=42)
    o2 = generate_orders_for_step(triples, hourly_dist, t=10, step_hours=1,
                                   small_share=0.0005, master_seed=42)
    assert len(o1) == len(o2)
    assert [o.order_id for o in o1] == [o.order_id for o in o2]


def test_product_rating_does_not_affect_arrivals():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    product = products[0]
    product.market_curve = [v * 1000 for v in product.market_curve]
    strong = StoreListing(
        product_id=product.product_id,
        sale_price=product.ref_price,
        rating_sum=500.0,
        rating_count=100.0,
    )
    weak = StoreListing(
        product_id=product.product_id,
        sale_price=product.ref_price,
        rating_sum=10.0,
        rating_count=100.0,
    )

    strong_orders = generate_orders_for_step(
        [(product, strong, "agent_0")], hourly_dist, t=10, step_hours=1,
        small_share=0.0005, master_seed=42,
    )
    weak_orders = generate_orders_for_step(
        [(product, weak, "agent_0")], hourly_dist, t=10, step_hours=1,
        small_share=0.0005, master_seed=42,
    )

    assert [o.order_id for o in strong_orders] == [o.order_id for o in weak_orders]
    assert [o.preset_anomaly for o in strong_orders] == [o.preset_anomaly for o in weak_orders]


def test_arrivals_change_with_seed():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    for p in products[:5]:
        p.market_curve = [v * 1000 for v in p.market_curve]
    listings = [StoreListing(product_id=p.product_id, sale_price=p.ref_price) for p in products[:5]]
    triples = [(p, l, "agent_0") for p, l in zip(products[:5], listings)]
    o1 = generate_orders_for_step(triples, hourly_dist, t=10, step_hours=1,
                                   small_share=0.0005, master_seed=42)
    o2 = generate_orders_for_step(triples, hourly_dist, t=10, step_hours=1,
                                   small_share=0.0005, master_seed=43)
    # different seed should change at least the order count or the anomaly mix
    counts_differ = len(o1) != len(o2)
    anomalies_differ = sorted(o.preset_anomaly for o in o1) != sorted(o.preset_anomaly for o in o2)
    assert counts_differ or anomalies_differ or len(o1) == 0


def test_per_agent_independent_streams():
    scenario = load_default_scenario()
    products, hourly_dist = generate(scenario)
    p = products[0]
    p.market_curve = [v * 1000 for v in p.market_curve]
    listing = StoreListing(product_id=p.product_id, sale_price=p.ref_price)
    t_a = [(p, listing, "agent_0")]
    t_b = [(p, listing, "agent_1")]
    o_a = generate_orders_for_step(t_a, hourly_dist, t=10, step_hours=1,
                                    small_share=0.0005, master_seed=42)
    o_b = generate_orders_for_step(t_b, hourly_dist, t=10, step_hours=1,
                                    small_share=0.0005, master_seed=42)
    assert all(o.agent_id == "agent_0" for o in o_a)
    assert all(o.agent_id == "agent_1" for o in o_b)
    # different agent_id => different RNG seed => order_ids prefixed differently
    if o_a:
        assert o_a[0].order_id.startswith("agent_0-")
    if o_b:
        assert o_b[0].order_id.startswith("agent_1-")


def test_normal_and_bad_review_get_random_settlement_delay():
    from core.demand import _make_order
    from core.entities import Product, StoreListing

    p = Product(
        product_id="P0", name="x", quantity=10,
        price=100.0, ref_price=100.0, supplier_id="s", supplier_name="S",
        ship_hours=1, logistics_hours=1, category="electronics",
        historical_avg_rating=4.5, shop_rating=4.5,
        return_buyer_rate=0.18, supplier_age_years=3.5,
        cancel_rate=0.0, refund_rate=0.0, only_refund_rate=0.0, bad_review_rate=0.0,
        max_quantity=100, hourly_increment=5,
        timeout_rate=0.0, price_change_rate=0.0, supplier_delist_rate=0.0,
        elasticity=1.0, market_curve=[1.0] * 365,
    )
    listing = StoreListing(product_id="P0", sale_price=100.0)
    normal_delays = {_make_order(p, listing, "agent_0", t=0, i=i, master_seed=42, normal_delay_steps=168).settlement_delay_steps
                     for i in range(50)}
    assert all(0 <= d <= 168 for d in normal_delays)
    assert len(normal_delays) > 1  # not all identical

    p_bad = Product(**{**p.__dict__, "bad_review_rate": 1.0, "cancel_rate": 0.0, "refund_rate": 0.0, "only_refund_rate": 0.0})
    bad_delays = {_make_order(p_bad, listing, "agent_0", t=0, i=i, master_seed=42, normal_delay_steps=168).settlement_delay_steps
                  for i in range(50)}
    assert all(0 <= d <= 168 for d in bad_delays)
    assert len(bad_delays) > 1
