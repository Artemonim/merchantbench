from pathlib import Path

import numpy as np
import pytest
from core.demand import (
    MAX_EXPECTED_DEMAND_PER_LISTING_STEP,
    EconomyV61,
    expected_demand,
    generate_orders_for_step,
)
from core.entities import Cash, Order, Product, StoreListing
from core.simulator import AgentState, Environment
from storage import db as dbm
from web.runner import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]

PLATFORM_RULES = {
    "cancel_penalty_amount": 0.0,
    "refund_penalty_amount": 8.0,
    "only_refund_penalty_amount": 0.0,
    "bad_review_penalty_amount": 5.0,
    "timeout_penalty_amount": 3.0,
    "stockout_penalty_amount": 5.0,
    "insufficient_balance_penalty_amount": 5.0,
    "max_promised_ship_hours": 48,
    "default_promised_ship_hours": 48,
}


def _mkproduct(ref_price=100.0, elasticity=2.0, market=1.0) -> Product:
    return Product(
        product_id="P0",
        name="x",
        quantity=10,
        price=80.0,
        ref_price=ref_price,
        supplier_id="s",
        supplier_name="S",
        ship_hours=1,
        logistics_hours=1,
        category="electronics",
        historical_avg_rating=4.5,
        shop_rating=4.5,
        return_buyer_rate=0.18,
        supplier_age_years=3.5,
        cancel_rate=0.0,
        refund_rate=0.0,
        only_refund_rate=0.0,
        bad_review_rate=0.0,
        max_quantity=100,
        hourly_increment=5,
        timeout_rate=0.0,
        price_change_rate=0.0,
        supplier_delist_rate=0.0,
        elasticity=elasticity,
        market_curve=[market] * 365,
    )


def _listing(sale_price: float) -> StoreListing:
    return StoreListing(product_id="P0", sale_price=sale_price, agent_id="agent_0")


def _hourly():
    return {"electronics": np.ones(24)}


def _demand_kwargs(scenario):
    cfg = EconomyV61.from_scenario(scenario)
    if not cfg.enabled:
        return {}
    return {
        "ces_multiplier_cap": cfg.ces_multiplier_cap,
        "max_expected_demand_per_listing_step": (cfg.max_expected_demand_per_listing_step),
    }


def test_flag_off_matches_missing_block_including_penny_price():
    product = _mkproduct()
    hourly = _hourly()
    disabled = {
        "economy_v6_1": {
            "enabled": False,
            "ces_multiplier_cap": 6.0,
            "violation_throttle_per_step": 5,
            "max_expected_demand_per_listing_step": 1000.0,
        }
    }
    assert _demand_kwargs({}) == {}
    assert _demand_kwargs(disabled) == {}
    for sale_price in (0.01, 100.0):
        listing = _listing(sale_price)
        q_missing = expected_demand(
            product,
            listing,
            hourly["electronics"],
            t=0,
            step_hours=1,
            small_share=1.0,
        )
        q_off = expected_demand(
            product,
            listing,
            hourly["electronics"],
            t=0,
            step_hours=1,
            small_share=1.0,
            **_demand_kwargs(disabled),
        )
        assert q_off == q_missing
        triples = [(product, listing, "agent_0")]
        o_missing = generate_orders_for_step(
            triples,
            hourly,
            t=0,
            step_hours=1,
            small_share=1.0,
            master_seed=42,
        )
        o_off = generate_orders_for_step(
            triples,
            hourly,
            t=0,
            step_hours=1,
            small_share=1.0,
            master_seed=42,
            **_demand_kwargs(disabled),
        )
        assert [o.order_id for o in o_off] == [o.order_id for o in o_missing]


def test_ces_cap_clamps_penny_price_to_m_times_base():
    product = _mkproduct(ref_price=100.0, elasticity=2.0, market=1.0)
    listing = _listing(0.01)
    hourly_w = np.ones(24)
    q_on = expected_demand(
        product,
        listing,
        hourly_w,
        t=0,
        step_hours=1,
        small_share=1.0,
        ces_multiplier_cap=6.0,
    )
    q_off = expected_demand(
        product,
        listing,
        hourly_w,
        t=0,
        step_hours=1,
        small_share=1.0,
    )
    assert q_off == MAX_EXPECTED_DEMAND_PER_LISTING_STEP
    assert q_on == pytest.approx(6.0)


def test_normal_discount_stays_below_cap():
    product = _mkproduct(ref_price=100.0, elasticity=2.0, market=1.0)
    hourly_w = np.ones(24)
    for sale_price in (50.0, 70.0):
        listing = _listing(sale_price)
        q_off = expected_demand(
            product,
            listing,
            hourly_w,
            t=0,
            step_hours=1,
            small_share=1.0,
        )
        q_on = expected_demand(
            product,
            listing,
            hourly_w,
            t=0,
            step_hours=1,
            small_share=1.0,
            ces_multiplier_cap=6.0,
        )
        ces = (sale_price / 100.0) ** (-2.0)
        assert ces < 6.0
        assert q_off == pytest.approx(ces)
        assert q_on == pytest.approx(q_off)


def test_max_expected_demand_knob_applies():
    product = _mkproduct(ref_price=100.0, elasticity=2.0, market=1.0)
    listing = _listing(0.01)
    hourly_w = np.ones(24)
    q = expected_demand(
        product,
        listing,
        hourly_w,
        t=0,
        step_hours=1,
        small_share=1.0,
        max_expected_demand_per_listing_step=10.0,
    )
    assert q == 10.0
    triples = [(product, listing, "agent_0")]
    orders = generate_orders_for_step(
        triples,
        _hourly(),
        t=0,
        step_hours=1,
        small_share=1.0,
        master_seed=1,
        max_expected_demand_per_listing_step=10.0,
    )
    assert len(orders) <= 40


def test_invalid_economy_v6_1_values_raise():
    with pytest.raises(ValueError, match="ces_multiplier_cap"):
        EconomyV61.from_scenario({"economy_v6_1": {"ces_multiplier_cap": 0}})
    with pytest.raises(ValueError, match="ces_multiplier_cap"):
        EconomyV61.from_scenario({"economy_v6_1": {"ces_multiplier_cap": "fast"}})
    with pytest.raises(ValueError, match="violation_throttle_per_step"):
        EconomyV61.from_scenario({"economy_v6_1": {"violation_throttle_per_step": -1}})
    with pytest.raises(ValueError, match="must be a mapping"):
        EconomyV61.from_scenario({"economy_v6_1": True})


def test_violation_throttle_zero_raises_regardless_of_enabled():
    for enabled in (True, False):
        with pytest.raises(
            ValueError,
            match=r"violation_throttle_per_step.*enabled: false",
        ):
            EconomyV61.from_scenario(
                {
                    "economy_v6_1": {
                        "enabled": enabled,
                        "violation_throttle_per_step": 0,
                    }
                }
            )


def _purchase_env(
    tmp_path,
    *,
    enabled=True,
    throttle_k=5,
    balance=0.0,
    listed_by_supplier=True,
):
    conn = dbm.open_db(str(tmp_path / "state.db"))
    product = _mkproduct()
    product.is_listed_by_supplier = listed_by_supplier
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    dbm.upsert_listing(conn, "v61", "agent_0", listing)
    state = AgentState(
        agent_id="agent_0",
        name="Agent",
        cash=Cash(balance, 1000.0),
        listings={"P0": listing},
    )
    scenario = {
        "run": {"step_hours": 1, "horizon_steps": 100, "master_seed": 42},
        "data": {"small_share": 0.01},
        "settlement": {"normal_delay_hours": 168},
        "platform_rules": PLATFORM_RULES,
        "supplier_ranges": {},
        "economy_v6_1": {
            "enabled": enabled,
            "ces_multiplier_cap": 6.0,
            "violation_throttle_per_step": throttle_k,
            "max_expected_demand_per_listing_step": 1000.0,
        },
    }
    env = Environment("v61", conn, scenario, str(tmp_path), [product], {}, {"agent_0": state})
    env._step_revenue = {aid: 0.0 for aid in env.agents}
    env._step_cost = {aid: 0.0 for aid in env.agents}
    return env


def _candidate(i: int) -> Order:
    return Order(
        order_id="O-{0}".format(i),
        product_id="P0",
        supplier_id="s",
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=2,
        sale_price=120.0,
        purchase_price=100.0,
    )


def test_violation_throttle_drops_extra_insufficient_balance(tmp_path):
    env = _purchase_env(tmp_path, enabled=True, throttle_k=5, balance=0.0)
    events = []
    kept = env._auto_purchase_new_orders(
        [_candidate(i) for i in range(12)],
        PLATFORM_RULES,
        events,
    )
    st = env.agents["agent_0"]
    assert len(kept) == 5
    assert all(o.current_status == "insufficient_balance" for o in kept)
    assert sum(1 for e in events if e.event_type == "order_insufficient_balance_violation") == 5
    assert st.cash.cumulative_fine == pytest.approx(25.0)
    assert st.cash.deposit_pool == pytest.approx(975.0)


def test_violation_throttle_drops_extra_stockout(tmp_path):
    # * Delisted supplier hits stockout before the cash check; extra candidates drop.
    env = _purchase_env(
        tmp_path,
        enabled=True,
        throttle_k=5,
        balance=10_000.0,
        listed_by_supplier=False,
    )
    events = []
    kept = env._auto_purchase_new_orders(
        [_candidate(i) for i in range(12)],
        PLATFORM_RULES,
        events,
    )
    st = env.agents["agent_0"]
    assert [o.order_id for o in kept] == ["O-{0}".format(i) for i in range(5)]
    assert all(o.current_status == "stockout" for o in kept)
    assert sum(1 for e in events if e.event_type == "order_stockout_violation") == 5
    assert st.cash.cumulative_fine == pytest.approx(25.0)
    assert st.cash.balance == pytest.approx(9_975.0)
    assert st.cash.deposit_pool == pytest.approx(1_000.0)
    assert st.cash.in_transit == pytest.approx(0.0)


def test_economy_v6_1_overlay_loads_enabled():
    scenario = load_scenario(str(REPO_ROOT / "env/scenarios/economy_v6_1.yaml"))
    block = scenario["economy_v6_1"]
    assert block["enabled"] is True
    assert block["ces_multiplier_cap"] == 6.0
    assert block["violation_throttle_per_step"] == 5
    assert block["max_expected_demand_per_listing_step"] == 1000.0
    assert scenario["economy_v6"]["enabled"] is True
