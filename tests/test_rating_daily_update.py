import sqlite3

import numpy as np
import pytest

from core.entities import Cash, Product, StoreListing
from core.listing_rating import compute_listing_rating
from core.simulator import AgentState, Environment
from storage import db as dbm


def _scenario():
    return {
        "run": {"step_hours": 1, "horizon_steps": 100, "master_seed": 42},
        "data": {"small_share": 0.01},
        "settlement": {"normal_delay_hours": 168},
        "platform_rules": {
            "default_promised_ship_hours": 48,
            "max_active_listings": 10,
        },
        "supplier_ranges": {},
        "rating_outcomes": {
            "normal_score": 4.5,
            "late_score": 3.0,
            "refund_score": 2.0,
            "only_refund_score": 1.5,
            "bad_review_score": 1.0,
            "stockout_score": 1.0,
            "normal_weight": 1.0,
            "late_weight": 1.0,
            "refund_weight": 1.0,
            "only_refund_weight": 2.0,
            "bad_review_weight": 2.0,
            "stockout_weight": 3.0,
        },
        "listing_rating": {
            "initial_rating": 4.0,
            "prior_weight": 20,
            "half_life_days": 90,
        },
        "shop_rating": {
            "enabled": True,
            "model": "order_outcome_v2",
            "initial_rating": 4.0,
            "prior_weight": 20,
            "half_life_days": 30,
            "bucket_thresholds": [2.5, 3.3, 3.8, 4.2],
            "star_multipliers": [0.1, 0.35, 0.8, 1.0, 1.2],
        },
    }


@pytest.fixture()
def daily_env(tmp_path):
    conn = dbm.open_db(str(tmp_path / "state.db"))
    run_id = "daily-rating"
    conn.execute(
        "INSERT INTO runs(run_id,name,scenario_yaml,master_seed,current_t,horizon,"
        "step_hours,status) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, "daily", "{}", 42, 0, 100, 1, "running"),
    )
    product = Product(
        product_id="p1", name="Widget", quantity=100, price=10.0,
        ref_price=15.0, supplier_id="s1", supplier_name="Supplier",
        ship_hours=1, logistics_hours=1, category="office",
        historical_avg_rating=5.0, shop_rating=5.0,
        return_buyer_rate=0.2, supplier_age_years=2.0,
        cancel_rate=0.0, refund_rate=0.0, only_refund_rate=0.0,
        bad_review_rate=0.0, max_quantity=100, hourly_increment=1,
        timeout_rate=0.0, price_change_rate=0.0, supplier_delist_rate=0.0,
        elasticity=1.0, market_curve=[1.0] * 365,
    )
    listing = StoreListing(product_id="p1", agent_id="agent_0", sale_price=15.0)
    dbm.upsert_listing(conn, run_id, "agent_0", listing)
    state = AgentState(
        agent_id="agent_0", name="Agent", cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    env = Environment(
        run_id, conn, _scenario(), str(tmp_path), [product],
        {"office": np.ones(24) / 24.0}, {"agent_0": state},
    )
    return env, conn, state, listing


def _insert_terminal_order(conn, status, settled_t, *, order_id, late_t=None):
    conn.execute(
        "INSERT INTO orders(run_id,order_id,agent_id,product_id,supplier_id,"
        "order_t,promised_delivery_t,sale_price,purchase_price,current_status,"
        "settled_t,late_t) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "daily-rating", order_id, "agent_0", "p1", "s1", 0, 1,
            15.0, 10.0, status, settled_t, late_t,
        ),
    )


def test_v2_rating_stays_at_prior_until_completed_day(daily_env):
    env, conn, state, listing = daily_env
    _insert_terminal_order(conn, "settled_bad_review", 5, order_id="bad")

    env.t = 22
    assert env._completed_day_cutoff_after_step() is None
    assert env._shop_rating_value(state) == 4.0
    assert compute_listing_rating(4.0, listing.rating_sum, listing.rating_count, 20) == 4.0


def test_v2_publishes_weighted_product_and_shop_rating_at_day_end(daily_env):
    env, conn, state, listing = daily_env
    _insert_terminal_order(conn, "settled_bad_review", 5, order_id="bad")
    _insert_terminal_order(conn, "settled_only_refund", 6, order_id="only")
    _insert_terminal_order(conn, "settled_normal", 7, order_id="normal")

    assert env._publish_daily_ratings(24)

    # bad review=(1*2, 2), only refund=(1.5*2, 2), normal=(4.5, 1)
    assert state.shop_rating_sum == pytest.approx(9.5)
    assert state.shop_rating_weight == pytest.approx(5.0)
    assert state.shop_rating_order_count == 3
    assert state.shop_rating_published_t == 24
    env.t = 24
    assert env._shop_rating_updated_through_step(state) == 23
    assert listing.rating_sum == pytest.approx(9.5)
    assert listing.rating_count == pytest.approx(5.0)
    assert env._shop_rating_value(state) == pytest.approx((80.0 + 9.5) / 25.0)


def test_v2_rehydrate_matches_daily_publication(daily_env):
    env, conn, _, _ = daily_env
    _insert_terminal_order(conn, "settled_normal", 3, order_id="normal")
    _insert_terminal_order(conn, "settled_bad_review", 25, order_id="future")
    env.t = 24
    env._publish_daily_ratings(24)
    expected = env._shop_rating_value(env.agents["agent_0"])

    listing = dbm.get_listing(conn, env.run_id, "agent_0", "p1")
    fresh_state = AgentState(
        agent_id="agent_0", name="Agent", cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    fresh = Environment(
        env.run_id, conn, env.scenario, env.runs_root,
        list(env.products.values()), env.hourly_dist, {"agent_0": fresh_state},
    )
    fresh.t = 24
    fresh.restore_rating_state()

    assert fresh._shop_rating_value(fresh_state) == pytest.approx(expected)
    assert fresh_state.shop_rating_order_count == 1
    assert fresh_state.listings["p1"].rating_count == pytest.approx(1.0)


def test_v2_shop_bucket_controls_demand_factor(daily_env):
    env, _, state, _ = daily_env
    state.shop_rating_sum = 55.0
    state.shop_rating_weight = 20.0
    # (80 + 55) / 40 = 3.375 => 3★ => 0.8x.
    assert env._shop_rating_value(state) == pytest.approx(3.375)
    assert env._compute_rating_factors() == {"agent_0": 0.8}


def test_v2_rating_metrics_are_sparse_daily_points(daily_env):
    env, conn, _, _ = daily_env
    env.t = 0
    env._write_metrics([], [], [], write_rating_metrics=True)
    env.t = 1
    env._write_metrics([], [], [], write_rating_metrics=False)
    env.t = 23
    env._write_metrics([], [], [], write_rating_metrics=True)

    series = dbm.load_metrics_bulk(conn, env.run_id, "agent_0", [
        "shop_rating_mean",
        "shop_rating_score",
        "shop_rating_stars",
        "shop_rating_order_count",
        "avg_listing_rating",
        "net_assets",
    ])
    assert series["shop_rating_mean"] == [(0, 4.0), (23, 4.0)]
    assert series["shop_rating_score"] == [(0, 4.0), (23, 4.0)]
    assert series["shop_rating_stars"] == [(0, 4.0), (23, 4.0)]
    assert series["shop_rating_order_count"] == [(0, 0.0), (23, 0.0)]
    assert series["avg_listing_rating"] == [(0, 4.0), (23, 4.0)]
    assert [t for t, _ in series["net_assets"]] == [0, 1, 23]


def test_v2_terminal_flush_closes_partial_day(daily_env):
    env, conn, state, _ = daily_env
    _insert_terminal_order(conn, "settled_bad_review", 25, order_id="partial")
    env.t = 30

    assert env.publish_terminal_ratings()
    assert state.shop_rating_published_t == 48
    assert env._shop_rating_updated_through_step(state) == 29
    assert state.shop_rating_order_count == 1
    assert dbm.load_metric_series(
        conn, env.run_id, "agent_0", "shop_rating_mean",
    )[-1][0] == 29
    assert not env.publish_terminal_ratings()


def test_v2_finished_partial_day_rehydrate_keeps_terminal_rating(daily_env):
    env, conn, _, _ = daily_env
    _insert_terminal_order(conn, "settled_bad_review", 25, order_id="partial")
    env.t = 30
    assert env.publish_terminal_ratings()
    expected = env._shop_rating_value(env.agents["agent_0"])

    listing = dbm.get_listing(conn, env.run_id, "agent_0", "p1")
    fresh_state = AgentState(
        agent_id="agent_0", name="Agent", cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    fresh = Environment(
        env.run_id, conn, env.scenario, env.runs_root,
        list(env.products.values()), env.hourly_dist, {"agent_0": fresh_state},
    )
    fresh.t = 30
    fresh.restore_rating_state(include_terminal_partial_day=True)

    assert fresh_state.shop_rating_published_t == 48
    assert fresh_state.shop_rating_order_count == 1
    assert fresh._shop_rating_value(fresh_state) == pytest.approx(expected)
    assert fresh_state.listings["p1"].rating_count == pytest.approx(2.0)
