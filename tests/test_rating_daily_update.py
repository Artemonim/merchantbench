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
        product_id="p1",
        name="Widget",
        quantity=100,
        price=10.0,
        ref_price=15.0,
        supplier_id="s1",
        supplier_name="Supplier",
        ship_hours=1,
        logistics_hours=1,
        category="office",
        historical_avg_rating=5.0,
        shop_rating=5.0,
        return_buyer_rate=0.2,
        supplier_age_years=2.0,
        cancel_rate=0.0,
        refund_rate=0.0,
        only_refund_rate=0.0,
        bad_review_rate=0.0,
        max_quantity=100,
        hourly_increment=1,
        timeout_rate=0.0,
        price_change_rate=0.0,
        supplier_delist_rate=0.0,
        elasticity=1.0,
        market_curve=[1.0] * 365,
    )
    listing = StoreListing(product_id="p1", agent_id="agent_0", sale_price=15.0)
    dbm.upsert_listing(conn, run_id, "agent_0", listing)
    state = AgentState(
        agent_id="agent_0",
        name="Agent",
        cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    env = Environment(
        run_id,
        conn,
        _scenario(),
        str(tmp_path),
        [product],
        {"office": np.ones(24) / 24.0},
        {"agent_0": state},
    )
    return env, conn, state, listing


def _insert_terminal_order(conn, status, settled_t, *, order_id, late_t=None):
    conn.execute(
        "INSERT INTO orders(run_id,order_id,agent_id,product_id,supplier_id,"
        "order_t,promised_delivery_t,sale_price,purchase_price,current_status,"
        "settled_t,late_t) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "daily-rating",
            order_id,
            "agent_0",
            "p1",
            "s1",
            0,
            1,
            15.0,
            10.0,
            status,
            settled_t,
            late_t,
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
        agent_id="agent_0",
        name="Agent",
        cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    fresh = Environment(
        env.run_id,
        conn,
        env.scenario,
        env.runs_root,
        list(env.products.values()),
        env.hourly_dist,
        {"agent_0": fresh_state},
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


def test_v3_combines_recent_quality_with_lifetime_reputation(daily_env):
    env, _, state, _ = daily_env
    env.scenario["shop_rating"].update(
        {
            "model": "order_outcome_v3",
            "prior_weight": 0,
            "half_life_days": 180,
            "star_multipliers": [0.1, 0.35, 0.8, 1.0, 1.12],
            "reputation_volume": {
                "min_multiplier": 0.8,
                "max_multiplier": 1.0,
                "half_saturation_orders": 20,
            },
        }
    )
    state.shop_rating_sum = 4.5
    state.shop_rating_weight = 1.0
    state.shop_rating_order_count = 1

    first_order_state = env._shop_rating_state(state)
    assert first_order_state["score"] == pytest.approx(4.5)
    assert first_order_state["quality_multiplier"] == pytest.approx(1.12)
    assert first_order_state["reputation_multiplier"] == pytest.approx(
        0.8 + 0.2 / 21,
    )
    assert first_order_state["demand_multiplier"] < 1.0

    state.shop_rating_order_count = 20
    established_state = env._shop_rating_state(state)
    assert established_state["score"] == first_order_state["score"]
    assert established_state["reputation_multiplier"] == pytest.approx(0.9)
    assert established_state["demand_multiplier"] == pytest.approx(1.008)
    assert env._compute_rating_factors() == {
        "agent_0": pytest.approx(1.008),
    }
    metrics = env._shop_rating_metric_values(state)
    assert metrics["shop_quality_multiplier"] == pytest.approx(1.12)
    assert metrics["shop_reputation_multiplier"] == pytest.approx(0.9)
    assert metrics["shop_demand_multiplier"] == pytest.approx(1.008)
    assert metrics["shop_reputation_evidence_count"] == 20.0


def test_public_reviews_rebuild_daily_and_survive_rehydrate(daily_env):
    env, conn, state, _ = daily_env
    env.scenario["shop_rating"].update(
        {
            "model": "order_outcome_v4",
            "prior_weight": 0,
            "half_life_days": 180,
            "star_multipliers": [0.1, 0.35, 0.8, 1.0, 1.12],
        }
    )
    env.scenario["public_reviews"] = {
        "enabled": True,
        "model": "self_selection_v1",
        "probability_by_star": [1, 1, 1, 1, 1],
    }
    _insert_terminal_order(conn, "settled_normal", 3, order_id="normal")
    _insert_terminal_order(conn, "settled_bad_review", 4, order_id="bad")

    assert env._publish_daily_ratings(24)

    public_state = env._public_review_state(state)
    assert state.shop_rating_order_count == 2
    assert public_state == {
        "model": "self_selection_v1",
        "rating": 3.0,
        "count": 2,
        "eligible_count": 2,
        "response_rate": 1.0,
        "full_response_rating": 3.0,
        "selection_gap": 0.0,
        "quality_gap": pytest.approx(3.0 - (6.5 / 3.0)),
        "affects_demand": True,
        "stars": 2.0,
        "confidence": pytest.approx(2 / 22),
        "raw_quality_multiplier": 0.35,
        "quality_multiplier": pytest.approx(1.0 - 0.65 * (2 / 22)),
        "reputation_multiplier": pytest.approx(0.8 + 0.2 * (2 / 22)),
        "demand_multiplier": pytest.approx((1.0 - 0.65 * (2 / 22)) * (0.8 + 0.2 * (2 / 22))),
    }
    metrics = env._shop_rating_metric_values(state)
    assert metrics["public_review_rating"] == 3.0
    assert metrics["public_review_count"] == 2.0
    assert metrics["public_review_selection_gap"] == 0.0
    assert metrics["shop_reputation_evidence_count"] == 2.0
    assert metrics["shop_qualified_transaction_count"] == 2.0
    assert env._shop_rating_state(state)["demand_source"] == "public_reviews"
    assert env._compute_rating_factors()["agent_0"] == pytest.approx(public_state["demand_multiplier"])
    expected_demand = env._compute_rating_factors()["agent_0"]
    state.shop_rating_sum = 15.0
    state.shop_rating_weight = 3.0
    assert env._compute_rating_factors()["agent_0"] == pytest.approx(expected_demand)

    listing = dbm.get_listing(conn, env.run_id, "agent_0", "p1")
    fresh_state = AgentState(
        agent_id="agent_0",
        name="Agent",
        cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    fresh = Environment(
        env.run_id,
        conn,
        env.scenario,
        env.runs_root,
        list(env.products.values()),
        env.hourly_dist,
        {"agent_0": fresh_state},
    )
    fresh.t = 24
    fresh.restore_rating_state()

    assert fresh._public_review_state(fresh_state) == public_state


def test_v3_ignores_v4_public_review_counters(daily_env):
    env, _, state, _ = daily_env
    env.scenario["shop_rating"].update(
        {
            "model": "order_outcome_v3",
            "prior_weight": 0,
            "reputation_volume": {
                "min_multiplier": 0.8,
                "max_multiplier": 1.0,
                "half_saturation_orders": 20,
            },
        }
    )
    env.scenario["public_reviews"] = {
        "enabled": True,
        "probability_by_star": [0.3, 0.18, 0.08, 0.06, 0.12],
    }
    state.shop_rating_sum = 4.5
    state.shop_rating_weight = 1.0
    state.shop_rating_order_count = 20
    expected = env._compute_rating_factors()

    state.public_review_sum = 1.0
    state.public_review_count = 1
    state.public_review_eligible_sum = 100.0
    state.public_review_eligible_count = 20

    assert env._public_review_state(state) is None
    assert env._compute_rating_factors() == expected


def test_v4_rejects_a_disabled_public_review_policy(daily_env):
    env, _, state, _ = daily_env
    env.scenario["shop_rating"]["model"] = "order_outcome_v4"
    env.scenario["public_reviews"] = {"enabled": False}

    with pytest.raises(
        ValueError,
        match="order_outcome_v4 requires public_reviews.enabled=true",
    ):
        env._shop_rating_state(state)


def test_v4_cold_start_does_not_publish_internal_quality_as_a_rating(daily_env):
    env, _, state, _ = daily_env
    env.scenario["shop_rating"].update(
        {
            "model": "order_outcome_v4",
            "prior_weight": 0,
            "half_life_days": 180,
            "star_multipliers": [0.1, 0.35, 0.8, 1.0, 1.12],
        }
    )
    env.scenario["public_reviews"] = {
        "enabled": True,
        "model": "self_selection_v1",
        "probability_by_star": [0.3, 0.18, 0.08, 0.06, 0.12],
    }
    state.shop_rating_sum = 1.0
    state.shop_rating_weight = 1.0

    rating_state = env._shop_rating_state(state)
    metrics = env._shop_rating_metric_values(state)

    assert rating_state["score"] is None
    assert rating_state["stars"] is None
    assert rating_state["rating_available"] is False
    assert rating_state["service_quality_score"] == 1.0
    assert rating_state["service_quality_stars"] == 1.0
    assert rating_state["quality_multiplier"] == 1.0
    assert rating_state["reputation_multiplier"] == 0.8
    assert rating_state["demand_multiplier"] == 0.8
    assert "shop_rating_score" not in metrics
    assert "shop_rating_stars" not in metrics
    assert "shop_rating_mean" not in metrics
    assert metrics["shop_service_quality_score"] == 1.0
    assert metrics["public_review_count"] == 0.0


def test_v2_rating_metrics_are_sparse_daily_points(daily_env):
    env, conn, _, _ = daily_env
    env.t = 0
    env._write_metrics([], [], [], write_rating_metrics=True)
    env.t = 1
    env._write_metrics([], [], [], write_rating_metrics=False)
    env.t = 23
    env._write_metrics([], [], [], write_rating_metrics=True)

    series = dbm.load_metrics_bulk(
        conn,
        env.run_id,
        "agent_0",
        [
            "shop_rating_mean",
            "shop_rating_score",
            "shop_rating_stars",
            "shop_rating_order_count",
            "avg_listing_rating",
            "net_assets",
        ],
    )
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
    assert (
        dbm.load_metric_series(
            conn,
            env.run_id,
            "agent_0",
            "shop_rating_mean",
        )[-1][0]
        == 29
    )
    assert not env.publish_terminal_ratings()


def test_v2_finished_partial_day_rehydrate_keeps_terminal_rating(daily_env):
    env, conn, _, _ = daily_env
    _insert_terminal_order(conn, "settled_bad_review", 25, order_id="partial")
    env.t = 30
    assert env.publish_terminal_ratings()
    expected = env._shop_rating_value(env.agents["agent_0"])

    listing = dbm.get_listing(conn, env.run_id, "agent_0", "p1")
    fresh_state = AgentState(
        agent_id="agent_0",
        name="Agent",
        cash=Cash(1000.0, 500.0),
        listings={"p1": listing},
    )
    fresh = Environment(
        env.run_id,
        conn,
        env.scenario,
        env.runs_root,
        list(env.products.values()),
        env.hourly_dist,
        {"agent_0": fresh_state},
    )
    fresh.t = 30
    fresh.restore_rating_state(include_terminal_partial_day=True)

    assert fresh_state.shop_rating_published_t == 48
    assert fresh_state.shop_rating_order_count == 1
    assert fresh._shop_rating_value(fresh_state) == pytest.approx(expected)
    assert fresh_state.listings["p1"].rating_count == pytest.approx(2.0)
