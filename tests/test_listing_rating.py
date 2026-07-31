"""Unit and integration tests for dynamic listing_rating."""
import pytest

from core import listing_rating as lr_mod


_PRIOR_WEIGHT = 10.0
_BUCKET_THRESHOLDS = [2.0, 3.0, 4.0, 4.6]
_RATING_MULTIPLIERS = [0.10, 0.30, 0.50, 1.00, 1.20]


class TestComputeListingRating:
    def test_initial_rating_uses_merchant_listing_prior(self):
        lr = lr_mod.compute_listing_rating(
            initial_rating=4.0,
            rating_sum=0.0,
            rating_count=0,
            prior_weight=_PRIOR_WEIGHT,
        )
        assert lr == pytest.approx(4.0)

    def test_normal_settlements_raise_rating(self):
        base = lr_mod.compute_listing_rating(4.0, 0.0, 0, _PRIOR_WEIGHT)
        after = lr_mod.compute_listing_rating(4.0, 50.0, 10, _PRIOR_WEIGHT)
        assert after > base

    def test_bad_outcomes_lower_rating(self):
        base = lr_mod.compute_listing_rating(4.0, 0.0, 0, _PRIOR_WEIGHT)
        after = lr_mod.compute_listing_rating(4.0, 10.0, 10, _PRIOR_WEIGHT)
        assert after < base

    @pytest.mark.parametrize(
        "status,late_t,expected",
        [
            ("settled_normal", None, 4.5),
            ("settled_normal", 3, 3.0),
            ("settled_refund", None, 2.0),
            ("settled_only_refund", None, 1.5),
            ("settled_bad_review", None, 1.0),
            ("stockout", None, 1.0),
            ("cancelled", None, None),
            ("insufficient_balance", None, None),
        ],
    )
    def test_final_order_outcome_score(self, status, late_t, expected):
        assert lr_mod.score_for_order_outcome(status, late_t) == expected

    def test_final_outcome_counts_once_even_if_late(self):
        assert lr_mod.score_for_order_outcome("settled_refund", late_t=3) == 2.0
        assert lr_mod.score_for_order_outcome("settled_bad_review", late_t=3) == 1.0

    @pytest.mark.parametrize(
        "status,late_t,expected",
        [
            ("settled_normal", None, (4.5, 1.0)),
            ("settled_normal", 3, (3.0, 1.0)),
            ("settled_refund", None, (2.0, 1.0)),
            ("settled_only_refund", None, (3.0, 2.0)),
            ("settled_bad_review", None, (2.0, 2.0)),
            ("stockout", None, (3.0, 3.0)),
            ("cancelled", None, None),
        ],
    )
    def test_weighted_order_contribution(self, status, late_t, expected):
        assert lr_mod.outcome_contribution(status, late_t) == expected

    def test_daily_decay_from_half_life(self):
        decay = lr_mod.decay_from_half_life(30)
        assert decay ** 30 == pytest.approx(0.5)

    def test_rebuild_evidence_uses_completed_days_and_counts_order_once(self):
        rows = [
            ("p1", "settled_normal", None, 0),
            ("p1", "settled_refund", 3, 23),
            ("p1", "settled_bad_review", 3, 24),
        ]
        out = lr_mod.rebuild_evidence(
            rows,
            cutoff_t=24,
            step_hours=1,
            half_life_days=30,
        )
        # The late refund contributes only its final refund result.  The order
        # at t=24 belongs to the incomplete next day and is excluded.
        assert out["p1"] == pytest.approx((6.5, 2.0, 2))

    def test_rating_stays_in_1_to_5(self):
        lr_low = lr_mod.compute_listing_rating(4.0, -1000.0, 1000, _PRIOR_WEIGHT)
        lr_high = lr_mod.compute_listing_rating(4.0, 5000.0, 1000, _PRIOR_WEIGHT)
        assert lr_low >= 1.0
        assert lr_high <= 5.0

    def test_prior_weight_controls_stability(self):
        diff_low = (
            lr_mod.compute_listing_rating(4.0, 25.0, 5, prior_weight=2.0)
            - lr_mod.compute_listing_rating(4.0, 0.0, 0, prior_weight=2.0)
        )
        diff_high = (
            lr_mod.compute_listing_rating(4.0, 25.0, 5, prior_weight=100.0)
            - lr_mod.compute_listing_rating(4.0, 0.0, 0, prior_weight=100.0)
        )
        assert diff_low > diff_high

    def test_formula_correctness(self):
        initial = 4.0
        rating_sum, rating_count = 82.5, 20
        pw = 10.0
        expected = (pw * initial + rating_sum) / (pw + rating_count)
        actual = lr_mod.compute_listing_rating(initial, rating_sum, rating_count, pw)
        assert actual == pytest.approx(expected)


class TestListingRatingIntegration:
    """Integration tests using a minimal Environment."""

    @pytest.fixture()
    def env_setup(self):
        """Create a minimal environment with one agent and one product."""
        import sqlite3
        import threading
        import numpy as np
        from core.entities import Cash, Product, StoreListing
        from core.simulator import AgentState, Environment
        from storage import db as dbm

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(dbm.SCHEMA)

        product = Product(
            product_id="p1", name="Widget", quantity=100, price=10.0,
            ref_price=15.0, supplier_id="s1", supplier_name="Sup",
            ship_hours=24, logistics_hours=48, category="electronics",
            historical_avg_rating=4.2, shop_rating=4.5,
            return_buyer_rate=0.3, supplier_age_years=2.0,
            cancel_rate=0.02, refund_rate=0.02,
            only_refund_rate=0.01, bad_review_rate=0.05,
            max_quantity=200, hourly_increment=5,
            timeout_rate=0.01, price_change_rate=0.01,
            supplier_delist_rate=0.01, elasticity=1.5,
            market_curve=[1.0] * 365,
        )

        listing = StoreListing(
            product_id="p1", agent_id="agent_0",
            sale_price=15.0, listed_at=0, first_listed_at=0,
            normal_count=0, bad_review_count=0,
            rating_sum=0.0, rating_count=0,
        )

        scenario = {
            "run": {"step_hours": 1, "horizon_steps": 100, "master_seed": 42},
            "data": {"small_share": 0.01},
            "settlement": {"normal_delay_hours": 168},
            "platform_rules": {"default_promised_ship_hours": 48, "max_active_listings": 100,
                               "bad_review_penalty_amount": 5.0},
            "shop_rating": {"enabled": True, "prior_good": 500.0, "prior_bad": 50.0,
                            "decay": 0.999, "bucket_thresholds": [0.5, 0.7, 0.85, 0.95],
                            "star_multipliers": [0.4, 0.7, 1.0, 1.2, 1.5]},
            "listing_rating": {"initial_rating": 4.0, "prior_weight": 10,
                               "bucket_thresholds": _BUCKET_THRESHOLDS,
                               "rating_multipliers": _RATING_MULTIPLIERS},
            "supplier_ranges": {},
        }

        agent_state = AgentState(
            agent_id="agent_0", name="Test Agent",
            cash=Cash(balance=1000.0, deposit_pool=500.0),
            listings={"p1": listing},
        )
        run_id = "test_run"
        conn.execute(
            "INSERT INTO runs(run_id, name, scenario_yaml, master_seed, status)"
            " VALUES (?,?,?,?,?)",
            (run_id, "test", "{}", 42, "running"),
        )
        dbm.upsert_listing(conn, run_id, "agent_0", listing)

        env = Environment(
            run_id=run_id, conn=conn, scenario=scenario,
            runs_root="/tmp", products=[product],
            hourly_dist={"electronics": np.ones(24) / 24.0},
            agents={"agent_0": agent_state},
        )
        env.lock = threading.RLock()
        return env, conn, product, listing

    def test_update_listing_ratings_normal(self, env_setup):
        from core.entities import Order
        env, conn, product, listing = env_setup
        env.t = 5
        order = Order(
            order_id="o1", agent_id="agent_0", product_id="p1",
            supplier_id="s1", order_t=0, promised_delivery_t=48,
            sale_price=15.0, purchase_price=10.0,
            current_status="settled_normal", settled_t=5,
        )
        env._update_listing_ratings([order])
        assert listing.normal_count == 1
        assert listing.bad_review_count == 0
        assert listing.rating_sum == pytest.approx(4.5)
        assert listing.rating_count == 1

    def test_update_listing_ratings_bad_review(self, env_setup):
        from core.entities import Order
        env, conn, product, listing = env_setup
        env.t = 5
        order = Order(
            order_id="o1", agent_id="agent_0", product_id="p1",
            supplier_id="s1", order_t=0, promised_delivery_t=48,
            sale_price=15.0, purchase_price=10.0,
            current_status="settled_bad_review", settled_t=5,
        )
        env._update_listing_ratings([order])
        assert listing.normal_count == 0
        assert listing.bad_review_count == 1
        assert listing.rating_sum == pytest.approx(1.0)
        assert listing.rating_count == 1

    def test_final_outcomes_update_rating_once(self, env_setup):
        from core.entities import Order
        env, conn, product, listing = env_setup
        env.t = 5
        orders = [
            Order(order_id="o1", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="settled_normal", settled_t=5),
            Order(order_id="o2", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="settled_normal", settled_t=5, late_t=3),
            Order(order_id="o3", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="settled_refund", settled_t=5, late_t=3),
            Order(order_id="o4", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="settled_only_refund", settled_t=5),
            Order(order_id="o5", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="settled_bad_review", settled_t=5),
            Order(order_id="o6", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="stockout", settled_t=5),
            Order(order_id="o7", agent_id="agent_0", product_id="p1",
                  supplier_id="s1", order_t=0, promised_delivery_t=48,
                  sale_price=15.0, purchase_price=10.0,
                  current_status="cancelled", settled_t=5),
        ]
        env._update_listing_ratings(orders)
        assert listing.normal_count == 2
        assert listing.bad_review_count == 1
        assert listing.rating_sum == pytest.approx(13.0)
        assert listing.rating_count == 6

    def test_db_persistence_of_counts(self, env_setup):
        from core.entities import Order
        from storage import db as dbm
        env, conn, product, listing = env_setup
        env.t = 5
        order = Order(
            order_id="o1", agent_id="agent_0", product_id="p1",
            supplier_id="s1", order_t=0, promised_delivery_t=48,
            sale_price=15.0, purchase_price=10.0,
            current_status="settled_normal", settled_t=5,
        )
        env._update_listing_ratings([order])
        db_listing = dbm.get_listing(conn, "test_run", "agent_0", "p1")
        assert db_listing.normal_count == 1
        assert db_listing.bad_review_count == 0
        assert db_listing.rating_sum == pytest.approx(4.5)
        assert db_listing.rating_count == 1

    def test_relist_restores_counts_from_orders(self, env_setup):
        from core.entities import EventLog, Order
        from storage import db as dbm
        import tools.tools as tt
        env, conn, product, listing = env_setup
        conn.execute(
            "INSERT INTO orders(run_id, order_id, agent_id, product_id, supplier_id,"
            " order_t, promised_delivery_t, sale_price, purchase_price, current_status,"
            " settled_t) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("test_run", "o1", "agent_0", "p1", "s1", 0, 48, 15.0, 10.0, "settled_normal", 3),
        )
        conn.execute(
            "INSERT INTO orders(run_id, order_id, agent_id, product_id, supplier_id,"
            " order_t, promised_delivery_t, sale_price, purchase_price, current_status,"
            " settled_t) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("test_run", "o2", "agent_0", "p1", "s1", 1, 49, 15.0, 10.0, "settled_bad_review", 4),
        )
        conn.execute(
            "INSERT INTO orders(run_id, order_id, agent_id, product_id, supplier_id,"
            " order_t, promised_delivery_t, sale_price, purchase_price, current_status,"
            " settled_t, late_t) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("test_run", "o3", "agent_0", "p1", "s1", 2, 50, 15.0, 10.0, "settled_refund", 5, 4),
        )
        conn.execute(
            "INSERT INTO orders(run_id, order_id, agent_id, product_id, supplier_id,"
            " order_t, promised_delivery_t, sale_price, purchase_price, current_status,"
            " settled_t) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("test_run", "o4", "agent_0", "p1", "s1", 2, 50, 15.0, 10.0, "stockout", 5),
        )
        dbm.write_events(conn, "test_run", [
            EventLog(t=0, event_type="agent_list_product", entity_id="p1",
                     agent_id="agent_0", payload={"sale_price": 15.0}),
        ])
        dbm.delete_listing(conn, "test_run", "agent_0", "p1")
        del env.agents["agent_0"].listings["p1"]
        env.t = 10
        result = tt._list_product_locked(env, "agent_0", "p1", 15.0)
        assert result["ok"]
        new_listing = env.agents["agent_0"].listings["p1"]
        assert new_listing.first_listed_at == 10
        assert new_listing.cum_sales == 3
        assert new_listing.cum_revenue == 45.0
        assert new_listing.normal_count == 1
        assert new_listing.bad_review_count == 1
        assert new_listing.rating_sum == pytest.approx(8.5)
        assert new_listing.rating_count == 4

    def test_write_metrics_includes_average_listing_sale_price_and_rating(self, env_setup):
        from dataclasses import replace
        from core.entities import StoreListing
        from storage import db as dbm

        env, conn, product, listing = env_setup
        env.t = 6
        listing.sale_price = 15.0
        listing.rating_sum = 5.0
        listing.rating_count = 1
        weak_listing = StoreListing(
            product_id="p2", agent_id="agent_0", sale_price=45.0,
            listed_at=0, first_listed_at=0,
            rating_sum=3.0, rating_count=1,
        )
        env.products["p2"] = replace(product, product_id="p2", price=25.0, ref_price=45.0)
        env.agents["agent_0"].listings["p2"] = weak_listing
        dbm.upsert_listing(conn, env.run_id, "agent_0", weak_listing)

        env._write_metrics([], [], [])

        series = dbm.load_metrics_bulk(conn, env.run_id, "agent_0", [
            "avg_listing_sale_price",
            "avg_listing_sale_price_count",
            "avg_listing_margin",
            "avg_listing_margin_count",
            "avg_listing_margin_ratio",
            "avg_listing_margin_ratio_count",
            "avg_listing_rating",
            "avg_listing_rating_count",
        ])
        assert series["avg_listing_sale_price"] == [(6, 30.0)]
        assert series["avg_listing_sale_price_count"] == [(6, 2.0)]
        assert series["avg_listing_margin"] == [(6, 12.5)]
        assert series["avg_listing_margin_count"] == [(6, 2.0)]
        assert series["avg_listing_margin_ratio"] == [
            (6, pytest.approx(((15.0 - 10.0) / 15.0 + (45.0 - 25.0) / 45.0) / 2))
        ]
        assert series["avg_listing_margin_ratio_count"] == [(6, 2.0)]
        assert series["avg_listing_rating"] == [(6, 4.0)]
        assert series["avg_listing_rating_count"] == [(6, 2.0)]
