from core.entities import StoreListing
from storage import db as dbm


def test_listing_reads_preserve_zero_first_listed_at(tmp_path):
    conn = dbm.open_db(str(tmp_path / "state.db"))
    try:
        listing = StoreListing(
            product_id="product-1",
            agent_id="agent-1",
            sale_price=10.0,
            listed_at=100,
            first_listed_at=0,
        )
        dbm.upsert_listing(conn, "run-1", "agent-1", listing)

        assert dbm.get_listing(
            conn, "run-1", "agent-1", "product-1"
        ).first_listed_at == 0
        assert dbm.list_listings(
            conn, "run-1", "agent-1"
        )[0].first_listed_at == 0
    finally:
        conn.close()


def test_listing_rating_aggregates_persist_and_reload(tmp_path):
    conn = dbm.open_db(str(tmp_path / "state.db"))
    try:
        listing = StoreListing(
            product_id="product-1",
            agent_id="agent-1",
            sale_price=10.0,
            listed_at=100,
            first_listed_at=100,
            rating_sum=8.5,
            rating_count=2,
        )
        dbm.upsert_listing(conn, "run-1", "agent-1", listing)

        reloaded = dbm.get_listing(conn, "run-1", "agent-1", "product-1")
        assert reloaded.rating_sum == 8.5
        assert reloaded.rating_count == 2
    finally:
        conn.close()


def test_open_db_migrates_listing_rating_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = dbm.open_db(str(db_path))
    conn.execute("ALTER TABLE store_listings DROP COLUMN rating_sum")
    conn.execute("ALTER TABLE store_listings DROP COLUMN rating_count")
    conn.close()

    migrated = dbm.open_db(str(db_path))
    try:
        columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(store_listings)").fetchall()
        }
        assert "rating_sum" in columns
        assert "rating_count" in columns
    finally:
        migrated.close()


def test_open_db_migrates_pending_hook_boundary_column(tmp_path):
    db_path = tmp_path / "legacy-run.db"
    conn = dbm.open_db(str(db_path))
    conn.execute("ALTER TABLE runs DROP COLUMN pending_hook_t")
    conn.execute("ALTER TABLE runs DROP COLUMN pending_hook_closed")
    conn.close()

    migrated = dbm.open_db(str(db_path))
    try:
        columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "pending_hook_t" in columns
        assert "pending_hook_closed" in columns
    finally:
        migrated.close()


def test_readonly_listing_read_handles_legacy_rating_columns(tmp_path):
    db_path = tmp_path / "legacy-readonly.db"
    conn = dbm.open_db(str(db_path))
    try:
        listing = StoreListing(
            product_id="product-1",
            agent_id="agent-1",
            sale_price=10.0,
            listed_at=100,
            first_listed_at=100,
            normal_count=2,
            bad_review_count=1,
        )
        dbm.upsert_listing(conn, "run-1", "agent-1", listing)
        conn.execute("ALTER TABLE store_listings DROP COLUMN rating_sum")
        conn.execute("ALTER TABLE store_listings DROP COLUMN rating_count")
    finally:
        conn.close()

    readonly = dbm.open_db_readonly(str(db_path))
    try:
        reloaded = dbm.get_listing(readonly, "run-1", "agent-1", "product-1")
        assert reloaded.rating_sum == 11.0
        assert reloaded.rating_count == 3
        assert dbm.list_listings(readonly, "run-1", "agent-1")[0].rating_sum == 11.0
    finally:
        readonly.close()


def test_metric_sampling_bounds_each_series_and_preserves_endpoints(tmp_path):
    conn = dbm.open_db(str(tmp_path / "metrics.db"))
    try:
        for t in range(100):
            dbm.write_metrics(
                conn,
                "run-1",
                "agent-1",
                t,
                {"x": float(t), "y": float(t * 2)},
            )

        sampled = dbm.load_metrics_bulk_sampled(
            conn,
            "run-1",
            "agent-1",
            ["x", "y"],
            max_points_per_key=10,
        )

        assert len(sampled["x"]) == 10
        assert sampled["x"][0] == (0, 0.0)
        assert sampled["x"][-1] == (99, 99.0)
        assert sampled["y"][0] == (0, 0.0)
        assert sampled["y"][-1] == (99, 198.0)
    finally:
        conn.close()


def test_daily_metric_lasts_preserve_exact_day_window_values(tmp_path):
    conn = dbm.open_db(str(tmp_path / "daily-metrics.db"))
    try:
        conn.executemany(
            "INSERT INTO metrics(run_id, agent_id, t, key, value)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                ("run-1", "agent-1", t, "cum_gmv", float(t))
                for t in range(365 * 24)
            ],
        )
        conn.executemany(
            "INSERT INTO metrics(run_id, agent_id, t, key, value)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                ("run-1", "_global", t, "orders_generated", 1.0)
                for t in range(365 * 24)
            ],
        )

        daily = dbm.load_metrics_bulk_daily_lasts(
            conn,
            "run-1",
            "agent-1",
            ["cum_gmv"],
            step_hours=1,
        )["cum_gmv"]
        cumulative = dbm.load_metric_cumulative_daily_lasts(
            conn,
            "run-1",
            "_global",
            "orders_generated",
            step_hours=1,
        )

        assert daily[:3] == [
            (0, 0.0),
            (23, 23.0),
            (47, 47.0),
        ]
        assert daily[2][1] - daily[1][1] == 24.0
        assert cumulative[:3] == [
            (0, 1.0),
            (23, 24.0),
            (47, 48.0),
        ]
        assert cumulative[2][1] - cumulative[1][1] == 24.0
    finally:
        conn.close()
