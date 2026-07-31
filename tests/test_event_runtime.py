import gzip
import json
import os
import sqlite3
import tempfile
from dataclasses import replace

from core.entities import EventLog, Product


def _mkproduct(**overrides) -> Product:
    base = dict(
        product_id="P00000",
        name="x",
        quantity=10,
        quantity_updated_t=0,
        price=100.0,
        ref_price=100.0,
        base_price=100.0,
        supplier_id="s",
        supplier_name="S",
        ship_hours=1,
        logistics_hours=1,
        category="office",
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
        elasticity=1.0,
        market_curve=[1.0] * 365,
    )
    base.update(overrides)
    return Product(**base)


def test_lazy_inventory_caps_and_materializes_before_consumption():
    from core.inventory import consume_quantity, effective_quantity

    p = _mkproduct(quantity=10, quantity_updated_t=2, hourly_increment=7, max_quantity=30)

    assert effective_quantity(p, t=4) == 24
    assert effective_quantity(p, t=5) == 30

    ok = consume_quantity(p, t=5, n=3)

    assert ok is True
    assert p.quantity == 27
    assert p.quantity_updated_t == 5


def test_scheduler_is_deterministic_and_reschedules_after_price_recover():
    from core.supplier_scheduler import apply_due_events, initial_supplier_events

    sup_cfg = {"recover_steps": [3, 3], "price_adjust_factor": [1.2, 1.2]}
    p1 = _mkproduct(product_id="P1", price_change_rate=0.25)
    p2 = _mkproduct(product_id="P1", price_change_rate=0.25)

    ev1 = initial_supplier_events([p1], master_seed=42, start_t=0, horizon=200)
    ev2 = initial_supplier_events([p2], master_seed=42, start_t=0, horizon=200)

    assert [e.to_row() for e in ev1] == [e.to_row() for e in ev2]
    price_event = next(e for e in ev1 if e.event_type == "price_change")

    logs, dirty, followups, cancel_ids = apply_due_events(
        {"P1": p1}, [price_event], price_event.due_t, master_seed=42,
        sup_cfg=sup_cfg, horizon=200,
    )

    assert dirty == {"P1"}
    assert cancel_ids == []
    assert p1.price == 120.0
    assert any(e.event_type == "price_change" for e in logs)
    recover = next(e for e in followups if e.event_type == "price_recover")
    assert recover.due_t == price_event.due_t + 3

    logs, dirty, followups, cancel_ids = apply_due_events(
        {"P1": p1}, [recover], recover.due_t, master_seed=42,
        sup_cfg=sup_cfg, horizon=200,
    )

    assert p1.price == 100.0
    assert dirty == {"P1"}
    assert any(e.event_type == "price_recover" for e in logs)
    assert any(e.event_type == "price_change" for e in followups)


def test_delta_snapshot_and_checkpoint_store_only_mutable_product_fields():
    from storage import snapshot as snap

    tmp = tempfile.mkdtemp()
    p = _mkproduct(product_id="P1", quantity=11, price=90.0)
    dirty = {"P1": p}

    delta_path = snap.write_env_delta_snapshot(
        tmp,
        "run-1",
        5,
        dirty_products=dirty.values(),
        agents=[],
        mutated_orders=[],
        events_this_step=[EventLog(t=5, event_type="price_change", entity_id="P1", payload={})],
        survival_state={"is_alive": True},
    )

    delta = json.load(open(delta_path, encoding="utf-8"))
    assert delta["kind"] == "delta"
    assert list(delta["products_delta"]) == ["P1"]
    assert "market_curve" not in delta["products_delta"]["P1"]
    assert "name" not in delta["products_delta"]["P1"]

    checkpoint_path = snap.write_env_checkpoint(tmp, "run-1", 7, [p])
    with gzip.open(checkpoint_path, "rt", encoding="utf-8") as f:
        checkpoint = json.load(f)
    assert checkpoint["kind"] == "checkpoint"
    assert list(checkpoint["products"]) == ["P1"]
    assert "market_curve" not in checkpoint["products"]["P1"]
    assert "name" not in checkpoint["products"]["P1"]


def test_replay_frame_reconstructs_product_state_from_checkpoint_and_deltas():
    from storage import snapshot as snap
    from storage.replay import ReplayFrameCache

    tmp = tempfile.mkdtemp()
    run_id = "run-1"
    p1 = _mkproduct(product_id="P1", quantity=11, price=90.0)
    p2 = _mkproduct(product_id="P2", quantity=22, price=120.0)
    snap.write_env_checkpoint(tmp, run_id, 0, [p1, p2], current_t=0)

    p1_t5 = replace(
        p1,
        quantity=7,
        quantity_updated_t=5,
        price=80.0,
        is_listed_by_supplier=False,
        delist_recover_t=12,
    )
    snap.write_env_delta_snapshot(
        tmp,
        run_id,
        5,
        dirty_products=[p1_t5],
        agents=[],
        mutated_orders=[],
        events_this_step=[],
        survival_state={},
        current_t=5,
    )

    p2_t8 = replace(
        p2,
        quantity=44,
        quantity_updated_t=8,
        price=150.0,
        timeout_active=True,
        timeout_recover_t=13,
    )
    snap.write_env_delta_snapshot(
        tmp,
        run_id,
        8,
        dirty_products=[p2_t8],
        agents=[],
        mutated_orders=[],
        events_this_step=[],
        survival_state={},
        current_t=8,
    )

    cache = ReplayFrameCache(tmp, max_frames=2)
    frame5 = cache.frame(run_id, 5)
    frame8 = cache.frame(run_id, 8)

    assert frame5.t == 5
    assert frame5.products["P1"]["price"] == 80.0
    assert frame5.products["P1"]["quantity"] == 7
    assert frame5.products["P1"]["is_listed_by_supplier"] is False
    assert frame5.products["P2"]["price"] == 120.0

    assert frame8.t == 8
    assert frame8.products["P1"]["price"] == 80.0
    assert frame8.products["P2"]["price"] == 150.0
    assert frame8.products["P2"]["timeout_active"] is True


def test_replay_frame_cache_holds_lock_while_serving_cached_frames(monkeypatch):
    from storage.replay import ReplayFrame, ReplayFrameCache

    class OwnedLock:
        def __init__(self):
            self.owned = False

        def __enter__(self):
            self.owned = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self.owned = False
            return False

    cache = ReplayFrameCache(tempfile.mkdtemp())
    cache._cache[("run-1", 0)] = ReplayFrame(t=0, products={}, agents=[], survival_state={})
    assert hasattr(cache, "_lock")
    guard = OwnedLock()
    cache._lock = guard

    def clone_requires_lock(frame):
        assert guard.owned is True
        return ReplayFrame(
            t=frame.t,
            products={pid: dict(values) for pid, values in frame.products.items()},
            agents=[dict(agent) for agent in frame.agents],
            survival_state=dict(frame.survival_state),
        )

    monkeypatch.setattr(cache, "_clone", clone_requires_lock)

    frame = cache.frame("run-1", 0)

    assert frame.t == 0


def test_runtime_db_has_supplier_events_and_effective_search():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    p = _mkproduct(product_id="P1", quantity=0, quantity_updated_t=0, hourly_increment=5)
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [p])

    dbm.insert_supplier_events(conn, "run-1", [
        {"due_t": 10, "product_id": "P1", "event_type": "price_change", "seq": 0, "payload": {}}
    ])
    due = dbm.load_supplier_events_due(conn, "run-1", 10)
    assert due[0]["product_id"] == "P1"

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="",
        filters={"quantity_min": 10},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=2,
    )
    assert [r["product_id"] for r in rows] == ["P1"]


def test_search_products_relevance_uses_bm25_for_short_chinese_terms():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(
            product_id="A001",
            name="耳机收纳盒桌面小物批发",
            category="office",
        ),
        _mkproduct(
            product_id="Z999",
            name="蓝牙耳机无线耳机运动耳机批发",
            category="office",
        ),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="耳机",
        filters={},
        sort_by="relevance",
        limit=2,
        offset=0,
        current_t=0,
    )

    assert rows[0]["product_id"] == "Z999"


def test_search_products_sql_does_not_fallback_for_single_chinese_character_queries():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(
            product_id="A001",
            name="蓝牙音箱",
            supplier_name="耳机配件厂",
        ),
        _mkproduct(
            product_id="Z999",
            name="蓝牙耳机",
            supplier_name="数码产品厂",
        ),
        _mkproduct(product_id="B002", name="桌面收纳盒"),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="耳",
        filters={},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=0,
    )

    assert rows == []


def test_search_products_relevance_keeps_ascii_substring_matches_with_fts():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(product_id="A001", name="Art paper notebook", category="office"),
        _mkproduct(product_id="Z999", name="Smart tape dispenser", category="office"),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="art",
        filters={},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=0,
    )

    assert {r["product_id"] for r in rows} == {"A001", "Z999"}


def test_search_products_relevance_does_not_expand_long_ascii_queries_to_trigrams():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(product_id="A001", name="Art paper notebook", category="office"),
        _mkproduct(product_id="Z999", name="Smart tape dispenser", category="office"),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="smart",
        filters={},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=0,
    )

    assert [r["product_id"] for r in rows] == ["Z999"]


def test_search_products_relevance_keeps_long_ascii_exact_in_mixed_cjk_query():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(product_id="A001", name="Art paper notebook", category="office"),
        _mkproduct(product_id="Z999", name="Smart tape dispenser 蓝牙耳机", category="office"),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="smart耳机",
        filters={},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=0,
    )

    assert [r["product_id"] for r in rows] == ["Z999"]


def test_search_products_uses_same_fts_matches_for_every_sort_mode():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(
            product_id="A001",
            name="蓝牙耳机",
            category="office",
            price=20.0,
            historical_avg_rating=4.0,
            shop_rating=4.9,
        ),
        _mkproduct(
            product_id="B002",
            name="Camera",
            category="electronics",
            price=10.0,
            historical_avg_rating=5.0,
            shop_rating=4.0,
        ),
        _mkproduct(
            product_id="C003",
            name="Unrelated product",
            category="toys",
        ),
    ])

    matched_ids = {}
    for sort_by in (
        "relevance",
        "price_asc",
        "price_desc",
        "rating",
        "supplier_rating",
        "logistics_speed",
    ):
        rows = dbm.search_products_sql(
            conn,
            "run-1",
            query="耳机 electronics",
            filters={},
            sort_by=sort_by,
            limit=10,
            offset=0,
            current_t=0,
        )
        matched_ids[sort_by] = {row["product_id"] for row in rows}

    assert all(ids == {"A001", "B002"} for ids in matched_ids.values())


def test_search_products_does_not_fallback_to_like_when_fts_has_no_match():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    conn = dbm.open_db(os.path.join(tmp, "test.db"))
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(product_id="A001", name="Smart tape dispenser"),
    ])

    rows = dbm.search_products_sql(
        conn,
        "run-1",
        query="smar",
        filters={},
        sort_by="relevance",
        limit=10,
        offset=0,
        current_t=0,
    )

    assert rows == []


def test_open_db_rebuilds_missing_catalog_fts_rows():
    from storage import db as dbm

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "test.db")
    conn = dbm.open_db(path)
    dbm.insert_run(conn, "run-1", "run", "{}", 42, 24, 1, "now")
    dbm.insert_products(conn, "run-1", [
        _mkproduct(product_id="A001", name="蓝牙耳机"),
    ])
    conn.execute("DELETE FROM catalog_search_fts")
    conn.close()

    migrated = dbm.open_db(path)
    try:
        rows = dbm.search_products_sql(
            migrated,
            "run-1",
            query="耳机",
            filters={},
            sort_by="relevance",
            limit=10,
            offset=0,
            current_t=0,
        )
        assert [row["product_id"] for row in rows] == ["A001"]
    finally:
        migrated.close()


def test_private_real_preflight_rejects_sha_mismatch(tmp_path):
    from data.build_private_real_db import build_private_real_db
    from data.generation_profiles import load_default_generation_params
    from data.private_real_preflight import preflight_dataset
    from tests.private_real_fixture import write_fixture_csv

    bench_path = write_fixture_csv(str(tmp_path))
    db_path = tmp_path / "private_real.sqlite"
    params = load_default_generation_params()
    params["supplier_item_count"] = dict(params.get("supplier_item_count", {}))
    params["supplier_item_count"]["min"] = 1
    build_private_real_db(
        bench_csv=bench_path,
        output_db=str(db_path),
        target_rows=10,
        seed=7,
        params=params,
    )

    result = preflight_dataset(str(db_path), expected_sha="definitely-wrong")

    assert result["ok"] is False
    assert "dataset_sha256" in result["errors"][0]


def _build_private_real_fixture(tmp_path, rows=10):
    from data.build_private_real_db import build_private_real_db
    from data.generation_profiles import load_default_generation_params
    from tests.private_real_fixture import write_fixture_csv

    tmp_path.mkdir(parents=True, exist_ok=True)
    bench_path = write_fixture_csv(str(tmp_path))
    db_path = tmp_path / f"private_real_{rows}.sqlite"
    params = load_default_generation_params()
    params["supplier_item_count"] = dict(params.get("supplier_item_count", {}))
    params["supplier_item_count"]["min"] = 1
    build_private_real_db(
        bench_csv=bench_path,
        output_db=str(db_path),
        target_rows=rows,
        seed=7,
        params=params,
    )
    return db_path


def test_private_real_preflight_rejects_unreadable_and_missing_tables(tmp_path):
    from data.private_real_preflight import preflight_dataset

    bad_path = tmp_path / "not.sqlite"
    bad_path.write_text("not sqlite", encoding="utf-8")
    bad_result = preflight_dataset(str(bad_path))
    assert bad_result["ok"] is False
    assert any("invalid DB" in err or "file is not a database" in err for err in bad_result["errors"])

    missing_path = tmp_path / "missing.sqlite"
    conn = sqlite3.connect(missing_path)
    conn.execute("CREATE TABLE dataset_meta(key TEXT, value TEXT)")
    conn.commit()
    conn.close()
    missing_result = preflight_dataset(str(missing_path))
    assert missing_result["ok"] is False
    assert any("missing table: products" in err for err in missing_result["errors"])


def test_private_real_preflight_rejects_market_curve_and_hourly_dist_drift(tmp_path):
    from data.private_real_preflight import preflight_dataset

    db_path = _build_private_real_fixture(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE products SET market_curve=? WHERE product_id=(SELECT product_id FROM products LIMIT 1)",
        (json.dumps([1.0, 2.0]),),
    )
    conn.commit()
    conn.close()
    curve_result = preflight_dataset(str(db_path))
    assert curve_result["ok"] is False
    assert any("market_curve length" in err for err in curve_result["errors"])

    db_path = _build_private_real_fixture(tmp_path / "hourly")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE hourly_dist SET w=0.0"
        " WHERE category=(SELECT category FROM hourly_dist LIMIT 1) AND hour=0"
    )
    conn.commit()
    conn.close()
    hourly_result = preflight_dataset(str(db_path))
    assert hourly_result["ok"] is False
    assert any("hourly_dist category" in err for err in hourly_result["errors"])


def test_private_real_preflight_rejects_supplier_profile_drift(tmp_path):
    from data.private_real_preflight import preflight_dataset

    db_path = _build_private_real_fixture(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = [row["name"] for row in conn.execute("PRAGMA table_info(products)").fetchall()]
    select_exprs = []
    params = []
    for col in cols:
        if col == "product_id":
            select_exprs.append("?")
            params.append("DRIFT_PRODUCT")
        elif col == "shop_rating":
            select_exprs.append("shop_rating - 0.5")
        else:
            select_exprs.append(col)
    conn.execute(
        f"INSERT INTO products({','.join(cols)})"
        f" SELECT {','.join(select_exprs)} FROM products LIMIT 1",
        params,
    )
    conn.commit()
    conn.close()

    result = preflight_dataset(str(db_path))

    assert result["ok"] is False
    assert any("supplier profile drift" in err for err in result["errors"])


def _tiny_scenario():
    from web.runner import load_default_scenario

    scen = load_default_scenario()
    scen["run"]["max_hook_seconds"] = 0.01
    scen["run"]["horizon_steps"] = 24
    scen["run"]["checkpoint_interval_steps"] = 4
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 12
    scen["data"]["num_suppliers"] = 3
    scen["data"]["num_categories"] = 2
    return scen


def test_create_run_initializes_supplier_event_queue():
    from storage import db as dbm
    from web.app import create_app

    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]

    assert dbm.count_supplier_events(app.registry.conn_for(run_id), run_id) > 0


def test_step_uses_event_scheduler_not_full_product_manager(monkeypatch):
    from web.app import create_app

    def explode(*args, **kwargs):
        raise AssertionError("old full product manager loop must not run")

    monkeypatch.setattr("core.simulator.pm.update_products", explode)
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
        c.post(f"/runs/{run_id}/step")

    snap_dir = os.path.join(tmp, "runs", run_id, "env_snapshot")
    snap0 = json.load(open(os.path.join(snap_dir, "t_00000.json"), encoding="utf-8"))
    assert snap0["kind"] == "delta"
    assert len(snap0["products_delta"]) < 12
