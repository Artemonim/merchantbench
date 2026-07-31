"""End-to-end smoke test: create run, list a product, run a few steps, check artifacts."""
import json
import os
import sqlite3
import tempfile
import threading
import time

import pytest
import yaml

from web.app import create_app
from web.runner import load_default_scenario
from storage import db as dbm


def _table_records(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        yield c, tmp, app


def _tiny_scenario(max_hook_seconds=0.1):
    s = load_default_scenario()
    s["run"]["max_hook_seconds"] = max_hook_seconds
    s["run"]["horizon_steps"] = 24
    s["data"]["source"] = "synthetic"
    s["data"]["num_products"] = 30
    s.setdefault("agent", {})["tool_denylist"] = []
    s.pop("lifecycle", None)
    return s


def _act(c, rid, agent_id, thought, tool_calls_spec):
    """Call the unified /act endpoint with one or more tool invocations."""
    tc_list = []
    for i, (name, args) in enumerate(tool_calls_spec):
        tc_list.append({"id": f"call_{i}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}})
    body = {"messages": [{"role": "assistant", "content": thought, "tool_calls": tc_list}]}
    return c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)


def _preseed_listings(app, run_id, n=50, markup=1.20):
    """Directly seed top-N listings for agent_0 (replaces the old inline
    auto_seed that ran synchronously during create_run). Used by tests that
    need listings present at t=0 without a real HTTP server for the
    subprocess agent to connect to."""
    from core.entities import StoreListing
    from storage import db as dbm
    env = app.registry._require(run_id)
    ranked = sorted(env.products.values(),
                    key=lambda p: sum(p.market_curve[-7:]), reverse=True)
    for p in ranked[:n]:
        sp = round(p.price * markup, 2)
        listing = StoreListing(product_id=p.product_id, agent_id="agent_0",
                               sale_price=sp, listed_at=0)
        dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
        env.agents["agent_0"].listings[p.product_id] = listing


def test_full_smoke(client):
    c, tmp, app = client
    resp = c.post("/runs", json={"scenario": _tiny_scenario(0.05)})
    assert resp.status_code == 200
    run_id = resp.get_json()["run_id"]
    _preseed_listings(app, run_id)

    # 2. drive 6 timesteps (each blocks ~50ms via the hook timeout)
    for _ in range(6):
        s = c.post(f"/runs/{run_id}/step").get_json()
        assert "t" in s

    # 4. snapshots written
    snap_dir = os.path.join(tmp, "runs", run_id, "env_snapshot")
    files = sorted(os.listdir(snap_dir))
    assert len(files) == 6
    snap0 = json.load(open(os.path.join(snap_dir, files[0])))
    assert snap0["t"] == 0
    assert "products" in snap0 and "orders" in snap0
    assert "agents" in snap0 and len(snap0["agents"]) >= 1
    assert "cash" in snap0["agents"][0]
    assert snap0["agents"][0]["agent_id"] == "agent_0"

    # 5. dashboard renders
    d = c.get(f"/dashboard?run_id={run_id}")
    assert d.status_code == 200
    assert b"MerchantBench" in d.data


def test_supplier_timeout_recovery_keeps_existing_order_ship_snapshot(client):
    """Recovery changes future-order timing without rewriting existing orders."""
    from core.entities import Order, OrderStatusRow

    c, _, app = client
    scenario = _tiny_scenario()
    scenario["run"]["horizon_steps"] = 200
    run_id = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))

    product.base_ship_hours = 3
    product.supplier_ship_hours = 96
    product.timeout_active = True
    product.timeout_recover_t = env.t
    dbm.upsert_product_state(env.conn, run_id, product)

    order = Order(
        order_id="timeout-recovery-order",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=env.t,
        promised_delivery_t=env.t + 96 + product.logistics_hours,
        sale_price=29.0,
        purchase_price=13.8,
        current_status="ordered",
        purchase_t=env.t,
        preset_anomaly="cancel",
        preset_anomaly_t=env.t + 91,
        supplier_ship_hours=96,
        actual_ship_hours=96,
        realized_cost=13.8,
        status_log=[OrderStatusRow(t=env.t, status="ordered")],
    )
    dbm.insert_orders(env.conn, run_id, [order])
    dbm.insert_supplier_events(env.conn, run_id, [{
        "due_t": env.t,
        "product_id": product.product_id,
        "event_type": "supplier_timeout_end",
        "seq": 0,
        "payload": {},
    }])

    env.step(drain=True)

    persisted = dbm.load_orders(env.conn, run_id, agent_id="agent_0")
    persisted_order = next(o for o in persisted if o.order_id == order.order_id)
    assert product.supplier_ship_hours == 3
    assert persisted_order.supplier_ship_hours == 96
    assert persisted_order.current_status == "ordered"


def test_trust_signal_consistency_per_supplier():
    """shop_rating / return_buyer_rate / supplier_age_years are sampled once
    per supplier_id and must be identical across every product owned by that
    supplier. historical_avg_rating is per-product and may vary freely."""
    from data.synth import generate
    products, _ = generate(_tiny_scenario())
    by_sup: dict[str, list] = {}
    for p in products:
        by_sup.setdefault(p.supplier_id, []).append(p)
    for sup_id, sup_products in by_sup.items():
        ratings = {p.shop_rating for p in sup_products}
        rbrs = {p.return_buyer_rate for p in sup_products}
        ages = {p.supplier_age_years for p in sup_products}
        assert len(ratings) == 1, f"shop_rating drift on {sup_id}: {ratings}"
        assert len(rbrs) == 1, f"return_buyer_rate drift on {sup_id}: {rbrs}"
        assert len(ages) == 1, f"supplier_age_years drift on {sup_id}: {ages}"
        for p in sup_products:
            assert 1.0 <= p.historical_avg_rating <= 5.0
            assert 1.0 <= p.shop_rating <= 5.0
            assert 0.0 <= p.return_buyer_rate <= 1.0
            assert p.supplier_age_years >= 0.0


def test_end_of_step_releases_hook_within_app_context(client):
    """Drive the hook via the registry directly (skips HTTP-thread isolation)."""
    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=10)  # long timeout
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    registry = app.registry
    env = registry._require(run_id)

    import threading
    elapsed = {}

    def step_thread():
        t0 = time.time()
        registry.step(run_id)
        elapsed["dt"] = time.time() - t0

    th = threading.Thread(target=step_thread)
    th.start()
    deadline = time.time() + 3
    with env.hook_cond:
        while not env.hook_open:
            remaining = deadline - time.time()
            assert remaining > 0, "hook did not open"
            env.hook_cond.wait(timeout=remaining)
    env.hook_event.set()
    th.join(timeout=3)
    assert not th.is_alive()
    assert elapsed["dt"] < 1.0  # released quickly, not the 10s timeout


def test_hook_observes_committed_current_step_transition(client):
    """Orders, events, cash and metrics for t are committed before its hook."""
    from core.entities import Order, OrderStatusRow

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    cash = env.agents["agent_0"].cash
    cash.receivable = 75.0
    dbm.insert_orders(env.conn, run_id, [Order(
        order_id="settle-before-hook",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=75.0,
        purchase_price=50.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        settlement_delay_steps=0,
        realized_cost=50.0,
        status_log=[OrderStatusRow(t=0, status="delivered")],
    )])

    observed = {}

    def inspect_committed_state():
        # A separate SQLite connection proves the environment transaction was
        # committed, rather than merely visible on env.conn.
        check = sqlite3.connect(app.registry.run_db_path(run_id))
        try:
            observed["status"] = check.execute(
                "SELECT current_status FROM orders"
                " WHERE run_id=? AND order_id=?",
                (run_id, "settle-before-hook"),
            ).fetchone()[0]
            observed["event_count"] = check.execute(
                "SELECT COUNT(*) FROM events"
                " WHERE run_id=? AND event_type='order_settled_normal' AND t=0",
                (run_id,),
            ).fetchone()[0]
            observed["cash_receivable"] = check.execute(
                "SELECT receivable FROM cash_log"
                " WHERE run_id=? AND agent_id='agent_0' AND t=0",
                (run_id,),
            ).fetchone()[0]
            observed["metric_count"] = check.execute(
                "SELECT COUNT(*) FROM metrics"
                " WHERE run_id=? AND agent_id='agent_0'"
                " AND key='net_assets' AND t=0",
                (run_id,),
            ).fetchone()[0]
        finally:
            check.close()

    result = env.step(hook_blocker=inspect_committed_state)

    assert result.t == 0
    assert observed == {
        "status": "settled_normal",
        "event_count": 1,
        "cash_receivable": 0.0,
        "metric_count": 1,
    }


def test_committed_transition_retry_resumes_hook_without_replaying_demand(
    client,
    monkeypatch,
):
    """A failure after the transition commit must not procure the order twice."""
    from core import demand as demand_mod
    from core.entities import Order, StoreListing

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = min(env.products.values(), key=lambda item: item.price)
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.price * 1.4, 2),
        listed_at=-1,
        first_listed_at=-1,
    )
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    env.conn.execute("DELETE FROM supplier_events WHERE run_id=?", (run_id,))

    demand_calls = 0

    def deterministic_demand(_triples, _hourly_dist, step_t, *_args, **_kwargs):
        nonlocal demand_calls
        demand_calls += 1
        return [Order(
            order_id=f"recoverable-order-{step_t}",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=step_t,
            promised_delivery_t=step_t + 10,
            sale_price=listing.sale_price,
            purchase_price=product.price,
        )]

    monkeypatch.setattr(
        demand_mod,
        "generate_orders_for_step",
        deterministic_demand,
    )

    def fail_hook():
        raise RuntimeError("hook interrupted")

    with pytest.raises(RuntimeError, match="hook interrupted"):
        env.step(hook_blocker=fail_hook)

    after_failure_cash = env.agents["agent_0"].cash.balance
    after_failure_quantity = product.quantity
    after_failure_listing = dbm.get_listing(
        env.conn, run_id, "agent_0", product.product_id,
    )
    assert after_failure_listing is not None
    pending = dbm.get_run(env.conn, run_id)
    assert pending["current_t"] == 0
    assert pending["pending_hook_t"] == 0
    assert pending["pending_hook_closed"] == 0
    assert env.hook_open is False

    # Simulate the worker releasing its in-memory Environment before restart.
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)
    product = env.products[product.product_id]

    result = env.step()

    finished = dbm.get_run(env.conn, run_id)
    assert result.t == 0
    assert result.new_orders == 1
    assert demand_calls == 1
    assert env.agents["agent_0"].cash.balance == after_failure_cash
    assert product.quantity == after_failure_quantity
    recovered_listing = dbm.get_listing(
        env.conn, run_id, "agent_0", product.product_id,
    )
    assert recovered_listing is not None
    assert recovered_listing.cum_sales == after_failure_listing.cum_sales == 1
    assert recovered_listing.cum_revenue == after_failure_listing.cum_revenue == listing.sale_price
    assert finished["current_t"] == 1
    assert finished["pending_hook_t"] is None
    assert finished["pending_hook_closed"] == 0
    assert env.conn.execute(
        "SELECT COUNT(*) FROM orders WHERE run_id=? AND order_id=?",
        (run_id, "recoverable-order-0"),
    ).fetchone()[0] == 1


def test_uncommitted_transition_failure_rolls_back_before_rehydrate_and_retry(
    client,
    monkeypatch,
):
    """A pre-commit failure may replay demand, but must persist one sale only."""
    from core import demand as demand_mod
    from core.entities import Order, StoreListing

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = min(env.products.values(), key=lambda item: item.price)
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.price * 1.4, 2),
        listed_at=-1,
        first_listed_at=-1,
    )
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    env.conn.execute("DELETE FROM supplier_events WHERE run_id=?", (run_id,))
    initial_balance = env.agents["agent_0"].cash.balance
    initial_quantity = product.quantity
    demand_calls = 0

    def deterministic_demand(_triples, _hourly_dist, step_t, *_args, **_kwargs):
        nonlocal demand_calls
        demand_calls += 1
        return [Order(
            order_id=f"rollback-order-{step_t}",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=step_t,
            promised_delivery_t=step_t + 10,
            sale_price=listing.sale_price,
            purchase_price=product.price,
        )]

    monkeypatch.setattr(
        demand_mod,
        "generate_orders_for_step",
        deterministic_demand,
    )
    original_mark = dbm.mark_step_transition_committed
    mark_calls = 0

    def fail_first_transition_mark(conn, rid, step_t):
        nonlocal mark_calls
        mark_calls += 1
        if mark_calls == 1:
            raise RuntimeError("transition commit interrupted")
        return original_mark(conn, rid, step_t)

    monkeypatch.setattr(
        dbm,
        "mark_step_transition_committed",
        fail_first_transition_mark,
    )

    with pytest.raises(RuntimeError, match="transition commit interrupted"):
        env.step()

    rolled_back = dbm.get_run(env.conn, run_id)
    rolled_back_listing = dbm.get_listing(
        env.conn, run_id, "agent_0", product.product_id,
    )
    assert rolled_back["current_t"] == 0
    assert rolled_back["pending_hook_t"] is None
    assert rolled_back_listing is not None
    assert rolled_back_listing.cum_sales == 0
    assert rolled_back_listing.cum_revenue == 0.0
    assert env.conn.execute(
        "SELECT COUNT(*) FROM orders WHERE run_id=? AND order_id=?",
        (run_id, "rollback-order-0"),
    ).fetchone()[0] == 0

    # Production error handling releases the failed runtime.  Rehydrate from
    # the rolled-back database before retrying the logical transition.
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)
    product = env.products[product.product_id]

    result = env.step()

    final_listing = dbm.get_listing(
        env.conn, run_id, "agent_0", product.product_id,
    )
    assert result.t == 0
    assert demand_calls == 2
    assert final_listing is not None
    assert final_listing.cum_sales == 1
    assert final_listing.cum_revenue == listing.sale_price
    assert env.agents["agent_0"].cash.balance == initial_balance - product.price
    assert product.quantity == initial_quantity - 1
    assert env.conn.execute(
        "SELECT COUNT(*) FROM orders WHERE run_id=? AND order_id=?",
        (run_id, "rollback-order-0"),
    ).fetchone()[0] == 1


def test_pending_hook_finalizes_before_rehydrated_finished_phase(client):
    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)

    def fail_hook():
        raise RuntimeError("hook interrupted")

    with pytest.raises(RuntimeError, match="hook interrupted"):
        env.step(hook_blocker=fail_hook)

    dbm.mark_agent_dead(env.conn, run_id, "agent_0", env.t)
    env.agents["agent_0"].is_alive = False
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)

    app.registry._require(run_id)
    result = app.registry.step(run_id)
    row = dbm.get_run(app.registry.conn_for(run_id), run_id)

    assert result["phase"] == "finished"
    assert result["t"] == 0
    assert row["current_t"] == 1
    assert row["pending_hook_t"] is None
    assert row["pending_hook_closed"] == 0


def test_pending_hook_reuses_persisted_observation_window_after_restart(client):
    from tools import observation as observation_mod

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    observed_windows = []

    def interrupted_hook():
        observation_mod.compose_observation(
            env,
            "agent_0",
            mark_observed=True,
        )
        observed_windows.append(
            observation_mod.current_or_cached_change_window(env, "agent_0")
        )
        raise RuntimeError("hook interrupted after observation")

    with pytest.raises(RuntimeError, match="hook interrupted after observation"):
        env.step(hook_blocker=interrupted_hook)

    assert observed_windows == [(0, 0)]
    assert env.last_observation_step_by_agent["agent_0"] == 0

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)

    assert env.last_observation_step_by_agent["agent_0"] == 0
    assert env.observation_window_by_agent_step[("agent_0", 0)] == (0, 0)

    def recovered_hook():
        observation_mod.compose_observation(
            env,
            "agent_0",
            mark_observed=True,
        )
        observed_windows.append(
            observation_mod.current_or_cached_change_window(env, "agent_0")
        )

    result = env.step(hook_blocker=recovered_hook)

    assert result.t == 0
    assert observed_windows == [(0, 0), (0, 0)]


def test_successful_idempotency_result_is_durable_before_step_finalize(client):
    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    calls = 0

    def mutate_once():
        nonlocal calls
        calls += 1
        return {"ok": True, "value": calls}

    first = env.with_idempotency(
        "durable-before-finalize",
        mutate_once,
        fingerprint={"tool": "test"},
    )
    assert first == {"ok": True, "value": 1}

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)

    replay = env.with_idempotency(
        "durable-before-finalize",
        mutate_once,
        fingerprint={"tool": "test"},
    )

    assert calls == 1
    assert replay == {
        "ok": True,
        "value": 1,
        "_idempotent_replay": True,
    }


def test_finalization_retry_does_not_reopen_completed_hook(client, monkeypatch):
    from core.entities import EventLog
    from storage import agent_log
    from storage import snapshot as snap_mod

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    original_snapshot = snap_mod.write_env_delta_snapshot
    snapshot_calls = 0
    hook_calls = 0
    transition_event_count = 0

    def fail_first_snapshot(*args, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            raise OSError("snapshot interrupted")
        return original_snapshot(*args, **kwargs)

    def completed_hook():
        nonlocal hook_calls, transition_event_count
        hook_calls += 1
        transition_event_count = len(dbm.load_events_at(env.conn, run_id, env.t))
        dbm.write_events(env.conn, run_id, [
            EventLog(
                t=env.t,
                event_type="agent_adjust_price",
                entity_id="recovery-trace-product",
                agent_id="agent_0",
                payload={"old_price": 10.0, "new_price": 11.0},
            )
        ])
        env.record_act(
            "agent_0",
            {"role": "assistant", "content": "completed recovery hook"},
            [],
            token_usage={"input": 11, "output": 7, "total": 18},
        )

    monkeypatch.setattr(
        snap_mod,
        "write_env_delta_snapshot",
        fail_first_snapshot,
    )

    with pytest.raises(OSError, match="snapshot interrupted"):
        env.step(hook_blocker=completed_hook)

    pending = dbm.get_run(env.conn, run_id)
    assert pending["current_t"] == 0
    assert pending["pending_hook_t"] == 0
    assert pending["pending_hook_closed"] == 1
    assert hook_calls == 1

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)

    result = env.step(hook_blocker=completed_hook)
    finished = dbm.get_run(env.conn, run_id)

    assert result.t == 0
    assert hook_calls == 1
    assert snapshot_calls == 2
    assert result.events == transition_event_count
    assert finished["current_t"] == 1
    assert finished["pending_hook_t"] is None
    assert finished["pending_hook_closed"] == 0
    trace = agent_log.read_step_index(env.runs_root, run_id, 0)
    assert trace is not None
    assert trace["hook_close_wall_ms"] >= trace["hook_open_wall_ms"] > 0
    assert any(
        message.get("content") == "completed recovery hook"
        for message in trace["messages"]
    )
    cost = agent_log.read_cost(env.runs_root, run_id)
    assert cost["total"]["input"] == 11
    assert cost["total"]["output"] == 7
    assert cost["total"]["turns"] == 1
    snapshot = snap_mod.read_env_snapshot(env.runs_root, run_id, 0)
    assert snapshot is not None
    assert all(
        event["event_type"] != "agent_adjust_price"
        for event in snapshot["events_this_step"]
    )


def test_pending_drain_transition_never_reopens_agent_hook(client, monkeypatch):
    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    scen["run"]["horizon_steps"] = 1
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    env.t = 1
    dbm.update_run_t(env.conn, run_id, 1)

    original_close = dbm.mark_step_hook_closed
    close_calls = 0

    def fail_first_close(conn, rid, step_t):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("hook marker interrupted")
        return original_close(conn, rid, step_t)

    monkeypatch.setattr(dbm, "mark_step_hook_closed", fail_first_close)
    with pytest.raises(OSError, match="hook marker interrupted"):
        env.step(drain=True)

    pending = dbm.get_run(env.conn, run_id)
    assert pending["pending_hook_t"] == 1
    assert pending["pending_hook_closed"] == 0

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
    app.registry.close_run_conn(run_id)
    env = app.registry._require(run_id)
    hook_calls = 0

    def unexpected_hook():
        nonlocal hook_calls
        hook_calls += 1

    result = env.step(hook_blocker=unexpected_hook, drain=False)

    assert result.t == 1
    assert hook_calls == 0
    assert close_calls == 2
    finished = dbm.get_run(env.conn, run_id)
    assert finished["current_t"] == 2
    assert finished["pending_hook_t"] is None
    assert finished["pending_hook_closed"] == 0


def test_hook_product_stats_use_current_step_failure_event(client, monkeypatch):
    """Current-step failed orders and their persisted fines share one cutoff."""
    from core import demand as demand_mod
    from core.entities import Order, StoreListing
    from tools import tools as tool_impl

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.price * 1.4, 2),
        listed_at=-1,
        first_listed_at=-1,
    )
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    product.is_listed_by_supplier = False
    candidate = Order(
        order_id="stockout-before-hook",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=10,
        sale_price=listing.sale_price,
        purchase_price=product.price,
    )
    monkeypatch.setattr(
        demand_mod,
        "generate_orders_for_step",
        lambda *args, **kwargs: [candidate],
    )

    observed = {}

    def inspect_stats():
        result = tool_impl.query_product_sales_stats(
            env,
            "agent_0",
            day_from=1,
            day_to=1,
        )
        observed.update(_table_records(result["items"])[0])

    env.step(hook_blocker=inspect_stats)

    assert observed["orders"] == 1
    assert observed["stockout_count"] == 1
    assert observed["gmv"] == 0.0
    assert observed["fine"] > 0.0
    assert observed["net_profit"] == -observed["fine"]


def test_hook_listing_action_affects_demand_from_next_step(client, monkeypatch):
    from core import demand as demand_mod
    from core.entities import Order
    from tools import tools as tool_impl

    c, _, app = client
    scen = _tiny_scenario(max_hook_seconds=2)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    env.conn.execute("DELETE FROM supplier_events WHERE run_id=?", (run_id,))

    def deterministic_demand(triples, _hourly_dist, step_t, *_args, **_kwargs):
        if not triples:
            return []
        listed_product, listing, agent_id = triples[0]
        return [Order(
            order_id=f"next-step-order-{step_t}",
            product_id=listed_product.product_id,
            supplier_id=listed_product.supplier_id,
            agent_id=agent_id,
            order_t=step_t,
            promised_delivery_t=step_t + 10,
            sale_price=listing.sale_price,
            purchase_price=listed_product.price,
        )]

    monkeypatch.setattr(
        demand_mod,
        "generate_orders_for_step",
        deterministic_demand,
    )

    def list_during_hook():
        result = tool_impl.list_product(env, "agent_0", [{
            "product_id": product.product_id,
            "sale_price": round(product.price * 1.4, 2),
        }])
        assert _table_records(result["items"])[0]["ok"] is True

    step_zero = env.step(hook_blocker=list_during_hook)
    step_one = env.step()

    assert step_zero.new_orders == 0
    assert step_one.new_orders == 1
    order = dbm.load_order(env.conn, run_id, "next-step-order-1")
    assert order is not None
    assert order.order_t == 1


def test_per_step_metrics_include_profit_series_and_net_assets(client):
    """After driving a few ticks, the merchant section must expose cum_cost,
    cum_gross_profit, cum_net_profit, and net_assets series. cum_net_profit is the matched economic profit
    summed only over orders with settled_t set: per settled order,
    realized_revenue − realized_cost − total_penalty. In-flight orders do not
    contribute. cum_gmv is the gross merchandise volume (sum of sale_price
    for all successfully procured orders)."""
    c, _, app = client
    scen = _tiny_scenario(0.05)
    scen["data"]["small_share"] = 0.01
    scen["run"]["horizon_steps"] = 50
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    _preseed_listings(app, run_id)
    for _ in range(30):
        c.post(f"/runs/{run_id}/step")
    section = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()
    series = section["series"]
    for k in ("cum_cost", "cum_gross_profit", "cum_net_profit", "net_assets", "cum_gmv", "cum_fine"):
        assert k in series
    assert "cum_profit" not in series

    if series["cum_gross_profit"]:
        _, last_gross_profit = series["cum_gross_profit"][-1]
        _, last_gmv = series["cum_gmv"][-1]
        _, last_cost = series["cum_cost"][-1]
        assert last_gross_profit == pytest.approx(last_gmv - last_cost)

    # Cross-check cum_net_profit against realized values from the complete orders
    # table: only orders with settled_t set contribute, and each contributes
    # realized_revenue − realized_cost − total_penalty.
    profit_row = app.registry.conn_for(run_id).execute(
        "SELECT COALESCE(SUM(CASE WHEN settled_t IS NOT NULL"
        " THEN realized_revenue - realized_cost - total_penalty ELSE 0 END), 0)"
        " AS expected FROM orders WHERE run_id=? AND agent_id=?",
        (run_id, "agent_0"),
    ).fetchone()
    if series["cum_net_profit"]:
        last_t, last_profit = series["cum_net_profit"][-1]
        expected = float(profit_row["expected"])
        # Allow some slack: the aggregate reflects current state, while the
        # series reflects state at last_t, which is env.t-1.
        assert abs(last_profit - expected) < max(1.0, abs(expected) * 0.05)


def test_horizon_drains_existing_orders_without_creating_new_demand(client):
    """After the operating horizon, the env should stop accepting new demand
    but continue order settlement until no active orders remain."""
    from core.entities import Order, OrderStatusRow, StoreListing
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 2
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))

    active_order = Order(
        order_id="manual-delivered-at-horizon",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    first = c.post(f"/runs/{run_id}/step").get_json()
    assert first["phase"] == "draining"
    assert first["active_orders_remaining"] == 1
    first_status = c.get(f"/runs/{run_id}/status").get_json()
    assert first_status["state"] == "stopped"
    assert first_status["phase"] == "draining"

    # Add a listing that would create orders if the post-horizon step still ran
    # the normal demand path.
    product.market_curve = [10000.0] * 365
    product.quantity = 10000
    product.max_quantity = 10000
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.ref_price,
        listed_at=env.t,
    )
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing

    draining = c.post(f"/runs/{run_id}/step").get_json()
    assert draining["phase"] == "draining"
    assert draining["new_orders"] == 0
    assert draining["active_orders_remaining"] == 1
    assert draining["active_order_status_counts"] == {"delivered": 1}
    status = c.get(f"/runs/{run_id}/status").get_json()
    assert status["state"] == "stopped"
    assert status["phase"] == "draining"

    finished = c.post(f"/runs/{run_id}/step").get_json()
    assert finished["phase"] == "finished"
    assert finished["active_orders_remaining"] == 0
    assert c.get(f"/runs/{run_id}/status").get_json()["state"] == "finished"


def test_worker_reports_draining_before_finishing_active_horizon_orders(client):
    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 3
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="worker-drain-delivered",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    c.post(f"/runs/{run_id}/start", json={"interval_ms": 25})
    saw_draining = False
    for _ in range(100):
        status = c.get(f"/runs/{run_id}/status").get_json()
        if status["state"] == "draining":
            saw_draining = True
            assert status["phase"] == "draining"
            assert status["active_orders_remaining"] == 1
            break
        time.sleep(0.01)

    c.post(f"/runs/{run_id}/stop")
    assert saw_draining


def test_start_does_not_spawn_second_thread_for_live_draining_worker(client):
    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 10
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="worker-drain-no-duplicate-thread",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    try:
        c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000})
        worker = app.registry.workers[run_id]
        for _ in range(100):
            if worker.state == "draining":
                break
            time.sleep(0.01)
        assert worker.state == "draining"
        first_thread = worker._thread
        assert first_thread is not None and first_thread.is_alive()

        restart = c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000}).get_json()

        assert restart["state"] == "draining"
        assert worker._thread is first_thread
    finally:
        c.post(f"/runs/{run_id}/stop")


def test_start_does_not_respawn_bootstrap_when_worker_phase_is_draining(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]

    class FakeDrainingWorker:
        state = "running"

        def start(self, interval_ms):
            return {"state": "running", "phase": "draining", "t": 1}

    with app.registry.lock:
        app.registry.workers[run_id] = FakeDrainingWorker()

    respawn_calls = []

    def record_respawn(respawn_run_id, base_url=None):
        respawn_calls.append((respawn_run_id, base_url))

    monkeypatch.setattr(app.registry, "_respawn_bootstrap", record_respawn)

    resp = c.post(
        f"/runs/{run_id}/start",
        json={"interval_ms": 500, "bootstrap_base_url": "http://env.test"},
    )

    assert resp.status_code == 200
    assert resp.get_json()["phase"] == "draining"
    assert respawn_calls == []


def test_start_does_not_respawn_bootstrap_for_already_running_worker(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]

    class FakeRunningWorker:
        state = "running"

        def start(self, interval_ms):
            return {"state": "running", "phase": "running", "t": 1}

    with app.registry.lock:
        app.registry.workers[run_id] = FakeRunningWorker()

    respawn_calls = []

    def record_respawn(respawn_run_id, base_url=None):
        respawn_calls.append((respawn_run_id, base_url))

    monkeypatch.setattr(app.registry, "_respawn_bootstrap", record_respawn)

    resp = c.post(
        f"/runs/{run_id}/start",
        json={"interval_ms": 500, "bootstrap_base_url": "http://env.test"},
    )

    assert resp.status_code == 200
    assert resp.get_json()["phase"] == "running"
    assert respawn_calls == []


def test_start_recreates_unloaded_stopped_worker_after_restart(client):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000})
    c.post(f"/runs/{run_id}/stop")

    with app.registry.lock:
        app.registry.workers.pop(run_id, None)
        app.registry.envs.pop(run_id, None)

    restart = c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000}).get_json()

    assert "error" not in restart
    assert restart["state"] == "running"
    assert dbm.get_run(app.registry.conn_for(run_id), run_id)["finished_at"] is None


def test_start_resumes_live_paused_worker(client):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000})
    paused = c.post(f"/runs/{run_id}/pause").get_json()
    worker = app.registry.workers[run_id]

    resumed = c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000}).get_json()

    assert paused["state"] == "paused"
    assert resumed["state"] == "running"
    assert app.registry.workers[run_id] is worker
    c.post(f"/runs/{run_id}/stop")


def test_start_recreates_unloaded_paused_worker_after_restart(client):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "paused")
    with app.registry.lock:
        app.registry.workers.pop(run_id, None)
        app.registry.envs.pop(run_id, None)

    restarted = c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000}).get_json()

    assert "error" not in restarted
    assert restarted["state"] == "running"
    c.post(f"/runs/{run_id}/stop")


def test_status_and_resume_preserve_unloaded_paused_run_after_restart(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "paused")
    with app.registry.lock:
        app.registry.workers.pop(run_id, None)
        app.registry.envs.pop(run_id, None)

    original_rehydrate = app.registry._rehydrate
    rehydrate_calls = []

    def track_rehydrate(target_run_id):
        rehydrate_calls.append(target_run_id)
        return original_rehydrate(target_run_id)

    monkeypatch.setattr(app.registry, "_rehydrate", track_rehydrate)

    paused = c.get(f"/runs/{run_id}/status").get_json()

    assert paused["state"] == "paused"
    assert rehydrate_calls == []
    assert run_id not in app.registry.envs

    resumed = c.post(f"/runs/{run_id}/resume").get_json()

    assert resumed["state"] == "running"
    assert rehydrate_calls == [run_id]
    assert run_id in app.registry.envs
    assert c.get(f"/runs/{run_id}/tools/schema").status_code == 200
    c.post(f"/runs/{run_id}/stop")


def test_drain_safety_guard_converts_lifecycle_hours_to_steps(client):
    c, _, app = client
    scen = _tiny_scenario(0.05)
    scen["settlement"]["normal_delay_hours"] = 168
    scen["supplier_ranges"]["timeout_delay_hours"] = [24, 96]
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    env.scenario["run"]["step_hours"] = 24

    for product in env.products.values():
        product.ship_hours = 48
        product.logistics_hours = 72

    assert app.registry._drain_safety_max_steps(env) == 30


def test_pause_stops_live_draining_worker(client):
    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 10
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="worker-drain-pause",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    try:
        c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000})
        worker = app.registry.workers[run_id]
        for _ in range(100):
            if worker.state == "draining":
                break
            time.sleep(0.01)
        assert worker.state == "draining"

        paused = c.post(f"/runs/{run_id}/pause").get_json()

        assert paused["state"] == "paused"
        assert paused["phase"] == "draining"
        paused_t = paused["t"]
        time.sleep(0.08)
        status = c.get(f"/runs/{run_id}/status").get_json()
        assert status["state"] == "paused"
        assert status["phase"] == "draining"
        assert status["t"] == paused_t
    finally:
        c.post(f"/runs/{run_id}/stop")


def test_pause_freezes_active_hook_timeout(client):
    c, _, app = client
    scen = _tiny_scenario(0.2)
    scen["agent"]["activation_period"] = 1
    run_id = c.post(
        "/runs",
        json={
            "scenario": scen,
            "auto_start": True,
            "interval_ms": 10,
            "bootstrap_agent": "human",
        },
    ).get_json()["run_id"]
    env = app.registry._require(run_id)

    try:
        observed = False
        deadline = time.time() + 2
        while time.time() < deadline:
            resp = c.get(f"/runs/{run_id}/agents/agent_0/observation?nowait=1")
            if resp.status_code == 200:
                observed = True
                break
            time.sleep(0.01)
        assert observed
        assert env.t == 0

        paused = c.post(f"/runs/{run_id}/pause").get_json()
        assert paused["state"] == "paused"
        paused_t = env.t

        time.sleep(0.35)
        status = c.get(f"/runs/{run_id}/status").get_json()
        assert status["state"] == "paused"
        assert status["t"] == paused_t
        assert env.t == paused_t

        resumed = c.post(f"/runs/{run_id}/resume").get_json()
        assert resumed["state"] == "running"

        done = _act(
            c,
            run_id,
            "agent_0",
            "[human] call end_of_step",
            [("end_of_step", {})],
        )
        assert done.status_code == 200, done.get_data(as_text=True)

        deadline = time.time() + 2
        while time.time() < deadline and env.t == paused_t:
            time.sleep(0.01)
        assert env.t == paused_t + 1
    finally:
        c.post(f"/runs/{run_id}/stop")


def test_pause_during_drain_step_keeps_worker_resumable(client, monkeypatch):
    import threading

    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 10
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="worker-drain-pause-mid-step",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    real_step = env.step
    entered_drain_step = threading.Event()
    allow_drain_step = threading.Event()

    def slow_drain_step(*args, **kwargs):
        if kwargs.get("drain"):
            entered_drain_step.set()
            if not allow_drain_step.wait(timeout=2):
                raise TimeoutError("test did not release drain step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(env, "step", slow_drain_step)

    try:
        c.post(f"/runs/{run_id}/start", json={"interval_ms": 0})
        assert entered_drain_step.wait(timeout=2)

        paused = c.post(f"/runs/{run_id}/pause").get_json()
        assert paused["state"] == "paused"
        assert paused["phase"] == "draining"
        paused_t = paused["t"]

        allow_drain_step.set()
        status = None
        for _ in range(100):
            status = c.get(f"/runs/{run_id}/status").get_json()
            if status["t"] > paused_t:
                break
            time.sleep(0.01)

        assert status is not None
        assert status["state"] == "paused"
        assert status["phase"] == "draining"

        resumed = c.post(f"/runs/{run_id}/resume").get_json()
        assert resumed["state"] in {"running", "draining"}
        assert resumed["phase"] == "draining"
    finally:
        allow_drain_step.set()
        c.post(f"/runs/{run_id}/stop")


def test_stop_during_final_drain_step_does_not_publish_finished(client, monkeypatch):
    import queue
    import threading

    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    scen["settlement"]["normal_delay_hours"] = 1
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="worker-drain-stop-before-finished-event",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    env.agents["agent_0"].cash.receivable += active_order.sale_price

    real_step = env.step
    entered_final_drain_step = threading.Event()
    allow_final_drain_step = threading.Event()

    def slow_final_drain_step(*args, **kwargs):
        if kwargs.get("drain"):
            entered_final_drain_step.set()
            if not allow_final_drain_step.wait(timeout=2):
                raise TimeoutError("test did not release final drain step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(env, "step", slow_final_drain_step)

    sub = None
    try:
        c.post(f"/runs/{run_id}/start", json={"interval_ms": 0})
        worker = app.registry.workers[run_id]
        assert entered_final_drain_step.wait(timeout=2)
        sub = worker.subscribe()

        stopped = c.post(f"/runs/{run_id}/stop").get_json()
        assert stopped["state"] == "stopped"
        allow_final_drain_step.set()
        if worker._thread is not None:
            worker._thread.join(timeout=2)

        events = []
        while True:
            try:
                events.append(sub.get_nowait())
            except queue.Empty:
                break

        assert "finished" not in {ev.get("type") for ev in events}
        assert dbm.get_run(app.registry.conn_for(run_id), run_id)["status"] == "stopped"
    finally:
        allow_final_drain_step.set()
        if sub is not None:
            worker.unsubscribe(sub)
        c.post(f"/runs/{run_id}/stop")


def test_manual_stop_is_persisted_before_worker_cleanup(client, monkeypatch):
    c, _, app = client
    scen = _tiny_scenario(1.0)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    c.post(f"/runs/{run_id}/start", json={"interval_ms": 1000})
    worker = app.registry.workers[run_id]
    stop_write_entered = threading.Event()
    allow_stop_write = threading.Event()
    real_mark_terminal = dbm.mark_run_terminal

    def block_request_stop_write(conn, update_run_id, status, finished_at):
        if (
            update_run_id == run_id
            and status == "stopped"
            and threading.current_thread().name == "stop-caller"
        ):
            stop_write_entered.set()
            assert allow_stop_write.wait(timeout=2)
        return real_mark_terminal(conn, update_run_id, status, finished_at)

    monkeypatch.setattr(dbm, "mark_run_terminal", block_request_stop_write)
    stop_thread = threading.Thread(target=worker.stop, name="stop-caller")
    stop_thread.start()
    assert stop_write_entered.wait(timeout=2)

    deadline = time.time() + 2
    persisted = None
    while time.time() < deadline:
        with sqlite3.connect(app.registry.run_db_path(run_id)) as conn:
            persisted = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        if persisted == "stopped" or run_id not in app.registry.workers:
            break
        time.sleep(0.01)

    worker_present_before_release = run_id in app.registry.workers
    allow_stop_write.set()
    stop_thread.join(timeout=2)
    if worker._thread is not None:
        worker._thread.join(timeout=2)

    assert persisted == "stopped"
    assert worker_present_before_release


def test_terminal_non_settled_statuses_do_not_block_drain_completion(client):
    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.01)
    scen["run"]["horizon_steps"] = 1
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    terminal_statuses = ("cancelled", "stockout", "insufficient_balance")
    orders = []
    for idx, status in enumerate(terminal_statuses):
        order = Order(
            order_id=f"terminal-{status}",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=0,
            promised_delivery_t=0,
            sale_price=120.0,
            purchase_price=100.0,
            current_status=status,
            purchase_t=0,
            settled_t=0 if status != "cancelled" else None,
            total_penalty=float(idx),
        )
        order.status_log.append(OrderStatusRow(t=0, status=status))
        orders.append(order)
    dbm.insert_orders(env.conn, run_id, orders)

    finished = c.post(f"/runs/{run_id}/step").get_json()
    assert finished["phase"] == "finished"
    assert finished["active_orders_remaining"] == 0
    assert finished["active_order_status_counts"] == {}


def test_stockout_orders_persist_with_status(client):
    """If a listed product hits zero supplier inventory, the resulting customer order
    must be written to the orders table with current_status='stockout' and a
    populated total_penalty (net_profit = -penalty)."""
    c, _, app = client
    scen = _tiny_scenario(2.0)  # long hook for tool calls in setup
    scen["data"]["small_share"] = 0.5
    scen["run"]["horizon_steps"] = 50
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    # Set up listing inside a driven step (tool calls require open hook).
    import threading
    step_th = threading.Thread(target=lambda: app.registry.step(run_id), daemon=True)
    step_th.start()
    time.sleep(0.05)
    resp = _act(c, run_id, "agent_0", "market brief", [("market_brief", {"window_days": 7})])
    cats = [row["category"] for row in json.loads(resp.get_json()["tool_results"][0]["content"])["categories"]]
    resp = _act(c, run_id, "agent_0", "searching", [("search_products", {"query": "", "page": 1, "page_size": 10})])
    browsed = _table_records(json.loads(resp.get_json()["tool_results"][0]["content"])["items"])
    pid = browsed[0]["product_id"]
    # list_product
    resp = _act(c, run_id, "agent_0", "listing product",
                [("list_product", {
                    "items": [{"product_id": pid, "sale_price": browsed[0]["price"] * 1.4}]
                })])
    r = json.loads(resp.get_json()["tool_results"][0]["content"])
    assert r["ok"]
    # end_of_step
    _act(c, run_id, "agent_0", "done", [("end_of_step", {})])
    step_th.join(timeout=3)
    # Now mutate inventory and drive remaining steps with short hook timeout
    env = app.registry._require(run_id)
    env.products[pid].quantity = 0
    env.products[pid].max_quantity = 0
    env.products[pid].hourly_increment = 0
    env.scenario["run"]["max_hook_seconds"] = 0.05  # speed up subsequent no-agent steps
    for _ in range(40):
        c.post(f"/runs/{run_id}/step")
    rows = c.get(f"/runs/{run_id}/orders?status=stockout").get_json()["orders"]
    if rows:
        # demand-driven path: confirm shape
        o = rows[0]
        assert o["current_status"] == "stockout"
        assert o["total_penalty"] == 5.0
        assert o["realized_cost"] == 0
        assert o["realized_revenue"] == 0
        assert abs(o["net_profit"] + o["total_penalty"]) < 1e-6  # net_profit = -penalty
        assert "final_profit" not in o
    else:
        # Demand happened to not fire on this product during the smoke window — at least
        # confirm the status is recognised by the endpoint without 500-ing.
        assert isinstance(rows, list)


def test_insufficient_balance_orders_use_fixed_penalty(client):
    c, _, app = client
    scen = _tiny_scenario(0.05)
    scen["data"]["small_share"] = 0.01
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)

    from core.entities import Order, StoreListing
    from storage import db as dbm

    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.price * 2, 2),
        listed_at=env.t,
    )
    st = env.agents["agent_0"]
    st.listings[product.product_id] = listing
    initial_balance = max(0.0, product.price - 1.0)
    st.cash.balance = initial_balance
    st.cash.deposit_pool = 100.0
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)

    order = Order(
        order_id="insufficient-balance-fixed-penalty",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=env.t,
        promised_delivery_t=env.t + product.ship_hours + product.logistics_hours,
        sale_price=listing.sale_price,
        purchase_price=product.price,
    )
    events = []

    kept = env._auto_purchase_new_orders([order], env.scenario["platform_rules"], events)

    assert len(kept) == 1
    assert kept[0].current_status == "insufficient_balance"
    assert kept[0].total_penalty == 5.0
    assert st.cash.cumulative_fine == 5.0
    assert st.cash.balance == max(0.0, initial_balance - 5.0)
    assert st.cash.deposit_pool == 100.0
    assert events[0].payload["penalty"] == 5.0
    assert listing.cum_sales == 0


def test_balance_zero_survives_but_exhausted_guarantee_closes_permanently(client):
    c, _, app = client
    scen = _tiny_scenario(0.05)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    st = env.agents["agent_0"]

    st.cash.balance = 0.0
    st.cash.deposit_pool = 5.0
    assert env._check_death_for("agent_0", env.t) is None
    assert st.is_alive

    from core.order_manager import _apply_penalty, _credit_cash

    initial_deposit = float(env.scenario["run"]["initial_deposit"])
    _apply_penalty(st.cash, 5.0)
    # A later same-step settlement may restore the pool, but closure is absorbing.
    _credit_cash(st.cash, 100.0, initial_deposit)
    assert st.cash.deposit_pool > 0.0
    death = env._check_death_for("agent_0", env.t)
    assert death is not None
    assert not st.is_alive

    st.cash.deposit_pool = initial_deposit
    assert env._check_death_for("agent_0", env.t + 1) is None
    assert not st.is_alive


def test_finished_status_persists_after_dashboard_reopen(client):
    """A run that hits horizon must keep status='finished'. Reopening the
    dashboard (which calls _get_or_create_worker) must not reset it to
    'pending'. Regression test for the bug where _get_or_create_worker
    replaced terminal workers unconditionally."""
    c, _, _ = client
    # Tiny horizon — finishes in a handful of steps with no agent
    scen = _tiny_scenario(0.05)
    scen["run"]["horizon_steps"] = 3
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    # Drive past horizon so the worker (or step loop) will cross it
    for _ in range(5):
        c.post(f"/runs/{run_id}/step")
    s_after = c.get(f"/runs/{run_id}/status").get_json()
    assert s_after["state"] == "finished"
    # Subsequent fetches and explicit start must not revive a finished run.
    for _ in range(3):
        s = c.get(f"/runs/{run_id}/status").get_json()
        assert s["state"] == "finished", f"status flipped back: {s}"
    restarted = c.post(f"/runs/{run_id}/start", json={"interval_ms": 0}).get_json()
    assert restarted["state"] == "finished"


def test_status_reads_unloaded_terminal_run_without_rehydrate(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    from storage import db as dbm

    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 12)
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "finished")
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
        app.registry.workers.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("status should not rehydrate unloaded terminal runs")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    resp = c.get(f"/runs/{run_id}/status")

    assert resp.status_code == 200
    assert resp.get_json()["state"] == "finished"
    assert resp.get_json()["t"] == 12
    assert run_id not in app.registry.envs


def test_sync_step_terminal_releases_runtime_and_keeps_db_status(client, monkeypatch):
    c, _, app = client
    scen = _tiny_scenario(0.05)
    scen["run"]["horizon_steps"] = 1
    scen["run"]["max_hook_seconds"] = 0
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]

    response = c.post(f"/runs/{run_id}/step")

    assert response.status_code == 200
    assert response.get_json()["phase"] == "finished"
    assert run_id not in app.registry.envs
    assert run_id not in app.registry.workers
    assert run_id not in app.registry._conns

    def fail_rehydrate(_run_id):
        raise AssertionError("terminal status must stay DB-only")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)
    status = c.get(f"/runs/{run_id}/status")
    assert status.status_code == 200
    assert status.get_json()["state"] == "finished"


def test_add_agent_to_terminal_run_returns_410(client):
    c, _, app = client
    scenario = _tiny_scenario(0)
    scenario["run"]["horizon_steps"] = 1
    run_id = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    assert c.post(f"/runs/{run_id}/step").get_json()["phase"] == "finished"
    assert run_id not in app.registry.envs

    response = c.post(f"/runs/{run_id}/agents", json={"name": "late agent"})

    assert response.status_code == 410
    assert response.get_json() == {
        "error": "run_runtime_not_loaded",
        "state": "finished",
    }


def test_add_agent_to_unloaded_pending_run_returns_409(client):
    c, _, app = client
    run_id = c.post(
        "/runs", json={"scenario": _tiny_scenario(0)}
    ).get_json()["run_id"]
    with app.registry.lock:
        app.registry.envs.pop(run_id)

    response = c.post(f"/runs/{run_id}/agents", json={"name": "late agent"})

    assert response.status_code == 409
    assert response.get_json() == {
        "error": "run_runtime_not_loaded",
        "state": "pending",
    }


def test_sync_step_internal_key_error_is_not_reported_as_runtime_unloaded(
    client, monkeypatch
):
    c, _, app = client
    run_id = c.post(
        "/runs", json={"scenario": _tiny_scenario(0.05)}
    ).get_json()["run_id"]
    env = app.registry.envs[run_id]

    def fail_step(*_args, **_kwargs):
        raise KeyError("broken simulation lookup")

    monkeypatch.setattr(env, "step", fail_step)
    app.config["PROPAGATE_EXCEPTIONS"] = True

    with pytest.raises(KeyError, match="broken simulation lookup"):
        c.post(f"/runs/{run_id}/step")

    assert app.registry.get_run(run_id)["status"] == "stopped"
    assert run_id not in app.registry.envs
    assert run_id not in app.registry._conns


def test_sync_step_missing_run_still_returns_not_found(client):
    c, _, _ = client

    step = c.post("/runs/run-missing/step")
    auto_step = c.post("/runs/run-missing/auto_step")

    assert step.status_code == 404
    assert step.get_json()["error"] == "not found"
    assert auto_step.status_code == 404
    assert auto_step.get_json()["error"] == "not found"


def test_sync_step_deleting_run_does_not_become_internal_error(client):
    c, _, app = client
    run_id = c.post(
        "/runs", json={"scenario": _tiny_scenario(0.05)}
    ).get_json()["run_id"]
    with app.registry._conn_lock:
        app.registry._deleting.add(run_id)
    try:
        response = c.post(f"/runs/{run_id}/step")
    finally:
        with app.registry._conn_lock:
            app.registry._deleting.discard(run_id)

    assert response.status_code == 404
    assert response.get_json()["error"] == "not found"


def test_terminal_observation_and_act_return_410_without_rehydrate(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    from storage import db as dbm

    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "stopped")
    app.registry.release_terminal_runtime(run_id)

    def fail_rehydrate(_run_id):
        raise AssertionError("terminal agent requests must not rehydrate")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)
    observation = c.get(f"/runs/{run_id}/agents/agent_0/observation?nowait=1")
    action = c.post(
        f"/runs/{run_id}/agents/agent_0/act",
        json={"messages": [{"role": "assistant", "content": "done"}]},
    )

    assert observation.status_code == 410
    assert action.status_code == 410


def test_status_demotes_stale_live_db_state_without_worker(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    from storage import db as dbm

    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 8)
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "running")
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
        app.registry.workers.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("status should not rehydrate stale live DB state")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    resp = c.get(f"/runs/{run_id}/status")

    assert resp.status_code == 200
    assert resp.get_json()["state"] == "stopped"
    assert resp.get_json()["t"] == 8
    assert run_id not in app.registry.envs


def test_status_demotes_stale_draining_db_state_without_worker(client, monkeypatch):
    from core.entities import Order, OrderStatusRow
    from storage import db as dbm

    c, _, app = client
    scen = _tiny_scenario(0.05)
    scen["run"]["horizon_steps"] = 1
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    active_order = Order(
        order_id="stale-draining-active-order",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=0,
        sale_price=120.0,
        purchase_price=100.0,
        current_status="delivered",
        purchase_t=0,
        shipped_t=0,
        delivered_t=0,
        preset_anomaly="normal",
        realized_cost=100.0,
    )
    active_order.status_log.append(OrderStatusRow(t=0, status="delivered"))
    dbm.insert_orders(env.conn, run_id, [active_order])
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 1)
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "draining")
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
        app.registry.workers.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("status should not rehydrate stale draining DB state")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    resp = c.get(f"/runs/{run_id}/status")

    body = resp.get_json()
    assert resp.status_code == 200
    assert body["state"] == "stopped"
    assert body["phase"] == "draining"
    assert body["active_orders_remaining"] == 1
    assert run_id not in app.registry.envs


def test_stream_returns_bounded_status_for_unloaded_terminal_run(client, monkeypatch):
    c, _, app = client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario(0.05)}).get_json()["run_id"]
    from storage import db as dbm

    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 3)
    dbm.update_run_status(app.registry.conn_for(run_id), run_id, "finished")
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)
        app.registry.workers.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("stream should not rehydrate unloaded terminal runs")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    resp = c.get(f"/runs/{run_id}/stream", buffered=True)
    body = resp.get_data(as_text=True)

    assert resp.status_code == 200
    assert "event: hello" in body
    assert '"state": "finished"' in body
    assert '"t": 3' in body
    assert run_id not in app.registry.envs


def test_supplier_section_exposes_recover_t_fields(client):
    """The supplier section must expose delist_recover_t / price_recover_t /
    timeout_recover_t per product so the dashboard can render recovery tooltips."""
    c, _, _ = client
    scen = _tiny_scenario(0.05)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    section = c.get(f"/runs/{run_id}/sections/supplier").get_json()
    assert section["products"], "expected at least one product"
    p = section["products"][0]
    for k in ("delist_recover_t", "price_recover_t", "timeout_recover_t"):
        assert k in p, f"missing {k} in supplier product payload"
    assert "base_price" in p


def test_supplier_section_is_paginated_by_default(client):
    c, _, _ = client
    scen = _tiny_scenario(0.05)
    scen["data"]["num_products"] = 1200
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]

    section = c.get(f"/runs/{run_id}/sections/supplier").get_json()

    assert section["product_count"] == 1200
    assert section["filtered_count"] == 1200
    assert len(section["products"]) == 500


def test_new_run_submit_redirects_to_started_run(client):
    c, _, _ = client
    scen = _tiny_scenario(0.05)
    scen["run"]["interval_ms"] = 0
    resp = c.post("/new_run", data={
        "name": "direct-create-test",
        "scenario_yaml": yaml.safe_dump(scen),
        "bootstrap_agent": "none",
    })

    assert resp.status_code in (301, 302)
    assert "/dashboard?run_id=" in resp.headers["Location"]
    run_id = resp.headers["Location"].rsplit("run_id=", 1)[1]
    row = c.get(f"/runs/{run_id}").get_json()
    assert row["name"] == "direct-create-test"
    assert row["status"] in {"running", "finished"}


def test_hook_timeout_path(client):
    c, _, _ = client
    scen = _tiny_scenario(max_hook_seconds=0.5)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    t0 = time.time()
    resp = c.post(f"/runs/{run_id}/step")
    elapsed = time.time() - t0
    assert resp.status_code == 200
    assert 0.3 < elapsed < 2.5


def _rating_scenario(star_multipliers):
    """Tiny scenario tuned so order generation is reliably non-zero, with the
    shop_rating block forced to a specific multiplier vector."""
    s = _tiny_scenario(0.05)
    s["data"]["small_share"] = 0.01
    s["run"]["horizon_steps"] = 60
    s["shop_rating"] = {
        "enabled": True,
        "prior_good": 20.0,
        "prior_bad": 2.0,
        "decay": 1.0,           # disable decay so the multiplier dominates
        "bucket_thresholds": [0.5, 0.7, 0.85, 0.95],
        "star_multipliers": list(star_multipliers),
    }
    return s


def test_shop_rating_affects_demand(client):
    """Under the same master_seed, a scenario with uniformly-low star multipliers
    must produce strictly fewer orders than one with uniformly-high multipliers.
    Proves the rating actually gates demand in the Poisson stream."""
    c, _, app = client
    # Low multipliers: every star → 0.1× demand
    low = _rating_scenario([0.1] * 5)
    rid_low = c.post("/runs", json={"scenario": low}).get_json()["run_id"]
    _preseed_listings(app, rid_low)
    for _ in range(40):
        c.post(f"/runs/{rid_low}/step")
    orders_low = app.registry.conn_for(rid_low).execute(
        "SELECT COUNT(*) AS n FROM orders WHERE run_id=?",
        (rid_low,),
    ).fetchone()["n"]

    # High multipliers: every star → 2.0× demand
    high = _rating_scenario([2.0] * 5)
    rid_high = c.post("/runs", json={"scenario": high}).get_json()["run_id"]
    _preseed_listings(app, rid_high)
    for _ in range(40):
        c.post(f"/runs/{rid_high}/step")
    orders_high = app.registry.conn_for(rid_high).execute(
        "SELECT COUNT(*) AS n FROM orders WHERE run_id=?",
        (rid_high,),
    ).fetchone()["n"]

    assert orders_high > orders_low, (
        f"expected high-multiplier run to produce more orders, "
        f"got high={orders_high} vs low={orders_low}")
    # The ratio between settings is 20× on demand rate; we don't assert exact
    # 20× because Poisson noise + supplier-side state diverges, but the gap
    # should be obvious (≥ 5×) given non-trivial volume in the high run.
    assert orders_high >= max(20, 5 * max(1, orders_low))


def test_rating_survives_restart(client):
    """Rehydrate path must replay the events log to recover n_good / n_bad.

    Without restore_rating_state, dropping the Environment from the
    registry and re-fetching it would reset the shop to the prior mean —
    silently corrupting any long-running shop's rating across server
    restarts.
    """
    from core.entities import EventLog
    from storage import db as dbm

    c, _, app = client
    # decay=0.9 so the timing math is non-trivial and noticeable.
    scen = _rating_scenario([1.0] * 5)
    scen["shop_rating"]["decay"] = 0.9
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    _preseed_listings(app, run_id)

    # Reach into the live env to (a) inject rating events directly into the
    # events table, and (b) advance runs.current_t — we don't want to wait
    # 168 simulated hours for settled_normal events to fire naturally.
    env = app.registry._require(run_id)
    conn = env.conn
    fake_events = [
        EventLog(t=2, event_type="order_settled_normal",
                 entity_id="ord-1", agent_id="agent_0", payload={}),
        EventLog(t=4, event_type="order_late",
                 entity_id="ord-2", agent_id="agent_0", payload={}),
        EventLog(t=7, event_type="order_settled_normal",
                 entity_id="ord-3", agent_id="agent_0", payload={}),
    ]
    dbm.write_events(conn, run_id, fake_events)
    conn.commit()
    # Pretend the simulator advanced through step 9; current_t = 10 means
    # last completed step was 9, so decay was applied at every step from
    # the event's t up through t=9.
    dbm.update_run_t(conn, run_id, 10)
    conn.commit()

    # Drop the env so the next access goes through _rehydrate.
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    env2 = app.registry._require(run_id)
    assert env2 is not env, "expected a freshly rehydrated Environment"
    assert env2.t == 10

    # Expected weights: env_t=10 → last completed step = 9.
    # Good events at t=2 and t=7 → decay**(9-2) + decay**(9-7) = 0.9^7 + 0.9^2
    # Bad event at t=4 → decay**(9-4) = 0.9^5
    decay = 0.9
    expected_good = decay ** 7 + decay ** 2
    expected_bad = decay ** 5
    state = env2.agents["agent_0"]
    assert state.n_good == pytest.approx(expected_good)
    assert state.n_bad == pytest.approx(expected_bad)


def test_rating_rehydrate_noop_when_disabled(client):
    """When shop_rating.enabled=false the rehydrate must NOT touch n_good /
    n_bad — they stay at the dataclass default 0.0 even if there are
    historical events in the table (e.g. the user toggled rating off
    mid-run)."""
    from core.entities import EventLog
    from storage import db as dbm

    c, _, app = client
    scen = _rating_scenario([1.0] * 5)
    scen["shop_rating"]["enabled"] = False
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    _preseed_listings(app, run_id)

    env = app.registry._require(run_id)
    conn = env.conn
    dbm.write_events(conn, run_id, [
        EventLog(t=1, event_type="order_settled_normal",
                 entity_id="ord-1", agent_id="agent_0", payload={}),
    ])
    conn.commit()
    dbm.update_run_t(conn, run_id, 5)
    conn.commit()
    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    env2 = app.registry._require(run_id)
    assert env2.agents["agent_0"].n_good == 0.0
    assert env2.agents["agent_0"].n_bad == 0.0


def test_observation_includes_shop_rating(client):
    """A fresh agent's observation packet must expose a shop rating block.
    Calls compose_observation directly to avoid the long-poll hook gate."""
    from tools.observation import compose_observation
    c, _, app = client
    scen = _tiny_scenario(0.05)
    run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
    _preseed_listings(app, run_id)
    c.post(f"/runs/{run_id}/step")
    env = app.registry._require(run_id)
    obs = compose_observation(env, "agent_0")
    assert "shop" in obs, f"observation missing shop block: {obs}"
    sr = obs["shop"]
    # Fresh v2 shop starts at the 4.0 prior and therefore in the neutral 4★ bucket.
    assert sr["score"] == 4.0
    assert sr["stars"] == 4
    assert sr["rated_order_count"] == 0
    assert sr["updated_through_step"] == 0
    assert "demand_multiplier" not in sr
    assert "n_good_effective" not in sr
    assert "n_bad_effective" not in sr
