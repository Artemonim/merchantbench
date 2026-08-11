"""Direct unit tests for tools.observation.compose_observation."""
import os
import tempfile

import pytest

from core.entities import EventLog, Order, OrderStatusRow, StoreListing
from storage import db as dbm
from tools import observation as obs_mod
from web.app import create_app
from web.runner import load_default_scenario


def _table_records(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


@pytest.fixture
def env_factory():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    client = app.test_client()

    def make(scen_overrides=None, hook_seconds=0.05):
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = hook_seconds
        scen["run"]["horizon_steps"] = 5
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        scen.setdefault("agent", {})["tool_denylist"] = []
        if scen_overrides:
            scen.setdefault("agent", {}).update(scen_overrides)
        rid = client.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        return app.registry._require(rid), rid, client

    yield make


def test_packet_has_grouped_store_snapshot_sections(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert set(p.keys()) == {"agent_id", "tick", "orders", "supply", "cash",
                             "shop", "daily_report_available", "text"}
    assert "computed_at_wall_ms" not in p
    assert set(p["orders"].keys()) == {
        "changes_since_last_observation",
        "totals",
    }
    assert set(p["supply"].keys()) == {
        "listings",
        "events_since_last_observation",
        "current_risks",
        "new_risks_since_last_observation",
    }
    assert p["supply"]["listings"]["max"] == 50


def test_observation_notices_unread_daily_report_without_injecting_it(
    env_factory, tmp_path
):
    report_dir = tmp_path / "daily_reports"
    report_dir.mkdir()
    (report_dir / "20250601.md").write_text(
        "# 【6月1日 市场商机速递】\n\nSynthetic test report.\n",
        encoding="utf-8",
    )
    (report_dir / "20250602.md").write_text(
        "# 【6月2日 市场商机速递】\n\nSynthetic test report.\n",
        encoding="utf-8",
    )
    env, _, _ = env_factory()
    env.scenario["data"]["daily_report_dir"] = str(report_dir)
    p = obs_mod.compose_observation(env, "agent_0")

    assert p["daily_report_available"] is True
    assert (
        "Daily report available: use get_daily_report for today's published market brief"
        in p["text"]
    )
    assert "# 【6月1日 市场商机速递】" not in p["text"]

    report = obs_mod.t.get_daily_report(env, "agent_0")
    assert report["ok"] is True
    after_read = obs_mod.compose_observation(env, "agent_0")
    assert after_read["daily_report_available"] is False
    assert "Daily report available:" not in after_read["text"]

    env.t = 24
    next_day = obs_mod.compose_observation(env, "agent_0")
    assert next_day["daily_report_available"] is True
    assert "Daily report available:" in next_day["text"]


def test_observation_omits_missing_daily_report_without_a_notice(env_factory, tmp_path):
    report_dir = tmp_path / "missing_daily_reports"
    report_dir.mkdir()
    env, _, _ = env_factory()
    env.scenario["data"]["daily_report_dir"] = str(report_dir)

    p = obs_mod.compose_observation(env, "agent_0")

    assert p["daily_report_available"] is False
    assert "Daily report available:" not in p["text"]
    assert "daily report not found" not in p["text"]
    assert "\n\nShop:\nInternal service quality (no direct v4 demand effect): " in p["text"]
    assert "Public reviews (drives demand): rating n/a / count 0" in p["text"]
    assert p["text"].endswith(
        "Continue operating the store. Goal: maximize net_assets."
    )


def test_observation_renders_shelf_utilization_without_guidance(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert "Shelf utilization: active 0 / max 50 / free 50" in p["text"]
    assert "Empty shelf slots reduce product exposure." not in p["text"]
    assert "Shelf utilization: active 0 / max 50 / free 50\n\nevents since last observation" in p["text"]
    assert "scarce business resource" not in p["text"]
    assert "listings: active" not in p["text"]
    assert "free_slots" not in p["text"]


def test_observation_text_defaults_to_english_when_language_missing(env_factory):
    env, _, _ = env_factory()
    env.scenario["agent"].pop("language", None)
    p = obs_mod.compose_observation(env, "agent_0")
    assert "Orders:" in p["text"]
    assert "\n\nSupply & listings:\n" in p["text"]
    assert "\n\nCash:\n" in p["text"]
    assert "\n\nShop:\n" in p["text"]
    assert "\n\nShop:\nInternal service quality (no direct v4 demand effect): " in p["text"]
    assert "Public reviews (drives demand): rating n/a / count 0" in p["text"]
    assert p["text"].endswith("Continue operating the store. Goal: maximize net_assets.")
    assert "我的商品异常" not in p["text"]


def test_tick_has_day_hour_and_default_virtual_datetime(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert set(p["tick"].keys()) == {"day", "hour", "datetime", "step"}
    assert p["tick"]["step"] == 0
    assert p["tick"]["day"] == 1
    assert p["tick"]["hour"] == 0
    assert p["tick"]["datetime"] == "2025-06-01T00:00:00"
    assert "t" not in p["tick"]
    assert "horizon" not in p["tick"]
    assert "sim_hours_elapsed" not in p["tick"]
    assert "step_hours" not in p["tick"]
    assert "max_hook_seconds" not in p["tick"]
    assert "hook_open" not in p["tick"]


def test_cash_has_full_agent_balance_fields(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert set(p["cash"].keys()) == {
        "balance",
        "deposit_pool",
        "in_transit",
        "receivable",
        "net_assets",
        "cumulative_fine",
    }


def test_orders_summary_has_changes_and_totals(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    expected_totals = {
        "total", "ordered", "late", "shipped", "delivered", "cancelled",
        "settled_normal", "settled_refund", "settled_only_refund",
        "settled_bad_review", "stockout", "insufficient_balance",
    }
    assert set(p["orders"]["changes_since_last_observation"].keys()) == expected_totals
    assert set(p["orders"]["totals"].keys()) == expected_totals
    assert p["orders"]["changes_since_last_observation"]["total"] == 0


def test_order_totals_are_current_status_counts(env_factory):
    env, _, _ = env_factory()
    statuses = [
        "ordered", "late", "shipped", "delivered", "cancelled",
        "settled_normal", "settled_refund", "settled_only_refund",
        "settled_bad_review", "stockout", "insufficient_balance",
    ]
    orders = []
    for i, status in enumerate(statuses):
        settled_t = i if status.startswith("settled_") else None
        late_t = i if status == "late" else None
        orders.append(Order(
            order_id=f"status-{i}",
            product_id="p",
            supplier_id="s",
            agent_id="agent_0",
            order_t=i,
            promised_delivery_t=i + 24,
            sale_price=100.0,
            purchase_price=80.0,
            current_status=status,
            settled_t=settled_t,
            late_t=late_t,
        ))
    dbm.insert_orders(env.conn, env.run_id, orders)

    p = obs_mod.compose_observation(env, "agent_0")

    assert p["orders"]["totals"] == {"total": len(statuses), **{s: 1 for s in statuses}}


def test_order_changes_are_status_entry_counts(env_factory):
    env, _, _ = env_factory()
    env.t = 20
    statuses = [
        "ordered", "late", "shipped", "delivered", "cancelled",
        "settled_normal", "settled_refund", "settled_only_refund",
        "settled_bad_review", "stockout", "insufficient_balance",
    ]
    orders = []
    for i, status in enumerate(statuses):
        orders.append(Order(
            order_id=f"changed-status-{i}",
            product_id="p",
            supplier_id="s",
            agent_id="agent_0",
            order_t=i,
            promised_delivery_t=i + 24,
            sale_price=100.0,
            purchase_price=80.0,
            current_status=status,
            settled_t=i if status.startswith("settled_") else None,
            late_t=i if status == "late" else None,
            status_log=[OrderStatusRow(t=i, status=status)],
        ))
    dbm.insert_orders(env.conn, env.run_id, orders)

    p = obs_mod.compose_observation(env, "agent_0")

    assert p["orders"]["changes_since_last_observation"] == {
        "total": len(statuses),
        **{status: 1 for status in statuses},
    }


def test_supply_event_counts_are_grouped(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert set(p["supply"]["events_since_last_observation"].keys()) == {
        "price_changes", "supplier_delists", "timeouts", "stockouts"}
    for v in p["supply"]["events_since_last_observation"].values():
        assert isinstance(v, int)


def test_day_hour_advance_with_t(env_factory):
    """t=25 with step_hours=1 → Day 2, Hour 1."""
    env, _, _ = env_factory()
    env.t = 25
    p = obs_mod.compose_observation(env, "agent_0")
    assert p["tick"]["day"] == 2
    assert p["tick"]["hour"] == 1


def test_supply_event_counts_t0(env_factory):
    env, _, _ = env_factory()
    p = obs_mod.compose_observation(env, "agent_0")
    assert p["supply"]["events_since_last_observation"] == {
        "price_changes": 0,
        "supplier_delists": 0,
        "timeouts": 0,
        "stockouts": 0,
    }


def test_supply_event_counts_zero_when_no_listings(env_factory):
    env, rid, client = env_factory()
    client.post(f"/runs/{rid}/step")
    p = obs_mod.compose_observation(env, "agent_0")
    assert p["supply"]["events_since_last_observation"] == {
        "price_changes": 0,
        "supplier_delists": 0,
        "timeouts": 0,
        "stockouts": 0,
    }


def _act(client, rid, agent_id, thought, tool_calls_spec):
    import json as _json
    tc_list = []
    for i, (name, args) in enumerate(tool_calls_spec):
        tc_list.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": _json.dumps(args)},
        })
    body = {
        "messages": [{"role": "assistant", "content": thought, "tool_calls": tc_list}],
    }
    return client.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)


def test_supply_event_counts_filter_to_my_products(env_factory):
    import json as _json, threading, time as _time
    env, rid, client = env_factory(hook_seconds=2.0)
    app = client.application
    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    deadline = _time.monotonic() + 2.0
    while not env.hook_open and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert env.hook_open
    r1 = _act(client, rid, "agent_0", "market", [("market_brief", {"window_days": 7})])
    cats = [row["category"] for row in _json.loads(r1.get_json()["tool_results"][0]["content"])["categories"]]
    r2 = _act(client, rid, "agent_0", "search", [("search_products", {"query": "", "page": 1, "page_size": 2})])
    browsed = _table_records(_json.loads(r2.get_json()["tool_results"][0]["content"])["items"])
    mine = browsed[0]["product_id"]
    other = browsed[1]["product_id"]
    _act(client, rid, "agent_0", "list", [("list_product", {
        "items": [{"product_id": mine, "sale_price": browsed[0]["price"] * 1.4}]
    })])
    _act(client, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)
    dbm.write_events(env.conn, env.run_id, [
        EventLog(t=0, event_type="price_change", entity_id=mine, agent_id=None,
                 payload={"new_price": 14.5, "ref_price": 12.0}),
        EventLog(t=0, event_type="price_change", entity_id=mine, agent_id=None,
                 payload={"new_price": 13.0, "ref_price": 12.0}),
        EventLog(t=0, event_type="supplier_delist", entity_id=mine, agent_id=None,
                 payload={"recover_t": 36}),
        EventLog(t=0, event_type="order_stockout_violation", entity_id=mine,
                 agent_id="agent_1", payload={"order_id": "other-agent-order"}),
        EventLog(t=0, event_type="price_change", entity_id=other, agent_id=None,
                 payload={"new_price": 9.9, "ref_price": 8.0}),
        EventLog(t=0, event_type="inventory_inc", entity_id=mine, agent_id=None,
                 payload={"from": 10, "to": 20}),
    ])
    env.t = 1
    p = obs_mod.compose_observation(env, "agent_0")
    assert p["supply"]["events_since_last_observation"] == {
        "price_changes": 2,
        "supplier_delists": 1,
        "timeouts": 0,
        "stockouts": 0,
    }


def test_observation_changes_cover_since_last_observation_not_previous_step(env_factory):
    env, _, _ = env_factory()
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price * 1.4,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    env.last_observation_step_by_agent = {"agent_0": 0}
    env.t = 12
    dbm.write_events(env.conn, env.run_id, [
        EventLog(t=0, event_type="price_change", entity_id=product.product_id,
                 agent_id=None, payload={"old_price": 10.0, "new_price": 12.0}),
        EventLog(t=5, event_type="supplier_timeout", entity_id=product.product_id,
                 agent_id=None, payload={"recover_t": 20}),
        EventLog(t=12, event_type="price_change", entity_id=product.product_id,
                 agent_id=None, payload={"old_price": 12.0, "new_price": 11.0}),
    ])
    dbm.insert_orders(env.conn, env.run_id, [
        Order(
            order_id="order-1",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=11,
            promised_delivery_t=15,
            sale_price=100.0,
            purchase_price=80.0,
            current_status="late",
            late_t=11,
            status_log=[
                OrderStatusRow(t=11, status="ordered"),
                OrderStatusRow(t=11, status="late"),
            ],
        )
    ])

    p = obs_mod.compose_observation(env, "agent_0")

    # Step 0 was already included in the previous complete observation; the
    # current completed step 12 is included in this one.
    assert p["supply"]["events_since_last_observation"]["price_changes"] == 1
    assert p["supply"]["events_since_last_observation"]["timeouts"] == 1
    changes = p["orders"]["changes_since_last_observation"]
    assert changes["ordered"] == 1
    assert changes["late"] == 1
