import os

import numpy as np
import pytest
from core import sim_time
from core.demand import expected_demand
from core.entities import EventLog, Order, OrderStatusRow, StoreListing
from storage import db as dbm
from tools import observation as obs_mod
from tools import tools as tools_mod
from web.app import create_app
from web.runner import load_default_scenario


def _tiny_scenario():
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 5
    scenario["run"]["max_hook_seconds"] = 0.01
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 20
    return scenario


def test_virtual_time_off_keeps_legacy_day_hour_shape():
    scenario = _tiny_scenario()
    scenario["run"].pop("virtual_time", None)

    assert sim_time.time_view(scenario, t=25, step_hours=1) == {"day": 2, "hour": 1}
    assert sim_time.demand_day_offset(scenario) == 0


def test_private_real_virtual_start_date_maps_to_calendar_and_curve_offset():
    scenario = _tiny_scenario()
    scenario["data"]["source"] = "private_real"
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-15",
    }

    view = sim_time.time_view(scenario, t=0, step_hours=1)

    assert view == {
        "day": 1,
        "hour": 0,
        "datetime": "2025-06-15T00:00:00",
    }
    assert sim_time.data_anchor_date(scenario).isoformat() == "2025-06-01"
    assert sim_time.demand_day_offset(scenario) == 14


def test_virtual_time_calendar_rolls_over_month_boundaries():
    scenario = _tiny_scenario()
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-30",
    }

    view = sim_time.time_view(scenario, t=24, step_hours=1)

    assert set(view.keys()) == {"day", "hour", "datetime"}
    assert view["datetime"] == "2025-07-01T00:00:00"
    assert view["day"] == 2
    assert view["hour"] == 0


def test_expected_demand_can_read_market_curve_from_virtual_offset():
    scenario = _tiny_scenario()
    from data.synth import generate

    products, hourly_dist = generate(scenario)
    product = products[0]
    product.market_curve = [float(i + 1) for i in range(365)]
    listing = StoreListing(product_id=product.product_id, sale_price=product.ref_price)
    hourly_w = np.ones(24)

    q_no_offset = expected_demand(
        product,
        listing,
        hourly_w,
        t=0,
        step_hours=1,
        small_share=1.0,
        day_offset=0,
    )
    q_offset = expected_demand(
        product,
        listing,
        hourly_w,
        t=0,
        step_hours=1,
        small_share=1.0,
        day_offset=5,
    )

    assert q_no_offset == pytest.approx(product.market_curve[0])
    assert q_offset == pytest.approx(product.market_curve[5])


def test_create_run_rejects_non_hourly_step_model(tmp_path):
    scenario = _tiny_scenario()
    scenario["run"]["step_hours"] = 5
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()

    response = client.post("/runs", json={"scenario": scenario})

    assert response.status_code == 400
    assert response.get_json()["error"] == "run.step_hours must be 1 in this release"


def test_observation_tick_includes_calendar_fields_when_virtual_time_enabled(tmp_path):
    scenario = _tiny_scenario()
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-15",
    }
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()
    run_id = client.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(run_id)
    env.t = 25

    packet = obs_mod.compose_observation(env, "agent_0")

    assert packet["tick"]["day"] == 2
    assert packet["tick"]["hour"] == 1
    assert packet["tick"]["step"] == 25
    assert set(packet["tick"].keys()) == {"day", "hour", "datetime", "step"}
    assert packet["tick"]["datetime"] == "2025-06-16T01:00:00"


def test_observation_text_uses_calendar_label_when_virtual_time_enabled(tmp_path):
    scenario = _tiny_scenario()
    scenario["agent"]["language"] = "zh"
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-10",
    }
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()
    run_id = client.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(run_id)

    packet = obs_mod.compose_observation(env, "agent_0")

    first_line = packet["text"].splitlines()[0]
    assert first_line == "Day 1, Hour 0 (2025-06-10T00:00:00)"


def test_supply_chain_anomaly_event_uses_safe_fields_when_virtual_time_enabled(tmp_path):
    scenario = _tiny_scenario()
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-10",
    }
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()
    run_id = client.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.t = 12
    dbm.write_events(
        env.conn,
        env.run_id,
        [
            EventLog(
                t=11,
                event_type="supplier_delist",
                entity_id=product.product_id,
                agent_id=None,
                payload={"recover_t": 36},
            ),
        ],
    )

    result = tools_mod.query_supply_chain_anomalies(env, "agent_0", mode="new")
    events = result["events"]

    assert events[0]["event_type"] == "supplier_delist"
    assert events[0]["product_id"] == product.product_id
    assert events[0]["before"] == {"supplier_listed": True}
    assert events[0]["after"] == {"supplier_listed": False}
    assert "recover_t" not in events[0]


def test_order_times_and_status_log_use_compact_virtual_time(tmp_path):
    scenario = _tiny_scenario()
    scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-01",
    }
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    client = app.test_client()
    run_id = client.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    order = Order(
        order_id="O_virtual_time",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=46,
        promised_delivery_t=72,
        sale_price=product.price + 1.0,
        purchase_price=product.price,
        current_status="ordered",
        status_log=[OrderStatusRow(t=46, status="ordered")],
    )
    dbm.insert_orders(env.conn, run_id, [order])

    detail = tools_mod.query_order_detail(env, "agent_0", "O_virtual_time")

    assert detail["order_time"] == {
        "day": 2,
        "hour": 22,
        "datetime": "2025-06-02T22:00:00",
    }
    assert detail["status_log"] == [
        {
            "time": {
                "day": 2,
                "hour": 22,
                "datetime": "2025-06-02T22:00:00",
            },
            "status": "ordered",
        }
    ]
