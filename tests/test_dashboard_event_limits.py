import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pytest
from core.entities import Cash, EventLog, Order, OrderStatusRow, StoreListing
from core.inventory import effective_quantity
from storage import db as dbm
from storage import snapshot as snap
from web.app import create_app
from web.runner import load_default_scenario


@pytest.fixture
def app_client():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"), runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        yield c, tmp, app


def _tiny_scenario():
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 24
    scenario["run"]["max_hook_seconds"] = 0.1
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 10
    return scenario


def _wait_for_catalog_diagnostics(c, run_id: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = c.get(f"/runs/{run_id}/sections/catalog_diagnostics")
        if response.status_code == 200:
            return
        assert response.status_code == 202
        time.sleep(0.01)
    raise AssertionError("catalog diagnostics did not finish materializing")


def _seed_dashboard_events(app, run_id: str, *, supply_count: int, order_count: int) -> None:
    env = app.registry._require(run_id)
    env.t = 50
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    events = []
    for i in range(supply_count):
        events.append(
            EventLog(
                t=i % 51,
                event_type="price_change" if i % 2 else "supplier_delist",
                entity_id=f"product-{i}",
                agent_id="",
                payload={"i": i},
            )
        )
    for i in range(order_count):
        events.append(
            EventLog(
                t=i % 51,
                event_type="order_created" if i % 2 else "order_shipped",
                entity_id=f"order-{i}",
                agent_id="agent_0",
                payload={"i": i},
            )
        )
    dbm.write_events(app.registry.conn_for(run_id), run_id, events)


def test_orders_section_no_longer_returns_recent_events(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    _seed_dashboard_events(app, run_id, supply_count=350, order_count=350)

    resp = c.get(f"/runs/{run_id}/sections/orders")

    assert resp.status_code == 200
    out = resp.get_json()
    assert "recent_events" not in out
    assert "event_meta" not in out
    assert "flow_series" not in out
    assert "status_series" not in out
    assert "status_cum_series" in out
    assert "orders_timeline" in out


def test_read_only_sections_do_not_rehydrate_unloaded_runs(app_client, monkeypatch):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("read-only dashboard section should not rehydrate")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    orders = c.get(f"/runs/{run_id}/sections/orders")
    supplier = c.get(f"/runs/{run_id}/sections/supplier?limit=5")

    assert orders.status_code == 200
    assert orders.get_json()["t"] == 0
    assert supplier.status_code == 200
    assert supplier.get_json()["t"] == 0


def test_dashboard_supplier_products_expose_debug_identity_fields(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    product = next(iter(app.registry._require(run_id).products.values()))

    resp = c.get(f"/runs/{run_id}/sections/supplier?q={product.product_id}&limit=1")

    assert resp.status_code == 200
    row = resp.get_json()["products"][0]
    assert row["product_id"] == product.product_id
    assert row["ref_price"] == product.ref_price
    assert row["supplier_name"] == product.supplier_name


def test_dashboard_merchant_listings_expose_debug_product_fields(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    product.supplier_ship_hours = int(product.base_ship_hours) + 10
    product.quantity = 1
    product.quantity_updated_t = 0
    product.hourly_increment = 3
    product.max_quantity = 20
    dbm.upsert_product_state(app.registry.conn_for(run_id), run_id, product)
    env.t = 2
    env.agents["agent_0"].listings[product.product_id] = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.ref_price * 1.1, 2),
        listed_at=3,
    )

    resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant")

    assert resp.status_code == 200
    row = resp.get_json()["listings"][0]
    assert row["ref_price"] == product.ref_price
    assert row["base_price"] == product.base_price
    assert row["supplier_id"] == product.supplier_id
    assert row["supplier_name"] == product.supplier_name
    assert row["listed_at"] == 3
    assert row["supplier_ship_hours"] == product.supplier_ship_hours
    assert row["supplier_log_hour"] == product.supplier_ship_hours + product.logistics_hours
    assert row["quantity"] == effective_quantity(product, env.t)
    assert row["cancel_rate"] == product.cancel_rate
    assert row["refund_rate"] == product.refund_rate
    assert row["only_refund_rate"] == product.only_refund_rate
    assert row["timeout_rate"] == product.timeout_rate
    assert row["bad_review_rate"] == product.bad_review_rate
    assert row["price_change_rate"] == product.price_change_rate
    assert row["supplier_delist_rate"] == product.supplier_delist_rate
    assert row["elasticity"] == product.elasticity
    assert row["downstream_rating"] == 4.0

    supplier_resp = c.get(f"/runs/{run_id}/sections/supplier?q={product.product_id}&limit=1")
    assert supplier_resp.status_code == 200
    supplier_row = supplier_resp.get_json()["products"][0]
    assert supplier_row["supplier_ship_hours"] == product.supplier_ship_hours


def test_merchant_section_includes_daily_sales_by_product(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    products = sorted(env.products.values(), key=lambda p: p.product_id)[:2]

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-p0-a",
                product_id=products[0].product_id,
                supplier_id=products[0].supplier_id,
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=1, status="ordered")],
            ),
            Order(
                order_id="order-p0-b",
                product_id=products[0].product_id,
                supplier_id=products[0].supplier_id,
                agent_id="agent_0",
                order_t=2,
                promised_delivery_t=30,
                sale_price=100.0,
                purchase_price=70.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=2, status="ordered")],
            ),
            Order(
                order_id="order-p1-a",
                product_id=products[1].product_id,
                supplier_id=products[1].supplier_id,
                agent_id="agent_0",
                order_t=26,
                promised_delivery_t=54,
                sale_price=90.0,
                purchase_price=60.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=26, status="ordered")],
            ),
        ],
    )
    env.t = 30
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    out = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()

    daily = out["daily_sales_by_product"]
    assert daily["grain"] == "day"
    assert daily["days"] == [1, 2]
    assert daily["buckets"] == [
        {"key": "D1", "label": "D1", "start_day": 1, "end_day": 1},
        {"key": "D2", "label": "D2", "start_day": 2, "end_day": 2},
    ]
    by_product = {row["product_id"]: row for row in daily["series"]}
    assert by_product[products[0].product_id]["name"] == products[0].name
    assert by_product[products[0].product_id]["data"] == [
        {
            "bucket": "D1",
            "label": "D1",
            "start_day": 1,
            "end_day": 1,
            "day": 1,
            "orders": 2,
            "value": 2,
            "gmv": 220.0,
            "gross_profit": 70.0,
            "net_profit": 0.0,
            "supply_chain_anomalies": 0,
            "order_anomalies": 0,
        },
        {
            "bucket": "D2",
            "label": "D2",
            "start_day": 2,
            "end_day": 2,
            "day": 2,
            "orders": 0,
            "value": 0,
            "gmv": 0.0,
            "gross_profit": 0.0,
            "net_profit": 0.0,
            "supply_chain_anomalies": 0,
            "order_anomalies": 0,
        },
    ]
    assert by_product[products[1].product_id]["data"] == [
        {
            "bucket": "D1",
            "label": "D1",
            "start_day": 1,
            "end_day": 1,
            "day": 1,
            "orders": 0,
            "value": 0,
            "gmv": 0.0,
            "gross_profit": 0.0,
            "net_profit": 0.0,
            "supply_chain_anomalies": 0,
            "order_anomalies": 0,
        },
        {
            "bucket": "D2",
            "label": "D2",
            "start_day": 2,
            "end_day": 2,
            "day": 2,
            "orders": 1,
            "value": 1,
            "gmv": 90.0,
            "gross_profit": 30.0,
            "net_profit": 0.0,
            "supply_chain_anomalies": 0,
            "order_anomalies": 0,
        },
    ]


def test_merchant_daily_sales_gross_profit_excludes_unprocured_orders(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-ok",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=1, status="ordered")],
            ),
            Order(
                order_id="order-stockout",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=2,
                promised_delivery_t=30,
                sale_price=200.0,
                purchase_price=50.0,
                current_status="stockout",
                settled_t=2,
                total_penalty=5.0,
                status_log=[OrderStatusRow(t=2, status="stockout")],
            ),
            Order(
                order_id="order-insufficient",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=30,
                sale_price=140.0,
                purchase_price=20.0,
                current_status="insufficient_balance",
                settled_t=3,
                total_penalty=5.0,
                status_log=[OrderStatusRow(t=3, status="insufficient_balance")],
            ),
        ],
    )
    env.t = 10
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    daily = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()["daily_sales_by_product"]

    row = next(item for item in daily["series"] if item["product_id"] == product.product_id)
    assert row["data"][0]["orders"] == 3
    assert row["data"][0]["gmv"] == 120.0
    assert row["data"][0]["gross_profit"] == 40.0


def test_merchant_daily_sales_payload_includes_metric_selector_fields(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-profit",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="settled_normal",
                settled_t=20,
                realized_revenue=120.0,
                realized_cost=80.0,
                total_penalty=3.0,
                status_log=[
                    OrderStatusRow(t=1, status="ordered"),
                    OrderStatusRow(t=20, status="settled_normal"),
                ],
            ),
            Order(
                order_id="order-late",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=2,
                promised_delivery_t=30,
                sale_price=90.0,
                purchase_price=50.0,
                current_status="late",
                late_t=15,
                total_penalty=11.0,
                status_log=[
                    OrderStatusRow(t=2, status="ordered"),
                    OrderStatusRow(t=15, status="late"),
                ],
            ),
        ],
    )
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(
                t=3,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id="",
                payload={"product_id": product.product_id},
            ),
            EventLog(
                t=15,
                event_type="order_late",
                entity_id="order-late",
                agent_id="agent_0",
                payload={"order_id": "order-late", "product_id": product.product_id},
            ),
            EventLog(
                t=16,
                event_type="order_late",
                entity_id="order-late",
                agent_id="agent_0",
                payload={"order_id": "order-late", "product_id": product.product_id},
            ),
            EventLog(
                t=17,
                event_type="order_insufficient_balance_violation",
                entity_id=product.product_id,
                agent_id="agent_0",
                payload={"order_id": "order-late"},
            ),
        ],
    )
    env.t = 30
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    daily = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()["daily_sales_by_product"]

    row = next(item for item in daily["series"] if item["product_id"] == product.product_id)
    assert row["data"][0]["orders"] == 2
    assert row["data"][0]["value"] == 2
    assert row["data"][0]["gross_profit"] == 80.0
    assert row["data"][0]["net_profit"] == 37.0
    assert row["data"][0]["supply_chain_anomalies"] == 1
    assert row["data"][0]["order_anomalies"] == 1


def test_merchant_section_includes_listing_operation_series(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    products = sorted(env.products.values(), key=lambda p: p.product_id)
    first = products[0].product_id
    second = products[1].product_id
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(
                t=1, event_type="agent_list_product", entity_id=first, agent_id="agent_0", payload={"product_id": first}
            ),
            EventLog(
                t=2, event_type="agent_adjust_price", entity_id=first, agent_id="agent_0", payload={"product_id": first}
            ),
            EventLog(
                t=24,
                event_type="agent_delist_product",
                entity_id=first,
                agent_id="agent_0",
                payload={"product_id": first},
            ),
            EventLog(
                t=25,
                event_type="agent_list_product",
                entity_id=second,
                agent_id="agent_0",
                payload={"product_id": second},
            ),
            EventLog(
                t=26,
                event_type="agent_list_product",
                entity_id=second,
                agent_id="agent_1",
                payload={"product_id": second},
            ),
        ],
    )
    env.t = 48
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    out = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()

    listing_ops = out["listing_ops"]
    assert listing_ops["grain"] == "day"
    assert listing_ops["series"]["ops"][:2] == [[1, 2], [2, 2]]
    assert listing_ops["series"]["list"][:2] == [[1, 1], [2, 1]]
    assert listing_ops["series"]["delist"][:2] == [[1, 0], [2, 1]]
    assert listing_ops["series"]["price"][:2] == [[1, 1], [2, 0]]


def test_merchant_daily_sales_uses_bucket_local_products_without_other(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    products = sorted(env.products.values(), key=lambda p: p.product_id)

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id=f"order-{i}",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=i * 24,
                promised_delivery_t=i * 24 + 30,
                sale_price=100.0 + i,
                purchase_price=70.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=i * 24, status="ordered")],
            )
            for i, product in enumerate(products)
        ],
    )
    env.t = 90 * 24
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    out = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant").get_json()

    daily = out["daily_sales_by_product"]
    assert daily["grain"] == "week"
    product_ids = {row["product_id"] for row in daily["series"]}
    assert "__other__" not in product_ids
    assert product_ids == {p.product_id for p in products}
    for row in daily["series"]:
        non_zero_buckets = [point["bucket"] for point in row["data"] if point["value"]]
        assert len(non_zero_buckets) == 1


def test_merchant_daily_sales_query_aggregates_orders_before_fetching(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id=f"order-{i}",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=i,
                promised_delivery_t=i + 30,
                sale_price=100.0,
                purchase_price=70.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=i, status="ordered")],
            )
            for i in range(3)
        ],
    )

    class AggregatedOrdersConnection:
        def __init__(self, raw):
            self.raw = raw

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(sql.split())
            flat_args = args[0] if args and isinstance(args[0], tuple) else args
            supplier_only_events = {"price_change", "supplier_delist", "supplier_timeout"}
            if " FROM events" in normalized and supplier_only_events.intersection(flat_args):
                assert "entity_id IN" in normalized
            if "SELECT order_id, product_id FROM orders" in normalized:
                assert "order_id IN" in normalized
            if "FROM orders o" in normalized and "LEFT JOIN products p" in normalized:
                assert "GROUP BY" in normalized
                assert "COUNT(" in normalized
            return self.raw.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.raw, name)

    daily = dbm.load_dashboard_merchant_daily_sales_by_product(
        AggregatedOrdersConnection(app.registry.conn_for(run_id)),
        run_id,
        "agent_0",
        current_t=30,
        step_hours=env.scenario["run"]["step_hours"],
    )

    assert daily["series"][0]["data"][0]["value"] == 3


def test_merchant_daily_sales_scopes_legacy_agentless_stockout_by_order_owner(
    app_client,
):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="agent-1-stockout",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_1",
                order_t=1,
                promised_delivery_t=30,
                sale_price=100.0,
                purchase_price=70.0,
                current_status="stockout",
                status_log=[OrderStatusRow(t=1, status="ordered")],
            ),
        ],
    )
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(
                t=2,
                event_type="order_stockout_violation",
                entity_id="agent-1-stockout",
                agent_id="",
                payload={
                    "order_id": "agent-1-stockout",
                    "product_id": product.product_id,
                },
            ),
        ],
    )
    merchant_products = {
        product.product_id: (product.name, product.category),
    }

    agent_0 = dbm.load_dashboard_merchant_daily_sales_by_product(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        current_t=3,
        step_hours=env.scenario["run"]["step_hours"],
        merchant_products=merchant_products,
    )
    agent_1 = dbm.load_dashboard_merchant_daily_sales_by_product(
        app.registry.conn_for(run_id),
        run_id,
        "agent_1",
        current_t=3,
        step_hours=env.scenario["run"]["step_hours"],
        merchant_products=merchant_products,
    )

    def anomaly_totals(payload):
        points = [point for series in payload["series"] for point in series["data"]]
        return (
            sum(point["supply_chain_anomalies"] for point in points),
            sum(point["order_anomalies"] for point in points),
        )

    assert anomaly_totals(agent_0) == (0, 0)
    assert anomaly_totals(agent_1) == (1, 1)


def test_orders_schema_has_merchant_daily_sales_index():
    assert "ix_orders_run_agent_t_product" in dbm.SCHEMA
    assert "ON orders(run_id, agent_id, order_t, product_id)" in dbm.SCHEMA


def test_dashboard_template_omits_removed_merchant_ship_promise():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    merchant_table = html.split('<table id="tbl-merchant"', 1)[1].split("</table>", 1)[0]
    colgroup = merchant_table.split("<thead>", 1)[0]
    field_headers = merchant_table.split('<tr class="field-header-row">', 1)[1].split("</tr>", 1)[0]

    assert "promised_ship_hours" not in html
    assert "merchant promise" not in html
    assert 'lineSeries("promise"' not in html
    assert '<th colspan="6">Supplier</th>' in merchant_table
    assert colgroup.count("<col ") == 31
    assert field_headers.count("<th ") == 31


def test_merchant_product_lifecycle_synthesizes_list_event_without_ship_promise(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.ref_price * 1.1, 2),
        listed_at=1,
    )
    dbm.upsert_listing(app.registry.conn_for(run_id), run_id, "agent_0", listing)

    lifecycle = dbm.load_dashboard_merchant_product_sales_lifecycle(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        product.product_id,
        current_t=2,
        step_hours=env.scenario["run"]["step_hours"],
    )

    list_event = next(event for event in lifecycle["events"] if event["event_type"] == "agent_list_product")
    assert list_event["payload"]["sale_price"] == listing.sale_price
    assert list_event["payload"]["synthetic"] is True
    assert "promised_ship_hours" not in list_event["payload"]


def test_merchant_product_detail_includes_market_copy_and_agent_lifecycle(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product, other_product = sorted(env.products.values(), key=lambda p: p.product_id)[:2]
    env.agents["agent_0"].listings[product.product_id] = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.ref_price * 1.1, 2),
        listed_at=1,
    )
    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-day-1",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=3, status="ordered")],
            ),
            Order(
                order_id="order-day-2",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=25,
                promised_delivery_t=54,
                sale_price=110.0,
                purchase_price=75.0,
                current_status="late",
                late_t=28,
                status_log=[
                    OrderStatusRow(t=25, status="ordered"),
                    OrderStatusRow(t=28, status="late"),
                ],
            ),
            Order(
                order_id="order-other-product",
                product_id=other_product.product_id,
                supplier_id=other_product.supplier_id,
                agent_id="agent_0",
                order_t=25,
                promised_delivery_t=54,
                sale_price=130.0,
                purchase_price=85.0,
                current_status="late",
                late_t=29,
                status_log=[
                    OrderStatusRow(t=25, status="ordered"),
                    OrderStatusRow(t=29, status="late"),
                ],
            ),
        ],
    )
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(
                t=2,
                event_type="agent_list_product",
                entity_id=product.product_id,
                agent_id="agent_0",
                payload={"sale_price": 132.0},
            ),
            EventLog(
                t=5,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id="",
                payload={"old_price": 80.0, "new_price": 85.0},
            ),
            EventLog(
                t=6,
                event_type="agent_adjust_price",
                entity_id=product.product_id,
                agent_id="agent_0",
                payload={"old_price": 132.0, "new_price": 125.0},
            ),
            EventLog(
                t=28,
                event_type="order_stockout_violation",
                entity_id="order-day-2",
                agent_id="agent_0",
                payload={"product_id": product.product_id},
            ),
            EventLog(
                t=29, event_type="order_late", entity_id="order-day-2", agent_id="agent_0", payload={"penalty": 5.0}
            ),
            EventLog(
                t=29,
                event_type="order_late",
                entity_id="order-other-product",
                agent_id="agent_0",
                payload={"penalty": 5.0},
            ),
        ],
    )
    env.t = 30
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)
    _wait_for_catalog_diagnostics(c, run_id)

    resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant/products/{product.product_id}")

    assert resp.status_code == 200
    out = resp.get_json()
    assert out["product"]["product_id"] == product.product_id
    assert len(out["market_curve"]) == 365
    assert set(out["category_band"]) == {"p10", "p50", "p90"}
    lifecycle = out["agent_sales_lifecycle"]
    assert lifecycle["days"] == [1, 2]
    assert lifecycle["series"]["new_orders"] == [[1, 1], [2, 1]]
    assert lifecycle["series"]["booked_gmv"] == [[1, 120.0], [2, 110.0]]
    assert lifecycle["series"]["gross_profit"] == [[1, 40.0], [2, 35.0]]
    assert [event["event_type"] for event in lifecycle["events"]] == [
        "agent_list_product",
        "price_change",
        "agent_adjust_price",
        "order_stockout_violation",
        "order_late",
    ]
    assert [event["event_group"] for event in lifecycle["events"]] == [
        "agent_operation",
        "supplier_anomaly",
        "agent_operation",
        "order_anomaly",
        "order_anomaly",
    ]
    assert lifecycle["summary"] == [
        {
            "key": "order_anomaly",
            "label": "Order anomalies",
            "count": 1,
            "event_count": 2,
            "denominator": 2,
            "rate": 0.5,
            "display": "1/2 · 50.0%",
            "by_type": [
                {"event_type": "order_late", "label": "late", "count": 1},
                {
                    "event_type": "order_stockout_violation",
                    "label": "stockout",
                    "count": 1,
                },
            ],
        },
        {
            "key": "supplier_anomaly",
            "label": "Supplier anomalies",
            "count": 1,
            "event_count": 1,
            "denominator": 2,
            "denominator_unit": "d",
            "display": "1/2d",
            "by_type": [
                {"event_type": "price_change", "label": "price change", "count": 1},
            ],
        },
        {
            "key": "agent_operation",
            "label": "Agent operations",
            "count": 2,
            "event_count": 2,
            "display": "2x",
            "by_type": [
                {"event_type": "agent_adjust_price", "label": "adjust price", "count": 1},
                {"event_type": "agent_list_product", "label": "list", "count": 1},
            ],
        },
    ]


def test_merchant_product_lifecycle_gross_profit_excludes_unprocured_orders(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    env.agents["agent_0"].listings[product.product_id] = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.ref_price * 1.1, 2),
        listed_at=1,
    )
    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="lifecycle-ok",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=1, status="ordered")],
            ),
            Order(
                order_id="lifecycle-stockout",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=2,
                promised_delivery_t=30,
                sale_price=200.0,
                purchase_price=50.0,
                current_status="stockout",
                settled_t=2,
                total_penalty=5.0,
                status_log=[OrderStatusRow(t=2, status="stockout")],
            ),
            Order(
                order_id="lifecycle-insufficient",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=30,
                sale_price=140.0,
                purchase_price=20.0,
                current_status="insufficient_balance",
                settled_t=3,
                total_penalty=5.0,
                status_log=[OrderStatusRow(t=3, status="insufficient_balance")],
            ),
        ],
    )
    env.t = 10
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)
    _wait_for_catalog_diagnostics(c, run_id)

    resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant/products/{product.product_id}")

    assert resp.status_code == 200
    lifecycle = resp.get_json()["agent_sales_lifecycle"]
    assert lifecycle["series"]["new_orders"] == [[1, 3]]
    assert lifecycle["series"]["booked_gmv"] == [[1, 120.0]]
    assert lifecycle["series"]["gross_profit"] == [[1, 40.0]]


def test_merchant_product_lifecycle_endpoint_avoids_full_diagnostics(app_client, monkeypatch):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-lifecycle",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=3, status="ordered")],
            ),
        ],
    )
    env.t = 30
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, env.t)

    from web import routes_dashboard

    def fail_full_diagnostics(*_args, **_kwargs):
        raise AssertionError("lifecycle endpoint should not build full diagnostics")

    monkeypatch.setattr(
        routes_dashboard.catalog_diagnostics,
        "build_product_diagnostics",
        fail_full_diagnostics,
    )

    resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant/products/{product.product_id}/lifecycle")

    assert resp.status_code == 200
    out = resp.get_json()
    assert out["agent_id"] == "agent_0"
    assert out["product_id"] == product.product_id
    assert out["agent_sales_lifecycle"]["series"]["new_orders"] == [[1, 1], [2, 0]]


def test_merchant_product_lifecycle_queries_events_by_entity_scope(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-lifecycle-scoped",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=3, status="ordered")],
            ),
        ],
    )

    class EntityScopedEventsConnection:
        def __init__(self, raw):
            self.raw = raw

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(sql.split())
            has_entity_filter = "entity_id=?" in normalized or "entity_id IN" in normalized
            if "FROM events" in normalized and not has_entity_filter:
                raise AssertionError("event lifecycle queries must be entity-scoped")
            return self.raw.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.raw, name)

    lifecycle = dbm.load_dashboard_merchant_product_sales_lifecycle(
        EntityScopedEventsConnection(app.registry.conn_for(run_id)),
        run_id,
        "agent_0",
        product.product_id,
        current_t=30,
        step_hours=env.scenario["run"]["step_hours"],
    )

    assert lifecycle["series"]["new_orders"] == [[1, 1], [2, 0]]


def test_events_schema_has_entity_scoped_lifecycle_index():
    assert "ix_events_run_entity_type_t" in dbm.SCHEMA
    assert "ON events(run_id, entity_id, event_type, t)" in dbm.SCHEMA


def test_merchant_section_falls_back_to_db_without_rehydrate(app_client, monkeypatch):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=round(product.price * 1.2, 2),
        cum_sales=3,
        cum_revenue=round(product.price * 1.2 * 3, 2),
        listed_at=2,
        promised_logistics_hours=48,
    )
    dbm.upsert_listing(app.registry.conn_for(run_id), run_id, "agent_0", listing)
    dbm.write_cash_log(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        7,
        Cash(
            balance=1234.0,
            deposit_pool=900.0,
            in_transit=12.0,
            receivable=34.0,
            cumulative_fine=5.0,
        ),
    )
    dbm.write_metrics(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        7,
        {
            "balance": 1234.0,
            "net_assets": 2180.0,
            "shop_rating_score": 0.91,
            "shop_rating_stars": 4,
            "shop_n_good_effective": 10,
            "shop_n_bad_effective": 1,
            # * v4 headline stars come from public reviews, not shop_rating_stars.
            "public_review_rating": 4.0,
            "public_review_count": 5,
        },
    )
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(t=1, event_type="price_change", entity_id="noise", agent_id="", payload={}),
            EventLog(
                t=7,
                event_type="order_created",
                entity_id="order-1",
                agent_id="agent_0",
                payload={"product_id": product.product_id},
            ),
        ],
    )
    env.t = 7
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 7)

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("merchant dashboard fallback should not rehydrate")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant")

    assert resp.status_code == 200
    out = resp.get_json()
    assert out["t"] == 7
    assert out["agent_id"] == "agent_0"
    assert out["cash"]["balance"] == 1234.0
    assert out["listings"][0]["product_id"] == product.product_id
    assert out["listings"][0]["ref_price"] == product.ref_price
    assert out["listings"][0]["base_price"] == product.base_price
    assert out["listings"][0]["supplier_id"] == product.supplier_id
    assert out["listings"][0]["supplier_name"] == product.supplier_name
    assert out["listings"][0]["listed_at"] == 2
    assert out["listings"][0]["supplier_log_hour"] == product.ship_hours + product.logistics_hours
    assert out["listings"][0]["cancel_rate"] == product.cancel_rate
    assert out["listings"][0]["refund_rate"] == product.refund_rate
    assert out["listings"][0]["only_refund_rate"] == product.only_refund_rate
    assert out["listings"][0]["timeout_rate"] == product.timeout_rate
    assert out["listings"][0]["bad_review_rate"] == product.bad_review_rate
    assert out["listings"][0]["price_change_rate"] == product.price_change_rate
    assert out["listings"][0]["supplier_delist_rate"] == product.supplier_delist_rate
    assert out["listings"][0]["elasticity"] == product.elasticity
    assert out["listings"][0]["promised_logistics_hours"] == 48
    assert out["listings"][0]["downstream_rating"] == 4.0
    assert out["series"]["balance"] == [[7, 1234.0]]
    assert out["shop_rating"]["stars"] == 4
    assert out["recent_actions"] == [
        {
            "t": 7,
            "event_type": "order_created",
            "entity_id": "order-1",
            "payload": '{"product_id": "%s"}' % product.product_id,
        }
    ]


def test_dashboard_sections_as_of_use_replay_state_and_cut_future_rows(app_client, monkeypatch):
    c, tmp, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    _wait_for_catalog_diagnostics(c, run_id)
    env = app.registry._require(run_id)
    product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    agent = env.agents["agent_0"]

    snap.write_env_checkpoint(
        app.registry.runs_root,
        run_id,
        0,
        list(env.products.values()),
        current_t=0,
    )

    product_t10 = replace(
        product,
        price=111.0,
        quantity=33,
        quantity_updated_t=10,
        supplier_ship_hours=int(product.base_ship_hours) + 10,
        is_listed_by_supplier=True,
    )
    product_t20 = replace(
        product,
        price=222.0,
        quantity=9,
        quantity_updated_t=20,
        supplier_ship_hours=int(product.base_ship_hours) + 20,
        is_listed_by_supplier=False,
    )

    agent.cash = Cash(balance=1000.0, deposit_pool=800.0, in_transit=10.0, receivable=20.0)
    agent.listings = {
        product.product_id: StoreListing(
            product_id=product.product_id,
            sale_price=150.0,
            cum_sales=1,
            cum_revenue=150.0,
            listed_at=10,
            promised_logistics_hours=48,
        )
    }
    snap.write_env_delta_snapshot(
        app.registry.runs_root,
        run_id,
        10,
        dirty_products=[product_t10],
        agents=[agent],
        mutated_orders=[],
        events_this_step=[],
        survival_state={},
        current_t=10,
    )

    agent.cash = Cash(balance=2000.0, deposit_pool=700.0, in_transit=30.0, receivable=40.0)
    agent.listings[product.product_id] = StoreListing(
        product_id=product.product_id,
        sale_price=300.0,
        cum_sales=7,
        cum_revenue=2100.0,
        listed_at=10,
        promised_logistics_hours=24,
    )
    snap.write_env_delta_snapshot(
        app.registry.runs_root,
        run_id,
        20,
        dirty_products=[product_t20],
        agents=[agent],
        mutated_orders=[],
        events_this_step=[],
        survival_state={},
        current_t=20,
    )

    env.products[product.product_id] = product_t20
    env.t = 20
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 20)
    dbm.write_cash_log(app.registry.conn_for(run_id), run_id, "agent_0", 10, Cash(balance=1000.0))
    dbm.write_cash_log(app.registry.conn_for(run_id), run_id, "agent_0", 20, Cash(balance=2000.0))
    dbm.write_metrics(
        app.registry.conn_for(run_id),
        run_id,
        "_global",
        10,
        {
            "product_avail_count": 5,
            "mean_supplier_price": 111,
            "total_supplier_qty": 33,
        },
    )
    dbm.write_metrics(
        app.registry.conn_for(run_id),
        run_id,
        "_global",
        20,
        {
            "product_avail_count": 2,
            "mean_supplier_price": 222,
            "total_supplier_qty": 9,
        },
    )
    dbm.write_metrics(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        10,
        {
            "balance": 1000,
            "net_assets": 1030,
            "n_active_listings": 1,
        },
    )
    dbm.write_metrics(
        app.registry.conn_for(run_id),
        run_id,
        "agent_0",
        20,
        {
            "balance": 2000,
            "net_assets": 2070,
            "n_active_listings": 1,
        },
    )
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [
            EventLog(
                t=10,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id="",
                payload={"product_id": product.product_id},
            ),
            EventLog(
                t=10,
                event_type="order_created",
                entity_id="order-at-10",
                agent_id="agent_0",
                payload={"product_id": product.product_id},
            ),
            EventLog(
                t=20,
                event_type="order_created",
                entity_id="order-at-20",
                agent_id="agent_0",
                payload={"product_id": product.product_id},
            ),
        ],
    )

    with app.registry.lock:
        app.registry.envs.pop(run_id, None)

    def fail_rehydrate(_run_id):
        raise AssertionError("as_of dashboard sections should not rehydrate")

    monkeypatch.setattr(app.registry, "_rehydrate", fail_rehydrate)

    supplier = c.get(f"/runs/{run_id}/sections/supplier?as_of=10&q={product.product_id}").get_json()
    merchant = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant?as_of=10").get_json()
    selected_resp = c.get(f"/runs/{run_id}/agents/agent_0/sections/merchant/products/{product.product_id}?as_of=10")
    selected = selected_resp.get_json()

    assert supplier["t"] == 10
    assert supplier["kpis"]["mean_supplier_price"] == 111
    assert supplier["series"]["mean_supplier_price"] == [[10, 111.0]]
    assert supplier["products"][0]["product_id"] == product.product_id
    assert supplier["products"][0]["price"] == 111.0
    assert supplier["products"][0]["ref_price"] == product.ref_price
    assert supplier["products"][0]["supplier_name"] == product.supplier_name
    assert supplier["products"][0]["quantity"] == 33
    assert supplier["products"][0]["supplier_ship_hours"] == product_t10.supplier_ship_hours
    assert supplier["products"][0]["is_listed_by_supplier"] is True

    assert merchant["t"] == 10
    assert merchant["cash"]["balance"] == 1000.0
    assert merchant["series"]["balance"] == [[10, 1000.0]]
    assert merchant["listings"][0]["sale_price"] == 150.0
    assert merchant["listings"][0]["ref_price"] == product.ref_price
    assert merchant["listings"][0]["base_price"] == product.base_price
    assert merchant["listings"][0]["supplier_id"] == product.supplier_id
    assert merchant["listings"][0]["supplier_name"] == product.supplier_name
    assert merchant["listings"][0]["listed_at"] == 10
    assert merchant["listings"][0]["supplier_ship_hours"] == product_t10.supplier_ship_hours
    assert merchant["listings"][0]["supplier_log_hour"] == product_t10.supplier_ship_hours + product.logistics_hours
    assert merchant["listings"][0]["cancel_rate"] == product.cancel_rate
    assert merchant["listings"][0]["refund_rate"] == product.refund_rate
    assert merchant["listings"][0]["only_refund_rate"] == product.only_refund_rate
    assert merchant["listings"][0]["timeout_rate"] == product.timeout_rate
    assert merchant["listings"][0]["bad_review_rate"] == product.bad_review_rate
    assert merchant["listings"][0]["price_change_rate"] == product.price_change_rate
    assert merchant["listings"][0]["supplier_delist_rate"] == product.supplier_delist_rate
    assert merchant["listings"][0]["elasticity"] == product.elasticity
    assert merchant["listings"][0]["cum_sales"] == 1
    assert merchant["listings"][0]["downstream_rating"] == 4.0
    assert [a["entity_id"] for a in merchant["recent_actions"]] == ["order-at-10"]
    daily_products = {row["product_id"]: row for row in merchant["daily_sales_by_product"]["series"]}
    assert daily_products[product.product_id]["data"][0]["supply_chain_anomalies"] == 1
    assert selected_resp.status_code == 200
    assert selected["product"]["product_id"] == product.product_id
    assert selected["product"]["price"] == 111.0
    assert selected["product"]["quantity"] == 33


def test_orders_section_as_of_reconstructs_status_at_cutoff(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-old",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=8,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="settled_normal",
                purchase_t=8,
                shipped_t=12,
                delivered_t=16,
                settled_t=18,
                realized_revenue=120.0,
                realized_cost=80.0,
                total_penalty=15.0,
                status_log=[
                    OrderStatusRow(t=8, status="ordered"),
                    OrderStatusRow(t=12, status="shipped"),
                    OrderStatusRow(t=16, status="delivered"),
                    OrderStatusRow(t=18, status="settled_normal"),
                ],
            ),
            Order(
                order_id="order-future",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=14,
                promised_delivery_t=30,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="ordered",
                status_log=[OrderStatusRow(t=14, status="ordered")],
            ),
        ],
    )
    env.t = 20
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 20)

    out = c.get(f"/runs/{run_id}/sections/orders?as_of=10").get_json()

    assert out["t"] == 10
    assert out["status_cum_series"]["status_ordered"] == [[8, 1]]
    assert out["status_cum_series"]["status_shipped"] == []
    assert out["status_counts"] == {"ordered": 1}
    assert out["status_cum"] == {"ordered": 1}
    assert [o["order_id"] for o in out["orders_timeline"]] == ["order-old"]
    assert out["orders_timeline"][0]["current_status"] == "ordered"
    assert out["orders_timeline"][0]["realized_revenue"] == 0.0
    assert out["orders_timeline"][0]["realized_cost"] == 80.0
    assert out["orders_timeline"][0]["total_penalty"] == 0.0
    assert out["orders_timeline"][0]["net_profit"] == -80.0
    assert "final_profit" not in out["orders_timeline"][0]
    assert out["orders_timeline"][0]["status_log"] == [{"t": 8, "status": "ordered"}]


def test_orders_section_as_of_uses_later_same_tick_status(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            Order(
                order_id="order-same-tick",
                product_id=product.product_id,
                supplier_id=product.supplier_id,
                agent_id="agent_0",
                order_t=0,
                promised_delivery_t=6,
                sale_price=120.0,
                purchase_price=80.0,
                current_status="shipped",
                purchase_t=0,
                shipped_t=5,
                actual_ship_hours=5,
                late_t=5,
                status_log=[
                    OrderStatusRow(t=0, status="ordered"),
                    OrderStatusRow(t=5, status="late"),
                    OrderStatusRow(t=5, status="shipped"),
                ],
            ),
        ],
    )
    env.t = 6
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 6)

    out = c.get(f"/runs/{run_id}/sections/orders?as_of=5").get_json()

    assert out["status_counts"] == {"shipped": 1}
    assert [o["order_id"] for o in out["orders_timeline"]] == ["order-same-tick"]
    assert out["orders_timeline"][0]["current_status"] == "shipped"


def test_orders_section_returns_true_cumulative_status_series(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    env = app.registry._require(run_id)
    product = next(iter(env.products.values()))

    def make_order(order_id: str, log: list[OrderStatusRow]) -> Order:
        return Order(
            order_id=order_id,
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=log[0].t,
            promised_delivery_t=30,
            sale_price=120.0,
            purchase_price=80.0,
            current_status=log[-1].status,
            status_log=log,
        )

    dbm.insert_orders(
        app.registry.conn_for(run_id),
        run_id,
        [
            make_order(
                "order-1",
                [
                    OrderStatusRow(t=1, status="ordered"),
                    OrderStatusRow(t=2, status="shipped"),
                    OrderStatusRow(t=4, status="delivered"),
                    OrderStatusRow(t=5, status="settled_normal"),
                ],
            ),
            make_order(
                "order-2",
                [
                    OrderStatusRow(t=2, status="ordered"),
                    OrderStatusRow(t=3, status="shipped"),
                    OrderStatusRow(t=5, status="delivered"),
                    OrderStatusRow(t=6, status="settled_normal"),
                ],
            ),
            make_order(
                "order-3",
                [
                    OrderStatusRow(t=5, status="ordered"),
                ],
            ),
        ],
    )
    env.t = 6
    dbm.update_run_t(app.registry.conn_for(run_id), run_id, 6)

    out = c.get(f"/runs/{run_id}/sections/orders").get_json()

    assert out["status_cum"] == {
        "ordered": 3,
        "shipped": 2,
        "delivered": 2,
        "settled_normal": 2,
    }
    assert out["status_cum_series"]["status_ordered"] == [
        [1, 1],
        [2, 2],
        [5, 3],
    ]
    assert out["status_cum_series"]["status_shipped"] == [
        [2, 1],
        [3, 2],
    ]
    assert out["status_cum_series"]["status_delivered"] == [
        [4, 1],
        [5, 2],
    ]
    assert out["status_cum_series"]["status_settled_normal"] == [
        [5, 1],
        [6, 2],
    ]


def test_event_helpers_force_event_type_index_and_return_matching_rows(app_client):
    c, _, app = app_client
    run_id = c.post("/runs", json={"scenario": _tiny_scenario()}).get_json()["run_id"]
    dbm.write_events(
        app.registry.conn_for(run_id),
        run_id,
        [EventLog(t=i, event_type="price_change", entity_id=f"noise-{i}", agent_id="", payload={}) for i in range(20)]
        + [
            EventLog(t=21, event_type="order_settled_normal", entity_id="order-good", agent_id="agent_0", payload={}),
            EventLog(t=22, event_type="order_created", entity_id="order-created", agent_id="agent_0", payload={}),
        ],
    )

    traced: list[str] = []
    app.registry.conn_for(run_id).set_trace_callback(traced.append)
    try:
        rating_rows = dbm.load_rating_events(
            app.registry.conn_for(run_id),
            run_id,
            ["order_settled_normal"],
        )
        assert hasattr(dbm, "load_dashboard_merchant_action_events")
        action_rows = dbm.load_dashboard_merchant_action_events(
            app.registry.conn_for(run_id),
            run_id,
            "agent_0",
        )
    finally:
        app.registry.conn_for(run_id).set_trace_callback(None)

    assert rating_rows == [("agent_0", "order_settled_normal", 21)]
    assert [dict(r) for r in action_rows] == [
        {"t": 22, "event_type": "order_created", "entity_id": "order-created", "payload": "{}"}
    ]
    index_queries = [sql for sql in traced if "FROM events" in sql]
    assert index_queries
    assert all("INDEXED BY ix_events_run_type_t" in sql for sql in index_queries)


def test_dashboard_template_removes_recent_events_panel_and_fetches_orders_plainly():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "Recent events" not in html
    assert "EVENT_RENDER_LIMIT" not in html
    assert "recent_events" not in html
    assert "/agent/tool_calls" not in html
    assert "function sectionUrl" in html
    assert 'qs.set("as_of", replayT);' in html
    assert 'qs.set("t_to", replayT);' in html
    assert "fetch(sectionUrl(`/runs/${RUN_ID}/sections/orders`), fetchOpts)" in html


def test_dashboard_order_charts_use_cumulative_status_series():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    orders_section = html.split("<!-- Section 3: Downstream Order Flow -->", 1)[1].split("<!-- Section 4: Merchant", 1)[
        0
    ]
    order_js = html.split("function applyOrdersCharts(d)", 1)[1].split("async function refreshOrders", 1)[0]

    assert "Status counts (stacked)" not in orders_section
    assert 'id="ch-ord-status"' not in orders_section
    assert 'makeChart("ch-ord-status"' not in html
    assert 'charts["ch-ord-status"]' not in html
    assert "flow_series" not in order_js
    assert "status_series" not in order_js

    assert 'const orderGeneratedKeys = ["status_ordered"];' in order_js
    assert "const anomalyStatusOrder = [" in order_js
    assert '"status_settled_refund"' in order_js
    assert '"status_stockout"' in order_js
    assert 'applyCumulativeStatusChart("ch-ord-gen", orderGeneratedKeys, statusColors' in order_js
    assert 'applyCumulativeStatusChart("ch-ord-anomaly", anomalyStatusOrder, statusColors' in order_js


def test_dashboard_template_connects_sse_only_for_live_states():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'const LIVE_STATES = new Set(["running", "paused", "draining"]);' in html
    assert "function isLiveState(state)" in html
    assert "function ensureLiveSSE()" in html
    assert "if (isLiveState(currentState) && !es) connectSSE();" in html
    assert "function syncSSEForState()" in html
    assert "if (isLiveState(currentState))" in html
    assert "closeSSE();" in html
    assert "ensureLiveSSE();" in html
    assert "syncSSEForState();" in html
    assert "if (!isLiveState(currentState)) {" in html
    assert "if (isLiveState(s.state))" in html


def test_dashboard_sse_error_lets_eventsource_handle_reconnects():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    connect_js = html.split("function connectSSE()", 1)[1].split("  // ---------- boot ----------", 1)[0]
    error_handler = connect_js.split('es.addEventListener("error"', 1)[1]

    assert "setTimeout" not in error_handler
    assert "connectSSE();" not in error_handler


def test_dashboard_agent_trace_renderer_is_structured_and_untruncated():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    agent_js = html.split("<!-- ---- Agent tab logic", 1)[1]

    assert "function safeJsonParse" in agent_js
    assert "function renderJsonValue" in agent_js
    assert "function renderToolResult" in agent_js
    assert "function renderToolCallArguments" in agent_js
    assert "function renderMarkdownPreview" in agent_js
    assert "Raw JSON" in agent_js
    assert "Raw Markdown" in agent_js

    assert "raw.slice" not in agent_js
    assert "argShort" not in agent_js


def test_dashboard_trace_source_defaults_by_framework_and_is_selectable():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    agent_js = html.split("<!-- ---- Agent tab logic", 1)[1]

    assert '<option value="hermes">Hermes active</option>' in html
    assert '<option value="hermes_raw">Hermes raw</option>' in html
    assert '<option value="env">Env</option>' in html
    assert 'id="agent-trace-source"' in html
    assert "function traceSourceForMeta(meta)" in agent_js
    assert 'return hasHermes ? "hermes" : "env";' in agent_js
    assert "agentTraceSourceUserSelected = true;" in agent_js
    assert "agentTraceSourceUserSelected && agentTraceSource" in agent_js
    assert "configureAgentTraceSource(meta);" in agent_js
    refresh_agent = agent_js.split("async function refreshAgent(focusT)", 1)[1].split("// SSE real-time fan-in", 1)[0]
    assert refresh_agent.index("configureAgentTraceSource(meta);") < refresh_agent.index(
        'getJSON(withAgentTraceParams(`/runs/${RUN_ID}/agent/${agentTraceEndpoint("index")}`))'
    )


def test_dashboard_agent_trace_renders_compact_tool_tables_before_json_objects():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    agent_js = html.split("<!-- ---- Agent tab logic", 1)[1]

    assert "function isCompactTable(value)" in agent_js
    assert "function compactTableToRecords(table)" in agent_js
    assert "function renderCompactTable(table)" in agent_js

    render_json = agent_js.split("function renderJsonValue(value, depth = 0)", 1)[1].split(
        "function renderRawBlock", 1
    )[0]
    assert "if (isCompactTable(value)) return renderCompactTable(value);" in render_json
    assert render_json.index("if (isCompactTable(value))") < render_json.index("if (isPlainObject(value))")


def test_dashboard_agent_trace_view_filters_to_replay_cutoff():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")
    agent_js = html.split("<!-- ---- Agent tab logic", 1)[1]

    assert "let agentReplayT = null" in agent_js
    assert 'document.addEventListener("merchantbench-replay-cutoff"' in agent_js
    assert "step.t <= agentReplayT" in agent_js
    assert "parseInt(k, 10) <= agentReplayT" in agent_js
    assert "if (agentReplayT != null) return;" in agent_js
