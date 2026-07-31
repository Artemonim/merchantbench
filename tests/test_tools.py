"""Tests for agent-facing tools: list_product, adjust_price, delist_product.

All tool calls go through the unified POST /runs/<rid>/agents/<aid>/act endpoint.
The hook window must be open — the `hook_session` fixture spawns a step in a
background thread so tools work, then releases the hook on teardown via end_of_step."""
import json
import os
import tempfile
import threading
import time

import pytest

from core.entities import EventLog, Order, OrderStatusRow, StoreListing
from storage import db as dbm
from tools import tools as tool_impl
from web.app import create_app
from web.runner import load_default_scenario


_call_counter = 0


def _act(c, rid, agent_id, thought, tool_calls_spec):
    """Helper: send an /act request with the given tool calls."""
    global _call_counter
    tc_list = []
    for i, (name, args) in enumerate(tool_calls_spec):
        _call_counter += 1
        tc_list.append({"id": f"call_{_call_counter}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}})
    body = {"messages": [{"role": "assistant", "content": thought, "tool_calls": tc_list}]}
    return c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)


def _tool_result(resp, index=0):
    """Extract parsed content from tool_results[index]."""
    data = resp.get_json()
    assert data["ok"], f"act failed: {data}"
    content = data["tool_results"][index]["content"]
    return json.loads(content)


def _table_records(payload, key=None):
    """Decode the compact table payload used by multi-row agent tools."""
    table = payload[key] if key is not None else payload
    assert isinstance(table, dict)
    assert set(table) >= {"columns", "rows"}
    columns = table["columns"]
    return [dict(zip(columns, row)) for row in table["rows"]]


def _records(payload, key):
    value = payload[key]
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        return _table_records(value)
    return value


def _wait_for_hook(env, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if env.hook_open:
            return
        time.sleep(0.01)
    raise RuntimeError(f"Hook window did not open within {timeout:g} seconds")


@pytest.fixture
def hook_session():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0  # long enough for sequential tool calls
        scen["run"]["horizon_steps"] = 24
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        # Clear tool denylist so all tools are available in tests
        scen.setdefault("agent", {})["tool_denylist"] = []
        run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        th = threading.Thread(target=lambda: app.registry.step(run_id), daemon=True)
        th.start()
        env = app.registry._require(run_id)
        _wait_for_hook(env)
        try:
            resp = _act(c, run_id, "agent_0", "get categories", [("market_brief", {"window_days": 7})])
            cats = [row["category"] for row in _tool_result(resp)["categories"]]
            resp = _act(c, run_id, "agent_0", "search", [("search_products", {"query": "", "page": 1, "page_size": 1})])
            browsed = _records(_tool_result(resp), "items")
            yield c, run_id, browsed[0]
        finally:
            _act(c, run_id, "agent_0", "done", [("end_of_step", {})])
            th.join(timeout=3)


def test_list_supplier_products_pages_after_live_visibility_filter(
    hook_session, monkeypatch,
):
    c, run_id, _prod = hook_session
    env = c.application.registry._require(run_id)
    selected = sorted(env.products.values(), key=lambda p: p.product_id)[:3]
    supplier_id = "supplier-pagination-test"

    with env.lock:
        for product in selected:
            product.supplier_id = supplier_id
            product.is_listed_by_supplier = True
        placeholders = ",".join("?" for _ in selected)
        env.conn.execute(
            f"UPDATE products SET supplier_id=?, is_listed_by_supplier=1"
            f" WHERE run_id=? AND product_id IN ({placeholders})",
            (supplier_id, run_id, *(p.product_id for p in selected)),
        )
        selected[0].is_listed_by_supplier = False

        class ProductsWithoutFullScan(dict):
            def values(self):
                raise AssertionError("supplier pagination must not scan all products")

        original_products = env.products
        env.products = ProductsWithoutFullScan(original_products)
        query_windows = []
        original_query = dbm.list_supplier_products_sql

        def tracked_query(*args, **kwargs):
            query_windows.append((kwargs["limit"], kwargs["offset"]))
            return original_query(*args, **kwargs)

        monkeypatch.setattr(dbm, "list_supplier_products_sql", tracked_query)
        try:
            first_page = tool_impl.list_supplier_products(
                env, supplier_id, page=1, page_size=1,
            )
            second_page = tool_impl.list_supplier_products(
                env, supplier_id, page=2, page_size=1,
            )
        finally:
            env.products = original_products

    assert [row["product_id"] for row in _table_records(first_page, "items")] == [
        selected[1].product_id,
    ]
    assert first_page["has_next"] is True
    assert [row["product_id"] for row in _table_records(second_page, "items")] == [
        selected[2].product_id,
    ]
    assert second_page["has_next"] is False
    assert query_windows
    assert max(limit for limit, _offset in query_windows) <= len(selected)


def test_list_product_updates_listing(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    # First listing - should return ok without error
    resp = _act(c, run_id, "agent_0", "list it", [("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})])
    payload = _tool_result(resp)
    assert payload["ok"] is True
    items = _table_records(payload, "items")
    assert len(items) == 1
    assert items[0]["ok"] is True
    assert items[0]["error"] is None
    assert items[0]["supplier_ship_hours"] == prod["supplier_ship_hours"]
    original_listed_at = dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).listed_at

    # Second listing (duplicate) updates the price without reporting an error.
    env.t = 24
    new_price = round(float(prod["price"]) + 5.0, 2)
    resp2 = _act(c, run_id, "agent_0", "list it again", [("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": new_price}]})])
    payload2 = _tool_result(resp2)
    items2 = _table_records(payload2, "items")
    assert items2[0]["ok"] is True
    assert items2[0]["error"] is None
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).listed_at == original_listed_at

    resp = _act(c, run_id, "agent_0", "check listing", [("query_my_listings", {})])
    payload = _tool_result(resp)
    assert payload["columns"] == [
        "product_id", "name", "sale_price", "supplier_price",
        "supplier_ship_hours", "supplier_logistics_hours",
        "procured_orders", "cum_gross_profit",
        "cum_net_profit", "cum_fine",
        "listing_rating",
    ]
    listings = _table_records(payload)
    listing = next(row for row in listings if row["product_id"] == prod["product_id"])
    assert "agent_id" not in listing
    assert listing["supplier_ship_hours"] == prod["supplier_ship_hours"]
    assert listing["supplier_logistics_hours"] == prod["logistics_hours"]


@pytest.mark.parametrize("bad_price", [float("nan"), float("inf")])
def test_listing_mutations_reject_nonfinite_prices(hook_session, bad_price):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    listed = _tool_result(_act(c, run_id, "agent_0", "bad listing price", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": bad_price,
        }]})
    ]))

    assert listed["ok"] is False
    assert listed["error"]["path"] == "$.items[0].sale_price"
    assert "finite" in listed["error"]["message"]
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ) is None

    _act(c, run_id, "agent_0", "list valid", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    before = dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).sale_price
    adjusted = _tool_result(_act(c, run_id, "agent_0", "bad adjusted price", [
        ("adjust_price", {"items": [{
            "product_id": prod["product_id"],
            "new_price": bad_price,
        }]})
    ]))

    assert adjusted["ok"] is False
    assert adjusted["error"]["path"] == "$.items[0].new_price"
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).sale_price == before


@pytest.mark.parametrize("bad_price", [0.0, -1.0, 0.009, 5e-324])
def test_listing_mutations_reject_prices_below_currency_floor(
    hook_session,
    bad_price,
):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    listed = _tool_result(_act(c, run_id, "agent_0", "too-small listing price", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": bad_price,
        }]})
    ]))

    assert listed["ok"] is False
    listed_rows = _table_records(listed["items"])
    assert listed_rows[0]["product_id"] == prod["product_id"]
    assert "at least 0.01" in listed_rows[0]["error"]
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ) is None

    _act(c, run_id, "agent_0", "list valid", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    before = dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).sale_price
    adjusted = _tool_result(_act(c, run_id, "agent_0", "too-small adjusted price", [
        ("adjust_price", {"items": [{
            "product_id": prod["product_id"],
            "new_price": bad_price,
        }]})
    ]))

    assert adjusted["ok"] is False
    adjusted_rows = _table_records(adjusted["items"])
    assert adjusted_rows[0]["product_id"] == prod["product_id"]
    assert "at least 0.01" in adjusted_rows[0]["error"]
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    ).sale_price == before


def test_list_product_rejects_supplier_delisted_new_listing(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    product = env.products[prod["product_id"]]
    product.is_listed_by_supplier = False

    result = _tool_result(_act(c, run_id, "agent_0", "list unavailable", [
        ("list_product", {"items": [{
            "product_id": product.product_id,
            "sale_price": product.price,
        }]})
    ]))
    row = _table_records(result, "items")[0]

    assert result["ok"] is False
    assert row["ok"] is False
    assert "not currently available" in row["error"]
    assert dbm.get_listing(
        env.conn, run_id, "agent_0", product.product_id,
    ) is None


def test_query_my_listings_reports_cumulative_gross_and_net_profit(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    sale_price = round(float(prod["price"]) + 20.0, 2)
    purchase_price = round(float(prod["price"]), 2)
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": sale_price}]})
    ])
    order = Order(
        order_id="O_listing_profit",
        product_id=prod["product_id"],
        supplier_id=prod["supplier_id"],
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=3,
        sale_price=sale_price,
        purchase_price=purchase_price,
        current_status="settled_bad_review",
        purchase_t=0,
        shipped_t=1,
        delivered_t=2,
        settled_t=3,
        supplier_ship_hours=prod["supplier_ship_hours"],
        actual_ship_hours=prod["supplier_ship_hours"],
        actual_logistics_hours=prod["logistics_hours"],
        realized_revenue=sale_price,
        realized_cost=purchase_price,
        total_penalty=5.0,
        status_log=[
            OrderStatusRow(t=0, status="ordered"),
            OrderStatusRow(t=3, status="settled_bad_review"),
        ],
    )
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [order])

    resp = _act(c, run_id, "agent_0", "check listing", [("query_my_listings", {})])
    listings = _table_records(_tool_result(resp))
    listing = next(row for row in listings if row["product_id"] == prod["product_id"])
    assert listing["cum_gross_profit"] == pytest.approx(20.0)
    assert listing["cum_net_profit"] == pytest.approx(15.0)
    assert listing["cum_fine"] == pytest.approx(5.0)


def test_query_my_listings_unknown_agent_returns_error(hook_session):
    c, run_id, _ = hook_session
    resp = _act(c, run_id, "missing_agent", "check listing", [("query_my_listings", {})])

    assert resp.status_code == 404
    assert "unknown agent" in resp.get_json()["error"]


def test_listing_mutation_tools_accept_batch_table_payloads(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    products = list(env.products.values())[:3]

    list_resp = _act(c, run_id, "agent_0", "batch list", [
        ("list_product", {"items": [
            {
                "product_id": products[0].product_id,
                "sale_price": products[0].price,
            },
            {
                "product_id": products[1].product_id,
                "sale_price": products[1].price + 1.0,
            },
        ]})
    ])
    listed = _tool_result(list_resp)
    assert listed["ok"] is True
    assert _table_records(listed, "items") == [
        {
            "product_id": products[0].product_id,
            "ok": True,
            "error": None,
            "supplier_ship_hours": products[0].supplier_ship_hours,
        },
        {
            "product_id": products[1].product_id,
            "ok": True,
            "error": None,
            "supplier_ship_hours": products[1].supplier_ship_hours,
        },
    ]

    price_resp = _act(c, run_id, "agent_0", "batch price", [
        ("adjust_price", {"items": [
            {"product_id": products[0].product_id, "new_price": products[0].price + 2.0},
            {"product_id": products[2].product_id, "new_price": products[2].price + 2.0},
        ]})
    ])
    priced = _tool_result(price_resp)
    assert priced["ok"] is False
    assert _table_records(priced, "items") == [
        {"product_id": products[0].product_id, "ok": True, "error": None},
        {"product_id": products[2].product_id, "ok": False, "error": "not currently listed"},
    ]

    delist_resp = _act(c, run_id, "agent_0", "batch delist", [
        ("delist_product", {"items": [
            {"product_id": products[0].product_id},
            {"product_id": products[2].product_id},
        ]})
    ])
    delisted = _tool_result(delist_resp)
    assert delisted["ok"] is False
    assert _table_records(delisted, "items") == [
        {"product_id": products[0].product_id, "ok": True, "error": None},
        {"product_id": products[2].product_id, "ok": False, "error": "not currently listed"},
    ]


def test_listing_batch_items_reject_unknown_arguments_without_mutating(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    products = list(env.products.values())[:2]

    list_resp = _act(c, run_id, "agent_0", "batch legacy list", [
        ("list_product", {"items": [{
            "product_id": products[0].product_id,
            "sale_price": products[0].price,
            "promised_ship_hours": 24,
        }]})
    ])
    listed = _tool_result(list_resp)
    assert listed["ok"] is False
    assert listed["error"]["code"] == "invalid_arguments"
    assert listed["error"]["path"] == "$.items[0].promised_ship_hours"
    assert dbm.get_listing(env.conn, run_id, "agent_0", products[0].product_id) is None

    _act(c, run_id, "agent_0", "list valid", [
        ("list_product", {"items": [{
            "product_id": products[0].product_id,
            "sale_price": products[0].price,
        }]})
    ])

    price_resp = _act(c, run_id, "agent_0", "batch price typo", [
        ("adjust_price", {"items": [{
            "product_id": products[0].product_id,
            "new_price": products[0].price + 3.0,
            "sale_price": products[0].price + 3.0,
        }]})
    ])
    priced = _tool_result(price_resp)
    assert priced["ok"] is False
    assert priced["error"]["code"] == "invalid_arguments"
    assert priced["error"]["path"] == "$.items[0].sale_price"

    delist_resp = _act(c, run_id, "agent_0", "batch delist typo", [
        ("delist_product", {"items": [{
            "product_id": products[0].product_id,
            "new_price": products[0].price + 1.0,
        }]})
    ])
    delisted = _tool_result(delist_resp)
    assert delisted["ok"] is False
    assert delisted["error"]["code"] == "invalid_arguments"
    assert delisted["error"]["path"] == "$.items[0].new_price"
    assert dbm.get_listing(env.conn, run_id, "agent_0", products[0].product_id) is not None


def test_list_product_rejects_legacy_ship_promise_param_without_mutating(hook_session):
    c, run_id, prod = hook_session
    resp = _act(c, run_id, "agent_0", "legacy list", [
        ("list_product", {
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
            "promised_ship_hours": 24,
        })
    ])
    r = _tool_result(resp)
    assert r["ok"] is False
    assert r["error"]["code"] == "invalid_arguments"
    assert r["error"]["path"] == "$.items"

    resp = _act(c, run_id, "agent_0", "check no listing", [("query_my_listings", {})])
    listings = _table_records(_tool_result(resp))
    assert all(row["product_id"] != prod["product_id"] for row in listings)


def test_list_product_enforces_active_listing_limit_per_agent():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0
        scen["run"]["horizon_steps"] = 24
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        scen["platform_rules"]["max_active_listings"] = 2
        scen.setdefault("agent", {})["tool_denylist"] = []
        run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        env = app.registry._require(run_id)
        products = list(env.products.values())[:3]
        th = threading.Thread(target=lambda: app.registry.step(run_id), daemon=True)
        th.start()
        for _ in range(50):
            if env.hook_open:
                break
            time.sleep(0.1)
        try:
            resp = _act(c, run_id, "agent_0", "list first two", [
                ("list_product", {"items": [
                    {"product_id": products[0].product_id, "sale_price": products[0].price},
                    {"product_id": products[1].product_id, "sale_price": products[1].price},
                ]}),
            ])
            data = resp.get_json()
            assert data["ok"], data
            r = json.loads(data["tool_results"][0]["content"])
            items = _table_records(r, "items")
            assert [item["ok"] for item in items] == [True, True]

            resp = _act(c, run_id, "agent_0", "relist existing", [
                ("list_product", {"items": [
                    {"product_id": products[0].product_id, "sale_price": products[0].price + 1.0},
                ]})
            ])
            r = _tool_result(resp)
            items = _table_records(r, "items")
            assert items[0]["ok"]

            resp = _act(c, run_id, "agent_0", "list one too many", [
                ("list_product", {"items": [
                    {"product_id": products[2].product_id, "sale_price": products[2].price},
                ]})
            ])
            r = _tool_result(resp)
            items = _table_records(r, "items")
            assert items[0]["ok"] is False
            assert "max_active_listings" in items[0]["error"]
            assert "2" in items[0]["error"]

            resp = _act(c, run_id, "agent_0", "delist one", [
                ("delist_product", {"items": [{"product_id": products[1].product_id}]})
            ])
            r = _tool_result(resp)
            assert r["ok"] is True
            items = _table_records(r, "items")
            assert items[0]["ok"] is True

            resp = _act(c, run_id, "agent_0", "list after delist", [
                ("list_product", {"items": [
                    {"product_id": products[2].product_id, "sale_price": products[2].price},
                ]})
            ])
            r = _tool_result(resp)
            items = _table_records(r, "items")
            assert items[0]["ok"]
        finally:
            _act(c, run_id, "agent_0", "done", [("end_of_step", {})])
            th.join(timeout=3)


def test_adjust_price_success_returns_only_ok(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list first", [("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})])

    resp = _act(c, run_id, "agent_0", "adjust price", [
        ("adjust_price", {"items": [{"product_id": prod["product_id"], "new_price": prod["price"] + 1.0}]})
    ])

    payload = _tool_result(resp)
    assert payload["ok"] is True


def test_listing_tools_emit_agent_operation_events(hook_session):
    c, run_id, prod = hook_session
    app = c.application

    _act(c, run_id, "agent_0", "list first", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])
    _act(c, run_id, "agent_0", "adjust price", [
        ("adjust_price", {"items": [{"product_id": prod["product_id"], "new_price": prod["price"] + 1.0}]})
    ])
    _act(c, run_id, "agent_0", "delist", [
        ("delist_product", {"items": [{"product_id": prod["product_id"]}]})
    ])

    rows = app.registry.conn_for(run_id).execute(
        "SELECT event_type, entity_id, agent_id, payload FROM events"
        " WHERE run_id=? AND entity_id=? AND agent_id=? ORDER BY rowid",
        (run_id, prod["product_id"], "agent_0"),
    ).fetchall()
    assert [row["event_type"] for row in rows] == [
        "agent_list_product",
        "agent_adjust_price",
        "agent_delist_product",
    ]
    payloads = [json.loads(row["payload"] or "{}") for row in rows]
    assert payloads[0]["sale_price"] == prod["price"]
    assert payloads[1]["new_price"] == prod["price"] + 1.0


def test_order_query_results_omit_agent_id():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0
        scen["run"]["horizon_steps"] = 24
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        run_id = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        env = app.registry._require(run_id)
        product = next(iter(env.products.values()))
        order = Order(
            order_id="O_format",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=0,
            promised_delivery_t=3,
            sale_price=product.price + 1.0,
            purchase_price=product.price,
            current_status="ordered",
            supplier_ship_hours=7,
            status_log=[OrderStatusRow(t=0, status="ordered")],
        )
        dbm.insert_orders(env.conn, run_id, [order])

        th = threading.Thread(target=lambda: app.registry.step(run_id), daemon=True)
        th.start()
        _wait_for_hook(env)
        try:
            resp = _act(c, run_id, "agent_0", "orders", [("query_my_orders", {"page_size": 1})])
            payload = _tool_result(resp)
            assert payload["orders"]["columns"] == [
                "order_id", "product_id", "product_name",
                "supplier_id", "supplier_name", "order_time",
                "sale_price", "purchase_price", "current_status",
                "supplier_ship_hours",
                "supplier_logistics_hours", "actual_logistics_hours",
                "realized_revenue", "realized_cost", "total_penalty",
                "net_profit", "profit_finalized",
            ]
            orders = _table_records(payload["orders"])
            assert orders and orders[0]["order_id"] == "O_format"
            assert orders[0]["product_name"] == product.name
            assert orders[0]["supplier_name"] == product.supplier_name
            assert "agent_id" not in orders[0]
            assert "promised_delivery_time" not in orders[0]
            assert orders[0]["supplier_ship_hours"] == 7
            assert orders[0]["supplier_logistics_hours"] == product.logistics_hours
            assert "actual_ship_hours" not in orders[0]
            assert orders[0]["actual_logistics_hours"] is None
            assert orders[0]["net_profit"] == 0.0
            assert orders[0]["profit_finalized"] is False

            resp = _act(c, run_id, "agent_0", "order detail", [
                ("query_order_detail", {"order_id": "O_format"})
            ])
            detail = _tool_result(resp)
            assert detail["order_id"] == "O_format"
            assert detail["product_name"] == product.name
            assert detail["supplier_name"] == product.supplier_name
            assert "agent_id" not in detail
            assert "promised_delivery_time" not in detail
            assert detail["supplier_ship_hours"] == 7
            assert detail["supplier_logistics_hours"] == product.logistics_hours
            assert "actual_ship_hours" not in detail
            assert detail["actual_logistics_hours"] is None
            assert detail["net_profit"] == 0.0
            assert detail["profit_finalized"] is False
        finally:
            _act(c, run_id, "agent_0", "done", [("end_of_step", {})])
            th.join(timeout=3)


def test_query_my_orders_pages_same_tick_orders_by_order_id(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    supplier_id = "stable-pagination-test"
    orders = [
        Order(
            order_id=order_id,
            product_id=prod["product_id"],
            supplier_id=supplier_id,
            agent_id="agent_0",
            order_t=10,
            promised_delivery_t=20,
            sale_price=100.0,
            purchase_price=80.0,
            current_status="ordered",
        )
        for order_id in ("O_page_b", "O_page_a", "O_page_c")
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    page_ids = []
    for page in (1, 2, 3):
        result = tool_impl.query_my_orders(
            env,
            "agent_0",
            supplier_id=supplier_id,
            page=page,
            page_size=1,
        )
        assert result["total_count"] == 3
        page_ids.append(_table_records(result["orders"])[0]["order_id"])

    assert page_ids == ["O_page_c", "O_page_b", "O_page_a"]


def test_order_profit_fields_report_current_value_and_finalization(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    order = Order(
        order_id="O_final_profit",
        product_id=prod["product_id"],
        supplier_id=prod["supplier_id"],
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=3,
        sale_price=100.0,
        purchase_price=60.0,
        current_status="settled_bad_review",
        delivered_t=2,
        settled_t=3,
        realized_revenue=100.0,
        realized_cost=60.0,
        total_penalty=5.0,
        status_log=[
            OrderStatusRow(t=0, status="ordered"),
            OrderStatusRow(t=3, status="settled_bad_review"),
        ],
    )
    dbm.insert_orders(env.conn, run_id, [order])
    env.products[prod["product_id"]].is_listed_by_supplier = False

    history = _tool_result(_act(c, run_id, "agent_0", "history", [
        ("query_my_orders", {"product_id": prod["product_id"], "page_size": 20})
    ]))
    row = next(
        item for item in _table_records(history["orders"])
        if item["order_id"] == order.order_id
    )
    assert row["product_name"] == prod["name"]
    assert row["net_profit"] == 35.0
    assert row["profit_finalized"] is True

    detail = _tool_result(_act(c, run_id, "agent_0", "detail", [
        ("query_order_detail", {"order_id": order.order_id})
    ]))
    assert detail["product_name"] == prod["name"]
    assert detail["net_profit"] == 35.0
    assert detail["profit_finalized"] is True
    assert detail["expected_delivery_time"] == {
        "day": 1,
        "hour": 3,
        "datetime": "2025-06-01T03:00:00",
    }
    assert detail["delivered_time"] == {
        "day": 1,
        "hour": 2,
        "datetime": "2025-06-01T02:00:00",
    }


def test_failed_order_has_no_expected_or_actual_delivery_time(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    order = Order(
        order_id="O_no_delivery",
        product_id=prod["product_id"],
        supplier_id=prod["supplier_id"],
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=12,
        sale_price=100.0,
        purchase_price=60.0,
        current_status="stockout",
        purchase_t=0,
        settled_t=0,
        supplier_ship_hours=4,
        total_penalty=5.0,
        status_log=[OrderStatusRow(t=0, status="stockout")],
    )
    dbm.insert_orders(env.conn, run_id, [order])

    detail = _tool_result(_act(c, run_id, "agent_0", "detail", [
        ("query_order_detail", {"order_id": order.order_id})
    ]))

    assert detail["expected_delivery_time"] is None
    assert detail["delivered_time"] is None


def test_query_my_orders_rejects_removed_limit_argument(hook_session):
    c, run_id, _ = hook_session
    resp = _act(c, run_id, "agent_0", "invalid orders", [
        ("query_my_orders", {"limit": 1})
    ])
    payload = _tool_result(resp)

    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_arguments"
    assert payload["error"]["path"] == "$.limit"


def test_legacy_tools_are_removed(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list first", [("list_product", {"product_id": prod["product_id"], "sale_price": prod["price"]})])
    resp = _act(c, run_id, "agent_0", "legacy update", [
        ("set_promised_logistics_hours", {"product_id": prod["product_id"], "hours": 12})
    ])
    r = _tool_result(resp)
    assert r["ok"] is False
    assert "unknown tool" in r["error"]

    resp = _act(c, run_id, "agent_0", "legacy update 2", [
        ("set_promised_ship_hours", {"product_id": prod["product_id"], "hours": 24})
    ])
    r = _tool_result(resp)
    assert r["ok"] is False
    assert "unknown tool" in r["error"]


def test_query_platform_rules_returns_only_system_prompt(hook_session):
    from tools.observation import compose_system_brief

    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    resp = _act(c, run_id, "agent_0", "check rules", [("query_platform_rules", {})])
    r = _tool_result(resp)
    assert set(r) == {"system_prompt"}
    assert r["system_prompt"] == compose_system_brief(env)["system_prompt"]
    assert "MerchantBench" in r["system_prompt"]


def test_query_balance_rounds_money_fields_to_two_decimals(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    cash = env.agents["agent_0"].cash
    cash.balance = 123.456
    cash.deposit_pool = 987.654
    cash.in_transit = 1.005
    cash.receivable = 2.004
    cash.cumulative_fine = 3.999

    resp = _act(c, run_id, "agent_0", "check balance", [("query_balance", {})])

    assert _tool_result(resp) == {
        "balance": 123.46,
        "deposit_pool": 987.65,
        "in_transit": 1.0,
        "receivable": 2.0,
        "cumulative_fine": 4.0,
    }


def test_query_supply_chain_anomalies_new_empty_at_t0(hook_session):
    """t=0 -> no previous step, returns []."""
    c, run_id, _ = hook_session
    resp = _act(c, run_id, "agent_0", "check events",
                [("query_supply_chain_anomalies", {"mode": "new"})])
    r = _tool_result(resp)
    assert r["mode"] == "new"
    assert r["events"] == []
    assert _table_records(r["listings"]) == []


def test_get_store_snapshot_reflects_same_hook_listing_mutation(hook_session):
    c, run_id, prod = hook_session
    obs = c.get(f"/runs/{run_id}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    before_resp = _act(c, run_id, "agent_0", "snapshot before", [("get_store_snapshot", {})])
    before = _tool_result(before_resp)
    before_active = before["supply"]["listings"]["active"]

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"] * 1.4}]})
    ])
    after_resp = _act(c, run_id, "agent_0", "snapshot after", [("get_store_snapshot", {})])
    after = _tool_result(after_resp)

    assert after["supply"]["listings"]["active"] == before_active + 1


def test_get_store_snapshot_order_totals_are_current_status_counts(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    statuses = [
        "ordered", "late", "shipped", "delivered", "cancelled",
        "settled_normal", "settled_refund", "settled_only_refund",
        "settled_bad_review", "stockout", "insufficient_balance",
    ]
    orders = [
        Order(
            order_id=f"snapshot-status-{i}",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=i,
            promised_delivery_t=i + 24,
            sale_price=100.0,
            purchase_price=80.0,
            current_status=status,
            settled_t=i if status.startswith("settled_") else None,
            late_t=i if status == "late" else None,
        )
        for i, status in enumerate(statuses)
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "snapshot", [("get_store_snapshot", {})])
    snapshot = _tool_result(resp)

    assert snapshot["orders"]["totals"] == {
        "total": len(statuses),
        **{status: 1 for status in statuses},
    }
    assert set(snapshot["orders"]["changes_since_last_observation"]) == set(
        snapshot["orders"]["totals"]
    )


def test_query_my_orders_status_filter_matches_exact_current_status(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    orders = [
        Order(
            order_id="order-settled-refund",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=1,
            promised_delivery_t=25,
            sale_price=100.0,
            purchase_price=80.0,
            current_status="settled_refund",
            settled_t=10,
        ),
        Order(
            order_id="order-settled-only-refund",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=2,
            promised_delivery_t=26,
            sale_price=100.0,
            purchase_price=80.0,
            current_status="settled_only_refund",
            settled_t=11,
        ),
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "orders", [
        ("query_my_orders", {"status": "settled_refund", "page_size": 20})
    ])
    filtered_payload = _tool_result(resp)
    filtered = _table_records(filtered_payload["orders"])
    resp = _act(c, run_id, "agent_0", "orders", [
        ("query_my_orders", {"status": "refund", "page_size": 20})
    ])
    aggregate_name = _tool_result(resp)

    assert [row["order_id"] for row in filtered] == ["order-settled-refund"]
    assert aggregate_name["ok"] is False
    assert aggregate_name["error"]["code"] == "invalid_arguments"
    assert aggregate_name["error"]["path"] == "$.status"


def test_query_open_orders_returns_compact_active_order_rows(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    product = env.products[prod["product_id"]]
    product.supplier_ship_hours = 4
    product.logistics_hours = 10
    env.t = 12
    orders = [
        Order(
            order_id="open-ordered",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=2,
            promised_delivery_t=20,
            sale_price=100.0,
            purchase_price=70.0,
            current_status="ordered",
            purchase_t=2,
            supplier_ship_hours=0,
            status_log=[OrderStatusRow(t=2, status="ordered")],
        ),
        Order(
            order_id="open-late",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=4,
            promised_delivery_t=18,
            sale_price=60.0,
            purchase_price=40.0,
            current_status="late",
            purchase_t=4,
            late_t=10,
            total_penalty=5.0,
            status_log=[
                OrderStatusRow(t=4, status="ordered"),
                OrderStatusRow(t=10, status="late"),
            ],
        ),
        Order(
            order_id="closed-settled",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=5,
            promised_delivery_t=10,
            sale_price=50.0,
            purchase_price=30.0,
            current_status="settled_normal",
            settled_t=10,
            status_log=[
                OrderStatusRow(t=5, status="ordered"),
                OrderStatusRow(t=10, status="settled_normal"),
            ],
        ),
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "open orders", [("query_open_orders", {})])
    result = _tool_result(resp)

    assert result["total_count"] == 2
    assert result["page"] == 1
    assert result["page_size"] == 20
    assert result["has_next"] is False
    assert "truncated" not in result
    assert result["by_status"] == {
        "ordered": 1,
        "late": 1,
        "shipped": 0,
        "delivered": 0,
    }
    assert set(result["orders"]) >= {"columns", "rows"}
    rows = {row["order_id"]: row for row in _table_records(result["orders"])}
    assert set(rows) == {"open-ordered", "open-late"}
    assert rows["open-late"] == {
        "order_id": "open-late",
        "product_id": product.product_id,
        "product_name": product.name,
        "supplier_id": product.supplier_id,
        "supplier_name": product.supplier_name,
        "current_status": "late",
        "order_time": {
            "day": 1,
            "hour": 4,
            "datetime": "2025-06-01T04:00:00",
        },
        "status_age_hours": 2,
        "expected_delivery_time": {
            "day": 1,
            "hour": 18,
            "datetime": "2025-06-01T18:00:00",
        },
        "delivered_time": None,
        "sale_price": 60.0,
        "purchase_price": 40.0,
        "total_penalty": 5.0,
        "net_profit": -5.0,
        "profit_finalized": False,
    }
    assert "margin" not in rows["open-late"]
    assert "supplier_ship_h" not in rows["open-late"]
    assert "promised_ship_h" not in rows["open-late"]

    first_page = tool_impl.query_open_orders(
        env, "agent_0", page=1, page_size=1,
    )
    second_page = tool_impl.query_open_orders(
        env, "agent_0", page=2, page_size=1,
    )
    assert first_page["has_next"] is True
    assert second_page["has_next"] is False
    assert {
        _table_records(first_page["orders"])[0]["order_id"],
        _table_records(second_page["orders"])[0]["order_id"],
    } == {"open-ordered", "open-late"}


def test_query_order_updates_coalesces_since_last_observation(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    product = env.products[prod["product_id"]]
    env.t = 12
    env.last_observation_step_by_agent = {"agent_0": 5}
    orders = [
        Order(
            order_id="changed-to-shipped",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=2,
            promised_delivery_t=20,
            sale_price=100.0,
            purchase_price=70.0,
            current_status="shipped",
            purchase_t=2,
            shipped_t=8,
            actual_logistics_hours=12,
            status_log=[
                OrderStatusRow(t=2, status="ordered"),
                OrderStatusRow(t=6, status="late"),
                OrderStatusRow(t=8, status="shipped"),
            ],
        ),
        Order(
            order_id="new-order",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=11,
            promised_delivery_t=16,
            sale_price=50.0,
            purchase_price=30.0,
            current_status="ordered",
            purchase_t=11,
            status_log=[OrderStatusRow(t=11, status="ordered")],
        ),
        Order(
            order_id="unchanged-delivered",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=1,
            promised_delivery_t=4,
            sale_price=40.0,
            purchase_price=20.0,
            current_status="delivered",
            purchase_t=1,
            delivered_t=4,
            status_log=[
                OrderStatusRow(t=1, status="ordered"),
                OrderStatusRow(t=4, status="delivered"),
            ],
        ),
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "updates", [("query_order_updates", {})])
    result = _tool_result(resp)

    assert result["window"] == {"from": "D1H6", "to": "D1H12"}
    assert result["total_count"] == 2
    assert result["page"] == 1
    assert result["page_size"] == 50
    assert result["has_next"] is False
    assert "truncated" not in result
    assert result["by_current"] == {
        "ordered": 1,
        "late": 0,
        "shipped": 1,
        "delivered": 0,
        "cancelled": 0,
        "settled_normal": 0,
        "settled_refund": 0,
        "settled_only_refund": 0,
        "settled_bad_review": 0,
        "stockout": 0,
        "insufficient_balance": 0,
    }
    assert set(result["orders"]) >= {"columns", "rows"}
    rows = {row["order_id"]: row for row in _table_records(result["orders"])}
    assert rows["changed-to-shipped"] == {
        "order_id": "changed-to-shipped",
        "product_id": prod["product_id"],
        "product_name": product.name,
        "supplier_id": product.supplier_id,
        "supplier_name": product.supplier_name,
        "previous_status": "ordered",
        "current_status": "shipped",
        "order_time": {
            "day": 1,
            "hour": 2,
            "datetime": "2025-06-01T02:00:00",
        },
        "status_age_hours": 4,
        "expected_delivery_time": {
            "day": 1,
            "hour": 20,
            "datetime": "2025-06-01T20:00:00",
        },
        "delivered_time": None,
        "sale_price": 100.0,
        "purchase_price": 70.0,
        "total_penalty": 0.0,
        "net_profit": 0.0,
        "profit_finalized": False,
    }
    assert rows["new-order"]["previous_status"] is None
    assert rows["new-order"]["current_status"] == "ordered"
    assert "at_step" not in rows["changed-to-shipped"]
    assert "margin" not in rows["changed-to-shipped"]

    first_page = tool_impl.query_order_updates(
        env, "agent_0", page=1, page_size=1,
    )
    second_page = tool_impl.query_order_updates(
        env, "agent_0", page=2, page_size=1,
    )
    assert first_page["has_next"] is True
    assert second_page["has_next"] is False
    assert {
        _table_records(first_page["orders"])[0]["order_id"],
        _table_records(second_page["orders"])[0]["order_id"],
    } == {"changed-to-shipped", "new-order"}


def test_query_supply_chain_anomalies_returns_safe_event_fields():
    """Two products listed; events of all 4 abnormal types + one non-abnormal
    (`order_created`) + one on a product I don't own. Tool must return only my
    products AND only the 4 abnormal event types."""
    import os, tempfile
    from core.entities import EventLog
    from storage import db as dbm
    from web.app import create_app
    from web.runner import load_default_scenario
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0
        scen["run"]["horizon_steps"] = 24
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        scen.setdefault("agent", {})["tool_denylist"] = []
        rid = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        env = app.registry._require(rid)
        th = threading.Thread(target=lambda: app.registry.step(rid), daemon=True)
        th.start()
        for _ in range(50):
            if env.hook_open:
                break
            time.sleep(0.1)
        resp = _act(c, rid, "agent_0", "get categories", [("market_brief", {"window_days": 7})])
        cats = [row["category"] for row in _tool_result(resp)["categories"]]
        resp = _act(c, rid, "agent_0", "search", [("search_products", {"query": "", "page": 1, "page_size": 3})])
        browsed = _records(_tool_result(resp), "items")
        mine_a = browsed[0]["product_id"]
        mine_b = browsed[1]["product_id"]
        other = browsed[2]["product_id"]
        # list two products
        _act(c, rid, "agent_0", "list a", [("list_product", {"items": [{"product_id": mine_a, "sale_price": browsed[0]["price"] * 1.4}]})])
        _act(c, rid, "agent_0", "list b", [("list_product", {"items": [{"product_id": mine_b, "sale_price": browsed[1]["price"] * 1.4}]})])
        # end step
        _act(c, rid, "agent_0", "done", [("end_of_step", {})])
        th.join(timeout=3)
        # advance the env clock so env.t > 0 and events sit at t-1.
        # Must be a multiple of activation_period (12) so the hook opens.
        # query_supply_chain_anomalies(mode="new") reuses the current
        # observation window, so write at t=11 for the t=12 hook.
        env.t = 12
        dbm.write_events(env.conn, env.run_id, [
            EventLog(t=11, event_type="price_change", entity_id=mine_a,
                     agent_id=None, payload={
                         "new_price": 14.5,
                         "ref_price": 12.0,
                         "base_price": 10.0,
                         "factor": 1.45,
                         "recover_t": 36,
                     }),
            EventLog(t=11, event_type="supplier_delist", entity_id=mine_b,
                     agent_id=None, payload={"recover_t": 36}),
            EventLog(t=11, event_type="order_stockout_violation",
                     entity_id=mine_a, agent_id="agent_0", payload={}),
            # agent-specific abnormal event on a product I also list: should NOT
            # leak another agent's order details.
            EventLog(t=11, event_type="order_stockout_violation",
                     entity_id=mine_a, agent_id="agent_1",
                     payload={"order_id": "other-agent-order", "penalty": 5.0}),
            EventLog(t=11, event_type="supplier_timeout", entity_id=mine_b,
                     agent_id=None, payload={
                         "recover_t": 12,
                         "before_supplier_ship_hours": 12,
                         "after_supplier_ship_hours": 60,
                     }),
            # non-abnormal: should NOT appear
            EventLog(t=11, event_type="order_created", entity_id=mine_a,
                     agent_id="agent_0", payload={}),
            # abnormal type but on a product I don't own: should NOT appear
            EventLog(t=11, event_type="price_change", entity_id=other,
                     agent_id=None, payload={"new_price": 9.9}),
        ])
        # Re-open a hook so the tool call is allowed.
        th2 = threading.Thread(target=lambda: app.registry.step(rid),
                                daemon=True)
        th2.start()
        # Wait for hook to open via observation long-poll
        obs_resp = c.get(f"/runs/{rid}/agents/agent_0/observation?timeout=3")
        assert obs_resp.status_code == 200
        try:
            resp = _act(c, rid, "agent_0", "check events",
                        [("query_supply_chain_anomalies", {"mode": "new"})])
            result = _tool_result(resp)
        finally:
            _act(c, rid, "agent_0", "done", [("end_of_step", {})])
            th2.join(timeout=3)
    events = result["events"]
    types = sorted(e["event_type"] for e in events)
    pids = sorted({e["product_id"] for e in events})
    assert types == ["order_stockout_violation", "price_change",
                     "supplier_delist", "supplier_timeout"], types
    assert pids == sorted([mine_a, mine_b]), pids
    assert all(other != e["product_id"] for e in events)
    by_type = {e["event_type"]: e for e in events}
    assert by_type["price_change"]["before"] == {"supplier_price": 10.0}
    assert by_type["price_change"]["after"] == {"supplier_price": 14.5}
    assert by_type["supplier_delist"]["before"] == {"supplier_listed": True}
    assert by_type["supplier_delist"]["after"] == {"supplier_listed": False}
    assert by_type["supplier_timeout"]["before"] == {"supplier_ship_hours": 12}
    assert by_type["supplier_timeout"]["after"] == {"supplier_ship_hours": 60}
    leaked = {"ref_price", "base_price", "factor", "recover_t", "recover_time", "risk"}
    for event in events:
        assert not leaked.intersection(event), event
        assert not leaked.intersection(event.get("before", {})), event
        assert not leaked.intersection(event.get("after", {})), event
    assert set(result["listings"]) >= {"columns", "rows"}
    assert {row["product_id"] for row in _table_records(result["listings"])} == {mine_a, mine_b}


def test_query_supply_chain_anomalies_now_reports_current_abnormal_listings(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    product = next(iter(env.products.values()))
    product.is_listed_by_supplier = False
    product.supplier_ship_hours = int(product.base_ship_hours) + 24
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price - 1.0,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing

    resp = _act(c, run_id, "agent_0", "check events",
                [("query_supply_chain_anomalies", {"mode": "now"})])
    result = _tool_result(resp)

    assert result["mode"] == "now"
    assert result["events"] == []
    assert set(result["listings"]) >= {"columns", "rows"}
    listings = _table_records(result["listings"])
    assert len(listings) == 1
    row = listings[0]
    assert row["product_id"] == product.product_id
    assert row["supplier_listed"] is False
    assert "timeout_active" not in row
    assert row["supplier_ship_hours"] == product.supplier_ship_hours
    assert row["supplier_price"] == round(product.price, 2)


def test_query_store_performance_returns_columnar_cumulative_buckets(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    env.t = 47
    dbm.write_metrics(env.conn, env.run_id, "agent_0", 23, {
        "cum_gmv": 100.0,
        "cum_cost": 60.0,
        "cum_gross_profit": 40.0,
        "cum_net_profit": 25.0,
        "cum_fine": 5.0,
        "net_assets": 3025.0,
    })
    dbm.write_metrics(env.conn, env.run_id, "agent_0", 47, {
        "cum_gmv": 180.0,
        "cum_cost": 105.0,
        "cum_gross_profit": 75.0,
        "cum_net_profit": 50.0,
        "cum_fine": 8.0,
        "net_assets": 3050.0,
    })

    resp = _act(c, run_id, "agent_0", "performance",
                [("query_store_performance", {
                    "day_from": 1,
                    "day_to": 2,
                    "level": "day",
                })])
    result = _tool_result(resp)

    assert result["bucket_label"] == ["D1", "D2"]
    assert result["bucket_day_from"] == [1, 2]
    assert result["bucket_day_to"] == [1, 2]
    assert result["cum_gmv"] == [100.0, 180.0]
    assert result["cum_cost"] == [60.0, 105.0]
    assert result["cum_gross_profit"] == [40.0, 75.0]
    assert result["cum_net_profit"] == [25.0, 50.0]
    assert result["cum_fine"] == [5.0, 8.0]
    assert result["net_assets"] == [3025.0, 3050.0]


def test_query_store_performance_week_buckets_include_partial_week(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    env.t = 215
    dbm.write_metrics(env.conn, env.run_id, "agent_0", 167, {
        "cum_gmv": 700.0,
        "cum_cost": 420.0,
        "cum_gross_profit": 280.0,
        "cum_net_profit": 210.0,
        "cum_fine": 12.0,
        "net_assets": 3210.0,
    })
    dbm.write_metrics(env.conn, env.run_id, "agent_0", 215, {
        "cum_gmv": 900.0,
        "cum_cost": 540.0,
        "cum_gross_profit": 360.0,
        "cum_net_profit": 270.0,
        "cum_fine": 15.0,
        "net_assets": 3270.0,
    })

    resp = _act(c, run_id, "agent_0", "performance",
                [("query_store_performance", {
                    "day_from": 1,
                    "day_to": 9,
                    "level": "week",
                })])
    result = _tool_result(resp)

    assert result["bucket_label"] == ["W1", "W2"]
    assert result["bucket_day_from"] == [1, 8]
    assert result["bucket_day_to"] == [7, 9]
    assert result["cum_gmv"] == [700.0, 900.0]
    assert result["net_assets"] == [3210.0, 3270.0]


@pytest.mark.parametrize("tool_name,args", [
    ("query_my_orders", {"day_to": 3}),
    ("query_store_performance", {
        "day_from": 1,
        "day_to": 3,
        "level": "day",
    }),
    ("query_product_sales_stats", {
        "day_from": 1,
        "day_to": 3,
        "sort_by": "orders",
    }),
])
def test_day_range_queries_reject_future_days(hook_session, tool_name, args):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    env.t = 24  # day 2

    result = _tool_result(_act(
        c,
        run_id,
        "agent_0",
        "future range",
        [(tool_name, args)],
    ))

    assert result["ok"] is False
    assert result["current_day"] == 2
    assert "cannot exceed current day" in result["error"]


def test_query_product_sales_stats_sorts_interval_and_defaults_limit(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    env.t = 47
    products = list(env.products.values())[:2]
    listing = StoreListing(
        product_id=products[0].product_id,
        agent_id="agent_0",
        sale_price=130.0,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, run_id, "agent_0", listing)
    env.agents["agent_0"].listings[products[0].product_id] = listing
    orders = [
        Order(
            order_id="product-high",
            product_id=products[0].product_id,
            supplier_id=products[0].supplier_id,
            agent_id="agent_0",
            order_t=25,
            promised_delivery_t=30,
            sale_price=120.0,
            purchase_price=70.0,
            current_status="settled_normal",
            purchase_t=25,
            shipped_t=26,
            delivered_t=28,
            settled_t=35,
            realized_revenue=120.0,
            realized_cost=70.0,
            total_penalty=5.0,
        ),
        Order(
            order_id="product-low",
            product_id=products[1].product_id,
            supplier_id=products[1].supplier_id,
            agent_id="agent_0",
            order_t=26,
            promised_delivery_t=31,
            sale_price=90.0,
            purchase_price=75.0,
            current_status="settled_refund",
            purchase_t=26,
            shipped_t=27,
            delivered_t=29,
            settled_t=36,
            realized_revenue=0.0,
            realized_cost=75.0,
            total_penalty=10.0,
        ),
    ]
    for order in orders:
        order.status_log.append(OrderStatusRow(t=order.order_t, status=order.current_status))
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "product stats",
                [("query_product_sales_stats", {
                    "day_from": 2,
                    "day_to": 2,
                    "sort_by": "net_profit",
                })])
    result = _tool_result(resp)

    assert result["limit"] == 10
    assert set(result["items"]) >= {"columns", "rows"}
    items = _table_records(result["items"])
    assert [item["product_id"] for item in items] == [
        products[0].product_id,
        products[1].product_id,
    ]
    assert items[0]["orders"] == 1
    assert items[0]["category"] == products[0].category
    assert items[0]["supplier_id"] == products[0].supplier_id
    assert items[0]["current_sale_price"] == 130.0
    assert items[0]["current_supplier_price"] == pytest.approx(products[0].price)
    assert items[0]["current_gross_margin_rate"] == pytest.approx(
        (130.0 - products[0].price) / 130.0,
        abs=0.0001,
    )
    assert items[0]["gmv"] == 120.0
    assert items[0]["gross_profit"] == 50.0
    assert items[0]["net_profit"] == 45.0
    assert items[1]["category"] == products[1].category
    assert items[1]["supplier_id"] == products[1].supplier_id
    assert items[1]["current_sale_price"] is None
    assert items[1]["current_supplier_price"] == pytest.approx(products[1].price)
    assert items[1]["current_gross_margin_rate"] is None
    assert items[1]["refund_count"] == 1
    assert not {
        "late_rate", "stockout_rate", "refund_rate", "bad_review_rate"
    } & set(result["items"]["columns"])


def test_query_product_sales_stats_attributes_realized_profit_to_settlement_day(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    env.t = 8 * 24
    product = next(iter(env.products.values()))
    order = Order(
        order_id="product-settles-later",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=24,
        sale_price=120.0,
        purchase_price=70.0,
        current_status="settled_bad_review",
        purchase_t=0,
        shipped_t=1,
        delivered_t=24,
        settled_t=8 * 24,
        realized_revenue=120.0,
        realized_cost=70.0,
        total_penalty=5.0,
    )
    order.status_log.append(OrderStatusRow(t=0, status="ordered"))
    order.status_log.append(OrderStatusRow(t=8 * 24, status="settled_bad_review"))
    dbm.insert_orders(env.conn, run_id, [order])
    dbm.write_events(env.conn, run_id, [EventLog(
        t=8 * 24,
        event_type="order_settled_bad_review",
        entity_id=order.order_id,
        agent_id="agent_0",
        payload={"penalty": 5.0},
    )])

    day1 = _tool_result(_act(c, run_id, "agent_0", "product stats", [
        ("query_product_sales_stats", {
            "day_from": 1,
            "day_to": 1,
            "sort_by": "orders",
        })
    ]))
    day9 = _tool_result(_act(c, run_id, "agent_0", "product stats", [
        ("query_product_sales_stats", {
            "day_from": 9,
            "day_to": 9,
            "sort_by": "net_profit",
        })
    ]))

    day1_items = _table_records(day1["items"])
    day9_items = _table_records(day9["items"])
    assert day1_items[0]["orders"] == 1
    assert day1_items[0]["gmv"] == 120.0
    assert day1_items[0]["net_profit"] == 0.0
    assert day1_items[0]["fine"] == 0.0
    assert day9_items[0]["orders"] == 0
    assert day9_items[0]["gmv"] == 0.0
    assert day9_items[0]["net_profit"] == 45.0
    assert day9_items[0]["fine"] == 5.0
    assert day9_items[0]["bad_review_count"] == 1
    assert not {
        "late_rate", "stockout_rate", "refund_rate", "bad_review_rate"
    } & set(day9["items"]["columns"])


def test_query_product_sales_stats_counts_failed_orders_without_gmv(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    product = next(iter(env.products.values()))
    dbm.insert_orders(env.conn, run_id, [
        Order(
            order_id="product-stockout",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=0,
            promised_delivery_t=24,
            sale_price=120.0,
            purchase_price=70.0,
            current_status="stockout",
            settled_t=0,
            total_penalty=5.0,
        ),
        Order(
            order_id="product-insufficient",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=1,
            promised_delivery_t=24,
            sale_price=90.0,
            purchase_price=60.0,
            current_status="insufficient_balance",
            settled_t=1,
            total_penalty=5.0,
        ),
    ])
    dbm.write_events(env.conn, run_id, [
        EventLog(
            t=0,
            event_type="order_stockout_violation",
            entity_id=product.product_id,
            agent_id="agent_0",
            payload={
                "order_id": "product-stockout",
                "product_id": product.product_id,
                "penalty": 5.0,
            },
        ),
        EventLog(
            t=1,
            event_type="order_insufficient_balance_violation",
            entity_id=product.product_id,
            agent_id="agent_0",
            payload={
                "order_id": "product-insufficient",
                "product_id": product.product_id,
                "penalty": 5.0,
            },
        ),
    ])

    result = _tool_result(_act(c, run_id, "agent_0", "product stats", [
        ("query_product_sales_stats", {
            "day_from": 1,
            "day_to": 1,
            "sort_by": "orders",
        })
    ]))
    item = _table_records(result["items"])[0]

    assert item["orders"] == 2
    assert item["gmv"] == 0.0
    assert item["gross_profit"] == 0.0
    assert item["net_profit"] == -10.0
    assert item["fine"] == 10.0
    assert item["stockout_count"] == 1
    assert item["insufficient_balance_count"] == 1


def test_query_product_sales_stats_attributes_late_fine_to_event_day(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    product = next(iter(env.products.values()))
    env.t = 10 * 24
    order = Order(
        order_id="product-late-open",
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id="agent_0",
        order_t=3 * 24,
        promised_delivery_t=4 * 24,
        sale_price=120.0,
        purchase_price=70.0,
        current_status="late",
        purchase_t=3 * 24,
        late_t=4 * 24,
        realized_cost=70.0,
        total_penalty=3.0,
        status_log=[
            OrderStatusRow(t=3 * 24, status="ordered"),
            OrderStatusRow(t=4 * 24, status="late"),
        ],
    )
    dbm.insert_orders(env.conn, run_id, [order])
    dbm.write_events(env.conn, run_id, [
        EventLog(
            t=4 * 24,
            event_type="order_late",
            entity_id=order.order_id,
            agent_id="agent_0",
            payload={"penalty": 3.0},
        ),
        EventLog(
            t=4 * 24,
            event_type="order_late",
            entity_id=order.order_id,
            agent_id="agent_1",
            payload={"penalty": 99.0},
        ),
    ])

    day5 = _tool_result(_act(c, run_id, "agent_0", "day 5 stats", [
        ("query_product_sales_stats", {
            "day_from": 5,
            "day_to": 5,
            "sort_by": "fine",
        })
    ]))
    day6 = _tool_result(_act(c, run_id, "agent_0", "day 6 stats", [
        ("query_product_sales_stats", {
            "day_from": 6,
            "day_to": 6,
            "sort_by": "fine",
        })
    ]))

    day5_item = _table_records(day5["items"])[0]
    assert day5_item["fine"] == 3.0
    assert day5_item["late_count"] == 1
    assert day5_item["net_profit"] == 0.0
    assert _table_records(day6["items"]) == []


def test_query_cash_pipeline_reports_receivable_aging_and_open_orders(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    products = list(env.products.values())[:6]
    env.t = 200
    env.scenario["settlement"]["normal_delay_hours"] = 240
    cash = env.agents["agent_0"].cash
    cash.balance = 1800.0
    cash.deposit_pool = 980.0
    cash.in_transit = 420.0
    cash.receivable = 760.0
    cash.cumulative_fine = 30.0
    orders = [
        Order(
            order_id="cash-due",
            product_id=products[0].product_id,
            supplier_id=products[0].supplier_id,
            agent_id="agent_0",
            order_t=20,
            promised_delivery_t=30,
            sale_price=150.0,
            purchase_price=80.0,
            current_status="delivered",
            purchase_t=20,
            shipped_t=25,
            delivered_t=100,
        ),
        Order(
            order_id="cash-later",
            product_id=products[1].product_id,
            supplier_id=products[1].supplier_id,
            agent_id="agent_0",
            order_t=100,
            promised_delivery_t=120,
            sale_price=200.0,
            purchase_price=90.0,
            current_status="delivered",
            purchase_t=100,
            shipped_t=110,
            delivered_t=190,
        ),
        Order(
            order_id="cash-shipped",
            product_id=products[2].product_id,
            supplier_id=products[2].supplier_id,
            agent_id="agent_0",
            order_t=160,
            promised_delivery_t=210,
            sale_price=120.0,
            purchase_price=70.0,
            current_status="shipped",
            purchase_t=160,
            shipped_t=180,
        ),
        Order(
            order_id="cash-ordered",
            product_id=products[3].product_id,
            supplier_id=products[3].supplier_id,
            agent_id="agent_0",
            order_t=180,
            promised_delivery_t=230,
            sale_price=60.0,
            purchase_price=30.0,
            current_status="ordered",
            purchase_t=180,
        ),
        Order(
            order_id="cash-late",
            product_id=products[4].product_id,
            supplier_id=products[4].supplier_id,
            agent_id="agent_0",
            order_t=150,
            promised_delivery_t=190,
            sale_price=90.0,
            purchase_price=40.0,
            current_status="late",
            purchase_t=150,
            late_t=199,
        ),
        Order(
            order_id="cash-settled",
            product_id=products[5].product_id,
            supplier_id=products[5].supplier_id,
            agent_id="agent_0",
            order_t=10,
            promised_delivery_t=20,
            sale_price=500.0,
            purchase_price=300.0,
            current_status="settled_normal",
            purchase_t=10,
            shipped_t=11,
            delivered_t=12,
            settled_t=190,
            realized_revenue=500.0,
            realized_cost=300.0,
        ),
    ]
    dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "cash",
                [("query_cash_pipeline", {"window_days": 7})])
    result = _tool_result(resp)

    assert result == {
        "window_days": 7,
        "cash_now": {
            "balance": 1800.0,
            "deposit_pool": 980.0,
            "in_transit": 420.0,
            "receivable": 760.0,
            "net_assets": 3960.0,
        },
        "receivable_aging": {
            "total": {
                "amount": 350.0,
                "order_count": 2,
            },
            "delivered_within_window": {
                "amount": 350.0,
                "order_count": 2,
            },
            "delivered_before_window": {
                "amount": 0.0,
                "order_count": 0,
            },
        },
        "settlement_policy": {
            "max_resolution_hours_after_delivery": 240,
            "exact_timing_known": False,
            "full_sale_proceeds_guaranteed": False,
        },
        "open_orders": {
            "count": 5,
            "purchase_cost": 310.0,
            "by_status": {
                "ordered": 1,
                "late": 1,
                "shipped": 1,
                "delivered": 2,
            },
        },
    }


def test_query_cash_pipeline_splits_receivable_by_age(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    product = next(iter(env.products.values()))
    env.t = 200
    env.scenario["settlement"]["normal_delay_hours"] = 48
    dbm.insert_orders(env.conn, run_id, [
        Order(
            order_id="cash-due-now",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=100,
            promised_delivery_t=130,
            sale_price=88.0,
            purchase_price=44.0,
            current_status="delivered",
            purchase_t=100,
            shipped_t=120,
            delivered_t=152,
        ),
    ])

    resp = _act(c, run_id, "agent_0", "cash",
                [("query_cash_pipeline", {"window_days": 1})])
    result = _tool_result(resp)

    assert result["receivable_aging"] == {
        "total": {"amount": 88.0, "order_count": 1},
        "delivered_within_window": {"amount": 0.0, "order_count": 0},
        "delivered_before_window": {"amount": 88.0, "order_count": 1},
    }


def test_query_cash_pipeline_ignores_hidden_settlement_outcomes(hook_session):
    c, run_id, _ = hook_session
    env = c.application.registry._require(run_id)
    product = next(iter(env.products.values()))
    env.t = 101
    env.scenario["settlement"]["normal_delay_hours"] = 240
    dbm.insert_orders(env.conn, run_id, [
        Order(
            order_id="cash-order-specific-due",
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            agent_id="agent_0",
            order_t=10,
            promised_delivery_t=40,
            sale_price=99.0,
            purchase_price=50.0,
            current_status="delivered",
            purchase_t=10,
            shipped_t=20,
            delivered_t=100,
            settlement_delay_steps=0,
        ),
    ])

    before = _tool_result(_act(
        c, run_id, "agent_0", "cash",
        [("query_cash_pipeline", {"window_days": 1})],
    ))
    env.conn.execute(
        "UPDATE orders SET settlement_delay_steps=?, preset_anomaly=?"
        " WHERE run_id=? AND order_id=?",
        (240, "only_refund", run_id, "cash-order-specific-due"),
    )
    after = _tool_result(_act(
        c, run_id, "agent_0", "cash again",
        [("query_cash_pipeline", {"window_days": 1})],
    ))

    assert before == after
    assert "receivable_due" not in before
    assert before["receivable_aging"]["total"] == {
        "amount": 99.0,
        "order_count": 1,
    }


def test_query_cash_pipeline_rejects_unsupported_window(hook_session):
    c, run_id, _ = hook_session

    resp = _act(c, run_id, "agent_0", "cash",
                [("query_cash_pipeline", {"window_days": 2})])
    result = _tool_result(resp)

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["path"] == "$.window_days"


# ---------- review_my_listings ----------


def test_review_my_listings_returns_expected_columns(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    payload = _tool_result(resp)

    assert payload["sort_by"] == "listing_age_days"
    assert payload["window_days"] == 7
    assert payload["columns"] == [
        "product_id", "name", "listing_age_days", "days_without_sales",
        "procured_orders", "fine", "open_orders",
        "listing_rating",
    ]
    rows = _table_records(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["product_id"] == prod["product_id"]
    assert row["name"] == prod["name"]
    assert row["listing_age_days"] == 0
    assert row["days_without_sales"] == 0
    assert row["procured_orders"] == 0
    assert row["fine"] == 0.0
    assert row["open_orders"] == 0


def test_review_my_listings_reflects_orders(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    order = Order(
        order_id="O_review_1",
        product_id=prod["product_id"],
        supplier_id=prod["supplier_id"],
        agent_id="agent_0",
        order_t=1,
        promised_delivery_t=4,
        sale_price=prod["price"],
        purchase_price=prod["price"] * 0.8,
        current_status="settled_bad_review",
        purchase_t=1, shipped_t=2, delivered_t=3, settled_t=4,
        supplier_ship_hours=prod["supplier_ship_hours"],
        total_penalty=2.5,
    )
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [order])
        dbm.write_events(env.conn, run_id, [EventLog(
            t=4,
            event_type="order_settled_bad_review",
            entity_id=order.order_id,
            agent_id="agent_0",
            payload={"penalty": 2.5},
        )])
    env.t = 4

    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    rows = _table_records(_tool_result(resp))
    row = next(r for r in rows if r["product_id"] == prod["product_id"])

    assert row["procured_orders"] >= 1
    assert row["fine"] == 2.5
    assert row["days_without_sales"] == 0


def test_review_my_listings_reports_full_days_without_sales(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])

    env.t = 20 * 24  # day 21
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [
            Order(
                order_id="O_review_old_sale",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=2 * 24,
                promised_delivery_t=2 * 24 + 3,
                sale_price=prod["price"],
                purchase_price=prod["price"] * 0.8,
                current_status="settled_normal",
                settled_t=2 * 24 + 3,
            ),
            Order(
                order_id="O_review_recent_sale",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=19 * 24,
                promised_delivery_t=19 * 24 + 3,
                sale_price=prod["price"],
                purchase_price=prod["price"] * 0.8,
                current_status="settled_normal",
                settled_t=19 * 24 + 3,
            ),
            Order(
                order_id="O_review_failed_today",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=20 * 24,
                promised_delivery_t=20 * 24 + 3,
                sale_price=prod["price"],
                purchase_price=prod["price"] * 0.8,
                current_status="stockout",
                settled_t=20 * 24,
            ),
        ])

    resp = _act(c, run_id, "agent_0", "review", [
        ("review_my_listings", {"sort_by": "days_without_sales"})
    ])
    row = next(
        item for item in _table_records(_tool_result(resp))
        if item["product_id"] == prod["product_id"]
    )

    assert row["listing_age_days"] == 20
    assert row["days_without_sales"] == 1
    assert row["procured_orders"] == 1


def test_review_my_listings_relisting_starts_fresh_no_sale_window(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [Order(
            order_id="O_review_before_relist",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=24,
            promised_delivery_t=27,
            sale_price=prod["price"],
            purchase_price=prod["price"] * 0.8,
            current_status="settled_normal",
            settled_t=27,
        )])

    env.t = 10 * 24
    _act(c, run_id, "agent_0", "delist", [
        ("delist_product", {"items": [{"product_id": prod["product_id"]}]})
    ])
    _act(c, run_id, "agent_0", "relist", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    env.t = 12 * 24

    resp = _act(c, run_id, "agent_0", "review", [
        ("review_my_listings", {})
    ])
    row = next(
        item for item in _table_records(_tool_result(resp))
        if item["product_id"] == prod["product_id"]
    )

    assert row["listing_age_days"] == 2
    assert row["days_without_sales"] == 2
    assert row["procured_orders"] == 0


def test_review_my_listings_excludes_order_created_before_same_step_relist(
    hook_session,
):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    env.t = 24
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [Order(
            order_id="O_review_same_step_before_relist",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=env.t,
            promised_delivery_t=env.t + 3,
            sale_price=prod["price"],
            purchase_price=prod["price"] * 0.8,
            current_status="ordered",
            purchase_t=env.t,
        )])

    _act(c, run_id, "agent_0", "delist", [
        ("delist_product", {"items": [{"product_id": prod["product_id"]}]})
    ])
    _act(c, run_id, "agent_0", "relist", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])

    result = _tool_result(_act(c, run_id, "agent_0", "review", [
        ("review_my_listings", {})
    ]))
    row = next(
        item for item in _table_records(result)
        if item["product_id"] == prod["product_id"]
    )

    assert row["listing_age_days"] == 0
    assert row["days_without_sales"] == 0
    assert row["procured_orders"] == 0
    assert row["open_orders"] == 1


def test_relisting_restores_only_procured_order_accumulators(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [
            Order(
                order_id="O_relist_procured",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=24,
                sale_price=100.0,
                purchase_price=60.0,
                current_status="settled_normal",
                settled_t=24,
            ),
            Order(
                order_id="O_relist_stockout",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=2,
                promised_delivery_t=24,
                sale_price=90.0,
                purchase_price=50.0,
                current_status="stockout",
                settled_t=2,
            ),
            Order(
                order_id="O_relist_insufficient",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=3,
                promised_delivery_t=24,
                sale_price=80.0,
                purchase_price=40.0,
                current_status="insufficient_balance",
                settled_t=3,
            ),
        ])

    _act(c, run_id, "agent_0", "delist", [
        ("delist_product", {"items": [{"product_id": prod["product_id"]}]})
    ])
    _act(c, run_id, "agent_0", "relist", [
        ("list_product", {"items": [{
            "product_id": prod["product_id"],
            "sale_price": prod["price"],
        }]})
    ])

    listing = dbm.get_listing(
        env.conn, run_id, "agent_0", prod["product_id"],
    )
    assert listing.cum_sales == 1
    assert listing.cum_revenue == 100.0

    rows = _table_records(_tool_result(_act(
        c,
        run_id,
        "agent_0",
        "check listings",
        [("query_my_listings", {})],
    )))
    row = next(item for item in rows if item["product_id"] == prod["product_id"])
    assert row["procured_orders"] == 1


def test_review_my_listings_uses_full_seven_days_for_non_divisor_step_hours(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    env.scenario["run"]["step_hours"] = 5
    env.t = 34  # 170 elapsed hours; t=1 is still within the previous 168 hours.
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [
            Order(
                order_id="O_review_window_boundary",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=1,
                promised_delivery_t=4,
                sale_price=prod["price"],
                purchase_price=prod["price"] * 0.8,
                current_status="settled_normal",
                settled_t=4,
            ),
            Order(
                order_id="O_review_before_window",
                product_id=prod["product_id"],
                supplier_id=prod["supplier_id"],
                agent_id="agent_0",
                order_t=0,
                promised_delivery_t=4,
                sale_price=prod["price"],
                purchase_price=prod["price"] * 0.8,
                current_status="settled_normal",
                settled_t=4,
            ),
        ])

    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    row = next(
        r for r in _table_records(_tool_result(resp))
        if r["product_id"] == prod["product_id"]
    )
    assert row["procured_orders"] == 1


def test_review_my_listings_counts_fines_by_penalty_event_time(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    env.t = 240
    order_id = "O_review_recent_penalty"
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [Order(
            order_id=order_id,
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=0,
            promised_delivery_t=24,
            sale_price=prod["price"],
            purchase_price=prod["price"] * 0.8,
            current_status="settled_bad_review",
            settled_t=239,
            total_penalty=2.5,
        )])
        dbm.write_events(env.conn, run_id, [EventLog(
            t=239,
            event_type="order_settled_bad_review",
            entity_id=order_id,
            agent_id="agent_0",
            payload={"penalty": 2.5},
        )])

    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    row = next(
        r for r in _table_records(_tool_result(resp))
        if r["product_id"] == prod["product_id"]
    )
    assert row["fine"] == 2.5


def test_review_my_listings_counts_open_orders_per_product(hook_session):
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)
    other_products = [
        product
        for product in env.products.values()
        if product.product_id != prod["product_id"]
    ][:2]
    assert len(other_products) == 2
    listed_products = [
        {
            "product_id": prod["product_id"],
            "supplier_id": prod["supplier_id"],
            "price": prod["price"],
            "supplier_ship_hours": prod["supplier_ship_hours"],
        },
        *[
            {
                "product_id": product.product_id,
                "supplier_id": product.supplier_id,
                "price": product.price,
                "supplier_ship_hours": product.supplier_ship_hours,
            }
            for product in other_products
        ],
    ]

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [
            {"product_id": product["product_id"], "sale_price": product["price"]}
            for product in listed_products
        ]})
    ])

    orders = []
    for product, count in zip(listed_products, (3, 2, 0)):
        orders.extend(
            Order(
                order_id=f"O_review_open_{product['product_id']}_{i}",
                product_id=product["product_id"],
                supplier_id=product["supplier_id"],
                agent_id="agent_0",
                order_t=i,
                promised_delivery_t=i + 10,
                sale_price=product["price"],
                purchase_price=product["price"] * 0.8,
                current_status="ordered",
                supplier_ship_hours=product["supplier_ship_hours"],
            )
            for i in range(count)
        )
    with env.lock:
        dbm.insert_orders(env.conn, run_id, orders)

    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    rows = _table_records(_tool_result(resp))
    open_orders_by_product = {
        row["product_id"]: row["open_orders"]
        for row in rows
    }
    assert open_orders_by_product == {
        product["product_id"]: expected
        for product, expected in zip(listed_products, (3, 2, 0))
    }


def test_review_my_listings_unknown_agent_returns_error(hook_session):
    c, run_id, _ = hook_session
    resp = _act(c, run_id, "missing_agent", "review", [("review_my_listings", {})])
    assert resp.status_code == 404
    assert "unknown agent" in resp.get_json()["error"]


def test_review_my_listings_sort_options(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    for sort_by in (
        "listing_age_days",
        "days_without_sales",
        "fine",
        "procured_orders",
    ):
        resp = _act(c, run_id, "agent_0", "review",
                    [("review_my_listings", {"sort_by": sort_by})])
        payload = _tool_result(resp)
        assert payload["sort_by"] == sort_by

    # Bad sort_by
    resp = _act(c, run_id, "agent_0", "review",
                [("review_my_listings", {"sort_by": "invalid"})])
    payload = _tool_result(resp)
    assert payload["ok"] is False
    assert payload["error"]["path"] == "$.sort_by"


def test_review_my_listings_rejects_non_string_sort_by(hook_session):
    c, run_id, _ = hook_session

    resp = _act(c, run_id, "agent_0", "review",
                [("review_my_listings", {"sort_by": []})])

    assert resp.status_code == 200
    payload = _tool_result(resp)
    assert payload["ok"] is False
    assert payload["error"]["path"] == "$.sort_by"


def test_review_my_listings_empty_when_no_listings(hook_session):
    c, run_id, _ = hook_session
    resp = _act(c, run_id, "agent_0", "review", [("review_my_listings", {})])
    payload = _tool_result(resp)
    assert payload["count"] == 0
    assert payload["rows"] == []


def test_review_my_listings_window_days_30(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    resp = _act(c, run_id, "agent_0", "review",
                [("review_my_listings", {"window_days": 30})])
    payload = _tool_result(resp)
    assert payload["window_days"] == 30


def test_review_my_listings_invalid_window_days(hook_session):
    c, run_id, prod = hook_session
    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    resp = _act(c, run_id, "agent_0", "review",
                [("review_my_listings", {"window_days": 14})])
    payload = _tool_result(resp)
    assert payload["ok"] is False
    assert payload["error"]["path"] == "$.window_days"


def test_review_my_listings_window_30_captures_old_orders(hook_session):
    """Orders outside 7-day window but inside 30-day window should appear only with window_days=30."""
    c, run_id, prod = hook_session
    env = c.application.registry._require(run_id)

    _act(c, run_id, "agent_0", "list it", [
        ("list_product", {"items": [{"product_id": prod["product_id"], "sale_price": prod["price"]}]})
    ])

    # Place an order 10 days ago (outside 7-day window, inside 30-day window).
    # With step_hours=1, t=240 is 10 days ago from t=480.
    env.t = 480
    with env.lock:
        dbm.insert_orders(env.conn, run_id, [Order(
            order_id="O_review_30day_window",
            product_id=prod["product_id"],
            supplier_id=prod["supplier_id"],
            agent_id="agent_0",
            order_t=240,  # 10 days ago
            promised_delivery_t=250,
            sale_price=prod["price"],
            purchase_price=prod["price"] * 0.8,
            current_status="settled_normal",
            settled_t=250,
        )])

    # 7-day window should not see this order.
    resp7 = _act(c, run_id, "agent_0", "review",
                 [("review_my_listings", {"window_days": 7})])
    row7 = next(
        r for r in _table_records(_tool_result(resp7))
        if r["product_id"] == prod["product_id"]
    )
    assert row7["procured_orders"] == 0

    # 30-day window should see this order.
    resp30 = _act(c, run_id, "agent_0", "review",
                  [("review_my_listings", {"window_days": 30})])
    row30 = next(
        r for r in _table_records(_tool_result(resp30))
        if r["product_id"] == prod["product_id"]
    )
    assert row30["procured_orders"] == 1
