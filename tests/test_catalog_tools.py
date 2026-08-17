"""Catalog/search tools for the paper-safe sourcing surface.

These tests intentionally exercise the unified /act endpoint because agent
tools are hook-gated and stored in the normal trace path.
"""
import json
import os
import tempfile
import threading
import time

import pytest

from core import sim_time
from storage import db as dbm
from web.app import create_app
from web.runner import load_default_scenario


VISIBLE_PRODUCT_KEYS = {
    "product_id",
    "name",
    "quantity",
    "price",
    "supplier_id",
    "supplier_name",
    "supplier_ship_hours",
    "logistics_hours",
    "category",
    "historical_avg_rating",
    "shop_rating",
    "supplier_age_years",
}


def _act(c, rid, agent_id, thought, tool_calls_spec):
    tc_list = []
    for i, (name, args) in enumerate(tool_calls_spec):
        tc_list.append({
            "id": f"call_{i}_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    body = {"messages": [{"role": "assistant", "content": thought,
                          "tool_calls": tc_list}]}
    return c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)


def _tool_result(resp, index=0):
    data = resp.get_json()
    assert data["ok"], data
    return json.loads(data["tool_results"][index]["content"])


def _table_records(table):
    assert isinstance(table, dict)
    assert set(table) >= {"columns", "rows"}
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def _records(payload, key):
    value = payload[key]
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        return _table_records(value)
    return value


@pytest.fixture
def hook_session():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0
        scen["run"]["horizon_steps"] = 24
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 80
        scen.setdefault("agent", {})["tool_denylist"] = []
        rid = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        th = threading.Thread(target=lambda: app.registry.step(rid), daemon=True)
        th.start()
        env = app.registry._require(rid)
        deadline = time.monotonic() + 5.0
        with env.hook_cond:
            while not env.hook_open:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "timed out waiting for catalog-tool hook"
                env.hook_cond.wait(timeout=remaining)
        try:
            yield c, app, rid
        finally:
            _act(c, rid, "agent_0", "done", [("end_of_step", {})])
            th.join(timeout=3)


def test_old_catalog_tools_are_not_registered(hook_session):
    c, _, rid = hook_session
    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    names = {tool["name"] for tool in schema["tools"]}
    assert "market_brief" in names
    assert "hot_search_terms" in names
    assert "search_products" in names
    assert "get_supplier_profile" in names
    assert "list_supplier_products" in names
    assert "list_category" not in names
    assert "websearch" not in names
    assert "browse_category" not in names


def test_market_brief_returns_all_categories_without_top_products(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    expected_categories = {p.category for p in env.products.values()}

    resp = _act(c, rid, "agent_0", "market", [("market_brief", {"window_days": 7})])
    out = _tool_result(resp)

    assert out["window_days"] == 7
    assert {row["category"] for row in out["categories"]} == expected_categories
    assert all(set(row) == {"category", "total_sales", "daily_avg_sales",
                             "total_gmv", "daily_avg_gmv", "avg_price"}
               for row in out["categories"])
    assert all(type(row["total_sales"]) is list for row in out["categories"])
    assert all(len(row["total_sales"]) == 7 for row in out["categories"])
    assert all(all(type(day_sales) is int for day_sales in row["total_sales"])
               for row in out["categories"])
    assert all(type(row["daily_avg_sales"]) is int for row in out["categories"])
    assert all(type(row["total_gmv"]) is list for row in out["categories"])
    assert all(len(row["total_gmv"]) == 7 for row in out["categories"])
    assert all(all(isinstance(v, float) for v in row["total_gmv"])
               for row in out["categories"])
    assert all(isinstance(row["daily_avg_gmv"], float) for row in out["categories"])
    assert all(isinstance(row["avg_price"], float) for row in out["categories"])
    assert "top" not in out
    assert "top_products" not in out


def test_market_brief_returns_daily_sales_series_oldest_to_newest(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    step_hours = int(env.scenario["run"]["step_hours"])
    latest_completed_idx = (
        sim_time.curve_day_index(env.scenario, env.t, step_hours) - 1
    ) % 365
    small_share = float(env.scenario["data"]["small_share"])

    resp = _act(c, rid, "agent_0", "market", [("market_brief", {"window_days": 7})])
    out = _tool_result(resp)

    for row in out["categories"]:
        raw_daily = [0.0 for _ in range(7)]
        raw_gmv_daily = [0.0 for _ in range(7)]
        category_prices = []
        for product in env.products.values():
            if product.category != row["category"]:
                continue
            category_prices.append(product.price)
            for offset, d in enumerate(range(6, -1, -1)):
                day_idx = (latest_completed_idx - d) % 365
                raw_daily[offset] += product.market_curve[day_idx]
                raw_gmv_daily[offset] += (
                    product.market_curve[day_idx] * product.ref_price
                )
        expected_daily = [round(value * small_share) for value in raw_daily]
        expected_gmv_daily = [round(value * small_share, 2) for value in raw_gmv_daily]
        expected_avg_gmv = round(sum(value * small_share for value in raw_gmv_daily) / 7, 2)
        expected_avg_price = round(sum(category_prices) / len(category_prices), 2) if category_prices else 0.0
        assert row["total_sales"] == expected_daily
        assert row["daily_avg_sales"] == round(sum(value * small_share for value in raw_daily) / 7)
        assert row["total_gmv"] == expected_gmv_daily
        assert row["daily_avg_gmv"] == expected_avg_gmv
        assert row["avg_price"] == expected_avg_price


def test_market_brief_historical_gmv_ignores_current_supplier_price_changes(
    hook_session,
):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    category = product.category

    before = _tool_result(_act(c, rid, "agent_0", "market before", [
        ("market_brief", {"window_days": 7})
    ]))
    before_row = next(
        row for row in before["categories"] if row["category"] == category
    )

    product.price *= 2.0

    after = _tool_result(_act(c, rid, "agent_0", "market after", [
        ("market_brief", {"window_days": 7})
    ]))
    after_row = next(
        row for row in after["categories"] if row["category"] == category
    )

    assert after_row["total_sales"] == before_row["total_sales"]
    assert after_row["total_gmv"] == before_row["total_gmv"]
    assert after_row["daily_avg_gmv"] == before_row["daily_avg_gmv"]
    assert after_row["avg_price"] != before_row["avg_price"]


def test_market_brief_rejects_non_public_windows(hook_session):
    c, _, rid = hook_session
    resp = _act(c, rid, "agent_0", "market", [("market_brief", {"window_days": 14})])
    out = _tool_result(resp)
    assert out["ok"] is False
    assert "window_days" in out["error"]


def test_market_brief_uses_virtual_calendar_curve_offset(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    env.scenario["run"]["virtual_time"] = {
        "enabled": True,
        "start_date": "2025-06-15",
    }
    env.scenario["data"]["calendar_anchor_date"] = "2025-06-01"
    for product in env.products.values():
        product.market_curve = [float(index) for index in range(365)]

    resp = _act(c, rid, "agent_0", "market offset", [
        ("market_brief", {"window_days": 7})
    ])
    out = _tool_result(resp)

    latest_completed_idx = (
        sim_time.curve_day_index(
            env.scenario, env.t, int(env.scenario["run"]["step_hours"])
        ) - 1
    ) % 365
    small_share = float(env.scenario["data"]["small_share"])
    for row in out["categories"]:
        product_count = sum(
            product.category == row["category"] for product in env.products.values()
        )
        expected = [
            round(
                product_count
                * ((latest_completed_idx - offset) % 365)
                * small_share
            )
            for offset in range(6, -1, -1)
        ]
        assert row["total_sales"] == expected


def test_hot_search_terms_ends_at_last_completed_virtual_day(
    hook_session, monkeypatch,
):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    env.t = 5 * 24 + 12
    captured = {}

    class CapturingIndex:
        def rank(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(env, "_hot_search_index", CapturingIndex(), raising=False)

    out = _tool_result(_act(c, rid, "agent_0", "hot terms", [
        ("hot_search_terms", {"window_days": 7})
    ]))

    current_idx = sim_time.curve_day_index(
        env.scenario,
        env.t,
        int(env.scenario["run"]["step_hours"]),
    )
    assert captured["today_idx"] == (current_idx - 1) % 365
    assert _table_records(out["trends"]) == []


def test_hot_search_terms_returns_fixed_ten_ranked_trend_rows(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    products = sorted(env.products.values(), key=lambda p: p.product_id)
    test_category = "test_hot_terms"
    phrases = [
        "折叠收纳箱", "厨房沥水架", "桌面收纳盒", "免打孔挂钩",
        "家用垃圾桶", "不锈钢置物架", "宿舍鞋架", "衣柜收纳袋",
        "旋转拖把桶", "食品保鲜盒", "浴室防滑垫", "衣架",
    ]
    for idx, phrase in enumerate(phrases):
        for dup in range(2):
            p = products[idx * 2 + dup]
            p.category = test_category
            p.name = f"{phrase} 家用"
            p.market_curve = [float(100 - idx)] * 365

    resp = _act(c, rid, "agent_0", "hot terms", [
        ("hot_search_terms", {"category": test_category, "window_days": 7})
    ])
    out = _tool_result(resp)

    assert set(out) == {"day", "date", "window_days", "category", "trends"}
    assert out["day"] == 1
    assert out["date"] == "2025-06-01"
    assert out["window_days"] == 7
    assert out["category"] == test_category
    assert out["trends"]["columns"] == [
        "rank", "keyword", "category", "trend", "change_pct", "rank_change",
    ]
    rows = _table_records(out["trends"])
    assert len(rows) == 10
    assert [row["rank"] for row in rows] == list(range(1, 11))
    assert all(row["category"] == test_category for row in rows)
    assert all(row["trend"] == "stable" for row in rows)
    assert all(row["change_pct"] == 0.0 for row in rows)
    assert "折叠收纳箱" in [row["keyword"] for row in rows[:3]]
    assert "厨房沥水架" in [row["keyword"] for row in rows[:4]]


def test_hot_search_terms_rejects_invalid_window_days(hook_session):
    c, _, rid = hook_session

    resp = _act(c, rid, "agent_0", "bad hot terms", [
        ("hot_search_terms", {"window_days": 14})
    ])
    out = _tool_result(resp)

    assert out["ok"] is False
    assert "window_days" in out["error"]


def test_search_products_paginates_visible_fields_and_filters_delisted(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    first_product = sorted(env.products.values(), key=lambda p: p.product_id)[0]
    first_product.is_listed_by_supplier = False

    resp1 = _act(c, rid, "agent_0", "search page 1", [
        ("search_products", {"query": "",
                             "page": 1, "page_size": 3})
    ])
    page1 = _tool_result(resp1)
    resp2 = _act(c, rid, "agent_0", "search page 2", [
        ("search_products", {"query": "",
                             "page": 2, "page_size": 3})
    ])
    page2 = _tool_result(resp2)

    assert page1["page"] == 1
    assert page1["page_size"] == 3
    page1_items = _records(page1, "items")
    page2_items = _records(page2, "items")
    assert len(page1_items) <= 3
    assert all(set(item) == VISIBLE_PRODUCT_KEYS for item in page1_items)
    assert first_product.product_id not in {item["product_id"] for item in page1_items}
    assert {item["product_id"] for item in page1_items}.isdisjoint(
        {item["product_id"] for item in page2_items}
    )


def test_get_product_detail_filters_supplier_delisted_product(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    product.is_listed_by_supplier = False

    resp = _act(c, rid, "agent_0", "product detail", [
        ("get_product_detail", {"product_id": product.product_id})
    ])
    result = _tool_result(resp)

    assert result == {"ok": False, "error": "not found"}


def test_get_product_detail_keeps_delisted_product_visible_to_listing_owner(
    hook_session,
):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    listed = _act(c, rid, "agent_0", "list product", [
        ("list_product", {"items": [{
            "product_id": product.product_id,
            "sale_price": product.price * 1.5,
        }]})
    ])
    assert _records(_tool_result(listed), "items")[0]["ok"] is True
    available_detail = _tool_result(_act(
        c, rid, "agent_0", "available product detail", [
            ("get_product_detail", {"product_id": product.product_id})
        ],
    ))
    assert available_detail["supplier_available"] is True
    product.is_listed_by_supplier = False

    detail = _tool_result(_act(c, rid, "agent_0", "product detail", [
        ("get_product_detail", {"product_id": product.product_id})
    ]))

    assert detail["product_id"] == product.product_id
    assert detail["supplier_available"] is False
    assert set(detail) == VISIBLE_PRODUCT_KEYS | {"supplier_available"}


def test_search_products_keyword_matches_name_substrings(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    product.name = "唯一星河耳机标记"
    dbm.insert_products(env.conn, rid, env.products.values())

    resp = _act(c, rid, "agent_0", "search keyword", [
        ("search_products", {"query": "星河耳机", "page": 1, "page_size": 10})
    ])
    out = _tool_result(resp)

    assert any(item["product_id"] == product.product_id for item in _records(out, "items"))


def test_search_products_schema_exposes_filters_and_sort_modes(hook_session):
    c, _, rid = hook_session
    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    hot_terms_tool = next(
        tool for tool in schema["tools"] if tool["name"] == "hot_search_terms"
    )
    hot_terms_schema = hot_terms_tool["parameters"]
    assert "limit" not in hot_terms_schema["properties"]
    assert hot_terms_schema["properties"]["window_days"]["enum"] == [7, 30]
    assert "demand proxy" in hot_terms_tool["description"].lower()
    assert "compact trend table" in hot_terms_tool["description"].lower()

    search_schema = next(
        tool for tool in schema["tools"] if tool["name"] == "search_products"
    )["parameters"]

    props = search_schema["properties"]
    for name in {
        "price_min",
        "price_max",
        "supplier_rating_min",
        "historical_rating_min",
        "logistics_hours_max",
        "supplier_ship_hours_max",
        "quantity_min",
        "sort_by",
    }:
        assert name in props
    assert props["sort_by"]["enum"] == [
        "relevance",
        "price_asc",
        "price_desc",
        "rating",
        "supplier_rating",
        "logistics_speed",
    ]
    assert props["page"]["maximum"] == 1_000_000


def test_search_products_rejects_nonfinite_filters_and_excessive_page(
    hook_session,
):
    c, _, rid = hook_session

    nonfinite = _tool_result(_act(c, rid, "agent_0", "bad price", [
        ("search_products", {"price_min": float("nan")})
    ]))
    excessive_page = _tool_result(_act(c, rid, "agent_0", "bad page", [
        ("search_products", {"page": 1_000_001})
    ]))

    assert nonfinite["ok"] is False
    assert nonfinite["error"]["path"] == "$.price_min"
    assert "finite" in nonfinite["error"]["message"]
    assert excessive_page["ok"] is False
    assert excessive_page["error"]["path"] == "$.page"


def test_search_products_filters_on_public_product_fields(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    target = sorted(env.products.values(), key=lambda p: p.product_id)[10]
    args = {
        "query": "",
        "price_min": target.price,
        "price_max": target.price,
        "supplier_rating_min": target.shop_rating,
        "historical_rating_min": target.historical_avg_rating,
        "logistics_hours_max": target.logistics_hours,
        "supplier_ship_hours_max": target.supplier_ship_hours,
        "quantity_min": target.quantity,
        "page": 1,
        "page_size": 50,
    }
    expected_ids = [
        p.product_id for p in sorted(
            env.products.values(), key=lambda p: (p.category, p.product_id)
        )
        if p.is_listed_by_supplier
        and p.price >= target.price
        and p.price <= target.price
        and p.shop_rating >= target.shop_rating
        and p.historical_avg_rating >= target.historical_avg_rating
        and p.logistics_hours <= target.logistics_hours
        and p.supplier_ship_hours <= target.supplier_ship_hours
        and p.quantity >= target.quantity
    ]

    resp = _act(c, rid, "agent_0", "filtered search", [("search_products", args)])
    out = _tool_result(resp)

    assert [item["product_id"] for item in _records(out, "items")] == expected_ids[:50]


def test_search_products_sorts_by_price_and_logistics_speed(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    products = [
        p for p in env.products.values()
        if p.is_listed_by_supplier
    ]

    price_resp = _act(c, rid, "agent_0", "price sort", [
        ("search_products", {
            "query": "", "sort_by": "price_asc",
            "page": 1, "page_size": 10,
        })
    ])
    price_out = _tool_result(price_resp)
    expected_price_ids = [
        p.product_id for p in sorted(products, key=lambda p: (p.price, p.product_id))
    ][:10]
    assert [item["product_id"] for item in _records(price_out, "items")] == expected_price_ids

    logistics_resp = _act(c, rid, "agent_0", "logistics sort", [
        ("search_products", {
            "query": "", "sort_by": "logistics_speed",
            "page": 1, "page_size": 10,
        })
    ])
    logistics_out = _tool_result(logistics_resp)
    expected_logistics_ids = [
        p.product_id for p in sorted(
            products,
            key=lambda p: (p.supplier_ship_hours + p.logistics_hours, p.product_id),
        )
    ][:10]
    assert [item["product_id"] for item in _records(logistics_out, "items")] == expected_logistics_ids


def test_search_products_rejects_single_chinese_query_terms(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    product.name = "蓝牙耳机"
    dbm.insert_products(env.conn, rid, env.products.values())

    resp = _act(c, rid, "agent_0", "single cjk query", [
        ("search_products", {"query": "耳", "page": 1, "page_size": 10})
    ])
    out = _tool_result(resp)

    assert out == {
        "ok": False,
        "error": "单字搜索不支持，请至少输入两个汉字",
    }


def test_search_products_relevance_allows_bm25_partial_term_matches(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    products = sorted(env.products.values(), key=lambda p: p.product_id)
    products[0].name = "蓝牙耳机"
    products[0].category = "electronics"
    products[1].name = "Camera"
    products[1].category = "electronics"
    dbm.insert_products(env.conn, rid, env.products.values())

    resp = _act(c, rid, "agent_0", "mixed short cjk query", [
        ("search_products", {
            "query": "耳机 electronics",
            "page": 1,
            "page_size": 20,
        })
    ])
    out = _tool_result(resp)
    returned_ids = {item["product_id"] for item in _records(out, "items")}

    assert products[0].product_id in returned_ids
    assert products[1].product_id in returned_ids


def test_supplier_tools_return_public_profile_and_paginated_products(hook_session):
    c, app, rid = hook_session
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))

    profile_resp = _act(c, rid, "agent_0", "supplier profile", [
        ("get_supplier_profile", {"supplier_id": product.supplier_id})
    ])
    profile = _tool_result(profile_resp)
    assert set(profile) == {
        "supplier_id",
        "supplier_name",
        "shop_rating",
        "return_buyer_rate",
        "supplier_age_years",
        "product_count",
    }
    assert profile["supplier_id"] == product.supplier_id
    assert profile["shop_rating"] == product.shop_rating

    products_resp = _act(c, rid, "agent_0", "supplier products", [
        ("list_supplier_products", {"supplier_id": product.supplier_id,
                                    "page": 1, "page_size": 5})
    ])
    out = _tool_result(products_resp)
    assert out["supplier_id"] == product.supplier_id
    assert out["page"] == 1
    assert out["page_size"] == 5
    items = _records(out, "items")
    assert all(set(item) == VISIBLE_PRODUCT_KEYS for item in items)
    assert all(item["supplier_id"] == product.supplier_id for item in items)


VISIBLE_PRODUCT_KEYS_V6 = VISIBLE_PRODUCT_KEYS | {
    "return_rate",
    "return_buyer_rate",
}


def test_v6_product_cards_include_return_rate_not_latent_refund_rate(hook_session):
    from core.economy_v6 import EconomyV6, public_return_rate

    c, app, rid = hook_session
    env = app.registry._require(rid)
    env.scenario.setdefault("economy_v6", {})["enabled"] = True
    env.economy_v6 = EconomyV6.from_scenario(env.scenario)
    product = next(p for p in env.products.values() if p.is_listed_by_supplier)
    expected_return_rate = public_return_rate(
        product.refund_rate, product.only_refund_rate,
    )

    detail = _tool_result(_act(c, rid, "agent_0", "product detail", [
        ("get_product_detail", {"product_id": product.product_id})
    ]))
    assert set(detail) == VISIBLE_PRODUCT_KEYS_V6 | {"supplier_available"}
    assert detail["return_rate"] == expected_return_rate
    assert detail["return_buyer_rate"] == product.return_buyer_rate
    assert "refund_rate" not in detail
    assert "only_refund_rate" not in detail
    assert "cancel_rate" not in detail

    search = _tool_result(_act(c, rid, "agent_0", "search", [
        ("search_products", {
            "query": product.name,
            "page": 1,
            "page_size": 5,
        })
    ]))
    items = _records(search, "items")
    assert items
    assert all(set(item) == VISIBLE_PRODUCT_KEYS_V6 for item in items)
    match = next(item for item in items if item["product_id"] == product.product_id)
    assert match["return_rate"] == expected_return_rate
    assert "refund_rate" not in match

    listed = _tool_result(_act(c, rid, "agent_0", "supplier products", [
        ("list_supplier_products", {
            "supplier_id": product.supplier_id,
            "page": 1,
            "page_size": 5,
        })
    ]))
    supplier_items = _records(listed, "items")
    assert supplier_items
    assert all(set(item) == VISIBLE_PRODUCT_KEYS_V6 for item in supplier_items)
