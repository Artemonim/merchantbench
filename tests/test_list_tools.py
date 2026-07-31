"""list_tools registry entry — hook-gated OpenAI schema discovery.

list_tools is a per-agent tool, so it's hook-gated like every other tool:
outside the hook window the /act route returns 425. Tests drive a step in a
background thread to open a hook window for the assertions."""
import json
import os
import tempfile
import threading
import time
from typing import get_args

import pytest

from core.entities import OrderStatus
from web.app import create_app
from web.runner import load_default_scenario


@pytest.fixture
def env():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        yield c, app


def _new_run(c, agent_overrides=None, hook_seconds: float = 2.0):
    scen = load_default_scenario()
    scen["run"]["horizon_steps"] = 3
    scen["run"]["max_hook_seconds"] = hook_seconds
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 30
    if agent_overrides:
        scen.setdefault("agent", {}).update(agent_overrides)
    return c.post("/runs", json={"scenario": scen}).get_json()["run_id"]


def _new_run_with_scenario(c, scenario):
    return c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]


def _make_tool_call(name, arguments, call_id=None):
    if call_id is None:
        call_id = f"call_{name}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
        },
    }


def _act_msg(tool_calls):
    """Build the request body for POST /act."""
    return {
        "messages": [{
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }]
    }


def _call_tool_via_act(c, rid, agent_id, tool_name, args=None):
    """Call a tool via the /act endpoint and return parsed tool result content."""
    tc = _make_tool_call(tool_name, args or {})
    body = _act_msg([tc])
    resp = c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)
    data = resp.get_json()
    if not data.get("ok"):
        return resp.status_code, data
    content = json.loads(data["tool_results"][0]["content"])
    return 200, content


def _in_hook(app, rid, c, fn):
    """Drive one step in another thread, run fn() while hook is open, then
    release via end_of_step tool. fn receives no args."""
    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    env = app.registry._require(rid)
    deadline = time.monotonic() + 5.0
    while not env.hook_open and time.monotonic() < deadline:
        time.sleep(0.01)
    if not env.hook_open:
        raise RuntimeError("Hook window did not open within 5 seconds")
    try:
        return fn()
    finally:
        # Release hook via end_of_step tool through /act
        end_body = _act_msg([_make_tool_call("end_of_step", {}, "call_end")])
        c.post(f"/runs/{rid}/agents/agent_0/act", json=end_body)
        th.join(timeout=3)


def test_list_tools_returns_openai_function_schemas(env):
    c, app = env
    rid = _new_run(c)

    def check():
        status, out = _call_tool_via_act(c, rid, "agent_0", "list_tools")
        assert status == 200
        assert isinstance(out, list) and out
        for row in out:
            assert set(row.keys()) == {"type", "function"}
            assert row["type"] == "function"
            fn = row["function"]
            assert set(fn.keys()) == {"name", "description", "parameters"}
            assert isinstance(fn["name"], str) and fn["name"]
            assert isinstance(fn["description"], str)
            assert isinstance(fn["parameters"], dict)

    _in_hook(app, rid, c, check)


def test_list_tools_includes_known_tools(env):
    c, app = env
    rid = _new_run(c)

    def check():
        status, out = _call_tool_via_act(c, rid, "agent_0", "list_tools")
        assert status == 200
        names = {r["function"]["name"] for r in out}
        assert "list_product" in names
        assert "adjust_price" in names
        assert "query_my_listings" in names
        assert "query_open_orders" in names
        assert "query_order_updates" in names
        assert "query_cash_pipeline" in names
        assert "list_tools" in names
        assert "get_observation" in names

    _in_hook(app, rid, c, check)


def test_listing_batch_schema_no_promised_ship_hours(env):
    """promised_ship_hours has been removed from list_product schema."""
    c, app = env
    scen = load_default_scenario()
    scen["run"]["horizon_steps"] = 3
    scen["run"]["max_hook_seconds"] = 2.0
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 30
    rid = _new_run_with_scenario(c, scen)

    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    tools = {t["name"]: t for t in schema.get("tools", [])}
    list_items = tools["list_product"]["parameters"]["properties"]["items"]
    item_props = list_items["items"]["properties"]

    assert list_items["maxItems"] == 100
    assert "promised_ship_hours" not in item_props
    assert "product_id" in item_props
    assert "sale_price" in item_props

    def check_list_tools():
        status, out = _call_tool_via_act(c, rid, "agent_0", "list_tools")
        assert status == 200
        specs = {row["function"]["name"]: row["function"] for row in out}
        list_params = specs["list_product"]["parameters"]
        list_items_schema = list_params["properties"]["items"]
        assert list_items_schema["maxItems"] == 100
        assert "promised_ship_hours" not in list_items_schema["items"]["properties"]

    _in_hook(app, rid, c, check_list_tools)


def test_query_my_orders_status_schema_uses_current_order_status_enum(env):
    c, _ = env
    rid = _new_run(c)
    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    tools = {t["name"]: t for t in schema.get("tools", [])}

    status_schema = tools["query_my_orders"]["parameters"]["properties"]["status"]

    assert status_schema["enum"] == list(get_args(OrderStatus))


def test_compact_order_tool_status_schemas(env):
    c, _ = env
    rid = _new_run(c)
    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    tools = {t["name"]: t for t in schema.get("tools", [])}

    open_statuses = tools["query_open_orders"]["parameters"]["properties"]["statuses"]
    update_statuses = tools["query_order_updates"]["parameters"]["properties"]["statuses"]

    assert open_statuses["items"]["enum"] == ["ordered", "late", "shipped", "delivered"]
    assert update_statuses["items"]["enum"] == list(get_args(OrderStatus))


def test_list_tools_appears_in_schema(env):
    """/tools/schema is NOT hook-gated (it's a registration-time fetch)."""
    c, _ = env
    rid = _new_run(c)
    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    names = {t["name"] for t in schema.get("tools", [])}
    assert "list_tools" in names


def test_list_tools_honors_denylist(env):
    c, app = env
    rid = _new_run(c, {"tool_denylist": ["adjust_price", "query_balance"]})

    def check():
        status, out = _call_tool_via_act(c, rid, "agent_0", "list_tools")
        assert status == 200
        names = {r["function"]["name"] for r in out}
        assert "adjust_price" not in names
        assert "query_balance" not in names
        assert len(names) > 0

    _in_hook(app, rid, c, check)


def test_list_tools_returns_425_outside_hook(env):
    """Calling list_tools outside the hook window returns 425."""
    c, _ = env
    rid = _new_run(c)
    tc = _make_tool_call("list_tools", {})
    body = _act_msg([tc])
    r = c.post(f"/runs/{rid}/agents/agent_0/act", json=body)
    assert r.status_code == 425
