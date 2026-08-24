"""End-to-end agent-route tests: register → schema → observation → act → cost.

Tests the unified /act endpoint that executes tools and stores traces."""

import json
import os
import sqlite3
import tempfile
import threading
import time

import pytest
from storage import agent_log
from storage import db as dbm
from web.app import create_app
from web.runner import load_default_scenario


def _table_records(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"), runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        yield c, tmp, app


def _agent_scenario(hook_seconds=0.1, **agent_overrides):
    s = load_default_scenario()
    s["run"]["max_hook_seconds"] = hook_seconds
    s["run"]["horizon_steps"] = 24
    s["data"]["source"] = "synthetic"
    s["data"]["num_products"] = 30
    s.setdefault("agent", {})["tool_denylist"] = []
    s.setdefault("agent", {}).update(agent_overrides)
    return s


def _wait_for_hook(env, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not env.hook_open and time.monotonic() < deadline:
        time.sleep(0.01)
    if not env.hook_open:
        raise RuntimeError(f"Hook window did not open within {timeout:g} seconds")


def _act(c, rid, agent_id, thought, tool_calls_spec, token_usage=None, assistant_extra=None, context=None):
    """Helper: build assistant msg and POST /act."""
    tc_list = []
    for i, (name, args) in enumerate(tool_calls_spec):
        tc_list.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    assistant_msg = {
        "role": "assistant",
        "content": thought,
        "tool_calls": tc_list,
    }
    if assistant_extra:
        assistant_msg.update(assistant_extra)
    body = {
        "messages": [assistant_msg],
        "token_usage": token_usage or {"input": 100, "output": 50, "total": 150},
    }
    if context is not None:
        body["context"] = context
    return c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)


def test_register_writes_meta(client):
    c, tmp, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    r = c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "framework": "test_agent", "model": "deterministic", "version": "1"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"]
    meta = c.get(f"/runs/{rid}/agent/meta").get_json()
    assert meta["agents"][0]["agent_id"] == "agent_0"
    assert meta["agents"][0]["framework"] == "test_agent"
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "framework": "test_agent", "model": "deterministic", "version": "2"},
    )
    meta2 = c.get(f"/runs/{rid}/agent/meta").get_json()
    assert len(meta2["agents"]) == 1
    assert meta2["agents"][0]["version"] == "2"


def test_register_requires_agent_id(client):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    r = c.post(f"/runs/{rid}/agent/register", json={"framework": "x"})
    assert r.status_code == 400


def test_tools_schema_endpoint(client):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    s = c.get(f"/runs/{rid}/tools/schema").get_json()
    names = {t["name"] for t in s["tools"]}
    assert "list_product" in names
    assert "get_observation" in names
    assert "get_store_snapshot" in names
    assert "query_supply_chain_anomalies" in names
    assert "query_store_performance" in names
    assert "query_product_sales_stats" in names
    assert "query_cash_pipeline" in names
    assert "read_memory_doc" in names
    assert "write_memory_doc" in names
    for t in s["tools"]:
        assert t["openai"]["type"] == "function"
        assert t["openai"]["function"]["name"] == t["name"]


def test_agent_token_is_limited_to_agent_api(client):
    c, _, app = client
    app.config["MERCHANTBENCH_REQUIRE_TOKENS"] = True
    app.config["MERCHANTBENCH_ADMIN_TOKEN"] = "admin-token"
    admin_headers = {"Authorization": "Bearer admin-token"}
    created = c.post(
        "/runs",
        json={"scenario": _agent_scenario()},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.get_data(as_text=True)
    body = created.get_json()
    rid = body["run_id"]
    agent_headers = {"Authorization": f"Bearer {body['agent_token']}"}

    schema = c.get(f"/runs/{rid}/tools/schema", headers=agent_headers)
    supplier = c.get(f"/runs/{rid}/sections/supplier", headers=agent_headers)
    step = c.post(f"/runs/{rid}/step", headers=agent_headers)
    anonymous = c.get(f"/runs/{rid}/sections/supplier")
    admin_merchant = c.get(
        f"/runs/{rid}/agents/agent_0/sections/merchant",
        headers=admin_headers,
    )

    assert schema.status_code == 200
    assert supplier.status_code == 403
    assert step.status_code == 403
    assert anonymous.status_code == 401
    assert admin_merchant.status_code == 200


def test_agent_token_cannot_act_as_another_agent(client):
    c, _, app = client
    app.config["MERCHANTBENCH_REQUIRE_TOKENS"] = True
    app.config["MERCHANTBENCH_ADMIN_TOKEN"] = "admin-token"
    admin_headers = {"Authorization": "Bearer admin-token"}
    created = c.post(
        "/runs",
        json={"scenario": _agent_scenario(hook_seconds=2)},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.get_data(as_text=True)
    body = created.get_json()
    rid = body["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    env = app.registry._require(rid)
    agent0_headers = {"Authorization": f"Bearer {body['agent_token']}"}
    act_body = {
        "messages": [
            {
                "role": "assistant",
                "content": "balance",
                "tool_calls": [
                    {
                        "id": "balance",
                        "type": "function",
                        "function": {"name": "query_balance", "arguments": json.dumps({})},
                    }
                ],
            }
        ]
    }
    usage_body = {
        "usage_id": "auth-scope",
        "step": 0,
        "token_usage": {"input": 1, "total": 1},
    }
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        own = c.post(
            f"/runs/{rid}/agents/agent_0/act",
            json=act_body,
            headers=agent0_headers,
        )
        other = c.post(
            f"/runs/{rid}/agents/agent_1/act",
            json=act_body,
            headers=agent0_headers,
        )
        other_register = c.post(
            f"/runs/{rid}/agent/register",
            json={"agent_id": "agent_1", "framework": "wrong-agent"},
            headers=agent0_headers,
        )
        own_usage = c.post(
            f"/runs/{rid}/agents/agent_0/usage",
            json=usage_body,
            headers=agent0_headers,
        )
        other_usage = c.post(
            f"/runs/{rid}/agents/agent_1/usage",
            json=usage_body,
            headers=agent0_headers,
        )
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert own.status_code == 200, own.get_data(as_text=True)
    assert other.status_code == 403
    assert other_register.status_code == 403
    assert own_usage.status_code == 200
    assert other_usage.status_code == 403


def test_tool_denylist_filters_schema(client):
    c, _, _ = client
    rid = c.post(
        "/runs", json={"scenario": _agent_scenario(tool_denylist=["adjust_price", "query_balance"])}
    ).get_json()["run_id"]
    s = c.get(f"/runs/{rid}/tools/schema").get_json()
    names = {t["name"] for t in s["tools"]}
    assert "adjust_price" not in names
    assert "query_balance" not in names


def test_observation_has_new_slim_shape(client):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1").get_json()
    assert "text" in obs
    assert "tick" in obs
    assert "brief" in obs
    assert "agent_id" not in obs
    assert "demand_multiplier" not in obs.get("text", "")


def test_duplicate_observation_fetch_in_same_step_keeps_change_counts(client):
    from core.entities import EventLog, StoreListing
    from storage import db as dbm

    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price * 1.4,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    env.t = 12
    dbm.write_events(
        env.conn,
        env.run_id,
        [
            EventLog(
                t=0,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id=None,
                payload={"old_price": 10.0, "new_price": 12.0},
            ),
            EventLog(
                t=11,
                event_type="supplier_timeout",
                entity_id=product.product_id,
                agent_id=None,
                payload={"recover_t": 20},
            ),
        ],
    )

    first = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1").get_json()
    second = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1").get_json()

    assert first["text"] == second["text"]
    assert "events since last observation: price_changes 1" in first["text"]
    assert "timeouts 1" in first["text"]


def test_nowait_outside_hook_does_not_consume_observation_window(client):
    from core.entities import EventLog, StoreListing
    from storage import db as dbm

    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price * 1.4,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    env.t = 6
    dbm.write_events(
        env.conn,
        env.run_id,
        [
            EventLog(
                t=0,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id=None,
                payload={"old_price": 10.0, "new_price": 12.0},
            ),
        ],
    )

    outside_hook = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert outside_hook.status_code == 200

    env.t = 12
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        real_hook = c.get(f"/runs/{rid}/agents/agent_0/observation?timeout=1")
    finally:
        with env.hook_cond:
            env.hook_open = False

    assert real_hook.status_code == 200
    assert "events since last observation: price_changes 1" in real_hook.get_json()["text"]


def test_observation_window_survives_rehydrate(client):
    from core.entities import EventLog, StoreListing
    from storage import db as dbm

    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    listing = StoreListing(
        product_id=product.product_id,
        agent_id="agent_0",
        sale_price=product.price * 1.4,
        listed_at=0,
    )
    dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
    env.agents["agent_0"].listings[product.product_id] = listing
    dbm.write_events(
        env.conn,
        env.run_id,
        [
            EventLog(
                t=0,
                event_type="price_change",
                entity_id=product.product_id,
                agent_id=None,
                payload={"old_price": 10.0, "new_price": 12.0},
            ),
        ],
    )

    env.t = 12
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    first = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert first.status_code == 200
    assert "events since last observation: price_changes 1" in first.get_json()["text"]

    env.t = 24
    second = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert second.status_code == 200
    assert "events since last observation: price_changes 0" in second.get_json()["text"]
    with env.hook_cond:
        env.hook_open = False

    dbm.update_run_t(app.registry.conn_for(rid), rid, 24)
    app.registry.envs.pop(rid, None)

    rehydrated = app.registry._require(rid)
    third = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")

    assert rehydrated is not env
    assert third.status_code == 200
    assert "events since last observation: price_changes 0" in third.get_json()["text"]


def test_observation_unknown_agent_returns_404(client):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    r = c.get(f"/runs/{rid}/agents/ghost/observation")
    assert r.status_code == 404


def test_observation_reports_existing_turn_count_for_same_step(client):
    c, _, app = client
    rid = c.post(
        "/runs",
        json={
            "scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=3),
        },
    ).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        first = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
        assert first.status_code == 200
        assert first.get_json()["turn_count"] == 0

        acted = _act(
            c,
            rid,
            "agent_0",
            "inspect balance",
            [
                ("query_balance", {}),
            ],
        )
        assert acted.status_code == 200
        assert acted.get_json()["turn_idx"] == 0

        resumed = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
        assert resumed.status_code == 200
        assert resumed.get_json()["tick"]["step"] == first.get_json()["tick"]["step"]
        assert resumed.get_json()["turn_count"] == 1
    finally:
        with env.hook_cond:
            env.hook_open = False


def test_act_executes_tools_and_stores_trace(client):
    """POST /act executes tool_calls and writes by_step."""
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    # Fetch observation first (stores as user msg in by_step)
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200
    # First: call market_brief via /act
    r = _act(c, rid, "agent_0", "let me check market", [("market_brief", {"window_days": 7})])
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"]
    assert body["turn_idx"] == 0
    assert not body["step_done"]
    assert len(body["tool_results"]) == 1
    brief = json.loads(body["tool_results"][0]["content"])
    assert isinstance(brief["categories"], list) and len(brief["categories"]) > 0
    # End the step
    r2 = _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    assert r2.get_json()["step_done"]
    th.join(timeout=3)
    # Verify by_step file was written
    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    assert os.path.exists(by_step)
    payload = json.load(open(by_step))
    assert payload["n_turns"] == 2
    assert "observation" not in payload  # observation lives in messages, not top-level
    assert len(payload["messages"]) >= 6  # system + user + assistant + tool + assistant + tool
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][2]["role"] == "assistant"
    # No trace.jsonl or trajectory/ should exist
    assert not os.path.exists(os.path.join(tmp, "runs", rid, "agent", "trace.jsonl"))
    assert not os.path.exists(os.path.join(tmp, "runs", rid, "trajectory"))


def test_act_records_user_messages_before_assistant(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    reminder = {
        "role": "user",
        "content": "[context-maintenance]\nThe conversation history is about to be compacted.",
    }
    assistant_msg = {
        "role": "assistant",
        "content": "saving memory",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "write_memory_doc",
                    "arguments": json.dumps({"content": "state before compaction"}),
                },
            }
        ],
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={"messages": [reminder, assistant_msg]},
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    roles = [m["role"] for m in payload["messages"][:5]]
    assert roles == ["system", "user", "user", "assistant", "tool"]
    assert payload["messages"][2]["role"] == reminder["role"]
    assert payload["messages"][2]["content"] == reminder["content"]
    assert "agent_id" not in payload["messages"][2]
    assert payload["message_agents"][2] == "agent_0"


def test_act_records_context_on_turn_and_trace_index(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    r = _act(
        c,
        rid,
        "agent_0",
        "context tracked",
        [("market_brief", {"window_days": 7})],
        context={"tokens": 123456, "compacted": True, "ignored": "drop"},
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    assert payload["turns"][0]["context"] == {
        "tokens": 123456,
        "compacted": True,
    }
    assert "context" not in payload["turns"][1]

    idx = c.get(f"/runs/{rid}/agent/all_traces_index").get_json()
    assert idx["steps"][0]["context"] == {
        "tokens": 123456,
        "compactions": 1,
    }


def test_act_records_system_messages_before_assistant(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [
                {"role": "system", "content": "agent-supplied system context"},
                {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "end_of_step", "arguments": "{}"},
                        }
                    ],
                },
            ]
        },
    )

    assert r.status_code == 200, r.get_data(as_text=True)
    th.join(timeout=3)
    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    roles = [m["role"] for m in payload["messages"][:4]]
    assert roles == ["system", "user", "system", "assistant"]


def test_act_rejects_non_context_messages_before_assistant(client):
    c, _tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [
                {"role": "tool", "content": "not valid before assistant"},
                {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "end_of_step", "arguments": "{}"},
                        }
                    ],
                },
            ]
        },
    )

    assert r.status_code == 400
    assert r.get_json()["error"] == "trace tool messages must include tool_origin"
    th.join(timeout=3)


def test_end_of_step_trace_recorded_before_hook_release(client, monkeypatch):
    """end_of_step must not wake the step thread before assistant/tool messages land."""
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)

    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    original_record_act = env.record_act

    def slow_record_act(*args, **kwargs):
        time.sleep(0.2)
        return original_record_act(*args, **kwargs)

    monkeypatch.setattr(env, "record_act", slow_record_act)

    r = _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    assert r.status_code == 200, r.get_data(as_text=True)
    th.join(timeout=3)

    by_step_0 = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    assert os.path.exists(by_step_0)
    payload_0 = json.load(open(by_step_0))
    roles_0 = [m["role"] for m in payload_0["messages"]]
    assert roles_0 == ["system", "user", "assistant", "tool"]
    assert payload_0["n_turns"] == 1

    by_step_1 = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00001.json")
    if os.path.exists(by_step_1):
        roles_1 = [m["role"] for m in json.load(open(by_step_1))["messages"]]
        assert "assistant" not in roles_1
        assert "tool" not in roles_1


def test_act_rechecks_stale_step_after_initial_validation(client, monkeypatch):
    """A step change after the early stale check must still reject before tools mutate."""
    c, _tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    env.hook_open = True
    env.t = 0
    product = next(iter(env.products.values()))

    import web.routes_agent as routes_agent

    original_check = routes_agent.check_stale_step
    calls = {"n": 0}

    def advance_after_first_check(env_arg):
        calls["n"] += 1
        if calls["n"] == 1:
            env_arg.t = 1
            return None
        return original_check(env_arg)

    monkeypatch.setattr(routes_agent, "check_stale_step", advance_after_first_check)

    msg = {
        "role": "assistant",
        "content": "list old decision",
        "tool_calls": [
            {
                "id": "call_list",
                "type": "function",
                "function": {
                    "name": "list_product",
                    "arguments": json.dumps(
                        {
                            "product_id": product.product_id,
                            "sale_price": round(product.price * 1.2, 2),
                        }
                    ),
                },
            }
        ],
    }
    resp = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={"messages": [msg]},
        headers={"X-Agent-Step": "0"},
    )

    assert resp.status_code == 425
    assert resp.get_json()["error"] == "stale_step"
    assert product.product_id not in env.agents["agent_0"].listings


def test_hook_cannot_close_while_act_dispatch_holds_env_lock(client, monkeypatch):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=0.1)}).get_json()["run_id"]
    env = app.registry._require(rid)

    import threading

    import web.routes_agent as routes_agent

    original_dispatch = routes_agent.dispatch_tool
    seen_hook_open_after_timeout = {"value": None}

    def slow_dispatch(*args, **kwargs):
        time.sleep(0.25)
        seen_hook_open_after_timeout["value"] = env.hook_open
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(routes_agent, "dispatch_tool", slow_dispatch)

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    r = _act(c, rid, "agent_0", "slow read", [("market_brief", {"window_days": 7})])
    assert r.status_code == 200, r.get_data(as_text=True)
    th.join(timeout=3)

    assert seen_hook_open_after_timeout["value"] is True
    assert not env.hook_open


def test_delete_run_waits_for_in_flight_act_request(client, monkeypatch):
    from web import routes_agent

    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()

    original_dispatch = routes_agent.dispatch_tool
    dispatch_entered = threading.Event()
    release_dispatch = threading.Event()

    def slow_dispatch(*args, **kwargs):
        dispatch_entered.set()
        release_dispatch.wait(timeout=2)
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(routes_agent, "dispatch_tool", slow_dispatch)
    act_response = {}

    def post_act():
        with app.test_client() as act_client:
            act_response["response"] = _act(
                act_client,
                rid,
                "agent_0",
                "checking market",
                [("market_brief", {"window_days": 7})],
            )

    act_thread = threading.Thread(target=post_act)
    act_thread.start()
    assert dispatch_entered.wait(timeout=2)

    delete_done = threading.Event()
    delete_result = {}

    def delete_run():
        delete_result.update(app.registry.delete_run(rid))
        delete_done.set()

    delete_thread = threading.Thread(target=delete_run)
    delete_thread.start()
    try:
        assert not delete_done.wait(timeout=0.2)
    finally:
        release_dispatch.set()
        act_thread.join(timeout=3)
        delete_thread.join(timeout=3)

    assert delete_done.is_set()
    assert delete_result["deleted"] is True
    assert act_response["response"].status_code == 200


def test_human_bootstrap_uses_same_act_trace_protocol(client):
    c, tmp, app = client
    rid = c.post(
        "/runs",
        json={
            "scenario": _agent_scenario(hook_seconds=2),
            "bootstrap_agent": "human",
        },
    ).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    listings = _act(
        c,
        rid,
        "agent_0",
        "[human] call query_my_listings",
        [("query_my_listings", {})],
        token_usage={"input": 0, "output": 0, "cache_read": 0, "total": 0},
    )
    assert listings.status_code == 200, listings.get_data(as_text=True)
    body = listings.get_json()
    assert body["ok"]
    assert body["tool_results"][0]["name"] == "query_my_listings"

    done = _act(
        c,
        rid,
        "agent_0",
        "[human] call end_of_step",
        [("end_of_step", {})],
        token_usage={"input": 0, "output": 0, "cache_read": 0, "total": 0},
    )
    assert done.status_code == 200, done.get_data(as_text=True)
    assert done.get_json()["step_done"] is True
    th.join(timeout=3)

    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    assistant_messages = [m for m in payload["messages"] if m["role"] == "assistant"]
    assert assistant_messages[0]["content"] == "[human] call query_my_listings"
    assert assistant_messages[1]["content"] == "[human] call end_of_step"
    assert payload["n_turns"] == 2


def test_human_auto_refresh_batches_safe_operational_tools_in_one_turn(client):
    c, tmp, app = client
    rid = c.post(
        "/runs",
        json={
            "scenario": _agent_scenario(hook_seconds=3),
            "bootstrap_agent": "human",
        },
    ).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    auto_calls = [
        ("get_store_snapshot", {}),
        (
            "query_order_updates",
            {
                "include_ordered": True,
                "page": 1,
                "page_size": 100,
            },
        ),
        ("query_open_orders", {"page": 1, "page_size": 50}),
        ("query_supply_chain_anomalies", {"mode": "new"}),
        ("query_supply_chain_anomalies", {"mode": "now"}),
        ("read_memory_doc", {}),
    ]
    refreshed = _act(
        c,
        rid,
        "agent_0",
        "[human:auto] refresh operational state",
        auto_calls,
        token_usage={"input": 0, "output": 0, "cached": 0, "total": 0},
    )

    assert refreshed.status_code == 200, refreshed.get_data(as_text=True)
    body = refreshed.get_json()
    assert body["turn_idx"] == 0
    assert body["step_done"] is False
    assert [result["name"] for result in body["tool_results"]] == [name for name, _args in auto_calls]

    done = _act(
        c,
        rid,
        "agent_0",
        "[human] end current step",
        [("end_of_step", {})],
        token_usage={"input": 0, "output": 0, "cached": 0, "total": 0},
    )
    assert done.status_code == 200
    assert done.get_json()["turn_idx"] == 1
    assert done.get_json()["step_done"] is True
    th.join(timeout=3)

    trace_path = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(trace_path))
    assert payload["n_turns"] == 2
    auto_message = next(
        message for message in payload["messages"] if message.get("content") == "[human:auto] refresh operational state"
    )
    assert [call["function"]["name"] for call in auto_message["tool_calls"]] == [name for name, _args in auto_calls]
    stored_tool_names = [message.get("name") for message in payload["messages"] if message.get("role") == "tool"]
    for name, _args in auto_calls:
        assert name in stored_tool_names


def test_act_preserves_reasoning_content_extension_in_trace(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    obs = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    assert obs.status_code == 200

    r = _act(
        c,
        rid,
        "agent_0",
        "done",
        [("end_of_step", {})],
        assistant_extra={"reasoning_content": "model thought trace"},
    )

    assert r.status_code == 200, r.get_data(as_text=True)
    th.join(timeout=3)
    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "model thought trace"


def test_act_writes_cost(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    _act(
        c,
        rid,
        "agent_0",
        "noop",
        [("end_of_step", {})],
        token_usage={"input": 75, "output": 50, "cache_read": 25, "total": 150},
    )
    th.join(timeout=3)
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    assert cost["total"]["input"] == 75
    assert cost["total"]["output"] == 50
    assert cost["total"]["cache_read"] == 25
    assert "cached" not in cost["total"]
    assert cost["total"]["turns"] == 1


def test_act_writes_cache_read_write_and_reasoning_cost(client):
    c, tmp, app = client
    scenario = _agent_scenario(
        hook_seconds=2,
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
            "cached_input_per_million": 100.0,
            "cache_write_input_per_million": 5000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    _act(
        c,
        rid,
        "agent_0",
        "noop",
        [("end_of_step", {})],
        token_usage={
            "input": 500,
            "output": 200,
            "cache_read": 300,
            "cache_write": 100,
            "reasoning": 50,
            "total": 1100,
        },
    )
    th.join(timeout=3)

    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    step = cost["by_step"]["0"]
    assert step["input"] == 500
    assert step["output"] == 200
    assert step["cache_read"] == 300
    assert step["cache_write"] == 100
    assert step["reasoning"] == 50
    assert "cached" not in step
    assert step["total"] == 1100
    assert step["usd"] == 1.43
    payload = json.load(open(os.path.join(tmp, "runs", rid, "agent", "cost.json")))
    assert payload["total"]["cache_read"] == 300
    assert payload["total"]["cache_write"] == 100
    assert payload["total"]["reasoning"] == 50


def test_delayed_checkpoint_review_usage_is_idempotent_and_billed(client):
    c, _, app = client
    scenario = _agent_scenario(
        hook_seconds=2,
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
            "cached_input_per_million": 100.0,
            "cache_write_input_per_million": 5000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "foreground-model"},
    )

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    _act(
        c,
        rid,
        "agent_0",
        "done",
        [("end_of_step", {})],
        token_usage={"input": 100, "output": 20, "total": 120},
    )
    th.join(timeout=3)

    body = {
        "usage_id": "review-10",
        "source": "checkpoint_review",
        "step": 0,
        "model": "foreground-model",
        "token_usage": {
            "input": 700,
            "output": 40,
            "cache_read": 200,
            "cache_write": 25,
            "reasoning": 10,
            "total": 975,
        },
    }
    first = c.post(f"/runs/{rid}/agents/agent_0/usage", json=body)
    duplicate = c.post(f"/runs/{rid}/agents/agent_0/usage", json=body)
    conflicting_body = dict(body)
    conflicting_body["token_usage"] = {"input": 1, "total": 1}
    conflict = c.post(f"/runs/{rid}/agents/agent_0/usage", json=conflicting_body)

    assert first.status_code == 200
    assert first.get_json()["recorded"] is True
    assert duplicate.get_json()["duplicate"] is True
    assert conflict.status_code == 409
    assert conflict.get_json()["error"] == "usage_id_conflict"
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    assert cost["total"]["input"] == 800
    assert cost["total"]["output"] == 60
    assert cost["total"]["cache_read"] == 200
    assert cost["total"]["cache_write"] == 25
    assert cost["total"]["total"] == 1095
    assert cost["total"]["turns"] == 1
    assert cost["total"]["usd"] == 1.085
    review_entry = cost["auxiliary"]["entries"]["agent_0:review-10"]
    assert review_entry["source"] == ("checkpoint_review")
    assert review_entry["reasoning_billed_separately"] is True


def test_reasoning_already_in_output_is_not_double_billed(client):
    c, _, _ = client
    scenario = _agent_scenario(
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "foreground-model"},
    )

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "reasoning-in-output",
            "step": 0,
            "model": "foreground-model",
            "token_usage": {
                "input": 100,
                "output": 20,
                "reasoning": 10,
                "total": 120,
            },
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:reasoning-in-output"]
    assert entry["reasoning_billed_separately"] is False
    assert entry["usd"] == 0.14


def test_auxiliary_model_uses_its_reported_cost_identity(client):
    c, _, _ = client
    scenario = _agent_scenario(
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "foreground-model"},
    )

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "routed-review",
            "source": "checkpoint_review",
            "step": 0,
            "model": "review-model",
            "provider": "review-provider",
            "cost_usd": 0.0123456,
            "cost_status": "known",
            "cost_source": "provider_response",
            "token_usage": {"input": 1000, "output": 100, "total": 1100},
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:routed-review"]
    assert entry["model"] == "review-model"
    assert entry["provider"] == "review-provider"
    assert entry["cost_status"] == "known"
    assert entry["cost_source"] == "provider_response"
    assert entry["pricing_mode"] == "reported_auxiliary_model"
    assert entry["usd"] == 0.012346
    assert cost["total"]["usd"] == 0.012346


def test_unpriced_auxiliary_model_keeps_tokens_without_false_usd(client):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "foreground-model"},
    )

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "unknown-price",
            "source": "compression",
            "step": 0,
            "model": "another-model",
            "cost_status": "unknown",
            "token_usage": {"input": 12, "output": 3, "total": 15},
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:unknown-price"]
    assert entry["pricing_mode"] == "unpriced_auxiliary_model"
    assert entry["usd"] is None
    assert cost["total"]["total"] == 15
    assert cost["total"]["usd"] == 0.0
    assert cost["total"]["unpriced_auxiliary"] == 1


def test_model_identity_without_registration_does_not_assume_foreground_rate(
    client,
):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "known-model-no-registration",
            "step": 0,
            "model": "review-model",
            "cost_usd": 0.25,
            "token_usage": {"input": 12, "output": 3, "total": 15},
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:known-model-no-registration"]
    assert entry["pricing_mode"] == "reported_auxiliary_model"
    assert entry["usd"] == 0.25


def test_missing_model_identity_uses_reported_auxiliary_cost(client):
    c, _, _ = client
    scenario = _agent_scenario(
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "reported-cost-without-model",
            "step": 0,
            "cost_usd": 0.25,
            "token_usage": {"input": 12, "output": 3, "total": 15},
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:reported-cost-without-model"]
    assert entry["pricing_mode"] == "reported_auxiliary_model"
    assert entry["usd"] == 0.25


def test_same_model_auxiliary_uses_frozen_scenario_pricing(client):
    c, _, _ = client
    scenario = _agent_scenario(
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "shared-model"},
    )

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "same-model",
            "step": 0,
            "model": "shared-model",
            "cost_usd": 999.0,
            "token_usage": {"input": 1000, "output": 100, "total": 1100},
        },
    )

    assert response.status_code == 200
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    entry = cost["auxiliary"]["entries"]["agent_0:same-model"]
    assert entry["pricing_mode"] == "scenario_foreground"
    assert entry["usd"] == 1.2
    assert cost["total"]["usd"] == 1.2


@pytest.mark.parametrize(
    "token_usage",
    [
        {"input": 10, "output": 2, "total": 99},
        {"input": -1, "output": 2, "total": 1},
        {"input": 1.5, "output": 2, "total": 3},
        {"input": "1", "output": 2, "total": 3},
        {"prompt_tokens": 1, "output": 2, "total": 3},
        {"input": 1, "cached": 2, "output": 0, "total": 3},
    ],
)
def test_auxiliary_usage_rejects_invalid_tokens_without_mutation(client, token_usage):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "bad-usage",
            "step": 0,
            "token_usage": token_usage,
        },
    )

    assert response.status_code == 400
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    assert "auxiliary" not in cost
    assert cost["total"]["total"] == 0


@pytest.mark.parametrize("payload", [[], "usage", 1, None])
def test_auxiliary_usage_rejects_non_object_body(client, payload):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json=payload,
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "ok": False,
        "error": "request body must be an object",
    }


@pytest.mark.parametrize("cost_usd", [-1, "nan", "not-a-number"])
def test_auxiliary_usage_rejects_invalid_cost_without_mutation(client, cost_usd):
    c, _, _ = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    c.post(
        f"/runs/{rid}/agent/register",
        json={"agent_id": "agent_0", "model": "foreground-model"},
    )

    response = c.post(
        f"/runs/{rid}/agents/agent_0/usage",
        json={
            "usage_id": "bad-cost",
            "step": 0,
            "model": "another-model",
            "cost_usd": cost_usd,
            "token_usage": {"input": 1, "output": 2, "total": 3},
        },
    )

    assert response.status_code == 400
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    assert "auxiliary" not in cost
    assert cost["total"]["total"] == 0


def test_cost_update_replaces_existing_step_in_total(tmp_path):
    from storage import agent_log

    turns = [
        {
            "token_usage": {
                "input": 13,
                "output": 5,
                "cache_read": 2,
                "total": 20,
            }
        }
    ]
    for _ in range(2):
        agent_log.update_cost(
            str(tmp_path),
            "run-cost-idempotent",
            7,
            turns,
            env_step_ms=9,
        )

    cost = agent_log.read_cost(str(tmp_path), "run-cost-idempotent")
    assert cost["by_step"]["7"]["total"] == 20
    assert cost["total"]["input"] == 13
    assert cost["total"]["output"] == 5
    assert cost["total"]["cache_read"] == 2
    assert cost["total"]["total"] == 20
    assert cost["total"]["turns"] == 1
    assert cost["total"]["env_step_ms"] == 9


def test_cost_recompute_preserves_delayed_auxiliary_usage(tmp_path):
    from storage import agent_log

    turns = [{"token_usage": {"input": 10, "output": 2, "total": 12}}]
    pricing = {"input_per_million": 1000.0, "output_per_million": 2000.0}
    agent_log.record_auxiliary_usage(
        str(tmp_path),
        "run-aux-recompute",
        "agent_0",
        "agent_0:review-1",
        {"input": 30, "output": 4, "total": 34},
        t=0,
        source="checkpoint_review",
        pricing=pricing,
    )
    agent_log.update_cost(
        str(tmp_path),
        "run-aux-recompute",
        0,
        turns,
        pricing=pricing,
    )

    cost = agent_log.read_cost(str(tmp_path), "run-aux-recompute")
    assert cost["by_step"]["0"]["input"] == 40
    assert cost["by_step"]["0"]["output"] == 6
    assert cost["by_step"]["0"]["total"] == 46
    assert cost["total"]["total"] == 46
    assert cost["total"]["turns"] == 1
    assert cost["total"]["usd"] == 0.052


def test_act_bills_cached_tokens_with_cached_input_rate(client):
    c, tmp, app = client
    scenario = _agent_scenario(
        hook_seconds=2,
        cost_pricing={
            "input_per_million": 1000.0,
            "output_per_million": 2000.0,
            "cached_input_per_million": 100.0,
        },
    )
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))

    _act(
        c,
        rid,
        "agent_0",
        "noop",
        [("end_of_step", {})],
        token_usage={"input": 1000, "output": 2000, "cached": 400, "total": 3000},
    )

    th.join(timeout=3)
    cost = c.get(f"/runs/{rid}/agent/cost").get_json()
    assert cost["by_step"]["0"]["usd"] == 4.64
    assert cost["total"]["usd"] == 4.64
    payload = json.load(open(os.path.join(tmp, "runs", rid, "agent", "cost.json")))
    assert payload["by_step"]["0"]["input"] == 600
    assert payload["by_step"]["0"]["cache_read"] == 400
    assert "cached" not in payload["by_step"]["0"]


def test_act_accepts_trace_only_hermes_native_batch(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    native_assistant = {
        "role": "assistant",
        "content": "I will inspect local context first.",
        "tool_origin": "hermes_native",
        "tool_calls": [
            {
                "id": "call_native_0",
                "type": "function",
                "tool_origin": "hermes_native",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "pwd"}),
                },
            }
        ],
    }
    native_tool = {
        "role": "tool",
        "tool_call_id": "call_native_0",
        "name": "terminal",
        "tool_origin": "hermes_native",
        "content": "/tmp/workspace",
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [native_assistant, native_tool],
            "token_usage": {"input": 1, "output": 1, "total": 2},
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["tool_results"] == []
    assert body["step_done"] is False
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    native_msgs = [m for m in trace["messages"] if m.get("tool_origin") == "hermes_native"]
    assert [m["role"] for m in native_msgs] == ["assistant", "tool"]
    assert native_msgs[1]["name"] == "terminal"
    assert trace["turns"][0]["message_origins"] == {"hermes_native": 2}
    assert trace["turns"][0]["tool_call_origins"] == {"hermes_native": 1}
    assert trace["turns"][0]["tool_result_origins"] == {"hermes_native": 1}


def test_act_hermes_native_error_message_is_not_counted_as_native_tool_call(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    native_error = {
        "role": "assistant",
        "content": "HTTP_STATUS/429 Throttling.AllocationQuota",
        "tool_origin": "hermes_native",
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [native_error],
            "token_usage": {"input": 1, "output": 1, "total": 2},
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["tool_results"] == []
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    assert trace["turns"][0]["message_origins"] == {"hermes_native": 1}
    assert trace["turns"][0]["tool_call_origins"] == {}
    assert trace["turns"][0]["tool_result_origins"] == {}


def test_act_infers_native_assistant_origin_from_tool_calls(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    native_assistant = {
        "role": "assistant",
        "content": "I will inspect local context first.",
        "tool_calls": [
            {
                "id": "call_native_0",
                "type": "function",
                "tool_origin": "hermes_native",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "pwd"}),
                },
            }
        ],
    }
    native_tool = {
        "role": "tool",
        "tool_call_id": "call_native_0",
        "name": "terminal",
        "tool_origin": "hermes_native",
        "content": "/tmp/workspace",
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [native_assistant, native_tool],
            "token_usage": {"input": 1, "output": 1, "total": 2},
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["tool_results"] == []
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    native_msgs = [m for m in trace["messages"] if m.get("tool_origin") == "hermes_native"]
    assert [m["role"] for m in native_msgs] == ["assistant", "tool"]
    assert "merchantbench_env" not in trace["turns"][0]["tool_origins"]


def test_act_executes_merchantbench_tool_even_when_not_final_message(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    env_assistant = {
        "role": "assistant",
        "content": "Check balance first.",
        "tool_origin": "merchantbench_env",
        "tool_calls": [
            {
                "id": "call_env_0",
                "type": "function",
                "tool_origin": "merchantbench_env",
                "function": {"name": "query_balance", "arguments": "{}"},
            }
        ],
    }
    native_assistant = {
        "role": "assistant",
        "content": "Then inspect local context.",
        "tool_origin": "hermes_native",
        "tool_calls": [
            {
                "id": "call_native_0",
                "type": "function",
                "tool_origin": "hermes_native",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "pwd"}),
                },
            }
        ],
    }
    native_tool = {
        "role": "tool",
        "tool_call_id": "call_native_0",
        "name": "terminal",
        "tool_origin": "hermes_native",
        "content": "/tmp/workspace",
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [env_assistant, native_assistant, native_tool],
            "token_usage": {"input": 1, "output": 1, "total": 2},
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert len(body["tool_results"]) == 1
    assert body["tool_results"][0]["name"] == "query_balance"

    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    env_tool = next(
        m for m in trace["messages"] if m.get("tool_origin") == "merchantbench_env" and m.get("name") == "query_balance"
    )
    native_tool = next(
        m for m in trace["messages"] if m.get("tool_origin") == "hermes_native" and m.get("name") == "terminal"
    )
    assert env_tool["tool_call_id"] == "call_env_0"
    assert native_tool["tool_call_id"] == "call_native_0"


def test_act_does_not_replay_env_tool_calls_that_already_have_results(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    historical_assistant = {
        "role": "assistant",
        "content": "Earlier I listed this product.",
        "tool_origin": "merchantbench_env",
        "tool_calls": [
            {
                "id": "call_env_old",
                "type": "function",
                "tool_origin": "merchantbench_env",
                "function": {
                    "name": "list_product",
                    "arguments": json.dumps(
                        {
                            "items": [
                                {
                                    "product_id": product.product_id,
                                    "sale_price": product.price * 1.3,
                                }
                            ],
                        }
                    ),
                },
            }
        ],
    }
    historical_tool = {
        "role": "tool",
        "tool_call_id": "call_env_old",
        "name": "list_product",
        "tool_origin": "merchantbench_env",
        "content": json.dumps({"ok": True}),
    }
    current_assistant = {
        "role": "assistant",
        "content": "Now check balance.",
        "tool_origin": "merchantbench_env",
        "tool_calls": [
            {
                "id": "call_env_current",
                "type": "function",
                "tool_origin": "merchantbench_env",
                "function": {"name": "query_balance", "arguments": "{}"},
            }
        ],
    }
    r = c.post(
        f"/runs/{rid}/agents/agent_0/act",
        json={
            "messages": [
                historical_assistant,
                historical_tool,
                current_assistant,
            ],
            "token_usage": {"input": 1, "output": 1, "total": 2},
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert [tr["name"] for tr in body["tool_results"]] == ["query_balance"]
    assert product.product_id not in env.agents["agent_0"].listings
    live_trace = agent_log.read_step_index(app.registry.runs_root, rid, 0)
    persisted_historical = next(msg for msg in live_trace["messages"] if msg.get("tool_call_id") == "call_env_old")
    persisted_execution = next(
        msg
        for msg in live_trace["messages"]
        if msg.get("tool_call_id") == "call_env_current" and msg.get("role") == "tool"
    )
    assert persisted_historical["runtime_historical"] is True
    assert persisted_execution["runtime_execution_id"].startswith("agent_0:0:0:call_env_current")

    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)


def test_act_idempotency(client):
    """Mutating tools are replayed only for the same scoped call and args."""
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    r = _act(c, rid, "agent_0", "market", [("market_brief", {"window_days": 7})])
    [row["category"] for row in json.loads(r.get_json()["tool_results"][0]["content"])["categories"]]
    r2 = _act(c, rid, "agent_0", "search cat", [("search_products", {"query": "", "page": 1, "page_size": 3})])
    products = _table_records(json.loads(r2.get_json()["tool_results"][0]["content"])["items"])
    pid = products[0]["product_id"]
    price = products[0]["price"]
    # List product with a specific call_id (same id = idempotent)
    body1 = {
        "messages": [
            {
                "role": "assistant",
                "content": "list",
                "tool_calls": [
                    {
                        "id": "idem_test_1",
                        "type": "function",
                        "function": {
                            "name": "list_product",
                            "arguments": json.dumps({"items": [{"product_id": pid, "sale_price": price * 1.3}]}),
                        },
                    }
                ],
            }
        ],
    }
    r3 = c.post(f"/runs/{rid}/agents/agent_0/act", json=body1)
    assert r3.status_code == 200
    res3 = json.loads(r3.get_json()["tool_results"][0]["content"])
    assert res3["ok"]
    # Same call_id and same arguments -> cached result.
    r4 = c.post(f"/runs/{rid}/agents/agent_0/act", json=body1)
    res4 = json.loads(r4.get_json()["tool_results"][0]["content"])
    assert res4.get("_idempotent_replay") is True
    # Same scoped call_id with different arguments is a protocol conflict.
    body2 = {
        "messages": [
            {
                "role": "assistant",
                "content": "list again",
                "tool_calls": [
                    {
                        "id": "idem_test_1",
                        "type": "function",
                        "function": {
                            "name": "list_product",
                            "arguments": json.dumps({"items": [{"product_id": pid, "sale_price": price * 5.0}]}),
                        },
                    }
                ],
            }
        ],
    }
    r5 = c.post(f"/runs/{rid}/agents/agent_0/act", json=body2)
    assert r5.status_code == 409
    assert r5.get_json()["error"] == "idempotency_conflict"
    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)


def test_act_idempotency_conflict_prevents_prior_batch_side_effects(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    products = list(env.products.values())
    first = products[0]
    second = products[1]
    first_args = {"items": [{"product_id": first.product_id, "sale_price": first.price * 1.3}]}
    second_args = {"items": [{"product_id": second.product_id, "sale_price": second.price * 1.3}]}
    conflicting_args = {"items": [{"product_id": first.product_id, "sale_price": first.price * 2.0}]}
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        seed_conflict = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "seed",
                    "tool_calls": [
                        {
                            "id": "conflicting-call",
                            "type": "function",
                            "function": {"name": "list_product", "arguments": json.dumps(first_args)},
                        }
                    ],
                }
            ]
        }
        seeded = c.post(f"/runs/{rid}/agents/agent_0/act", json=seed_conflict)
        assert seeded.status_code == 200, seeded.get_data(as_text=True)

        batch = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "batch",
                    "tool_calls": [
                        {
                            "id": "new-call-before-conflict",
                            "type": "function",
                            "function": {"name": "list_product", "arguments": json.dumps(second_args)},
                        },
                        {
                            "id": "conflicting-call",
                            "type": "function",
                            "function": {"name": "list_product", "arguments": json.dumps(conflicting_args)},
                        },
                    ],
                }
            ]
        }
        conflict = c.post(f"/runs/{rid}/agents/agent_0/act", json=batch)
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert conflict.status_code == 409
    assert first.product_id in env.agents["agent_0"].listings
    assert second.product_id not in env.agents["agent_0"].listings


def test_act_idempotency_is_scoped_by_agent(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    product = next(iter(env.products.values()))
    args = {"items": [{"product_id": product.product_id, "sale_price": product.price * 1.3}]}
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "list",
                    "tool_calls": [
                        {
                            "id": "shared-call-id",
                            "type": "function",
                            "function": {"name": "list_product", "arguments": json.dumps(args)},
                        }
                    ],
                }
            ]
        }
        r0 = c.post(f"/runs/{rid}/agents/agent_0/act", json=body)
        r1 = c.post(f"/runs/{rid}/agents/agent_1/act", json=body)
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r0.status_code == 200, r0.get_data(as_text=True)
    assert r1.status_code == 200, r1.get_data(as_text=True)
    res1 = json.loads(r1.get_json()["tool_results"][0]["content"])
    assert res1["ok"]
    assert product.product_id in env.agents["agent_1"].listings


def test_multi_agent_observation_brief_is_per_agent(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")

    obs0 = c.get(f"/runs/{rid}/agents/agent_0/observation?nowait=1")
    obs1 = c.get(f"/runs/{rid}/agents/agent_1/observation?nowait=1")

    assert obs0.status_code == 200
    assert obs1.status_code == 200
    assert "brief" in obs0.get_json()
    assert "brief" in obs1.get_json()


def test_multi_agent_turn_quota_is_per_agent(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=1)}).get_json()["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r0 = _act(c, rid, "agent_0", "balance", [("query_balance", {})])
        r1 = _act(c, rid, "agent_1", "balance", [("query_balance", {})])
        r0_again = _act(c, rid, "agent_0", "again", [("query_balance", {})])
    finally:
        with env.hook_cond:
            env.hook_open = False

    assert r0.status_code == 200, r0.get_data(as_text=True)
    assert r1.status_code == 200, r1.get_data(as_text=True)
    assert r0.get_json()["turn_idx"] == 0
    assert r1.get_json()["turn_idx"] == 0
    assert r0_again.status_code == 429
    assert r0_again.get_json()["error"] == "max_turns_per_step=1 reached"


def test_multi_agent_end_of_step_waits_for_all_live_agents(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=3)}).get_json()["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    app.registry._require(rid)

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))
    r0 = _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    time.sleep(0.1)
    assert r0.status_code == 200, r0.get_data(as_text=True)
    assert r0.get_json()["step_done"] is True
    assert r0.get_json()["hook_released"] is False
    assert th.is_alive()

    r1 = _act(c, rid, "agent_1", "done", [("end_of_step", {})])
    th.join(timeout=3)

    assert r1.status_code == 200, r1.get_data(as_text=True)
    assert r1.get_json()["step_done"] is True
    assert r1.get_json()["hook_released"] is True
    assert not th.is_alive()
    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    assert {turn["agent_id"] for turn in payload["turns"]} == {"agent_0", "agent_1"}
    assert set(payload["message_agents"]) == {"agent_0", "agent_1"}
    assert all("agent_id" not in msg for msg in payload["messages"])


def test_multi_agent_trace_keeps_message_order_and_openai_shape(client, monkeypatch):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=3)}).get_json()["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    env = app.registry._require(rid)
    from storage import agent_log

    monkeypatch.setattr(agent_log, "now_ms", lambda: 12345)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r1 = _act(c, rid, "agent_1", "agent1 turn", [("query_balance", {})])
        r0 = _act(c, rid, "agent_0", "agent0 turn", [("query_balance", {})])
    finally:
        with env.hook_cond:
            env.hook_open = False

    assert r1.status_code == 200, r1.get_data(as_text=True)
    assert r0.status_code == 200, r0.get_data(as_text=True)
    by_step = os.path.join(tmp, "runs", rid, "agent", "by_step", "t_00000.json")
    payload = json.load(open(by_step))
    assistant_contents = [msg["content"] for msg in payload["messages"] if msg["role"] == "assistant"]
    assert assistant_contents == ["agent1 turn", "agent0 turn"]
    assert all("agent_id" not in msg for msg in payload["messages"])
    assistant_agents = [
        agent for msg, agent in zip(payload["messages"], payload["message_agents"]) if msg["role"] == "assistant"
    ]
    assert assistant_agents == ["agent_1", "agent_0"]
    assert [turn["agent_id"] for turn in payload["turns"]] == ["agent_1", "agent_0"]


def test_historical_observation_filters_trace_to_requested_agent(client):
    c, tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=3)}).get_json()["run_id"]
    app.registry.add_agent(rid, "agent_1", "Agent 1")
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r1 = _act(
            c,
            rid,
            "agent_1",
            "agent1 private turn",
            [("query_balance", {})],
            context={"tokens": 111},
        )
        r0 = _act(
            c,
            rid,
            "agent_0",
            "agent0 private turn",
            [("query_balance", {})],
            context={"tokens": 222},
        )
    finally:
        with env.hook_cond:
            env.hook_open = False

    assert r1.status_code == 200, r1.get_data(as_text=True)
    assert r0.status_code == 200, r0.get_data(as_text=True)

    agent0_history = c.get(f"/runs/{rid}/agents/agent_0/observation?t=0")
    agent1_history = c.get(f"/runs/{rid}/agents/agent_1/observation?t=0")
    ghost_history = c.get(f"/runs/{rid}/agents/ghost/observation?t=0")

    assert agent0_history.status_code == 200
    assert agent1_history.status_code == 200
    assert ghost_history.status_code == 404
    agent0_payload = agent0_history.get_json()
    agent1_payload = agent1_history.get_json()
    assert {msg["content"] for msg in agent0_payload["messages"] if msg["role"] == "assistant"} == {
        "agent0 private turn"
    }
    assert {msg["content"] for msg in agent1_payload["messages"] if msg["role"] == "assistant"} == {
        "agent1 private turn"
    }
    assert set(agent0_payload["message_agents"]) == {"agent_0"}
    assert set(agent1_payload["message_agents"]) == {"agent_1"}
    assert {turn["agent_id"] for turn in agent0_payload["turns"]} == {"agent_0"}
    assert {turn["agent_id"] for turn in agent1_payload["turns"]} == {"agent_1"}

    agent0_index = c.get(f"/runs/{rid}/agents/agent_0/trace_index")
    agent1_index = c.get(f"/runs/{rid}/agents/agent_1/trace_index")
    ghost_index = c.get(f"/runs/{rid}/agents/ghost/trace_index")
    assert agent0_index.status_code == 200
    assert agent1_index.status_code == 200
    assert ghost_index.status_code == 404
    assert agent0_index.get_json()["steps"] == [
        {
            "t": 0,
            "n": len(agent0_payload["messages"]),
            "context": {"tokens": 222},
        }
    ]
    assert agent1_index.get_json()["steps"] == [
        {
            "t": 0,
            "n": len(agent1_payload["messages"]),
            "context": {"tokens": 111},
        }
    ]
    assert agent0_index.get_json()["total"] == len(agent0_payload["messages"])
    assert agent1_index.get_json()["total"] == len(agent1_payload["messages"])

    agent0_trace = c.get(f"/runs/{rid}/agents/agent_0/trace?t=0")
    agent1_trace = c.get(f"/runs/{rid}/agents/agent_1/trace?t=0")
    assert agent0_trace.status_code == 200
    assert agent1_trace.status_code == 200
    assert {msg["content"] for msg in agent0_trace.get_json()["messages"] if msg["role"] == "assistant"} == {
        "agent0 private turn"
    }
    assert {msg["content"] for msg in agent1_trace.get_json()["messages"] if msg["role"] == "assistant"} == {
        "agent1 private turn"
    }

    all_traces = c.get(f"/runs/{rid}/agent/all_traces").get_json()
    step0 = next(step for step in all_traces["steps"] if step["t"] == 0)
    assistants = [
        (msg["content"], owner)
        for msg, owner in zip(step0["messages"], step0["message_agents"])
        if msg["role"] == "assistant"
    ]
    assert assistants == [
        ("agent1 private turn", "agent_1"),
        ("agent0 private turn", "agent_0"),
    ]
    assert all("agent_id" not in msg for msg in step0["messages"])

    # A malformed multi-agent trace must fail closed instead of treating all
    # unowned legacy messages as agent_0 data.
    trace_path = os.path.join(
        tmp,
        "runs",
        rid,
        "agent",
        "by_step",
        "t_00000.json",
    )
    malformed = json.load(open(trace_path))
    malformed.pop("message_agents", None)
    with open(trace_path, "w") as trace_file:
        json.dump(malformed, trace_file)
    assert c.get(f"/runs/{rid}/agents/agent_0/observation?t=0").get_json()["messages"] == []
    assert c.get(f"/runs/{rid}/agents/agent_1/observation?t=0").get_json()["messages"] == []
    assert c.get(f"/runs/{rid}/agents/agent_0/trace_index").get_json() == {"steps": [], "total": 0}


def test_agent_scoped_trace_remains_available_after_runtime_release(client):
    c, _tmp, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=3)}).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        response = _act(
            c,
            rid,
            "agent_0",
            "terminal trace",
            [("query_balance", {})],
        )
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert response.status_code == 200

    with app.registry.lease_conn_for(rid) as conn:
        dbm.update_run_status(conn, rid, "finished")
    assert app.registry.release_terminal_runtime(rid) is True
    assert app.registry.get_env(rid) is None

    index = c.get(f"/runs/{rid}/agents/agent_0/trace_index")
    detail = c.get(f"/runs/{rid}/agents/agent_0/trace?t=0")
    assert index.status_code == 200
    assert index.get_json()["steps"] == [
        {
            "t": 0,
            "n": len(detail.get_json()["messages"]),
        }
    ]
    assert detail.status_code == 200
    assert {msg["content"] for msg in detail.get_json()["messages"] if msg["role"] == "assistant"} == {"terminal trace"}
    assert c.get(f"/runs/{rid}/agents/ghost/trace_index").status_code == 404
    assert c.get(f"/runs/{rid}/agents/ghost/trace?t=0").status_code == 404


def test_act_outside_hook_returns_425(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    r = _act(c, rid, "agent_0", "test", [("market_brief", {"window_days": 7})])
    assert r.status_code == 425
    body = r.get_json()
    assert body["error"] == "hook_closed"
    runtime = agent_log.read_runtime_events(app.registry.runs_root, rid)
    assert runtime is not None
    assert runtime["events"][-1]["event_type"] == "merchantbench_api_failed_attempt"
    assert runtime["events"][-1]["payload"] == {
        "status": 425,
        "error": "hook_closed",
    }


def test_runtime_events_append_jsonl_and_read_legacy(tmp_path):
    runs_root = str(tmp_path)
    run_id = "run-runtime-events"
    agent_log.init_runtime_events(runs_root, run_id)
    for t in (1, 2):
        agent_log.record_runtime_event(
            runs_root,
            run_id,
            agent_id="agent_0",
            t=t,
            event_type="merchantbench_api_failed_attempt",
            payload={"status": 425},
        )

    base = tmp_path / run_id / "agent"
    stream_path = base / "runtime_events.jsonl"
    assert len(stream_path.read_text(encoding="utf-8").splitlines()) == 3

    (base / "runtime_events.json").write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {
                        "agent_id": "agent_0",
                        "t": 0,
                        "event_type": "merchantbench_api_failed_attempt",
                        "payload": {"status": 400},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime = agent_log.read_runtime_events(runs_root, run_id)
    assert runtime is not None
    assert runtime["version"] == 2
    assert runtime["capabilities"] == {"merchantbench_api_failed_attempts": True}
    markers = [event for event in runtime["events"] if event["event_type"] == "runtime_telemetry_started"]
    assert [(event["agent_id"], event["t"]) for event in markers] == [
        ("agent_0", 0),
    ]
    failures = [event for event in runtime["events"] if event["event_type"] == "merchantbench_api_failed_attempt"]
    assert [event["t"] for event in failures] == [0, 1, 2]


def test_act_rejects_unknown_agent_before_end_of_step(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r = _act(c, rid, "ghost", "end", [("end_of_step", {})])
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r.status_code == 404
    assert env.hook_event.is_set() is False


def test_act_validates_arguments_before_dispatch(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "bad args",
                    "tool_calls": [
                        {"id": "bad_args", "type": "function", "function": {"name": "query_balance", "arguments": "1"}}
                    ],
                }
            ]
        }
        r = c.post(f"/runs/{rid}/agents/agent_0/act", json=body)
        body2 = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "missing args",
                    "tool_calls": [
                        {
                            "id": "missing_args",
                            "type": "function",
                            "function": {"name": "list_product", "arguments": json.dumps({})},
                        }
                    ],
                }
            ]
        }
        r2 = c.post(f"/runs/{rid}/agents/agent_0/act", json=body2)
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r.status_code == 200, r.get_data(as_text=True)
    result = json.loads(r.get_json()["tool_results"][0]["content"])
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert r2.status_code == 200, r2.get_data(as_text=True)
    result2 = json.loads(r2.get_json()["tool_results"][0]["content"])
    assert result2["ok"] is False
    assert result2["error"]["code"] == "invalid_arguments"
    assert result2["error"]["path"] == "$.items"


def test_act_rejects_dead_agent_before_dispatch(client):
    c, _, app = client
    rid = c.post(
        "/runs",
        json={"scenario": _agent_scenario(hook_seconds=2)},
    ).get_json()["run_id"]
    env = app.registry._require(rid)
    env.agents["agent_0"].is_alive = False
    env.agents["agent_0"].died_at_t = env.t
    with env.hook_cond:
        env.hook_open = True
        env.hook_event.clear()
        env.hook_cond.notify_all()
    try:
        observation = c.get(
            f"/runs/{rid}/agents/agent_0/observation?nowait=1",
        )
        response = _act(
            c,
            rid,
            "agent_0",
            "keep acting",
            [
                ("end_of_step", {}),
            ],
        )
    finally:
        with env.hook_cond:
            env.hook_open = False

    assert observation.status_code == 410
    assert observation.get_json()["error"] == "agent_dead"
    assert response.status_code == 410
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"] == "agent_dead"
    assert body["died_at"]["day"] == 1
    assert env.hook_event.is_set() is False


def test_act_end_of_step_obeys_denylist(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, tool_denylist=["end_of_step"])}).get_json()[
        "run_id"
    ]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r = _act(c, rid, "agent_0", "end", [("end_of_step", {})])
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r.status_code == 200
    body = r.get_json()
    assert body["step_done"] is False
    result = json.loads(body["tool_results"][0]["content"])
    assert result["ok"] is False
    assert "not available" in result["error"]
    assert env.hook_event.is_set() is False


def test_act_requires_end_of_step_to_be_last(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    env = app.registry._require(rid)
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r = _act(
            c,
            rid,
            "agent_0",
            "bad eos order",
            [
                ("end_of_step", {}),
                ("query_balance", {}),
            ],
        )
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r.status_code == 400
    assert r.get_json()["error"] == "end_of_step_must_be_last"
    assert env.hook_event.is_set() is False


def test_act_end_of_step_allowed_when_hook_closed(client):
    """end_of_step should work even if hook is technically in a grey zone."""
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    r = _act(c, rid, "agent_0", "end", [("end_of_step", {})])
    assert r.status_code == 200
    assert r.get_json()["step_done"]
    th.join(timeout=3)


def test_end_of_step_over_quota_still_releases_hook(client):
    """Pure end_of_step may close the hook after the action turn budget is spent."""
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=1)}).get_json()["run_id"]
    env = app.registry._require(rid)
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    r1 = _act(c, rid, "agent_0", "probe", [("market_brief", {"window_days": 7})])
    assert r1.status_code == 200, r1.get_data(as_text=True)

    r2 = _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    assert r2.status_code == 200, r2.get_data(as_text=True)
    body = r2.get_json()
    assert body["step_done"] is True
    assert body["hook_released"] is True
    th.join(timeout=3)
    assert not th.is_alive()
    assert env.t == 1


def test_act_turn_quota_blocks_mutation_before_side_effect(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2, max_turns_per_step=1)}).get_json()["run_id"]
    env = app.registry._require(rid)
    product = next(iter(env.products.values()))
    with env.hook_cond:
        env.hook_open = True
        env.hook_cond.notify_all()
    try:
        r1 = _act(c, rid, "agent_0", "first turn", [("query_balance", {})])
        assert r1.status_code == 200, r1.get_data(as_text=True)
        r2 = _act(
            c,
            rid,
            "agent_0",
            "over quota",
            [("list_product", {"items": [{"product_id": product.product_id, "sale_price": product.price * 1.3}]})],
        )
    finally:
        with env.hook_cond:
            env.hook_open = False
    assert r2.status_code == 429
    assert product.product_id not in env.agents["agent_0"].listings


def test_environment_step_serializes_full_hook_window(client):
    c, _, app = client
    scenario = _agent_scenario(hook_seconds=2, activation_period=1)
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    env = app.registry._require(rid)
    gate = threading.Event()
    blocker_lock = threading.Lock()
    active_blockers = {"n": 0, "max": 0}

    def blocker():
        with blocker_lock:
            active_blockers["n"] += 1
            active_blockers["max"] = max(active_blockers["max"], active_blockers["n"])
            if active_blockers["n"] == 1:
                gate.set()
        time.sleep(0.2)
        with blocker_lock:
            active_blockers["n"] -= 1

    first = threading.Thread(target=lambda: env.step(hook_blocker=blocker))
    second = threading.Thread(target=lambda: env.step(hook_blocker=blocker))
    first.start()
    assert gate.wait(timeout=1)
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert active_blockers["max"] == 1
    assert env.t == 2


def test_act_fires_turn_listener(client):
    """record_act should notify turn listeners (SSE)."""
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    from web.run_worker import RunWorker

    with app.registry.lock:
        worker = app.registry.workers.get(rid) or RunWorker(app.registry, rid)
        app.registry.workers[rid] = worker
    worker.start(interval_ms=0)
    try:
        sub = worker.subscribe()
        import threading

        def drive():
            app.registry.step(rid)

        th = threading.Thread(target=drive)
        th.start()
        _wait_for_hook(app.registry._require(rid))
        _act(c, rid, "agent_0", "probe", [("end_of_step", {})])
        th.join(timeout=3)
        seen_turn = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                ev = sub.get(timeout=0.2)
            except Exception:
                continue
            if ev.get("type") == "turn":
                seen_turn = ev
                break
        assert seen_turn is not None, "no SSE 'turn' event published"
        assert seen_turn["turn_idx"] == 0
        assert seen_turn["t"] == 0
    finally:
        worker.stop()


def test_stop_detaches_turn_listener(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=0.1)}).get_json()["run_id"]
    from web.run_worker import RunWorker

    with app.registry.lock:
        worker = app.registry.workers.get(rid) or RunWorker(app.registry, rid)
        app.registry.workers[rid] = worker
    worker.start(interval_ms=0)
    env = app.registry._require(rid)
    assert worker._on_turn in env.turn_listeners
    worker.stop()
    assert worker._on_turn not in env.turn_listeners


def test_trace_endpoint_reads_by_step(client):
    """GET /agent/trace?t=N returns by_step data."""
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]
    import threading

    def drive():
        app.registry.step(rid)

    th = threading.Thread(target=drive)
    th.start()
    _wait_for_hook(app.registry._require(rid))
    _act(c, rid, "agent_0", "test", [("end_of_step", {})])
    th.join(timeout=3)
    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    assert trace["n_turns"] == 1
    assert len(trace["messages"]) >= 2


def _write_hermes_state_db(app, rid, rows, *, system_prompt=None):
    from storage import agent_log

    db_dir = os.path.join(agent_log.agent_dir(app.registry.runs_root, rid), "hermes_home")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "state.db")
    root_session = f"merchantbench-{rid}"

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
            "source TEXT, system_prompt TEXT, started_at REAL)"
        )
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, "
            "tool_name TEXT, reasoning TEXT, reasoning_content TEXT, "
            "reasoning_details TEXT, active INTEGER, compacted INTEGER, "
            "timestamp REAL)"
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, NULL, 'merchantbench', ?, 0)",
            (root_session, system_prompt),
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_hermes_trace_endpoint_reads_session_db(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    root_session = f"merchantbench-{rid}"
    _write_hermes_state_db(
        app,
        rid,
        [
            (1, root_session, "user", "Day 1, Hour 0\nSupply & listings:", None, None, None, None, None, None, 1, 0, 1),
            (
                2,
                root_session,
                "assistant",
                "search",
                None,
                json.dumps(
                    [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "merchantbench__search_products", "arguments": "{}"},
                        }
                    ]
                ),
                None,
                None,
                None,
                None,
                1,
                0,
                2,
            ),
            (3, root_session, "user", "Day 1, Hour 12\nNext observation", None, None, None, None, None, None, 1, 0, 3),
            (
                4,
                root_session,
                "assistant",
                "done",
                None,
                json.dumps(
                    [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "end_of_step", "arguments": "{}"},
                        }
                    ]
                ),
                None,
                None,
                None,
                None,
                1,
                0,
                4,
            ),
        ],
    )

    idx = c.get(f"/runs/{rid}/agent/hermes_all_traces_index").get_json()
    assert idx["steps"] == [{"t": 0, "n": 2}, {"t": 12, "n": 2}]
    trace = c.get(f"/runs/{rid}/agent/hermes_trace?t=0").get_json()
    assert trace["trace_source"] == "hermes"
    assert [m["role"] for m in trace["messages"]] == ["user", "assistant"]
    assert trace["messages"][1]["tool_calls"][0]["tool_origin"] == "merchantbench_env"


def test_hermes_trace_includes_persisted_session_system_prompt(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    root_session = f"merchantbench-{rid}"
    _write_hermes_state_db(
        app,
        rid,
        [
            (1, root_session, "user", "Day 1, Hour 0\nSupply & listings:", None, None, None, None, None, None, 1, 0, 1),
            (2, root_session, "assistant", "search", None, None, None, None, None, None, 1, 0, 2),
        ],
        system_prompt="Hermes base prompt\n\nMerchantBench operating rules",
    )

    idx = c.get(f"/runs/{rid}/agent/hermes_all_traces_index").get_json()
    assert idx["steps"] == [{"t": 0, "n": 3}]

    trace = c.get(f"/runs/{rid}/agent/hermes_trace?t=0").get_json()
    assert [m["role"] for m in trace["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert trace["messages"][0]["content"] == "Hermes base prompt\n\nMerchantBench operating rules"
    assert trace["messages"][0]["trace_source"] == "hermes"


def test_hermes_trace_merges_env_turn_context(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    root_session = f"merchantbench-{rid}"
    _write_hermes_state_db(
        app,
        rid,
        [
            (1, root_session, "user", "Day 1, Hour 0\nSupply & listings:", None, None, None, None, None, None, 1, 0, 1),
            (2, root_session, "assistant", "search", None, None, None, None, None, None, 1, 0, 2),
        ],
    )

    from storage import agent_log

    agent_log.write_step_index(
        app.registry.runs_root,
        rid,
        0,
        [{"role": "assistant", "content": "env trace"}],
        [
            {"turn_idx": 0, "context": {"tokens": 123456}},
            {"turn_idx": 1, "context": {"tokens": 90000, "compacted": True}},
        ],
    )

    idx = c.get(f"/runs/{rid}/agent/hermes_all_traces_index").get_json()
    assert idx["steps"] == [
        {
            "t": 0,
            "n": 2,
            "context": {"tokens": 123456, "compactions": 1},
        }
    ]

    trace = c.get(f"/runs/{rid}/agent/hermes_trace?t=0").get_json()
    assert trace["trace_source"] == "hermes"
    assert [m["role"] for m in trace["messages"]] == ["user", "assistant"]
    assert trace["n_turns"] == 2
    assert trace["turns"][0]["context"] == {"tokens": 123456}
    assert trace["turns"][1]["context"] == {
        "tokens": 90000,
        "compacted": True,
    }


def test_hermes_trace_defaults_to_active_messages(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    root_session = f"merchantbench-{rid}"
    _write_hermes_state_db(
        app,
        rid,
        [
            (
                1,
                root_session,
                "user",
                "Day 1, Hour 0\nCompacted observation",
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                1,
                1,
            ),
            (2, root_session, "assistant", "compacted old decision", None, None, None, None, None, None, 0, 1, 2),
            (3, root_session, "user", "Day 1, Hour 0\nActive observation", None, None, None, None, None, None, 1, 0, 3),
            (4, root_session, "assistant", "active decision", None, None, None, None, None, None, 1, 0, 4),
        ],
    )

    idx = c.get(f"/runs/{rid}/agent/hermes_all_traces_index").get_json()
    assert idx["steps"] == [{"t": 0, "n": 2}]
    assert idx["total_compacted"] == 2
    trace = c.get(f"/runs/{rid}/agent/hermes_trace?t=0").get_json()
    assert [m["content"] for m in trace["messages"]] == [
        "Day 1, Hour 0\nActive observation",
        "active decision",
    ]

    raw = c.get(f"/runs/{rid}/agent/hermes_trace?t=0&include_compacted=1").get_json()
    assert len(raw["messages"]) == 4


def test_hermes_trace_index_does_not_parse_messages(client, monkeypatch):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario()}).get_json()["run_id"]
    root_session = f"merchantbench-{rid}"
    _write_hermes_state_db(
        app,
        rid,
        [
            (1, root_session, "user", "Day 1, Hour 0\nObservation", None, None, None, None, None, None, 1, 0, 1),
        ],
    )

    import web.routes_agent as routes_agent

    monkeypatch.setattr(
        routes_agent,
        "_parse_hermes_tool_calls",
        lambda raw: (_ for _ in ()).throw(AssertionError("index parsed messages")),
    )

    idx = c.get(f"/runs/{rid}/agent/hermes_all_traces_index").get_json()
    assert idx["steps"] == [{"t": 0, "n": 1}]


def test_tool_calls_endpoint_returns_only_merchantbench_env_tool_calls(client):
    c, _, app = client
    rid = c.post("/runs", json={"scenario": _agent_scenario(hook_seconds=2)}).get_json()["run_id"]

    from storage import agent_log

    agent_log.write_step_index(
        app.registry.runs_root,
        rid,
        0,
        [
            {
                "role": "assistant",
                "content": "mixed",
                "tool_origin": "mixed",
                "tool_calls": [
                    {
                        "id": "call_env_0",
                        "type": "function",
                        "tool_origin": "merchantbench_env",
                        "function": {"name": "query_balance", "arguments": "{}"},
                    },
                    {
                        "id": "call_native_0",
                        "type": "function",
                        "tool_origin": "hermes_native",
                        "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                    },
                    {
                        "id": "call_end_0",
                        "type": "function",
                        "tool_origin": "merchantbench_env",
                        "function": {"name": "end_of_step", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "legacy",
                "tool_calls": [
                    {
                        "id": "call_legacy_0",
                        "type": "function",
                        "function": {"name": "query_my_orders", "arguments": "{}"},
                    }
                ],
            },
        ],
        [],
    )

    body = c.get(f"/runs/{rid}/agent/tool_calls").get_json()

    assert body == [
        {"t": 0, "tool_name": "query_balance"},
        {"t": 0, "tool_name": "query_my_orders"},
    ]


def test_trace_preserves_long_memory_markdown_arguments_and_results(client):
    c, _, app = client
    scenario = _agent_scenario(hook_seconds=2)
    scenario.setdefault("agent", {})["tool_denylist"] = []
    rid = c.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
    import threading

    th = threading.Thread(target=lambda: app.registry.step(rid))
    th.start()
    _wait_for_hook(app.registry._require(rid))

    long_md = (
        "# Memory\n\n"
        + "\n".join(f"- item {i}: keep the full markdown content visible in trace" for i in range(300))
        + "\n"
    )
    r = _act(
        c,
        rid,
        "agent_0",
        "store and read memory",
        [
            ("write_memory_doc", {"content": long_md}),
            ("read_memory_doc", {}),
        ],
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    read_result = json.loads(r.get_json()["tool_results"][1]["content"])
    assert read_result["content"] == long_md

    _act(c, rid, "agent_0", "done", [("end_of_step", {})])
    th.join(timeout=3)

    trace = c.get(f"/runs/{rid}/agent/trace?t=0").get_json()
    assistant = next(
        m
        for m in trace["messages"]
        if any(tc.get("function", {}).get("name") == "write_memory_doc" for tc in m.get("tool_calls", []))
    )
    write_call = next(tc for tc in assistant["tool_calls"] if tc["function"]["name"] == "write_memory_doc")
    assert json.loads(write_call["function"]["arguments"])["content"] == long_md

    read_tool_msg = next(m for m in trace["messages"] if m.get("role") == "tool" and m.get("name") == "read_memory_doc")
    assert json.loads(read_tool_msg["content"])["content"] == long_md
