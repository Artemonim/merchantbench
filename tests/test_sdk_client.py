"""SDK unit tests using a real Flask test server (werkzeug make_server)."""

import json
import os
import socket
import tempfile
import threading
import time

import pytest
from sdk.merchantbench_tool_client import MerchantBenchToolClient
from web.app import create_app
from web.runner import load_default_scenario
from werkzeug.serving import make_server


def _table_records(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"), runs_root=os.path.join(tmp, "runs"))
    port = _free_port()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            time.sleep(0.01)
    else:
        srv.shutdown()
        th.join(timeout=2)
        raise RuntimeError("test server did not start")
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        srv.shutdown()
        th.join(timeout=2)


def _new_run(base, hook_seconds: float = 0.05):
    import requests

    scen = load_default_scenario()
    scen["run"]["max_hook_seconds"] = hook_seconds
    scen["run"]["horizon_steps"] = 5
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 30
    scen.setdefault("agent", {})["tool_denylist"] = []
    return requests.post(f"{base}/runs", json={"scenario": scen}, timeout=5).json()["run_id"]


def _make_assistant_msg(tool_calls):
    """Build an OpenAI-format assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }


def _make_tool_call(name, arguments, call_id=None):
    """Build a single OpenAI-format tool_call entry."""
    if call_id is None:
        call_id = f"call_{name}_{id(arguments)}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
        },
    }


def _end_of_step_msg():
    """Build an assistant message that only calls end_of_step."""
    return _make_assistant_msg([_make_tool_call("end_of_step", {})])


def test_sdk_default_http_timeout_is_long_enough_for_batch_runs(monkeypatch):
    monkeypatch.setattr(MerchantBenchToolClient, "refresh_schema", lambda self: None)

    client = MerchantBenchToolClient("http://merchantbench.test", "run_1", "agent_0")

    assert client.timeout == 600.0


def _wait_for_hook(env, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if env.hook_open:
            return
        time.sleep(0.01)
    raise RuntimeError(f"Hook window did not open within {timeout:g} seconds")


def _with_hook(app, rid, base, fn):
    """Drive one step in another thread so the hook is open while fn() runs.
    fn receives no args. Releases hook via end_of_step tool + joins step."""
    import requests

    step_th = threading.Thread(target=lambda: app.registry.step(rid), daemon=True)
    step_th.start()
    _wait_for_hook(app.registry._require(rid))
    try:
        return fn()
    finally:
        try:
            # end_of_step via /act
            msg = _end_of_step_msg()
            requests.post(
                f"{base}/runs/{rid}/agents/agent_0/act",
                json={"messages": [msg]},
                timeout=2,
            )
        except Exception:
            pass
        step_th.join(timeout=3)


def test_tools_returns_openai_function_schemas(server):
    base, _ = server
    rid = _new_run(base)
    client = MerchantBenchToolClient(base, rid, "agent_0")
    tools = client.tools()
    assert isinstance(tools, list) and tools
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]
    names = {t["function"]["name"] for t in tools}
    assert "list_product" in names
    assert "adjust_price" in names
    assert "list_tools" in names


def test_act_dispatches_read_tool(server):
    """client.act() with a market_brief tool_call returns category rows."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        resp = client.act(msg)
        assert resp["ok"] is True
        assert "turn_idx" in resp
        assert isinstance(resp["tool_results"], list)
        content = json.loads(resp["tool_results"][0]["content"])
        assert isinstance(content["categories"], list) and content["categories"]
        return resp

    _with_hook(app, rid, base, body)


def test_act_dispatches_mutating_tool(server):
    """client.act() can list a product (mutating tool)."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg1 = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        r1 = client.act(msg1)
        [row["category"] for row in json.loads(r1["tool_results"][0]["content"])["categories"]]
        msg2 = _make_assistant_msg([_make_tool_call("search_products", {"query": "", "page": 1, "page_size": 1})])
        r2 = client.act(msg2)
        items = json.loads(r2["tool_results"][0]["content"])["items"]
        browsed = _table_records(items)
        pid = browsed[0]["product_id"]
        # List the product (mutating)
        msg3 = _make_assistant_msg(
            [
                _make_tool_call(
                    "list_product",
                    {
                        "items": [
                            {
                                "product_id": pid,
                                "sale_price": browsed[0]["price"] * 1.4,
                            }
                        ],
                    },
                )
            ]
        )
        r3 = client.act(msg3)
        result = json.loads(r3["tool_results"][0]["content"])
        assert result.get("ok") is True
        return result

    _with_hook(app, rid, base, body)


def test_act_with_token_usage(server):
    """client.act() accepts token_usage without error."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        resp = client.act(msg, token_usage={"input": 100, "output": 50, "total": 150})
        assert resp["ok"] is True
        return resp

    _with_hook(app, rid, base, body)


def test_act_sends_explicit_messages_payload():
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "tool_results": [], "step_done": False}

    class _Session:
        def __init__(self):
            self.post_json = None

        def post(self, _url, json=None, headers=None, timeout=None):
            self.post_json = json
            return _Resp()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://merchantbench.test"
    client.run_id = "run-1"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client._latest_env_t = None
    client._session = _Session()

    reminder = {"role": "user", "content": "[context-maintenance] compact soon"}
    assistant = _end_of_step_msg()
    client.act(messages=[reminder, assistant])

    assert client._session.post_json["messages"] == [reminder, assistant]


def test_act_sends_context_payload():
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "tool_results": [], "step_done": False}

    class _Session:
        def __init__(self):
            self.post_json = None

        def post(self, _url, json=None, headers=None, timeout=None):
            self.post_json = json
            return _Resp()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://merchantbench.test"
    client.run_id = "run-1"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client._latest_env_t = None
    client._session = _Session()

    assistant = _end_of_step_msg()
    client.act(assistant, context={"tokens": 123456, "compacted": True})

    assert client._session.post_json["context"] == {
        "tokens": 123456,
        "compacted": True,
    }


def test_record_usage_sends_idempotent_auxiliary_payload():
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "recorded": True}

    class _Session:
        def __init__(self):
            self.url = None
            self.body = None

        def post(self, url, json=None, timeout=None):
            self.url = url
            self.body = json
            return _Resp()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://merchantbench.test"
    client.run_id = "run-1"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client._latest_env_t = 2148
    client._session = _Session()

    result = client.record_usage(
        {"input": 700, "output": 40, "total": 740},
        usage_id="review-final",
        source="checkpoint_review",
        model="review-model",
        provider="review-provider",
        cost_usd=0.1234567,
        cost_status="known",
        cost_source="provider_response",
    )

    assert result["recorded"] is True
    assert client._session.url.endswith("/runs/run-1/agents/agent_0/usage")
    assert client._session.body == {
        "usage_id": "review-final",
        "source": "checkpoint_review",
        "token_usage": {"input": 700, "output": 40, "total": 740},
        "step": 2148,
        "model": "review-model",
        "provider": "review-provider",
        "cost_usd": 0.1234567,
        "cost_status": "known",
        "cost_source": "provider_response",
    }


def test_act_end_of_step_sets_step_done(server):
    """Calling end_of_step tool via act returns step_done=True."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    step_th = threading.Thread(target=lambda: app.registry.step(rid), daemon=True)
    step_th.start()
    _wait_for_hook(app.registry._require(rid))
    try:
        msg = _end_of_step_msg()
        resp = client.act(msg)
        assert resp["ok"] is True
        assert resp["step_done"] is True
    finally:
        step_th.join(timeout=3)


def test_act_without_end_of_step_has_step_done_false(server):
    """A normal tool call without end_of_step returns step_done=False."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        resp = client.act(msg)
        assert resp["step_done"] is False
        return resp

    _with_hook(app, rid, base, body)


def test_act_unknown_tool_returns_error_in_results(server):
    """Unknown tool doesn't crash; it returns an error in tool_results."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg = _make_assistant_msg([_make_tool_call("definitely_not_a_real_tool", {})])
        resp = client.act(msg)
        assert resp["ok"] is True
        content = json.loads(resp["tool_results"][0]["content"])
        assert content.get("ok") is False
        assert "unknown tool" in content.get("error", "")
        return resp

    _with_hook(app, rid, base, body)


# ---------- X-Agent-Step auto-injection + stale-step rejection ----------


def test_observation_records_env_t_and_act_injects_x_agent_step(server):
    """After client.observation(), subsequent client.act() must auto-attach
    X-Agent-Step: <env.t> matching env.t."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")
    assert client.latest_env_t() is None  # not observed yet

    def body():
        pkt = client.observation(timeout=1.0)
        assert pkt["tick"]["day"] == 1
        assert pkt["tick"]["hour"] == 0
        assert client.latest_env_t() == 0
        # act() should succeed — proves X-Agent-Step matched.
        msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        resp = client.act(msg)
        assert resp["ok"] is True
        return resp

    _with_hook(app, rid, base, body)


def test_observation_uses_raw_step_when_tick_exposes_it():
    """day/hour cannot be inverted when step_hours != 1; raw tick is authoritative."""

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok", "tick": {"day": 1, "hour": 2, "step": 1}}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://env"
    client.run_id = "run-x"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client.observation_timeout = 30.0
    client._session = FakeSession()
    client._latest_env_t = None

    packet = client.observation(nowait=True)

    assert packet["tick"]["step"] == 1
    assert client.latest_env_t() == 1


def test_act_rejects_stale_step(server):
    """If the agent's recorded step is older than env.t, env returns 425
    stale_step. Simulate by manually setting client._latest_env_t to a past
    value before calling."""
    import requests as req_lib

    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        # Pretend we observed at step 99, but env is actually at step 0.
        client._latest_env_t = 99
        try:
            msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
            client.act(msg)
        except req_lib.HTTPError as e:
            assert e.response.status_code == 425
            resp_body = e.response.json()
            assert resp_body["error"] == "stale_step"
            assert resp_body["agent_step"] == 99
            assert resp_body["env_step"] == 0
            return "rejected"
        return "accepted"  # would be a bug

    result = _with_hook(app, rid, base, body)
    assert result == "rejected", "stale-step call should have been rejected"


def test_act_without_observation_omits_x_agent_step(server):
    """Before any observation, the SDK must NOT inject X-Agent-Step. This
    preserves backward compatibility for callers that don't use the
    long-poll observation flow."""
    base, app = server
    rid = _new_run(base, hook_seconds=2.0)
    client = MerchantBenchToolClient(base, rid, "agent_0")

    def body():
        msg = _make_assistant_msg([_make_tool_call("market_brief", {"window_days": 7})])
        resp = client.act(msg)
        assert resp["ok"] is True
        return resp

    _with_hook(app, rid, base, body)


def test_observation_retries_transient_long_poll_disconnects():
    """Network-layer idle disconnects during long-polling should behave like
    HTTP 408: retry the safe observation request instead of forcing the agent
    process to reconnect itself."""
    import requests as req_lib

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok", "tick": {"day": 1, "hour": 0}}

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise req_lib.ReadTimeout("idle long-poll disconnected")
            return FakeResponse()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://env"
    client.run_id = "run-x"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client.observation_timeout = 30.0
    client._session = FakeSession()
    client._latest_env_t = None

    packet = client.observation()

    assert packet["text"] == "ok"
    assert client.latest_env_t() == 0
    assert client._session.calls == 2


def test_observation_reraises_persistent_connection_errors():
    """Persistent connection failures should surface instead of making the
    agent wait forever on a misconfigured or dead env endpoint."""
    import requests as req_lib

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok", "tick": {"day": 1, "hour": 0}}

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls <= 4:
                raise req_lib.ConnectionError("env refused connection")
            return FakeResponse()

    client = MerchantBenchToolClient.__new__(MerchantBenchToolClient)
    client.base = "http://env"
    client.run_id = "run-x"
    client.agent_id = "agent_0"
    client.timeout = 15.0
    client.observation_timeout = 30.0
    client.observation_connection_retries = 3
    client._session = FakeSession()
    client._latest_env_t = None

    with pytest.raises(req_lib.ConnectionError):
        client.observation()

    assert client._session.calls == 4
