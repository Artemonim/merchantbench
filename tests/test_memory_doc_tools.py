"""Markdown memory document tools.

These tools are agent-facing and go through the normal /act path so traces and
the dashboard see them like any other tool.
"""

import json
import os
import tempfile
import threading
import time
import uuid

import pytest
from web.app import create_app
from web.runner import load_default_scenario


def _make_tool_call(name, arguments, call_id=None):
    if call_id is None:
        call_id = f"call_{name}_{uuid.uuid4().hex[:8]}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _act_msg(tool_calls):
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        ]
    }


def _call_tool(c, rid, agent_id, name, args=None):
    body = _act_msg([_make_tool_call(name, args or {})])
    resp = c.post(f"/runs/{rid}/agents/{agent_id}/act", json=body)
    data = resp.get_json()
    assert data["ok"], data
    return json.loads(data["tool_results"][0]["content"])


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
    app = create_app(db_path=os.path.join(tmp, "test.db"), runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        scen = load_default_scenario()
        scen["run"]["max_hook_seconds"] = 5.0
        scen["run"]["horizon_steps"] = 3
        scen["data"]["source"] = "synthetic"
        scen["data"]["num_products"] = 30
        scen.setdefault("agent", {})["tool_denylist"] = []
        rid = c.post("/runs", json={"scenario": scen}).get_json()["run_id"]
        app.registry.add_agent(rid, "agent_1", "Agent 1")
        th = threading.Thread(target=lambda: app.registry.step(rid), daemon=True)
        th.start()
        _wait_for_hook(app.registry._require(rid))
        try:
            yield c, rid
        finally:
            for agent_id in ("agent_0", "agent_1"):
                c.post(
                    f"/runs/{rid}/agents/{agent_id}/act",
                    json=_act_msg([_make_tool_call("end_of_step", {}, f"call_end_{agent_id}")]),
                )
            th.join(timeout=3)


def test_memory_doc_tools_appear_in_schema_and_list_tools(hook_session):
    c, rid = hook_session

    schema = c.get(f"/runs/{rid}/tools/schema").get_json()
    schema_names = {t["name"] for t in schema["tools"]}
    assert "read_memory_doc" in schema_names
    assert "write_memory_doc" in schema_names

    listed = _call_tool(c, rid, "agent_0", "list_tools")
    listed_names = {row["function"]["name"] for row in listed}
    assert "read_memory_doc" in listed_names
    assert "write_memory_doc" in listed_names


def test_memory_doc_read_write_round_trips_markdown(hook_session):
    c, rid = hook_session
    content = "# Plan\n\n- keep product p_001 listed\n- check refunds tomorrow\n"

    initial = _call_tool(c, rid, "agent_0", "read_memory_doc")
    assert initial == {"ok": True, "content": "", "bytes": 0}

    written = _call_tool(c, rid, "agent_0", "write_memory_doc", {"content": content})
    assert written == {"ok": True, "bytes": len(content.encode("utf-8"))}

    reread = _call_tool(c, rid, "agent_0", "read_memory_doc")
    assert reread == {"ok": True, "content": content, "bytes": len(content.encode("utf-8"))}


def test_memory_doc_writes_keep_markdown_history_file(hook_session):
    c, rid = hook_session
    first = "# Memory v1\n\n- list p_001\n"
    second = "# Memory v2\n\n- delist p_001\n"

    _call_tool(c, rid, "agent_0", "write_memory_doc", {"content": first})
    _call_tool(c, rid, "agent_0", "write_memory_doc", {"content": second})

    latest = _call_tool(c, rid, "agent_0", "read_memory_doc")
    assert latest["content"] == second

    history_path = os.path.join(
        c.application.registry.runs_root,
        rid,
        "agent",
        "memory",
        "agent_0.history.md",
    )
    with open(history_path, "r", encoding="utf-8") as f:
        history = f.read()

    assert history.count("<!-- merchantbench-memory-version ") == 2
    assert "## Memory version 1" in history
    assert "## Memory version 2" in history
    assert "step: 0" in history
    assert first in history
    assert second in history
    assert history.index(first) < history.index(second)


def test_memory_doc_is_isolated_by_agent_id(hook_session):
    c, rid = hook_session

    _call_tool(c, rid, "agent_0", "write_memory_doc", {"content": "# Agent 0\n"})
    _call_tool(c, rid, "agent_1", "write_memory_doc", {"content": "# Agent 1\n"})

    assert _call_tool(c, rid, "agent_0", "read_memory_doc")["content"] == "# Agent 0\n"
    assert _call_tool(c, rid, "agent_1", "read_memory_doc")["content"] == "# Agent 1\n"
