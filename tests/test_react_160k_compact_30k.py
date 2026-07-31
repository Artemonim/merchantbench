"""ReAct 160k->30k compaction baseline behavior."""
import json
from types import SimpleNamespace

import requests

from agent.baselines import react_160k_compact_30k as compact_react


class _FakeClient:
    def __init__(self):
        self.acts = []
        self.messages = []
        self.contexts = []
        self.register_kwargs = None

    def register(self, **kwargs):
        self.register_kwargs = kwargs
        return {"ok": True}

    def tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_memory_doc",
                    "description": "write memory",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "end_of_step",
                    "description": "done",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    def act(self, assistant_msg=None, token_usage=None, *, messages=None, context=None):
        if messages is None:
            messages = [assistant_msg]
        assistant_msg = messages[-1]
        self.messages.append(messages)
        self.acts.append(assistant_msg)
        self.contexts.append(context)
        name = assistant_msg["tool_calls"][0]["function"]["name"]
        return {
            "ok": True,
            "turn_idx": len(self.acts) - 1,
            "step_done": name == "end_of_step",
            "tool_results": [{
                "tool_call_id": assistant_msg["tool_calls"][0]["id"],
                "name": name,
                "content": '{"ok": true}',
            }],
        }


class _FakeClientNoMemory(_FakeClient):
    def tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "end_of_step",
                    "description": "done",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]


class _FakeOpenAI:
    def __init__(self, prompt_tokens=0, usage=True):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.calls = []
        self.prompt_tokens = prompt_tokens
        self.usage = usage

    def create(self, **kwargs):
        self.calls.append(kwargs)
        saw_reminder = any(
            "[context-maintenance]" in str(m.get("content", ""))
            for m in kwargs["messages"]
        )
        if saw_reminder:
            tool_call = SimpleNamespace(
                id="call_memory",
                function=SimpleNamespace(
                    name="write_memory_doc",
                    arguments='{"content":"long-term strategy"}',
                ),
            )
        else:
            tool_call = SimpleNamespace(
                id="call_end",
                function=SimpleNamespace(name="end_of_step", arguments="{}"),
            )
        message = SimpleNamespace(content="ok", tool_calls=[tool_call])
        usage = None
        if self.usage:
            usage = SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=0,
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=usage)


class _FakeFailingThenSuccessOpenAI:
    def __init__(self, failures):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.calls = []
        self.failures = list(failures)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        tool_call = SimpleNamespace(
            id="call_end",
            function=SimpleNamespace(name="end_of_step", arguments="{}"),
        )
        message = SimpleNamespace(content="release after retry",
                                  tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=None)


class _FakeAlwaysFailOpenAI:
    def __init__(self, message):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.calls = []
        self.message = message

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError(self.message)


class _FakeActFailingClient(_FakeClient):
    def __init__(self, status_code=500):
        super().__init__()
        self.status_code = status_code
        self.failures_remaining = 1

    def act(self, assistant_msg=None, token_usage=None, *, messages=None, context=None):
        if self.failures_remaining:
            self.failures_remaining -= 1
            response = requests.Response()
            response.status_code = self.status_code
            response.url = "http://merchantbench.test/act"
            raise requests.HTTPError(f"{self.status_code} Server Error", response=response)
        return super().act(
            assistant_msg,
            token_usage,
            messages=messages,
            context=context,
        )


class _FakeToolOpenAI:
    def __init__(self, tool_name="market_brief"):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.calls = []
        self.tool_name = tool_name

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_name}",
            function=SimpleNamespace(name=self.tool_name, arguments="{}"),
        )
        message = SimpleNamespace(content="use tool", tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class _FakeNoToolOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="thinking only", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _burst_error():
    return (
        "BadRequestError: Error code: 400 - {'code': 'MPE-429', "
        "'detailMessage': '{\"code\":\"Throttling.BurstRate\","
        "\"message\":\"Request rate increased too quickly.\"}'}"
    )


def _allocation_error():
    return (
        "BadRequestError: Error code: 400 - {'code': 'MPE-429', "
        "'detailMessage': '{\"code\":\"Throttling.AllocationQuota\","
        "\"message\":\"Allocated quota exceeded\"}'}"
    )


def _rate_quota_error():
    return (
        "BadRequestError: Error code: 400 - {'code': 'MPE-429', "
        "'detailMessage': '{\"code\":\"Throttling.RateQuota\","
        "\"message\":\"Requests rate limit exceeded\"}'}"
    )


def _forbidden_error():
    return (
        "BadRequestError: Error code: 400 - {'code': 'MPE-001', "
        "'detailMessage': '{\"error\":{\"code\":\"Forbidden\","
        "\"message\":\"temporarily blocked\"}}'}"
    )


def _invalid_function_arguments_error():
    return (
        "BadRequestError: Error code: 400 - {'success': False, "
        "'message': '模型提供方错误', 'code': 'MPE-001', "
        "'detailMessage': '{\"code\":\"InvalidParameter\","
        "\"message\":\"<400> InternalError.Algo.InvalidParameter: "
        "The \\\\\"function.arguments\\\\\" parameter of the code model "
        "must be in JSON format.\"}'}"
    )


def _new_compact_agent(openai_client):
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.system_prompt = "system rules"
    agent.language = "zh"
    agent.client = _FakeClient()
    agent.openai = openai_client
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.max_hops_per_step = 1
    agent.temperature = None
    agent.context_window_tokens = 30000
    agent.compact_trigger_tokens = 160000
    agent.compact_keep_tokens = 30000
    agent.compaction_pending = False
    agent.last_prompt_tokens = 0
    agent._pending_pre_assistant_messages = []
    agent.history = []
    return agent


def test_compact_react_declares_runtime_health_capabilities():
    agent = _new_compact_agent(_FakeOpenAI())

    agent.register()

    capabilities = agent.client.register_kwargs["extra"]["runtime_health_capabilities"]
    assert capabilities == {
        "provider_api_failed_attempts": "reported",
        "retry_exhausted": "reported",
        "memory_compactions": "reported",
        "skills_evolutions": "not_applicable",
    }


def test_compact_react_waits_for_provider_prompt_usage_before_compacting():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.compact_trigger_tokens = 120
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent._pending_pre_assistant_messages = []
    agent.last_prompt_tokens = 119
    agent.history = [
        {"role": "user", "content": "old state " + ("x" * 500)},
        {"role": "assistant", "content": "middle state"},
        {"role": "user", "content": "recent state"},
    ]

    agent._maybe_add_compaction_reminder(_FakeClient().tools())

    assert not agent.compaction_pending
    assert agent._pending_pre_assistant_messages == []
    assert "old state" in str(agent.history)


def test_compact_react_falls_back_to_history_estimate_when_usage_missing():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.compact_trigger_tokens = 120
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent._pending_pre_assistant_messages = []
    agent.last_prompt_tokens = 0
    agent.history = [
        {"role": "user", "content": "old state " + ("x" * 500)},
        {"role": "assistant", "content": "middle state"},
        {"role": "user", "content": "recent state"},
    ]

    agent._maybe_add_compaction_reminder(_FakeClient().tools())

    assert agent.compaction_pending
    assert any(
        "[context-maintenance]" in str(m.get("content", ""))
        for m in agent._pending_pre_assistant_messages
    )
    assert "Conversation reached the 120-token limit" in agent._pending_pre_assistant_messages[0]["content"]


def test_compact_react_warns_memory_then_trims_history_to_keep_window():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.system_prompt = "system rules"
    agent.language = "zh"
    agent.client = _FakeClient()
    agent.openai = _FakeOpenAI()
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.max_hops_per_step = 2
    agent.temperature = None
    agent.context_window_tokens = 200
    agent.compact_trigger_tokens = 120
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent.last_prompt_tokens = 120
    agent.history = [
        {"role": "user", "content": "old state " + ("x" * 500)},
        {"role": "assistant", "content": "middle state"},
        {"role": "user", "content": "recent state"},
    ]

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    first_call_messages = agent.openai.calls[0]["messages"]
    assert any(
        "[context-maintenance]" in str(m.get("content", ""))
        for m in first_call_messages
    )
    assert "Conversation reached the 120-token limit" in first_call_messages[-1]["content"]
    assert "write_memory_doc" in first_call_messages[-1]["content"]
    assert agent.client.acts[0]["tool_calls"][0]["function"]["name"] == "write_memory_doc"
    assert "context_maintenance" not in agent.client.acts[0]
    assert agent.client.messages[0] == [
        {
            "role": "user",
            "content": compact_react._compaction_reminder(120, 50),
        },
        agent.client.acts[0],
    ]
    assert agent.client.contexts[0]["compacted"] is True
    second_call_history = agent.openai.calls[1]["messages"][1:]
    assert compact_react._history_token_estimate(second_call_history) <= 60
    assert "old state" not in str(agent.history)


def test_compact_react_records_context_message_when_memory_tool_unavailable():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.system_prompt = "system rules"
    agent.language = "zh"
    agent.client = _FakeClientNoMemory()
    agent.openai = _FakeOpenAI()
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.max_hops_per_step = 1
    agent.temperature = None
    agent.context_window_tokens = 200
    agent.compact_trigger_tokens = 120
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent.last_prompt_tokens = 120
    agent._pending_pre_assistant_messages = []
    agent.history = [
        {"role": "user", "content": "old state " + ("x" * 500)},
        {"role": "assistant", "content": "middle state"},
        {"role": "user", "content": "recent state"},
    ]

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    first_call_messages = agent.openai.calls[0]["messages"]
    assert not any(
        "write_memory_doc" in str(m.get("content", ""))
        for m in first_call_messages
    )
    assert any(
        "[context-maintenance]" in str(m.get("content", ""))
        for m in first_call_messages
    )
    assert agent.client.messages[0][0]["role"] == "user"
    assert "[context-maintenance]" in agent.client.messages[0][0]["content"]
    assert "Conversation reached the 120-token limit" in agent.client.messages[0][0]["content"]
    assert "memory" not in agent.client.messages[0][0]["content"].lower()
    assert "tool" not in agent.client.messages[0][0]["content"].lower()
    assert "old state" not in str(agent.history)
    assert compact_react._history_token_estimate(agent.history) <= 80


def test_compact_react_records_provider_prompt_usage_for_next_compaction_check():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.system_prompt = "system rules"
    agent.language = "zh"
    agent.client = _FakeClient()
    agent.openai = _FakeOpenAI(prompt_tokens=123)
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.max_hops_per_step = 1
    agent.temperature = None
    agent.context_window_tokens = 200
    agent.compact_trigger_tokens = 160000
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent.last_prompt_tokens = 0
    agent._pending_pre_assistant_messages = []
    agent.history = []

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    assert agent.last_prompt_tokens == 123
    assert agent.client.contexts[0] == {"tokens": 123}


def test_compact_react_reports_estimated_context_tokens_when_usage_missing():
    agent = _new_compact_agent(_FakeOpenAI(usage=False))

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    expected_tokens = compact_react._history_token_estimate(
        agent.openai.calls[0]["messages"]
    )
    assert agent.client.contexts[0] == {"tokens": expected_tokens}


def test_compact_react_clears_stale_prompt_usage_when_provider_omits_usage():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.system_prompt = "system rules"
    agent.language = "zh"
    agent.client = _FakeClient()
    agent.openai = _FakeOpenAI(usage=False)
    agent.model = "fake-model"
    agent.run_id = "test-run"
    agent.max_hops_per_step = 1
    agent.temperature = None
    agent.context_window_tokens = 200
    agent.compact_trigger_tokens = 160000
    agent.compact_keep_tokens = 50
    agent.compaction_pending = False
    agent.last_prompt_tokens = 123
    agent._pending_pre_assistant_messages = []
    agent.history = []

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    assert agent.last_prompt_tokens == 0


def test_compact_react_retries_burst_rate_then_continues_without_forcing_eos(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeFailingThenSuccessOpenAI([_burst_error(), _burst_error()])
    agent = _new_compact_agent(openai)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 3
    assert sleeps == [2, 5]
    assert len(agent.client.acts) == 1
    assert agent.client.acts[0]["content"] == "release after retry"
    assert not agent.client.acts[0]["content"].startswith("[llm-error]")
    assert agent.client.contexts[0]["provider_api_failed_attempts"] == 2
    assert "retry_exhausted" not in agent.client.contexts[0]


def test_compact_react_retries_rate_quota_then_continues_without_forcing_eos(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeFailingThenSuccessOpenAI([_rate_quota_error(), _rate_quota_error()])
    agent = _new_compact_agent(openai)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 3
    assert sleeps == [2, 5]
    assert len(agent.client.acts) == 1
    assert agent.client.acts[0]["content"] == "release after retry"
    assert not agent.client.acts[0]["content"].startswith("[llm-error]")


def test_compact_react_retries_allocation_quota_eight_times_then_forces_eos(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeAlwaysFailOpenAI(_allocation_error())
    agent = _new_compact_agent(openai)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 9
    assert sleeps == [30, 60, 120, 120, 120, 120, 120, 120]
    assert len(agent.client.acts) == 1
    assert "detail_code=Throttling.AllocationQuota retries=8" in agent.client.acts[0]["content"]
    assert agent.client.contexts[0]["provider_api_failed_attempts"] == 9
    assert agent.client.contexts[0]["retry_exhausted"] == 1


def test_compact_react_extracts_provider_specific_cached_tokens():
    assert compact_react._extract_cached_tokens(SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=128),
    )) == 128
    assert compact_react._extract_cached_tokens(SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        cacheReadInputTokensCompatible=64,
    )) == 64
    assert compact_react._extract_cached_tokens(SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        cache_read_input_tokens=32,
    )) == 32


def test_compact_react_adds_end_of_step_to_thought_only_message_in_same_turn():
    agent = _new_compact_agent(_FakeNoToolOpenAI())

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(agent.client.acts) == 1
    assert agent.client.acts[0] == {
        "role": "assistant",
        "content": "thinking only",
        "tool_calls": [{
            "id": "call_eos",
            "type": "function",
            "function": {"name": "end_of_step", "arguments": "{}"},
        }],
    }
    assert agent.history == [
        {"role": "user", "content": "day 1 observation"},
        {"role": "assistant", "content": "thinking only"},
    ]


def test_compact_react_claude_cache_breakpoints_skip_empty_text_blocks():
    messages = [
        {"role": "user", "content": "previous observation"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "search_products", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a", "name": "search_products",
         "content": '{"ok": true}'},
        {"role": "user", "content": "new observation"},
    ]

    out = compact_react._add_cache_breakpoints(messages)

    assert out[1]["content"] == ""
    assert out[-1]["content"] == [
        {"type": "text", "text": "new observation",
         "cache_control": {"type": "ephemeral"}},
    ]
    assert "cache_control" not in str(out[1])


def test_compact_react_sanitizes_malformed_tool_call_arguments_for_llm_request():
    bad_args = '{"items": '
    history = [
        {"role": "user", "content": "previous observation"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_bad", "type": "function",
             "function": {"name": "list_product", "arguments": bad_args}},
        ]},
        {"role": "tool", "tool_call_id": "call_bad", "name": "list_product",
         "content": '{"ok": false, "error": {"code": "invalid_arguments"}}'},
    ]

    messages = compact_react._build_llm_messages("system", history, 10000)

    sanitized_args = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sanitized_args) == {"_invalid_json_arguments": bad_args}
    assert history[1]["tool_calls"][0]["function"]["arguments"] == bad_args


def test_compact_react_claude_omits_empty_cached_system_prompt():
    agent = _new_compact_agent(_FakeOpenAI())
    agent.model = "claude-opus-4-8"
    agent.system_prompt = ""

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    request = agent.openai.calls[0]
    assert "extra_body" not in request


def test_compact_react_keeps_end_of_step_result_out_of_llm_history():
    agent = compact_react.ReActAgent.__new__(compact_react.ReActAgent)
    agent.history = []
    assistant_msg = {
        "role": "assistant",
        "content": "No changes; wait for the next observation.",
        "tool_calls": [{
            "id": "call_eos",
            "type": "function",
            "function": {"name": "end_of_step", "arguments": "{}"},
        }],
    }
    act_resp = {
        "tool_results": [{
            "tool_call_id": "call_eos",
            "name": "end_of_step",
            "content": '{"ok": true}',
        }],
    }

    compact_react.ReActAgent._remember_act(agent, assistant_msg, act_resp)

    assert agent.history == [{
        "role": "assistant",
        "content": "No changes; wait for the next observation.",
    }]


def test_compact_react_forbidden_redacts_last_tool_turn_and_retries_once(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeFailingThenSuccessOpenAI([_forbidden_error()])
    agent = _new_compact_agent(openai)
    agent.history = [
        {"role": "user", "content": "previous observation"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "search_products", "arguments": "{}"}},
            {"id": "call_b", "type": "function",
             "function": {"name": "query_my_listings", "arguments": "{}"}},
        ], "reasoning_content": "keep reasoning"},
        {"role": "tool", "tool_call_id": "call_a", "name": "search_products",
         "content": "性感 吊带 大露背"},
        {"role": "tool", "tool_call_id": "call_b", "name": "query_my_listings",
         "content": "成人 烟具 药"},
    ]

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 2
    assert sleeps == []
    assert agent.history[1]["reasoning_content"] == "keep reasoning"
    assert agent.history[2]["tool_call_id"] == "call_a"
    assert agent.history[3]["tool_call_id"] == "call_b"
    assert "redacted_for_policy_retry" in agent.history[2]["content"]
    assert "redacted_for_policy_retry" in agent.history[3]["content"]
    assert "性感" not in agent.history[2]["content"]
    assert "成人" not in agent.history[3]["content"]


def test_compact_react_forbidden_retry_then_transient_uses_retry_budget(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeFailingThenSuccessOpenAI([_forbidden_error(), _burst_error()])
    agent = _new_compact_agent(openai)
    agent.history = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "search_products", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a", "name": "search_products",
         "content": "性感 吊带 大露背"},
    ]

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 3
    assert sleeps == [2]
    assert len(agent.client.acts) == 1
    assert agent.client.acts[0]["content"] == "release after retry"


def test_compact_react_invalid_function_arguments_error_does_not_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeAlwaysFailOpenAI(_invalid_function_arguments_error())
    agent = _new_compact_agent(openai)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 1
    assert sleeps == []
    assert len(agent.client.acts) == 1
    assert "detail_code=InvalidParameter retries=0" in agent.client.acts[0]["content"]


def test_compact_react_keeps_pending_compaction_message_when_act_fails():
    agent = _new_compact_agent(_FakeToolOpenAI("write_memory_doc"))
    agent.client = _FakeActFailingClient(status_code=425)
    agent.max_hops_per_step = 1
    agent.compact_trigger_tokens = 120
    agent.compact_keep_tokens = 50
    agent.last_prompt_tokens = 120
    agent.history = [
        {"role": "user", "content": "old state " + ("x" * 500)},
        {"role": "assistant", "content": "middle state"},
    ]

    agent._drive_step({"text": "new observation"}, t_key=0, verbose=False)

    assert agent.compaction_pending
    assert len(agent._pending_pre_assistant_messages) == 1
    assert "[context-maintenance]" in agent._pending_pre_assistant_messages[0]["content"]


def test_compact_react_forces_end_of_step_after_non_stale_act_error():
    agent = _new_compact_agent(_FakeToolOpenAI("market_brief"))
    agent.client = _FakeActFailingClient(status_code=500)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(agent.client.acts) == 1
    assert agent.client.acts[0]["tool_calls"][0]["function"]["name"] == "end_of_step"
    assert agent.client.acts[0]["content"].startswith("[act-error]")


def test_compact_react_unknown_llm_error_does_not_retry_or_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(compact_react.time, "sleep", lambda seconds: sleeps.append(seconds))
    openai = _FakeAlwaysFailOpenAI("BadRequestError: unexpected provider failure")
    agent = _new_compact_agent(openai)

    agent._drive_step({"text": "day 1 observation"}, t_key=0, verbose=False)

    assert len(openai.calls) == 1
    assert sleeps == []
    assert len(agent.client.acts) == 1
    assert "detail_code=unknown retries=0" in agent.client.acts[0]["content"]
    assert agent.client.contexts[0]["provider_api_failed_attempts"] == 1
    assert "retry_exhausted" not in agent.client.contexts[0]
