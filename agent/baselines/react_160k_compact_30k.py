"""ReAct 160k->30k compaction baseline for MerchantBench.

Uses the unified /act endpoint: each LLM hop produces an assistant message
with tool_calls, sent to the env via client.act(). The env executes tools
and returns results, which are fed back to the LLM for the next hop.

Run locally:
  cp .env.example .env   # fill in OPENAI_API_KEY etc.
  cd env && python run.py --port 5050 &
  .venv/bin/python agent/baselines/react_160k_compact_30k.py \
      --run-id <rid> --base-url http://localhost:5050
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("react_160k_compact_30k requires 'requests': "
          ".venv/bin/python -m pip install -r agent/requirements.txt",
          file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("react_160k_compact_30k requires 'openai>=1.40': "
          ".venv/bin/python -m pip install -r agent/requirements.txt",
          file=sys.stderr)
    raise

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_kw):
        return False

_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from sdk.merchantbench_tool_client import MerchantBenchToolClient


VERSION = "2.1-react-160k-compact-30k"
FRAMEWORK = "react_160k_compact_30k"
DEFAULT_MAX_HOPS = 30
DEFAULT_MAX_STEPS = 2200
DEFAULT_CONTEXT_WINDOW_TOKENS = 160000
DEFAULT_COMPACT_TRIGGER_TOKENS = 160000
DEFAULT_COMPACT_KEEP_TOKENS = 30000
_LLM_RETRY_DELAYS = [2, 5, 10, 20, 40, 60, 60, 60]
_LLM_ALLOCATION_RETRY_DELAYS = [30, 60, 120, 120, 120, 120, 120, 120]
_RETRYABLE_LLM_DETAIL_CODES = {
    "Throttling.BurstRate",
    "Throttling.RateQuota",
    "Throttling.AllocationQuota",
    "transient",
}
_POLICY_REDACTED_TOOL_CONTENT = json.dumps({
    "redacted_for_policy_retry": True,
    "reason": (
        "provider returned Forbidden; previous tool output was removed "
        "from the LLM prompt"
    ),
    "hint": "continue with known numeric facts or query narrower if needed",
}, ensure_ascii=False, sort_keys=True)


def _est_tokens(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    ascii_chars = sum(1 for ch in value if ord(ch) < 128)
    non_ascii_chars = len(value) - ascii_chars
    return max(1, (ascii_chars + 2 * non_ascii_chars) // 4)


def _message_token_estimate(message: dict) -> int:
    return _est_tokens(message) + 4


def _history_token_estimate(messages: list[dict]) -> int:
    return sum(_message_token_estimate(msg) for msg in messages)


def _tool_available(tools: list[dict], name: str) -> bool:
    return any(
        isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == name
        for tool in tools
    )


def _trim_messages_to_token_budget(messages: list[dict],
                                   max_tokens: int) -> list[dict]:
    if max_tokens <= 0:
        return []
    kept_reversed: list[dict] = []
    total = 0
    for msg in reversed(messages):
        cost = _message_token_estimate(msg)
        if kept_reversed and total + cost > max_tokens:
            break
        kept_reversed.append(msg)
        total += cost
        if total >= max_tokens:
            break
    kept = list(reversed(kept_reversed))
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    return kept


def _invalid_json_arguments(value: str) -> str:
    return json.dumps(
        {"_invalid_json_arguments": value},
        ensure_ascii=False,
        sort_keys=True,
    )


def _coerce_json_argument_string(value: Any) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            value = str(value)
    if not value.strip():
        return "{}"
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return _invalid_json_arguments(value)
    if not isinstance(parsed, dict):
        return _invalid_json_arguments(value)
    return value


def _sanitize_message_for_llm(message: dict) -> dict:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    sanitized_calls = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            sanitized_calls.append(tc)
            continue
        func = tc.get("function")
        if not isinstance(func, dict):
            sanitized_calls.append(tc)
            continue
        tc_copy = dict(tc)
        func_copy = dict(func)
        func_copy["arguments"] = _coerce_json_argument_string(
            func.get("arguments")
        )
        tc_copy["function"] = func_copy
        sanitized_calls.append(tc_copy)
    message_copy = dict(message)
    message_copy["tool_calls"] = sanitized_calls
    return message_copy


def _build_llm_messages(system_prompt: Optional[str], history: list[dict],
                        max_history_tokens: int,
                        *, exclude_system: bool = False) -> list[dict]:
    messages: list[dict] = []
    if system_prompt and not exclude_system:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        _sanitize_message_for_llm(msg)
        for msg in _trim_messages_to_token_budget(history, max_history_tokens)
    )
    return messages


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def _add_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """Add cache_control breakpoints at: last assistant, [-2], [-1]."""
    if not messages:
        return messages
    messages = [dict(m) for m in messages]

    indices: set[int] = set()
    indices.add(len(messages) - 1)
    if len(messages) >= 2:
        indices.add(len(messages) - 2)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            indices.add(i)
            break

    for idx in indices:
        msg = dict(messages[idx])
        content = msg.get("content", "")
        if isinstance(content, str):
            if not content:
                messages[idx] = msg
                continue
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
            ]
        elif isinstance(content, list) and content:
            content = [dict(b) if isinstance(b, dict) else b for b in content]
            for block_idx in range(len(content) - 1, -1, -1):
                block = content[block_idx]
                if isinstance(block, dict) and block.get("text"):
                    content[block_idx] = dict(block)
                    content[block_idx]["cache_control"] = {"type": "ephemeral"}
                    msg["content"] = content
                    break
        messages[idx] = msg

    return messages


def _is_only_end_of_step(assistant_msg: dict) -> bool:
    tool_calls = assistant_msg.get("tool_calls") or []
    return len(tool_calls) == 1 and (
        tool_calls[0].get("function", {}).get("name") == "end_of_step"
    )


def _end_of_step_tool_call() -> dict:
    return {
        "id": "call_eos",
        "type": "function",
        "function": {"name": "end_of_step", "arguments": "{}"},
    }


def _extract_reasoning_content(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None and hasattr(message, "model_dump"):
        value = message.model_dump().get("reasoning_content")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_cached_tokens(usage: Any) -> int:
    pt_details = getattr(usage, "prompt_tokens_details", None)
    if pt_details is not None:
        value = getattr(pt_details, "cached_tokens", None)
        if value:
            return int(value)
    for attr in ("cacheReadInputTokensCompatible", "cache_read_input_tokens"):
        value = getattr(usage, attr, None)
        if value:
            return int(value)
    return 0


def _llm_detail_code(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    if (
        "InvalidParameter" in text
        and (
            "function.arguments" in text
            or "must be in JSON format" in text
        )
    ):
        return "InvalidParameter"
    if "Throttling.BurstRate" in text:
        return "Throttling.BurstRate"
    if "Throttling.RateQuota" in text:
        return "Throttling.RateQuota"
    if "Throttling.AllocationQuota" in text:
        return "Throttling.AllocationQuota"
    if "Forbidden" in text:
        return "Forbidden"
    if any(marker in text for marker in (
        "APIConnectionError",
        "APITimeoutError",
        "Connection error",
        "ReadTimeout",
        "Timeout",
        "timed out",
    )):
        return "transient"
    response = getattr(e, "response", None)
    status = getattr(response, "status_code", None)
    if status in (500, 502, 503, 504):
        return "transient"
    return "unknown"


def _sleep_for_llm_retry(retry_idx: int, detail_code: str) -> Optional[int]:
    delays = (
        _LLM_ALLOCATION_RETRY_DELAYS
        if detail_code == "Throttling.AllocationQuota"
        else _LLM_RETRY_DELAYS
    )
    if retry_idx >= len(delays):
        return None
    return delays[retry_idx]


def _compaction_reminder(trigger_tokens: int, keep_tokens: int) -> str:
    return (
        "[context-maintenance]\n"
        f"Conversation reached the {trigger_tokens:,}-token limit. "
        f"After this turn, history will be compacted to the latest ~{keep_tokens:,} estimated tokens. "
        "Call write_memory_doc now if important details should be kept."
    )


def _compaction_notice(trigger_tokens: int, keep_tokens: int) -> str:
    return (
        "[context-maintenance]\n"
        f"Conversation reached the {trigger_tokens:,}-token limit. "
        f"History was compacted to the latest ~{keep_tokens:,} estimated tokens."
    )


class ReActAgent:
    def __init__(self, base_url: str, run_id: str, agent_id: str,
                 *, openai_client: OpenAI, model: str,
                 max_hops_per_step: int = DEFAULT_MAX_HOPS,
                 temperature: Optional[float] = None,
                 context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
                 compact_trigger_tokens: int = DEFAULT_COMPACT_TRIGGER_TOKENS,
                 compact_keep_tokens: int = DEFAULT_COMPACT_KEEP_TOKENS):
        self.base = base_url.rstrip("/")
        self.run_id = run_id
        self.agent_id = agent_id
        self.client = MerchantBenchToolClient(base_url, run_id, agent_id)
        self.system_prompt: Optional[str] = None
        self.language: str = "en"
        self.openai = openai_client
        self.model = model
        self.max_hops_per_step = max_hops_per_step
        self.temperature = temperature
        self.context_window_tokens = context_window_tokens
        self.compact_trigger_tokens = compact_trigger_tokens
        self.compact_keep_tokens = compact_keep_tokens
        self.last_prompt_tokens = 0
        self.compaction_pending = False
        self._pending_pre_assistant_messages: list[dict] = []
        self._pending_runtime_health = {
            "provider_api_failed_attempts": 0,
            "retry_exhausted": 0,
        }
        self.history: list[dict] = []

    def register(self) -> None:
        self.client.register(
            framework=FRAMEWORK,
            model=self.model,
            version=VERSION,
            extra={"max_hops_per_step": self.max_hops_per_step,
                   "context_window_tokens": self.context_window_tokens,
                   "compact_trigger_tokens": self.compact_trigger_tokens,
                   "compact_keep_tokens": self.compact_keep_tokens,
                   "runtime_health_version": 1,
                   "runtime_health_capabilities": {
                       "provider_api_failed_attempts": "reported",
                       "retry_exhausted": "reported",
                       "memory_compactions": "reported",
                       "skills_evolutions": "not_applicable",
                   }},
        )

    def _note_runtime_health(self, key: str, count: int = 1) -> None:
        pending = getattr(self, "_pending_runtime_health", None)
        if not isinstance(pending, dict):
            pending = {}
            self._pending_runtime_health = pending
        pending[key] = int(pending.get(key, 0) or 0) + int(count)

    def _ack_runtime_health(self) -> None:
        self._pending_runtime_health = {
            "provider_api_failed_attempts": 0,
            "retry_exhausted": 0,
        }

    def _update_brief_from_obs(self, obs: dict) -> None:
        """Extract brief from observation packet (included on first observation)."""
        brief = obs.get("brief")
        if brief and self.system_prompt is None:
            base_prompt = brief.get("system_prompt", "") or ""
            self.system_prompt = base_prompt
            self.language = brief.get("language", "en")

    def _append_tool_results(self, act_resp: dict) -> None:
        for tr in act_resp.get("tool_results", []):
            self.history.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "name": tr["name"],
                "content": tr["content"],
            })

    def _remember_act(self, assistant_msg: dict, act_resp: dict) -> None:
        if _is_only_end_of_step(assistant_msg):
            content = assistant_msg.get("content") or ""
            if content and not content.startswith("[llm-error]"):
                history_msg = {"role": "assistant", "content": content}
                if assistant_msg.get("reasoning_content"):
                    history_msg["reasoning_content"] = assistant_msg["reasoning_content"]
                self.history.append(history_msg)
            return
        self.history.append(assistant_msg)
        self._append_tool_results(act_resp)

    def _redact_last_tool_turn_for_policy_retry(self) -> int:
        idx = len(self.history) - 1
        while idx >= 0 and self.history[idx].get("role") != "tool":
            idx -= 1
        redacted = 0
        while idx >= 0 and self.history[idx].get("role") == "tool":
            self.history[idx]["content"] = _POLICY_REDACTED_TOOL_CONTENT
            redacted += 1
            idx -= 1
        return redacted

    def _chat_completion_with_retries(self, request: dict, verbose: bool) -> Any:
        self._last_llm_error_detail_code = "unknown"
        self._last_llm_error_retries = 0
        retries_done = 0
        forbidden_retried = False

        while True:
            try:
                return self.openai.chat.completions.create(**request)
            except Exception as e:
                self._note_runtime_health("provider_api_failed_attempts")
                detail_code = _llm_detail_code(e)
                self._last_llm_error_detail_code = detail_code
                self._last_llm_error_retries = retries_done

                if detail_code == "Forbidden" and not forbidden_retried:
                    self._redact_last_tool_turn_for_policy_retry()
                    request = {
                        **request,
                        "messages": _build_llm_messages(
                            self.system_prompt,
                            self.history,
                            self.context_window_tokens,
                            exclude_system=_is_claude_model(self.model),
                        ),
                    }
                    forbidden_retried = True
                    # Don't increment retries_done - Forbidden retry is a special
                    # one-shot attempt, not counted in the regular retry budget
                    self._last_llm_error_retries = retries_done
                    continue

                if detail_code not in _RETRYABLE_LLM_DETAIL_CODES:
                    if retries_done > 0 or forbidden_retried:
                        self._note_runtime_health("retry_exhausted")
                    raise
                delay = _sleep_for_llm_retry(retries_done, detail_code)
                if delay is None:
                    self._note_runtime_health("retry_exhausted")
                    raise
                if verbose:
                    print(
                        f"[llm retry] detail_code={detail_code} "
                        f"retry={retries_done + 1} sleep={delay}s",
                        file=sys.stderr,
                    )
                time.sleep(delay)
                retries_done += 1
                self._last_llm_error_retries = retries_done

    def _maybe_add_compaction_reminder(self, tools: list[dict]) -> None:
        if self.compaction_pending:
            return
        if self.compact_trigger_tokens <= 0:
            return
        last_prompt_tokens = int(getattr(self, "last_prompt_tokens", 0) or 0)
        trigger_tokens = (
            last_prompt_tokens
            if last_prompt_tokens > 0
            else _history_token_estimate(self.history)
        )
        if trigger_tokens < self.compact_trigger_tokens:
            return
        if not _tool_available(tools, "write_memory_doc"):
            self.history = _trim_messages_to_token_budget(
                self.history,
                self.compact_keep_tokens,
            )
            notice_msg = {
                "role": "user",
                "content": _compaction_notice(
                    self.compact_trigger_tokens,
                    self.compact_keep_tokens,
                ),
            }
            self.history.append(notice_msg)
            self.compaction_pending = True
            self._pending_pre_assistant_messages = [dict(notice_msg)]
            return
        reminder = _compaction_reminder(
            self.compact_trigger_tokens,
            self.compact_keep_tokens,
        )
        reminder_msg = {
            "role": "user",
            "content": reminder,
        }
        self.history.append(reminder_msg)
        self._pending_pre_assistant_messages = [dict(reminder_msg)]
        self.compaction_pending = True

    def _act_messages(self, assistant_msg: dict) -> list[dict]:
        # Prepend pending pre-assistant messages (e.g., compaction reminders).
        # They are cleared only after /act succeeds, otherwise a stale-step or
        # transient /act failure would permanently lose the reminder.
        return [*self._pending_pre_assistant_messages, assistant_msg]

    def _compact_history_if_pending(self) -> None:
        if not self.compaction_pending:
            return
        self.history = _trim_messages_to_token_budget(
            self.history,
            self.compact_keep_tokens,
        )
        self.last_prompt_tokens = 0
        self.compaction_pending = False
        self._pending_pre_assistant_messages = []

    def _act_context(self, llm_messages: Optional[list[dict]] = None) -> Optional[dict]:
        prompt_tokens = int(getattr(self, "last_prompt_tokens", 0) or 0)
        if prompt_tokens <= 0:
            if llm_messages is None:
                llm_messages = _build_llm_messages(
                    self.system_prompt,
                    self.history,
                    self.context_window_tokens,
                    exclude_system=_is_claude_model(self.model),
                )
            prompt_tokens = _history_token_estimate(llm_messages)
        context = {}
        if prompt_tokens > 0:
            context["tokens"] = prompt_tokens
        if self.compaction_pending:
            context["compacted"] = True
        pending = getattr(self, "_pending_runtime_health", {})
        for key in ("provider_api_failed_attempts", "retry_exhausted"):
            count = int(pending.get(key, 0) or 0) if isinstance(pending, dict) else 0
            if count:
                context[key] = count
        return context or None

    def _drive_step(self, obs: dict, t_key: int, verbose: bool) -> None:
        tools = self.client.tools()
        self._update_brief_from_obs(obs)
        # Record history length before appending observation so we can truncate
        # back on HTTP 425 (stale step) to avoid accumulating duplicate observations.
        history_len_before = len(self.history)
        self.history.append({"role": "user", "content": obs.get("text", "")})

        for hop in range(self.max_hops_per_step):
            self._maybe_add_compaction_reminder(tools)
            is_claude = _is_claude_model(self.model)
            llm_messages = _build_llm_messages(
                self.system_prompt,
                self.history,
                self.context_window_tokens,
                exclude_system=is_claude,
            )
            try:
                request = {
                    "model": self.model,
                    "messages": llm_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "extra_headers": {
                        "x-idealab-session-id": self.run_id,
                    },
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                if is_claude:
                    request["messages"] = _add_cache_breakpoints(llm_messages)
                    if self.system_prompt:
                        request["extra_body"] = {
                            "extendParams": {
                                "system": [
                                    {
                                        "type": "text",
                                        "text": self.system_prompt,
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                ],
                            },
                        }
                resp = self._chat_completion_with_retries(request, verbose)
            except Exception as e:
                if verbose:
                    print(f"[llm error hop={hop}] {e}", file=sys.stderr)
                detail_code = getattr(
                    self, "_last_llm_error_detail_code", _llm_detail_code(e))
                retries = getattr(self, "_last_llm_error_retries", 0)
                self._force_end_of_step(
                    f"[llm-error] {type(e).__name__}: {e} "
                    f"detail_code={detail_code} retries={retries}"
                )
                return

            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            token_usage = None
            self.last_prompt_tokens = 0
            if usage is not None:
                inp = int(getattr(usage, "prompt_tokens", 0) or 0)
                out = int(getattr(usage, "completion_tokens", 0) or 0)
                cached = _extract_cached_tokens(usage)
                self.last_prompt_tokens = inp
                token_usage = {
                    "input": max(0, inp - cached),
                    "output": out,
                    "cache_read": cached,
                    "total": inp + out,
                }
            context = self._act_context(request["messages"])

            tool_calls_raw = msg.tool_calls or []
            thought = msg.content or ""
            reasoning_content = _extract_reasoning_content(msg)

            if not tool_calls_raw:
                assistant_msg = {
                    "role": "assistant",
                    "content": thought or "[fallback] no tool call — release hook",
                    "tool_calls": [_end_of_step_tool_call()],
                }
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                try:
                    act_resp = self.client.act(
                        token_usage=token_usage,
                        messages=self._act_messages(assistant_msg),
                        context=context,
                    )
                    self._ack_runtime_health()
                except requests.HTTPError as e:
                    if verbose:
                        print(f"[act error hop={hop}] {e}", file=sys.stderr)
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 425:
                        # Stale step: env already advanced. Discard this turn's
                        # messages (observation + any hops) to prevent accumulation
                        # of duplicate observations on re-observe.
                        self.history = self.history[:history_len_before]
                    else:
                        self._force_end_of_step(
                            f"[act-error] {type(e).__name__}: {e}"
                        )
                    return
                self._remember_act(assistant_msg, act_resp)
                self._compact_history_if_pending()
                return

            # Build assistant message in OpenAI format
            tc_block = []
            for tc in tool_calls_raw:
                tc_block.append({
                    "id": tc.id or f"call_{hop}_{tool_calls_raw.index(tc)}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                })

            assistant_msg = {
                "role": "assistant",
                "content": thought,
                "tool_calls": tc_block,
            }
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content

            # Send to env — executes tools and records trace
            try:
                act_resp = self.client.act(
                    token_usage=token_usage,
                    messages=self._act_messages(assistant_msg),
                    context=context,
                )
                self._ack_runtime_health()
            except requests.HTTPError as e:
                if verbose:
                    print(f"[act error hop={hop}] {e}", file=sys.stderr)
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 425:
                    # Stale step: env already advanced. Discard this turn's
                    # messages (observation + any hops) to prevent accumulation
                    # of duplicate observations on re-observe.
                    self.history = self.history[:history_len_before]
                else:
                    self._force_end_of_step(
                        f"[act-error] {type(e).__name__}: {e}"
                    )
                return

            self._remember_act(assistant_msg, act_resp)
            self._compact_history_if_pending()

            # Check if step is done (end_of_step was in the tool_calls)
            if act_resp.get("step_done"):
                if verbose:
                    print(f"  hop {hop}: end_of_step", file=sys.stderr)
                return

            if verbose:
                names = ",".join(tc["function"]["name"] for tc in tc_block)
                print(f"  hop {hop}: {names}", file=sys.stderr)

        # Hop budget exhausted
        if verbose:
            print(f"[budget] t={t_key} hop budget exhausted, force EOS",
                  file=sys.stderr)
        self._force_end_of_step("[fallback] hop budget exhausted — release hook")

    def _force_end_of_step(self, reason: str) -> None:
        assistant_msg = {
            "role": "assistant",
            "content": reason,
            "tool_calls": [_end_of_step_tool_call()],
        }
        try:
            act_resp = self.client.act(
                messages=self._act_messages(assistant_msg),
                context=self._act_context(),
            )
            self._ack_runtime_health()
            self._remember_act(assistant_msg, act_resp)
            self._compact_history_if_pending()
        except requests.HTTPError:
            # /act failed (e.g., stale step). Reset compaction state so the
            # reminder can be re-added on the next step instead of staying
            # permanently pending and blocking future reminders.
            if self.compaction_pending:
                self.compaction_pending = False
                self._pending_pre_assistant_messages = []
                self.last_prompt_tokens = 0

    def run(self, max_steps: int = DEFAULT_MAX_STEPS,
            verbose: bool = True) -> None:
        self.register()
        if verbose:
            print(f"[{FRAMEWORK}] registered model={self.model} max_steps={max_steps} "
                  f"max_hops={self.max_hops_per_step} "
                  f"context_window={self.context_window_tokens} "
                  f"compact={self.compact_trigger_tokens}->{self.compact_keep_tokens}",
                  file=sys.stderr)
        while True:
            try:
                obs = self.client.observation()
            except requests.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 410:
                    if verbose:
                        print("[done] run finished", file=sys.stderr)
                    return
                if verbose:
                    print(f"[obs error] {e}", file=sys.stderr)
                continue
            tk = obs["tick"]
            t = (tk["day"] - 1) * 24 + tk["hour"]
            if t >= max_steps:
                if verbose:
                    print(f"[done] reached t={t}", file=sys.stderr)
                return
            if verbose:
                print(f"[step] day={tk['day']} hour={tk['hour']}", file=sys.stderr)
            try:
                self._drive_step(obs, t, verbose)
            except Exception as e:
                if verbose:
                    print(f"[step crash] t={t} {type(e).__name__}: {e}",
                          file=sys.stderr)
                self._force_end_of_step(
                    f"[step-crash] {type(e).__name__}: {e}"
                )


def _build_openai_client() -> tuple[OpenAI, str]:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("MODEL_NAME", "qwen3.5-27b")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set — copy .env.example to .env and fill it in,"
            " or export OPENAI_API_KEY before running.")
    if not base_url:
        raise SystemExit(
            "OPENAI_BASE_URL not set — set it in .env or the environment.")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    return client, model


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id",
                    default=(os.environ.get("MERCHANTBENCH_RUN_ID")
                             or os.environ.get("REALSHOP_RUN_ID")),
                    help="env-run id; falls back to MERCHANTBENCH_RUN_ID")
    ap.add_argument("--base-url",
                    default=os.environ.get(
                        "MERCHANTBENCH_BASE_URL",
                        os.environ.get("REALSHOP_BASE_URL", "http://localhost:5050"),
                    ))
    ap.add_argument("--agent-id",
                    default=os.environ.get(
                        "MERCHANTBENCH_AGENT_ID",
                        os.environ.get("REALSHOP_AGENT_ID", "agent_0"),
                    ))
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("MAX_STEPS", DEFAULT_MAX_STEPS)))
    ap.add_argument("--max-hops", type=int,
                    default=int(os.environ.get("MAX_HOPS_PER_STEP", DEFAULT_MAX_HOPS)))
    ap.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE"),
                    help=("Optional sampling temperature. Omitted by default so "
                          "each model/provider uses its own default."))
    ap.add_argument("--model",
                    default=None,
                    help="OpenAI-compatible model name; overrides MODEL_NAME")
    ap.add_argument("--context-window-tokens", type=int,
                    default=int(os.environ.get(
                        "CONTEXT_WINDOW_TOKENS",
                        DEFAULT_CONTEXT_WINDOW_TOKENS,
                    )))
    ap.add_argument("--compact-trigger-tokens", type=int,
                    default=int(os.environ.get(
                        "COMPACT_TRIGGER_TOKENS",
                        DEFAULT_COMPACT_TRIGGER_TOKENS,
                    )))
    ap.add_argument("--compact-keep-tokens", type=int,
                    default=int(os.environ.get(
                        "COMPACT_KEEP_TOKENS",
                        DEFAULT_COMPACT_KEEP_TOKENS,
                    )))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.run_id:
        ap.error("--run-id is required (MERCHANTBENCH_RUN_ID or legacy REALSHOP_RUN_ID)")

    openai_client, default_model = _build_openai_client()
    model = args.model or default_model
    ReActAgent(
        args.base_url, args.run_id, args.agent_id,
        openai_client=openai_client, model=model,
        max_hops_per_step=args.max_hops,
        temperature=args.temperature,
        context_window_tokens=args.context_window_tokens,
        compact_trigger_tokens=args.compact_trigger_tokens,
        compact_keep_tokens=args.compact_keep_tokens,
    ).run(max_steps=args.max_steps, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
