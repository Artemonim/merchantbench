"""MerchantBench single-file tool client.

Copy this file into your agent project. It is framework-neutral: the agent
sends an OpenAI-format assistant message or full messages batch to a single
/act endpoint; the env executes MerchantBench tool calls and returns results.

What this SDK does:
  - Fetches /tools/schema once on construction and caches the OpenAI tool
    schemas. `client.tools()` returns the list you pass to your LLM's
    `tools=` argument.
  - `client.register(framework=..., model=..., ...)` — POST /agent/register.
  - `client.observation(timeout=30)` long-polls the env's per-step
    notification. Auto-retries on HTTP 408. HTTP 410 = no more agent hooks.
    The first observation automatically includes a `brief` field with the
    env-issued system prompt + platform rules.
  - `client.act(assistant_message, token_usage=None, messages=None,
    context=None)` — POST
    /act with an assistant message or full OpenAI-format messages batch. The
    env executes `merchantbench_env` tool calls, preserves non-env/native traces
    tagged with `tool_origin`, and stores the protocol log.
  - Auth: pass `agent_token=` or set `MERCHANTBENCH_AGENT_TOKEN`; the SDK sends it
    as `Authorization: Bearer ...` on every request.

Dependencies: requests.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests


class MerchantBenchToolClient:
    def __init__(self, base_url: str, run_id: str, agent_id: str,
                 timeout: float = 600.0,
                 observation_timeout: float = 30.0,
                 observation_connection_retries: int = 3,
                 agent_token: Optional[str] = None):
        self.base = base_url.rstrip("/")
        self.run_id = run_id
        self.agent_id = agent_id
        self.timeout = timeout
        self.observation_timeout = observation_timeout
        self.observation_connection_retries = observation_connection_retries
        self._session = requests.Session()
        token = agent_token or os.environ.get("MERCHANTBENCH_AGENT_TOKEN")
        if token:
            self._session.headers.update({"Authorization": f"Bearer {token}"})
        self._schema: list[dict] = []
        self._latest_env_t: Optional[int] = None
        self.refresh_schema()

    # ---------- schema ----------

    def refresh_schema(self) -> None:
        url = f"{self.base}/runs/{self.run_id}/tools/schema"
        r = self._session.get(url, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        self._schema = body.get("tools", [])

    def tools(self) -> list[dict]:
        """OpenAI-format tool schemas for chat completion's tools= argument."""
        return [s["openai"] for s in self._schema]

    # ---------- one-shot lifecycle ----------

    def register(self, *, framework: str = "unknown",
                 model: Optional[str] = None,
                 version: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 extra: Optional[dict] = None) -> dict:
        url = f"{self.base}/runs/{self.run_id}/agent/register"
        body: dict[str, Any] = {"agent_id": self.agent_id, "framework": framework}
        if model is not None:
            body["model"] = model
        if version is not None:
            body["version"] = version
        if prompt_template is not None:
            body["prompt_template"] = prompt_template
        if extra is not None:
            body["extra"] = extra
        r = self._session.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---------- observation ----------

    def observation(self, timeout: Optional[float] = None,
                    nowait: bool = False) -> dict:
        """Long-poll for the next hook to open. Auto-retries 408.
        Raises HTTPError(410) when no more agent hooks will open."""
        url = (f"{self.base}/runs/{self.run_id}/agents/{self.agent_id}"
               f"/observation")
        eff_timeout = timeout if timeout is not None else self.observation_timeout
        params = {"nowait": "1"} if nowait else {"timeout": str(eff_timeout)}
        http_timeout = eff_timeout + 5 if not nowait else self.timeout
        connection_errors = 0
        while True:
            try:
                r = self._session.get(url, params=params, timeout=http_timeout)
            except requests.ReadTimeout:
                if not nowait:
                    continue
                raise
            except requests.ConnectionError:
                if nowait:
                    raise
                connection_errors += 1
                if connection_errors > getattr(self, "observation_connection_retries", 3):
                    raise
                continue
            except requests.Timeout:
                if nowait:
                    raise
                connection_errors += 1
                if connection_errors > getattr(self, "observation_connection_retries", 3):
                    raise
                continue
            connection_errors = 0
            if r.status_code == 408 and not nowait:
                continue
            r.raise_for_status()
            packet = r.json()
            tick = packet.get("tick") or {}
            if tick.get("step") is not None:
                self._latest_env_t = int(tick["step"])
            else:
                day = int(tick.get("day", 0))
                hour = int(tick.get("hour", 0))
                if day > 0:
                    self._latest_env_t = (day - 1) * 24 + hour
            return packet

    def latest_env_t(self) -> Optional[int]:
        return self._latest_env_t

    # ---------- act (unified tool execution) ----------

    def act(self, assistant_message: Optional[dict] = None,
            token_usage: Optional[dict] = None, *,
            messages: Optional[list[dict]] = None,
            context: Optional[dict] = None) -> dict:
        """Send an assistant message or messages batch to the env.

        Args:
            assistant_message: OpenAI-format assistant msg with tool_calls.
                Example: {"role": "assistant", "content": "...",
                          "tool_calls": [{"id": "call_1", "type": "function",
                           "function": {"name": "...", "arguments": "..."}}]}
                Kept for the common one-assistant-message case.
            token_usage: Optional LLM token counts for cost tracking.
                Example: {"input": 200, "output": 50,
                          "cache_read": 0, "cache_write": 0,
                          "reasoning": 0, "total": 250}
            messages: Optional full OpenAI-format messages payload for this
                /act turn. Assistant tool calls tagged with
                tool_origin="merchantbench_env" are executed by the env; other
                origins are recorded for trace continuity but not executed.
            context: Optional agent-reported context metrics for this turn.
                Supported fields include {"tokens": int, "compacted": bool,
                "provider_api_failed_attempts": int,
                "retry_exhausted": int, "skills_evolutions": int}.

        Returns:
            {"ok": True, "turn_idx": N, "tool_results": [...],
             "step_done": bool, "hook_released": bool}
            tool_results items: {"tool_call_id": str, "name": str,
                                 "tool_origin": "merchantbench_env",
                                 "content": str}
        """
        url = (f"{self.base}/runs/{self.run_id}/agents/{self.agent_id}/act")
        if messages is None:
            if assistant_message is None:
                raise ValueError("assistant_message or messages is required")
            messages = [assistant_message]
        body: dict[str, Any] = {"messages": messages}
        if token_usage is not None:
            body["token_usage"] = token_usage
        if context is not None:
            body["context"] = context
        headers: dict[str, str] = {}
        if self._latest_env_t is not None:
            headers["X-Agent-Step"] = str(self._latest_env_t)
        r = self._session.post(url, json=body, headers=headers,
                               timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def record_usage(
        self,
        token_usage: dict,
        *,
        usage_id: str,
        source: str = "auxiliary",
        step: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
    ) -> dict:
        """Idempotently record delayed auxiliary LLM usage.

        Unlike :meth:`act`, this remains valid after a hook closes so the last
        checkpoint review of a run is not lost from the environment ledger.
        """
        url = f"{self.base}/runs/{self.run_id}/agents/{self.agent_id}/usage"
        body: dict[str, Any] = {
            "usage_id": usage_id,
            "source": source,
            "token_usage": token_usage,
        }
        effective_step = self._latest_env_t if step is None else step
        if effective_step is not None:
            body["step"] = int(effective_step)
        for key, value in (
            ("model", model),
            ("provider", provider),
            ("cost_status", cost_status),
            ("cost_source", cost_source),
        ):
            if value is not None:
                body[key] = str(value)
        if cost_usd is not None:
            body["cost_usd"] = float(cost_usd)
        r = self._session.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
