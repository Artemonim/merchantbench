"""Submission stub — replace this with your agent.

Minimum contract:
  1. Read MERCHANTBENCH_BASE_URL / MERCHANTBENCH_RUN_ID / MERCHANTBENCH_AGENT_ID /
     MERCHANTBENCH_AGENT_TOKEN from env. The SDK reads MERCHANTBENCH_AGENT_TOKEN
     automatically and sends it as the bearer token.
  2. Use sdk.merchantbench_tool_client.MerchantBenchToolClient:
       client.observation()  -> long-poll the next env tick
       client.tools()        -> OpenAI-format tool schemas for your LLM
       client.act(msg, ...)  -> send assistant message, env executes tools
       client.act(messages=[...]) -> send a full OpenAI messages batch
  3. Include "end_of_step" in tool_calls to release the per-step hook.
  4. Exit cleanly when observation() raises HTTP 410 (no more agent hooks).

For a real reference implementation, see
  agent/baselines/react_160k_compact_30k.py      (long-context ReAct loop)
  agent/baselines/rule_based.py                  (daily-report/random rule loop)

This stub does the bare minimum: it just calls end_of_step on every step.
"""
from __future__ import annotations

import os
import sys

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sdk.merchantbench_tool_client import MerchantBenchToolClient


def main() -> int:
    base_url = os.environ.get("MERCHANTBENCH_BASE_URL", "http://localhost:5050")
    run_id = os.environ.get("MERCHANTBENCH_RUN_ID")
    agent_id = os.environ.get("MERCHANTBENCH_AGENT_ID", "agent_0")
    if not run_id:
        raise SystemExit("MERCHANTBENCH_RUN_ID is required (set by the harness).")

    client = MerchantBenchToolClient(base_url, run_id, agent_id)
    client.register(framework="stub", model="noop", version="0.1")

    while True:
        try:
            obs = client.observation()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 410:
                return 0
            raise
        # Replace this with your LLM decision logic.
        client.act({
            "role": "assistant",
            "content": "[noop] releasing hook",
            "tool_calls": [{
                "id": "call_eos",
                "type": "function",
                "function": {"name": "end_of_step", "arguments": "{}"},
            }],
        })


if __name__ == "__main__":
    sys.exit(main())
