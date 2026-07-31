# `agent/` — MerchantBench SDK + baselines + submission template

This directory is the **public** half of MerchantBench. Submitters see this; the
env-side simulator (`env/`) is opaque to them. You can carve `agent/` into
its own repo or PyPI package without touching `env/`.

| Path | Role |
| --- | --- |
| [`sdk/`](sdk/) | The public tool client every submission imports. **Stable interface — don't fork.** |
| [`baselines/`](baselines/) | Reference implementations: `rule_based.py` (daily-report/random deterministic loop) and `react_160k_compact_30k.py` (long-context ReAct with optional memory-aware compaction when memory tools are enabled). |
| [`submission_template/`](submission_template/) | Copy this and replace `my_agent.py` with your code. |

The root README provides the artifact-level installation and reproduction
workflow.

---

## 1. The 30-second submission flow

You ship a Docker image. The evaluator runs it against a hidden scenario and
a fixed `master_seed`, then reports final net assets. That's the entire interaction.

Two equivalent flows:

**Flow A — edit the template in place** (simplest):

```bash
# 1. Edit agent/submission_template/my_agent.py with your agent code.
# 2. Build from repo root. Build context is agent/ so the Dockerfile
#    can COPY sdk/ from a sibling directory.
docker build -f agent/submission_template/Dockerfile -t my-agent:v1 agent/
# 3. Hand the image (or a tarball) to the evaluator.
```

**Flow B — copy template to a separate dir** (recommended for teams):

```bash
# 1. Copy under agent/ so the Dockerfile's COPY paths still resolve.
cp -r agent/submission_template/ agent/my_submission/

# 2. Replace agent/my_submission/my_agent.py with your agent.
#    Edit agent/my_submission/Dockerfile so the COPY line points to
#    `my_submission/my_agent.py` (instead of `submission_template/...`).

# 3. Build:
docker build -f agent/my_submission/Dockerfile -t my-agent:v1 agent/
```

Either way, only the runtime contract below matters for evaluation.

### Runtime contract

Your image's entrypoint must, at runtime:

1. Read four environment variables (the evaluator harness injects them):

   | Variable | Meaning |
   | --- | --- |
   | `MERCHANTBENCH_BASE_URL` | URL of the env (e.g. `http://env:5000`) |
   | `MERCHANTBENCH_RUN_ID`   | The run your agent should drive |
   | `MERCHANTBENCH_AGENT_ID` | Default `agent_0` |
   | `MERCHANTBENCH_AGENT_TOKEN` | Bearer token for the agent-facing env API |

2. Drive the run via [`sdk/merchantbench_tool_client.py`](sdk/merchantbench_tool_client.py):

   The SDK reads `MERCHANTBENCH_AGENT_TOKEN` automatically and sends it as the
   `Authorization: Bearer ...` header.

   - `register()` once (POST `/runs/<rid>/agent/register`)
   - long-poll `client.observation()` in a loop
   - send assistant messages with tool_calls via `client.act(msg, ...)`
   - include `end_of_step` in tool_calls to release the per-step hook
   - exit cleanly when `observation()` raises HTTP 410 (agent-facing run is terminal)

3. Receive your LLM provider credentials via env vars (the harness forwards
   `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME` from the evaluator's
   `.env`). **Do not bake API keys into the image.**

---

## 2. The SDK (`sdk/merchantbench_tool_client.py`)

A single-file Python client. Copy it into your project, or import it
directly if your build context includes this directory. Only depends on
`requests`.

It is intentionally framework-neutral:

- **`client.tools()`** — caches the env's `/tools/schema` and returns the
  OpenAI-format tool list, ready to pass straight to your LLM's `tools=`
  argument.
- **`client.register(framework=..., model=..., ...)`** — POST
  `/agent/register` once at startup. Required when scenario has
  `agent.require_register: true`; harmless otherwise.
- **`client.observation(timeout=30)`** — long-polls the env's per-step
  notification. Auto-retries on HTTP 408 (long-poll expired); raises
  `requests.HTTPError` on HTTP 410 when the operating horizon is over and
  no more agent hooks will open. The env may still be draining active
  orders internally, but the agent loop should treat 410 as a clean exit.
  **The first observation automatically includes a `brief` field** with the
  system prompt + platform rules; no need to call `brief()` separately. The
  prompt injects the active scenario's exact shop-rating outcomes, aggregation
  settings, star thresholds, and traffic multipliers when outcome-based rating
  is enabled, so submissions do not need to hard-code leaderboard values. The
  observation `tick` includes raw `step` plus agent-facing `{day, hour}` /
  optional calendar datetime.
- **`client.act(assistant_message, token_usage=None, messages=None)`** —
  POST `/act` with either a single OpenAI-format assistant message or a full
  OpenAI-format messages batch. Tool calls default to `tool_origin:
  "merchantbench_env"` for legacy agents and are executed by env; non-env/native
  tool calls tagged with another `tool_origin` are preserved in trace but not
  executed. A thought-only assistant message is also valid for trace
  continuity; send a follow-up `end_of_step` tool call to release the hook.
- **`client.record_usage(token_usage, usage_id=..., source=..., model=...)`** —
  idempotently records delayed auxiliary LLM work, such as checkpoint review.
  This cost-only call remains valid after the hook closes; it neither creates
  an agent turn nor executes tools. Auxiliary work routed to another model
  should also include `provider`, `cost_usd`, `cost_status`, and `cost_source`;
  unknown cross-model prices remain explicitly unpriced instead of being
  charged at the foreground model's scenario rate.
- **Optional run-local memory tools** — `read_memory_doc` and
  `write_memory_doc` still exist, but the default benchmark scenario denies
  them. If a scenario explicitly enables them, they read/write
  `runs/<run_id>/agent/memory/<agent_id>.md` through the same `/act`
  trace path.

It deliberately does **not**:

- bundle an LLM provider — wire the schemas into whichever client you use
- decide how to structure multi-hop reasoning — see
  [`baselines/`](baselines/) for two opinionated takes

### Why the `X-Agent-Step` header matters

Under network latency, an `act()` call decided at `t=42` might arrive when
env has already advanced to `t=43`. Without protection, the mutation lands
on the wrong simulation step. The SDK guards against this:

- `client.observation()` records the env step it just observed.
- Every subsequent `client.act(...)` auto-injects `X-Agent-Step: <N>`.
- Env compares against current `env.t`. Mismatch → HTTP 425 with
  `{error: "stale_step", agent_step: N, env_step: M}`.

On 425, **discard all pending work for this turn, re-fetch observation, and
re-plan**.

---

## 3. Minimal end-to-end loop

```python
from sdk.merchantbench_tool_client import MerchantBenchToolClient

client = MerchantBenchToolClient(base_url, run_id, agent_id)
client.register(framework="my-framework", model="my-model")

while True:
    try:
        obs = client.observation()       # long-poll the next tick
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 410:
            return                       # no more agent hooks; clean exit
        raise

    # Decide what to do based on obs (rules, LLM, whatever).
    # The observation is intentionally slim (tick + cash + event counts) —
    # use search_products / query_my_listings / get_product_detail in
    # tool_calls for detailed info.

    # Send an assistant message with tool_calls; env executes and returns results.
    # Full messages batches are also accepted via client.act(messages=[...]).
    result = client.act({
        "role": "assistant",
        "content": "Adjusting price for P00017",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "adjust_price",
                          "arguments": '{"items": [{"product_id": "P00017", "new_price": 41.5}]}'}},
            {"id": "call_eos", "type": "function",
             "function": {"name": "end_of_step", "arguments": "{}"}},
        ],
    }, token_usage={"input": 200, "output": 50, "cache_read": 0, "total": 250})
    # result = {"ok": True, "turn_idx": 0, "tool_results": [...],
    #           "step_done": True, "hook_released": True}
```

For a full LLM-driven loop with multi-hop ReAct, real provider-reported
token usage, a hop budget, and compaction from 160k estimated tokens to the
latest 30k, see
[`baselines/react_160k_compact_30k.py`](baselines/react_160k_compact_30k.py).
It only prompts for memory writes when `write_memory_doc` is available in the
scenario tool schema.
To run the current batch helper against a local env, use
[`../scripts/run_batch.py`](../scripts/run_batch.py).

For a deterministic non-LLM loop demonstrating the same wire protocol with
daily-report and random sourcing modes
(register → observation → act → end_of_step),
see [`baselines/rule_based.py`](baselines/rule_based.py).

---

## 4. Run a baseline locally (no Docker)

You need an env server up first:

```bash
# In one terminal:
cd ../env && python run.py --port 5050
```

In another terminal, create a run and start a baseline:

```bash
cd ..                                                       # repo root
.venv/bin/python -m pip install -r agent/requirements.txt
cp .env.example .env                                        # for ReAct only

RID=$(curl -s -X POST http://127.0.0.1:5050/runs \
  -H 'Content-Type: application/json' \
  -d '{"scenario_path":"scenarios/default.yaml","auto_start":true,"interval_ms":200}' \
  | .venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")

# Rule-based random sourcing (deterministic, no LLM):
.venv/bin/python agent/baselines/rule_based.py --selection-mode random --selection-seed 42 --run-id "$RID" --base-url http://127.0.0.1:5050

# Or LLM-driven (needs OPENAI_API_KEY etc. in .env):
.venv/bin/python agent/baselines/react_160k_compact_30k.py --run-id "$RID" --base-url http://127.0.0.1:5050
```

Or skip the curl: open `http://127.0.0.1:5050/new_run` in a browser and
choose **Human**, **Rule-based**, `react_160k_compact_30k`, or **Hermes**
when an external adapter checkout is configured. The dashboard starts the
run immediately and, for subprocess baselines/adapters, watches their trace
turn by turn.

Hermes bootstrap runs use the external Hermes checkout as runtime code, but
MerchantBench creates a fresh profile per run at
`runs/<run_id>/agent/hermes_home`. The profile is seeded from the checkout's
`skills/` directory, and Hermes tool commands default to
`runs/<run_id>/agent/hermes_workspace`, so parallel and sequential experiments
do not share Hermes memory, sessions, or skill edits.

### Configuration for the ReAct baseline

`react_160k_compact_30k.py` reads from `.env` (in the repo root) at startup via
`python-dotenv`:

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://...
MODEL_NAME=qwen3.5-27b
```

When run inside a hosted-eval Docker container, the harness forwards these
vars from the operator's `.env` via `--env-file`. Submission images **never
bundle credentials**.

---

## 5. Test locally with the harness

```bash
# 1. Build your agent image (from repo root)
docker build -f agent/submission_template/Dockerfile -t my-agent:v1 agent/

# 2. Build the env image used by the hosted-eval harness
docker build -f eval/env_image/Dockerfile -t merchantbench-env:dev env/

# 3. Let the harness start env + agent containers on a bridge network
.venv/bin/python -m eval.run_eval --agent-image my-agent:v1 \
    --host-port 5050 --keep-containers \
    --output result.json
```

`--keep-containers` keeps the env container alive after the run finishes so
you can browse the dashboard at `http://127.0.0.1:5050/dashboard?run_id=<run_id>` and
inspect the trace.

---

## 6. Result metric (what you're being optimized for)

```
score = final_net_assets
```

`net_assets` = cash + deposit + in-transit + receivable, read from the
env's metric series at the last tick. Fines have already reduced cash or
deposit when applied, so `cum_fine` is reported for transparency and is
not subtracted again. The `score` field in `result.json` is kept as a
compatibility alias for `final_net_assets`. Additional reported fields:
`net_profit`, canonical 1–5 `shop_rating_mean`, and the same-value compatibility
alias `shop_rating_score`; `shop_rating_scale` distinguishes new `1-5` results
from historical `0-1` runs. Exact computation in
[`../eval/scoring.py`](../eval/scoring.py).

---

## 7. Files in `submission_template/`

| File | What you do |
| --- | --- |
| `my_agent.py` | **Replace this.** Stub that registers, observes, and no-ops every step. |
| `Dockerfile`  | Mostly leave alone. Customize if you need extra deps. |

If you copy the template out (`cp -r submission_template my_sub/`) you'll
typically build with `-f my_sub/Dockerfile agent/` or restructure paths to
taste — only the runtime contract in §1 matters for evaluation.

---

## 8. Versioning

The SDK is intentionally minimal. The env's HTTP shape may change between
major versions; the recommended approach is to **vendor**
`sdk/merchantbench_tool_client.py` next to your agent code and update it
alongside env upgrades.
