# `env/` — MerchantBench simulator

This directory is a **standalone Flask app**. It is the simulation backend;
agents talk to it over HTTP. It has no dependency on `agent/` or `eval/`,
and can be carved out into its own repository without code changes.

## Run it

Requires Python 3.10+. From the repo root, create `.venv` with a supported
interpreter first.

```bash
conda create -p ./.venv python=3.11  # or: python3.11 -m venv .venv
cd env
python -m pip install -r requirements.txt
python run.py --port 5050
```

This starts the env on port 5050 and serves both the JSON HTTP API and the
dashboard at `http://127.0.0.1:5050/`.

State is written under `runs/<run_id>/` inside this directory. Each run owns
its own SQLite database at `runs/<run_id>/state.db`; per-step environment
snapshots are compact deltas, and sparse checkpoints are written as gzip files
under `env_checkpoint/`.

## Build the docker image (used by the hosted-eval harness)

From the **repo root**:

```bash
docker build -f eval/env_image/Dockerfile -t merchantbench-env:dev env/
```

Build context is `env/` so the Dockerfile only sees env-side code.

## Dashboard niceties

The `New Run` form has a `Bootstrap agent` selector:

| Option | Behavior |
| --- | --- |
| `none` | Empty shop; attach an external agent through the HTTP API. |
| `human` | Opens the browser playground for `agent_0`; it uses the same observation and `/act` protocol as external agents. |
| `rule_based` | Spawns `agent/baselines/rule_based.py`; selects either daily-report sourcing with stale-product cleanup or minimal reproducible-random sourcing without business-value filters; both modes list up to `run.rule_based_count` and delist upstream-abnormal products. |
| `react_160k_compact_30k` | Spawns the long-context ReAct baseline, compacting history to the latest 30k estimated tokens; it only prompts memory writes when memory tools are enabled. |
| `hermes` | Spawns an external Hermes adapter repo from `MERCHANTBENCH_HERMES_AGENT_ROOT` or sibling `../hermes-agent`; the adapter talks to env through the public observation + `/act` protocol. |

Subprocess baselines and the Hermes adapter launcher are **local development
conveniences**. When `env/` is deployed without the demo repo's sibling
`agent/` directory or without the external Hermes checkout, the human
playground still works, but those subprocess launchers are not available.
The Hermes launcher uses `MERCHANTBENCH_HERMES_PYTHON` first, then
`hermes-agent/.venv/bin/python` when present, then the current interpreter.
It passes `--max-hops-per-step 30`, matching the long-context ReAct baseline.
For benchmark isolation, each Hermes subprocess receives a run-local
`HERMES_HOME` at `runs/<run_id>/agent/hermes_home` and a run-local
`TERMINAL_CWD` at `runs/<run_id>/agent/hermes_workspace`. On first spawn for a
run, MerchantBench copies the external Hermes checkout's `skills/` directory into
that home; respawning the same run reuses the existing home instead of
overwriting it.

## The artifact evaluation scenario

`scenarios/default.yaml` is the artifact's single source of truth. The harness
pins `master_seed=42` unless explicitly overridden:

| Param | Value |
| --- | --- |
| Horizon | 8 760 steps (365 days × 24h, env ticks hourly) |
| `step_hours` | 1 |
| `agent.activation_period` | 12 (agent activates every 12 simulated hours) |
| `agent.max_turns_per_step` | 30 `/act` turns |
| `agent.language` | `en` by default; brief/observation text can also render `zh` |
| `master_seed` | 42 by default; `eval/run_eval.py --master-seed` overrides it for sweeps |
| Initial capital | 2 000 cash + 1 000 deposit |
| Catalog | deterministic synthetic data, 98 843 products and 36 576 suppliers |
| Virtual time | enabled from `2025-06-01` |

Run phases are explicit in `/runs/<rid>/status` and worker events:

| Phase | Meaning |
| --- | --- |
| `running` | `env.t < horizon_steps` and at least one agent is alive; demand, hooks, supplier events, orders, metrics, and snapshots all advance. |
| `draining` | The horizon is reached or all agents are dead, and active orders remain; new demand and agent hooks stop, bootstrap subprocesses are killed, long-polling agents receive HTTP 410, and existing orders continue to settlement. |
| `finished` | The horizon is reached or all agents are dead, and no active orders remain; `finished_at` is persisted and terminal dashboard/event consumers can stop polling. |

## Shop reputation methodology

The default `order_outcome_v3` policy separates two signals that buyers can
observe independently:

- **Recent service quality** is the weighted 1–5 outcome score of terminal
  orders. Evidence decays with a 180-day half-life, so current performance can
  recover or deteriorate without a single month erasing the store's history.
- **Reputation volume** is the lifetime number of rated terminal orders. It
  never decays and therefore records how established the seller is, regardless
  of whether those ratings were positive or negative.

Demand uses `quality_multiplier × reputation_multiplier`. Quality maps through
the configured star buckets. Reputation volume follows the bounded curve
`min + (max - min) × n / (n + half_saturation_orders)`, giving strong diminishing
returns. With the default values, a new seller starts at `0.80×` volume trust,
earns half of the trust gap after 20 ratings, and asymptotically approaches
`1.00×`. There are no synthetic reviews: the initial 4.0 score is only the
display fallback before real evidence exists.

`order_outcome_v2` remains supported for replaying historical scenarios. It
uses the earlier single quality multiplier and optional synthetic prior mass.

The research evaluation catalog and daily opportunity reports are not
redistributed. The artifact defaults to synthetic data so that the simulator,
agent protocol, scoring, and determinism can be inspected and tested without
external datasets. The optional `data.private_real` loader remains available
for researchers who supply their own compatible SQLite catalog.

## Runtime storage layout

`runs/<run_id>/` contains:

| Path | Meaning |
| --- | --- |
| `meta.json` | Scenario/run metadata, including `dataset_id`, `dataset_rows`, `dataset_sha256` when using `private_real` |
| `state.db` | Run-local SQLite database for orders, listings, metrics, events, and aggregate tables |
| `env_snapshot/t_NNNNN.json` | Per-step `kind: "delta"` snapshot: dirty product mutable fields, order delta, agents/cash, events, survival |
| `env_checkpoint/t_NNNNN.json.gz` | Sparse checkpoint, default every 168 steps via `run.checkpoint_interval_steps`; contains mutable product overlay only |
| `agent/by_step/t_NNNNN.json` | Agent OpenAI-message trace for steps with activity |
| `agent/memory/<agent_id>.md` | Optional run-local Markdown scratchpad used only when `read_memory_doc` / `write_memory_doc` are enabled |
| `agent/cost.json` | Token/cost aggregation, including idempotent delayed auxiliary usage such as checkpoint review |
| `agent/observation_state.json` | Per-agent last served observation step, used to keep "since last observation" windows stable across rehydrate |

Supplier runtime is event-driven. `supplier_events(run_id, due_t, product_id,
event_type, seq, payload)` stores only the next scheduled supplier events;
each step loads due rows, applies transitions, then schedules follow-up
events. Inventory replenishment is lazy via `quantity_updated_t`, so
search/detail/order/checkpoint reads effective quantity without writing 10w
natural refill updates per hour.

Catalog diagnostics in the dashboard use profit/risk-oriented views:
gross_profit365, net_profit365, order-risk scatter/outliers, selected product
drilldown, and merchant-side gross-profit ranking by product.

Manual catalog-scale benchmark checklist:

```bash
PYTHONPATH=../env:../agent ../.venv/bin/python -m pytest \
  ../tests/test_event_runtime.py -q

# Then run a 1k and 10w scenario and record:
# create_run time, one-step time, dirty product count, delta snapshot size,
# checkpoint size, runtime DB size, and full 4320-step baseline runtime.
```

## Going deeper

The code under `core/`, `tools/`, `storage/`, and `web/` is organized around
the simulator state machine, HTTP tool protocol, persistence, and dashboard
respectively. The root README provides the artifact-level reproduction path.
