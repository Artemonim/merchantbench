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

## Synthetic catalog economics (v5)

The public artifact default is a **synthetic 1000-SKU** catalog
(`data.num_products: 1000` in `scenarios/default.yaml`). The 98 843 products /
36 576 suppliers row in the table above describes the private research catalog;
it is not redistributed.

v5 generation is margin-consistent: supplier `cost` is strictly below consumer
`ref_price`, CES elasticity is `ε = ref / (ref − cost)`, and `ref_price` is the
theoretically optimal sale price. Per-SKU `base_demand` is sampled from
`[0.02, 1.02]` so expected listing-day demand at `sale = ref` (lifecycle=1,
rating=1) is about 0.52 — roughly 26 shop-day orders with 50 active listings.

Knobs live under `generation_params`:

- `pricing_model`: `margin_consistent_v1` (default) or `legacy_anchor_at_cost`
- `base_demand`: calibrated `[0.02, 1.02]` vs legacy `[1.0, 50.0]`
- per-category `retail_margin` (used only by `margin_consistent_v1`)

One-axis overlays are in `scenarios/ablations/` (`pricing_only`, `demand_only`,
`both`, `legacy_v4`). Platform fines remain **fixed RMB amounts**, not
percentages of ticket size.

## Shop reputation methodology

The default `order_outcome_v4` policy separates operational truth from the
buyer-visible reputation that drives traffic:

- **Internal service quality** is the weighted 1–5 outcome score of terminal
  orders. Evidence decays with a 180-day half-life, so current performance can
  recover or deteriorate without a single month erasing the store's history.
  Hermes and the evaluator both see it as an operational KPI, but it does not
  directly enter v4 demand.
- **Public review rating and count** are the lifetime buyer-visible reputation.
  They are sampled from qualified terminal experiences and are visible to
  Hermes, the buyer-demand model, the human playground, and the evaluator.

V4 demand uses one shared public-reputation formula. For `n` public reviews and
`h=20`, confidence is `c = n / (n + h)`. The raw star-bucket effect `M` is
shrunk toward neutral as `Q = 1 + c × (M - 1)`, so one extreme review cannot
carry the weight of an established history. Review-volume trust is
`T = min + (max - min) × c`, and final traffic is `Q × T`. With the defaults,
no reviews produce neutral rating quality × `0.80` cold-start trust; 20 reviews
provide 50% confidence; trust asymptotically approaches `1.00`. Until the first
public review, buyer-visible `score` and `stars` are absent rather than copied
from the internal service-quality KPI.

The default `self_selection_v1` policy maps each qualified terminal experience
to the nearest 1–5 star value and samples whether it becomes public. The
configured response probabilities by star are `30%, 18%, 8%, 6%, 12%`, giving
an explicit U-shaped extremity bias with stronger negative selection. An
existing `settled_bad_review` outcome is public by definition. These
probabilities are transparent synthetic stress parameters, not claimed
universal marketplace estimates; scenario ablations should vary them explicitly.

Sampling uses an independent deterministic RNG stream derived from the master
seed, merchant, and order, so it does not advance shared economic RNG state.
The prompt, observation, dashboard, replay, and run summary expose the same
public rating/count/confidence/demand state. The all-response counterfactual and
selection gap remain comparison diagnostics exposed alongside that shared
state, but they do not affect demand. The public state itself is a first-class
v4 economic input rather than side telemetry.

`order_outcome_v3` remains available through
`env/scenarios/agents/hermes_v3.yaml` for paired compatibility experiments; it
uses internal recent quality × lifetime qualified-transaction volume. V2 also
remains supported for historical replay with its earlier single quality
multiplier and optional synthetic prior mass.

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
