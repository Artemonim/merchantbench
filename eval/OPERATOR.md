# MerchantBench hosted-eval — operator manual

You receive a submission as a Docker image (or tarball). This document
is for **you, the evaluator** — submitters read
[`agent/README.md`](../agent/README.md) instead.

## One-time setup

```bash
# 1. Install harness deps from the repo root, using the project Python 3.10+ venv
.venv/bin/python -m pip install -r eval/requirements.txt

# 2. Provide LLM credentials. These are forwarded into the agent
#    container at runtime via --env-file. Submissions never bundle keys.
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME

# 3. Build the env image (run once per env code change)
docker build -f eval/env_image/Dockerfile -t merchantbench-env:dev env/

# 4. (Optional) build the official ReAct baseline as a reference image
docker build -f agent/submission_template/Dockerfile \
             -t merchantbench-react-baseline:dev agent/
```

## Run an evaluation

```bash
.venv/bin/python -m eval.run_eval --agent-image submitter-team-x:v1 \
                                  --output result.json
```

What the harness does:

1. Creates a docker bridge network and an env container from
   `merchantbench-env:dev`, exposing port 5000 internally on a random
   host port (override with `--host-port 5050`).
2. POSTs `env/scenarios/default.yaml` (with `master_seed=42` pinned)
   to the env to create a fresh run.
3. Starts the agent container on the same network with
   `MERCHANTBENCH_BASE_URL=http://<env>:5000`, `MERCHANTBENCH_RUN_ID`,
   `MERCHANTBENCH_AGENT_ID`, `MERCHANTBENCH_AGENT_TOKEN`, and the OpenAI creds from
   `.env` injected via env vars.
4. Polls `/runs/<rid>` until status is `finished`.
5. Pulls `/runs/<rid>/agents/agent_0/sections/merchant`, computes the
   result metrics via [`scoring.py`](scoring.py), writes `result.json`.
6. Tears down everything (containers + network).

## Watching the run live

Pin a host port if you need to inspect the hosted-eval env container while
the run executes:

```bash
.venv/bin/python -m eval.run_eval --agent-image my-image:v1 \
                                  --host-port 5050 \
                                  --keep-containers \
                                  --output result.json
```

Hosted eval starts the env with token enforcement enabled. Agent containers
receive only `MERCHANTBENCH_AGENT_TOKEN`, which is limited to the agent API;
dashboard and run-control endpoints require the harness admin token. For an
unauthenticated browser dashboard, run the dev env directly with
`cd env && python run.py --port 5050`.

With `--keep-containers` the env stays up after the run finishes; clean up
manually with `docker rm -f merchantbench-env-<tag> merchantbench-agent-<tag>` and
`docker network rm merchantbench-net-<tag>`.

## Result metric

```python
score = final_net_assets
```

`final_net_assets` is the last point of the env's `net_assets` metric
series for `agent_0`. The `score` field is kept as a compatibility alias
for `final_net_assets`; there is no separate conversion from starting
capital. Additional reported fields are `net_profit`, canonical 1–5 downstream
`shop_rating_mean`, the same-value compatibility alias `shop_rating_score`, and
`shop_rating_scale` (`1-5` for current runs, `0-1` for historical runs).
See [`scoring.py`](scoring.py) to adjust.

## result.json shape

```json
{
  "agent_image": "submitter-team-x:v1",
  "env_image": "merchantbench-env:dev",
  "scenario": "default",
  "master_seed": 42,
  "run_id": "run-...",
  "terminal_status": "finished",
  "agent_id": "agent_0",
  "score": 18340.27,
  "final_net_assets": 18340.27,
  "net_profit": 13205.10,
  "shop_rating_mean": 3.91,
  "shop_rating_score": 3.91,
  "shop_rating_scale": "1-5",
  "is_alive": true,
  "died_at_t": null,
  "n_steps": 4320,
  "agent_logs_tail": ["..."]
}
```

## Multi-seed sweep

Run the same image against several seeds and average:

```bash
for seed in 1 2 3 4 5; do
  .venv/bin/python -m eval.run_eval --agent-image my-image:v1 \
      --master-seed $seed --output result-seed-$seed.json
done
```

Note: the official leaderboard uses `master_seed=42` only — sweeps are
for internal stability checks.

## What's NOT here

- No image scanning, sandboxing, or resource limits — the harness
  trusts the image. Add `--cpus`, `--memory`, network restrictions to
  `eval/run_eval.py`'s `containers.run` call when deploying for real
  external submissions.
- No leaderboard / submission UI — `result.json` is the deliverable.
- No multi-agent runs — `agent_0` is the only slot.
