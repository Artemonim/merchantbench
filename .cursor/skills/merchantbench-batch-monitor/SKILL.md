---
name: merchantbench-batch-monitor
description: >-
  Runs MerchantBench Hermes batch experiments (smoke or multi-day), monitors
  progress on a fixed interval, and reports cost/wall-time plus horizon
  projections from run_summary/batch_summaries. Use when the user asks to run
  a MerchantBench bench, batch queue, Hermes smoke, parallel seeds, monitor a
  running batch every N minutes, or extrapolate 30/90/365d cost/time.
disable-model-invocation: true
---

# MerchantBench batch + monitor

Windows / PowerShell workflow for Hermes runs against the local simulator.

## Preconditions

1. Repo root: MerchantBench checkout on the working branch (do not invent branch switches unless asked).
2. `.env` present (gitignored). Required:
   - `OPENROUTER_API_KEY` / `OPENAI_API_KEY`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
   - `MODEL_NAME` (default `deepseek/deepseek-v4-flash-0731`)
   - `MERCHANTBENCH_HERMES_AGENT_ROOT` → official adapter checkout with `merchantbench_adapter/`
   - `MERCHANTBENCH_HERMES_PYTHON` → that checkout’s `.venv/Scripts/python.exe`
   - `MERCHANTBENCH_HERMES_PROFILE_SEED=1` (optional OpenRouter profile seed)
   - `MERCHANTBENCH_NODE_HOME` → portable Node ≥22.22 if system Node is wrong
3. **Never** point `MERCHANTBENCH_HERMES_AGENT_ROOT` at personal `G:\Hermes` (that is HERMES_HOME, not the agent source tree).
4. Official adapter expected at sibling-style path, e.g. `G:\GitHubImports\hermes-agent` @ `realshop-integration`.
5. Activate MerchantBench venv: `.venv/Scripts/Activate.ps1`.

## Queues

| Goal | Queue |
|---|---|
| Hermes flash (edit `days` / CLI `--days`) | `scripts/batch_queue_hermes_flash0731.yaml` |
| Two parallel 7d replicates (seeds 42/43) | `scripts/batch_queue_hermes_flash0731_7d_x2.yaml` |

Override horizon with `--days N`. Parallelism: YAML `max_parallel` or `--max-parallel`.

## Launch pattern (one shell)

Load `.env` into the process, free port 5050, start env, then batch:

```powershell
Set-Location <MerchantBenchRoot>
.\.venv\Scripts\Activate.ps1
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k,$v = $_.Split('=',2)
  Set-Item -Path ("Env:" + $k.Trim()) -Value $v.Trim()
}
$env:PYTHONPATH = "env;agent"
if ($env:MERCHANTBENCH_NODE_HOME) {
  $env:Path = "$($env:MERCHANTBENCH_NODE_HOME);" + $env:Path
}

$existing = Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing -and $existing.OwningProcess) {
  Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 1
}

$envProc = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList @("run.py","--port","5050") `
  -WorkingDirectory (Join-Path (Get-Location) "env") `
  -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput "env_smoke_stdout.log" `
  -RedirectStandardError "env_smoke_stderr.log"

# Wait until http://127.0.0.1:5050/ responds, then:
python scripts/run_batch.py --queue scripts/<queue>.yaml --days <N> --max-parallel <K>
$batchExit = $LASTEXITCODE
Write-Output ("BATCH_EXIT=" + $batchExit)
if (-not $envProc.HasExited) { Stop-Process -Id $envProc.Id -Force }
Write-Output "ENV_STOPPED"
```

- Foreground by default. Use background **only** if the user explicitly asks.
- Capture run IDs from `dashboard: .../run_id=...` lines.

## Monitor loop

When the user asks to monitor every N minutes (e.g. 10m):

1. Follow the Cursor `loop` skill: fixed interval, unique sentinel, `notify_on_output`.
2. PowerShell shape:

```powershell
while ($true) {
  Start-Sleep -Seconds <N*60>
  Write-Output 'AGENT_LOOP_TICK_mbatch {"prompt":"<monitor prompt with run_ids and batch shell id>"}'
}
```

3. On each tick, poll:
   - `GET http://127.0.0.1:5050/runs/<rid>/status` → `state`, `phase`, `t`
   - `GET http://127.0.0.1:5050/runs/<rid>/agent/cost` → `total.usd`, `turns`, `total`, `by_step` keys
4. Brief Russian status table per tick (phase / t / usd / turns / tokens / windows).
5. Horizon for `--days D` with `step_hours=1` is `t → D*24` while **running**; higher `t` after that is **draining** (order settlement), not more agent wakeups.
6. Stop the loop when batch prints `BATCH_EXIT=` / `ENV_STOPPED`, or both runs are `finished` and `env/batch_summaries/latest.json` exists. Kill the loop process; do not leave orphan tickers.

## Phases (do not confuse)

| Phase | Meaning |
|---|---|
| `running` | `t < horizon_steps`; agent hooks active |
| `draining` | Horizon hit; agent killed; existing orders settle |
| `finished` | No active orders; summaries written |

Agent activation: every `activation_period` hours (default 12) → ~2 wakeups/sim-day.

## Artifacts to read at the end

Prefer persisted summaries over hand-recomputing:

1. **`experiments/run_history.json`** (git-friendly ledger of all finished runs) — start here for cross-run history
2. `experiments/run_history.jsonl` — append/upsert source for that ledger
3. `env/runs/<rid>/agent/run_summary.json` — full per-run snapshot (local; gitignored under `env/runs/*`)
4. `env/runs/<rid>/agent/cost.json` — `by_step` + `total`
5. `env/batch_summaries/latest.json` — multi-run aggregate + mean-rate projections (local)
6. Hermes logs (official adapter): `env/runs/<rid>/agent/hermes_home/logs/agent.log`

Rebuild ledger from local runs: `python scripts/rebuild_run_history.py`

If `run_summary.json` is missing on an old run, fall back to `cost.json` + DB metrics (`net_assets`, etc.).

## Final report template (Russian)

```markdown
### Итог batch
| Run | Seed | Status | Wall | USD | Tokens | Turns | Net assets |
|---|---|---|---|---|---|---|---|

### Прогноз (linear from measured per-sim-day rates)
| Horizon | USD | Wall |
| 30d | | |
| 90d | | |
| 365d | | |

Caveat: first-wakeup pathology, cache, and compaction make long horizons non-linear.
```

Use `projections` / `projections_from_mean_rates` from the JSON artifacts when present.

## Guardrails

- Do not commit `.env`, `env/runs/*`, or secrets.
- Do not use personal Hermes home as adapter root.
- Do not thrash `/act` after `max_turns_per_step`; pure `end_of_step` may close the hook past quota (env fix).
- Parallel Hermes jobs share OpenRouter limits — expect staggered `t` progress.
- Final user-facing reports in Russian unless asked otherwise.

## Additional Information

- Update this SKILL yourself as needed.
