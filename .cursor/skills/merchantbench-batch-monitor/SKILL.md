---
name: merchantbench-batch-monitor
description: >-
  Runs MerchantBench Hermes batch experiments (smoke or multi-day), monitors
  progress via AwaitShell on the batch terminal (not a loop ticker), and
  reports cost/wall-time plus horizon projections from run_summary /
  batch_summaries / experiments/run_history. Use when the user asks to run a
  MerchantBench bench, batch queue, Hermes smoke, parallel seeds, monitor a
  running batch, or extrapolate 30/90/365d cost/time.
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
| Two parallel 7d on v3 + CoreWeave FP8 | `scripts/batch_queue_hermes_flash0731_7d_x2_v3_coreweave.yaml` |
| Nine parallel 30d context matrix (200k/350k/1M × seeds 42–44) | `scripts/batch_queue_hermes_flash0731_30d_x9_ctx_matrix.yaml` |
| Three parallel 7d reproducing the legacy-v2 zero-prior condition | `scripts/batch_queue_hermes_flash0731_7d_x3_zero_prior.yaml` (`order_outcome_v2`, `shop_rating.prior_weight=0`) |

The default scenario uses `order_outcome_v4`: deterministic self-selected
public rating/count are visible to Hermes and drive buyer demand through
confidence shrinkage plus review-volume trust. Internal recent service quality
remains visible as a separate operational KPI. Compare final
`public_review_rating`, `public_review_count`, `public_review_confidence`,
`public_review_demand_multiplier`, `public_review_selection_gap`, and
`service_quality_score` when evaluating new runs. OpenRouter routing for Hermes is pinned in
`scripts/hermes_openrouter_profile.snippet.yaml` (currently `coreweave/fp8`).
Use the zero-prior queue only to reproduce the earlier v2 experiment, not as
the current cold-start baseline.

Hermes context is configurable via scenario `agent.hermes`:

```yaml
agent:
  hermes:
    context_length: 262144   # written to model.context_length
    compression_threshold: 0.85
```

Variants: `hermes_ctx_200k.yaml`, `hermes_ctx_350k.yaml`, `hermes_ctx_1m.yaml`.
For a paired pre-review baseline, use `env/scenarios/agents/hermes_v3.yaml`.
Default Hermes and the evaluator share the same v4 public-reputation state.

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

- For long multi-day batches: start this shell with `block_until_ms: 0` (background).
- Capture the batch **shell_id** and run IDs from `dashboard: .../run_id=...`.
- Foreground (`block_until_ms` covering full runtime) is fine for short smokes.

## Monitoring — AwaitShell only (no loop)

**Do not** use the Cursor `loop` skill. **Do not** start a second `while { Start-Sleep ... }` ticker shell.

Use **AwaitShell** on the batch shell:

1. After background launch, note `shell_id`.
2. Progress cadence (default 10 minutes unless user overrides):

   ```
   AwaitShell(
     shell_id=<batch_shell_id>,
     pattern="BATCH_EXIT=|ENV_STOPPED",
     block_until_ms=<N_minutes * 60_000>
   )
   ```

3. When AwaitShell returns:
   - If output matched `BATCH_EXIT=` / `ENV_STOPPED` or the shell exited → go to **Final report**.
   - Else (slice timeout): poll status/cost (below), brief Russian progress table, then **AwaitShell again** with the same pattern/shell_id.
4. Optional early wake: also match useful batch lines, e.g. `phase=finished` / `phase=draining`, but keep the completion pattern as `BATCH_EXIT=|ENV_STOPPED`.
5. Size slices to expected runtime (7d×2–3 parallel often ~40–90+ min wall). Prefer 10–15 min slices for progress updates; avoid vacuous 1h+ sleeps when the user asked for periodic updates.

### Progress poll (each AwaitShell return while still running)

- `GET http://127.0.0.1:5050/runs/<rid>/status` → `state`, `phase`, `t`
- `GET http://127.0.0.1:5050/runs/<rid>/agent/cost` → `total.usd`, `turns`, `total`, `by_step` keys
- Brief Russian table: phase / t / usd / turns / tokens / windows
- Horizon for `--days D` with `step_hours=1` is `t → D*24` while **running**; higher `t` is **draining** (settlement), not more agent wakeups

## Phases (do not confuse)

| Phase | Meaning |
|---|---|
| `running` | `t < horizon_steps`; agent hooks active |
| `draining` | Horizon hit; agent killed; existing orders settle |
| `finished` | No active orders; summaries written |

Agent activation: every `activation_period` hours (default 12) → ~2 wakeups/sim-day.

## Artifacts to read at the end

Prefer persisted summaries over hand-recomputing:

1. **`experiments/run_history.json`** — git-friendly ledger of finished runs (start here for history)
2. `experiments/run_history.jsonl` — upsert source for that ledger
3. `env/runs/<rid>/agent/run_summary.json` — full per-run snapshot (local; under gitignored `env/runs/*`)
4. `env/runs/<rid>/agent/cost.json` — `by_step` + `total`
5. `env/batch_summaries/latest.json` — multi-run aggregate + mean-rate projections (local)
6. Hermes logs: `env/runs/<rid>/agent/hermes_home/logs/agent.log`

Rebuild ledger: `python scripts/rebuild_run_history.py`

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

Use `projections` / `projections_from_mean_rates` from JSON artifacts when present.

## Guardrails

- Do not commit `.env`, `env/runs/*`, or secrets.
- Do not use personal Hermes home as adapter root.
- Do not thrash `/act` after `max_turns_per_step`; pure `end_of_step` may close the hook past quota.
- Parallel Hermes jobs share OpenRouter limits — expect staggered `t` progress.
- Final user-facing reports in Russian unless asked otherwise.
- **No loop skill / no sleep-ticker shell** for monitoring — only AwaitShell on the batch terminal.
