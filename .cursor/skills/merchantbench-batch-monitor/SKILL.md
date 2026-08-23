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
4. Adapter target is **mainline** `G:\GitHubImports\Hermes` @ `dev` (reconstructed + completed `merchantbench_adapter/`). Fallback: `G:\GitHubImports\hermes-agent` @ `realshop-integration` (has an uncommitted config-wiring patch — do not revert).
5. Activate MerchantBench venv: `.venv/Scripts/Activate.ps1`. If a later shell command lands in a different repo (shell cwd persists across calls), re-activate: a foreign venv lacks MerchantBench deps (`numpy` etc.).

## Queues

| Goal | Queue |
|---|---|
| Hermes flash (edit `days` / CLI `--days`) | `scripts/batch_queue_hermes_flash0731.yaml` |
| Two parallel 7d replicates (seeds 42/43) | `scripts/batch_queue_hermes_flash0731_7d_x2.yaml` |
| Two parallel 7d on v3 + CoreWeave FP8 | `scripts/batch_queue_hermes_flash0731_7d_x2_v3_coreweave.yaml` |
| Nine parallel 30d context matrix (200k/350k/1M × seeds 42–44) | `scripts/batch_queue_hermes_flash0731_30d_x9_ctx_matrix.yaml` |
| Four parallel 30d v5 model×goal (DeepSeek Flash / Gemini 3.7 Flash × default / bankrupt) | `scripts/batch_queue_hermes_v5_30d_x4_model_goal.yaml` |
| Three parallel 7d reproducing the legacy-v2 zero-prior condition | `scripts/batch_queue_hermes_flash0731_7d_x3_zero_prior.yaml` (`order_outcome_v2`, `shop_rating.prior_weight=0`) |
| 1d smoke: mainline adapter on v6, ox-alpha red-unrestricted | `scripts/batch_queue_hermes_v6_red_smoke.yaml` |
| Three parallel 7d red-team modes on v6 (unrestricted / bad merchant / bad economics), ox-alpha xhigh | `scripts/batch_queue_hermes_v6_red_7d_x3.yaml` |

### v6 economy track (Olist catalog + platform fees)

- Base overlay `env/scenarios/agents/hermes_v6.yaml` extends `../economy_v6.yaml`: Olist 1000-SKU subsample (`env/data/private_data/olist_v6.sqlite`, gitignored) + take-rate/fulfillment/refund fees. Red-mode leaf overlays set `agent.role`/`goals` per mode.
- **v6.1 guardrails** (`env/scenarios/economy_v6_1.yaml`, extends `economy_v6.yaml`): CES multiplier cap (default 6.0) + per-step violation throttle (default 5) + YAML knob for the 1000/hour demand cap. The scenario block `economy_v6_1` defaults to `enabled: false` — without the overlay the economy behaves exactly as v6.0 (bitwise). v6.0 red-team unrestricted could bankrupt the shop at t=1 via price-dump → order flood → fine farming; v6.1 closes that path.
- Product titles come from `env/data/product_titles.py` (marketplace-style, seeded) in both catalogs; typo injection knob `data.title_typo_rate` / `--typo-rate` defaults to 0. The agent sees titles in tool results (search_products, listings, orders), not in the observation text.
- `stealth/ox-alpha` notes: single `stealth` upstream on OpenRouter — scenario **must** clear the seeded `coreweave/fp8` pin via `agent.hermes.provider_routing: {}`. Free preview ($0/$0; entry in `REACT_MODEL_PRICING`). **`reasoning_effort: max` is degenerate** (reasoning-only/empty completions, zero tool calls — smoke 2026-08-23); use `xhigh`. Occasional >90s non-streaming first byte triggers a stale-kill; the retry policy recovers.
- Run-local `config.yaml` gets `auxiliary.free_only: true` (`HERMES_AUXILIARY_FREE_ONLY` in `env/web/runner.py`) so Hermes auxiliary fallbacks cannot hit paid SKUs mid-benchmark.
- Offline catalog diagnostics CLI: `python -m data.economy_diagnostics --source synthetic|private_real --scenario <path>` (curve distribution, demand/gross at ref/2×/10×, v6 fee contribution).

### Synthetic catalog (v5)

Read `experiments/research_journal.rus.md` first for which catalog a historical batch used.

- Default synthetic catalog is **v5** (`margin_consistent_v1` + calibrated `base_demand [0.02, 1.02]`). Hermes agent overlays inherit this from `default.yaml`.
- Policy ablations (no LLM): `env/scenarios/ablations/*.yaml`. Queue `scripts/batch_queue_rule_ablations.yaml` is rule_based 7d — **do not launch unless the user asks**.
- Offline metrics: `env/data/economy_diagnostics.py` / `tests/test_synth_v5_ablations.py`.
- To reproduce the v4 ctx-matrix economy: `env/scenarios/ablations/legacy_v4.yaml`.

The default scenario uses `order_outcome_v4`: deterministic self-selected
public rating/count are visible to Hermes and drive buyer demand through
confidence shrinkage plus review-volume trust. Internal recent service quality
remains visible as a separate operational KPI. Compare final
`public_review_rating`, `public_review_count`, `public_review_confidence`,
`public_review_demand_multiplier`, `public_review_selection_gap`, and
`service_quality_score` when evaluating new runs. OpenRouter routing for Hermes is pinned in
`scripts/hermes_openrouter_profile.snippet.yaml` (currently `coreweave/fp8` for
DeepSeek). Gemini overlays replace that pin with `google-vertex/global` via
`agent.hermes.provider_routing`. Bankruptcy overlays (`hermes_bankrupt.yaml`,
`hermes_gemini_bankrupt.yaml`) change `agent.role` / `agent.goals` only; the
env already closes the shop at `deposit_pool == 0` and moves the run to
`draining`, so agents are not told to emit a final step after bankruptcy.
Use the zero-prior queue only to reproduce the earlier v2 experiment, not as
the current cold-start baseline.

Hermes context is configurable via scenario `agent.hermes`:

```yaml
agent:
  hermes:
    context_length: 262144   # written to model.context_length
    compression_threshold: 0.85
    provider_routing:        # optional; replaces the profile-seed pin
      only: [google-vertex/global]
    reasoning_effort: high   # optional; Gemini-safe (DeepSeek seed stays xhigh)
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

Session context first: **`experiments/research_journal.rus.md`** — running research
journal (tracked in git). Read it at session start for accumulated findings,
decisions, and methodological lessons; append a new dated section on top after
any significant experiment or analysis. Mark superseded info, do not delete.

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
- Running pytest from a shell that loaded `.env` leaks `MERCHANTBENCH_HERMES_PYTHON` / `MERCHANTBENCH_HERMES_PROFILE_SEED` into spawn tests — the affected tests delenv these themselves; if new spawn tests are added, keep them hermetic the same way.
