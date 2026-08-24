# CI TODO

Local CI plan for `MerchantBench`, adapted to the [Agent Enforcer 2](https://github.com/Artemonim/AgentEnforcer2) blueprint (three-tier: `run.ps1` thin wrapper -> `build.ps1` orchestrator -> `build.py` Python stage logic; stage statuses `ok|warn|fail|cached|skip`; hash + trust-stamp cache in `.ci_cache/`; unified `report.json`; `.enforcer/` logs).

## Stage Matrix

| Stage | Decision | Reason |
|---|---|---|
| `self-check` | `implement` | The CI layer is PowerShell-based; validate `run.ps1`, `build.ps1`, `build.py`, `pyproject.toml`, `PSScriptAnalyzerSettings.psd1` via parser diagnostics + PSScriptAnalyzer (1.24.0 installed) before project checks. Fail on any finding (AE2 recommends FAIL even on WARN for CI itself). |
| `fmt` | `implement` | `ruff format` in **fix mode** (Architect-approved auto-fix). After a successful mutating run the cache key is recomputed from the post-mutation disk state and re-stamped. |
| `lint` | `implement` | `ruff check --fix` (safe fixes only) with a focused ruleset; non-fixable findings fail the stage and are cleaned up manually once during introduction. |
| `line-limits` | `skip` | No file-size/directory policy exists for this repo; introducing global thresholds needs a separate calibration (Hermes excluded it for the same reason). TODO candidate below. |
| `compile` | `implement` | `python -m compileall -q` over all repo Python files catches syntax breakage cheaply before tests. `mypy` is **not** enabled: the codebase has no type-annotation baseline, strict mode would produce a warn-storm (see TODO). |
| `test` | `implement` | `pytest` suite (49 files, ~900 tests) is the main quality gate. Runs via `.venv` interpreter with `pythonpath` from `pyproject.toml`; parallel via `pytest-xdist` (`-n auto --maxprocesses=8 --dist=loadscope`). |
| `coverage` | `implement` | Required by AE2. `pytest-cov` collects line coverage over `env/`, `agent/`, `eval/`, `scripts/`; safe default policy without fail thresholds (see Coverage Policy). |
| `security` | `skip` | `pip-audit` needs network access to advisory feeds and there is no lock file; a blocking dependency audit would be dishonest. See TODO to enable it warn-only later. |
| `build` | `not_applicable` | No distributable artifact pipeline exists; quality is enforced at compile/test/coverage level. |
| `e2e` | `not_applicable` | End-to-end scenarios already live inside `tests/` (smoke/agent-routes suites); no separate e2e harness. |
| `launch` | `skip` | `env/run.py` is a long-running Flask server and scenarios are started individually via `POST /runs`; there is no single foreground app to launch after CI. `-SkipLaunch` is accepted for AE2 compatibility and records an explicit skip reason. |
| `archive` | `not_applicable` | CI produces no release artifacts. |
| `codebase-memory` | `implement` (warn-only, Full profile) | Refreshes the local `codebase-memory-mcp` graph so agents navigate fresh line numbers after local edits. Missing CLI or reindex errors never block CI: the index is an auxiliary navigation cache, not a source of truth. |

## Default Toolset

| Tool | Purpose | Why it is the default here |
|---|---|---|
| `ruff format` | Python code formatting (fix mode) | Single fast tool; Architect approved auto-fixing. Normalizes the whole tree once, then keeps it clean via cache. |
| `ruff check` | Python linting (fix mode, safe fixes) | Baseline quality signal (unused imports/vars, bugs) in the same tool; `# noqa: <CODE> (reason)` for suppressions. |
| `python -m compileall -q` | Syntax compilation check | Catches broken Python even before lint/test; near-zero cost. |
| `pytest` | Unit/integration test runner | Existing suite; documented entry point. |
| `pytest-xdist` | Parallel test execution (`-n auto --maxprocesses=8 --dist=loadscope`) | Suite takes ~3.5 min serially; `loadscope` keeps module-scoped fixtures (simulation runs) inside one worker. Cap of 8 keeps the dev machine responsive. |
| `pytest-cov` | Coverage collection | Mandatory AE2 coverage stage; integrates with `pytest` directly. |
| `PSScriptAnalyzer` | Self-check of the PowerShell CI layer | Validates the CI scripts themselves; integrates with the IDE via `.vscode/settings.json` / `.cursor` equivalent. |

## Profiles And Flags

- Default profile: **Full** local CI (all `implement` stages).
- `-Fast`: keeps `self-check`, `fmt`, `lint`, `compile`, `test`; skips `coverage` and `codebase-memory`.
- `-SkipLaunch`: accepted for AE2 compatibility; `launch` is always `skip` in this repo and the flag only records the explicit reason.
- `-NoCache`: ignores cache hits but rewrites trust stamps on success.
- `-ForceAll`: equivalent to `-NoCache` for the current pipeline (re-runs every cacheable stage).
- `-Clean`: removes `.ci_cache/` before the run.

## Coverage Policy

- Source of truth: `pytest-cov` JSON report + console summary.
- Start policy: safe default **without fail thresholds** (AE2 requirement when no threshold policy is chosen).
- Warning threshold: `75%` line coverage (AE2-recommended warn threshold).
- Current behavior: `coverage >= 75%` -> `ok`; `< 75%` -> `warn`; hard `fail` only if coverage cannot be collected or the test run itself fails.
- AE2 reference thresholds for a future policy: warn `75%`, fail `60%`.

## AE2 Reference Thresholds (not enforced yet)

- Coverage: warn `75%` / fail `60%` (currently warn-only at `75%`).
- Line limits (if introduced): executable-lines warn `1500` / fail `2500` per file; files-per-directory warn `20` / fail `50`.
- Self-check: AE2 recommends failing the pipeline even on WARN findings of the CI layer itself - adopted (`self-check` fails on any PSScriptAnalyzer/parser finding).

## Expected Files

- `run.ps1` - thin wrapper with help and validation of the flag set above.
- `build.ps1` - AE2-style orchestrator: stages, cache, logs, report, compact summary. Compatible with Windows PowerShell 5.1 and PowerShell 7+.
- `build.py` - Python stage runner for `fmt`, `lint`, `compile`, `test`, `coverage`; emits one JSON stage-result line on stdout for the orchestrator. (`codebase-memory` lives directly in `build.ps1`.)
- `pyproject.toml` - `[tool.pytest.ini_options]` (`testpaths`, `pythonpath`, `norecursedirs`), `[tool.ruff]`, `[tool.coverage.*]`. Not a packaging manifest.
- `requirements-dev.txt` - dev-only tools (`ruff`, `pytest-cov`, `pytest-xdist`) on top of `requirements.txt`; runtime requirement files stay untouched.
- `PSScriptAnalyzerSettings.psd1` - self-check policy for the PowerShell layer.
- `.ci_cache/` (gitignored) - `report.json`, per-stage `.sha256`/`.trusted`, `logs/`.
- `.enforcer/` (gitignored) - `Enforcer_last_check.log`, `Enforcer_stats.log`.
- `AGENTS.md` (gitignored, local) - operational contract with the final-verification command `./run.ps1 -Fast -SkipLaunch`.

## Notes

- Cache inputs use `git ls-files --cached --others --exclude-standard` (respects `.gitignore`; no submodules exist). `self-check` hashes the CI files themselves; `fmt`/`lint`/`compile` hash all tracked+untracked `*.py` + `pyproject.toml` + `build.py`. `test` and `coverage` are intentionally **not cached**: they are the honest gate and stay cheap enough with `pytest-xdist`. A global suffix (`CACHE_SCHEMA_VERSION` + interpreter/tool versions) invalidates all stamps honestly on toolchain changes.
- Mutating stages (`fmt`, `lint`) recompute their cache key from the post-mutation disk state after a successful run, otherwise the next run is a guaranteed miss.
- File bytes are hashed as-is (no CRLF normalization): the repo has no `.gitattributes`, so line endings are platform state, not content.
- `pyproject.toml` sets `pythonpath = ["env", "agent", "."]`, which fixes the documented Linux-style `PYTHONPATH=env:agent` on Windows and makes bare `pytest` collect only `tests/` (previously it also collected ~159 sandbox tests from `env/runs/*`).
- `test`/`coverage` run with a subprocess timeout. The pre-CI baseline (2026-08-24) was `895 passed / 5 failed / 2 skipped / ~202s` serial; the 5 failures (2 stale test fixtures, 2 hook-timing races, 1 Windows file-lock race in `env/web/runner.py` + `env/web/catalog_diagnostics.py`) were fixed structurally, and fixed `time.sleep` hook waits were replaced with condition polling. Current baseline: `900 passed / 0 failed / 2 skipped / ~66s` under xdist. Remaining known slowness: intentional `slow_dispatch`/`hook_blocker` simulation sleeps.
- Console output prints one result line per stage (`[OK] fmt (1.2s)`); full tool output goes to `.ci_cache/logs/<stage>.log`; `report.json` always stores the untruncated issue list plus git metadata of `HEAD` and `ci.passed`.
- TODO (follow-ups, not this iteration): enable `pip-audit` warn-only once dependency pinning exists; introduce `mypy` warn-only after a type-annotation baseline appears; consider `line-limits` after a calibration pass; consolidate duplicated test fixtures (`create_app` x11, `_tiny_scenario` x6, `_act`/`_table_records` helpers) into `tests/conftest.py` / `tests/helpers.py`; optional `pre-push` hook requiring a fresh green `report.json`; CI sounds (`assets/ci_sounds`) intentionally not added.
