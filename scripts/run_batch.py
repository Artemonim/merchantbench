#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "env"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from web.runner import load_default_scenario, load_scenario  # noqa: E402
from web.routes_dashboard import REACT_MODEL_PRICING_BY_MODEL  # noqa: E402


DEFAULT_QUEUE_PATH = ROOT / "scripts" / "batch_queue.yaml"
DEFAULT_SCENARIO_PATH = "env/scenarios/default.yaml"
HERMES_SCENARIO_PATH = "env/scenarios/agents/hermes.yaml"
DEFAULT_BOOTSTRAP_AGENT = "react_160k_compact_30k"
SUPPORTED_BOOTSTRAP_AGENTS = {
    "react_160k_compact_30k",
    "hermes",
    "rule_based",
}
RULE_BASED_SELECTION_MODES = {"daily_report", "random"}
DEFAULT_RUN_DAYS = 90
DEFAULT_MASTER_SEED = 42
DEFAULT_BASE_URL = "http://127.0.0.1:5050"
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))


class BatchRunError(RuntimeError):
    def __init__(self, message: str, *, failures: list[dict], results: list):
        super().__init__(message)
        self.failures = failures
        self.results = results


def request_json(method: str, base_url: str, path: str,
                 body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(f"{base_url.rstrip('/')}{path}",
                  data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
    except HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {e.code} {detail}") from e
    except TimeoutError as e:
        raise RuntimeError(
            f"{method} {path} timed out after {REQUEST_TIMEOUT_SECONDS:g}s; "
            "check /runs before retrying because the server may still finish it."
        ) from e
    except URLError as e:
        raise RuntimeError(f"cannot reach {base_url}: {e}") from e
    return json.loads(raw.decode("utf-8")) if raw else {}


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in (ROOT, ENV_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return ROOT / candidate


def load_queue_config(path: str | Path = DEFAULT_QUEUE_PATH) -> dict[str, Any]:
    queue_path = _resolve_repo_path(path)
    with open(queue_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"queue config must be a YAML mapping: {queue_path}")
    config.setdefault("scenario_path", DEFAULT_SCENARIO_PATH)
    config.setdefault("bootstrap_agent", config.get("framework", DEFAULT_BOOTSTRAP_AGENT))
    config.setdefault("days", DEFAULT_RUN_DAYS)
    config.setdefault("seed", DEFAULT_MASTER_SEED)
    config.setdefault("poll_seconds", 30)
    config.setdefault("base_url", DEFAULT_BASE_URL)
    config.setdefault("queue", [])
    return config


def _coerce_run_days(raw: Any) -> int:
    days = int(raw)
    if days < 1:
        raise ValueError("days must be at least 1")
    return days


def _coerce_seed(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer: {raw!r}") from exc


def _horizon_steps_for_days(days: int, step_hours: int | float) -> int:
    step_hours_float = float(step_hours)
    if step_hours_float <= 0:
        raise ValueError("scenario run.step_hours must be positive")
    return int(math.ceil((days * 24) / step_hours_float))


def _set_scenario_run_days(scenario: dict[str, Any], days: int) -> None:
    run_cfg = scenario.setdefault("run", {})
    step_hours = run_cfg.get("step_hours", 1)
    run_cfg["horizon_steps"] = _horizon_steps_for_days(days, step_hours)


def jobs_from_config(
    config: dict[str, Any],
    *,
    days_override: int | None = None,
    seed_override: int | None = None,
) -> list[dict[str, Any]]:
    raw_queue = config.get("queue")
    if not isinstance(raw_queue, list):
        raise ValueError("queue config must define `queue:` as a list")

    jobs = []
    for item in raw_queue:
        if isinstance(item, str):
            job = {"model": item}
        elif isinstance(item, dict):
            job = dict(item)
        else:
            raise ValueError(f"queue item must be a model string or mapping: {item!r}")

        job.setdefault("scenario_path", config["scenario_path"])
        if "detailed" not in job and "detailed" in config:
            job["detailed"] = config["detailed"]
        if "detailed" in job and not isinstance(job["detailed"], bool):
            raise ValueError(
                f"queue item `detailed` must be true or false: {job['detailed']!r}"
            )
        bootstrap_agent = str(
            job.get("bootstrap_agent")
            or job.get("framework")
            or config.get("bootstrap_agent")
            or config.get("framework")
            or DEFAULT_BOOTSTRAP_AGENT
        ).strip()
        if bootstrap_agent not in SUPPORTED_BOOTSTRAP_AGENTS:
            raise ValueError(
                f"unsupported bootstrap_agent for this runner: {bootstrap_agent!r}"
            )
        job["bootstrap_agent"] = bootstrap_agent

        if bootstrap_agent == "rule_based":
            selection_mode = str(
                job.get("selection_mode")
                or config.get("selection_mode")
                or "random"
            ).strip()
            if selection_mode not in RULE_BASED_SELECTION_MODES:
                raise ValueError(
                    "rule_based selection_mode must be one of "
                    f"{sorted(RULE_BASED_SELECTION_MODES)}, got {selection_mode!r}"
                )
            job["selection_mode"] = selection_mode
            model = str(job.get("model") or selection_mode).strip()
        else:
            model = str(job.get("model", "")).strip()
            if not model:
                raise ValueError(f"queue item is missing `model`: {item!r}")
        job["model"] = model

        raw_days = (
            days_override
            if days_override is not None
            else job.get("days", config.get("days", DEFAULT_RUN_DAYS))
        )
        job["days"] = _coerce_run_days(raw_days)
        raw_seed = (
            seed_override
            if seed_override is not None
            else job.get("seed", config.get("seed", DEFAULT_MASTER_SEED))
        )
        job["seed"] = _coerce_seed(raw_seed)
        if bootstrap_agent == "rule_based":
            raw_selection_seed = job.get(
                "selection_seed",
                config.get("selection_seed", job["seed"]),
            )
            job["selection_seed"] = _coerce_seed(raw_selection_seed)
        jobs.append(job)
    return jobs


def _load_scenario_for_path(scenario_path: str | None) -> dict:
    if not scenario_path:
        return load_default_scenario()
    return load_scenario(str(_resolve_repo_path(scenario_path)))


def _scenario_path_for_bootstrap(
    bootstrap_agent: str,
    scenario_path: str | None,
) -> str | None:
    if bootstrap_agent == "hermes" and scenario_path in (None, DEFAULT_SCENARIO_PATH):
        return HERMES_SCENARIO_PATH
    return scenario_path


def scenario_for(
    model: str,
    scenario_path: str | None = DEFAULT_SCENARIO_PATH,
    days: int | None = DEFAULT_RUN_DAYS,
    detailed: bool | None = None,
) -> dict:
    scenario = _load_scenario_for_path(scenario_path)
    if days is not None:
        _set_scenario_run_days(scenario, _coerce_run_days(days))
    if detailed is not None:
        if not isinstance(detailed, bool):
            raise ValueError("detailed must be true or false")
        scenario.setdefault("agent", {})["detailed"] = detailed
    pricing = REACT_MODEL_PRICING_BY_MODEL.get(model)
    if pricing is None:
        raise ValueError(f"model is not in REACT_MODEL_PRICING: {model}")
    scenario.setdefault("agent", {})["cost_pricing"] = {
        "input_per_million": float(pricing["input"]),
        "output_per_million": float(pricing["output"]),
        "cached_input_per_million": float(pricing["cached_input"]),
    }
    return scenario


def _coerce_job(job: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(job, str):
        return {
            "model": job,
            "scenario_path": DEFAULT_SCENARIO_PATH,
            "bootstrap_agent": DEFAULT_BOOTSTRAP_AGENT,
            "days": DEFAULT_RUN_DAYS,
            "seed": DEFAULT_MASTER_SEED,
        }
    normalized = dict(job)
    bootstrap_agent = str(
        normalized.get("bootstrap_agent") or DEFAULT_BOOTSTRAP_AGENT
    )
    if bootstrap_agent == "rule_based":
        selection_mode = str(normalized.get("selection_mode") or "random")
        normalized.setdefault("model", selection_mode)
    return normalized


def create_run(base_url: str, job: str | dict[str, Any]) -> str:
    job = _coerce_job(job)
    bootstrap_agent = str(job.get("bootstrap_agent") or DEFAULT_BOOTSTRAP_AGENT)
    if bootstrap_agent not in SUPPORTED_BOOTSTRAP_AGENTS:
        raise ValueError(
            f"unsupported bootstrap_agent for this runner: {bootstrap_agent!r}"
        )
    seed = _coerce_seed(job.get("seed", DEFAULT_MASTER_SEED))
    model = str(job.get("model") or "").strip()
    if bootstrap_agent == "rule_based":
        selection_mode = str(job.get("selection_mode") or "random").strip()
        if selection_mode not in RULE_BASED_SELECTION_MODES:
            raise ValueError(
                "rule_based selection_mode must be one of "
                f"{sorted(RULE_BASED_SELECTION_MODES)}, got {selection_mode!r}"
            )
        if not model:
            model = selection_mode
        selection_seed = _coerce_seed(job.get("selection_seed", seed))
        scenario = _load_scenario_for_path(
            _scenario_path_for_bootstrap(
                bootstrap_agent,
                job.get("scenario_path"),
            )
        )
        days = job.get("days", DEFAULT_RUN_DAYS)
        if days is not None:
            _set_scenario_run_days(scenario, _coerce_run_days(days))
        bootstrap_config = {
            "selection_mode": selection_mode,
            "selection_seed": selection_seed,
        }
    else:
        if not model:
            raise ValueError("queue item is missing `model`")
        scenario = scenario_for(
            model,
            _scenario_path_for_bootstrap(
                bootstrap_agent,
                job.get("scenario_path"),
            ),
            days=job.get("days", DEFAULT_RUN_DAYS),
            detailed=job.get("detailed"),
        )
        bootstrap_config = {"react_model": model}

    safe_model = model.replace("/", "-")
    scenario.setdefault("run", {})["master_seed"] = seed
    body = {
        "name": job.get("name") or (
            f"{bootstrap_agent}-{safe_model}-seed-{seed}"
            if bootstrap_agent == "rule_based"
            else f"{bootstrap_agent}-{safe_model}"
        ),
        "scenario": scenario,
        "master_seed": seed,
        "bootstrap_agent": bootstrap_agent,
        "bootstrap_config": bootstrap_config,
        "auto_start": True,
    }
    return request_json("POST", base_url, "/runs", body=body)["run_id"]


def wait_finished(base_url: str, run_id: str, poll_seconds: float) -> dict:
    last_line = None
    while True:
        status = request_json("GET", base_url, f"/runs/{run_id}/status")
        line = (
            f"{run_id}: state={status.get('state')} "
            f"phase={status.get('phase')} t={status.get('t')}"
        )
        if line != last_line:
            print(line, flush=True)
            last_line = line
        if status.get("phase") == "finished":
            return status
        if status.get("state") in {"stopped", "error"}:
            raise RuntimeError(f"{run_id} ended early: {status}")
        time.sleep(poll_seconds)


def get_run_status(base_url: str, run_id: str) -> dict:
    return request_json("GET", base_url, f"/runs/{run_id}/status")


def resolve_max_parallel(cli_value: int | None, config: dict[str, Any]) -> int:
    raw = cli_value
    if raw is None:
        raw = os.environ.get("MAX_PARALLEL")
    if raw is None:
        raw = config.get("max_parallel", 1)
    value = int(raw)
    if value < 1:
        raise ValueError("max_parallel must be at least 1")
    return value


def _collect_orphaned_runs(
    active: dict[int, tuple[dict[str, Any], str]],
    failures: list[dict[str, Any]],
    ordered_results: list,
    *,
    reason: str,
    exclude_index: int | None = None,
) -> None:
    """Add still-active runs to failures so callers can track or stop them."""
    for index, (job, run_id) in active.items():
        if index == exclude_index:
            continue
        if ordered_results[index] is not None:
            continue
        if any(f.get("index") == index for f in failures):
            continue
        failures.append({
            "index": index,
            "model": job["model"],
            "run_id": run_id,
            "error": reason,
        })


def run_models(jobs: list[dict[str, Any]] | tuple[str, ...],
               base_url: str = DEFAULT_BASE_URL,
               poll_seconds: float = 30,
               max_parallel: int = 1) -> list[tuple[str, str, dict]]:
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    normalized_jobs = [_coerce_job(job) for job in jobs]
    ordered_results: list[tuple[str, str, dict] | None] = [None] * len(normalized_jobs)
    active: dict[int, tuple[dict[str, Any], str]] = {}
    failures: list[dict[str, Any]] = []
    next_index = 0
    last_lines: dict[str, str] = {}

    while next_index < len(normalized_jobs) or active:
        while next_index < len(normalized_jobs) and len(active) < max_parallel:
            job = normalized_jobs[next_index]
            model = str(job["model"])
            print(f"starting {model} via {job.get('bootstrap_agent')}", flush=True)
            try:
                run_id = create_run(base_url, job)
            except Exception as exc:
                failures.append({"index": next_index, "model": model, "error": str(exc)})
                _collect_orphaned_runs(active, failures, ordered_results,
                                       reason="abandoned by batch error")
                raise BatchRunError(
                    f"creation failed for {model}; stopped filling the queue",
                    failures=failures,
                    results=[result for result in ordered_results if result is not None],
                ) from exc
            active[next_index] = (job, run_id)
            print(
                f"dashboard: {base_url.rstrip('/')}/dashboard?run_id={run_id}",
                flush=True,
            )
            next_index += 1

        completed: list[int] = []
        for index, (job, run_id) in list(active.items()):
            try:
                status = get_run_status(base_url, run_id)
            except Exception as exc:
                failures.append({
                    "index": index,
                    "model": job["model"],
                    "run_id": run_id,
                    "error": str(exc),
                })
                _collect_orphaned_runs(active, failures, ordered_results,
                                       reason="abandoned by batch error",
                                       exclude_index=index)
                raise BatchRunError(
                    f"status request failed for {run_id}; stopped filling the queue",
                    failures=failures,
                    results=[result for result in ordered_results if result is not None],
                ) from exc
            line = (
                f"{run_id}: state={status.get('state')} "
                f"phase={status.get('phase')} t={status.get('t')}"
            )
            if last_lines.get(run_id) != line:
                print(line, flush=True)
                last_lines[run_id] = line
            if status.get("state") in {"stopped", "error"}:
                failures.append({
                    "index": index,
                    "model": job["model"],
                    "run_id": run_id,
                    "status": status,
                })
                completed.append(index)
            elif status.get("phase") == "finished":
                ordered_results[index] = (str(job["model"]), run_id, status)
                completed.append(index)
        for index in completed:
            active.pop(index, None)

        if active and not completed:
            time.sleep(poll_seconds)

    results = [result for result in ordered_results if result is not None]
    if failures:
        raise BatchRunError(
            f"{len(failures)} run(s) failed", failures=failures, results=results
        )
    return results


def _read_run_summary_file(run_id: str) -> dict[str, Any]:
    path = ENV_ROOT / "runs" / run_id / "agent" / "run_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_batch_summary(
    results: list[tuple[str, str, dict]],
    *,
    queue_path: str | Path,
    days: int | None,
    max_parallel: int,
) -> Path:
    """Persist a batch-level summary under env/batch_summaries/."""
    from datetime import datetime, timezone

    from storage import agent_log

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ENV_ROOT / "batch_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    usd_rates: list[float] = []
    wall_rates: list[float] = []
    for model, run_id, status in results:
        summary = _read_run_summary_file(run_id)
        rates = summary.get("rates") or {}
        if rates.get("usd_per_sim_day") is not None:
            usd_rates.append(float(rates["usd_per_sim_day"]))
        if rates.get("wall_ms_per_sim_day") is not None:
            wall_rates.append(float(rates["wall_ms_per_sim_day"]))
        runs.append({
            "model": model,
            "run_id": run_id,
            "status": status,
            "summary": summary,
        })
    avg_usd = sum(usd_rates) / len(usd_rates) if usd_rates else 0.0
    avg_wall = sum(wall_rates) / len(wall_rates) if wall_rates else 0.0
    payload = {
        "written_at": stamp,
        "queue_path": str(queue_path),
        "days": days,
        "max_parallel": max_parallel,
        "run_count": len(runs),
        "runs": runs,
        "aggregate_rates": {
            "usd_per_sim_day_mean": round(avg_usd, 6),
            "wall_ms_per_sim_day_mean": round(avg_wall, 3),
            "n_rate_samples": len(usd_rates),
        },
        "projections_from_mean_rates": agent_log.build_horizon_projections(
            usd_per_sim_day=avg_usd,
            wall_ms_per_sim_day=avg_wall,
        ),
        "caveat": (
            "Linear projections from mean measured per-sim-day rates across "
            "finished runs in this batch."
        ),
    }
    out_path = out_dir / f"batch-{stamp}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    latest = out_dir / "latest.json"
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run queued MerchantBench bootstrap agents."
    )
    parser.add_argument(
        "--queue",
        default=os.environ.get("MERCHANTBENCH_QUEUE", str(DEFAULT_QUEUE_PATH)),
        help="YAML queue file. Defaults to scripts/batch_queue.yaml.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Run horizon in simulated days. Defaults to YAML `days`, "
            f"or {DEFAULT_RUN_DAYS} when omitted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Environment master seed. Precedence: CLI > per-job YAML `seed` "
            f"> top-level YAML `seed` > {DEFAULT_MASTER_SEED}."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum active runs (CLI > MAX_PARALLEL > YAML > 1).",
    )
    args = parser.parse_args()

    config = load_queue_config(args.queue)
    days_override = _coerce_run_days(args.days) if args.days is not None else None
    seed_override = _coerce_seed(args.seed) if args.seed is not None else None
    jobs = jobs_from_config(
        config,
        days_override=days_override,
        seed_override=seed_override,
    )
    if not jobs:
        raise ValueError(f"queue has no enabled jobs: {args.queue}")

    base_url = os.environ.get("MERCHANTBENCH_BASE_URL", config["base_url"]).rstrip("/")
    poll_seconds = float(os.environ.get("POLL_SECONDS", config["poll_seconds"]))
    max_parallel = resolve_max_parallel(args.max_parallel, config)
    try:
        results = run_models(
            jobs,
            base_url=base_url,
            poll_seconds=poll_seconds,
            max_parallel=max_parallel,
        )
    except BatchRunError as exc:
        print(str(exc), file=sys.stderr)
        for failure in exc.failures:
            print(json.dumps(failure, ensure_ascii=False, default=str), file=sys.stderr)
        if exc.results:
            try:
                path = write_batch_summary(
                    list(exc.results),
                    queue_path=args.queue,
                    days=days_override if days_override is not None else config.get("days"),
                    max_parallel=max_parallel,
                )
                print(f"BATCH_SUMMARY={path}", flush=True)
            except Exception as summary_exc:  # noqa: BLE001
                print(f"batch summary failed: {summary_exc}", file=sys.stderr)
        return 1
    try:
        path = write_batch_summary(
            results,
            queue_path=args.queue,
            days=days_override if days_override is not None else config.get("days"),
            max_parallel=max_parallel,
        )
        print(f"BATCH_SUMMARY={path}", flush=True)
    except Exception as summary_exc:  # noqa: BLE001
        print(f"batch summary failed: {summary_exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
