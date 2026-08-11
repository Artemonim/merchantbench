"""Agent trace store — one file per step.

Layout under runs/<run_id>/agent/:
  meta.json              — list of register payloads (one per agent_id)
  by_step/t_NNNNN.json  — per-step messages (OpenAI format) + turn metadata
  cost.json             — token + USD aggregation, by_step + total
  run_summary.json       — terminal shop/cost/wall snapshot + horizon projections
  (also upserts experiments/run_history.jsonl at the repo root)
  observation_state.json — per-agent last served observation step
  idem_cache.json       — small LRU of (idempotency_key -> cached result)
  runtime_events.jsonl  — append-only runtime-health events

All writers are crash-safe via write-temp-then-rename.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------- paths ----------

def agent_dir(runs_root: str, run_id: str) -> str:
    return os.path.join(runs_root, run_id, "agent")


def _ensure(runs_root: str, run_id: str) -> str:
    p = agent_dir(runs_root, run_id)
    os.makedirs(os.path.join(p, "by_step"), exist_ok=True)
    return p


def _atomic_write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


# ---------- meta (one record per agent_id, idempotent by agent_id) ----------

_META_LOCK = threading.Lock()
_RUNTIME_EVENTS_LOCK = threading.Lock()


def write_meta(runs_root: str, run_id: str, payload: dict) -> dict:
    """Idempotently upsert by payload['agent_id']. Returns the persisted record."""
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "meta.json")
    with _META_LOCK:
        current = _read_json(path, default={"agents": []})
        if "agents" not in current:
            current = {"agents": []}
        aid = payload.get("agent_id")
        agents = current["agents"]
        replaced = False
        for i, rec in enumerate(agents):
            if rec.get("agent_id") == aid:
                agents[i] = {**rec, **payload, "registered_at_wall_ms": rec.get("registered_at_wall_ms", _now_ms())}
                replaced = True
                break
        if not replaced:
            agents.append({**payload, "registered_at_wall_ms": _now_ms()})
        _atomic_write_json(path, current)
    return payload


def read_meta(runs_root: str, run_id: str) -> dict:
    path = os.path.join(agent_dir(runs_root, run_id), "meta.json")
    return _read_json(path, default={"agents": []})


# ---------- runtime health events ----------

def init_runtime_events(
    runs_root: str,
    run_id: str,
    *,
    agent_id: str = "agent_0",
    t: int = 0,
) -> None:
    """Mark MerchantBench API runtime-health telemetry as available for a run.

    The start marker records when coverage became active so earlier simulation
    weeks are not retroactively reported as healthy zeroes.
    """
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "runtime_events.jsonl")
    with _RUNTIME_EVENTS_LOCK:
        marker_exists = False
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if (
                            isinstance(event, dict)
                            and event.get("event_type") == "runtime_telemetry_started"
                            and str(event.get("agent_id") or "") == str(agent_id)
                        ):
                            marker_exists = True
                            break
            except OSError:
                pass
        if not marker_exists:
            marker = {
                "agent_id": str(agent_id),
                "t": int(t),
                "event_type": "runtime_telemetry_started",
                "wall_ms": now_ms(),
                "payload": {"version": 2},
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(marker, ensure_ascii=False, default=str) + "\n")


def record_runtime_event(
    runs_root: str,
    run_id: str,
    *,
    agent_id: str,
    t: int,
    event_type: str,
    payload: Optional[dict] = None,
) -> None:
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "runtime_events.jsonl")
    event = {
        "agent_id": str(agent_id),
        "t": int(t),
        "event_type": str(event_type),
        "wall_ms": now_ms(),
        "payload": dict(payload or {}),
    }
    with _RUNTIME_EVENTS_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str))
            f.write("\n")


def read_runtime_events(runs_root: str, run_id: str) -> Optional[dict]:
    base = agent_dir(runs_root, run_id)
    legacy_path = os.path.join(base, "runtime_events.json")
    stream_path = os.path.join(base, "runtime_events.jsonl")
    if not os.path.exists(legacy_path) and not os.path.exists(stream_path):
        return None
    events = []
    version = 2
    if os.path.exists(legacy_path):
        legacy = _read_json(legacy_path, default=None)
        if isinstance(legacy, dict):
            try:
                version = max(version, int(legacy.get("version") or 1))
            except (TypeError, ValueError):
                pass
            events.extend(
                event for event in legacy.get("events", [])
                if isinstance(event, dict)
            )
    if os.path.exists(stream_path):
        try:
            with open(stream_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        # A process crash may leave one incomplete final line.
                        continue
                    if isinstance(event, dict):
                        events.append(event)
        except OSError:
            pass
    return {
        "version": version,
        "capabilities": {"merchantbench_api_failed_attempts": True},
        "events": events,
    }


# ---------- wall clock ----------


def now_ms() -> int:
    """Shared wall-clock helper. Use this everywhere — duplicating
    `int(time.time() * 1000)` across modules is how clock-source drift
    creeps in."""
    return int(time.time() * 1000)


_now_ms = now_ms


# ---------- by_step (one file per step, OpenAI messages format) ----------

def write_step_live(runs_root: str, run_id: str, t: int,
                    messages: list[dict], turns_meta: list[dict],
                    *, message_agents: Optional[list[Optional[str]]] = None,
                    hook_open_wall_ms: int = 0) -> str:
    """Atomic write of in-progress step data (called after each /act)."""
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "by_step", f"t_{t:05d}.json")
    payload = {
        "t": t, "n_turns": len(turns_meta),
        "hook_open_wall_ms": hook_open_wall_ms,
        "hook_close_wall_ms": 0,
        "messages": messages,
        "message_agents": message_agents or [None] * len(messages),
        "turns": turns_meta,
    }
    _atomic_write_json(path, payload)
    return path


def write_step_index(runs_root: str, run_id: str, t: int,
                     messages: list[dict], turns_meta: list[dict],
                     *, message_agents: Optional[list[Optional[str]]] = None,
                     hook_open_wall_ms: int = 0,
                     hook_close_wall_ms: int = 0) -> str:
    """Final write at phase 7 with hook_close timing."""
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "by_step", f"t_{t:05d}.json")
    payload = {
        "t": t, "n_turns": len(turns_meta),
        "hook_open_wall_ms": hook_open_wall_ms,
        "hook_close_wall_ms": hook_close_wall_ms,
        "messages": messages,
        "message_agents": message_agents or [None] * len(messages),
        "turns": turns_meta,
    }
    _atomic_write_json(path, payload)
    return path


def read_step_index(runs_root: str, run_id: str, t: int) -> Optional[dict]:
    p = os.path.join(agent_dir(runs_root, run_id), "by_step", f"t_{t:05d}.json")
    if not os.path.exists(p):
        return None
    return _read_json(p, default=None)


# ---------- cost.json (per-step + total token + USD aggregation) ----------

_COST_LOCK = threading.Lock()


def _zero_cost() -> dict:
    return {"input": 0, "output": 0, "cache_read": 0,
            "cache_write": 0, "reasoning": 0, "total": 0,
            "usd": 0.0, "turns": 0, "env_step_ms": 0,
            "unpriced_auxiliary": 0}


def normalize_token_usage(token_usage: Optional[dict]) -> dict:
    if not token_usage:
        return {}
    input_tokens = int(token_usage.get("input", 0) or 0)
    output_tokens = int(token_usage.get("output", 0) or 0)
    cache_write_tokens = int(token_usage.get("cache_write", 0) or 0)
    reasoning_tokens = int(token_usage.get("reasoning", 0) or 0)
    if "cache_read" in token_usage:
        cache_read_tokens = int(token_usage.get("cache_read", 0) or 0)
        canonical_input = input_tokens
    else:
        cache_read_tokens = int(token_usage.get("cached", 0) or 0)
        canonical_input = max(0, input_tokens - cache_read_tokens)
    total_tokens = int(
        token_usage.get(
            "total",
            canonical_input + output_tokens + cache_read_tokens + cache_write_tokens,
        )
        or 0
    )
    return {
        "input": canonical_input,
        "output": output_tokens,
        "cache_read": cache_read_tokens,
        "cache_write": cache_write_tokens,
        "reasoning": reasoning_tokens,
        "total": total_tokens,
    }


def update_cost(runs_root: str, run_id: str, t: int, turns: list[dict],
                pricing: Optional[dict] = None, *,
                env_step_ms: int = 0) -> dict:
    """Aggregate token_usage from turns into cost.json for step t and total."""
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "cost.json")
    in_per_m = float((pricing or {}).get("input_per_million", 0.0))
    out_per_m = float((pricing or {}).get("output_per_million", 0.0))
    cached_in_per_m = float((pricing or {}).get("cached_input_per_million", 0.0))
    cache_write_in_per_m = float(
        (pricing or {}).get("cache_write_input_per_million", in_per_m)
    )
    step_agg = _zero_cost()
    for turn in turns:
        tu = normalize_token_usage(turn.get("token_usage") or {})
        input_tokens = int(tu.get("input", 0) or 0)
        output_tokens = int(tu.get("output", 0) or 0)
        cache_read_tokens = int(tu.get("cache_read", 0) or 0)
        cache_write_tokens = int(tu.get("cache_write", 0) or 0)
        reasoning_tokens = int(tu.get("reasoning", 0) or 0)
        total_tokens = int(tu.get("total", 0) or 0)
        step_agg["input"] += input_tokens
        step_agg["output"] += output_tokens
        step_agg["cache_read"] += cache_read_tokens
        step_agg["cache_write"] += cache_write_tokens
        step_agg["reasoning"] += reasoning_tokens
        step_agg["total"] += total_tokens
        step_agg["turns"] += 1
        step_agg["usd"] += (
            (input_tokens / 1_000_000.0) * in_per_m
            + (cache_read_tokens / 1_000_000.0) * cached_in_per_m
            + (cache_write_tokens / 1_000_000.0) * cache_write_in_per_m
            + (output_tokens / 1_000_000.0) * out_per_m
        )
    step_agg["usd"] = round(step_agg["usd"], 6)
    step_agg["env_step_ms"] = env_step_ms
    with _COST_LOCK:
        current = _read_json(path, default={"by_step": {}, "total": _zero_cost()})
        if "by_step" not in current:
            current = {"by_step": {}, "total": _zero_cost()}
        # Auxiliary model work can finish after the hook that triggered it has
        # already closed.  Its idempotent entries live outside the mutable turn
        # buffer, so fold matching entries back into a recomputed step instead
        # of letting this replacement pass erase them.
        auxiliary_entries = (
            (current.get("auxiliary") or {}).get("entries") or {}
        )
        for entry in auxiliary_entries.values():
            entry_step = entry.get("step", -1)
            if int(-1 if entry_step is None else entry_step) != int(t):
                continue
            for key in (
                "input", "output", "cache_read", "cache_write",
                "reasoning", "total",
            ):
                step_agg[key] += int(entry.get(key, 0) or 0)
            if entry.get("usd") is None:
                step_agg["unpriced_auxiliary"] += 1
            else:
                step_agg["usd"] += float(entry.get("usd", 0.0) or 0.0)
        step_agg["usd"] = round(step_agg["usd"], 6)
        previous = current["by_step"].get(str(t)) or {}
        current["by_step"][str(t)] = step_agg
        total = current["total"]
        total.pop("cached", None)
        for k in ("input", "output", "cache_read", "cache_write", "reasoning",
                  "total", "turns", "unpriced_auxiliary"):
            total[k] = (
                int(total.get(k, 0))
                - int(previous.get(k, 0) or 0)
                + step_agg[k]
            )
        total["env_step_ms"] = (
            int(total.get("env_step_ms", 0))
            - int(previous.get("env_step_ms", 0) or 0)
            + step_agg["env_step_ms"]
        )
        total["usd"] = round(
            float(total.get("usd", 0.0))
            - float(previous.get("usd", 0.0) or 0.0)
            + step_agg["usd"],
            6,
        )
        _atomic_write_json(path, current)
    return step_agg


def record_auxiliary_usage(
    runs_root: str,
    run_id: str,
    agent_id: str,
    usage_id: str,
    token_usage: Optional[dict],
    *,
    t: int,
    source: str,
    pricing: Optional[dict] = None,
    model: str = "",
    provider: str = "",
    cost_usd: Optional[float] = None,
    cost_status: str = "",
    cost_source: str = "",
    use_scenario_pricing: bool = True,
) -> dict:
    """Idempotently add delayed auxiliary-model usage to ``cost.json``.

    Unlike an agent turn, checkpoint review may complete after ``end_of_step``
    has released the hook.  Keeping these entries in a separate durable ledger
    lets late accounting survive both request retries and a later recomputation
    of the owning step from its persisted turns.
    """
    raw_usage = token_usage or {}
    allowed_usage_keys = {
        "input", "output", "cache_read", "cached", "cache_write",
        "reasoning", "total",
    }
    unknown_keys = set(raw_usage) - allowed_usage_keys
    if unknown_keys:
        return {
            "ok": False,
            "recorded": False,
            "duplicate": False,
            "error": "invalid_token_usage",
            "field": sorted(map(str, unknown_keys))[0],
        }
    for key in allowed_usage_keys:
        if key not in raw_usage:
            continue
        value = raw_usage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return {
                "ok": False,
                "recorded": False,
                "duplicate": False,
                "error": "invalid_token_usage",
                "field": key,
            }
    if (
        "cached" in raw_usage
        and int(raw_usage.get("cached", 0) or 0)
        > int(raw_usage.get("input", 0) or 0)
    ):
        return {
            "ok": False,
            "recorded": False,
            "duplicate": False,
            "error": "invalid_token_usage",
            "field": "cached",
        }
    try:
        normalized = normalize_token_usage(raw_usage)
    except (TypeError, ValueError, OverflowError):
        return {
            "ok": False,
            "recorded": False,
            "duplicate": False,
            "error": "invalid_token_usage",
        }
    if not normalized:
        return {"ok": True, "recorded": False, "duplicate": False}
    normalized = {
        key: max(0, int(normalized.get(key, 0) or 0))
        for key in (
            "input", "output", "cache_read", "cache_write", "reasoning",
            "total",
        )
    }
    if not any(normalized.values()):
        return {"ok": True, "recorded": False, "duplicate": False}
    base_total = sum(
        normalized[key]
        for key in ("input", "output", "cache_read", "cache_write")
    )
    # Some transports expose reasoning as a subset of output; others expose it
    # as a separate billed bucket. Accept both canonical shapes, but reject a
    # caller-provided total that matches neither.
    allowed_totals = {base_total, base_total + normalized["reasoning"]}
    if normalized["total"] not in allowed_totals:
        return {
            "ok": False,
            "recorded": False,
            "duplicate": False,
            "error": "invalid_token_total",
            "expected_total": sorted(allowed_totals),
        }
    separate_reasoning_tokens = (
        normalized["reasoning"]
        if normalized["reasoning"]
        and normalized["total"] == base_total + normalized["reasoning"]
        else 0
    )

    reported_cost: Optional[float]
    if cost_usd is None:
        reported_cost = None
    else:
        try:
            reported_cost = float(cost_usd)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_cost_usd"}
        if not math.isfinite(reported_cost) or reported_cost < 0:
            return {"ok": False, "error": "invalid_cost_usd"}

    in_per_m = float((pricing or {}).get("input_per_million", 0.0))
    out_per_m = float((pricing or {}).get("output_per_million", 0.0))
    cached_in_per_m = float(
        (pricing or {}).get("cached_input_per_million", 0.0)
    )
    cache_write_in_per_m = float(
        (pricing or {}).get("cache_write_input_per_million", in_per_m)
    )
    entry = {
        "agent_id": str(agent_id),
        "source": str(source),
        "step": int(t),
        "model": str(model or ""),
        "provider": str(provider or ""),
        "cost_status": str(cost_status or ""),
        "cost_source": str(cost_source or ""),
        "reasoning_billed_separately": bool(separate_reasoning_tokens),
        **normalized,
    }
    if use_scenario_pricing:
        entry["usd"] = round(
            (entry["input"] / 1_000_000.0) * in_per_m
            + (entry["cache_read"] / 1_000_000.0) * cached_in_per_m
            + (entry["cache_write"] / 1_000_000.0) * cache_write_in_per_m
            + (
                (entry["output"] + separate_reasoning_tokens)
                / 1_000_000.0
            ) * out_per_m,
            6,
        )
        entry["pricing_mode"] = "scenario_foreground"
    elif reported_cost is not None:
        entry["usd"] = round(reported_cost, 6)
        entry["pricing_mode"] = "reported_auxiliary_model"
    else:
        entry["usd"] = None
        entry["pricing_mode"] = "unpriced_auxiliary_model"

    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "cost.json")
    with _COST_LOCK:
        current = _read_json(
            path,
            default={"by_step": {}, "total": _zero_cost()},
        )
        current.setdefault("by_step", {})
        current.setdefault("total", _zero_cost())
        entries = current.setdefault("auxiliary", {}).setdefault("entries", {})
        if usage_id in entries:
            if entries[usage_id] != entry:
                return {
                    "ok": False,
                    "recorded": False,
                    "duplicate": False,
                    "error": "usage_id_conflict",
                }
            return {"ok": True, "recorded": False, "duplicate": True}

        entries[usage_id] = entry
        step_agg = current["by_step"].setdefault(str(int(t)), _zero_cost())
        total_agg = current["total"]
        for key in (
            "input", "output", "cache_read", "cache_write", "reasoning",
            "total",
        ):
            value = int(entry.get(key, 0) or 0)
            step_agg[key] = int(step_agg.get(key, 0) or 0) + value
            total_agg[key] = int(total_agg.get(key, 0) or 0) + value
        if entry["usd"] is None:
            step_agg["unpriced_auxiliary"] = int(
                step_agg.get("unpriced_auxiliary", 0) or 0
            ) + 1
            total_agg["unpriced_auxiliary"] = int(
                total_agg.get("unpriced_auxiliary", 0) or 0
            ) + 1
        else:
            step_agg["usd"] = round(
                float(step_agg.get("usd", 0.0) or 0.0) + entry["usd"], 6
            )
            total_agg["usd"] = round(
                float(total_agg.get("usd", 0.0) or 0.0) + entry["usd"], 6
            )
        _atomic_write_json(path, current)
    return {"ok": True, "recorded": True, "duplicate": False}


def read_cost(runs_root: str, run_id: str) -> dict:
    path = os.path.join(agent_dir(runs_root, run_id), "cost.json")
    return _read_json(path, default={"by_step": {}, "total": _zero_cost()})


# ---------- run_summary.json (terminal snapshot) ----------

RUN_SUMMARY_FILENAME = "run_summary.json"
_DEFAULT_PROJECTION_HORIZONS_DAYS = (30, 90, 365)


def run_summary_path(runs_root: str, run_id: str) -> str:
    return os.path.join(agent_dir(runs_root, run_id), RUN_SUMMARY_FILENAME)


def read_run_summary(runs_root: str, run_id: str) -> dict:
    return _read_json(run_summary_path(runs_root, run_id), default={})


def build_horizon_projections(
    *,
    usd_per_sim_day: float,
    wall_ms_per_sim_day: float,
    horizons_days: tuple[int, ...] = _DEFAULT_PROJECTION_HORIZONS_DAYS,
) -> dict[str, dict[str, float]]:
    """Linear extrapolations from measured per-sim-day rates.

    These are planning estimates only: compaction, cache hit rate, and first-
    wakeup pathology make long-horizon cost/time non-linear.
    """
    out: dict[str, dict[str, float]] = {}
    for days in horizons_days:
        key = f"{int(days)}d"
        out[key] = {
            "sim_days": float(days),
            "usd": round(float(usd_per_sim_day) * days, 6),
            "wall_hours": round(
                (float(wall_ms_per_sim_day) * days) / 3_600_000.0, 4
            ),
        }
    return out


def write_run_summary(runs_root: str, run_id: str, payload: dict) -> str:
    """Persist a terminal run summary under ``agent/run_summary.json``."""
    _ensure(runs_root, run_id)
    path = run_summary_path(runs_root, run_id)
    body = dict(payload or {})
    body.setdefault("run_id", run_id)
    _atomic_write_json(path, body)
    try:
        append_run_history_from_summary(runs_root, body)
    except Exception:
        # * History is best-effort; never block the per-run summary write.
        pass
    return path


# ---------- experiments/run_history (git-friendly ledger) ----------

_HISTORY_LOCK = threading.Lock()
RUN_HISTORY_JSONL = "run_history.jsonl"
RUN_HISTORY_JSON = "run_history.json"


def repo_root_from_runs_root(runs_root: str) -> str:
    """``runs_root`` is ``<repo>/env/runs`` → return ``<repo>``."""
    return os.path.dirname(os.path.dirname(os.path.abspath(runs_root)))


def experiment_history_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "experiments")


def run_history_paths(repo_root: str) -> tuple[str, str]:
    base = experiment_history_dir(repo_root)
    return (
        os.path.join(base, RUN_HISTORY_JSONL),
        os.path.join(base, RUN_HISTORY_JSON),
    )


def compact_run_history_entry(
    summary: dict,
    *,
    model: Optional[str] = None,
    batch_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """Build a small ledger row from a full ``run_summary`` payload."""
    result = summary.get("result") if isinstance(summary.get("result"), dict) else {}
    cost = (
        summary.get("cost_total")
        if isinstance(summary.get("cost_total"), dict)
        else {}
    )
    hermes = summary.get("hermes") if isinstance(summary.get("hermes"), dict) else {}
    rates = summary.get("rates") if isinstance(summary.get("rates"), dict) else {}
    run_id = str(summary.get("run_id") or "")
    return {
        "run_id": run_id,
        "recorded_at": summary.get("written_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": summary.get("status"),
        "bootstrap_agent": summary.get("bootstrap_agent"),
        "model": model,
        "master_seed": summary.get("master_seed"),
        "sim_days": summary.get("sim_days"),
        "horizon_steps": summary.get("horizon_steps"),
        "activation_windows": summary.get("activation_windows"),
        "usd": cost.get("usd", result.get("usd")),
        "tokens": cost.get("total", result.get("tokens")),
        "turns": cost.get("turns", result.get("turns")),
        "elapsed_ms": result.get("elapsed_ms"),
        "final_net_assets": result.get("final_net_assets"),
        "cum_orders": result.get("cum_orders"),
        "cum_fine": result.get("cum_fine"),
        "shop_rating_mean": result.get("shop_rating_mean"),
        "rates": {
            "usd_per_sim_day": rates.get("usd_per_sim_day"),
            "wall_ms_per_sim_day": rates.get("wall_ms_per_sim_day"),
        },
        "projections": summary.get("projections") or {},
        "hermes_git_commit": hermes.get("hermes_git_commit"),
        "hermes_root": hermes.get("hermes_root"),
        "batch_id": batch_id,
        "notes": notes,
        "summary_relpath": (
            f"env/runs/{run_id}/agent/run_summary.json" if run_id else None
        ),
    }


def _load_history_jsonl(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        return []
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("run_id"):
                rows.append(row)
    return rows


def _rewrite_history_index(jsonl_path: str, index_path: str) -> dict:
    """Last-wins by ``run_id``, sorted by recorded_at."""
    by_id: dict[str, dict] = {}
    for row in _load_history_jsonl(jsonl_path):
        by_id[str(row["run_id"])] = row
    runs = sorted(
        by_id.values(),
        key=lambda r: str(r.get("recorded_at") or r.get("run_id") or ""),
    )
    payload = {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_count": len(runs),
        "runs": runs,
    }
    _atomic_write_json(index_path, payload)
    return payload


def append_run_history_entry(
    repo_root: str,
    entry: dict,
    *,
    replace_existing: bool = True,
) -> dict:
    """Append one compact row to ``experiments/run_history.jsonl`` and refresh JSON.

    When ``replace_existing`` is true and ``run_id`` is already present, rewrite
    the jsonl without the old row, then append the new one (upsert).
    """
    run_id = str((entry or {}).get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run history entry requires run_id")
    os.makedirs(experiment_history_dir(repo_root), exist_ok=True)
    jsonl_path, index_path = run_history_paths(repo_root)
    with _HISTORY_LOCK:
        existing = _load_history_jsonl(jsonl_path)
        if replace_existing:
            existing = [row for row in existing if str(row.get("run_id")) != run_id]
        elif any(str(row.get("run_id")) == run_id for row in existing):
            return _rewrite_history_index(jsonl_path, index_path)
        existing.append(dict(entry))
        tmp = f"{jsonl_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in existing:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, jsonl_path)
        return _rewrite_history_index(jsonl_path, index_path)


def append_run_history_from_summary(
    runs_root: str,
    summary: dict,
    *,
    model: Optional[str] = None,
    batch_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """Compact a run_summary and upsert it into the repo experiment ledger."""
    repo_root = repo_root_from_runs_root(runs_root)
    entry = compact_run_history_entry(
        summary, model=model, batch_id=batch_id, notes=notes
    )
    return append_run_history_entry(repo_root, entry)


def read_run_history(repo_root: str) -> dict:
    """Load ``experiments/run_history.json`` (rebuild from jsonl if missing)."""
    jsonl_path, index_path = run_history_paths(repo_root)
    data = _read_json(index_path, default={})
    if data:
        return data
    if os.path.exists(jsonl_path):
        return _rewrite_history_index(jsonl_path, index_path)
    return {"version": 1, "updated_at": None, "run_count": 0, "runs": []}


def rebuild_run_history_from_runs(runs_root: str) -> dict:
    """Scan ``env/runs/*/agent/run_summary.json`` and rebuild the ledger."""
    repo_root = repo_root_from_runs_root(runs_root)
    os.makedirs(experiment_history_dir(repo_root), exist_ok=True)
    jsonl_path, index_path = run_history_paths(repo_root)
    rows: list[dict] = []
    if not os.path.isdir(runs_root):
        with _HISTORY_LOCK:
            open(jsonl_path, "w", encoding="utf-8").close()
            return _rewrite_history_index(jsonl_path, index_path)
    for name in sorted(os.listdir(runs_root)):
        summary = read_run_summary(runs_root, name)
        if summary:
            rows.append(compact_run_history_entry(summary))
            continue
        cost = read_cost(runs_root, name)
        total = cost.get("total") if isinstance(cost.get("total"), dict) else {}
        if not total or not int(total.get("turns") or 0):
            continue
        rows.append({
            "run_id": name,
            "recorded_at": None,
            "status": "unknown",
            "bootstrap_agent": None,
            "model": None,
            "master_seed": None,
            "sim_days": None,
            "horizon_steps": None,
            "activation_windows": len(cost.get("by_step") or {}),
            "usd": total.get("usd"),
            "tokens": total.get("total"),
            "turns": total.get("turns"),
            "elapsed_ms": total.get("env_step_ms"),
            "final_net_assets": None,
            "cum_orders": None,
            "cum_fine": None,
            "shop_rating_mean": None,
            "rates": {},
            "projections": {},
            "hermes_git_commit": None,
            "hermes_root": None,
            "batch_id": None,
            "notes": "backfill from cost.json only (no run_summary.json)",
            "summary_relpath": None,
        })
    with _HISTORY_LOCK:
        tmp = f"{jsonl_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, jsonl_path)
        return _rewrite_history_index(jsonl_path, index_path)


# ---------- observation state ----------

_OBS_STATE_LOCK = threading.Lock()


def load_observation_state(runs_root: str, run_id: str) -> dict[str, int]:
    path = os.path.join(agent_dir(runs_root, run_id), "observation_state.json")
    data = _read_json(path, default={})
    raw_steps = {}
    if isinstance(data, dict):
        candidate = data.get("last_observation_step_by_agent")
        raw_steps = candidate if isinstance(candidate, dict) else data
    out: dict[str, int] = {}
    for agent_id, step in raw_steps.items():
        try:
            out[str(agent_id)] = int(step)
        except (TypeError, ValueError):
            continue
    return out


def load_daily_report_read_dates(runs_root: str, run_id: str) -> dict[str, str]:
    """Load the most recent successfully-read daily report date per agent."""
    path = os.path.join(agent_dir(runs_root, run_id), "observation_state.json")
    data = _read_json(path, default={})
    raw_dates = (
        data.get("daily_report_read_date_by_agent", {})
        if isinstance(data, dict)
        else {}
    )
    if not isinstance(raw_dates, dict):
        return {}
    return {
        str(agent_id): str(report_date)
        for agent_id, report_date in raw_dates.items()
        if report_date
    }


def load_observation_windows(
    runs_root: str,
    run_id: str,
) -> dict[tuple[str, int], Optional[tuple[int, int]]]:
    """Load the most recently served change window for each agent.

    A committed step can reopen the same hook after a process restart.  The
    last-observed cursor alone cannot reconstruct that hook's window because
    it has already advanced to the current step, so persist the exact window
    alongside the cursor.
    """
    path = os.path.join(agent_dir(runs_root, run_id), "observation_state.json")
    data = _read_json(path, default={})
    raw_windows = (
        data.get("change_window_by_agent", {})
        if isinstance(data, dict)
        else {}
    )
    if not isinstance(raw_windows, dict):
        return {}

    out: dict[tuple[str, int], Optional[tuple[int, int]]] = {}
    for agent_id, raw in raw_windows.items():
        if not isinstance(raw, dict):
            continue
        try:
            step = int(raw["step"])
        except (KeyError, TypeError, ValueError):
            continue
        t_from = raw.get("from")
        t_to = raw.get("to")
        if t_from is None and t_to is None:
            out[(str(agent_id), step)] = None
            continue
        try:
            parsed = (int(t_from), int(t_to))
        except (TypeError, ValueError):
            continue
        if parsed[0] <= parsed[1]:
            out[(str(agent_id), step)] = parsed
    return out


def persist_observation_state(runs_root: str, run_id: str,
                              steps_by_agent: dict[str, int], *,
                              windows_by_agent_step: Optional[
                                  dict[tuple[str, int], Optional[tuple[int, int]]]
                              ] = None,
                              daily_report_read_dates_by_agent: Optional[
                                  dict[str, str]
                              ] = None) -> None:
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "observation_state.json")
    clean = {}
    for agent_id, step in steps_by_agent.items():
        try:
            clean[str(agent_id)] = int(step)
        except (TypeError, ValueError):
            continue
    latest_windows: dict[str, dict[str, Optional[int]]] = {}
    for key, window in (windows_by_agent_step or {}).items():
        try:
            agent_id, raw_step = key
            step = int(raw_step)
        except (TypeError, ValueError):
            continue
        agent_key = str(agent_id)
        previous = latest_windows.get(agent_key)
        if previous is not None and int(previous["step"]) > step:
            continue
        if window is None:
            latest_windows[agent_key] = {
                "step": step,
                "from": None,
                "to": None,
            }
            continue
        try:
            t_from, t_to = int(window[0]), int(window[1])
        except (TypeError, ValueError, IndexError):
            continue
        if t_from <= t_to:
            latest_windows[agent_key] = {
                "step": step,
                "from": t_from,
                "to": t_to,
            }

    payload = {
        "last_observation_step_by_agent": clean,
        "change_window_by_agent": latest_windows,
        "daily_report_read_date_by_agent": {
            str(agent_id): str(report_date)
            for agent_id, report_date in (
                daily_report_read_dates_by_agent or {}
            ).items()
            if report_date
        },
        "updated_at_wall_ms": now_ms(),
    }
    with _OBS_STATE_LOCK:
        _atomic_write_json(path, payload)


# ---------- idempotency cache (per-run LRU + JSON persist) ----------

@dataclass
class IdemCache:
    cap: int = 256
    _data: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            v = self._data.get(key)
            if v is not None:
                self._data.move_to_end(key)
            return v

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.cap:
                self._data.popitem(last=False)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


_IDEM_LOCK = threading.Lock()


def persist_idem(runs_root: str, run_id: str, cache: IdemCache) -> None:
    base = _ensure(runs_root, run_id)
    path = os.path.join(base, "idem_cache.json")
    with _IDEM_LOCK:
        _atomic_write_json(path, cache.snapshot())


def load_idem(runs_root: str, run_id: str, cap: int = 256) -> IdemCache:
    path = os.path.join(agent_dir(runs_root, run_id), "idem_cache.json")
    cache = IdemCache(cap=cap)
    data = _read_json(path, default={})
    if isinstance(data, dict):
        for k, v in data.items():
            cache.put(k, v)
    return cache
