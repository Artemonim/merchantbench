"""Run management + dashboard routes + per-section data endpoints + SSE."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import yaml
from flask import (Blueprint, Response, abort, g, jsonify, redirect, render_template,
                   request, stream_with_context, url_for)

from core.inventory import effective_quantity
from storage import db as dbm
from storage import snapshot as snap
from storage.replay import ReplayFrame, ReplayFrameCache
from tools import observation as observation_tools
from data import private_real
from data.private_real import PrivateRealDataError
from web import catalog_diagnostics
from web.experiment_groups import (
    FRAMEWORK_PRESETS,
    MODEL_PRESETS,
    ExperimentGroupStore,
    ExperimentGroupStoreError,
)
from web.leaderboard import (
    RUN_RESULT_STATUSES,
    build_charts,
    build_leaderboard,
    build_run_results,
    decorate_run,
    merge_chart_payloads,
)
from web.runner import (
    RunPhaseClosedError,
    RunRuntimeUnavailableError,
    _now_iso,
    load_default_scenario,
    load_scenario,
)
from web.run_worker import RunWorker


ORDER_STATUS_KEYS = [
    "status_ordered", "status_shipped", "status_late",
    "status_stockout", "status_insufficient_balance",
    "status_delivered", "status_cancelled",
    "status_settled_normal", "status_settled_refund",
    "status_settled_only_refund", "status_settled_bad_review",
]
MERCHANT_METRIC_KEYS = [
    "balance", "deposit_pool", "in_transit", "receivable", "cum_fine",
    "n_active_listings", "cum_gmv", "revenue_rate",
    "cum_cost", "cum_gross_profit", "cum_net_profit", "cum_fee", "net_assets",
    # Shop rating series (only written when scenario.shop_rating.enabled).
    # Frontend tolerates missing keys; merchant section just shows empty charts
    # for rating-disabled scenarios.
    "shop_rating_mean", "shop_rating_score", "shop_rating_stars",
    "shop_rating_order_count",
    "shop_quality_multiplier", "shop_reputation_multiplier",
    "shop_demand_multiplier",
    "shop_reputation_evidence_count", "shop_qualified_transaction_count",
    "shop_service_quality_score", "shop_service_quality_stars",
    "shop_service_quality_multiplier",
    "public_review_rating", "public_review_count",
    "public_review_eligible_count", "public_review_response_rate",
    "public_review_full_response_rating", "public_review_selection_gap",
    "public_review_quality_gap", "public_review_confidence",
    "public_review_raw_quality_multiplier",
    "public_review_quality_multiplier",
    "public_review_reputation_multiplier",
    "public_review_demand_multiplier",
    "shop_n_good_effective", "shop_n_bad_effective",
]


class CatalogDiagnosticsNotMaterialized(Exception):
    pass


_LEADERBOARD_MIN_INTERVAL = 30.0
_TERMINAL_CHART_CACHE_VERSION = 9
_TERMINAL_CHART_CACHE_FILENAME = ".dashboard_terminal_charts.v9.json.gz"


REACT_MODEL_PRICING = [
    {
        "model": "qwen3.7-max",
        "input": 1.66,
        "output": 4.97,
        "cached_input": 0.17,
    },
    {
        "model": "qwen3.7-plus",
        "input": 0.28,
        "output": 1.10,
        "cached_input": 0.06,
    },
    {
        "model": "qwen3.5-27b",
        "input": 0.083,
        "output": 0.66,
        "cached_input": 0.008,
    },
    {
        "model": "bailian/deepseek-v4-pro",
        "input": 0.435,
        "output": 0.87,
        "cached_input": 0.004,
    },
    {
        "model": "bailian/deepseek-v4-flash",
        "input": 0.14,
        "output": 0.28,
        "cached_input": 0.03,
    },
    {
        # OpenRouter CoreWeave FP8 list price (USD per 1M tokens).
        "model": "deepseek/deepseek-v4-flash-0731",
        "input": 0.13,
        "output": 0.28,
        "cached_input": 0.07,
    },
    {
        # OpenRouter Google Vertex global listed price after the current
        # Vertex 50% promo (USD per 1M tokens, default/standard tier).
        # Google intro list through 2026-12-31 is $0.75 / $3.75 / $0.075.
        "model": "google/gemini-3.7-flash",
        "input": 0.375,
        "output": 1.875,
        "cached_input": 0.0375,
    },
    {
        "model": "bailian/kimi-k2.6",
        "input": 0.90,
        "output": 3.75,
        "cached_input": 0.15,
    },
    {
        "model": "moonshot/kimi-k3",
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
    },
    {
        "model": "bailian/glm-5.1",
        "input": 0.83,
        "output": 3.31,
        "cached_input": 0.15,
    },
    {
        "model": "bailian/glm-5.2",
        "input": 1.10,
        "output": 3.87,
        "cached_input": 0.22,
    },
    {
        "model": "claude-sonnet-4-6",
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
    },
    {
        # Anthropic introductory pricing through 2026-08-31.
        "model": "claude-sonnet-5",
        "input": 2.00,
        "output": 10.00,
        "cached_input": 0.20,
    },
    {
        "model": "claude-opus-4-7",
        "input": 5.00,
        "output": 25.00,
        "cached_input": 0.50,
    },
    {
        "model": "claude-opus-4-8",
        "input": 5.00,
        "output": 25.00,
        "cached_input": 0.50,
    },
    {
        "model": "gemini-3.5-flash",
        "input": 1.50,
        "output": 9.00,
        "cached_input": 0.15,
    },
    {
        "model": "gemini-3.1-pro-preview",
        "input": 2.00,
        "output": 12.00,
        "cached_input": 0.20,
    },
    {
        "model": "gpt-5.5-0424-global",
        "input": 5.00,
        "output": 30.00,
        "cached_input": 0.50,
    },
    {
        "model": "gpt-5.6-sol",
        "input": 5.00,
        "output": 30.00,
        "cached_input": 0.50,
    },
    {
        # OpenRouter stealth preview (single `stealth` upstream): free during
        # the preview window, 1M context. Re-check the catalog before assuming
        # zero cost — stealth previews can gain list pricing without notice.
        "model": "stealth/ox-alpha",
        "input": 0.0,
        "output": 0.0,
        "cached_input": 0.0,
    },
]
REACT_MODEL_PRICING_BY_MODEL = {
    row["model"]: row for row in REACT_MODEL_PRICING
}
HUMAN_MODEL_PRESETS = [
    "Beethoven",
    "Mozart",
    "Chopin",
    "Bach",
    "Liszt",
    "Schubert",
    "Tchaikovsky",
    "Vivaldi",
    "Rachmaninoff",
    "Debussy",
]

def _list_snapshot_steps(runs_root: str, run_id: str) -> list[int]:
    snap_dir = os.path.join(snap.run_dir(runs_root, run_id), "env_snapshot")
    if not os.path.isdir(snap_dir):
        return []
    out: list[int] = []
    for name in os.listdir(snap_dir):
        if name.startswith("t_") and name.endswith(".json"):
            try:
                out.append(int(name[2:-5]))
            except ValueError:
                pass
    out.sort()
    return out


def make_blueprint(registry) -> Blueprint:
    bp = Blueprint("dashboard", __name__)
    replay_cache = ReplayFrameCache(registry.runs_root)
    experiment_group_store = ExperimentGroupStore(registry.runs_root)
    # Cache per app/registry. Run membership and status changes rebuild
    # immediately; current_t-only changes are throttled below.
    leaderboard_lock = threading.Lock()
    leaderboard_cache: dict = {
        "structure": None,
        "progress": None,
        "payload": None,
        "ts": 0.0,
    }
    terminal_cache_lock = threading.Lock()
    terminal_chart_build_lock = threading.Lock()
    terminal_cache: dict = {
        "loaded": False,
        "fingerprints": {},
        "run_results": [],
        "charts": {},
    }
    terminal_cache_path = os.path.join(
        registry.runs_root,
        _TERMINAL_CHART_CACHE_FILENAME,
    )

    def _is_cacheable_terminal(row: dict) -> bool:
        # A terminal row without a durable end timestamp still has a
        # wall-clock-dependent elapsed_ms and therefore is not immutable.
        return (
            row.get("status") in RUN_RESULT_STATUSES
            and bool(row.get("finished_at"))
        )

    def _terminal_fingerprint(row: dict) -> str:
        identity = [
            row.get("run_id"),
            row.get("status"),
            row.get("current_t"),
            row.get("finished_at"),
            row.get("name"),
            row.get("bootstrap_agent"),
            row.get("bootstrap_config_json"),
            row.get("scenario_yaml"),
        ]
        return hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _load_terminal_cache() -> None:
        if terminal_cache["loaded"]:
            return
        terminal_cache["loaded"] = True
        try:
            with gzip.open(terminal_cache_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("version") != _TERMINAL_CHART_CACHE_VERSION:
            return
        if not isinstance(payload.get("fingerprints"), dict):
            return
        if not isinstance(payload.get("run_results"), list):
            return
        if not isinstance(payload.get("charts"), dict):
            return
        terminal_cache["fingerprints"] = payload["fingerprints"]
        terminal_cache["run_results"] = payload["run_results"]
        terminal_cache["charts"] = payload["charts"]

    def _write_terminal_cache() -> None:
        os.makedirs(registry.runs_root, exist_ok=True)
        temporary_path = (
            f"{terminal_cache_path}.tmp.{os.getpid()}."
            f"{threading.get_ident()}"
        )
        payload = {
            "version": _TERMINAL_CHART_CACHE_VERSION,
            "fingerprints": terminal_cache["fingerprints"],
            "run_results": terminal_cache["run_results"],
            "charts": terminal_cache["charts"],
        }
        try:
            with gzip.open(temporary_path, "wt", encoding="utf-8") as f:
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            os.replace(temporary_path, terminal_cache_path)
        except OSError:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    def _prepare_terminal_data(
        terminal_rows: list[dict],
    ) -> tuple[list[dict], dict, list[dict], dict[str, str]]:
        """Prepare cached summaries while holding only the short metadata lock."""
        with terminal_cache_lock:
            _load_terminal_cache()
            current_fingerprints = {
                str(row["run_id"]): _terminal_fingerprint(row)
                for row in terminal_rows
            }
            old_fingerprints = dict(terminal_cache["fingerprints"])
            cached_results = {
                str(row.get("run_id")): row
                for row in terminal_cache["run_results"]
                if row.get("run_id") is not None
            }
            unchanged_ids = {
                run_id
                for run_id, fingerprint in current_fingerprints.items()
                if old_fingerprints.get(run_id) == fingerprint
                and run_id in cached_results
            }
            changed_rows = [
                row
                for row in terminal_rows
                if str(row["run_id"]) not in unchanged_ids
            ]
            changed_results = build_run_results(registry, changed_rows)
            results_by_run = {
                run_id: cached_results[run_id]
                for run_id in unchanged_ids
            }
            results_by_run.update({
                str(row["run_id"]): row
                for row in changed_results
            })
            terminal_results = [
                results_by_run[str(row["run_id"])]
                for row in terminal_rows
                if str(row["run_id"]) in results_by_run
            ]
            terminal_results.sort(
                key=lambda row: row.get("started_at") or "",
                reverse=True,
            )

            cached_chart_ids = {
                str(row.get("run_id"))
                for row in (terminal_cache["charts"].get("runs") or [])
                if row.get("run_id") is not None
            }
            reusable_chart_ids = unchanged_ids & cached_chart_ids
            reusable_results = [
                row for row in terminal_results
                if str(row["run_id"]) in reusable_chart_ids
            ]
            terminal_result_ids = {
                str(row["run_id"]) for row in terminal_results
            }
            if reusable_chart_ids == terminal_result_ids:
                reusable_charts = terminal_cache["charts"]
            else:
                reusable_charts = (
                    merge_chart_payloads(
                        [terminal_cache["charts"]],
                        reusable_results,
                    )
                    if reusable_results else {}
                )

            terminal_charts = reusable_charts
            chart_build_results = [
                row for row in terminal_results
                if str(row["run_id"]) not in reusable_chart_ids
            ]

            cache_changed = (
                current_fingerprints != old_fingerprints
                or terminal_results != terminal_cache["run_results"]
                or terminal_charts != terminal_cache["charts"]
            )
            terminal_cache["fingerprints"] = current_fingerprints
            terminal_cache["run_results"] = terminal_results
            terminal_cache["charts"] = terminal_charts
            if cache_changed:
                _write_terminal_cache()
            return (
                terminal_results,
                terminal_charts,
                chart_build_results,
                current_fingerprints,
            )

    def _terminal_run_data(
        terminal_rows: list[dict],
        *,
        include_charts: bool,
    ) -> tuple[list[dict], dict]:
        """Reuse immutable terminal runs and only build new/changed fragments."""
        if not include_charts:
            terminal_results, terminal_charts, _, _ = (
                _prepare_terminal_data(terminal_rows)
            )
            return terminal_results, terminal_charts

        # Full chart builds can take minutes on a cold cache. Serialize those
        # builders without blocking the lightweight HTML/summary path.
        with terminal_chart_build_lock:
            (
                terminal_results,
                reusable_charts,
                chart_build_results,
                fingerprint_snapshot,
            ) = _prepare_terminal_data(terminal_rows)
            if not chart_build_results:
                return terminal_results, reusable_charts

            changed_charts = build_charts(registry, chart_build_results)
            terminal_charts = merge_chart_payloads(
                [reusable_charts, changed_charts],
                terminal_results,
            )
            with terminal_cache_lock:
                # An overview request may have observed a newer run set while
                # charts were building. Do not overwrite that newer cache.
                if terminal_cache["fingerprints"] == fingerprint_snapshot:
                    terminal_cache["charts"] = terminal_charts
                    _write_terminal_cache()
            return terminal_results, terminal_charts

    @bp.teardown_request
    def _release_request_conns(exc):
        stack = g.pop("_merchantbench_conn_stack", None)
        if stack is not None:
            stack.close()
        g.pop("_merchantbench_conn_cache", None)

    def _conn(run_id: str):
        try:
            stack = getattr(g, "_merchantbench_conn_stack", None)
            if stack is None:
                stack = ExitStack()
                g._merchantbench_conn_stack = stack
                g._merchantbench_conn_cache = {}
            cache = g._merchantbench_conn_cache
            if run_id not in cache:
                cache[run_id] = stack.enter_context(registry.lease_conn_for(run_id))
            return cache[run_id]
        except KeyError as e:
            if "is being deleted" in str(e):
                abort(409, description="run is being deleted")
            abort(404)
        except ValueError:
            abort(404)
        except sqlite3.Error:
            abort(503)

    def _get_or_create_worker(run_id: str, allow_restart: bool = False) -> RunWorker | None:
        """Look up the worker for a run, creating one only when truly absent.

        Pass ``allow_restart=True`` only from an explicit start/resume path.
        Those are the only actions where replacing a terminal in-memory worker
        with a fresh pending one is correct. Read-only endpoints (status,
        stream, ...) must keep observing the existing terminal worker so its
        state ('finished') is reported truthfully — otherwise reopening a
        finished run would show 'pending' indefinitely.
        """
        if allow_restart:
            registry._require(run_id)
        with registry.lock:
            w = registry.workers.get(run_id)
            if w is None and not allow_restart:
                return None
            if w is None:
                w = RunWorker(registry, run_id)
                if allow_restart and getattr(w, "state", None) in ("stopped", "finished"):
                    w.state = "pending"
                registry.workers[run_id] = w
            elif allow_restart and getattr(w, "state", None) in ("stopped", "finished"):
                w = RunWorker(registry, run_id)
                w.state = "pending"
                registry.workers[run_id] = w
            return w

    # ---------- run management ----------

    @bp.post("/runs")
    def create_run():
        body = request.get_json(silent=True) or {}
        if "scenario_path" in body:
            scenario = load_scenario(body["scenario_path"])
        elif "scenario" in body:
            scenario = body["scenario"]
        else:
            scenario = load_default_scenario()
        master_seed = body.get("master_seed")
        name = body.get("name")
        bootstrap_agent = body.get("bootstrap_agent", "none")
        bootstrap_config = body.get("bootstrap_config") or {}
        # If the caller didn't pin a base_url for spawned baselines, use
        # the URL this request came in on. That way an env on a non-default
        # port (e.g. dashboard at :5050) doesn't have to be told twice — the
        # spawned subprocess connects back to the same host:port.
        bootstrap_base_url = body.get("bootstrap_base_url") or \
            request.host_url.rstrip("/")
        auto_start = bool(body.get("auto_start", False))  # API default: do not auto-start
        interval_ms = int(body.get("interval_ms", 500))
        try:
            run_id = registry.create_run(scenario, master_seed, name,
                                         bootstrap_agent=bootstrap_agent,
                                         bootstrap_base_url=bootstrap_base_url,
                                         auto_start=auto_start,
                                         interval_ms=interval_ms,
                                         bootstrap_config=bootstrap_config)
        except (ValueError, PrivateRealDataError) as e:
            return jsonify({"error": str(e)}), 400
        row = dbm.get_run(_conn(run_id), run_id)
        out = {"run_id": run_id, "name": row.get("name"), "t": 0}
        auth = registry.auth_for_run(run_id)
        if auth and auth.get("agent_token"):
            out["agent_token"] = auth["agent_token"]
        return jsonify(out)

    @bp.get("/runs")
    def list_runs():
        return jsonify(registry.list_runs())

    @bp.get("/runs/<run_id>")
    def get_run(run_id):
        row = dbm.get_run(_conn(run_id), run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify(row)

    @bp.delete("/runs/<run_id>")
    def delete_run(run_id):
        result = registry.delete_run(run_id)
        if not result.get("deleted"):
            status_code = 409 if result.get("status") == "busy" else 404
            return jsonify({
                "error": result.get("reason", "not found"),
                "run_id": result.get("run_id", run_id),
                "status": result.get("status", "not_found"),
            }), status_code
        return jsonify(result)

    @bp.post("/runs/<run_id>/step")
    def step(run_id):
        try:
            return jsonify(registry.step(run_id))
        except RunRuntimeUnavailableError:
            row = registry.get_run(run_id)
            if not row:
                return jsonify({"error": "not found"}), 404
            code = 410 if row.get("status") in ("finished", "stopped") else 409
            return jsonify({"error": "run_runtime_not_loaded", "state": row.get("status")}), code

    @bp.post("/runs/<run_id>/auto_step")
    def auto_step(run_id):
        n = int(request.args.get("n", 1))
        try:
            return jsonify(registry.auto_step(run_id, n))
        except RunRuntimeUnavailableError:
            row = registry.get_run(run_id)
            if not row:
                return jsonify({"error": "not found"}), 404
            code = 410 if row.get("status") in ("finished", "stopped") else 409
            return jsonify({"error": "run_runtime_not_loaded", "state": row.get("status")}), code

    # ---------- run control ----------

    @bp.post("/runs/<run_id>/start")
    def start_run(run_id):
        with registry.runtime_lifecycle_for(run_id):
            body = request.get_json(silent=True) or {}
            interval_ms = int(body.get("interval_ms", 500))
            persisted = registry.get_run(run_id)
            if not persisted:
                return jsonify({"error": "not found"}), 404
            if persisted.get("status") == "finished":
                return jsonify(_status_from_db(run_id, persisted))
            # The only endpoint allowed to replace a stopped worker with a
            # fresh pending one and rehydrate its Environment.
            w = _get_or_create_worker(run_id, allow_restart=True)
            worker_state = getattr(w, "state", None)
            was_live = worker_state in ("running", "paused", "draining")
            result = w.resume() if worker_state == "paused" else w.start(interval_ms)
            if (
                not was_live
                and result.get("state") == "running"
                and result.get("phase") == "running"
            ):
                base_url = body.get("bootstrap_base_url") or request.host_url.rstrip("/")
                registry._respawn_bootstrap(run_id, base_url)
            return jsonify(result)

    @bp.post("/runs/<run_id>/pause")
    def pause_run(run_id):
        with registry.runtime_lifecycle_for(run_id):
            w = _get_or_create_worker(run_id)
            if w is None:
                status = _status_from_db(run_id)
                return jsonify(status or {"error": "not found"}), 200 if status else 404
            return jsonify(w.pause())

    @bp.post("/runs/<run_id>/resume")
    def resume_run(run_id):
        with registry.runtime_lifecycle_for(run_id):
            persisted = registry.get_run(run_id)
            if not persisted:
                return jsonify({"error": "not found"}), 404
            with registry.lock:
                live_worker = registry.workers.get(run_id)
            if live_worker is None and persisted.get("status") == "paused":
                # A service restart discards the worker thread but deliberately
                # leaves the durable paused status behind. Resume is an explicit
                # request to rehydrate that runtime and continue it.
                w = _get_or_create_worker(run_id, allow_restart=True)
                return jsonify(w.start())
            w = live_worker
            if w is None:
                return jsonify(_status_from_db(run_id, persisted))
            return jsonify(w.resume())

    @bp.post("/runs/<run_id>/stop")
    def stop_run(run_id):
        with registry.runtime_lifecycle_for(run_id):
            with registry.lock:
                w = registry.workers.get(run_id)
            if w is None:
                row = registry.get_run(run_id)
                if not row:
                    return jsonify({"error": "not found"}), 404
                if row.get("status") not in ("finished", "stopped"):
                    conn = registry.conn_for(run_id)
                    dbm.mark_run_terminal(conn, run_id, "stopped", _now_iso())
                    registry.release_terminal_runtime(run_id)
                status = _status_from_db(run_id)
                return jsonify(status or {"error": "not found"}), 200 if status else 404
            result = w.stop()
            # Kill the bootstrap subprocess immediately rather than waiting for
            # the worker thread to exit. The thread will call
            # release_terminal_runtime when it finishes, which is idempotent.
            registry._kill_bootstrap(run_id)
            thread = getattr(w, "_thread", None)
            if thread is None or not thread.is_alive():
                registry.release_terminal_runtime(run_id, worker=w)
            return jsonify(result)

    def _parse_wall_clock_iso(value: str | None):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    def _now_for_elapsed(start_dt):
        if getattr(start_dt, "tzinfo", None) is None:
            return datetime.now()
        return datetime.now(timezone.utc)

    def _align_elapsed_clocks(start_dt, end_dt):
        start_aware = getattr(start_dt, "tzinfo", None) is not None
        end_aware = getattr(end_dt, "tzinfo", None) is not None
        if start_aware == end_aware:
            return start_dt, end_dt
        if start_aware:
            start_dt = start_dt.astimezone().replace(tzinfo=None)
        if end_aware:
            end_dt = end_dt.astimezone().replace(tzinfo=None)
        return start_dt, end_dt

    def _elapsed_ms(row: dict) -> int | None:
        start_dt = _parse_wall_clock_iso((row or {}).get("started_at"))
        if start_dt is None:
            return None
        finished_at = (row or {}).get("finished_at")
        if finished_at:
            end_dt = _parse_wall_clock_iso(finished_at) or _now_for_elapsed(start_dt)
        else:
            end_dt = _now_for_elapsed(start_dt)
        start_dt, end_dt = _align_elapsed_clocks(start_dt, end_dt)
        return int((end_dt - start_dt).total_seconds() * 1000)

    def _status_from_db(run_id: str, row: dict | None = None) -> dict | None:
        conn = _conn(run_id)
        row = row or dbm.get_run(conn, run_id)
        if not row:
            return None
        state = row.get("status") or "pending"
        # A persisted running/draining state only means a worker existed in some
        # process. Without an in-process worker, the dashboard must not imply
        # ticks are still advancing. Paused is different: it already promises
        # no progress and is a durable, explicitly resumable state.
        if state in ("running", "draining"):
            state = "stopped"
        t = int(row.get("current_t") or 0)
        active_counts = dbm.active_order_status_counts(conn, run_id)
        active_count = sum(active_counts.values())
        horizon = int(row.get("horizon") or 0)
        if state == "finished":
            phase = "finished"
        elif t >= horizon and active_count > 0:
            phase = "draining"
        elif t >= horizon:
            phase = "finished"
        else:
            phase = "running"
        return {
            "run_id": run_id,
            "state": state,
            "phase": phase,
            "t": t,
            "interval_ms": 0,
            "last_step_ms": 0,
            "elapsed_ms": _elapsed_ms(row),
            "active_orders_remaining": active_count,
            "active_order_status_counts": active_counts,
        }

    def _status_for_worker(run_id: str, w, row: dict | None = None) -> dict:
        result = w.status()
        db_t = int((row or {}).get("current_t") or 0)
        if not result.get("t") and db_t:
            result["t"] = db_t
        if row is None:
            try:
                row = dbm.get_run(_conn(run_id), run_id)
            except (sqlite3.Error, OSError):
                row = None
        result["elapsed_ms"] = _elapsed_ms(row)
        return result

    @bp.get("/runs/<run_id>/status")
    def status_run(run_id):
        with registry.lock:
            w = registry.workers.get(run_id)
        # If a worker exists, prioritize its state even if DB is unavailable
        if w is not None:
            try:
                row = dbm.get_run(_conn(run_id), run_id)
            except (sqlite3.Error, OSError):
                row = None
            return jsonify(_status_for_worker(run_id, w, row))
        # No worker - fall back to DB
        try:
            row = dbm.get_run(_conn(run_id), run_id)
        except (sqlite3.Error, OSError):
            row = None
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify(_status_from_db(run_id, row))

    @bp.get("/runs/<run_id>/stream")
    def stream_run(run_id):
        with registry.lock:
            w = registry.workers.get(run_id)
            sub = w.subscribe() if w is not None else None
        if w is None:
            status = _status_from_db(run_id)
            if status is None:
                return jsonify({"error": "not found"}), 404

            def status_only_gen():
                yield f"event: hello\ndata: {json.dumps(status)}\n\n"
                if status.get("state") == "finished":
                    yield f"event: finished\ndata: {json.dumps(status)}\n\n"

            resp = Response(status_only_gen(), mimetype="text/event-stream")
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp

        @stream_with_context
        def gen():
            # initial snapshot frame
            yield f"event: hello\ndata: {json.dumps(_status_for_worker(run_id, w))}\n\n"
            last_beat = time.time()
            while True:
                try:
                    ev = sub.get(timeout=15.0)
                    yield f"event: {ev.get('type', 'tick')}\ndata: {json.dumps(ev)}\n\n"
                    last_beat = time.time()
                    if ev.get("type") in ("finished", "error", "stopped"):
                        break
                except queue.Empty:
                    yield f"event: heartbeat\ndata: {{\"t\":{time.time():.0f}}}\n\n"

        resp = Response(gen(), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        resp.call_on_close(lambda: w.unsubscribe(sub))
        return resp

    # ---------- agents ----------

    @bp.get("/runs/<run_id>/agents")
    def list_agents(run_id):
        return jsonify(registry.list_agents(run_id))

    @bp.post("/runs/<run_id>/agents")
    def add_agent(run_id):
        body = request.get_json(force=True) or {}
        try:
            existing = registry.list_agents(run_id)
            agent_id = body.get("agent_id") or f"agent_{len(existing)}"
            name = body.get("name") or agent_id
            out = registry.add_agent(run_id, agent_id, name)
            return jsonify(out)
        except RunPhaseClosedError as exc:
            return jsonify({
                "error": "agent_addition_closed",
                "state": exc.phase,
            }), 410 if exc.phase == "finished" else 409
        except RunRuntimeUnavailableError:
            row = registry.get_run(run_id)
            if not row:
                return jsonify({"error": "not found"}), 404
            state = row.get("status")
            code = 410 if state in ("finished", "stopped") else 409
            return jsonify({
                "error": "run_runtime_not_loaded",
                "state": state,
            }), code

    # ---------- snapshots ----------

    @bp.get("/runs/<run_id>/snapshots/<int:t>")
    def get_snapshot(run_id, t):
        out = snap.read_env_snapshot(registry.runs_root, run_id, t)
        if out is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(out)

    @bp.get("/runs/<run_id>/metrics/daily")
    def get_daily(run_id):
        return jsonify(dbm.load_daily_aggregates(_conn(run_id), run_id))

    @bp.get("/runs/<run_id>/events/<int:t>")
    def get_events(run_id, t):
        return jsonify(dbm.load_events_at(_conn(run_id), run_id, t))

    @bp.get("/runs/<run_id>/snapshot_steps")
    def list_snapshot_steps(run_id):
        return jsonify(_list_snapshot_steps(registry.runs_root, run_id))

    # ---------- section endpoints ----------

    def _series_from(conn, run_id: str, agent_id: str, keys: list[str],
                     t_from, t_to) -> dict[str, list[list]]:
        bulk = dbm.load_metrics_bulk(conn, run_id, agent_id, keys, t_from, t_to)
        return {k: [[t, v] for (t, v) in bulk.get(k, [])] for k in keys}

    def _current_t_for_dashboard(run_id: str) -> int | None:
        env = registry.get_env(run_id)
        if env is not None:
            return int(env.t)
        row = dbm.get_run(_conn(run_id), run_id)
        if not row:
            return None
        return int(row.get("current_t") or 0)

    def _as_of_arg() -> int | None:
        return int(request.args["as_of"]) if "as_of" in request.args else None

    def _default_t_to(t_to: int | None, as_of: int | None) -> int | None:
        return as_of if as_of is not None and t_to is None else t_to

    _PRODUCT_MUTABLE_FIELDS = {
        "quantity", "quantity_updated_t", "price", "is_listed_by_supplier",
        "delist_recover_t", "price_recover_t", "timeout_active",
        "timeout_recover_t", "supplier_ship_hours",
    }

    def _overlay_product_rows(rows: list[dict], frame: ReplayFrame | None) -> list[dict]:
        if frame is None:
            return rows
        out = []
        for row in rows:
            item = dict(row)
            overlay = frame.products.get(item.get("product_id"))
            if overlay:
                for key in _PRODUCT_MUTABLE_FIELDS:
                    if key in overlay:
                        item[key] = overlay[key]
            out.append(item)
        return out

    def _product_rows_by_id(
        run_id: str, product_ids: list[str], frame: ReplayFrame | None
    ) -> dict[str, dict]:
        if not product_ids:
            return {}
        qmarks = ",".join("?" for _ in product_ids)
        rows = _conn(run_id).execute(
            "SELECT product_id, name, category, supplier_id, supplier_name,"
            " price, ref_price, base_price, quantity, max_quantity,"
            " is_listed_by_supplier, timeout_active, logistics_hours,"
            " base_ship_hours, supplier_ship_hours,"
            " historical_avg_rating, shop_rating, return_buyer_rate, supplier_age_years,"
            " cancel_rate, refund_rate, only_refund_rate, timeout_rate, bad_review_rate,"
            " price_change_rate, supplier_delist_rate, elasticity,"
            " delist_recover_t, price_recover_t, timeout_recover_t"
            f" FROM products WHERE run_id=? AND product_id IN ({qmarks})",
            (run_id, *product_ids),
        ).fetchall()
        return {
            row["product_id"]: row
            for row in _overlay_product_rows([dict(r) for r in rows], frame)
        }

    def _supplier_log_hour(ship_hours, logistics_hours):
        if logistics_hours is None:
            return None
        if ship_hours is None:
            return logistics_hours
        return ship_hours + logistics_hours

    def _latest_metric_values(series: dict, keys: list[str]) -> dict:
        out = {}
        for key in keys:
            values = series.get(key) or []
            out[key] = float(values[-1][1]) if values else 0.0
        return out

    def _per_listing_pnl(run_id: str, agent_id: str, t_to: int | None = None) -> dict:
        out = {}
        order_clause = ""
        settled_clause = ""
        params: list = []
        if t_to is not None:
            order_clause = " AND order_t<=?"
            settled_clause = " AND settled_t<=?"
            params.extend([int(t_to), int(t_to), int(t_to)])
        params.extend([run_id, agent_id])
        for r in _conn(run_id).execute(
            "SELECT product_id,"
            " COALESCE(SUM(CASE WHEN current_status NOT IN"
            "   ('stockout','insufficient_balance')"
            f"{order_clause}"
            "   THEN sale_price - purchase_price ELSE 0 END), 0) AS gross_profit,"
            " COALESCE(SUM(CASE WHEN settled_t IS NOT NULL"
            f"{settled_clause}"
            f"   THEN {dbm.order_net_profit_sql()} ELSE 0 END), 0) AS profit,"
            " COALESCE(SUM(CASE WHEN settled_t IS NOT NULL"
            f"{settled_clause}"
            "   THEN total_penalty ELSE 0 END), 0) AS fine"
            " FROM orders WHERE run_id=? AND agent_id=?"
            " GROUP BY product_id",
            tuple(params),
        ).fetchall():
            out[r["product_id"]] = {
                "cum_gross_profit": round(float(r["gross_profit"]), 2),
                "cum_net_profit": round(float(r["profit"]), 2),
                "cum_fine": round(float(r["fine"]), 2),
            }
        return out

    def _attach_listing_pnl(listings: list[dict], per_listing_pnl: dict) -> list[dict]:
        for listing in listings:
            pnl = per_listing_pnl.get(listing["product_id"], {})
            listing["cum_gross_profit"] = pnl.get("cum_gross_profit", 0.0)
            listing["cum_net_profit"] = pnl.get("cum_net_profit", 0.0)
            listing["cum_fine"] = pnl.get("cum_fine", 0.0)
        return listings

    def _human_platform_rules_payload(conn, run_id: str, scenario: dict) -> dict:
        """Render public platform rules in Chinese without changing agent tools."""
        scenario_zh = copy.deepcopy(scenario)
        scenario_zh.setdefault("agent", {})["language"] = "zh"
        categories = [
            str(row["category"])
            for row in conn.execute(
                "SELECT DISTINCT category FROM products"
                " WHERE run_id=? AND category IS NOT NULL"
                " ORDER BY category",
                (run_id,),
            ).fetchall()
        ]
        products = {
            f"category_{index}": SimpleNamespace(category=category)
            for index, category in enumerate(categories)
        }
        brief = observation_tools.compose_system_brief(SimpleNamespace(
            scenario=scenario_zh,
            products=products,
        ))
        context = brief["context"]
        public_keys = (
            "default_promised_ship_hours",
            "max_active_listings",
            "horizon_days",
            "activation_hours",
            "normal_settlement_delay_hours",
            "initial_cash",
            "initial_deposit",
            "penalties",
            "field_logic",
        )
        return {
            "system_prompt": brief["system_prompt"],
            "important_rules": {key: context[key] for key in public_keys},
        }

    def _human_listing_period_stats(
        conn,
        run_id: str,
        agent_id: str,
        current_t: int,
        step_hours: int,
        listings: list[dict],
    ) -> dict[str, dict]:
        """Current-listing-period no-sale facts for the Human UI only."""
        anchors = {
            str(listing.get("product_id")): int(listing.get("listed_at") or 0)
            for listing in listings
            if isinstance(listing, dict) and listing.get("product_id")
        }
        if not anchors:
            return {}
        values_sql = ",".join("(?, ?)" for _ in anchors)
        values_params = [
            value
            for product_id, listed_at in anchors.items()
            for value in (product_id, listed_at)
        ]
        rows = conn.execute(
            f"WITH listing_periods(product_id, listed_at) AS (VALUES {values_sql})"
            " SELECT lp.product_id, lp.listed_at, MAX(o.order_t) AS last_sale_t"
            " FROM listing_periods lp"
            " LEFT JOIN orders o"
            " ON o.run_id=? AND o.agent_id=? AND o.product_id=lp.product_id"
            " AND o.order_t>lp.listed_at AND o.order_t<=?"
            " AND o.current_status NOT IN ('stockout','insufficient_balance')"
            " GROUP BY lp.product_id, lp.listed_at",
            (*values_params, run_id, agent_id, int(current_t)),
        ).fetchall()
        out = {}
        for row in rows:
            listed_at = int(row["listed_at"] or 0)
            last_sale_t = (
                int(row["last_sale_t"])
                if row["last_sale_t"] is not None
                else None
            )
            anchor_t = last_sale_t if last_sale_t is not None else listed_at
            out[str(row["product_id"])] = {
                "first_listed_day": (listed_at * step_hours) // 24 + 1,
                "last_sale_day": (
                    (last_sale_t * step_hours) // 24 + 1
                    if last_sale_t is not None
                    else 0
                ),
                "days_without_sales": (
                    max(0, (int(current_t) - anchor_t) * step_hours) // 24
                ),
            }
        return out

    def _human_safe_merchant_payload(
        run_id: str,
        agent_id: str,
        payload: dict,
        *,
        level: str | None = None,
        t_from: int | None = None,
        t_to: int | None = None,
    ) -> dict:
        """Remove administrator-only merchant diagnostics before browser delivery."""
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            return {"error": "not found"}
        scenario = yaml.safe_load(row["scenario_yaml"])
        payload_t = payload.get("t")
        current_t = int(
            payload_t if payload_t is not None else row.get("current_t") or 0
        )
        step_hours = _step_hours_from_scenario(scenario)
        listing_periods = _human_listing_period_stats(
            conn,
            run_id,
            agent_id,
            current_t,
            step_hours,
            list(payload.get("listings") or []),
        )
        safe_listing_keys = (
            "product_id", "name", "category", "quantity", "sale_price",
            "supplier_price", "supplier_id", "supplier_name",
            "supplier_ship_hours", "supplier_logistics_hours",
            "historical_avg_rating", "shop_rating", "supplier_age_years",
            "cum_gross_profit", "cum_net_profit", "cum_fine",
        )
        listings = []
        for source in payload.get("listings") or []:
            listing = {key: source.get(key) for key in safe_listing_keys}
            supplier_price = source.get("supplier_price")
            sale_price = source.get("sale_price")
            listing["price_ratio"] = (
                round(float(sale_price) / float(supplier_price), 4)
                if supplier_price is not None
                and float(supplier_price) > 0
                and sale_price is not None
                else None
            )
            listing["listing_rating"] = source.get("downstream_rating")
            listing["procured_orders"] = int(source.get("cum_sales") or 0)
            listing.update(listing_periods.get(str(source.get("product_id")), {}))
            listings.append(listing)

        safe_series_keys = (
            "balance", "deposit_pool", "in_transit", "receivable",
            "net_assets", "n_active_listings", "cum_gmv", "cum_cost",
            "cum_gross_profit", "cum_net_profit", "cum_fine", "cum_fee",
            "shop_rating_mean", "shop_rating_score",
        )
        series = payload.get("series") or {}
        cash = payload.get("cash") or {}
        rating = payload.get("shop_rating") or {}
        product_name_rows = conn.execute(
            "SELECT DISTINCT o.product_id, COALESCE(p.name, '') AS name"
            " FROM orders o LEFT JOIN products p"
            " ON p.run_id=o.run_id AND p.product_id=o.product_id"
            " WHERE o.run_id=? AND o.agent_id=? AND o.order_t<=?",
            (run_id, agent_id, current_t),
        ).fetchall()
        listing_ops = payload.get("listing_ops") or {}
        daily_sales = payload.get("daily_sales_by_product") or {}
        if level in {"day", "week"}:
            listing_ops = dbm.load_dashboard_merchant_listing_ops(
                conn,
                run_id,
                agent_id,
                current_t=current_t,
                step_hours=step_hours,
                level=level,
                t_from=t_from,
                t_to=t_to,
            )
            daily_sales = dbm.load_dashboard_merchant_daily_sales_by_product(
                conn,
                run_id,
                agent_id,
                current_t=current_t,
                step_hours=step_hours,
                level=level,
                t_from=t_from,
                t_to=t_to,
            )

        def safe_buckets(source: dict) -> list[dict]:
            return [
                {
                    key: bucket.get(key)
                    for key in ("key", "label", "start_day", "end_day")
                }
                for bucket in source.get("buckets") or []
                if isinstance(bucket, dict)
            ]

        safe_listing_ops = {
            "grain": listing_ops.get("grain"),
            "days": list(listing_ops.get("days") or []),
            "buckets": safe_buckets(listing_ops),
            "series": {
                key: list((listing_ops.get("series") or {}).get(key) or [])
                for key in ("ops", "list", "delist", "price", "promise")
            },
        }
        safe_daily_sales = {
            "grain": daily_sales.get("grain"),
            "days": list(daily_sales.get("days") or []),
            "buckets": safe_buckets(daily_sales),
            "series": [],
        }
        for product in daily_sales.get("series") or []:
            if not isinstance(product, dict):
                continue
            safe_daily_sales["series"].append({
                key: product.get(key)
                for key in ("product_id", "name", "category")
            } | {
                "data": [
                    {
                        key: point.get(key)
                        for key in (
                            "bucket", "label", "start_day", "end_day", "day",
                            "orders", "value", "gmv", "gross_profit",
                            "net_profit", "supply_chain_anomalies",
                            "order_anomalies",
                        )
                    }
                    for point in product.get("data") or []
                    if isinstance(point, dict)
                ],
            })
        safe_shop_rating = None
        if rating:
            safe_shop_rating = {
                key: rating.get(key)
                for key in (
                    "enabled", "model", "score", "stars",
                    "rated_order_count", "qualified_transaction_count",
                    "reputation_evidence_count", "quality_multiplier",
                    "reputation_multiplier", "demand_multiplier",
                    "service_quality_score", "service_quality_stars",
                    "service_quality_multiplier", "rating_available",
                    "demand_source",
                )
            }
            public_reviews = rating.get("public_reviews")
            if isinstance(public_reviews, dict):
                safe_shop_rating["public_reviews"] = {
                    key: public_reviews.get(key)
                    for key in (
                        "model", "rating", "count", "eligible_count",
                        "response_rate", "full_response_rating",
                        "selection_gap", "quality_gap", "affects_demand",
                        "stars", "confidence", "raw_quality_multiplier",
                        "quality_multiplier", "reputation_multiplier",
                        "demand_multiplier",
                    )
                }
        return {
            "t": current_t,
            "agent_id": agent_id,
            "cash": {
                key: cash.get(key)
                for key in (
                    "balance", "deposit_pool", "in_transit", "receivable",
                    "cumulative_fine", "net_assets",
                )
            },
            "listings": listings,
            "series": {key: series.get(key, []) for key in safe_series_keys},
            "listing_ops": safe_listing_ops,
            "daily_sales_by_product": safe_daily_sales,
            "shop_rating": safe_shop_rating,
            "product_names": {
                str(item["product_id"]): item["name"]
                for item in product_name_rows
            },
        }

    def _series_latest(series: dict, key: str):
        values = series.get(key) or []
        return values[-1][1] if values else None

    def _public_review_payload(state: dict) -> dict:
        """Serialize one canonical public-review state for dashboard clients."""
        payload = {
            "model": str(state["model"]),
            "rating": (
                round(float(state["rating"]), 4)
                if state["rating"] is not None else None
            ),
            "count": int(state["count"]),
            "eligible_count": int(state["eligible_count"]),
            "response_rate": round(float(state["response_rate"]), 4),
            "full_response_rating": (
                round(float(state["full_response_rating"]), 4)
                if state["full_response_rating"] is not None else None
            ),
            "selection_gap": (
                round(float(state["selection_gap"]), 4)
                if state["selection_gap"] is not None else None
            ),
            "quality_gap": (
                round(float(state["quality_gap"]), 4)
                if state["quality_gap"] is not None else None
            ),
            "affects_demand": bool(state["affects_demand"]),
        }
        for key in (
            "stars",
            "confidence",
            "raw_quality_multiplier",
            "quality_multiplier",
            "reputation_multiplier",
            "demand_multiplier",
        ):
            value = state.get(key)
            if value is not None:
                payload[key] = round(float(value), 4)
        return payload

    def _public_reviews_from_series(
        scenario: dict, series: dict, quality_score: float,
    ) -> dict | None:
        from core import listing_rating as listing_rating_mod
        from core import public_reviews as public_reviews_mod
        from core import rating as rating_mod

        review_cfg = scenario.get("public_reviews") or {}
        rating_cfg = scenario.get("shop_rating") or {}
        rating_model = str(rating_cfg.get("model") or "beta_event_v1")
        if (
            rating_model != listing_rating_mod.PUBLIC_REVIEW_RATING_MODEL
            or not review_cfg.get("enabled", False)
        ):
            return None
        count = int(_series_latest(series, "public_review_count") or 0)
        eligible_count = int(
            _series_latest(series, "public_review_eligible_count") or 0
        )
        rating = _series_latest(series, "public_review_rating")
        full_response_rating = _series_latest(
            series, "public_review_full_response_rating",
        )
        selection_gap = _series_latest(series, "public_review_selection_gap")
        quality_gap = _series_latest(series, "public_review_quality_gap")
        response_rate = _series_latest(series, "public_review_response_rate")
        if response_rate is None:
            response_rate = count / eligible_count if eligible_count else 0.0
        if (
            selection_gap is None
            and rating is not None
            and full_response_rating is not None
        ):
            selection_gap = float(rating) - float(full_response_rating)
        if quality_gap is None and rating is not None:
            quality_gap = float(rating) - float(quality_score)
        state = {
            "model": str(review_cfg.get("model") or "self_selection_v1"),
            "rating": rating,
            "count": count,
            "eligible_count": eligible_count,
            "response_rate": response_rate,
            "full_response_rating": full_response_rating,
            "selection_gap": selection_gap,
            "quality_gap": quality_gap,
            "affects_demand": True,
        }
        state["stars"] = (
            float(rating_mod.stars_from_score(
                rating, rating_cfg["bucket_thresholds"],
            ))
            if rating is not None else None
        )
        metric_keys = {
            "confidence": "public_review_confidence",
            "raw_quality_multiplier": (
                "public_review_raw_quality_multiplier"
            ),
            "quality_multiplier": "public_review_quality_multiplier",
            "reputation_multiplier": (
                "public_review_reputation_multiplier"
            ),
            "demand_multiplier": "public_review_demand_multiplier",
        }
        factors = {
            key: _series_latest(series, metric_key)
            for key, metric_key in metric_keys.items()
        }
        if any(value is None for value in factors.values()):
            factors = public_reviews_mod.public_review_demand_factors(
                rating,
                count,
                bucket_thresholds=list(rating_cfg["bucket_thresholds"]),
                star_multipliers=list(rating_cfg["star_multipliers"]),
                config=review_cfg,
            )
        state.update(factors)
        return _public_review_payload(state)

    def _shop_rating_from_series(scenario: dict, series: dict) -> dict | None:
        from core import listing_rating as listing_rating_mod
        from core import rating as rating_mod

        rating_cfg = scenario.get("shop_rating") or {}
        if not rating_cfg.get("enabled", False):
            return None
        model = str(rating_cfg.get("model") or "beta_event_v1")
        if model in listing_rating_mod.ORDER_OUTCOME_RATING_MODELS:
            uses_public_reviews = (
                model == listing_rating_mod.PUBLIC_REVIEW_RATING_MODEL
            )
            score = _series_latest(series, "shop_rating_mean")
            if score is None:
                score = _series_latest(series, "shop_rating_score")
            if score is None and not uses_public_reviews:
                score = float(rating_cfg["initial_rating"])
            stars = _series_latest(series, "shop_rating_stars")
            if stars is None and score is not None:
                stars = rating_mod.stars_from_score(
                    score, rating_cfg["bucket_thresholds"])
            if stars is not None:
                stars = int(stars)
            qualified_count = _series_latest(
                series, "shop_qualified_transaction_count",
            )
            if qualified_count is None:
                qualified_count = _series_latest(
                    series, "shop_rating_order_count",
                )
            reputation_evidence_count = _series_latest(
                series, "shop_reputation_evidence_count",
            )
            service_quality_score = _series_latest(
                series, "shop_service_quality_score",
            )
            if service_quality_score is None:
                service_quality_score = (
                    float(rating_cfg["initial_rating"])
                    if uses_public_reviews else score
                )
            service_quality_stars = _series_latest(
                series, "shop_service_quality_stars",
            )
            if service_quality_stars is None:
                service_quality_stars = rating_mod.stars_from_score(
                    service_quality_score, rating_cfg["bucket_thresholds"],
                )
            service_quality_multiplier = _series_latest(
                series, "shop_service_quality_multiplier",
            )
            if service_quality_multiplier is None:
                service_quality_multiplier = rating_mod.multiplier_from_stars(
                    int(service_quality_stars),
                    rating_cfg["star_multipliers"],
                )
            public_reviews = _public_reviews_from_series(
                scenario, series, float(service_quality_score),
            )
            if uses_public_reviews:
                score = (
                    public_reviews.get("rating")
                    if public_reviews is not None else None
                )
                stars = (
                    int(public_reviews["stars"])
                    if (
                        public_reviews is not None
                        and public_reviews.get("stars") is not None
                    ) else None
                )
            if reputation_evidence_count is None:
                reputation_evidence_count = (
                    public_reviews.get("count", 0)
                    if uses_public_reviews and public_reviews is not None
                    else qualified_count
                )
            quality_multiplier = _series_latest(
                series, "shop_quality_multiplier",
            )
            if quality_multiplier is None:
                quality_multiplier = (
                    public_reviews.get("quality_multiplier")
                    if (
                        uses_public_reviews and public_reviews is not None
                    )
                    else rating_mod.multiplier_from_stars(
                        stars, rating_cfg["star_multipliers"],
                    )
                )
            reputation_multiplier = _series_latest(
                series, "shop_reputation_multiplier",
            )
            if reputation_multiplier is None:
                if (
                    uses_public_reviews and public_reviews is not None
                ):
                    reputation_multiplier = public_reviews.get(
                        "reputation_multiplier", 1.0,
                    )
                elif model == listing_rating_mod.REPUTATION_VOLUME_RATING_MODEL:
                    reputation_multiplier = (
                        listing_rating_mod.reputation_volume_multiplier(
                            qualified_count or 0,
                            rating_cfg.get("reputation_volume"),
                        )
                    )
                else:
                    reputation_multiplier = 1.0
            demand_multiplier = _series_latest(series, "shop_demand_multiplier")
            if demand_multiplier is None:
                demand_multiplier = quality_multiplier * reputation_multiplier
            result = {
                "enabled": True,
                "model": model,
                "score": (
                    round(float(score), 4) if score is not None else None
                ),
                "stars": stars,
                "quality_multiplier": round(float(quality_multiplier), 4),
                "reputation_multiplier": round(
                    float(reputation_multiplier), 4,
                ),
                "demand_multiplier": round(float(demand_multiplier), 4),
                "rated_order_count": int(qualified_count or 0),
                "qualified_transaction_count": int(qualified_count or 0),
                "reputation_evidence_count": int(
                    reputation_evidence_count or 0
                ),
                "service_quality_score": round(
                    float(service_quality_score), 4,
                ),
                "service_quality_stars": int(service_quality_stars),
                "service_quality_multiplier": round(
                    float(service_quality_multiplier), 4,
                ),
                "rating_available": (
                    score is not None if uses_public_reviews else True
                ),
                "demand_source": (
                    "public_reviews"
                    if model == listing_rating_mod.PUBLIC_REVIEW_RATING_MODEL
                    else "service_quality_and_transaction_volume"
                    if model == listing_rating_mod.REPUTATION_VOLUME_RATING_MODEL
                    else "service_quality"
                ),
                "bucket_thresholds": list(rating_cfg["bucket_thresholds"]),
                "star_multipliers": list(rating_cfg["star_multipliers"]),
            }
            if public_reviews is not None:
                result["public_reviews"] = public_reviews
            return result
        prior_good = float(rating_cfg["prior_good"])
        prior_bad = float(rating_cfg["prior_bad"])
        score = _series_latest(series, "shop_rating_score")
        if score is None:
            score = rating_mod.posterior_mean(0.0, 0.0, prior_good, prior_bad)
        stars = _series_latest(series, "shop_rating_stars")
        if stars is None:
            stars = rating_mod.stars_from_score(score, rating_cfg["bucket_thresholds"])
        stars = int(stars)
        n_good = _series_latest(series, "shop_n_good_effective")
        n_bad = _series_latest(series, "shop_n_bad_effective")
        return {
            "enabled": True,
            "model": model,
            "score": round(float(score), 4),
            "stars": stars,
            "quality_multiplier": rating_mod.multiplier_from_stars(
                stars, rating_cfg["star_multipliers"]),
            "reputation_multiplier": 1.0,
            "demand_multiplier": rating_mod.multiplier_from_stars(
                stars, rating_cfg["star_multipliers"]),
            "n_good_effective": round(float(n_good or 0.0), 2),
            "n_bad_effective": round(float(n_bad or 0.0), 2),
            "bucket_thresholds": list(rating_cfg["bucket_thresholds"]),
            "star_multipliers": list(rating_cfg["star_multipliers"]),
        }

    def _downstream_listing_rating(
        scenario: dict, rating_sum: float, rating_count: float,
    ) -> float | None:
        from core import listing_rating as listing_rating_mod

        cfg = scenario.get("listing_rating") or {}
        if not cfg:
            return None
        return round(listing_rating_mod.compute_listing_rating(
            float(cfg.get("initial_rating", 4.0)),
            float(rating_sum or 0.0),
            float(rating_count or 0.0),
            float(cfg.get("prior_weight", 20.0)),
        ), 4)

    def _attach_downstream_listing_ratings(
        listings: list[dict], scenario: dict,
    ) -> list[dict]:
        for listing in listings:
            listing["downstream_rating"] = _downstream_listing_rating(
                scenario,
                listing.get("rating_sum", 0.0),
                listing.get("rating_count", 0.0),
            )
        return listings

    def _step_hours_from_scenario(scenario: dict) -> int:
        return int((scenario.get("run") or {}).get("step_hours", 1))

    def _daily_sales_by_product(
        run_id: str,
        agent_id: str,
        current_t: int,
        scenario: dict,
        merchant_products: dict[str, tuple[str, str]] | None = None,
    ) -> dict:
        return dbm.load_dashboard_merchant_daily_sales_by_product(
            _conn(run_id),
            run_id,
            agent_id,
            current_t=current_t,
            step_hours=_step_hours_from_scenario(scenario),
            merchant_products=merchant_products,
        )

    def _listing_ops(
        run_id: str,
        agent_id: str,
        current_t: int,
        scenario: dict,
    ) -> dict:
        return dbm.load_dashboard_merchant_listing_ops(
            _conn(run_id),
            run_id,
            agent_id,
            current_t=int(current_t),
            step_hours=_step_hours_from_scenario(scenario),
        )

    def _merchant_product_lifecycle_payload(
        run_id: str,
        agent_id: str,
        product_id: str,
        current_t: int,
        scenario: dict,
    ) -> dict:
        return {
            "agent_id": agent_id,
            "product_id": product_id,
            "agent_sales_lifecycle": (
                dbm.load_dashboard_merchant_product_sales_lifecycle(
                    _conn(run_id),
                    run_id,
                    agent_id,
                    product_id,
                    current_t=int(current_t),
                    step_hours=_step_hours_from_scenario(scenario),
                )
            ),
        }

    def _catalog_product_diagnostics_as_of(
        run_id: str,
        product_id: str,
        as_of: int | None,
        scenario: dict,
        *,
        initial_catalog: bool = False,
    ) -> dict | None:
        artifact = catalog_diagnostics.read_catalog_diagnostics_artifact(
            registry.catalog_diagnostics_path(run_id)
        )
        if artifact is None:
            raise CatalogDiagnosticsNotMaterialized
        product = dbm.load_product(
            _conn(run_id), run_id, product_id, initial=initial_catalog
        )
        if product is None:
            return None
        if as_of is not None and not initial_catalog:
            overlay = replay_cache.frame(run_id, int(as_of)).products.get(product_id)
            if overlay:
                product = replace(
                    product,
                    **{
                        key: overlay[key]
                        for key in _PRODUCT_MUTABLE_FIELDS
                        if key in overlay
                    },
                )
        small_share = float((scenario.get("data") or {}).get("small_share", 1.0))
        artifact_params = artifact.get("parameters") or {}
        category_band = (artifact.get("category_bands") or {}).get(product.category) or {
            "p10": [], "p50": [], "p90": []
        }
        return catalog_diagnostics.build_materialized_product_diagnostics(
            product,
            small_share=small_share,
            category_band=category_band,
            refund_penalty=float(artifact_params.get("refund_penalty_amount", 8.0)),
            bad_review_penalty=float(artifact_params.get("bad_review_penalty_amount", 5.0)),
        )

    def _catalog_diagnostics_unavailable(run_id: str):
        status_payload = catalog_diagnostics.read_catalog_diagnostics_status(
            registry.catalog_diagnostics_status_path(run_id)
        )
        state = str((status_payload or {}).get("status") or "missing")
        if state == "pending":
            response = jsonify({
                "error": "catalog_diagnostics_pending",
                "status": "pending",
                "retry_after_ms": 300,
            })
            response.status_code = 202
            response.headers["Retry-After"] = "1"
            return response
        if state == "failed":
            return jsonify({
                "error": "catalog_diagnostics_failed",
                "status": "failed",
            }), 503
        return jsonify({
            "error": "catalog_diagnostics_not_materialized",
            "status": state,
        }), 409

    def _merchant_section_from_db(run_id: str, agent_id: str, t_from, t_to):
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        agents = {a.agent_id: a for a in dbm.list_agents(conn, run_id)}
        agent = agents.get(agent_id)
        if agent is None:
            return jsonify({"error": "unknown agent"}), 404
        scenario = yaml.safe_load(row["scenario_yaml"])
        current_t = int(row.get("current_t") or 0)
        series = _series_from(conn, run_id, agent_id, MERCHANT_METRIC_KEYS, t_from, t_to)
        listings = dbm.load_dashboard_merchant_listings(conn, run_id, agent_id, current_t=current_t)
        _attach_downstream_listing_ratings(listings, scenario)
        _attach_listing_pnl(listings, _per_listing_pnl(run_id, agent_id))
        cash = dbm.load_latest_cash(conn, run_id, agent_id)
        cash_dict = cash.to_dict() if cash else {
            "balance": float((scenario.get("run") or {}).get("initial_cash", 0.0)),
            "deposit_pool": float((scenario.get("run") or {}).get("initial_deposit", 0.0)),
            "in_transit": 0.0,
            "receivable": 0.0,
            "cumulative_fine": 0.0,
        }
        return jsonify({
            "t": current_t,
            "agent_id": agent_id,
            "name": agent.name,
            "is_alive": agent.is_alive,
            "died_at_t": agent.died_at_t,
            "cash": cash_dict,
            "listings": listings,
            "series": series,
            "recent_actions": dbm.load_dashboard_merchant_action_events(
                conn, run_id, agent_id),
            "shop_rating": _shop_rating_from_series(scenario, series),
            "listing_ops": _listing_ops(run_id, agent_id, current_t, scenario),
            "daily_sales_by_product": _daily_sales_by_product(
                run_id, agent_id, current_t, scenario),
        })

    def _merchant_section_as_of(run_id: str, agent_id: str, as_of: int, t_from, t_to):
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        agents = {a.agent_id: a for a in dbm.list_agents(conn, run_id)}
        agent = agents.get(agent_id)
        if agent is None:
            return jsonify({"error": "unknown agent"}), 404
        scenario = yaml.safe_load(row["scenario_yaml"])
        frame = replay_cache.frame(run_id, as_of)
        agent_blob = next(
            (a for a in frame.agents if a.get("agent_id") == agent_id),
            None,
        )
        series = _series_from(conn, run_id, agent_id, MERCHANT_METRIC_KEYS, t_from, t_to)
        raw_listings = list((agent_blob or {}).get("store_listings") or [])
        products = _product_rows_by_id(
            run_id, [l.get("product_id") for l in raw_listings if l.get("product_id")], frame)
        per_listing_pnl = _per_listing_pnl(run_id, agent_id, as_of)
        merchant_product_scope = {
            str(pid): (
                (products.get(pid) or {}).get("name") or "",
                (products.get(pid) or {}).get("category") or "",
            )
            for pid in (l.get("product_id") for l in raw_listings)
            if pid
        }
        listings = []
        for listing in raw_listings:
            pid = listing.get("product_id")
            product = products.get(pid, {})
            sale_price = float(listing.get("sale_price") or 0.0)
            supplier_price = product.get("price")
            margin = (
                ((sale_price - float(supplier_price)) / sale_price)
                if sale_price and supplier_price is not None
                else 0.0
            )
            pnl = per_listing_pnl.get(pid, {})
            listings.append({
                "product_id": pid,
                "name": product.get("name") or "",
                "category": product.get("category") or "",
                "sale_price": listing.get("sale_price"),
                "supplier_price": supplier_price,
                "ref_price": product.get("ref_price"),
                "base_price": product.get("base_price"),
                "supplier_id": product.get("supplier_id"),
                "supplier_name": product.get("supplier_name"),
                "margin_ratio": round(margin, 4),
                "quantity": product.get("quantity"),
                "is_listed_by_supplier": product.get("is_listed_by_supplier"),
                "listed_at": listing.get("listed_at"),
                "promised_logistics_hours": listing.get("promised_logistics_hours"),
                "supplier_ship_hours": product.get("supplier_ship_hours"),
                "supplier_logistics_hours": product.get("logistics_hours"),
                "supplier_log_hour": _supplier_log_hour(
                    product.get("supplier_ship_hours"), product.get("logistics_hours")),
                "historical_avg_rating": product.get("historical_avg_rating"),
                "shop_rating": product.get("shop_rating"),
                "downstream_rating": _downstream_listing_rating(
                    scenario,
                    listing.get("rating_sum", 0.0),
                    listing.get("rating_count", 0.0),
                ),
                "return_buyer_rate": product.get("return_buyer_rate"),
                "supplier_age_years": product.get("supplier_age_years"),
                "cancel_rate": product.get("cancel_rate"),
                "refund_rate": product.get("refund_rate"),
                "only_refund_rate": product.get("only_refund_rate"),
                "timeout_rate": product.get("timeout_rate"),
                "bad_review_rate": product.get("bad_review_rate"),
                "price_change_rate": product.get("price_change_rate"),
                "supplier_delist_rate": product.get("supplier_delist_rate"),
                "elasticity": product.get("elasticity"),
                "cum_sales": listing.get("cum_sales", 0),
                "cum_gross_profit": pnl.get("cum_gross_profit", 0.0),
                "cum_net_profit": pnl.get("cum_net_profit", 0.0),
                "cum_fine": pnl.get("cum_fine", 0.0),
            })
        cash = (agent_blob or {}).get("cash")
        if not cash:
            cash_row = dbm.load_latest_cash_at(conn, run_id, agent_id, as_of)
            cash = cash_row.to_dict() if cash_row else {
                "balance": float((scenario.get("run") or {}).get("initial_cash", 0.0)),
                "deposit_pool": float((scenario.get("run") or {}).get("initial_deposit", 0.0)),
                "in_transit": 0.0,
                "receivable": 0.0,
                "cumulative_fine": 0.0,
            }
        return jsonify({
            "t": int(as_of),
            "agent_id": agent_id,
            "name": (agent_blob or {}).get("name") or agent.name,
            "is_alive": (agent_blob or {}).get("is_alive", agent.is_alive),
            "died_at_t": (agent_blob or {}).get("died_at_t", agent.died_at_t),
            "cash": cash,
            "listings": listings,
            "series": series,
            "recent_actions": dbm.load_dashboard_merchant_action_events(
                conn, run_id, agent_id, t_to=as_of),
            "shop_rating": _shop_rating_from_series(scenario, series),
            "listing_ops": _listing_ops(run_id, agent_id, int(as_of), scenario),
            "daily_sales_by_product": _daily_sales_by_product(
                run_id, agent_id, int(as_of), scenario, merchant_product_scope),
        })

    @bp.get("/runs/<run_id>/sections/supplier")
    def section_supplier(run_id):
        t_from = int(request.args["t_from"]) if "t_from" in request.args else None
        t_to = int(request.args["t_to"]) if "t_to" in request.args else None
        as_of = _as_of_arg()
        t_to = _default_t_to(t_to, as_of)
        limit = max(1, min(1000, int(request.args.get("limit", 500))))
        offset = max(0, int(request.args.get("offset", 0)))
        query = request.args.get("q", "")
        selected_product_id = request.args.get("selected_product_id", "").strip()
        sort_by = request.args.get("sort_by", "product_id")
        sort_dir = request.args.get("sort_dir", "asc")
        current_t = as_of if as_of is not None else _current_t_for_dashboard(run_id)
        if current_t is None:
            return jsonify({"error": "not found"}), 404
        frame = replay_cache.frame(run_id, as_of) if as_of is not None else None
        conn = _conn(run_id)
        # per-step series for supplier metrics
        series = _series_from(conn, run_id, "_global",
                              ["product_avail_count", "mean_supplier_price",
                               "total_supplier_qty", "delist_events",
                               "relist_events", "price_changes",
                               "supplier_timeout_events"],
                              t_from, t_to)
        products = dbm.list_dashboard_supplier_products(
            conn, run_id,
            query=query,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
            current_t=current_t,
        )
        products = _overlay_product_rows(products, frame)
        if (
            selected_product_id
            and all(str(p.get("product_id")) != selected_product_id for p in products)
        ):
            selected_product = dbm.get_dashboard_supplier_product(
                conn,
                run_id,
                selected_product_id,
                current_t=current_t,
            )
            if selected_product is not None:
                products.extend(_overlay_product_rows([selected_product], frame))
        product_count = dbm.count_dashboard_supplier_products(conn, run_id)
        filtered_count = (
            dbm.count_dashboard_supplier_products(conn, run_id, query=query)
            if query else product_count
        )
        # KPIs from latest metric value
        if as_of is not None:
            latest = _latest_metric_values(
                series, ["product_avail_count", "mean_supplier_price", "total_supplier_qty"])
        else:
            latest = {}
            for k in ("product_avail_count", "mean_supplier_price", "total_supplier_qty"):
                row = conn.execute(
                    "SELECT value FROM metrics WHERE run_id=? AND agent_id='_global' AND key=? ORDER BY t DESC LIMIT 1",
                    (run_id, k),
                ).fetchone()
                latest[k] = float(row["value"]) if row else 0.0
        return jsonify({
            "t": current_t,
            "kpis": latest,
            "products": products,
            "product_count": product_count,
            "filtered_count": filtered_count,
            "limit": limit,
            "offset": offset,
            "series": series,
        })

    @bp.get("/runs/<run_id>/sections/catalog_diagnostics")
    def section_catalog_diagnostics(run_id):
        if registry.get_run(run_id) is None:
            return jsonify({"error": "not found"}), 404
        sample_size = max(1, min(2000, int(request.args.get("sample_size", 1200))))
        top_n = max(1, min(100, int(request.args.get("top_n", 20))))
        artifact = catalog_diagnostics.read_catalog_diagnostics_artifact(
            registry.catalog_diagnostics_path(run_id)
        )
        if artifact is None:
            return _catalog_diagnostics_unavailable(run_id)
        return jsonify(catalog_diagnostics.diagnostics_from_artifact(
            artifact,
            sample_size=sample_size,
            top_n=top_n,
        ))

    @bp.get("/runs/<run_id>/sections/catalog_diagnostics/products/<product_id>")
    def section_catalog_diagnostics_product(run_id, product_id):
        row = dbm.get_run(_conn(run_id), run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        scenario = yaml.safe_load(row["scenario_yaml"])
        try:
            out = _catalog_product_diagnostics_as_of(
                run_id, product_id, None, scenario, initial_catalog=True
            )
        except CatalogDiagnosticsNotMaterialized:
            return _catalog_diagnostics_unavailable(run_id)
        if out is None:
            return jsonify({"error": "unknown product"}), 404
        return jsonify(out)

    @bp.get("/runs/<run_id>/agents/<agent_id>/sections/merchant/products/<product_id>")
    def section_merchant_product(run_id, agent_id, product_id):
        as_of = _as_of_arg()
        if as_of is not None:
            conn = _conn(run_id)
            row = dbm.get_run(conn, run_id)
            if not row:
                return jsonify({"error": "not found"}), 404
            agents = {a.agent_id for a in dbm.list_agents(conn, run_id)}
            if agent_id not in agents:
                return jsonify({"error": "unknown agent"}), 404
            scenario = yaml.safe_load(row["scenario_yaml"])
            try:
                out = _catalog_product_diagnostics_as_of(
                    run_id, product_id, int(as_of), scenario)
            except CatalogDiagnosticsNotMaterialized:
                return _catalog_diagnostics_unavailable(run_id)
            if out is None:
                return jsonify({"error": "unknown product"}), 404
            out = dict(out)
            out.update(
                _merchant_product_lifecycle_payload(
                    run_id,
                    agent_id,
                    product_id,
                    int(as_of),
                    scenario,
                )
            )
            return jsonify(out)
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        if agent_id not in {a.agent_id for a in dbm.list_agents(conn, run_id)}:
            return jsonify({"error": "unknown agent"}), 404
        scenario = yaml.safe_load(row["scenario_yaml"])
        try:
            out = _catalog_product_diagnostics_as_of(
                run_id, product_id, None, scenario
            )
        except CatalogDiagnosticsNotMaterialized:
            return _catalog_diagnostics_unavailable(run_id)
        if out is None:
            return jsonify({"error": "unknown product"}), 404
        current_t = _current_t_for_dashboard(run_id)
        if current_t is None:
            return jsonify({"error": "not found"}), 404
        out = dict(out)
        out.update(
            _merchant_product_lifecycle_payload(
                run_id,
                agent_id,
                product_id,
                int(current_t),
                scenario,
            )
        )
        return jsonify(out)

    @bp.get("/runs/<run_id>/agents/<agent_id>/sections/merchant/products/<product_id>/lifecycle")
    def section_merchant_product_lifecycle(run_id, agent_id, product_id):
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        agents = {a.agent_id for a in dbm.list_agents(conn, run_id)}
        if agent_id not in agents:
            return jsonify({"error": "unknown agent"}), 404
        product = conn.execute(
            "SELECT 1 FROM products WHERE run_id=? AND product_id=? LIMIT 1",
            (run_id, product_id),
        ).fetchone()
        if product is None:
            return jsonify({"error": "unknown product"}), 404
        as_of = _as_of_arg()
        current_t = as_of if as_of is not None else _current_t_for_dashboard(run_id)
        if current_t is None:
            return jsonify({"error": "not found"}), 404
        scenario = yaml.safe_load(row["scenario_yaml"])
        return jsonify(
            _merchant_product_lifecycle_payload(
                run_id,
                agent_id,
                product_id,
                int(current_t),
                scenario,
            )
        )

    @bp.get("/runs/<run_id>/sections/orders")
    def section_orders(run_id):
        t_from = int(request.args["t_from"]) if "t_from" in request.args else None
        t_to = int(request.args["t_to"]) if "t_to" in request.args else None
        as_of = _as_of_arg()
        t_to = _default_t_to(t_to, as_of)
        current_t = as_of if as_of is not None else _current_t_for_dashboard(run_id)
        if current_t is None:
            return jsonify({"error": "not found"}), 404
        status_names = [k[len("status_"):] for k in ORDER_STATUS_KEYS]
        conn = _conn(run_id)
        raw_status_cum_series = dbm.load_order_status_cum_series(conn, run_id, status_names, t_from, t_to)
        status_cum_series = {
            f"status_{status}": [[t, n] for (t, n) in rows]
            for status, rows in raw_status_cum_series.items()
        }
        if as_of is not None:
            status_counts = dbm.load_order_status_counts_as_of(conn, run_id, as_of)
            status_cum = dbm.load_order_status_cum_as_of(conn, run_id, as_of)
            orders_timeline = dbm.load_orders_with_log_as_of(conn, run_id, as_of, limit=200)
        else:
            row = conn.execute(
                "SELECT current_status, COUNT(*) AS n FROM orders WHERE run_id=? GROUP BY current_status",
                (run_id,),
            ).fetchall()
            status_counts = {r["current_status"]: r["n"] for r in row}
            cum_row = conn.execute(
                "SELECT status, COUNT(DISTINCT order_id) AS n FROM order_status WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
            status_cum = {r["status"]: r["n"] for r in cum_row}
            orders_timeline = dbm.load_orders_with_log(conn, run_id, limit=200)
        return jsonify({
            "t": current_t,
            "status_counts": status_counts,
            "status_cum": status_cum,
            "status_cum_series": status_cum_series,
            "orders_timeline": orders_timeline,
        })

    @bp.get("/runs/<run_id>/orders")
    def list_orders_with_log(run_id):
        limit = int(request.args.get("limit", 200))
        status = request.args.get("status") or None
        agent_id = request.args.get("agent_id") or None
        conn = _conn(run_id)
        rows = dbm.load_orders_with_log(conn, run_id, limit=limit, status=status, agent_id=agent_id)
        return jsonify({"orders": rows})

    @bp.get("/runs/<run_id>/agents/<agent_id>/sections/merchant")
    def section_merchant(run_id, agent_id):
        t_from = int(request.args["t_from"]) if "t_from" in request.args else None
        t_to = int(request.args["t_to"]) if "t_to" in request.args else None
        as_of = _as_of_arg()
        t_to = _default_t_to(t_to, as_of)
        if as_of is not None:
            return _merchant_section_as_of(run_id, agent_id, as_of, t_from, t_to)
        env = registry.get_env(run_id)
        if env is None:
            return _merchant_section_from_db(run_id, agent_id, t_from, t_to)

        # Snapshot env state under lock to avoid reading partially-updated state
        # while worker thread modifies it in env.step()
        with env.lock:
            if agent_id not in env.agents:
                return jsonify({"error": "unknown agent"}), 404
            st = env.agents[agent_id]
            current_t = env.t
            scenario = env.scenario

            # Copy listings to avoid mutation during iteration
            listings_snapshot = dict(st.listings)
            products_snapshot = {pid: env.products.get(pid) for pid in listings_snapshot}

            # Copy agent state fields we need
            agent_name = st.name
            agent_is_alive = st.is_alive
            agent_died_at_t = st.died_at_t
            agent_cash = st.cash.to_dict()
            agent_n_good = st.n_good
            agent_n_bad = st.n_bad
            agent_shop_rating_order_count = st.shop_rating_order_count
            agent_shop_rating_state = env._shop_rating_state(st)
            agent_public_review_state = env._public_review_state(st)
            agent_shop_rating_updated_through_step = (
                env._shop_rating_updated_through_step(st)
            )

        conn = _conn(run_id)
        series = _series_from(conn, run_id, agent_id, MERCHANT_METRIC_KEYS, t_from, t_to)
        per_listing_pnl = _per_listing_pnl(run_id, agent_id)

        # Build listings from snapshot
        listings = []
        for pid, l in listings_snapshot.items():
            p = products_snapshot.get(pid)
            sup_price = p.price if p else 0
            margin = ((l.sale_price - sup_price) / l.sale_price) if l.sale_price else 0
            pnl = per_listing_pnl.get(pid, {})
            listings.append({
                "product_id": pid,
                "name": p.name if p else "",
                "category": p.category if p else "",
                "sale_price": l.sale_price,
                "supplier_price": p.price if p else None,
                "ref_price": p.ref_price if p else None,
                "base_price": p.base_price if p else None,
                "supplier_id": p.supplier_id if p else None,
                "supplier_name": p.supplier_name if p else None,
                "margin_ratio": round(margin, 4),
                "quantity": effective_quantity(p, current_t) if p else None,
                "is_listed_by_supplier": p.is_listed_by_supplier if p else None,
                "listed_at": l.listed_at,
                "promised_logistics_hours": l.promised_logistics_hours,
                "supplier_ship_hours": p.supplier_ship_hours if p else None,
                "supplier_logistics_hours": p.logistics_hours if p else None,
                "supplier_log_hour": (
                    _supplier_log_hour(p.supplier_ship_hours, p.logistics_hours)
                    if p else None
                ),
                "historical_avg_rating": p.historical_avg_rating if p else None,
                "shop_rating": p.shop_rating if p else None,
                "downstream_rating": _downstream_listing_rating(
                    scenario, l.rating_sum, l.rating_count,
                ),
                "return_buyer_rate": p.return_buyer_rate if p else None,
                "supplier_age_years": p.supplier_age_years if p else None,
                "cancel_rate": p.cancel_rate if p else None,
                "refund_rate": p.refund_rate if p else None,
                "only_refund_rate": p.only_refund_rate if p else None,
                "timeout_rate": p.timeout_rate if p else None,
                "bad_review_rate": p.bad_review_rate if p else None,
                "price_change_rate": p.price_change_rate if p else None,
                "supplier_delist_rate": p.supplier_delist_rate if p else None,
                "elasticity": p.elasticity if p else None,
                "cum_sales": l.cum_sales,
                "cum_gross_profit": pnl.get("cum_gross_profit", 0.0),
                "cum_net_profit": pnl.get("cum_net_profit", 0.0),
                "cum_fine": pnl.get("cum_fine", 0.0),
            })

        # * The live headline uses the simulator's canonical quality/trust state.
        # * The time series in `series` remains the source for the rating chart.
        rating_cfg = scenario.get("shop_rating") or {}
        shop_rating = None
        if rating_cfg.get("enabled", False):
            raw_score = agent_shop_rating_state["score"]
            raw_stars = agent_shop_rating_state["stars"]
            shop_rating = {
                "enabled": True,
                "model": str(rating_cfg.get("model") or "beta_event_v1"),
                "score": (
                    round(float(raw_score), 4)
                    if raw_score is not None else None
                ),
                "stars": int(raw_stars) if raw_stars is not None else None,
                "quality_multiplier": round(
                    agent_shop_rating_state["quality_multiplier"], 4,
                ),
                "reputation_multiplier": round(
                    agent_shop_rating_state["reputation_multiplier"], 4,
                ),
                "demand_multiplier": round(
                    agent_shop_rating_state["demand_multiplier"], 4,
                ),
                "service_quality_score": round(
                    agent_shop_rating_state["service_quality_score"], 4,
                ),
                "service_quality_stars": int(
                    agent_shop_rating_state["service_quality_stars"]
                ),
                "service_quality_multiplier": round(
                    agent_shop_rating_state["service_quality_multiplier"], 4,
                ),
                "rating_available": bool(
                    agent_shop_rating_state["rating_available"]
                ),
                "demand_source": str(
                    agent_shop_rating_state["demand_source"]
                ),
                "bucket_thresholds": list(rating_cfg["bucket_thresholds"]),
                "star_multipliers": list(rating_cfg["star_multipliers"]),
            }
            if env._uses_order_outcome_rating():
                reputation_evidence_count = agent_shop_rating_order_count
                if env._uses_public_review_demand():
                    reputation_evidence_count = int(
                        agent_public_review_state["count"]
                        if agent_public_review_state is not None else 0
                    )
                shop_rating.update({
                    "rated_order_count": int(agent_shop_rating_order_count),
                    "qualified_transaction_count": int(
                        agent_shop_rating_order_count
                    ),
                    "reputation_evidence_count": int(
                        reputation_evidence_count
                    ),
                    "updated_through_step": agent_shop_rating_updated_through_step,
                })
                if agent_public_review_state is not None:
                    shop_rating["public_reviews"] = _public_review_payload(
                        agent_public_review_state,
                    )
            else:
                shop_rating.update({
                    "n_good_effective": round(agent_n_good, 2),
                    "n_bad_effective": round(agent_n_bad, 2),
                })
        return jsonify({
            "t": current_t,
            "agent_id": agent_id,
            "name": agent_name,
            "is_alive": agent_is_alive,
            "died_at_t": agent_died_at_t,
            "cash": agent_cash,
            "listings": listings,
            "series": series,
            "recent_actions": dbm.load_dashboard_merchant_action_events(conn, run_id, agent_id),
            "shop_rating": shop_rating,
            "listing_ops": _listing_ops(run_id, agent_id, current_t, scenario),
            "daily_sales_by_product": _daily_sales_by_product(
                run_id, agent_id, current_t, scenario),
        })

    @bp.get("/runs/<run_id>/agents/<agent_id>/playground/dashboard-data")
    def human_playground_dashboard_data(run_id, agent_id):
        """Safe Human UI projection of the existing Merchant Dashboard data."""
        if "as_of" in request.args:
            return jsonify({"error": "as_of is not supported by this endpoint"}), 400
        level = request.args.get("level")
        if level not in (None, "day", "week"):
            return jsonify({"error": "level must be 'day' or 'week'"}), 400
        try:
            t_from = (
                int(request.args["t_from"])
                if "t_from" in request.args
                else None
            )
            t_to = (
                int(request.args["t_to"])
                if "t_to" in request.args
                else None
            )
        except (TypeError, ValueError):
            return jsonify({"error": "t_from and t_to must be integers"}), 400
        merchant_response = section_merchant(run_id, agent_id)
        status = 200
        response = merchant_response
        if isinstance(merchant_response, tuple):
            response = merchant_response[0]
            if len(merchant_response) > 1:
                status = int(merchant_response[1])
        else:
            status = int(getattr(response, "status_code", 200))
        if status != 200:
            return merchant_response
        payload = response.get_json(silent=True) or {}
        return jsonify(_human_safe_merchant_payload(
            run_id,
            agent_id,
            payload,
            level=level,
            t_from=t_from,
            t_to=t_to,
        ))

    # ---------- New Run form ----------

    def _list_scenarios() -> list[dict]:
        """Enumerate scenario templates.

        Top-level scenarios/*.yaml are served as raw file content to preserve
        comments. Nested profile files are served resolved because /new_run
        submits textarea YAML directly rather than a scenario path.
        """
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scen_dir = os.path.join(here, "scenarios")
        entries: list[dict] = []
        if os.path.isdir(scen_dir):
            for fname in sorted(os.listdir(scen_dir)):
                if not fname.endswith(".yaml"):
                    continue
                with open(os.path.join(scen_dir, fname), "r") as f:
                    entries.append({"name": fname[:-5], "yaml": f.read()})
            for root, _dirs, files in os.walk(scen_dir):
                if root == scen_dir:
                    continue
                for fname in sorted(files):
                    if not fname.endswith(".yaml"):
                        continue
                    path = os.path.join(root, fname)
                    rel = os.path.relpath(path, scen_dir)
                    name = os.path.splitext(rel)[0].replace(os.sep, "/")
                    resolved = load_scenario(path)
                    entries.append({
                        "name": name,
                        "yaml": yaml.safe_dump(
                            resolved,
                            sort_keys=False,
                            allow_unicode=True,
                        ),
                    })
        entries.sort(key=lambda e: (0 if e["name"] == "default" else 1, e["name"]))
        return entries

    @bp.get("/scenarios")
    def list_scenarios():
        return jsonify(_list_scenarios())

    # ---------- dashboard setup ----------

    def _bool_value(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @bp.get("/experiments")
    def experiments_redirect():
        return redirect("/dashboard", code=302)

    @bp.get("/leaderboard")
    def leaderboard_page():
        return redirect("/dashboard", code=302)

    @bp.get("/new_run")
    def new_run_page():
        scenarios = _list_scenarios()
        default_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:4]}"
        default_scenario = load_default_scenario()
        private_db_path = private_real.resolve_dataset_path(
            (default_scenario.get("data") or {}).get("private_real_db_path")
        )
        data_sources = {
            "private_real_available": private_real.dataset_available(private_db_path),
            "private_real_db_path": private_db_path,
        }
        return render_template("new_run.html",
                               scenarios=scenarios,
                               default_name=default_name,
                               data_sources=data_sources,
                               human_model_presets=HUMAN_MODEL_PRESETS,
                               default_human_model=HUMAN_MODEL_PRESETS[0],
                               default_model=os.environ.get("MODEL_NAME", "qwen3.5-27b"),
                               model_pricing=REACT_MODEL_PRICING)

    def _pricing_from_form(field: str, preset: dict | None,
                           default: float = 0.0) -> float:
        million_name = f"cost_{field}_per_million"
        raw = (request.form.get(million_name) or "").strip()
        if raw:
            return float(raw)
        if preset and field in preset:
            return float(preset[field])
        return float(default)

    def _apply_new_run_options(scenario: dict) -> dict:
        scenario.setdefault("run", {})
        scenario.setdefault("agent", {})
        scenario.setdefault("data", {})

        if _bool_value(request.form.get("virtual_time_enabled"), False):
            start_date = (request.form.get("virtual_start_date") or "").strip()
            if not start_date:
                raise ValueError("virtual_start_date is required when virtual time is enabled")
            datetime.fromisoformat(start_date)
            scenario["run"]["virtual_time"] = {
                "enabled": True,
                "start_date": start_date,
            }
        else:
            scenario["run"].pop("virtual_time", None)

        data_anchor = (request.form.get("data_anchor_date") or "").strip()
        if data_anchor:
            datetime.fromisoformat(data_anchor)
            scenario["data"]["calendar_anchor_date"] = data_anchor

        model_name = (request.form.get("react_model") or "").strip()
        preset = REACT_MODEL_PRICING_BY_MODEL.get(model_name)
        pricing = {
            "input_per_million": _pricing_from_form("input", preset, 0.0),
            "output_per_million": _pricing_from_form("output", preset, 0.0),
            "cached_input_per_million": _pricing_from_form("cached_input", preset, 0.0),
        }
        scenario["agent"]["cost_pricing"] = pricing
        return pricing

    def _scenario_interval_ms(scenario: dict, default: int = 500) -> int:
        raw = (scenario.get("run") or {}).get("interval_ms", default)
        return max(0, int(raw))

    @bp.post("/new_run")
    def new_run_submit():
        name = (request.form.get("name") or "").strip() or None
        scenario_yaml = request.form.get("scenario_yaml") or ""
        bootstrap = request.form.get("bootstrap_agent") or "none"
        human_model = (request.form.get("human_model") or "").strip() or \
            HUMAN_MODEL_PRESETS[0]
        react_model = (request.form.get("react_model") or "").strip() or \
            os.environ.get("MODEL_NAME", "qwen3.5-27b")
        if bootstrap == "human" and len(human_model) > 80:
            return Response(
                "Human model name must be at most 80 characters",
                status=400,
                mimetype="text/plain",
            )
        try:
            scenario = yaml.safe_load(scenario_yaml)
            if not isinstance(scenario, dict) or "run" not in scenario:
                raise ValueError("YAML must define a `run:` section")
            _apply_new_run_options(scenario)
            interval_ms = _scenario_interval_ms(scenario)
        except Exception as e:  # noqa: BLE001
            return Response(f"YAML parse error: {e}", status=400, mimetype="text/plain")
        if bootstrap == "human":
            bootstrap_config = {"human_model": human_model}
        elif bootstrap == "rule_based":
            bootstrap_config = {
                "selection_mode": (
                    request.form.get("rule_based_mode") or "daily_report"
                ).strip(),
            }
        elif bootstrap in {"react_160k_compact_30k", "hermes"} and react_model:
            bootstrap_config = {"react_model": react_model}
        else:
            bootstrap_config = {}
        try:
            run_id = registry.create_run(
                scenario,
                name=(
                    name or f"run-{datetime.now().strftime('%Y%m%d-%H%M')}"
                ),
                bootstrap_agent=bootstrap,
                bootstrap_base_url=request.host_url.rstrip("/"),
                auto_start=True,
                interval_ms=interval_ms,
                bootstrap_config=bootstrap_config,
            )
        except (ValueError, PrivateRealDataError) as e:
            return Response(str(e), status=400, mimetype="text/plain")
        except Exception as e:  # noqa: BLE001
            return Response(str(e), status=500, mimetype="text/plain")
        if bootstrap == "human":
            return redirect(url_for(
                "dashboard.human_playground",
                run_id=run_id,
                agent_id="agent_0",
            ), code=302)
        return redirect(url_for("dashboard.dashboard", run_id=run_id), code=302)

    @bp.get("/runs/<run_id>/playground")
    def human_playground(run_id):
        conn = _conn(run_id)
        row = dbm.get_run(conn, run_id)
        if not row:
            abort(404)
        agent_id = request.args.get("agent_id") or "agent_0"
        agents = {a.agent_id: a for a in dbm.list_agents(conn, run_id)}
        agent = agents.get(agent_id)
        if agent is None:
            abort(404)
        scenario = yaml.safe_load(row["scenario_yaml"])
        run_cfg = scenario.get("run") or {}
        agent_cfg = scenario.get("agent") or {}
        bootstrap_cfg = dbm.get_bootstrap_config(conn, run_id)
        human_model = str(
            bootstrap_cfg.get("human_model") or "human-playground"
        ).strip()
        playground_config = {
            "runId": run_id,
            "agentId": agent_id,
            "agentName": agent.name,
            "modelName": human_model,
            "maxHookSeconds": int(run_cfg.get("max_hook_seconds", 1200)),
            "activationPeriod": int(agent_cfg.get("activation_period") or 1),
            "maxTurnsPerStep": int(agent_cfg.get("max_turns_per_step") or 0),
            "stepHours": int(run_cfg.get("step_hours") or 1),
            "toolsSchemaUrl": f"/runs/{run_id}/tools/schema",
            "registerUrl": f"/runs/{run_id}/agent/register",
            "observationUrl": f"/runs/{run_id}/agents/{agent_id}/observation",
            "actUrl": f"/runs/{run_id}/agents/{agent_id}/act",
            "statusUrl": f"/runs/{run_id}/status",
            "streamUrl": f"/runs/{run_id}/stream",
            "pauseUrl": f"/runs/{run_id}/pause",
            "resumeUrl": f"/runs/{run_id}/resume",
            "traceIndexUrl": (
                f"/runs/{run_id}/agents/{agent_id}/trace_index"
            ),
            "traceUrl": f"/runs/{run_id}/agents/{agent_id}/trace",
            "dashboardDataUrl": (
                f"/runs/{run_id}/agents/{agent_id}/playground/dashboard-data"
            ),
            "platformRules": _human_platform_rules_payload(
                conn, run_id, scenario,
            ),
        }
        return render_template(
            "human_playground.html",
            run=row,
            agent=agent,
            playground_config=playground_config,
        )

    # ---------- minimal HTML dashboard ----------

    def _leaderboard_summary(
        runs: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Build result and ranking rows without building chart histories."""
        terminal_rows = [
            row for row in runs
            if _is_cacheable_terminal(row)
        ]
        uncached_rows = [
            row for row in runs
            if not _is_cacheable_terminal(row)
        ]
        terminal_results, _ = _terminal_run_data(
            terminal_rows,
            include_charts=False,
        )
        run_results = terminal_results + build_run_results(
            registry,
            uncached_rows,
        )
        run_results.sort(
            key=lambda row: row.get("started_at") or "",
            reverse=True,
        )
        return run_results, build_leaderboard(run_results)

    def _leaderboard_payload():
        base_runs = registry.list_runs()
        has_uncacheable_terminal = any(
            row.get("status") in RUN_RESULT_STATUSES
            and not row.get("finished_at")
            for row in base_runs
        )
        structure = tuple(sorted(
            (str(r.get("run_id")), r.get("status"))
            for r in base_runs
        ))
        progress = tuple(sorted(
            (str(r.get("run_id")), r.get("current_t"))
            for r in base_runs
        ))
        cached = leaderboard_cache
        if (
            cached["payload"] is not None
            and not has_uncacheable_terminal
            and cached["structure"] == structure
            and cached["progress"] == progress
        ):
            return cached["payload"]
        if (
            cached["payload"] is not None
            and cached["structure"] == structure
            and time.monotonic() - cached["ts"] < _LEADERBOARD_MIN_INTERVAL
        ):
            return cached["payload"]
        with leaderboard_lock:
            # Re-check under the lock: another thread may have just rebuilt it.
            if (
                leaderboard_cache["payload"] is not None
                and not has_uncacheable_terminal
                and leaderboard_cache["structure"] == structure
                and leaderboard_cache["progress"] == progress
            ):
                return leaderboard_cache["payload"]
            if (
                leaderboard_cache["payload"] is not None
                and leaderboard_cache["structure"] == structure
                and time.monotonic() - leaderboard_cache["ts"]
                < _LEADERBOARD_MIN_INTERVAL
            ):
                return leaderboard_cache["payload"]
            runs = [decorate_run(r, registry.runs_root) for r in base_runs]
            terminal_rows = [
                row for row in runs
                if _is_cacheable_terminal(row)
            ]
            uncached_rows = [
                row for row in runs
                if not _is_cacheable_terminal(row)
            ]
            terminal_results, terminal_charts = _terminal_run_data(
                terminal_rows,
                include_charts=True,
            )
            uncached_results = build_run_results(registry, uncached_rows)
            uncached_charts = (
                build_charts(registry, uncached_results)
                if uncached_results else {}
            )
            run_results = terminal_results + uncached_results
            run_results.sort(
                key=lambda row: row.get("started_at") or "",
                reverse=True,
            )
            leaderboard_rows = build_leaderboard(run_results)
            leaderboard_charts = merge_chart_payloads(
                [terminal_charts, uncached_charts],
                run_results,
            )
            payload = {
                "run_results": run_results,
                "leaderboard": leaderboard_rows,
                "charts": leaderboard_charts,
            }
            leaderboard_cache["structure"] = structure
            leaderboard_cache["progress"] = progress
            leaderboard_cache["payload"] = payload
            leaderboard_cache["ts"] = time.monotonic()
            return payload

    @bp.get("/dashboard/leaderboard.json")
    def dashboard_leaderboard_json():
        return jsonify(_leaderboard_payload())

    def _experiment_group_run_options() -> list[dict]:
        options = []
        for raw in registry.list_runs():
            row = decorate_run(raw, registry.runs_root)
            horizon = row.get("horizon")
            step_hours = row.get("step_hours") or 1
            try:
                horizon_days = (
                    float(horizon) * float(step_hours) / 24
                    if horizon is not None else None
                )
            except (TypeError, ValueError):
                horizon_days = None
            options.append({
                "run_id": row.get("run_id"),
                "framework": row.get("framework") or "None",
                "framework_key": row.get("framework_key")
                or row.get("bootstrap_agent")
                or "none",
                "model": row.get("model") or "—",
                "status": row.get("status") or "unknown",
                "started_at": row.get("started_at"),
                "horizon": horizon,
                "step_hours": step_hours,
                "horizon_days": horizon_days,
            })
        return options

    @bp.get("/dashboard/experiment-run-options.json")
    def dashboard_experiment_run_options_json():
        return jsonify({"run_options": _experiment_group_run_options()})

    @bp.get("/dashboard/experiment-groups.json")
    def dashboard_experiment_groups_json():
        try:
            payload = experiment_group_store.load()
        except ExperimentGroupStoreError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({
            **payload,
            "framework_presets": FRAMEWORK_PRESETS,
            "model_presets": MODEL_PRESETS,
            "run_options": _experiment_group_run_options(),
        })

    @bp.put("/dashboard/experiment-groups.json")
    def update_dashboard_experiment_groups_json():
        try:
            payload = experiment_group_store.save(request.get_json(silent=True))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except ExperimentGroupStoreError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({
            **payload,
            "framework_presets": FRAMEWORK_PRESETS,
            "model_presets": MODEL_PRESETS,
            "run_options": _experiment_group_run_options(),
        })

    @bp.get("/dashboard")
    def dashboard():
        run_id = request.args.get("run_id")
        runs = []
        leaderboard_payload = None
        if not run_id:
            # Keep the initial HTML lightweight. The summary table only needs
            # final values; full chart histories load asynchronously from the
            # JSON endpoint after the page becomes usable.
            runs = [
                decorate_run(r, registry.runs_root)
                for r in registry.list_runs()
            ]
            run_results, leaderboard_rows = _leaderboard_summary(runs)
            summary_charts = merge_chart_payloads([], run_results)
            leaderboard_payload = {
                "run_results": run_results,
                "leaderboard": leaderboard_rows,
                "charts": summary_charts,
            }
            results_by_run = {
                row["run_id"]: row["result"]
                for row in run_results
            }
            empty_result = {
                "t": None, "elapsed_ms": None, "final_net_assets": None,
                "cum_gmv": None, "tokens": None, "usd": None,
            }
            for row in runs:
                row["result"] = results_by_run.get(row["run_id"], empty_result)
        ctx = {
            "runs": runs,
            "run_id": run_id,
            "data": None,
            "virtual_time": None,
            "run_results": (
                leaderboard_payload["run_results"]
                if leaderboard_payload else []
            ),
            "leaderboard": (
                leaderboard_payload["leaderboard"]
                if leaderboard_payload else []
            ),
            "leaderboard_charts": (
                leaderboard_payload["charts"]
                if leaderboard_payload else {}
            ),
            "leaderboard_payload": leaderboard_payload,
        }
        if run_id:
            conn = _conn(run_id)
            row = dbm.get_run(conn, run_id)
            ctx["data"] = {"run": row}
            scenario = yaml.safe_load(row.get("scenario_yaml") or "{}") if row else {}
            vt = (((scenario or {}).get("run") or {}).get("virtual_time") or {})
            if vt.get("enabled") and vt.get("start_date"):
                ctx["virtual_time"] = {
                    "enabled": True,
                    "startDate": str(vt["start_date"]),
                }
        return render_template("dashboard.html", **ctx)

    @bp.get("/")
    def index():
        return redirect("/dashboard", code=302)

    return bp
