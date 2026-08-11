"""Active-run registry + step hook orchestration.

One Environment per run_id. Each run has N agents (default 1: agent_0).
step() blocks until either end_of_step is called or max_hook_seconds elapses.
auto_step loops step() N times synchronously (legacy/scripted path).
For interactive Start/Pause/Resume/Stop control, see web.run_worker.RunWorker.

Bootstrap agents:
The env is an agent test harness, so /new_run lets you preload an agent. Options:
  * "none"       — no agent at start (user attaches their own external agent)
  * "human"      — open a browser playground for agent_0; no subprocess
  * "rule_based" — spawn baselines/rule_based.py with daily-report or
                   reproducible-random sourcing.
  * "auto_seed"  — legacy daily-report compatibility entry point.
  * "react_160k_compact_30k"
                 — spawn agents/react_160k_compact_30k.py. LLM tool-calling
                   loop with a 160k context window and 30k compaction target.
  * "hermes"     — spawn an external Hermes adapter repo. The adapter remains
                   a separate agent-side project and talks to this env via the
                   public observation + /act SDK protocol; launcher passes the
                   same 30-hop per-hook budget as react_160k and gives each run
                   an isolated HERMES_HOME under runs/<run_id>/agent/.
Tracked subprocesses live on RunRegistry.bootstrap_procs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import queue
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import yaml
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

MIN_PYTHON_VERSION = (3, 10)


def _ensure_supported_python(version_info=sys.version_info) -> None:
    major_minor = (int(version_info[0]), int(version_info[1]))
    if major_minor < MIN_PYTHON_VERSION:
        current = ".".join(str(part) for part in version_info[:3])
        required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
        raise RuntimeError(
            f"MerchantBench requires Python {required}+; current interpreter is {current}. "
            "Create the project virtualenv with Python 3.10 or newer."
        )


_ensure_supported_python()


class RunRuntimeUnavailableError(KeyError):
    """A run cannot provide an in-memory runtime to the caller."""


class RunRuntimeNotLoadedError(RunRuntimeUnavailableError):
    """The run exists on disk, but its in-memory runtime is not loaded."""


class RunPhaseClosedError(RunRuntimeUnavailableError):
    """A lifecycle mutation was requested after normal operation closed."""

    def __init__(self, run_id: str, phase: str):
        self.run_id = run_id
        self.phase = phase
        super().__init__(f"run {run_id} no longer accepts agents in phase {phase}")


class _RuntimeLifecycleEntry:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.users = 0


class _BoundedDaemonJobScheduler:
    """One daemon worker for heavyweight best-effort background artifacts."""

    def __init__(self, name: str):
        self._name = name
        self._queue: queue.Queue[
            tuple[tuple[str, str], Callable[[], None]]
        ] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: set[tuple[str, str]] = set()
        self._thread: Optional[threading.Thread] = None

    def submit(self, key: tuple[str, str], job: Callable[[], None]) -> bool:
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name=self._name,
                )
                self._thread.start()
        self._queue.put((key, job))
        return True

    def _run(self) -> None:
        while True:
            key, job = self._queue.get()
            try:
                job()
            except Exception:  # noqa: BLE001
                log.exception("unhandled background job failure for %s", key)
            finally:
                with self._lock:
                    self._pending.discard(key)
                self._queue.task_done()


_CATALOG_DIAGNOSTICS_SCHEDULER = _BoundedDaemonJobScheduler(
    "catalog-diagnostics-worker"
)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


from core.entities import Cash
from core.simulator import AgentState, Environment
from core.difficulty import apply_difficulty_rate
from core import supplier_scheduler
from data import private_real, synth
from storage import db as dbm
from storage import snapshot as snap
from web import catalog_diagnostics


class _BufferedCursor:
    """Small cursor-shaped object backed by rows materialized under the DB lock."""
    def __init__(self, rows, *, description=None, lastrowid=None, rowcount=-1):
        self._rows = list(rows)
        self._idx = 0
        self.description = description
        self._lastrowid = lastrowid
        self._rowcount = rowcount

    def fetchone(self):
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self):
        rows = self._rows[self._idx:]
        self._idx = len(self._rows)
        return rows

    def fetchmany(self, n=1):
        end = min(len(self._rows), self._idx + int(n))
        rows = self._rows[self._idx:end]
        self._idx = end
        return rows

    def __iter__(self):
        return iter(self.fetchall())

    @property
    def lastrowid(self):
        return self._lastrowid

    @property
    def rowcount(self):
        return self._rowcount


class LockedConnection:
    """Thread-safe wrapper around a sqlite3.Connection.

    sqlite3 connections allow cross-thread use (check_same_thread=False) but
    cursors are not thread-safe. For result-producing statements, this wrapper
    materializes rows while holding the shared lock so no raw cursor outlives
    its critical section.
    """
    def __init__(self, raw):
        self._raw = raw
        self._lock = threading.RLock()
    def execute(self, sql, *a, **kw):
        with self._lock:
            cursor = self._raw.execute(sql, *a, **kw)
            description = getattr(cursor, "description", None)
            lastrowid = getattr(cursor, "lastrowid", None)
            rowcount = getattr(cursor, "rowcount", -1)
            rows = cursor.fetchall() if description is not None else []
            return _BufferedCursor(
                rows,
                description=description,
                lastrowid=lastrowid,
                rowcount=rowcount,
            )

    def executemany(self, sql, *a, **kw):
        with self._lock:
            cursor = self._raw.executemany(sql, *a, **kw)
            return _BufferedCursor(
                [],
                description=getattr(cursor, "description", None),
                lastrowid=getattr(cursor, "lastrowid", None),
                rowcount=getattr(cursor, "rowcount", -1),
            )
    def executescript(self, sql, *a, **kw):
        with self._lock:
            cursor = self._raw.executescript(sql, *a, **kw)
            description = getattr(cursor, "description", None)
            rows = cursor.fetchall() if description is not None else []
            return _BufferedCursor(
                rows,
                description=description,
                lastrowid=getattr(cursor, "lastrowid", None),
                rowcount=getattr(cursor, "rowcount", -1),
            )
    def close(self):
        with self._lock:
            return self._raw.close()
    def commit(self):
        with self._lock:
            return self._raw.commit()
    def rollback(self):
        with self._lock:
            return self._raw.rollback()
    def __getattr__(self, name):
        # Only reached for attributes not explicitly defined above.
        # Wrap callables with the lock to prevent unsynchronised access.
        attr = getattr(self._raw, name)
        if callable(attr):
            def _locked(*a, **kw):
                with self._lock:
                    return attr(*a, **kw)
            return _locked
        return attr


BOOTSTRAP_AGENTS = (
    "none",
    "human",
    "rule_based",
    "auto_seed",
    "react_160k_compact_30k",
    "hermes",
)
RULE_BASED_SELECTION_MODES = ("daily_report", "random")
HERMES_MAX_HOPS_PER_STEP = 30
HERMES_HOME_DIRNAME = "hermes_home"
HERMES_WORKSPACE_DIRNAME = "hermes_workspace"
HERMES_PROFILE_MANIFEST_FILENAME = "hermes_profile_manifest.json"
HERMES_PROFILE_CONFIG_FILENAME = "config.yaml"
HERMES_CONTEXT_LENGTH = 262_144
HERMES_MAX_TOKENS = 16_384
HERMES_CONTEXT_FILE_MAX_CHARS = 80_000
# * Compression fires when estimated context / context_length exceeds this ratio.
HERMES_COMPRESSION_THRESHOLD = 0.85


class RunRegistry:
    def __init__(self, db_path: str, runs_root: str, run_db_filename: str = "state.db"):
        self.legacy_db_path = db_path
        self.runs_root = runs_root
        self.run_db_filename = run_db_filename
        self._conns: dict[str, LockedConnection] = {}
        self._read_conns: dict[str, LockedConnection] = {}  # read-only connections
        self._conn_lock = threading.RLock()
        self._conn_cond = threading.Condition(self._conn_lock)
        self._conn_active: dict[str, int] = {}
        self._deleting: set[str] = set()  # run_ids currently being deleted
        self.envs: dict[str, Environment] = {}
        self.workers: dict[str, "object"] = {}  # type: web.run_worker.RunWorker
        # tracked bootstrap agent subprocesses (rule_based / react_160k / hermes)
        self.bootstrap_procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self._runtime_locks: dict[str, _RuntimeLifecycleEntry] = {}
        self._run_auth: dict[str, dict[str, Any]] = {}
        self._shutdown_event = threading.Event()
        self._reconcile_orphaned_runs()
        self._resume_catalog_diagnostics_jobs()

    def _reconcile_orphaned_runs(self) -> None:
        """Make persisted lifecycle state truthful after a server restart."""
        if not os.path.isdir(self.runs_root):
            return
        stopped_at = _now_iso()
        recovered = 0
        for entry in os.scandir(self.runs_root):
            if not entry.is_dir() or not entry.name.startswith("run-"):
                continue
            path = os.path.join(entry.path, self.run_db_filename)
            try:
                if dbm.stop_orphaned_run(path, entry.name, stopped_at):
                    recovered += 1
            except (OSError, sqlite3.Error) as exc:
                log.warning("failed to reconcile orphaned run %s: %s", entry.name, exc)
        if recovered:
            log.info("marked %d orphaned run(s) as stopped", recovered)

    def shutdown(self) -> None:
        """Stop owned runtimes and subprocesses before the server exits."""
        self._shutdown_event.set()
        with self.lock:
            workers = list(self.workers.values())
        for worker in workers:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.exception("failed to stop worker during registry shutdown")
        for worker in workers:
            thread = getattr(worker, "_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        with self.lock:
            bootstrap_run_ids = list(self.bootstrap_procs)
        for run_id in bootstrap_run_ids:
            self._kill_bootstrap(run_id)
        with self._conn_lock:
            open_run_ids = set(self._conns) | set(self._read_conns)
        for run_id in open_run_ids:
            self.close_run_conn(run_id)

    @property
    def db_path(self) -> str:
        """Legacy shared DB path kept for migration tooling and old callers."""
        return self.legacy_db_path

    @property
    def conn(self):
        raise AttributeError("RunRegistry.conn was removed; use conn_for(run_id)")

    def _run_dir(self, run_id: str) -> str:
        runs_root_abs = os.path.realpath(self.runs_root)
        target = os.path.realpath(os.path.join(self.runs_root, run_id))
        if target == runs_root_abs or not target.startswith(runs_root_abs + os.sep):
            raise ValueError(f"invalid run_id path component: {run_id!r}")
        return target

    def run_db_path(self, run_id: str) -> str:
        return os.path.join(self._run_dir(run_id), self.run_db_filename)

    def catalog_diagnostics_path(self, run_id: str) -> str:
        return os.path.join(
            self._run_dir(run_id), catalog_diagnostics.CATALOG_DIAGNOSTICS_FILENAME
        )

    def catalog_diagnostics_status_path(self, run_id: str) -> str:
        return os.path.join(
            self._run_dir(run_id),
            catalog_diagnostics.CATALOG_DIAGNOSTICS_STATUS_FILENAME,
        )

    def _catalog_diagnostics_job_key(self, run_id: str) -> tuple[str, str]:
        return os.path.abspath(self.runs_root), f"{run_id}:{id(self)}"

    def _schedule_catalog_diagnostics(
        self,
        run_id: str,
        *,
        mark_pending: bool = False,
    ) -> bool:
        if self._shutdown_event.is_set() or not os.path.isdir(self._run_dir(run_id)):
            return False
        artifact_path = self.catalog_diagnostics_path(run_id)
        if (
            catalog_diagnostics.read_catalog_diagnostics_artifact(artifact_path)
            is not None
        ):
            catalog_diagnostics.write_catalog_diagnostics_status(
                self.catalog_diagnostics_status_path(run_id), "ready"
            )
            return False
        if mark_pending:
            catalog_diagnostics.write_catalog_diagnostics_status(
                self.catalog_diagnostics_status_path(run_id), "pending"
            )
        return _CATALOG_DIAGNOSTICS_SCHEDULER.submit(
            self._catalog_diagnostics_job_key(run_id),
            lambda: self._materialize_catalog_diagnostics(run_id),
        )

    def _resume_catalog_diagnostics_jobs(self) -> None:
        if not os.path.isdir(self.runs_root):
            return
        for entry in os.scandir(self.runs_root):
            if not entry.is_dir() or not entry.name.startswith("run-"):
                continue
            status = catalog_diagnostics.read_catalog_diagnostics_status(
                self.catalog_diagnostics_status_path(entry.name)
            )
            if (
                status
                and status.get("status") == "pending"
                and catalog_diagnostics.read_catalog_diagnostics_artifact(
                    self.catalog_diagnostics_path(entry.name)
                ) is None
            ):
                self._schedule_catalog_diagnostics(entry.name)

    def _materialize_catalog_diagnostics(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if self._shutdown_event.is_set() or not os.path.isdir(run_dir):
            return
        try:
            with self.lease_conn_for(run_id) as conn:
                row = dbm.get_run(conn, run_id)
                if not row:
                    return
                scenario = yaml.safe_load(row["scenario_yaml"])
                products = dbm.load_products(conn, run_id, initial=True)
                hourly_dist = dbm.load_hourly_dist(conn, run_id)
            run_meta = snap.read_meta(self.runs_root, run_id) or {
                "run_id": run_id,
                "started_at": row.get("started_at"),
            }
            diagnostics_source = SimpleNamespace(
                products={
                    product.product_id: product
                    for product in products
                },
                scenario=scenario,
                hourly_dist=hourly_dist,
            )
            artifact = catalog_diagnostics.build_catalog_diagnostics_artifact(
                diagnostics_source,
                run_meta=run_meta,
            )
            if self._shutdown_event.is_set() or not os.path.isdir(run_dir):
                return
            catalog_diagnostics.write_catalog_diagnostics_artifact(
                self.catalog_diagnostics_path(run_id), artifact
            )
            catalog_diagnostics.write_catalog_diagnostics_status(
                self.catalog_diagnostics_status_path(run_id), "ready"
            )
        except Exception as exc:  # noqa: BLE001
            if self._shutdown_event.is_set() or not os.path.isdir(run_dir):
                return
            try:
                catalog_diagnostics.write_catalog_diagnostics_status(
                    self.catalog_diagnostics_status_path(run_id),
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except OSError:
                return
            log.warning(
                "catalog diagnostics failed for %s",
                run_id,
                exc_info=True,
            )

    def auth_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        auth = self._run_auth.get(run_id)
        if auth is not None:
            return auth
        path = os.path.join(self._run_dir(run_id), "auth.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(data, dict):
            self._run_auth[run_id] = data
            return data
        return None

    def _write_auth_for_run(self, run_id: str, auth: dict[str, Any]) -> None:
        self._run_auth[run_id] = auth
        path = os.path.join(self._run_dir(run_id), "auth.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(auth, f, ensure_ascii=False, indent=2)

    def _ensure_agent_token(self, run_id: str, agent_id: str) -> str:
        auth = dict(self.auth_for_run(run_id) or {})
        agent_tokens = auth.get("agent_tokens")
        if not isinstance(agent_tokens, dict):
            agent_tokens = {}
            legacy_token = auth.get("agent_token")
            if legacy_token:
                agent_tokens["agent_0"] = legacy_token
        token = agent_tokens.get(agent_id)
        if not token:
            token = secrets.token_urlsafe(32)
            agent_tokens[agent_id] = token
        auth["agent_tokens"] = agent_tokens
        if agent_id == "agent_0":
            auth["agent_token"] = token
        self._write_auth_for_run(run_id, auth)
        return token

    @contextmanager
    def runtime_lifecycle_for(self, run_id: str):
        with self.lock:
            entry = self._runtime_locks.setdefault(run_id, _RuntimeLifecycleEntry())
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self.lock:
                entry.users -= 1
                if (
                    entry.users == 0
                    and self._runtime_locks.get(run_id) is entry
                ):
                    self._runtime_locks.pop(run_id, None)

    def conn_for(self, run_id: str, *, create: bool = False) -> LockedConnection:
        with self._conn_lock:
            if run_id in self._deleting:
                raise KeyError(f"run {run_id} is being deleted")
            conn = self._conns.get(run_id)
            if conn is not None:
                return conn
            path = self.run_db_path(run_id)
            if create:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            elif not os.path.exists(path):
                raise KeyError(f"unknown run {run_id}")
        conn = LockedConnection(dbm.open_db(path))
        with self._conn_lock:
            if run_id in self._deleting:
                conn.close()
                raise KeyError(f"run {run_id} is being deleted")
            existing = self._conns.get(run_id)
            if existing is not None:
                conn.close()
                return existing
            self._conns[run_id] = conn
            return conn

    @contextmanager
    def lease_conn_for(self, run_id: str, *, create: bool = False):
        conn = self._acquire_leased_conn(run_id, create=create)
        try:
            yield conn
        finally:
            self._release_conn_lease(run_id)

    @contextmanager
    def lease_env_for(self, run_id: str):
        with self.lease_conn_for(run_id):
            env = self.get_env(run_id)
            if env is None:
                raise RunRuntimeNotLoadedError(
                    f"run {run_id} runtime is not loaded"
                )
            yield env

    def _acquire_leased_conn(self, run_id: str, *, create: bool = False) -> LockedConnection:
        with self._conn_cond:
            if run_id in self._deleting:
                raise RunRuntimeUnavailableError(
                    f"run {run_id} is being deleted"
                )
            conn = self._conns.get(run_id)
            self._conn_active[run_id] = self._conn_active.get(run_id, 0) + 1
            if conn is not None:
                return conn
            path = self.run_db_path(run_id)
            if create:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            elif not os.path.exists(path):
                self._release_conn_lease(run_id)
                raise RunRuntimeUnavailableError(f"unknown run {run_id}")
        try:
            conn = LockedConnection(dbm.open_db(path))
        except Exception:
            self._release_conn_lease(run_id)
            raise
        with self._conn_lock:
            if run_id in self._deleting:
                conn.close()
                self._release_conn_lease(run_id)
                raise RunRuntimeUnavailableError(
                    f"run {run_id} is being deleted"
                )
            existing = self._conns.get(run_id)
            if existing is not None:
                conn.close()
                return existing
            self._conns[run_id] = conn
            return conn

    def _release_conn_lease(self, run_id: str) -> None:
        with self._conn_cond:
            active = self._conn_active.get(run_id, 0)
            if active <= 1:
                self._conn_active.pop(run_id, None)
                self._conn_cond.notify_all()
            else:
                self._conn_active[run_id] = active - 1

    @contextmanager
    def read_conn_for(self, run_id: str):
        # Try to get cached connection without holding lock during yield
        # Check writable connections first (they can handle reads too)
        with self._conn_lock:
            if run_id in self._deleting:
                raise KeyError(f"run {run_id} is being deleted")
            self._conn_active[run_id] = self._conn_active.get(run_id, 0) + 1
            cached = self._conns.get(run_id)
            if cached is not None:
                conn = cached
                is_cached = True
                is_writable = True
            else:
                # Check read-only connection cache
                cached = self._read_conns.get(run_id)
                if cached is not None:
                    conn = cached
                    is_cached = True
                    is_writable = False
                else:
                    is_cached = False
                    is_writable = False
                    conn = None

        if is_cached:
            # Yield cached connection without holding lock
            try:
                yield conn
            finally:
                self._release_conn_lease(run_id)
            return

        # Create new connection outside lock (read-only, lightweight)
        path = self.run_db_path(run_id)
        if not os.path.exists(path):
            self._release_conn_lease(run_id)
            raise KeyError(f"unknown run {run_id}")
        try:
            conn = LockedConnection(dbm.open_db_readonly(path))
        except sqlite3.OperationalError:
            # File may have been deleted between exists check and open (race
            # with delete_run). Surface as KeyError so callers treat it the
            # same as a missing run.
            self._release_conn_lease(run_id)
            raise KeyError(f"unknown run {run_id}")
        except Exception:
            # DatabaseError (corruption) or other errors: release the lease
            # before propagating to prevent permanent _conn_active leak that
            # would deadlock close_run_conn / delete_run.
            self._release_conn_lease(run_id)
            raise
        try:
            # Re-check _deleting after expensive operation to avoid race
            # and try to cache the connection
            with self._conn_lock:
                if run_id in self._deleting:
                    conn.close()
                    raise KeyError(f"run {run_id} is being deleted")
                # Check if a connection was added while we were creating ours
                existing = self._conns.get(run_id) or self._read_conns.get(run_id)
                if existing is not None:
                    conn.close()
                    conn = existing
                else:
                    # Cache our new read-only connection
                    self._read_conns[run_id] = conn
            yield conn
        finally:
            self._release_conn_lease(run_id)

    def close_run_conn(self, run_id: str) -> None:
        with self._conn_cond:
            while self._conn_active.get(run_id, 0) > 0:
                self._conn_cond.wait()
            conn = self._conns.pop(run_id, None)
            read_conn = self._read_conns.pop(run_id, None)
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                log.warning("error closing conn for run %s: %s", run_id, e)
        if read_conn is not None:
            try:
                read_conn.close()
            except Exception as e:
                log.warning("error closing read conn for run %s: %s", run_id, e)

    def get_run(self, run_id: str) -> Optional[dict]:
        try:
            with self.read_conn_for(run_id) as conn:
                return dbm.get_run(conn, run_id)
        except (KeyError, sqlite3.Error, OSError):
            return None

    def list_runs(self) -> list[dict]:
        import concurrent.futures

        rows: list[dict] = []
        if not os.path.isdir(self.runs_root):
            return rows

        # Collect run directories first
        run_entries = []
        for entry in os.scandir(self.runs_root):
            # Only directories matching run_id format (run-YYYYMMDD-...)
            if not entry.is_dir() or not entry.name.startswith("run-"):
                continue
            path = os.path.join(entry.path, self.run_db_filename)
            run_entries.append((path, entry.name))

        # Parallelize DB opens — get_run_lightweight is thread-safe (read-only, short-lived)
        # Use bounded thread pool to avoid overwhelming system with many runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_run = {
                executor.submit(dbm.get_run_lightweight, path, run_id): (path, run_id)
                for path, run_id in run_entries
            }
            for future in concurrent.futures.as_completed(future_to_run):
                path, run_id = future_to_run[future]
                try:
                    row = future.result()
                    if row:
                        rows.append(row)
                except Exception as e:
                    log.warning("failed to read run metadata for %s: %s", run_id, e)

        rows.sort(key=lambda r: (r.get("started_at") or "", r.get("run_id") or ""), reverse=True)
        return rows


    def get_env(self, run_id: str) -> Optional[Environment]:
        with self.lock:
            return self.envs.get(run_id)

    def release_terminal_runtime(self, run_id: str, *, worker=None) -> bool:
        """Drop only in-memory runtime state; persisted run artifacts remain."""
        released = False
        with self.runtime_lifecycle_for(run_id):
            if worker is None:
                row = self.get_run(run_id)
                if not row or row.get("status") not in ("finished", "stopped"):
                    return False
            with self.lock:
                current_worker = self.workers.get(run_id)
                if worker is not None and current_worker is not worker:
                    return False
                if worker is not None and getattr(worker, "state", None) not in (
                    "finished", "stopped"
                ):
                    return False
                if worker is not None:
                    self.workers.pop(run_id, None)
                elif current_worker is None or getattr(current_worker, "state", None) in (
                    "finished", "stopped"
                ):
                    self.workers.pop(run_id, None)
                else:
                    return False
                self.envs.pop(run_id, None)
            self._kill_bootstrap(run_id)
            self.close_run_conn(run_id)
            released = True
        return released

    def create_run(self, scenario: dict, master_seed: Optional[int] = None,
                   name: Optional[str] = None,
                   bootstrap_agent: str = "none",
                   bootstrap_base_url: Optional[str] = None,
                   auto_start: bool = True, interval_ms: int = 500,
                   bootstrap_config: Optional[dict] = None) -> str:
        if bootstrap_agent not in BOOTSTRAP_AGENTS:
            raise ValueError(f"bootstrap_agent must be one of {BOOTSTRAP_AGENTS}, got {bootstrap_agent!r}")
        bootstrap_config = dict(bootstrap_config or {})
        step_hours = int((scenario.get("run") or {}).get("step_hours", 1))
        if step_hours != 1:
            raise ValueError("run.step_hours must be 1 in this release")
        if master_seed is not None:
            scenario["run"]["master_seed"] = int(master_seed)
        if bootstrap_agent == "rule_based":
            selection_mode = str(
                bootstrap_config.get("selection_mode") or "daily_report"
            )
            if selection_mode not in RULE_BASED_SELECTION_MODES:
                raise ValueError(
                    "rule_based selection_mode must be one of "
                    f"{RULE_BASED_SELECTION_MODES}, got {selection_mode!r}"
                )
            try:
                selection_seed = int(
                    bootstrap_config.get(
                        "selection_seed",
                        scenario["run"]["master_seed"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rule_based selection_seed must be an integer"
                ) from exc
            bootstrap_config = {
                **bootstrap_config,
                "selection_mode": selection_mode,
                "selection_seed": selection_seed,
            }
        products, hourly_dist, data_meta = self._load_catalog_for_scenario(scenario)
        applied_difficulty_rate = apply_difficulty_rate(products, scenario)
        run_id = f"run-{_now_compact()}-{uuid.uuid4().hex[:6]}"
        run_name = name or f"run-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:4]}"
        agent_token = secrets.token_urlsafe(32)
        auth = {
            "agent_token": agent_token,
            "agent_tokens": {"agent_0": agent_token},
        }
        scenario_yaml = yaml.safe_dump(scenario, sort_keys=False)
        started_at = _now_iso()
        conn = self.conn_for(run_id, create=True)
        self._write_auth_for_run(run_id, auth)
        # Generate supplier events before opening the transaction: this is pure
        # CPU work and must not keep a write transaction open.
        initial_events = supplier_scheduler.initial_supplier_events(
            products,
            int(scenario["run"]["master_seed"]),
            start_t=0,
            horizon=int(scenario["run"]["horizon_steps"]),
        )
        # Write the entire catalog in one transaction. The connection is in
        # autocommit mode (isolation_level=None), so without this each
        # executemany would commit per-row, and on NFS every commit is an fsync
        # round-trip — turning creation into a 30-minute stall.
        initial_cash = float(scenario["run"]["initial_cash"])
        initial_deposit = float(scenario["run"].get("initial_deposit", 1000.0))
        cash = Cash(balance=initial_cash, deposit_pool=initial_deposit)
        conn.execute("BEGIN")
        try:
            dbm.insert_run(
                conn, run_id, run_name, scenario_yaml,
                int(scenario["run"]["master_seed"]),
                int(scenario["run"]["horizon_steps"]),
                int(scenario["run"]["step_hours"]),
                started_at,
                bootstrap_agent=bootstrap_agent,
                bootstrap_config=bootstrap_config,
            )
            dbm.insert_products(conn, run_id, products)
            dbm.write_hourly_dist(conn, run_id, hourly_dist)
            dbm.insert_supplier_events(conn, run_id, [e.to_row() for e in initial_events])
            # default agent_0 with separate balance + deposit
            dbm.insert_agent(conn, run_id, "agent_0", "Agent 0", started_at)
            dbm.write_cash_log(conn, run_id, "agent_0", -1, cash)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        agents = {"agent_0": AgentState(agent_id="agent_0", name="Agent 0", cash=cash)}
        env = Environment(run_id, conn, scenario, self.runs_root,
                          products, hourly_dist, agents)
        run_meta = {
            "run_id": run_id,
            "name": run_name,
            "scenario": scenario,
            "master_seed": int(scenario["run"]["master_seed"]),
            "started_at": started_at,
            "runtime_mode": "event_driven",
            "checkpoint_interval_steps": int(scenario["run"].get("checkpoint_interval_steps", 168)),
            "difficulty_rate": applied_difficulty_rate,
            **data_meta,
        }
        snap.write_meta(self.runs_root, run_id, run_meta)
        with self.lock:
            self.envs[run_id] = env

        self._schedule_catalog_diagnostics(run_id, mark_pending=True)

        with self.runtime_lifecycle_for(run_id):
            if auto_start:
                self._auto_start(run_id, interval_ms)
            # Spawn while holding the lifecycle lock so a terminal worker
            # cannot finish cleanup before its bootstrap process is registered.
            if bootstrap_agent == "rule_based":
                self._spawn_rule_based(
                    run_id,
                    bootstrap_base_url,
                    scenario,
                    selection_mode=bootstrap_config["selection_mode"],
                    selection_seed=bootstrap_config["selection_seed"],
                )
            elif bootstrap_agent == "auto_seed":
                self._spawn_auto_seed(run_id, bootstrap_base_url, scenario)
            elif bootstrap_agent == "react_160k_compact_30k":
                self._spawn_react_160k_compact_30k(
                    run_id, bootstrap_base_url,
                    model=bootstrap_config.get("react_model"),
                    max_steps=int(scenario["run"]["horizon_steps"]))
            elif bootstrap_agent == "hermes":
                self._spawn_hermes(
                    run_id, bootstrap_base_url,
                    model=bootstrap_config.get("react_model"),
                    max_steps=int(scenario["run"]["horizon_steps"]),
                    scenario=scenario)
        return run_id

    def _load_catalog_for_scenario(self, scenario: dict) -> tuple[list, dict, dict[str, str]]:
        data_cfg = scenario.setdefault("data", {})
        source = str(data_cfg.get("source", "synthetic") or "synthetic")
        if source not in {"synthetic", "private_real"}:
            raise ValueError("data.source must be one of: synthetic, private_real")
        data_cfg["source"] = source
        if source == "synthetic":
            products, hourly_dist = synth.generate(scenario)
            return products, hourly_dist, {"data_source": "synthetic"}

        db_path = private_real.resolve_dataset_path(data_cfg.get("private_real_db_path"))
        products, hourly_dist, meta = private_real.load_dataset(db_path)
        categories = sorted({p.category for p in products})
        suppliers = {p.supplier_id for p in products}
        data_cfg["private_real_db_path"] = db_path
        data_cfg["num_products"] = len(products)
        data_cfg["num_categories"] = len(categories)
        data_cfg["num_suppliers"] = len(suppliers)
        data_cfg["category_pool"] = categories
        return products, hourly_dist, {
            "data_source": "private_real",
            "dataset_id": meta.get("dataset_id", "private_real"),
            "dataset_rows": meta.get("dataset_rows", str(len(products))),
            "dataset_sha256": meta.get("dataset_sha256", ""),
        }

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

    def _agent_baselines_dir(self) -> str:
        """Locate the demo repo's agent/baselines/ directory.

        Layout (since the env↔agent split):
          <repo_root>/env/web/runner.py   <- this file
          <repo_root>/agent/baselines/    <- baselines live here

        `__file__` = env/web/runner.py, so going up three levels gets
        us to the repo root. When env and agent are deployed as
        separate repos the baselines dir won't exist; the dashboard's
        rule_based / react_160k bootstrap options will then be no-ops.
        """
        return os.path.join(self._repo_root(), "agent", "baselines")

    def _hermes_agent_root(self) -> str:
        """Locate the external Hermes adapter repo.

        Defaults to a sibling checkout named hermes-agent:
          <parent>/merchantbench-dev
          <parent>/hermes-agent
        Operators can override it with MERCHANTBENCH_HERMES_AGENT_ROOT.
        """
        configured = os.environ.get("MERCHANTBENCH_HERMES_AGENT_ROOT")
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        return os.path.join(os.path.dirname(self._repo_root()), "hermes-agent")

    def _hermes_python_executable(self, hermes_root: Optional[str] = None) -> str:
        configured = os.environ.get("MERCHANTBENCH_HERMES_PYTHON")
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        if hermes_root:
            # * POSIX and Windows venv layouts both need to resolve.
            for rel in (
                os.path.join(".venv", "bin", "python"),
                os.path.join("venv", "bin", "python"),
                os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join("venv", "Scripts", "python.exe"),
            ):
                candidate = os.path.join(hermes_root, rel)
                if os.path.isfile(candidate) and (
                    os.name == "nt" or os.access(candidate, os.X_OK)
                ):
                    return candidate
        return sys.executable

    def _hermes_profile_paths(self, run_id: str) -> dict[str, str]:
        agent_dir = os.path.join(self._run_dir(run_id), "agent")
        return {
            "agent_dir": agent_dir,
            "home": os.path.join(agent_dir, HERMES_HOME_DIRNAME),
            "workspace": os.path.join(agent_dir, HERMES_WORKSPACE_DIRNAME),
            "manifest": os.path.join(agent_dir, HERMES_PROFILE_MANIFEST_FILENAME),
        }

    def _git_output(self, cwd: str, *args: str) -> str:
        if not os.path.exists(os.path.join(cwd, ".git")):
            return ""
        try:
            return subprocess.check_output(
                ["git", "-C", cwd, *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    def _directory_sha256(self, root: str) -> str:
        if not os.path.isdir(root):
            return ""
        digest = hashlib.sha256()
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                if name in {".DS_Store"}:
                    continue
                path = os.path.join(current, name)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                digest.update(rel.encode("utf-8", "surrogateescape"))
                digest.update(b"\0")
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
        return digest.hexdigest()

    def _hermes_copy_ignore(self, _dir: str, names: list[str]) -> set[str]:
        ignored = {".DS_Store", "__pycache__"}
        return {name for name in names if name in ignored}

    def _write_hermes_profile_manifest(
        self,
        manifest_path: str,
        *,
        hermes_root: str,
        hermes_home: str,
        hermes_workspace: str,
        skills_source: str,
    ) -> None:
        data = {
            "created_at": _now_iso(),
            "hermes_root": hermes_root,
            "hermes_git_commit": self._git_output(hermes_root, "rev-parse", "HEAD"),
            "hermes_git_dirty": bool(self._git_output(hermes_root, "status", "--porcelain")),
            "hermes_home": hermes_home,
            "hermes_workspace": hermes_workspace,
            "skills_source": skills_source if os.path.isdir(skills_source) else "",
            "skills_sha256": self._directory_sha256(os.path.join(hermes_home, "skills")),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _hermes_profile_seed_path(self) -> Optional[str]:
        # * Opt-in seed for local OpenRouter defaults (Baidu FP8 + max reasoning).
        #   Set MERCHANTBENCH_HERMES_PROFILE_SEED to an absolute path, or to "1"/"true"
        #   to use scripts/hermes_openrouter_profile.snippet.yaml.
        configured = os.environ.get("MERCHANTBENCH_HERMES_PROFILE_SEED", "").strip()
        if not configured:
            return None
        if configured.lower() in {"1", "true", "yes", "on"}:
            return os.path.join(
                self._repo_root(),
                "scripts",
                "hermes_openrouter_profile.snippet.yaml",
            )
        return os.path.abspath(os.path.expanduser(configured))

    def _load_hermes_profile_seed(self) -> dict[str, Any]:
        path = self._hermes_profile_seed_path()
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Hermes profile seed must be a mapping: {path}")
        return loaded

    def _resolve_hermes_profile_settings(
        self,
        scenario: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Resolve run-local Hermes context/compression from scenario overrides.

        Scenario path: ``agent.hermes.context_length`` /
        ``agent.hermes.compression_threshold``. Missing keys keep the
        MerchantBench defaults used for profile seeding.
        """
        context_length = HERMES_CONTEXT_LENGTH
        compression_threshold = HERMES_COMPRESSION_THRESHOLD
        hermes_cfg: dict[str, Any] = {}
        if isinstance(scenario, dict):
            agent_cfg = scenario.get("agent")
            if isinstance(agent_cfg, dict):
                raw = agent_cfg.get("hermes")
                if isinstance(raw, dict):
                    hermes_cfg = raw
        if hermes_cfg.get("context_length") is not None:
            context_length = int(hermes_cfg["context_length"])
            if context_length <= 0:
                raise ValueError("agent.hermes.context_length must be positive")
        if hermes_cfg.get("compression_threshold") is not None:
            compression_threshold = float(hermes_cfg["compression_threshold"])
            if not 0.0 < compression_threshold <= 1.0:
                raise ValueError(
                    "agent.hermes.compression_threshold must be in (0, 1]"
                )
        return {
            "context_length": context_length,
            "compression_threshold": compression_threshold,
        }

    def _write_hermes_profile_config(
        self,
        hermes_home: str,
        *,
        scenario: Optional[dict] = None,
    ) -> str:
        config_path = os.path.join(hermes_home, HERMES_PROFILE_CONFIG_FILENAME)
        config: dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"Hermes profile config must be a mapping: {config_path}"
                    )
                config = loaded
        else:
            # * Only seed when MERCHANTBENCH_HERMES_PROFILE_SEED is set (keeps unit tests clean).
            config = self._load_hermes_profile_seed()

        settings = self._resolve_hermes_profile_settings(scenario)
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            model_config = {}
        model_config["context_length"] = int(settings["context_length"])
        model_config["max_tokens"] = HERMES_MAX_TOKENS
        config["model"] = model_config
        config["context_file_max_chars"] = HERMES_CONTEXT_FILE_MAX_CHARS

        compression_config = config.get("compression")
        if not isinstance(compression_config, dict):
            compression_config = {}
        compression_config["threshold"] = float(settings["compression_threshold"])
        compression_config["abort_on_summary_failure"] = False
        config["compression"] = compression_config

        auxiliary = config.get("auxiliary")
        if not isinstance(auxiliary, dict):
            auxiliary = {}
        compression = auxiliary.get("compression")
        if not isinstance(compression, dict):
            compression = {}
        # Make context compression part of the evaluated model's own behavior.
        # ``auto`` reuses the adapter's live main runtime, including the model,
        # endpoint, and credential passed at launch.  An explicit ``main``
        # provider instead re-resolves from config.yaml; MerchantBench intentionally
        # keeps provider credentials out of that run-local file, so that route
        # can lose the live API key and fail every summary request.
        compression["provider"] = "auto"
        for key in ("model", "base_url", "api_key", "context_length"):
            compression.pop(key, None)
        auxiliary["compression"] = compression
        config["auxiliary"] = auxiliary

        tmp_path = f"{config_path}.tmp-{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp_path, config_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return config_path

    def _prepare_hermes_run_profile(
        self,
        run_id: str,
        hermes_root: str,
        *,
        scenario: Optional[dict] = None,
    ) -> dict[str, str]:
        paths = self._hermes_profile_paths(run_id)
        os.makedirs(paths["agent_dir"], exist_ok=True)
        os.makedirs(paths["workspace"], exist_ok=True)
        if os.path.exists(paths["home"]):
            self._write_hermes_profile_config(paths["home"], scenario=scenario)
            return paths

        skills_source = os.path.join(hermes_root, "skills")
        tmp_home = f"{paths['home']}.tmp-{uuid.uuid4().hex[:8]}"
        shutil.rmtree(tmp_home, ignore_errors=True)
        try:
            os.makedirs(tmp_home, exist_ok=True)
            skills_target = os.path.join(tmp_home, "skills")
            if os.path.isdir(skills_source):
                shutil.copytree(
                    skills_source,
                    skills_target,
                    ignore=self._hermes_copy_ignore,
                )
            else:
                os.makedirs(skills_target, exist_ok=True)
            self._write_hermes_profile_config(tmp_home, scenario=scenario)
            os.replace(tmp_home, paths["home"])
            self._write_hermes_profile_manifest(
                paths["manifest"],
                hermes_root=hermes_root,
                hermes_home=paths["home"],
                hermes_workspace=paths["workspace"],
                skills_source=skills_source,
            )
        except Exception:
            shutil.rmtree(tmp_home, ignore_errors=True)
            raise
        return paths

    def _repo_dotenv_values(self) -> dict[str, str]:
        path = os.path.join(self._repo_root(), ".env")
        if not os.path.exists(path):
            return {}
        values: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return values
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[key] = value
        return values

    def _spawn_baseline(self, run_id: str, base_url: Optional[str],
                         script_name: str, label: str,
                         pass_environ: bool = False,
                         extra_args: Optional[list[str]] = None,
                         extra_env: Optional[dict[str, str]] = None) -> None:
        """Shared subprocess spawner for built-in baseline agents.

        script_name is the file under agent/baselines/ (e.g.
        "rule_based.py" or "react_160k_compact_30k.py"). Tracked on
        self.bootstrap_procs so delete_run can kill it.
        """
        if base_url is None:
            base_url = os.environ.get("MERCHANTBENCH_BASE_URL", "http://127.0.0.1:5000")
        baselines_dir = self._agent_baselines_dir()
        script = os.path.join(baselines_dir, script_name)
        if not os.path.exists(script):
            log.error("%s baseline script missing at %s "
                      "(env and agent split into separate deployments?)",
                      label, script)
            return
        repo_root = os.path.dirname(os.path.dirname(baselines_dir))
        cmd = [sys.executable, script,
                "--run-id", run_id,
                "--base-url", base_url,
                "--agent-id", "agent_0",
                "--quiet"]
        if extra_args:
            cmd.extend(extra_args)
        popen_kw = {"cwd": repo_root}
        auth = self.auth_for_run(run_id) or {}
        agent_tokens = auth.get("agent_tokens")
        agent_token = (
            agent_tokens.get("agent_0")
            if isinstance(agent_tokens, dict)
            else auth.get("agent_token")
        )
        if pass_environ or extra_env or agent_token:
            # ReAct baselines load .env via dotenv from cwd. Forward the
            # full parent environ so any OPENAI_* already exported by
            # the operator also reaches the child.
            env = os.environ.copy()
            if extra_env:
                env.update({str(k): str(v) for k, v in extra_env.items()})
            if agent_token:
                env["MERCHANTBENCH_AGENT_TOKEN"] = str(agent_token)
            popen_kw["env"] = env
        try:
            with self.lock:
                existing = self.bootstrap_procs.get(run_id)
                if existing is not None:
                    if existing.poll() is None:
                        log.info("bootstrap agent already running for %s (pid=%s); skip spawning %s",
                                 run_id, existing.pid, label)
                        return
                    self.bootstrap_procs.pop(run_id, None)
                proc = subprocess.Popen(cmd, **popen_kw)
                self.bootstrap_procs[run_id] = proc
        except OSError as e:
            log.error("failed to spawn %s agent: %s", label, e)
            return
        log.info("spawned %s agent for %s (pid=%s)", label, run_id, proc.pid)

    def _spawn_auto_seed(self, run_id: str, base_url: Optional[str],
                          scenario: dict) -> None:
        """Launch the legacy daily-report compatibility baseline."""
        n = int(scenario["run"].get(
            "rule_based_count",
            scenario["run"].get("auto_seed_count", 50),
        ))
        self._spawn_baseline(run_id, base_url, "auto_seed.py", "auto_seed",
                              extra_args=[
                                  "--seed-count", str(n),
                                  "--max-steps", str(int(scenario["run"]["horizon_steps"])),
                              ])

    def _spawn_rule_based(
        self,
        run_id: str,
        base_url: Optional[str],
        scenario: dict,
        *,
        selection_mode: str,
        selection_seed: int,
    ) -> None:
        n = int(scenario["run"].get(
            "rule_based_count",
            scenario["run"].get("auto_seed_count", 50),
        ))
        self._spawn_baseline(
            run_id,
            base_url,
            "rule_based.py",
            "rule_based",
            extra_args=[
                "--selection-mode", selection_mode,
                "--selection-seed", str(int(selection_seed)),
                "--seed-count", str(n),
                "--max-steps", str(int(scenario["run"]["horizon_steps"])),
            ],
        )

    def _spawn_react_160k_compact_30k(
        self,
        run_id: str,
        base_url: Optional[str],
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        extra_args: list[str] = []
        extra_env: dict[str, str] = {}
        if model:
            extra_args.extend(["--model", str(model)])
            extra_env["MODEL_NAME"] = str(model)
        if max_steps is not None:
            extra_args.extend(["--max-steps", str(int(max_steps))])
        extra_args.extend([
            "--context-window-tokens", "160000",
            "--compact-trigger-tokens", "160000",
            "--compact-keep-tokens", "30000",
        ])
        self._spawn_baseline(
            run_id,
            base_url,
            "react_160k_compact_30k.py",
            "react_160k_compact_30k",
            pass_environ=True,
            extra_args=extra_args,
            extra_env=extra_env,
        )

    def _spawn_hermes(
        self,
        run_id: str,
        base_url: Optional[str],
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        scenario: Optional[dict] = None,
    ) -> None:
        if base_url is None:
            base_url = os.environ.get("MERCHANTBENCH_BASE_URL", "http://127.0.0.1:5000")
        hermes_root = self._hermes_agent_root()
        adapter_entry = os.path.join(hermes_root, "merchantbench_adapter", "__main__.py")
        if not os.path.exists(adapter_entry):
            log.error(
                "Hermes adapter missing at %s; set MERCHANTBENCH_HERMES_AGENT_ROOT "
                "to the external hermes-agent checkout",
                adapter_entry,
            )
            return
        with self.lock:
            existing = self.bootstrap_procs.get(run_id)
            if existing is not None and existing.poll() is None:
                log.info(
                    "bootstrap agent already running for %s (pid=%s); "
                    "skip preparing and spawning hermes",
                    run_id, existing.pid,
                )
                return
        if scenario is None:
            env = self.envs.get(run_id)
            if env is not None:
                scenario = getattr(env, "scenario", None)
        repo_dotenv_values = self._repo_dotenv_values()
        try:
            profile_paths = self._prepare_hermes_run_profile(
                run_id,
                hermes_root,
                scenario=scenario,
            )
        except Exception as e:
            log.error("failed to prepare hermes run profile: %s", e)
            return

        cmd = [
            self._hermes_python_executable(hermes_root), "-m", "merchantbench_adapter",
            "--run-id", run_id,
            "--base-url", base_url,
            "--agent-id", "agent_0",
            "--max-hops-per-step", str(HERMES_MAX_HOPS_PER_STEP),
            "--quiet",
        ]
        extra_env: dict[str, str] = {}
        if model:
            cmd.extend(["--model", str(model)])
            extra_env["MODEL_NAME"] = str(model)
        if max_steps is not None:
            cmd.extend(["--max-steps", str(int(max_steps))])

        auth = self.auth_for_run(run_id) or {}
        agent_tokens = auth.get("agent_tokens")
        agent_token = (
            agent_tokens.get("agent_0")
            if isinstance(agent_tokens, dict)
            else auth.get("agent_token")
        )
        env = os.environ.copy()
        for key, value in repo_dotenv_values.items():
            env.setdefault(key, value)
        env.update(extra_env)
        if agent_token:
            env["MERCHANTBENCH_AGENT_TOKEN"] = str(agent_token)
        env["HERMES_HOME"] = profile_paths["home"]
        env["TERMINAL_CWD"] = profile_paths["workspace"]
        sdk_root = os.path.join(self._repo_root(), "agent")
        env["MERCHANTBENCH_AGENT_SDK_ROOT"] = sdk_root
        pythonpath_parts = [sdk_root]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        # * Prefer the portable Node from the personal Hermes install when present.
        #   System Node on this machine can be outside Hermes engines.node range.
        node_home = (
            env.get("MERCHANTBENCH_NODE_HOME")
            or repo_dotenv_values.get("MERCHANTBENCH_NODE_HOME")
            or ""
        ).strip()
        if node_home and os.path.isdir(node_home):
            env["PATH"] = node_home + os.pathsep + env.get("PATH", "")
            env["MERCHANTBENCH_NODE_HOME"] = node_home

        # * OpenRouter App attribution for the Hermes subprocess.
        env.setdefault(
            "OPENROUTER_X_TITLE",
            repo_dotenv_values.get("OPENROUTER_X_TITLE", "MerchantBench"),
        )
        env.setdefault(
            "OPENROUTER_HTTP_REFERER",
            repo_dotenv_values.get(
                "OPENROUTER_HTTP_REFERER",
                "https://github.com/Artemonim/merchantbench",
            ),
        )
        # Keep OpenAI-compatible clients authenticated when only OPENROUTER_* is set.
        if env.get("OPENROUTER_API_KEY") and not env.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]

        log_path = os.path.join(profile_paths["agent_dir"], "bootstrap_hermes.log")
        try:
            with self.lock:
                existing = self.bootstrap_procs.get(run_id)
                if existing is not None:
                    if existing.poll() is None:
                        log.info(
                            "bootstrap agent already running for %s (pid=%s); "
                            "skip spawning hermes",
                            run_id, existing.pid,
                        )
                        return
                    self.bootstrap_procs.pop(run_id, None)
                log_file = open(log_path, "ab", buffering=0)
                proc = subprocess.Popen(
                    cmd,
                    cwd=hermes_root,
                    env=env,
                    stdout=log_file,
                    stderr=log_file,
                )
                self.bootstrap_procs[run_id] = proc
        except OSError as e:
            log.error("failed to spawn hermes agent: %s", e)
            return
        log.info("spawned hermes agent for %s (pid=%s)", run_id, proc.pid)

    def _kill_bootstrap(self, run_id: str) -> None:
        with self.lock:
            proc = self.bootstrap_procs.pop(run_id, None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

    def _respawn_bootstrap(self, run_id: str, base_url: Optional[str] = None) -> None:
        """Re-spawn the bootstrap agent for a run based on its persisted type."""
        conn = self.conn_for(run_id)
        ba = dbm.get_bootstrap_agent(conn, run_id)
        if ba == "rule_based":
            cfg = dbm.get_bootstrap_config(conn, run_id)
            env = self._require(run_id)
            self._spawn_rule_based(
                run_id,
                base_url,
                env.scenario,
                selection_mode=str(
                    cfg.get("selection_mode") or "daily_report"
                ),
                selection_seed=int(
                    cfg.get(
                        "selection_seed",
                        env.scenario["run"]["master_seed"],
                    )
                ),
            )
        elif ba == "auto_seed":
            env = self._require(run_id)
            self._spawn_auto_seed(run_id, base_url, env.scenario)
        elif ba == "react_160k_compact_30k":
            cfg = dbm.get_bootstrap_config(conn, run_id)
            env = self._require(run_id)
            self._spawn_react_160k_compact_30k(
                run_id,
                base_url,
                model=cfg.get("react_model"),
                max_steps=int(env.scenario["run"]["horizon_steps"]),
            )
        elif ba == "hermes":
            cfg = dbm.get_bootstrap_config(conn, run_id)
            env = self._require(run_id)
            self._spawn_hermes(
                run_id,
                base_url,
                model=cfg.get("react_model"),
                max_steps=int(env.scenario["run"]["horizon_steps"]),
                scenario=env.scenario,
            )

    def _auto_start(self, run_id: str, interval_ms: int) -> None:
        """Spin up a RunWorker and start it. Imported lazily to avoid circular
        import (run_worker uses registry connection helpers)."""
        from web.run_worker import RunWorker
        with self.lock:
            w = self.workers.get(run_id)
            if w is None:
                w = RunWorker(self, run_id)
                self.workers[run_id] = w
        w.start(interval_ms)

    def add_agent(self, run_id: str, agent_id: str, name: str) -> dict:
        with self.lease_env_for(run_id) as env:
            with env.lock:
                active_count, _ = self._active_order_summary(run_id)
                phase = self._phase_for(env, active_count)
                if phase != "running":
                    raise RunPhaseClosedError(run_id, phase)
                initial = float(env.scenario["run"]["initial_cash"])
                deposit = float(env.scenario["run"].get("initial_deposit", 1000.0))
                dbm.insert_agent(
                    self.conn_for(run_id),
                    run_id,
                    agent_id,
                    name,
                    _now_iso(),
                )
                env.add_agent(agent_id, name, initial, deposit)
                token = self._ensure_agent_token(run_id, agent_id)
                return {
                    "agent_id": agent_id,
                    "name": name,
                    "agent_token": token,
                }

    def list_agents(self, run_id: str) -> list[dict]:
        with self.lease_conn_for(run_id) as conn:
            rows = dbm.list_agents(conn, run_id)
        return [{"agent_id": a.agent_id, "name": a.name, "created_at": a.created_at,
                 "is_alive": a.is_alive, "died_at_t": a.died_at_t} for a in rows]

    def active_order_status_counts(self, run_id: str) -> dict[str, int]:
        with self.lease_conn_for(run_id) as conn:
            return dbm.active_order_status_counts(conn, run_id)

    def _active_order_summary(self, run_id: str) -> tuple[int, dict[str, int]]:
        counts = self.active_order_status_counts(run_id)
        return sum(counts.values()), counts

    def _phase_for(self, env: Environment, active_count: Optional[int] = None) -> str:
        horizon = int(env.scenario["run"]["horizon_steps"])
        with env.lock:
            terminal_started = bool(
                env.finished or getattr(env, "drain_started_t", None) is not None
            )
            has_live_agents = any(
                agent.is_alive for agent in env.agents.values()
            )
            current_t = int(env.t)
        if not terminal_started and has_live_agents and current_t < horizon:
            return "running"
        if active_count is None:
            active_count, _ = self._active_order_summary(env.run_id)
        return "draining" if active_count > 0 else "finished"

    def _pending_hook_t(self, env: Environment) -> Optional[int]:
        """Return the durable hook awaiting finalization, if any."""
        try:
            row = dbm.get_run(self.conn_for(env.run_id), env.run_id) or {}
        except (KeyError, sqlite3.Error, OSError):
            return None
        value = row.get("pending_hook_t")
        return None if value is None else int(value)

    def _wake_agents_for_terminal_phase(self, env: Environment) -> None:
        with env.lock:
            with env.hook_cond:
                env.finished = True
                env.hook_open = False
                env.hook_cond.notify_all()

    def _drain_safety_max_steps(self, env: Environment) -> int:
        run_cfg = env.scenario["run"]
        step_hours = max(1, int(run_cfg.get("step_hours", 1)))
        explicit = run_cfg.get("drain_safety_max_steps")
        if explicit is not None:
            return max(1, int(explicit))
        products = list(env.products.values())
        def hours_to_steps(hours) -> int:
            return int(math.ceil(max(0.0, float(hours)) / step_hours))

        max_ship_steps = max((hours_to_steps(p.ship_hours) for p in products), default=0)
        max_logistics_steps = max((hours_to_steps(p.logistics_hours) for p in products), default=0)
        timeout_hi_steps = hours_to_steps(
            (env.scenario.get("supplier_ranges") or {})
            .get("timeout_delay_hours", [0, 0])[-1]
        )
        normal_delay_steps = max(
            1,
            hours_to_steps(env.scenario["settlement"]["normal_delay_hours"]),
        )
        refund_tail_steps = hours_to_steps(95)
        estimated_lifecycle_cap = (
            max_ship_steps + timeout_hi_steps + max_logistics_steps
            + normal_delay_steps + refund_tail_steps
        )
        day_steps = int(math.ceil(24 / step_hours))
        return max(30 * day_steps, estimated_lifecycle_cap + day_steps)

    def _mark_finished(self, env: Environment) -> None:
        with env.lock:
            env.publish_terminal_ratings()
        self._wake_agents_for_terminal_phase(env)
        try:
            conn = self.conn_for(env.run_id)
            dbm.mark_run_terminal(conn, env.run_id, "finished", _now_iso())
            self._persist_run_summary(env.run_id, conn=conn)
        except (KeyError, sqlite3.Error) as e:
            log.warning("failed to mark run %s as finished: %s", env.run_id, e)

    def _persist_run_summary(self, run_id: str, *, conn) -> None:
        """Write agent/run_summary.json with cost, wall time, and projections."""
        from storage import agent_log
        from web.leaderboard import compute_run_result

        try:
            row = dbm.get_run(conn, run_id) or {}
            result = compute_run_result(self, run_id, conn=conn, row=row) or {}
            cost = agent_log.read_cost(self.runs_root, run_id)
            total = dict(cost.get("total") or {})
            by_step = cost.get("by_step") or {}
            step_hours = 1.0
            scenario_horizon = 0
            activation_period = 12
            try:
                env = self._require(run_id)
                run_cfg = env.scenario.get("run") or {}
                agent_cfg = env.scenario.get("agent") or {}
                step_hours = float(run_cfg.get("step_hours") or 1.0)
                scenario_horizon = int(run_cfg.get("horizon_steps") or 0)
                activation_period = int(agent_cfg.get("activation_period") or 12)
            except KeyError:
                env = None
            # * Prefer configured operating horizon over drain tail current_t.
            horizon_steps = scenario_horizon or int(
                row.get("current_t") or result.get("t") or 0
            )
            sim_days = (horizon_steps * step_hours) / 24.0 if horizon_steps else 0.0
            elapsed_ms = result.get("elapsed_ms")
            usd = float(total.get("usd") or result.get("usd") or 0.0)
            usd_per_day = (usd / sim_days) if sim_days > 0 else 0.0
            wall_ms_per_day = (
                (float(elapsed_ms) / sim_days) if sim_days > 0 and elapsed_ms else 0.0
            )
            windows = len(by_step)
            manifest_path = os.path.join(
                agent_log.agent_dir(self.runs_root, run_id),
                HERMES_PROFILE_MANIFEST_FILENAME,
            )
            hermes_meta = None
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        hermes_meta = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("failed reading hermes manifest for %s: %s", run_id, exc)
            summary = {
                "run_id": run_id,
                "written_at": _now_iso(),
                "status": row.get("status") or "finished",
                "bootstrap_agent": row.get("bootstrap_agent"),
                "master_seed": row.get("master_seed"),
                "horizon_steps": horizon_steps,
                "sim_days": round(sim_days, 4),
                "activation_period": activation_period,
                "activation_windows": windows,
                "result": result,
                "cost_total": total,
                "cost_by_step": {
                    str(step): {
                        "turns": payload.get("turns"),
                        "usd": payload.get("usd"),
                        "total": payload.get("total"),
                        "env_step_ms": payload.get("env_step_ms"),
                        "input": payload.get("input"),
                        "output": payload.get("output"),
                    }
                    for step, payload in by_step.items()
                    if isinstance(payload, dict)
                },
                "rates": {
                    "usd_per_sim_day": round(usd_per_day, 6),
                    "wall_ms_per_sim_day": round(wall_ms_per_day, 3),
                    "usd_per_window": round((usd / windows), 6) if windows else 0.0,
                    "wall_ms_per_window": (
                        round(float(elapsed_ms) / windows, 3)
                        if windows and elapsed_ms
                        else 0.0
                    ),
                },
                "projections": agent_log.build_horizon_projections(
                    usd_per_sim_day=usd_per_day,
                    wall_ms_per_sim_day=wall_ms_per_day,
                ),
                "hermes": hermes_meta,
                "caveat": (
                    "Projections are linear in simulated days from this run's "
                    "measured rates; first-wakeup pathology, cache, and "
                    "compaction make longer horizons non-linear."
                ),
            }
            path = agent_log.write_run_summary(self.runs_root, run_id, summary)
            log.info("wrote run summary for %s -> %s", run_id, path)
        except Exception as exc:  # noqa: BLE001
            # * Summary is diagnostics only; never fail terminalization on it.
            log.warning("failed to persist run_summary for %s: %s", run_id, exc)

    def _mark_draining(self, env: Environment) -> None:
        if not hasattr(env, "drain_started_t") or env.drain_started_t is None:
            env.drain_started_t = env.t
        self._wake_agents_for_terminal_phase(env)
        try:
            dbm.update_run_status(self.conn_for(env.run_id), env.run_id, "draining")
        except (KeyError, sqlite3.Error) as e:
            log.warning("failed to mark run %s as draining: %s", env.run_id, e)
        self._kill_bootstrap(env.run_id)

    def _step_payload(self, result, phase: str, active_count: int,
                      active_counts: dict[str, int], current_t: int) -> dict:
        return {
            "t": result.t if result is not None else current_t,
            "new_orders": result.new_orders if result is not None else 0,
            "state_transitions": result.state_transitions if result is not None else 0,
            "events": result.events if result is not None else 0,
            "phase": phase,
            "active_orders_remaining": active_count,
            "active_order_status_counts": active_counts,
        }

    def step(self, run_id: str) -> dict:
        env = None
        try:
            with self.lease_env_for(run_id) as env:
                result = self._step_with_env(env)
        except Exception:
            if env is not None:
                try:
                    conn = self.conn_for(run_id)
                    row = dbm.get_run(conn, run_id)
                    if not row or row.get("status") not in ("finished",):
                        dbm.mark_run_terminal(conn, run_id, "stopped", _now_iso())
                except (KeyError, sqlite3.Error, OSError) as cleanup_error:
                    log.warning(
                        "failed to persist synchronous step error for %s: %s",
                        run_id,
                        cleanup_error,
                    )
                self.release_terminal_runtime(run_id)
            raise
        if result.get("phase") == "finished" or result.get("error"):
            self.release_terminal_runtime(run_id)
        return result

    def _step_with_env(self, env: Environment) -> dict:
        run_id = env.run_id
        max_secs = float(env.scenario["run"]["max_hook_seconds"])

        def blocker():
            env.hook_event.wait(timeout=max_secs)

        active_before, counts_before = self._active_order_summary(run_id)
        phase_before = self._phase_for(env, active_before)
        pending_hook_t = self._pending_hook_t(env)
        if phase_before == "finished" and pending_hook_t is None:
            self._mark_finished(env)
            return self._step_payload(None, "finished", active_before,
                                      counts_before, env.t)

        drain = phase_before == "draining"
        if drain:
            self._mark_draining(env)
            drain_started_t = getattr(env, "drain_started_t", env.t)
            if env.t - drain_started_t > self._drain_safety_max_steps(env):
                conn = self.conn_for(run_id)
                dbm.mark_run_terminal(conn, run_id, "stopped", _now_iso())
                return {
                    **self._step_payload(None, "draining", active_before,
                                         counts_before, env.t),
                    "error": "drain_safety_max_steps_exceeded",
                }

        result = env.step(hook_blocker=blocker, drain=drain)
        active_after, counts_after = self._active_order_summary(run_id)
        phase_after = self._phase_for(env, active_after)
        if phase_after == "draining":
            self._mark_draining(env)
        elif phase_after == "finished":
            self._mark_finished(env)
        return self._step_payload(result, phase_after, active_after,
                                  counts_after, env.t)

    def auto_step(self, run_id: str, n: int) -> dict:
        out = []
        for _ in range(n):
            step_result = self.step(run_id)
            out.append(step_result)
            if step_result.get("phase") == "finished" or step_result.get("error"):
                break
        env = self.get_env(run_id)
        current_t = env.t if env is not None else (out[-1]["t"] if out else 0)
        return {"steps": out, "current_t": current_t}

    def delete_run(self, run_id: str) -> dict:
        """Stop the worker, drop in-memory state, and remove the run directory.
        Safe to call on a missing run — returns {'deleted': False} in that case."""
        target = self._run_dir(run_id)
        if not os.path.isdir(target):
            return {"deleted": False, "reason": "not found"}

        # Check if this looks like a run directory. We accept it if state.db exists
        # OR the agent/ subdirectory exists. This allows deletion of corrupted or
        # partially-deleted run directories while protecting non-run directories.
        db_path = self.run_db_path(run_id)
        agent_dir = os.path.join(target, "agent")
        if not os.path.exists(db_path) and not os.path.isdir(agent_dir):
            return {"deleted": False, "reason": "not found"}

        # If state.db exists, try to read metadata. Corruption is logged but doesn't block deletion.
        if os.path.exists(db_path):
            try:
                dbm.get_run_lightweight(db_path, run_id)
            except sqlite3.DatabaseError as e:
                log.warning("corrupted run DB %s, proceeding with deletion: %s", db_path, e)

        # Mark as deleting FIRST to prevent race with concurrent operations
        with self._conn_lock:
            if run_id in self._deleting:
                return {"deleted": False, "reason": "already deleting", "run_id": run_id}
            self._deleting.add(run_id)

        try:
            with self.lock:
                worker = self.workers.get(run_id)

            # Stop worker BEFORE closing connection to prevent "closed database" errors
            worker_stuck = False
            if worker is not None:
                try:
                    worker.stop(persist=False)
                    thread = getattr(worker, "_thread", None)
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=2)
                    if thread is not None and thread.is_alive():
                        log.warning("worker for run %s did not stop before timeout, killing bootstrap", run_id)
                        self._kill_bootstrap(run_id)
                        worker_stuck = True
                except Exception as e:  # noqa: BLE001
                    log.exception("failed to stop worker before deleting %s", run_id)
                    self._kill_bootstrap(run_id)
                    worker_stuck = True

            if worker_stuck:
                return {
                    "deleted": False,
                    "reason": "worker is still stopping",
                    "status": "busy",
                    "run_id": run_id,
                }

            with self.lock:
                self.workers.pop(run_id, None)
                self.envs.pop(run_id, None)

            # Kill any bootstrap subprocess tied to this run
            self._kill_bootstrap(run_id)

            # Checkpoint WAL before closing to ensure durability
            # New leases are blocked by _deleting; wait for existing leases
            # before touching the DB or removing the run directory.
            with self._conn_cond:
                while self._conn_active.get(run_id, 0) > 0:
                    self._conn_cond.wait()

            temp_conn = None
            conn = self._conns.get(run_id)
            if conn is None and os.path.exists(db_path):
                try:
                    temp_conn = LockedConnection(dbm.open_db(db_path))
                    conn = temp_conn
                except sqlite3.Error as e:
                    log.warning("failed to open DB before deleting %s: %s", run_id, e)
                    conn = None
            if conn is not None:
                try:
                    dbm.update_run_status(conn, run_id, "stopped")
                    # Force WAL checkpoint before closing
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as e:
                    log.warning("WAL checkpoint failed for %s: %s", run_id, e)
                finally:
                    if temp_conn is not None:
                        try:
                            temp_conn.close()
                        except Exception as e:
                            log.warning("error closing temporary conn for run %s: %s", run_id, e)

            # Now safe to close connection
            self.close_run_conn(run_id)

            # Remove directory
            if os.path.isdir(target):
                try:
                    shutil.rmtree(target)
                except OSError as e:
                    log.warning("failed to remove run directory %s: %s", target, e)
                    return {
                        "deleted": False,
                        "reason": f"failed to remove run directory: {e}",
                        "status": "error",
                        "run_id": run_id,
                    }

            return {"deleted": True, "run_id": run_id}
        finally:
            with self._conn_lock:
                self._deleting.discard(run_id)

    def _require(self, run_id: str) -> Environment:
        with self.runtime_lifecycle_for(run_id):
            env = self.get_env(run_id)
            if env is None:
                env = self._rehydrate(run_id)
                if env is None:
                    raise KeyError(f"unknown run {run_id}")
            return env

    def _rehydrate(self, run_id: str) -> Optional[Environment]:
        try:
            conn = self.conn_for(run_id)
        except KeyError:
            return None
        try:
            row = dbm.get_run(conn, run_id)
            if not row:
                return None
            scenario = yaml.safe_load(row["scenario_yaml"])
            products = dbm.load_products(conn, run_id)
            hourly_dist = dbm.load_hourly_dist(conn, run_id)
            initial = float(scenario["run"]["initial_cash"])
            agent_rows = dbm.list_agents(conn, run_id)
            deposit = float(scenario["run"].get("initial_deposit", 1000.0))
            agents: dict[str, AgentState] = {}
            for a in agent_rows:
                cash = dbm.load_latest_cash(conn, run_id, a.agent_id) or \
                    Cash(balance=initial, deposit_pool=deposit)
                st = AgentState(agent_id=a.agent_id, name=a.name, cash=cash,
                                is_alive=a.is_alive, died_at_t=a.died_at_t)
                agents[a.agent_id] = st
        except (sqlite3.Error, OSError) as e:
            log.warning("failed to rehydrate run %s: %s", run_id, e)
            return None
        env = Environment(run_id, conn, scenario, self.runs_root,
                          products, hourly_dist, agents)
        env.t = int(row["current_t"])
        env.reload_all_listings()
        # Rebuild the configured rating model after listings are loaded.  V1
        # replays events; V2 replays terminal downstream orders through the
        # last completed virtual day.  A pending hook means transition env.t
        # is already durable even though current_t has not advanced yet, so
        # rating replay must include that completed transition as t + 1.
        hook_t = env.t
        pending_hook_t = row.get("pending_hook_t")
        if pending_hook_t is not None and int(pending_hook_t) == hook_t:
            env.t = hook_t + 1
        try:
            env.restore_rating_state(
                include_terminal_partial_day=(row.get("status") == "finished"),
            )
        finally:
            env.t = hook_t
        with self.lock:
            existing = self.envs.get(run_id)
            if existing is not None:
                return existing
            self.envs[run_id] = env
            return env


def load_default_scenario() -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_scenario(os.path.join(here, "scenarios", "default.yaml"))


def load_scenario(path: str) -> dict:
    return _load_scenario_file(os.path.abspath(path), stack=[])


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_scenario_file(path: str, *, stack: list[str]) -> dict:
    real_path = os.path.realpath(path)
    if real_path in stack:
        chain = " -> ".join([*stack, real_path])
        raise ValueError(f"scenario extends cycle: {chain}")

    with open(real_path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f) or {}
    if not isinstance(scenario, dict):
        raise ValueError(f"scenario YAML must be a mapping: {real_path}")

    extends = scenario.pop("extends", None)
    if extends is None:
        return scenario
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError(f"scenario extends must be a non-empty string: {real_path}")

    parent_path = extends
    if not os.path.isabs(parent_path):
        parent_path = os.path.join(os.path.dirname(real_path), parent_path)
    parent = _load_scenario_file(parent_path, stack=[*stack, real_path])
    return _deep_merge_dicts(parent, scenario)
