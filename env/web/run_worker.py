"""Per-run background worker for Start/Pause/Resume/Stop control.

Wraps env.step() in a thread loop. Publishes tick events to subscribers via
thread-safe queues, used by the SSE endpoint /runs/<rid>/stream.

State machine:
  pending  -> start() -> running
  running  -> pause() -> paused
  paused   -> resume() -> running
  *        -> stop()  -> stopped

In addition to the step loop, this module hosts the subscriber API
for SSE (Server-Sent Events) used by the dashboard's live stream.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class RunWorker:
    def __init__(self, registry, run_id: str):
        self.registry = registry
        self.run_id = run_id
        # If the run already terminated in a previous process (server restart,
        # GC, etc.) the DB has the truth — hydrate from there so /status doesn't
        # lie and the dashboard's replay bar appears on reopened terminal runs.
        # Live states (running/paused) belong to a process that no longer exists,
        # so demote them to "stopped" rather than claim we're still ticking.
        from storage import db as dbm
        try:
            row = dbm.get_run(registry.conn_for(run_id), run_id)
        except (sqlite3.Error, OSError, KeyError) as e:
            # KeyError: run is being deleted (conn_for raises KeyError from _deleting set)
            # sqlite3.Error/OSError: DB corruption or I/O failure
            log.warning("failed to read persisted state for worker %s: %s", run_id, e)
            row = None
        persisted = (row or {}).get("status") or "pending"
        if persisted in ("stopped", "finished", "draining"):
            self.state = persisted
        elif persisted in ("running", "paused"):
            self.state = "stopped"
        else:
            self.state = "pending"
        self.interval_ms: int = 500
        self.last_step_ms: int = 0
        self._pause = threading.Event()
        self._pause.set()  # starts un-paused; .clear() to pause
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._stop_callback = None  # set by registry to remove from workers dict
        self._persist_on_stop = True

    # ----- subscriber API for SSE -----

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=128)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _publish(self, event: dict) -> None:
        with self._sub_lock:
            dead: list[queue.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    # ----- control -----

    def start(self, interval_ms: int = 500) -> dict:
        if self.state in ("running", "paused") or (
            self.state == "draining"
            and self._thread is not None
            and self._thread.is_alive()
        ):
            return self.status()
        if self.state in ("stopped", "finished"):
            # cannot restart a stopped worker; caller should create a new one
            return {"error": f"worker already {self.state}", **self.status()}
        self.interval_ms = max(0, int(interval_ms))
        self._stop.clear()
        self._pause.set()
        self.state = "running"
        try:
            from storage import db as dbm
            dbm.mark_run_running(self.registry.conn_for(self.run_id), self.run_id)
        except (KeyError, sqlite3.OperationalError):
            return {"error": "run deleted", "run_id": self.run_id}
        except sqlite3.Error as e:
            log.warning("unexpected DB error starting run %s: %s", self.run_id, e)
            return {"error": "run deleted", "run_id": self.run_id}
        self._thread = threading.Thread(target=self._loop, name=f"runworker-{self.run_id}",
                                        daemon=True)
        self._thread.start()
        env = self.registry.get_env(self.run_id)
        if env is not None:
            with env.lock:
                if self._on_turn not in env.turn_listeners:
                    env.turn_listeners.append(self._on_turn)
        return self.status()

    def _on_turn(self, turn: dict) -> None:
        """env.turn_listeners callback. Translates a recorded turn into a
        compact SSE 'turn' event; the dashboard reacts by refreshing the
        agent panel. The full turn payload is fetched lazily over HTTP."""
        self._publish({
            "type": "turn",
            "t": turn.get("t"),
            "agent_id": turn.get("agent_id"),
            "turn_idx": turn.get("turn_idx"),
            "turn_id": turn.get("turn_id"),
        })

    def pause(self) -> dict:
        if self.state in ("running", "draining"):
            self._pause.clear()
            self.state = "paused"
            try:
                from storage import db as dbm
                dbm.update_run_status(self.registry.conn_for(self.run_id), self.run_id, "paused")
            except (KeyError, sqlite3.OperationalError):
                pass  # run may be deleted concurrently
            except sqlite3.Error as e:
                log.warning("unexpected DB error pausing run %s: %s", self.run_id, e)
        return self.status()

    def _preserve_pause_after_step(self) -> bool:
        if self._stop.is_set() or self._pause.is_set():
            return False
        self.state = "paused"
        try:
            from storage import db as dbm
            dbm.update_run_status(self.registry.conn_for(self.run_id), self.run_id, "paused")
        except (KeyError, sqlite3.OperationalError):
            pass  # run may be deleted concurrently
        except sqlite3.Error as e:
            log.warning("unexpected DB error pausing run %s: %s", self.run_id, e)
        return True

    def resume(self) -> dict:
        if self.state == "paused":
            self._pause.set()
            self.state = "running"
            try:
                from storage import db as dbm
                dbm.mark_run_running(self.registry.conn_for(self.run_id), self.run_id)
            except (KeyError, sqlite3.OperationalError):
                pass  # run may be deleted concurrently
            except sqlite3.Error as e:
                log.warning("unexpected DB error resuming run %s: %s", self.run_id, e)
        return self.status()

    def stop(self, *, persist: bool = True) -> dict:
        with self.registry.runtime_lifecycle_for(self.run_id):
            with self.registry.lock:
                is_current = self.registry.workers.get(self.run_id) is self
            if not is_current:
                return self.status()
            return self._stop_with_lifecycle(persist=persist)

    def _stop_with_lifecycle(self, *, persist: bool) -> dict:
        self._stop.set()
        self._pause.set()  # release any pause-wait so the loop can exit
        self._persist_on_stop = persist
        # also release any open hook so step() returns promptly
        env = self.registry.get_env(self.run_id)
        if env is not None:
            env.hook_event.set()
            # detach turn listener so the stopped worker doesn't leak into
            # a future fresh worker's event stream.
            with env.lock:
                try:
                    env.turn_listeners.remove(self._on_turn)
                except ValueError:
                    pass
        was_finished = self.state == "finished"
        self.state = "stopped"
        if not persist:
            return {
                "run_id": self.run_id,
                "state": self.state,
                "phase": self.state,
                "t": env.t if env is not None else 0,
                "interval_ms": self.interval_ms,
                "last_step_ms": self.last_step_ms,
                "subscribers": len(self._subscribers),
                "active_orders_remaining": 0,
                "active_order_status_counts": {},
            }
        if not was_finished:
            # Check if the loop thread already persisted a terminal status
            # (e.g. step-error or drain-safety). Avoid overwriting its
            # finished_at with a later timestamp.
            already_terminal = False
            try:
                from storage import db as dbm
                conn = self.registry.conn_for(self.run_id)
                row = dbm.get_run(conn, self.run_id)
                if row and row.get("status") in ("finished", "stopped"):
                    already_terminal = True
            except (KeyError, sqlite3.Error):
                pass
            if not already_terminal:
                self._persist_stopped()
        # Suppress the loop thread's own _persist_stopped call to prevent
        # a double-write race on finished_at. The synchronous persist above
        # is the canonical one.
        self._persist_on_stop = False
        return self.status()

    def _persist_stopped(self) -> None:
        try:
            from storage import db as dbm
            from web.runner import _now_iso
            conn = self.registry.conn_for(self.run_id)
            dbm.mark_run_terminal(conn, self.run_id, "stopped", _now_iso())
        except (KeyError, sqlite3.OperationalError):
            pass  # run may be deleted concurrently
        except sqlite3.Error as exc:
            log.warning("unexpected DB error stopping run %s: %s", self.run_id, exc)

    def status(self) -> dict:
        env = self.registry.get_env(self.run_id)
        row = self.registry.get_run(self.run_id) if env is None else None
        t = env.t if env else int((row or {}).get("current_t") or 0)
        try:
            active_counts = (
                self.registry.active_order_status_counts(self.run_id)
                if env is not None else {}
            )
        except Exception:
            active_counts = {}
        active_count = sum(active_counts.values())
        phase = self.registry._phase_for(env, active_count) if env is not None else self.state
        return {
            "run_id": self.run_id,
            "state": self.state,
            "phase": phase,
            "t": t,
            "interval_ms": self.interval_ms,
            "last_step_ms": self.last_step_ms,
            "subscribers": len(self._subscribers),
            "active_orders_remaining": active_count,
            "active_order_status_counts": active_counts,
        }

    # ----- loop -----

    def _wait_for_hook_or_timeout(self, env, max_secs: float) -> None:
        """Wait for end_of_step while freezing the hook timeout during pause."""
        remaining = max(0.0, float(max_secs))
        poll_secs = 0.05
        while remaining > 0 and not self._stop.is_set():
            if env.hook_event.is_set():
                return

            if not self._pause.is_set():
                env.hook_event.wait(timeout=poll_secs)
                continue

            wait_for = min(poll_secs, remaining)
            started = time.monotonic()
            if env.hook_event.wait(timeout=wait_for):
                return
            if self._pause.is_set():
                remaining -= max(0.0, time.monotonic() - started)

    def _loop(self) -> None:
        env = self.registry.get_env(self.run_id)
        if env is None:
            log.error("worker started but env missing for %s", self.run_id)
            self.state = "stopped"
            try:
                from storage import db as dbm
                from web.runner import _now_iso
                conn = self.registry.conn_for(self.run_id)
                dbm.mark_run_terminal(conn, self.run_id, "stopped", _now_iso())
            except (KeyError, sqlite3.Error, OSError):
                pass
            self.registry.release_terminal_runtime(self.run_id, worker=self)
            self._publish({
                "type": "stopped",
                "t": 0,
                "phase": "stopped",
                "active_orders_remaining": 0,
                "active_order_status_counts": {},
            })
            return
        max_secs = float(env.scenario["run"]["max_hook_seconds"])

        def blocker():
            self._wait_for_hook_or_timeout(env, max_secs)

        def remove_turn_listener():
            try:
                env.turn_listeners.remove(self._on_turn)
            except ValueError:
                pass

        manual_stop_published = False
        while not self._stop.is_set():
            self._pause.wait()
            if self._stop.is_set():
                # _persist_stopped is deferred to the post-loop block to
                # avoid a double-write race with _stop_with_lifecycle.
                remove_turn_listener()
                self._publish({
                    "type": "stopped",
                    "t": env.t,
                    "phase": self.state,
                    "active_orders_remaining": 0,
                    "active_order_status_counts": {},
                })
                manual_stop_published = True
                break

            active_before, counts_before = self.registry._active_order_summary(self.run_id)
            phase_before = self.registry._phase_for(env, active_before)
            pending_hook_t = self.registry._pending_hook_t(env)
            if phase_before == "finished" and pending_hook_t is None:
                self.state = "finished"
                self.registry._mark_finished(env)
                remove_turn_listener()
                self._publish({
                    "type": "finished",
                    "t": env.t,
                    "phase": "finished",
                    "active_orders_remaining": active_before,
                    "active_order_status_counts": counts_before,
                })
                break

            drain = phase_before == "draining"
            if drain:
                self.state = "draining"
                self.registry._mark_draining(env)
                drain_started_t = getattr(env, "drain_started_t", env.t)
                if env.t - drain_started_t > self.registry._drain_safety_max_steps(env):
                    try:
                        from storage import db as dbm
                        from web.runner import _now_iso
                        conn = self.registry.conn_for(self.run_id)
                        dbm.mark_run_terminal(conn, self.run_id, "stopped", _now_iso())
                    except (KeyError, sqlite3.OperationalError):
                        pass  # run may be deleted concurrently
                    except sqlite3.Error as e:
                        log.warning("unexpected DB error in drain safety for run %s: %s", self.run_id, e)
                    self._persist_on_stop = False
                    self.state = "stopped"
                    remove_turn_listener()
                    self._publish({
                        "type": "error",
                        "error": "drain_safety_max_steps_exceeded",
                        "phase": "draining",
                        "active_orders_remaining": active_before,
                        "active_order_status_counts": counts_before,
                    })
                    break

            t0 = time.time()
            try:
                result = env.step(hook_blocker=blocker, drain=drain)
                dt_ms = int((time.time() - t0) * 1000)
                self.last_step_ms = dt_ms
                active_after, counts_after = self.registry._active_order_summary(self.run_id)
                phase_after = self.registry._phase_for(env, active_after)
                if phase_after == "draining" and not self._stop.is_set():
                    self.registry._mark_draining(env)
                    if not self._preserve_pause_after_step():
                        self.state = "draining"
                elif phase_after == "finished" and not self._stop.is_set():
                    self.state = "finished"
                    self.registry._mark_finished(env)
                self._publish({
                    "type": "tick",
                    "t": result.t,
                    "new_orders": result.new_orders,
                    "state_transitions": result.state_transitions,
                    "events": result.events,
                    "phase": phase_after,
                    "active_orders_remaining": active_after,
                    "active_order_status_counts": counts_after,
                    "step_ms": dt_ms,
                })
                if phase_after == "finished" and not self._stop.is_set():
                    remove_turn_listener()
                    self._publish({
                        "type": "finished",
                        "t": env.t,
                        "phase": "finished",
                        "active_orders_remaining": active_after,
                        "active_order_status_counts": counts_after,
                    })
                    break
            except Exception as step_error:  # noqa: BLE001
                if self._stop.is_set():
                    remove_turn_listener()
                    break
                log.exception("step failed")
                self.state = "stopped"
                try:
                    from storage import db as dbm
                    from web.runner import _now_iso
                    conn = self.registry.conn_for(self.run_id)
                    dbm.mark_run_terminal(conn, self.run_id, "stopped", _now_iso())
                except (KeyError, sqlite3.OperationalError):
                    pass  # run may be deleted concurrently
                except sqlite3.Error as db_error:
                    log.warning("unexpected DB error in step exception handler for run %s: %s", self.run_id, db_error)
                self._persist_on_stop = False
                remove_turn_listener()
                self._publish({"type": "error", "error": str(step_error)})
                break

            if self._stop.is_set():
                break
            # interruptible sleep
            if self.interval_ms > 0:
                self._stop.wait(self.interval_ms / 1000.0)
        if self.state == "stopped" and self._stop.is_set():
            if self._persist_on_stop:
                self._persist_stopped()
            if not manual_stop_published:
                remove_turn_listener()
                self._publish({
                    "type": "stopped",
                    "t": env.t,
                    "phase": "stopped",
                    "active_orders_remaining": 0,
                    "active_order_status_counts": {},
                })
        if self.state in ("finished", "stopped"):
            self.registry.release_terminal_runtime(self.run_id, worker=self)
