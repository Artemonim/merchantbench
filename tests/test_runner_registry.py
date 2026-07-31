import os
import copy
import sqlite3
import threading
import time
from datetime import datetime

import pytest

from data import synth
from storage import db as dbm
from web import runner as runner_mod
from web.app import create_app
from web.runner import load_default_scenario


def _tiny_scenario():
    scenario = load_default_scenario()
    scenario["run"]["horizon_steps"] = 4
    scenario["run"]["max_hook_seconds"] = 0.01
    scenario["data"]["source"] = "synthetic"
    scenario["data"]["num_products"] = 20
    return scenario


def test_create_runs_use_isolated_state_dbs(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry

    first_id = registry.create_run(_tiny_scenario(), auto_start=False, name="first")
    second_id = registry.create_run(_tiny_scenario(), auto_start=False, name="second")

    first_db = registry.run_db_path(first_id)
    second_db = registry.run_db_path(second_id)

    assert first_db == os.path.join(tmp_path, "runs", first_id, "state.db")
    assert second_db == os.path.join(tmp_path, "runs", second_id, "state.db")
    assert first_db != second_db
    assert os.path.exists(first_db)
    assert os.path.exists(second_db)

    with sqlite3.connect(first_db) as conn:
        first_runs = conn.execute("SELECT run_id FROM runs").fetchall()
        first_products = conn.execute("SELECT DISTINCT run_id FROM products").fetchall()
    with sqlite3.connect(second_db) as conn:
        second_runs = conn.execute("SELECT run_id FROM runs").fetchall()
        second_products = conn.execute("SELECT DISTINCT run_id FROM products").fetchall()

    assert first_runs == [(first_id,)]
    assert first_products == [(first_id,)]
    assert second_runs == [(second_id,)]
    assert second_products == [(second_id,)]


def test_phase_stays_running_while_any_agent_lives_and_ends_when_all_die(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    single_id = registry.create_run(_tiny_scenario(), auto_start=False)
    single_env = registry.envs[single_id]
    single_env.agents["agent_0"].is_alive = False
    assert registry._phase_for(single_env, active_count=1) == "draining"
    assert registry._phase_for(single_env, active_count=0) == "finished"

    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    registry.add_agent(run_id, "agent_1", "Agent 1")
    env = registry.envs[run_id]
    env.agents["agent_0"].is_alive = False
    assert registry._phase_for(env, active_count=1) == "running"

    env.agents["agent_1"].is_alive = False
    assert registry._phase_for(env, active_count=1) == "draining"
    assert registry._phase_for(env, active_count=0) == "finished"

    env.finished = True
    env.drain_started_t = env.t
    env.agents["agent_1"].is_alive = True
    assert registry._phase_for(env, active_count=1) == "draining"
    with pytest.raises(runner_mod.RunPhaseClosedError):
        registry.add_agent(run_id, "agent_2", "Agent 2")
    assert "agent_2" not in env.agents

    response = app.test_client().post(
        f"/runs/{run_id}/agents",
        json={"agent_id": "agent_2", "name": "Agent 2"},
    )
    assert response.status_code == 410
    assert response.get_json() == {
        "error": "agent_addition_closed",
        "state": "finished",
    }


def test_catalog_materialization_failure_is_reported_without_removing_run(
    tmp_path, monkeypatch
):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )

    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(
        runner_mod.catalog_diagnostics,
        "build_catalog_diagnostics_artifact",
        fail_materialization,
    )

    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    status_path = app.registry.catalog_diagnostics_status_path(run_id)
    deadline = time.monotonic() + 5
    status = None
    while time.monotonic() < deadline:
        status = runner_mod.catalog_diagnostics.read_catalog_diagnostics_status(
            status_path
        )
        if status and status["status"] == "failed":
            break
        time.sleep(0.01)

    assert status == {
        "status": "failed",
        "error": "RuntimeError: diagnostics failed",
    }
    assert app.registry.get_run(run_id) is not None
    assert run_id in app.registry.envs
    assert not os.path.exists(app.registry.catalog_diagnostics_path(run_id))
    with app.test_client() as client:
        response = client.get(f"/runs/{run_id}/sections/catalog_diagnostics")
    assert response.status_code == 503
    assert response.get_json() == {
        "error": "catalog_diagnostics_failed",
        "status": "failed",
    }


def test_catalog_materialization_uses_creation_snapshot(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    entered = threading.Event()
    release = threading.Event()
    captured = {}
    real_build = runner_mod.catalog_diagnostics.build_catalog_diagnostics_artifact

    def delayed_materialization(source, **kwargs):
        captured["source"] = source
        entered.set()
        assert release.wait(timeout=5)
        return real_build(source, **kwargs)

    monkeypatch.setattr(
        runner_mod.catalog_diagnostics,
        "build_catalog_diagnostics_artifact",
        delayed_materialization,
    )

    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    assert entered.wait(timeout=5)
    with app.test_client() as client:
        pending = client.get(f"/runs/{run_id}/sections/catalog_diagnostics")
    assert pending.status_code == 202
    assert pending.get_json() == {
        "error": "catalog_diagnostics_pending",
        "status": "pending",
        "retry_after_ms": 300,
    }
    env = app.registry.envs[run_id]
    product_id = sorted(env.products)[0]
    initial_price = captured["source"].products[product_id].price
    env.products[product_id].price = initial_price + 999
    release.set()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = runner_mod.catalog_diagnostics.read_catalog_diagnostics_status(
            app.registry.catalog_diagnostics_status_path(run_id)
        )
        if status and status["status"] == "ready":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("catalog diagnostics did not finish materializing")

    assert captured["source"].products[product_id].price == initial_price
    assert captured["source"].products[product_id] is not env.products[product_id]


def test_pending_catalog_materialization_does_not_recreate_deleted_run(
    tmp_path, monkeypatch
):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_build = runner_mod.catalog_diagnostics.build_catalog_diagnostics_artifact

    def delayed_materialization(source, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        try:
            return real_build(source, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(
        runner_mod.catalog_diagnostics,
        "build_catalog_diagnostics_artifact",
        delayed_materialization,
    )

    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    assert entered.wait(timeout=5)
    run_dir = os.path.join(app.registry.runs_root, run_id)
    assert app.registry.delete_run(run_id)["deleted"]
    release.set()
    assert finished.wait(timeout=5)
    time.sleep(0.05)

    assert not os.path.exists(run_dir)


def test_pending_catalog_materialization_resumes_after_registry_restart(
    tmp_path, monkeypatch
):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    artifact_path = app.registry.catalog_diagnostics_path(run_id)
    status_path = app.registry.catalog_diagnostics_status_path(run_id)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = runner_mod.catalog_diagnostics.read_catalog_diagnostics_status(
            status_path
        )
        if status and status["status"] == "ready":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("initial catalog diagnostics did not finish")

    os.remove(artifact_path)
    runner_mod.catalog_diagnostics.write_catalog_diagnostics_status(
        status_path, "pending"
    )
    resumed = threading.Event()
    real_build = runner_mod.catalog_diagnostics.build_catalog_diagnostics_artifact

    def observe_resume(source, **kwargs):
        resumed.set()
        return real_build(source, **kwargs)

    monkeypatch.setattr(
        runner_mod.catalog_diagnostics,
        "build_catalog_diagnostics_artifact",
        observe_resume,
    )

    restarted = runner_mod.RunRegistry(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    assert resumed.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = runner_mod.catalog_diagnostics.read_catalog_diagnostics_status(
            status_path
        )
        if status and status["status"] == "ready":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("resumed catalog diagnostics did not finish")

    assert os.path.exists(artifact_path)
    restarted.shutdown()
    app.registry.shutdown()


def test_initial_quantity_is_write_once_when_product_state_changes(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    run_id = app.registry.create_run(_tiny_scenario(), auto_start=False)
    product = next(iter(app.registry.envs[run_id].products.values()))
    initial_quantity = product.quantity

    product.quantity = max(0, product.quantity - 3)
    dbm.upsert_product_state(app.registry.conn_for(run_id), run_id, product)
    row = app.registry.conn_for(run_id).execute(
        "SELECT quantity, initial_quantity FROM products WHERE run_id=? AND product_id=?",
        (run_id, product.product_id),
    ).fetchone()

    assert row["quantity"] == product.quantity
    assert row["initial_quantity"] == initial_quantity


def test_list_runs_scans_per_run_state_dbs(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry

    first_id = registry.create_run(_tiny_scenario(), auto_start=False, name="first")
    second_id = registry.create_run(_tiny_scenario(), auto_start=False, name="second")
    registry.conn_for(first_id).execute(
        "UPDATE runs SET started_at=? WHERE run_id=?",
        ("2026-06-16T00:00:00", first_id),
    )
    registry.conn_for(second_id).execute(
        "UPDATE runs SET started_at=? WHERE run_id=?",
        ("2026-06-17T00:00:00", second_id),
    )
    dbm.update_run_finished_at(registry.conn_for(first_id), first_id, "2026-06-16T00:00:00")
    dbm.update_run_finished_at(registry.conn_for(second_id), second_id, "2026-06-17T00:00:00")

    rows = registry.list_runs()

    assert [row["run_id"] for row in rows] == [second_id, first_id]
    assert [row["name"] for row in rows] == ["second", "first"]


def test_registry_startup_reconciles_orphaned_live_states(tmp_path):
    runs_root = os.path.join(tmp_path, "runs")
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=runs_root,
    )
    registry = app.registry
    running_id = registry.create_run(_tiny_scenario(), auto_start=False)
    draining_id = registry.create_run(_tiny_scenario(), auto_start=False)
    paused_id = registry.create_run(_tiny_scenario(), auto_start=False)
    old_finished_at = "2026-06-16T00:00:00"

    for run_id, status in (
        (running_id, "running"),
        (draining_id, "draining"),
        (paused_id, "paused"),
    ):
        conn = registry.conn_for(run_id)
        dbm.update_run_status(conn, run_id, status)
        dbm.update_run_finished_at(conn, run_id, old_finished_at)
        registry.close_run_conn(run_id)

    restarted = runner_mod.RunRegistry(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=runs_root,
    )

    running = restarted.get_run(running_id)
    draining = restarted.get_run(draining_id)
    paused = restarted.get_run(paused_id)
    assert running["status"] == "stopped"
    assert draining["status"] == "stopped"
    assert running["finished_at"] != old_finished_at
    assert draining["finished_at"] != old_finished_at
    assert paused["status"] == "paused"
    assert paused["finished_at"] == old_finished_at


def test_registry_shutdown_persists_live_worker_as_stopped(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    scenario = _tiny_scenario()
    scenario["run"]["horizon_steps"] = 100
    scenario["run"]["max_hook_seconds"] = 1.0
    run_id = registry.create_run(scenario, auto_start=False)
    registry._auto_start(run_id, interval_ms=1000)

    registry.shutdown()

    with sqlite3.connect(registry.run_db_path(run_id)) as conn:
        status, finished_at = conn.execute(
            "SELECT status, finished_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    assert status == "stopped"
    assert finished_at is not None
    assert registry.bootstrap_procs == {}
    assert registry._conns == {}
    assert registry._read_conns == {}


def test_deleting_run_blocks_cached_connections(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)

    cached = registry.conn_for(run_id)
    assert registry._conns[run_id] is cached

    with registry._conn_lock:
        registry._deleting.add(run_id)
    try:
        with pytest.raises(KeyError, match="being deleted"):
            registry.conn_for(run_id)
        with pytest.raises(KeyError, match="being deleted"):
            with registry.read_conn_for(run_id):
                pass
    finally:
        with registry._conn_lock:
            registry._deleting.discard(run_id)


def test_delete_run_does_not_remove_non_run_directory(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = "run-not-a-real-run"
    target = os.path.join(registry.runs_root, run_id)
    os.makedirs(target)
    marker = os.path.join(target, "keep.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("not a run")

    result = registry.delete_run(run_id)

    assert result["deleted"] is False
    assert result["reason"] == "not found"
    assert os.path.exists(marker)


def test_delete_run_waits_for_borrowed_connection_before_close(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    delete_done = threading.Event()
    delete_result = {}

    with registry.read_conn_for(run_id) as conn:
        def delete_run():
            delete_result.update(registry.delete_run(run_id))
            delete_done.set()

        thread = threading.Thread(target=delete_run)
        thread.start()
        try:
            assert not delete_done.wait(timeout=0.2)
            row = conn.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            assert row["run_id"] == run_id
        finally:
            thread.join(timeout=0)

    assert delete_done.wait(timeout=2)
    assert delete_result["deleted"] is True
    assert not os.path.exists(registry.run_db_path(run_id))


def test_delete_run_waits_for_in_flight_registry_db_operation(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    entered = threading.Event()
    release = threading.Event()
    list_done = threading.Event()
    list_error = {}
    delete_done = threading.Event()
    delete_result = {}
    original_list_agents = runner_mod.dbm.list_agents

    def slow_list_agents(conn, *args, **kwargs):
        entered.set()
        release.wait(timeout=2)
        return original_list_agents(conn, *args, **kwargs)

    monkeypatch.setattr(runner_mod.dbm, "list_agents", slow_list_agents)

    def list_agents():
        try:
            registry.list_agents(run_id)
        except Exception as e:  # noqa: BLE001
            list_error["error"] = e
        finally:
            list_done.set()

    list_thread = threading.Thread(target=list_agents)
    list_thread.start()
    assert entered.wait(timeout=2)

    def delete_run():
        delete_result.update(registry.delete_run(run_id))
        delete_done.set()

    delete_thread = threading.Thread(target=delete_run)
    delete_thread.start()
    try:
        assert not delete_done.wait(timeout=0.2)
    finally:
        release.set()
        list_thread.join(timeout=3)
        delete_thread.join(timeout=3)

    assert list_done.is_set()
    assert list_error == {}
    assert delete_done.is_set()
    assert delete_result["deleted"] is True


def test_late_rehydrate_does_not_overwrite_live_env(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    live_env = registry._require(run_id)
    with registry.lock:
        registry.envs.pop(run_id)

    original_environment = runner_mod.Environment

    def environment_appears_during_rehydrate(*args, **kwargs):
        stale_env = original_environment(*args, **kwargs)
        with registry.lock:
            registry.envs[run_id] = live_env
        return stale_env

    monkeypatch.setattr(runner_mod, "Environment", environment_appears_during_rehydrate)

    rehydrated = registry._rehydrate(run_id)

    assert rehydrated is live_env
    assert registry.get_env(run_id) is live_env


def test_terminal_cleanup_identity_guard_does_not_remove_replacement_worker(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    old_worker = object()
    replacement_worker = object()
    registry.workers[run_id] = replacement_worker

    released = registry.release_terminal_runtime(run_id, worker=old_worker)

    assert released is False
    assert registry.workers[run_id] is replacement_worker
    assert run_id in registry.envs


def test_runtime_lifecycle_keeps_one_lock_while_waiter_is_registered(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    waiter_entered = threading.Event()
    release_waiter = threading.Event()
    contender_entered = threading.Event()

    def waiter_body():
        with registry.runtime_lifecycle_for(run_id):
            waiter_entered.set()
            assert release_waiter.wait(timeout=2)

    def contender_body():
        with registry.runtime_lifecycle_for(run_id):
            contender_entered.set()

    with registry.runtime_lifecycle_for(run_id):
        waiter = threading.Thread(target=waiter_body)
        waiter.start()
        deadline = time.time() + 2
        users = 0
        while time.time() < deadline:
            with registry.lock:
                users = getattr(registry._runtime_locks[run_id], "users", 0)
            if users == 2:
                break
            time.sleep(0.01)
        assert users == 2

    assert waiter_entered.wait(timeout=2)
    contender = threading.Thread(target=contender_body)
    contender.start()
    assert not contender_entered.wait(timeout=0.1)
    release_waiter.set()
    waiter.join(timeout=2)
    contender.join(timeout=2)

    assert not waiter.is_alive()
    assert not contender.is_alive()
    assert contender_entered.is_set()
    with registry.lock:
        assert run_id not in registry._runtime_locks


def test_rehydrate_waits_until_terminal_cleanup_has_closed_old_runtime(
    tmp_path, monkeypatch
):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    dbm.update_run_status(registry.conn_for(run_id), run_id, "stopped")

    class StoppedWorker:
        state = "stopped"

    old_worker = StoppedWorker()
    registry.workers[run_id] = old_worker
    cleanup_reached_close = threading.Event()
    allow_cleanup_to_close = threading.Event()
    rehydrate_done = threading.Event()
    rehydrated = {}
    real_close = registry.close_run_conn

    def blocked_close(close_run_id):
        cleanup_reached_close.set()
        assert allow_cleanup_to_close.wait(timeout=2)
        real_close(close_run_id)

    monkeypatch.setattr(registry, "close_run_conn", blocked_close)
    cleanup_thread = threading.Thread(
        target=lambda: registry.release_terminal_runtime(run_id, worker=old_worker)
    )
    cleanup_thread.start()
    assert cleanup_reached_close.wait(timeout=2)

    def rehydrate():
        rehydrated["env"] = registry._require(run_id)
        rehydrate_done.set()

    rehydrate_thread = threading.Thread(target=rehydrate)
    rehydrate_thread.start()
    rehydrate_was_blocked = not rehydrate_done.wait(timeout=0.1)
    allow_cleanup_to_close.set()
    cleanup_thread.join(timeout=2)
    rehydrate_thread.join(timeout=2)

    assert rehydrate_was_blocked
    assert rehydrate_done.is_set()
    assert rehydrated["env"].conn.execute("SELECT 1").fetchone()[0] == 1


def test_create_keeps_bootstrap_spawn_inside_runtime_lifecycle(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    cleanup_complete = threading.Event()
    cleanup_threads = []

    class FinishedWorker:
        state = "finished"

    class BootstrapProc:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    def finish_during_auto_start(run_id, _interval_ms):
        worker = FinishedWorker()
        registry.workers[run_id] = worker
        dbm.update_run_status(registry.conn_for(run_id), run_id, "finished")

        def cleanup():
            registry.release_terminal_runtime(run_id, worker=worker)
            cleanup_complete.set()

        thread = threading.Thread(target=cleanup)
        cleanup_threads.append(thread)
        thread.start()

    def spawn_after_cleanup_window(run_id, _base_url, _scenario):
        cleanup_complete.wait(timeout=0.1)
        registry.bootstrap_procs[run_id] = BootstrapProc()

    monkeypatch.setattr(registry, "_auto_start", finish_during_auto_start)
    monkeypatch.setattr(registry, "_spawn_auto_seed", spawn_after_cleanup_window)

    run_id = registry.create_run(
        _tiny_scenario(), auto_start=True, bootstrap_agent="auto_seed"
    )
    for thread in cleanup_threads:
        thread.join(timeout=2)

    assert cleanup_complete.is_set()
    assert run_id not in registry.bootstrap_procs


def test_sync_step_exception_persists_stopped_and_releases_runtime(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    env = registry.envs[run_id]

    def fail_step(*_args, **_kwargs):
        raise RuntimeError("sync step failed")

    monkeypatch.setattr(env, "step", fail_step)
    with pytest.raises(RuntimeError, match="sync step failed"):
        registry.step(run_id)

    assert registry.get_run(run_id)["status"] == "stopped"
    assert run_id not in registry.envs
    assert run_id not in registry._conns


def test_auto_step_stops_after_error_result(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "legacy.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)

    def drain_error(env):
        dbm.update_run_status(registry.conn_for(run_id), run_id, "stopped")
        return {
            "phase": "draining",
            "error": "drain_safety_max_steps_exceeded",
            "t": env.t,
        }

    monkeypatch.setattr(registry, "_step_with_env", drain_error)

    result = registry.auto_step(run_id, 2)

    assert result == {
        "steps": [{
            "phase": "draining",
            "error": "drain_safety_max_steps_exceeded",
            "t": 0,
        }],
        "current_t": 0,
    }


def test_delete_run_keeps_state_when_worker_is_still_stopping(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    env = registry._require(run_id)
    real_step = env.step
    step_entered = threading.Event()
    release_step = threading.Event()

    def slow_step(*args, **kwargs):
        step_entered.set()
        release_step.wait(timeout=5)
        return real_step(*args, **kwargs)

    env.step = slow_step
    registry._auto_start(run_id, interval_ms=0)
    worker = registry.workers[run_id]
    assert step_entered.wait(timeout=2)

    try:
        with app.test_client() as client:
            response = client.delete(f"/runs/{run_id}")

        assert response.status_code == 409
        payload = response.get_json()
        assert payload["status"] == "busy"
        assert payload["run_id"] == run_id
        assert worker._thread is not None and worker._thread.is_alive()
        assert os.path.exists(registry.run_db_path(run_id))
        assert registry.get_env(run_id) is env
        assert run_id in registry._conns
    finally:
        release_step.set()
        if worker._thread is not None:
            worker._thread.join(timeout=3)
        registry.delete_run(run_id)

    assert not os.path.exists(registry.run_db_path(run_id))


def test_delete_run_reports_failed_directory_removal_and_marks_stopped(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)

    def fail_rmtree(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(runner_mod.shutil, "rmtree", fail_rmtree)

    result = registry.delete_run(run_id)

    assert result["deleted"] is False
    assert result["status"] == "error"
    assert "permission denied" in result["reason"]
    assert os.path.exists(registry.run_db_path(run_id))
    row = dbm.get_run(registry.conn_for(run_id), run_id)
    assert row["status"] == "stopped"


def test_rehydrate_returns_none_for_corrupt_run_db(tmp_path, monkeypatch):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    with registry.lock:
        registry.envs.pop(run_id, None)

    def fail_load_products(*args, **kwargs):
        raise sqlite3.DatabaseError("corrupt page")

    monkeypatch.setattr(runner_mod.dbm, "load_products", fail_load_products)

    assert registry._rehydrate(run_id) is None


def test_run_and_agent_wall_clock_timestamps_use_local_time_without_offset(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry

    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    run = dbm.get_run(registry.conn_for(run_id), run_id)
    agent = dbm.list_agents(registry.conn_for(run_id), run_id)[0]

    assert datetime.fromisoformat(run["started_at"]).tzinfo is None
    assert datetime.fromisoformat(agent.created_at).tzinfo is None


def test_status_elapsed_uses_local_wall_clock_timestamps(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    run_id = registry.create_run(_tiny_scenario(), auto_start=False)
    registry.conn_for(run_id).execute(
        "UPDATE runs SET status='finished', started_at=?, finished_at=? WHERE run_id=?",
        (
            "2026-06-21T12:00:00",
            "2026-06-21T13:00:00",
            run_id,
        ),
    )

    with app.test_client() as client:
        status = client.get(f"/runs/{run_id}/status").get_json()

    assert status["elapsed_ms"] == 60 * 60 * 1000


def test_create_run_applies_difficulty_rate_to_probability_fields(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    scenario = _tiny_scenario()
    baseline_products, _ = synth.generate(copy.deepcopy(scenario))
    baseline_by_id = {p.product_id: p for p in baseline_products}
    scenario["difficulty_rate"] = {
        "cancel_rate": 0.7,
        "refund_rate": 0.5,
        "only_refund_rate": 0.25,
        "bad_review_rate": 0.0,
        "timeout_rate": 0.8,
        "price_change_rate": 0.6,
        "supplier_delist_rate": 0.4,
    }

    run_id = registry.create_run(scenario, auto_start=False)

    product = next(iter(registry._require(run_id).products.values()))
    baseline = baseline_by_id[product.product_id]
    for field, multiplier in scenario["difficulty_rate"].items():
        assert getattr(product, field) == pytest.approx(
            min(getattr(baseline, field) * multiplier, 1.0)
        )
    assert product.price == baseline.price


def test_create_run_rejects_negative_difficulty_rate(tmp_path):
    app = create_app(
        db_path=os.path.join(tmp_path, "test.db"),
        runs_root=os.path.join(tmp_path, "runs"),
    )
    registry = app.registry
    scenario = _tiny_scenario()
    scenario["difficulty_rate"] = {"refund_rate": -0.1}

    with pytest.raises(ValueError, match="difficulty_rate.refund_rate"):
        registry.create_run(scenario, auto_start=False)


def test_locked_connection_materializes_select_before_releasing_lock():
    class SlowCursor:
        description = (("value",),)
        lastrowid = None
        rowcount = -1

        def __init__(self, release_fetch):
            self._release_fetch = release_fetch

        def fetchall(self):
            self._release_fetch.wait(timeout=1)
            return [("row",)]

    class WriteCursor:
        description = None
        lastrowid = 1
        rowcount = 1

    class RawConnection:
        row_factory = None

        def __init__(self):
            self.release_fetch = threading.Event()
            self.select_started = threading.Event()
            self.write_finished = threading.Event()

        def execute(self, sql, *args, **kwargs):
            if sql.startswith("SELECT"):
                self.select_started.set()
                return SlowCursor(self.release_fetch)
            self.write_finished.set()
            return WriteCursor()

    raw = RawConnection()
    conn = runner_mod.LockedConnection(raw)
    execute_returned = threading.Event()
    allow_fetch = threading.Event()

    def run_select():
        cursor = conn.execute("SELECT slow")
        execute_returned.set()
        allow_fetch.wait(timeout=1)
        cursor.fetchall()

    select_thread = threading.Thread(
        target=run_select,
        daemon=True,
    )
    select_thread.start()
    assert raw.select_started.wait(timeout=1)

    write_thread = threading.Thread(
        target=lambda: conn.execute("UPDATE rows SET value=1"),
        daemon=True,
    )
    write_thread.start()
    time.sleep(0.05)

    assert not execute_returned.is_set()
    assert not raw.write_finished.is_set()

    allow_fetch.set()
    raw.release_fetch.set()
    select_thread.join(timeout=1)
    write_thread.join(timeout=1)

    assert execute_returned.is_set()
    assert raw.write_finished.is_set()
