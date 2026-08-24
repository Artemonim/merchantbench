"""HTTP routes for agent interaction with the environment.

All under /runs/<run_id>/... — agent_id is in the path for multi-agent runs.

Endpoints:
  POST /runs/<rid>/agent/register
  GET  /runs/<rid>/agent/meta
  GET  /runs/<rid>/tools/schema
  GET  /runs/<rid>/agents/<aid>/observation
  GET  /runs/<rid>/agents/<aid>/trace?t=<N>
  GET  /runs/<rid>/agents/<aid>/trace_index
  POST /runs/<rid>/agents/<aid>/act          ★ unified tool execution + trace
  POST /runs/<rid>/agents/<aid>/usage        delayed auxiliary cost only
  GET  /runs/<rid>/agent/cost
  GET  /runs/<rid>/agent/trace?t=<N>
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time as _time
from contextlib import ExitStack
from typing import Optional

from flask import Blueprint, abort, after_this_request, g, jsonify, request

from compat import (
    API_FAILED_EVENT,
    ENV_TOOL_ORIGIN,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    canonical_tool_origin,
    is_env_tool_origin,
    tool_schema_sha256,
)
from storage import agent_log
from storage import snapshot as snap
from tools import observation as obs_mod
from tools import registry
from tools import tools as tool_impl
from tools.dispatch import dispatch_tool


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


HERMES_TOOL_ORIGIN = "hermes_native"
_HERMES_OBSERVATION_RE = re.compile(r"^Day\s+(\d+),\s*Hour\s+(\d+)\b")


def _dead_agent_payload(env, agent_id: str) -> dict:
    agent_state = env.agents[agent_id]
    return {
        "ok": False,
        "error": "agent_dead",
        "message": f"agent {agent_id} is dead (deposit exhausted)",
        "died_at": tool_impl.t_to_agent_time_optional(
            env, agent_state.died_at_t,
        ),
    }


def _normalize_context(value) -> Optional[dict]:
    if not isinstance(value, dict):
        return None
    out = {}
    if "tokens" in value:
        try:
            tokens = int(value.get("tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        if tokens > 0:
            out["tokens"] = tokens
    compacted = value.get("compacted")
    if isinstance(compacted, bool):
        out["compacted"] = compacted
    for key in (
        "provider_api_failed_attempts",
        "retry_exhausted",
        "skills_evolutions",
    ):
        if key not in value:
            continue
        try:
            count = int(value.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            out[key] = count
    return out or None


def _context_summary_from_turns(turns: list[dict]) -> Optional[dict]:
    tokens = 0
    compactions = 0
    for turn in turns or []:
        ctx = turn.get("context") if isinstance(turn, dict) else None
        if not isinstance(ctx, dict):
            continue
        try:
            tokens = max(tokens, int(ctx.get("tokens") or 0))
        except (TypeError, ValueError):
            pass
        if ctx.get("compacted") is True:
            compactions += 1
    out = {}
    if tokens > 0:
        out["tokens"] = tokens
    if compactions:
        out["compactions"] = compactions
    return out or None


def _env_turns_for_step(runs_root: str, rid: str, t: int) -> list[dict]:
    idx = agent_log.read_step_index(runs_root, rid, t)
    if not isinstance(idx, dict):
        return []
    turns = idx.get("turns")
    return turns if isinstance(turns, list) else []


def _env_context_summary_by_step(runs_root: str, rid: str) -> dict[int, dict]:
    base = agent_log.agent_dir(runs_root, rid)
    by_step_dir = os.path.join(base, "by_step")
    summaries: dict[int, dict] = {}
    if not os.path.isdir(by_step_dir):
        return summaries
    for fname in sorted(os.listdir(by_step_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(by_step_dir, fname)
        try:
            with open(path) as f:
                step_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        context = _context_summary_from_turns(step_data.get("turns", []))
        if context:
            try:
                summaries[int(step_data.get("t", 0))] = context
            except (TypeError, ValueError):
                continue
    return summaries


def _turn_quota_error(env, agent_id: str) -> Optional[dict]:
    max_turns = int((env.scenario.get("agent", {}) or {}).get("max_turns_per_step", 0))
    if not max_turns:
        return None
    with env.turn_lock:
        if len(env.turns_meta_by_agent.get(agent_id, [])) >= max_turns:
            return {"ok": False, "error": f"max_turns_per_step={max_turns} reached"}
    return None


def _tool_call_origin(msg: dict, call: dict) -> str:
    if call.get("tool_origin") is not None:
        return canonical_tool_origin(call.get("tool_origin"))
    if msg.get("tool_origin") is not None:
        return canonical_tool_origin(msg.get("tool_origin"))
    return ENV_TOOL_ORIGIN


def _is_merchantbench_env_tool_call(msg: dict, call: dict) -> bool:
    return is_env_tool_origin(_tool_call_origin(msg, call))


def _hermes_tool_origin(name: str) -> str:
    name = str(name or "")
    if name == "end_of_step" or name.startswith(("merchantbench__", "realshop__")):
        return ENV_TOOL_ORIGIN
    return HERMES_TOOL_ORIGIN


def _parse_hermes_tool_calls(raw) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    out = []
    for tc in value:
        if not isinstance(tc, dict):
            continue
        tc = dict(tc)
        fn = tc.get("function") or {}
        name = str(fn.get("name") or tc.get("name") or "")
        tc.setdefault("tool_origin", _hermes_tool_origin(name))
        out.append(tc)
    return out


def _hermes_observation_t(content: str) -> Optional[int]:
    match = _HERMES_OBSERVATION_RE.match(str(content or "").lstrip())
    if not match:
        return None
    day = int(match.group(1))
    hour = int(match.group(2))
    return (day - 1) * 24 + hour if day > 0 else hour


def _idempotency_conflict(env, key: Optional[str], fingerprint: Optional[dict]) -> bool:
    if not key:
        return False
    hit = env.idem_cache.get(key)
    return (
        isinstance(hit, dict)
        and "__idem_result" in hit
        and hit.get("__idem_fingerprint") != fingerprint
    )


def check_stale_step(env) -> Optional[tuple]:
    """Reject if X-Agent-Step header doesn't match env.t."""
    declared = request.headers.get("X-Agent-Step")
    if declared is None:
        return None
    try:
        declared_t = int(declared)
    except ValueError:
        return jsonify({"ok": False,
                         "error": "invalid X-Agent-Step header",
                         "value": declared}), 400
    if declared_t != env.t:
        return jsonify({"ok": False, "error": "stale_step",
                         "agent_step": declared_t,
                         "env_step": env.t,
                         "hint": "your decision was based on an older"
                                 " observation; re-fetch /observation"
                                 " and re-plan"}), 425
    return None


def _filter_step_index_for_agent(
    idx: dict,
    agent_id: str,
    *,
    allow_legacy_agent0: bool = True,
) -> dict:
    out = dict(idx)
    messages = list(out.get("messages") or [])
    message_agents = out.get("message_agents")
    if isinstance(message_agents, list) and len(message_agents) == len(messages):
        kept = [
            (msg, owner)
            for msg, owner in zip(messages, message_agents)
            if owner == agent_id
        ]
        out["messages"] = [msg for msg, _ in kept]
        out["message_agents"] = [owner for _, owner in kept]
    elif agent_id != "agent_0" or not allow_legacy_agent0:
        out["messages"] = []
        out["message_agents"] = []
    else:
        out["message_agents"] = [agent_id] * len(messages)

    turns = list(out.get("turns") or [])
    if any(isinstance(turn, dict) and "agent_id" in turn for turn in turns):
        out["turns"] = [
            turn for turn in turns
            if isinstance(turn, dict) and turn.get("agent_id") == agent_id
        ]
    elif agent_id != "agent_0" or not allow_legacy_agent0:
        out["turns"] = []
    out["n_turns"] = len(out.get("turns") or [])
    return out


def make_blueprint(registry_obj) -> Blueprint:
    bp = Blueprint("agent_routes", __name__)

    @bp.teardown_request
    def _release_request_envs(exc):
        stack = g.pop("_merchantbench_agent_env_stack", None)
        if stack is not None:
            stack.close()
        g.pop("_merchantbench_agent_env_cache", None)

    TERMINAL_STATUSES = ("finished", "stopped", "error")

    def env_of(run_id):
        stack = getattr(g, "_merchantbench_agent_env_stack", None)
        if stack is None:
            stack = ExitStack()
            g._merchantbench_agent_env_stack = stack
            g._merchantbench_agent_env_cache = {}
        cache = g._merchantbench_agent_env_cache
        if run_id not in cache:
            try:
                cache[run_id] = stack.enter_context(registry_obj.lease_env_for(run_id))
            except KeyError as e:
                if "being deleted" in str(e):
                    abort(409, description="run is being deleted")
                # Runtime not loaded — check if the run is terminal so
                # callers get 410 (stop polling) rather than 404.
                row = registry_obj.get_run(run_id)
                if row and row.get("status") in TERMINAL_STATUSES:
                    abort(410, description="run_finished")
                abort(404)
            except ValueError:
                abort(404)
        return cache[run_id]

    def _resolve_run_dir(run_id):
        """Return (runs_root, run_id) — from live env or disk fallback.

        Read-only endpoints (cost, trace, all_traces, etc.) only need the
        on-disk artifact path.  When the in-memory Environment has been
        released (finished / stopped / server restart), we can still serve
        data directly from disk without aborting with 410.
        """
        try:
            env = env_of(run_id)
            return env.runs_root, env.run_id
        except Exception:
            # env_of calls abort() which raises an HTTPException.
            # Fall through to disk-only path.
            pass
        # Validate the run exists on disk.
        run_dir = os.path.join(registry_obj.runs_root, run_id)
        if os.path.isdir(run_dir):
            return registry_obj.runs_root, run_id
        # Last resort: check the DB catalog.
        row = registry_obj.get_run(run_id)
        if row:
            return registry_obj.runs_root, run_id
        abort(404)

    def _resolve_agent_trace_scope(run_id: str, agent_id: str):
        """Resolve disk-backed trace access without requiring a live runtime."""
        runs_root, rid = _resolve_run_dir(run_id)
        try:
            agents = registry_obj.list_agents(rid)
        except (KeyError, ValueError, sqlite3.Error, OSError):
            return None
        agent_ids = {
            str(agent.get("agent_id"))
            for agent in agents
            if isinstance(agent, dict) and agent.get("agent_id")
        }
        if agent_id not in agent_ids:
            return None
        return runs_root, rid, len(agent_ids) == 1

    def _hermes_state_db_path(runs_root: str, rid: str) -> str:
        return os.path.join(agent_log.agent_dir(runs_root, rid), "hermes_home", "state.db")

    def _include_compacted_hermes_messages() -> bool:
        value = str(request.args.get("include_compacted") or "").lower()
        return value in ("1", "true", "yes", "on")

    def _hermes_session_ids(conn: sqlite3.Connection, rid: str) -> list[str]:
        for root_session in (f"merchantbench-{rid}", f"realshop-{rid}"):
            rows = conn.execute(
                """
                WITH RECURSIVE lineage(id) AS (
                    SELECT id FROM sessions WHERE id = ?
                    UNION ALL
                    SELECT s.id
                    FROM sessions s
                    JOIN lineage l ON s.parent_session_id = l.id
                )
                SELECT id FROM lineage
                """,
                (root_session,),
            ).fetchall()
            session_ids = [str(row["id"]) for row in rows]
            if session_ids:
                return session_ids
        rows = conn.execute(
            "SELECT id FROM sessions WHERE source IN ('merchantbench', 'realshop') "
            "ORDER BY started_at"
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def _hermes_message_from_row(row: sqlite3.Row) -> dict:
        msg = {
            "role": row["role"],
            "content": row["content"],
            "hermes_message_id": row["id"],
            "hermes_session_id": row["session_id"],
            "active": bool(row["active"]),
            "compacted": bool(row["compacted"]),
            "trace_source": "hermes",
        }
        if row["tool_call_id"]:
            msg["tool_call_id"] = row["tool_call_id"]
        if row["tool_name"]:
            msg["name"] = row["tool_name"]
            msg["tool_origin"] = _hermes_tool_origin(row["tool_name"])
        tool_calls = _parse_hermes_tool_calls(row["tool_calls"])
        if tool_calls:
            msg["tool_calls"] = tool_calls
            origins = {_tool_call_origin(msg, tc) for tc in tool_calls}
            if len(origins) == 1:
                msg["tool_origin"] = next(iter(origins))
            elif origins:
                msg["tool_origin"] = "mixed"
        if row["reasoning"]:
            msg["reasoning"] = row["reasoning"]
        if row["reasoning_content"]:
            msg["reasoning_content"] = row["reasoning_content"]
        if row["reasoning_details"]:
            msg["reasoning_details"] = row["reasoning_details"]
        return msg

    def _hermes_message_stats(conn: sqlite3.Connection, session_ids: list[str]) -> dict:
        placeholders = ",".join("?" for _ in session_ids)
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_raw,
                SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS total_active,
                SUM(CASE WHEN active = 0 OR compacted = 1 THEN 1 ELSE 0 END) AS total_compacted
            FROM messages
            WHERE session_id IN ({placeholders})
            """,
            session_ids,
        ).fetchone()
        return {
            "total_raw": int(row["total_raw"] or 0),
            "total_active": int(row["total_active"] or 0),
            "total_compacted": int(row["total_compacted"] or 0),
        }

    def _hermes_session_system_prompt(
        conn: sqlite3.Connection,
        session_ids: list[str],
    ) -> Optional[str]:
        for session_id in session_ids:
            try:
                row = conn.execute(
                    "SELECT system_prompt FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            except sqlite3.Error:
                return None
            if row and row["system_prompt"]:
                return str(row["system_prompt"])
        return None

    def _hermes_system_prompt_message(system_prompt: Optional[str]) -> Optional[dict]:
        if not system_prompt:
            return None
        return {
            "role": "system",
            "content": str(system_prompt),
            "trace_source": "hermes",
            "active": True,
            "compacted": False,
        }

    def _read_hermes_trace_index(
        runs_root: str,
        rid: str,
        *,
        include_compacted: bool = False,
    ) -> dict:
        db_path = _hermes_state_db_path(runs_root, rid)
        empty = {"steps": [], "total": 0, "total_raw": 0,
                 "total_active": 0, "total_compacted": 0}
        if not os.path.exists(db_path):
            return empty
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            session_ids = _hermes_session_ids(conn, rid)
            if not session_ids:
                return empty
            stats = _hermes_message_stats(conn, session_ids)
            system_prompt = _hermes_session_system_prompt(conn, session_ids)
            placeholders = ",".join("?" for _ in session_ids)
            active_filter = "" if include_compacted else "AND active = 1"
            rows = conn.execute(
                f"""
                SELECT role, content
                FROM messages
                WHERE session_id IN ({placeholders}) {active_filter}
                ORDER BY timestamp, id
                """,
                session_ids,
            ).fetchall()
        except sqlite3.Error:
            return empty
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        steps_by_t: dict[int, int] = {}
        current_t: Optional[int] = None
        for row in rows:
            observed_t = (
                _hermes_observation_t(row["content"] or "")
                if row["role"] == "user"
                else None
            )
            if observed_t is not None:
                current_t = observed_t
            if current_t is None:
                continue
            steps_by_t[current_t] = steps_by_t.get(current_t, 0) + 1

        context_by_step = _env_context_summary_by_step(runs_root, rid)
        steps = []
        system_prompt_t = min(steps_by_t) if system_prompt and steps_by_t else None
        for t, n in sorted(steps_by_t.items()):
            if not n:
                continue
            display_n = n + (1 if t == system_prompt_t else 0)
            item = {"t": t, "n": display_n}
            if t in context_by_step:
                item["context"] = context_by_step[t]
            steps.append(item)
        return {
            "steps": steps,
            "total": sum(step["n"] for step in steps),
            **stats,
        }

    def _read_hermes_trace_steps(
        runs_root: str,
        rid: str,
        *,
        include_compacted: bool = False,
    ) -> list[dict]:
        db_path = _hermes_state_db_path(runs_root, rid)
        if not os.path.exists(db_path):
            return []
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            session_ids = _hermes_session_ids(conn, rid)
            if not session_ids:
                return []
            system_prompt = _hermes_session_system_prompt(conn, session_ids)
            placeholders = ",".join("?" for _ in session_ids)
            active_filter = "" if include_compacted else "AND active = 1"
            rows = conn.execute(
                f"""
                SELECT id, session_id, role, content, tool_call_id, tool_calls,
                       tool_name, reasoning, reasoning_content, reasoning_details,
                       active, compacted, timestamp
                FROM messages
                WHERE session_id IN ({placeholders}) {active_filter}
                ORDER BY timestamp, id
                """,
                session_ids,
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        steps_by_t: dict[int, list[dict]] = {}
        current_t: Optional[int] = None
        for row in rows:
            msg = _hermes_message_from_row(row)
            observed_t = (
                _hermes_observation_t(msg.get("content") or "")
                if msg.get("role") == "user"
                else None
            )
            if observed_t is not None:
                current_t = observed_t
            if current_t is None:
                continue
            msg["hermes_step_t"] = current_t
            steps_by_t.setdefault(current_t, []).append(msg)

        steps = []
        system_prompt_msg = _hermes_system_prompt_message(system_prompt)
        system_prompt_t = min(steps_by_t) if system_prompt_msg and steps_by_t else None
        for t, messages in sorted(steps_by_t.items()):
            display_messages = (
                [system_prompt_msg, *messages]
                if t == system_prompt_t
                else messages
            )
            turns = _env_turns_for_step(runs_root, rid, t)
            n_turns = (
                len(turns)
                if turns
                else sum(1 for msg in messages if msg.get("role") == "assistant")
            )
            steps.append({
                "t": t,
                "n_turns": n_turns,
                "messages": display_messages,
                "message_agents": ["agent_0"] * len(display_messages),
                "turns": turns,
                "trace_source": "hermes",
            })
        return steps

    # ---------- register ----------

    @bp.post("/runs/<run_id>/agent/register")
    def register_agent(run_id):
        env = env_of(run_id)
        body = request.get_json(force=True, silent=True) or {}
        agent_id = body.get("agent_id")
        if not agent_id:
            return jsonify({"ok": False, "error": "agent_id is required"}), 400
        payload = {
            "agent_id": agent_id,
            "framework": body.get("framework", "unknown"),
            "model": body.get("model"),
            "prompt_template": body.get("prompt_template"),
            "version": body.get("version"),
            "extra": body.get("extra", {}),
        }
        agent_log.write_meta(env.runs_root, env.run_id, payload)
        agent_log.init_runtime_events(
            env.runs_root,
            env.run_id,
            agent_id=agent_id,
            t=env.t,
        )
        return jsonify({"ok": True, **payload})

    @bp.get("/runs/<run_id>/agent/meta")
    def get_meta(run_id):
        runs_root, rid = _resolve_run_dir(run_id)
        return jsonify(agent_log.read_meta(runs_root, rid))

    # ---------- tool schema ----------

    @bp.get("/runs/<run_id>/tools/schema")
    def get_tools_schema(run_id):
        env = env_of(run_id)
        deny = (env.scenario.get("agent", {}) or {}).get("tool_denylist")
        specs = registry.all_specs(deny)
        out = []
        for s in specs:
            out.append({
                "name": s.name,
                "description": s.description,
                "parameters": registry.parameters_for_env(s, env),
                "examples": s.examples,
                "mutating": s.mutating,
                "openai": registry.openai_schema_for_env(s, env),
            })
        run_meta = snap.read_meta(env.runs_root, env.run_id) or {}
        schema_hash = tool_schema_sha256(item["openai"] for item in out)
        return jsonify({
            "tools": out,
            "base_path": f"/runs/{run_id}/agents/<agent_id>/act",
            "protocol": {
                "name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "legacy_input_names": ["realshop"],
            },
            "tool_schema_sha256": schema_hash,
            "scenario_id": (
                env.scenario.get("scenario_id")
                or run_meta.get("scenario_id")
                or "default"
            ),
            "dataset": {
                "id": run_meta.get("dataset_id", run_meta.get("data_source", "unknown")),
                "sha256": run_meta.get("dataset_sha256", ""),
                "rows": run_meta.get("dataset_rows"),
            },
        })

    # ---------- observation ----------

    @bp.get("/runs/<run_id>/agents/<agent_id>/observation")
    def get_observation(run_id, agent_id):
        env = env_of(run_id)
        if agent_id not in env.agents:
            return jsonify({"ok": False, "error": f"unknown agent {agent_id}"}), 404
        if "t" in request.args:
            try:
                tn = int(request.args["t"])
            except ValueError:
                return jsonify({"ok": False, "error": "t must be int"}), 400
            idx = agent_log.read_step_index(env.runs_root, env.run_id, tn)
            if idx is None:
                return jsonify({"ok": False, "error": "no agent log for that step"}), 404
            return jsonify(_filter_step_index_for_agent(
                idx,
                agent_id,
                allow_legacy_agent0=len(env.agents) == 1,
            ))
        if not env.agents[agent_id].is_alive:
            return jsonify(_dead_agent_payload(env, agent_id)), 410
        nowait = request.args.get("nowait") in ("1", "true", "yes")
        if not nowait:
            try:
                timeout = float(request.args.get("timeout", 30.0))
            except ValueError:
                timeout = 30.0
            deadline = _time.time() + max(0.0, timeout)
            with env.hook_cond:
                while not env.hook_open:
                    if not env.agents[agent_id].is_alive:
                        return jsonify(_dead_agent_payload(env, agent_id)), 410
                    if env.finished:
                        return jsonify({"ok": False, "error": "run_finished"}), 410
                    remaining = deadline - _time.time()
                    if remaining <= 0:
                        return jsonify({"ok": False, "error": "hook_not_open_within_timeout",
                                         "timeout": timeout}), 408
                    env.hook_cond.wait(timeout=remaining)
        if not env.agents[agent_id].is_alive:
            return jsonify(_dead_agent_payload(env, agent_id)), 410
        include_brief = agent_id not in env.brief_served_by_agent
        mark_observed = (not nowait) or bool(env.hook_open)
        packet = obs_mod.compose_observation(env, agent_id,
                                             include_brief=include_brief,
                                             mark_observed=mark_observed)
        if include_brief:
            env.brief_served_by_agent.add(agent_id)
        with env.turn_lock:
            env._ensure_agent_protocol_state(agent_id)
            turn_count = len(env.turns_meta_by_agent.get(agent_id, []))
            if env.observation_packet is None:
                env.observation_packet = packet
            env.observation_packet_by_agent[agent_id] = packet
            if include_brief and packet.get("brief"):
                env._record_agent_message(
                    agent_id,
                    {"role": "system", "content": packet["brief"]["system_prompt"]},
                )
            env._record_agent_message(
                agent_id,
                {"role": "user", "content": packet.get("text", "")},
            )
        agent_facing = {
            "text": packet.get("text", ""),
            "tick": packet.get("tick", {}),
            "turn_count": turn_count,
        }
        if include_brief and packet.get("brief"):
            agent_facing["brief"] = packet["brief"]
        return jsonify(agent_facing)

    @bp.get("/runs/<run_id>/agents/<agent_id>/trace_index")
    def get_agent_trace_index(run_id, agent_id):
        """Return a message-count index scoped to one agent's trace."""
        scope = _resolve_agent_trace_scope(run_id, agent_id)
        if scope is None:
            return jsonify({"ok": False, "error": f"unknown agent {agent_id}"}), 404
        runs_root, rid, allow_legacy_agent0 = scope
        base = agent_log.agent_dir(runs_root, rid)
        by_step_dir = os.path.join(base, "by_step")
        steps = []
        total = 0
        if not os.path.isdir(by_step_dir):
            return jsonify({"steps": [], "total": 0})
        for fname in sorted(os.listdir(by_step_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(by_step_dir, fname)
            try:
                with open(path) as trace_file:
                    step_data = json.load(trace_file)
            except (json.JSONDecodeError, OSError):
                continue
            filtered = _filter_step_index_for_agent(
                step_data,
                agent_id,
                allow_legacy_agent0=allow_legacy_agent0,
            )
            n_messages = len(filtered.get("messages") or [])
            if not n_messages:
                continue
            item = {"t": filtered.get("t", 0), "n": n_messages}
            context = _context_summary_from_turns(filtered.get("turns") or [])
            if context:
                item["context"] = context
            steps.append(item)
            total += n_messages
        return jsonify({"steps": steps, "total": total})

    @bp.get("/runs/<run_id>/agents/<agent_id>/trace")
    def get_agent_trace(run_id, agent_id):
        """Return one disk-backed trace step scoped to a persisted agent."""
        scope = _resolve_agent_trace_scope(run_id, agent_id)
        if scope is None:
            return jsonify({"ok": False, "error": f"unknown agent {agent_id}"}), 404
        if "t" not in request.args:
            return jsonify({
                "turns": [],
                "count": 0,
                "hint": "pass ?t=N to read a specific step",
            })
        try:
            t_arg = int(request.args["t"])
        except ValueError:
            return jsonify({"ok": False, "error": "t must be int"}), 400
        runs_root, rid, allow_legacy_agent0 = scope
        idx = agent_log.read_step_index(runs_root, rid, t_arg)
        if idx is None:
            return jsonify({
                "ok": False,
                "error": "no agent log for that step",
            }), 404
        return jsonify(_filter_step_index_for_agent(
            idx,
            agent_id,
            allow_legacy_agent0=allow_legacy_agent0,
        ))

    # ---------- act (unified tool execution + trace) ----------

    @bp.post("/runs/<run_id>/agents/<agent_id>/usage")
    def record_agent_auxiliary_usage(run_id, agent_id):
        """Record delayed, non-turn LLM usage such as checkpoint review.

        This endpoint intentionally does not require an open hook: a review is
        launched after its foreground turn finishes and the final review can
        complete after the last agent hook has closed.
        """
        env = env_of(run_id)
        if agent_id not in env.agents:
            return jsonify({"ok": False, "error": f"unknown agent {agent_id}"}), 404
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "request body must be an object"}), 400
        usage_id = str(body.get("usage_id") or "").strip()
        source = str(body.get("source") or "auxiliary").strip()
        model = str(body.get("model") or "").strip()
        provider = str(body.get("provider") or "").strip()
        cost_status = str(body.get("cost_status") or "").strip()
        cost_source = str(body.get("cost_source") or "").strip()
        if not usage_id:
            return jsonify({"ok": False, "error": "usage_id is required"}), 400
        if not source:
            return jsonify({"ok": False, "error": "source is required"}), 400
        if (
            len(usage_id) > 200
            or len(source) > 100
            or len(model) > 200
            or len(provider) > 100
            or len(cost_status) > 100
            or len(cost_source) > 200
        ):
            return jsonify({"ok": False, "error": "usage metadata too long"}), 400
        token_usage = body.get("token_usage")
        if not isinstance(token_usage, dict):
            return jsonify({"ok": False, "error": "token_usage is required"}), 400
        try:
            usage_step = int(body.get("step", env.t))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "step must be int"}), 400
        if usage_step < 0 or usage_step > int(env.t):
            return jsonify({"ok": False, "error": "step out of range"}), 400
        meta = agent_log.read_meta(env.runs_root, run_id)
        foreground_model = next(
            (
                str(record.get("model") or "")
                for record in (meta.get("agents") or [])
                if str(record.get("agent_id") or "") == str(agent_id)
            ),
            "",
        )
        use_scenario_pricing = bool(
            model and foreground_model and model == foreground_model
        )
        result = agent_log.record_auxiliary_usage(
            env.runs_root,
            run_id,
            agent_id,
            f"{agent_id}:{usage_id}",
            token_usage,
            t=usage_step,
            source=source,
            pricing=(env.scenario.get("agent", {}) or {}).get("cost_pricing"),
            model=model,
            provider=provider,
            cost_usd=body.get("cost_usd"),
            cost_status=cost_status,
            cost_source=cost_source,
            use_scenario_pricing=use_scenario_pricing,
        )
        if result.get("error") == "usage_id_conflict":
            status = 409
        elif result.get("error"):
            status = 400
        else:
            status = 200
        return jsonify(result), status

    @bp.post("/runs/<run_id>/agents/<agent_id>/act")
    def act(run_id, agent_id):
        env = env_of(run_id)
        if agent_id not in env.agents:
            return jsonify({"ok": False, "error": f"unknown agent {agent_id}"}), 404

        @after_this_request
        def record_failed_act(response):
            if int(response.status_code or 0) < 400:
                return response
            body = response.get_json(silent=True)
            error = body.get("error") if isinstance(body, dict) else None
            try:
                agent_log.record_runtime_event(
                    env.runs_root,
                    env.run_id,
                    agent_id=agent_id,
                    t=env.t,
                    event_type=API_FAILED_EVENT,
                    payload={"status": int(response.status_code), "error": error},
                )
            except OSError:
                # The run may have been deleted after request dispatch.  Do not
                # replace the original protocol error with a telemetry error.
                pass
            return response

        if not env.agents[agent_id].is_alive:
            return jsonify(_dead_agent_payload(env, agent_id)), 410
        blocked = check_stale_step(env)
        if blocked is not None:
            return blocked
        body = request.get_json(force=True, silent=True) or {}
        messages = body.get("messages", [])
        token_usage = body.get("token_usage")
        context = _normalize_context(body.get("context"))

        if not messages:
            return jsonify({"ok": False, "error": "messages is required"}), 400

        completed_env_tool_call_indexes: dict[str, list[int]] = {}
        for message_idx, raw_msg in enumerate(messages):
            if not isinstance(raw_msg, dict) or raw_msg.get("role") != "tool":
                continue
            if not is_env_tool_origin(raw_msg.get("tool_origin")):
                continue
            tc_id = raw_msg.get("tool_call_id")
            if tc_id:
                completed_env_tool_call_indexes.setdefault(str(tc_id), []).append(message_idx)

        normalized_messages = []
        env_tool_calls = []
        seen_tc_ids = set()
        for message_idx, raw_msg in enumerate(messages):
            if not isinstance(raw_msg, dict):
                return jsonify({"ok": False, "error": "messages entries must be objects"}), 400
            role = raw_msg.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                return jsonify({"ok": False, "error": "messages must use OpenAI message roles"}), 400
            msg = dict(raw_msg)
            if role == "assistant":
                explicit_msg_origin = msg.get("tool_origin")
                tool_calls = msg.get("tool_calls", [])
                if tool_calls is None:
                    tool_calls = []
                if not isinstance(tool_calls, list):
                    return jsonify({"ok": False, "error": "assistant tool_calls must be a list"}), 400
                normalized_tool_calls = []
                call_origins = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        return jsonify({"ok": False, "error": "tool_call entries must be objects"}), 400
                    tc = dict(tc)
                    origin = canonical_tool_origin(
                        tc.get("tool_origin") or explicit_msg_origin,
                        default=ENV_TOOL_ORIGIN,
                    )
                    tc["tool_origin"] = origin
                    call_origins.append(origin)
                    tc_id = tc.get("id")
                    if not tc_id:
                        return jsonify({"ok": False, "error": "tool_call id is required"}), 400
                    if tc_id in seen_tc_ids:
                        return jsonify({"ok": False, "error": f"duplicate tool_call id: {tc_id}"}), 400
                    seen_tc_ids.add(tc_id)
                    normalized_tool_calls.append(tc)
                    has_recorded_result = any(
                        result_idx > message_idx
                        for result_idx in completed_env_tool_call_indexes.get(str(tc_id), [])
                    )
                    if origin == ENV_TOOL_ORIGIN and not has_recorded_result:
                        env_tool_calls.append((message_idx, tc))
                msg["tool_calls"] = normalized_tool_calls
                if explicit_msg_origin:
                    msg["tool_origin"] = canonical_tool_origin(explicit_msg_origin)
                elif call_origins:
                    unique_origins = set(call_origins)
                    if len(unique_origins) == 1:
                        msg["tool_origin"] = call_origins[0]
                    else:
                        msg.pop("tool_origin", None)
                else:
                    msg["tool_origin"] = ENV_TOOL_ORIGIN
            elif role == "tool":
                if not msg.get("tool_origin"):
                    return jsonify({"ok": False,
                                    "error": "trace tool messages must include tool_origin"}), 400
                msg["tool_origin"] = canonical_tool_origin(msg.get("tool_origin"))
                if is_env_tool_origin(msg.get("tool_origin")) and not msg.get("runtime_execution_id"):
                    # Tool messages supplied by the client describe already
                    # completed history; they were not executed by this request.
                    msg["runtime_historical"] = True
            normalized_messages.append(msg)

        tool_results = []
        generated_tool_msgs_by_message_idx: dict[int, list[dict]] = {}
        step_done = False
        hook_released = False

        tool_names = [tc.get("function", {}).get("name") for _, tc in env_tool_calls]
        has_end_of_step = "end_of_step" in tool_names
        if has_end_of_step and tool_names[-1] != "end_of_step":
            return jsonify({"ok": False, "error": "end_of_step_must_be_last"}), 400

        # Gate: hook must be open unless this is a pure end_of_step request.
        only_eos = has_end_of_step and len(env_tool_calls) == 1
        deny = set((env.scenario.get("agent", {}) or {}).get("tool_denylist") or [])

        with env.lock:
            if not env.agents[agent_id].is_alive:
                return jsonify(_dead_agent_payload(env, agent_id)), 410
            blocked = check_stale_step(env)
            if blocked is not None:
                return blocked
            if not env.hook_open and not only_eos:
                return jsonify({"ok": False, "error": "hook_closed",
                                 "hint": "wait for GET /observation"}), 425
            # * Pure end_of_step is protocol control, not agent work: allow it
            #   past turn quota so the hook closes instead of burning
            #   max_hook_seconds after the agent exhausted action turns.
            quota_error = _turn_quota_error(env, agent_id)
            if quota_error is not None and not only_eos:
                return jsonify(quota_error), 429

            runtime_turn_idx = len(env.turns_meta_by_agent.get(agent_id, []))
            planned_calls = []
            for message_idx, tc in env_tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args_dict = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError) as e:
                    args_dict = None
                tc_id = tc.get("id", "")
                spec = registry.get(tool_name)
                idem_key = (
                    f"{agent_id}:{env.t}:{tc_id}"
                    if spec and spec.mutating else None
                )
                idem_fingerprint = (
                    {"tool_name": tool_name, "arguments": _canonical_json(args_dict)}
                    if idem_key and args_dict is not None else None
                )
                if _idempotency_conflict(env, idem_key, idem_fingerprint):
                    return jsonify({"ok": False, "error": "idempotency_conflict"}), 409
                planned_calls.append((message_idx, tc, tool_name, args_dict, idem_key, idem_fingerprint))

            for message_idx, tc, tool_name, args_dict, idem_key, idem_fingerprint in planned_calls:
                func = tc.get("function", {})
                tc_id = tc.get("id", "")
                if args_dict is None:
                    result = {"ok": False,
                              "error": {
                                  "code": "invalid_arguments",
                                  "path": "$",
                                  "message": f"invalid JSON in arguments: {func.get('arguments', '{}')}",
                              }}
                elif tool_name == "end_of_step":
                    if tool_name in deny:
                        result = {"ok": False,
                                  "error": f"tool '{tool_name}' is not available in this scenario"}
                    else:
                        result = tool_impl.end_of_step_result(env)
                        step_done = True
                        hook_released = env.mark_agent_step_done(agent_id)
                else:
                    result = dispatch_tool(env, agent_id, tool_name, args_dict,
                                           idempotency_key=idem_key,
                                           idempotency_fingerprint=idem_fingerprint,
                                           denylist=deny)
                    if isinstance(result, dict) and result.get("_http_status") == 409:
                        return jsonify({"ok": False, "error": "idempotency_conflict"}), 409
                content = json.dumps(result, ensure_ascii=False, default=str)
                tool_results.append({
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "tool_origin": ENV_TOOL_ORIGIN,
                    "content": content,
                })
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "tool_origin": ENV_TOOL_ORIGIN,
                    "content": content,
                    "runtime_execution_id": (
                        f"{agent_id}:{env.t}:{runtime_turn_idx}:{tc_id}"
                    ),
                }
                generated_tool_msgs_by_message_idx.setdefault(message_idx, []).append(tool_msg)
                if step_done:
                    break

            recorded_messages = []
            for message_idx, msg in enumerate(normalized_messages):
                recorded_messages.append(msg)
                recorded_messages.extend(generated_tool_msgs_by_message_idx.get(message_idx, []))

            record_result = env.record_act(
                agent_id,
                {},
                [],
                token_usage,
                recorded_messages=recorded_messages,
                context=context,
                ignore_turn_quota=only_eos,
            )
            if hook_released:
                env.hook_event.set()
        if not record_result.get("ok"):
            return jsonify(record_result), 429

        return jsonify({
            "ok": True,
            "turn_idx": record_result["turn_idx"],
            "tool_results": tool_results,
            "step_done": step_done,
            "hook_released": hook_released,
        })

    # ---------- read helpers (dashboard / debug) ----------

    @bp.get("/runs/<run_id>/agent/cost")
    def get_cost(run_id):
        runs_root, rid = _resolve_run_dir(run_id)
        return jsonify(agent_log.read_cost(runs_root, rid))

    @bp.get("/runs/<run_id>/agent/trace")
    def get_trace(run_id):
        runs_root, rid = _resolve_run_dir(run_id)
        if "t" not in request.args:
            return jsonify({"turns": [], "count": 0,
                            "hint": "pass ?t=N to read a specific step"})
        try:
            t_arg = int(request.args["t"])
        except ValueError:
            return jsonify({"ok": False, "error": "t must be int"}), 400
        idx = agent_log.read_step_index(runs_root, rid, t_arg)
        if idx is None:
            return jsonify({"turns": [], "count": 0})
        return jsonify(idx)

    @bp.get("/runs/<run_id>/agent/all_traces")
    def get_all_traces(run_id):
        """Return all messages across all steps, each annotated with its step number."""
        runs_root, rid = _resolve_run_dir(run_id)
        base = agent_log.agent_dir(runs_root, rid)
        by_step_dir = os.path.join(base, "by_step")
        steps = []
        if not os.path.isdir(by_step_dir):
            return jsonify({"steps": []})
        for fname in sorted(os.listdir(by_step_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(by_step_dir, fname)
            try:
                with open(path) as f:
                    step_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            t = step_data.get("t", 0)
            messages = step_data.get("messages", [])
            message_agents = step_data.get("message_agents")
            if messages:
                steps.append({
                    "t": t,
                    "messages": messages,
                    "message_agents": message_agents,
                })
        return jsonify({"steps": steps})

    @bp.get("/runs/<run_id>/agent/all_traces_index")
    def get_all_traces_index(run_id):
        """Lightweight index: just step numbers and message counts (no message bodies)."""
        runs_root, rid = _resolve_run_dir(run_id)
        base = agent_log.agent_dir(runs_root, rid)
        by_step_dir = os.path.join(base, "by_step")
        steps = []
        total = 0
        if not os.path.isdir(by_step_dir):
            return jsonify({"steps": [], "total": 0})
        for fname in sorted(os.listdir(by_step_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(by_step_dir, fname)
            try:
                with open(path) as f:
                    step_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            t = step_data.get("t", 0)
            n = len(step_data.get("messages", []))
            if n:
                item = {"t": t, "n": n}
                context = _context_summary_from_turns(step_data.get("turns", []))
                if context:
                    item["context"] = context
                steps.append(item)
                total += n
        return jsonify({"steps": steps, "total": total})

    @bp.get("/runs/<run_id>/agent/hermes_trace")
    def get_hermes_trace(run_id):
        """Return Hermes SessionDB messages for one parsed MerchantBench step."""
        runs_root, rid = _resolve_run_dir(run_id)
        if "t" not in request.args:
            return jsonify({"turns": [], "count": 0,
                            "hint": "pass ?t=N to read a specific step"})
        try:
            t_arg = int(request.args["t"])
        except ValueError:
            return jsonify({"ok": False, "error": "t must be int"}), 400
        include_compacted = _include_compacted_hermes_messages()
        steps = _read_hermes_trace_steps(
            runs_root, rid, include_compacted=include_compacted)
        for step in steps:
            if int(step.get("t", -1)) == t_arg:
                step["include_compacted"] = include_compacted
                return jsonify(step)
        return jsonify({"t": t_arg, "messages": [], "message_agents": [],
                        "turns": [], "n_turns": 0, "trace_source": "hermes",
                        "include_compacted": include_compacted})

    @bp.get("/runs/<run_id>/agent/hermes_all_traces")
    def get_hermes_all_traces(run_id):
        """Return Hermes SessionDB messages grouped by parsed MerchantBench step."""
        runs_root, rid = _resolve_run_dir(run_id)
        include_compacted = _include_compacted_hermes_messages()
        steps = _read_hermes_trace_steps(
            runs_root, rid, include_compacted=include_compacted)
        return jsonify({"steps": [
            {
                "t": step["t"],
                "messages": step.get("messages", []),
                "message_agents": step.get("message_agents"),
            }
            for step in steps
            if step.get("messages")
        ], "include_compacted": include_compacted})

    @bp.get("/runs/<run_id>/agent/hermes_all_traces_index")
    def get_hermes_all_traces_index(run_id):
        """Lightweight Hermes SessionDB step index."""
        runs_root, rid = _resolve_run_dir(run_id)
        include_compacted = _include_compacted_hermes_messages()
        return jsonify(_read_hermes_trace_index(
            runs_root, rid, include_compacted=include_compacted))

    @bp.get("/runs/<run_id>/agent/tool_calls")
    def get_tool_calls(run_id):
        """Return all agent tool calls across all steps (for dashboard events panel)."""
        runs_root, rid = _resolve_run_dir(run_id)
        base = agent_log.agent_dir(runs_root, rid)
        by_step_dir = os.path.join(base, "by_step")
        results = []
        if not os.path.isdir(by_step_dir):
            return jsonify(results)
        for fname in sorted(os.listdir(by_step_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(by_step_dir, fname)
            try:
                with open(path) as f:
                    step_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            t = step_data.get("t", 0)
            for msg in step_data.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls", []):
                    if not _is_merchantbench_env_tool_call(msg, tc):
                        continue
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if name and name != "end_of_step":
                        results.append({"t": t, "tool_name": name})
        return jsonify(results)

    return bp
