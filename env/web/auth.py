"""Optional bearer-token auth for hosted evaluation mode."""

from __future__ import annotations

import re
from typing import Any, Optional

from flask import current_app, jsonify, request

_RUN_RE = re.compile(r"^/runs/([^/]+)(?:/|$)")
_AGENT_ID_RE = re.compile(r"^/runs/[^/]+/agents/([^/]+)/(?:observation|act|usage)$")
_REGISTER_RE = re.compile(r"^/runs/[^/]+/agent/register$")
_AGENT_ALLOWED = (
    re.compile(r"^/runs/[^/]+/agent/register$"),
    re.compile(r"^/runs/[^/]+/tools/schema$"),
    re.compile(r"^/runs/[^/]+/agents/[^/]+/observation$"),
    re.compile(r"^/runs/[^/]+/agents/[^/]+/act$"),
    re.compile(r"^/runs/[^/]+/agents/[^/]+/usage$"),
)


def _bearer_token() -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if auth.startswith(prefix):
        return auth[len(prefix) :].strip()
    return request.headers.get("X-MerchantBench-Token") or request.headers.get("X-RealShop-Token")


def _run_id_from_path(path: str) -> Optional[str]:
    m = _RUN_RE.match(path)
    return m.group(1) if m else None


def _agent_id_from_path(path: str) -> Optional[str]:
    m = _AGENT_ID_RE.match(path)
    if m:
        return m.group(1)
    if _REGISTER_RE.match(path):
        body = request.get_json(silent=True) or {}
        agent_id = body.get("agent_id") if isinstance(body, dict) else None
        return str(agent_id) if agent_id else None
    return None


def _is_agent_allowed(path: str) -> bool:
    return any(pattern.match(path) for pattern in _AGENT_ALLOWED)


def _agent_token_matches(run_auth: dict[str, Any], token: str, path: str) -> bool:
    agent_id = _agent_id_from_path(path)
    agent_tokens = run_auth.get("agent_tokens")
    if isinstance(agent_tokens, dict):
        if agent_id:
            return token == agent_tokens.get(agent_id)
        return token in set(agent_tokens.values())
    # Legacy auth.json files had a single agent_token; keep that token limited
    # to the default agent instead of leaving it run-wide.
    legacy_token = run_auth.get("agent_token")
    if token != legacy_token:
        return False
    return agent_id in (None, "agent_0")


def enforce_optional_auth():
    require_tokens = current_app.config.get("MERCHANTBENCH_REQUIRE_TOKENS") or current_app.config.get(
        "REALSHOP_REQUIRE_TOKENS"
    )
    if not require_tokens:
        return None
    token = _bearer_token()
    if not token:
        return jsonify({"ok": False, "error": "auth_required"}), 401
    admin_token = current_app.config.get("MERCHANTBENCH_ADMIN_TOKEN") or current_app.config.get("REALSHOP_ADMIN_TOKEN")
    if admin_token and token == admin_token:
        return None
    run_id = _run_id_from_path(request.path)
    if not run_id:
        return jsonify({"ok": False, "error": "admin_token_required"}), 403
    registry = current_app.registry
    run_auth = registry.auth_for_run(run_id)
    if run_auth and _is_agent_allowed(request.path) and _agent_token_matches(run_auth, token, request.path):
        return None
    return jsonify({"ok": False, "error": "forbidden"}), 403
