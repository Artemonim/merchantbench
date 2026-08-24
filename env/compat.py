"""Backward-compatible names for the public MerchantBench protocol.

All new state is written with MerchantBench names. Legacy RealShop names are
accepted only at process boundaries and while reading existing run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Optional

PROTOCOL_NAME = "merchantbench"
PROTOCOL_VERSION = 2

ENV_TOOL_ORIGIN = "merchantbench_env"
LEGACY_ENV_TOOL_ORIGINS = frozenset({"realshop_env"})
ENV_TOOL_ORIGINS = frozenset({ENV_TOOL_ORIGIN, *LEGACY_ENV_TOOL_ORIGINS})

API_FAILED_EVENT = "merchantbench_api_failed_attempt"
LEGACY_API_FAILED_EVENTS = frozenset({"realshop_api_failed_attempt"})
API_FAILED_EVENTS = frozenset({API_FAILED_EVENT, *LEGACY_API_FAILED_EVENTS})

MEMORY_VERSION_MARKER = "<!-- merchantbench-memory-version "
LEGACY_MEMORY_VERSION_MARKERS = ("<!-- realshop-memory-version ",)


def env_value(primary: str, *legacy: str, default: Optional[str] = None) -> Optional[str]:
    """Return the canonical environment value, falling back to legacy names."""
    for name in (primary, *legacy):
        value = os.environ.get(name)
        if value:
            return value
    return default


def canonical_tool_origin(value: Any, *, default: Optional[str] = None) -> str:
    text = str(value or "")
    if text in LEGACY_ENV_TOOL_ORIGINS:
        return ENV_TOOL_ORIGIN
    if text:
        return text
    return str(default or "")


def is_env_tool_origin(value: Any) -> bool:
    return str(value or "") in ENV_TOOL_ORIGINS


def is_api_failed_event(value: Any) -> bool:
    return str(value or "") in API_FAILED_EVENTS


def tool_schema_sha256(openai_schemas: Iterable[dict]) -> str:
    payload = json.dumps(
        list(openai_schemas),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
