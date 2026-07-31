"""Simulation time projection helpers.

The simulator core stores raw integer ticks. Agent-facing APIs continue to
expose legacy ``day`` and ``hour`` fields, and optionally add a compact ISO
``datetime`` field when ``run.virtual_time.enabled`` is true.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional


PRIVATE_REAL_DEFAULT_ANCHOR_DATE = date(2025, 6, 1)
def legacy_day_hour(t: int, step_hours: int) -> dict:
    sim_hour = int(t) * int(step_hours)
    return {"day": sim_hour // 24 + 1, "hour": sim_hour % 24}


def _parse_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


def virtual_time_config(scenario: dict) -> dict:
    return ((scenario.get("run") or {}).get("virtual_time") or {})


def virtual_time_enabled(scenario: dict) -> bool:
    cfg = virtual_time_config(scenario)
    return bool(cfg.get("enabled") and cfg.get("start_date"))


def start_date(scenario: dict) -> Optional[date]:
    cfg = virtual_time_config(scenario)
    if not cfg.get("enabled") or not cfg.get("start_date"):
        return None
    return _parse_date(cfg["start_date"])


def data_anchor_date(scenario: dict) -> Optional[date]:
    data_cfg = scenario.get("data") or {}
    explicit = data_cfg.get("calendar_anchor_date")
    if explicit:
        return _parse_date(explicit)
    if str(data_cfg.get("source", "synthetic") or "synthetic") == "private_real":
        return PRIVATE_REAL_DEFAULT_ANCHOR_DATE
    return None


def demand_day_offset(scenario: dict) -> int:
    """Return the market-curve day offset implied by virtual calendar time.

    Synthetic data keeps the historical behavior unless a caller explicitly
    supplies ``data.calendar_anchor_date``.
    """
    start = start_date(scenario)
    anchor = data_anchor_date(scenario)
    if start is None or anchor is None:
        return 0
    return (start - anchor).days


def curve_day_index(scenario: dict, t: int, step_hours: int) -> int:
    day = int((int(t) * int(step_hours)) // 24)
    return (day + demand_day_offset(scenario)) % 365


def time_view(scenario: dict, t: int, step_hours: int) -> dict:
    """Project a raw tick to the agent/dashboard time shape.

    Off/default mode intentionally returns exactly ``{"day": ..., "hour": ...}``.
    Virtual-time mode keeps day/hour for tool arguments and adds only an ISO
    datetime to avoid repeating date/weekday variants in every tool result.
    """
    base = legacy_day_hour(t, step_hours)
    start = start_date(scenario)
    if start is None:
        return base

    sim_hour = int(t) * int(step_hours)
    dt = datetime.combine(start, time.min) + timedelta(hours=sim_hour)
    out = dict(base)
    out["datetime"] = dt.isoformat(timespec="seconds")
    return out
