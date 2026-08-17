"""Leaderboard helpers derived from completed run data.

Paper-facing aggregate definitions:
  - net_profit_margin = final cumulative net profit / final cumulative GMV
  - contribution_margin_pct = (GMV - COGS - fee_total) / GMV * 100, or 0
    when GMV is 0. fee_total is commission + logistics + reverse fulfillment.
    Fines are excluded. Distinct from net_profit_margin.
  - order_anomaly_rate = orders with a realized late/cancel/refund/bad-review/
    stockout/insufficient-balance outcome / all orders
  - average_active_listings = arithmetic mean of per-step active-listing samples
    within the configured evaluation horizon (excluding draining)
  - effective_window_rate = hook windows with at least one non-end environment
    tool call / scheduled hook windows
  - total_tool_calls = all MerchantBench environment tool calls, including
    ``end_of_step`` and excluding native/non-environment tools
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
import re
import sqlite3
import threading
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Optional

import yaml

log = logging.getLogger(__name__)

from core import listing_rating as lr_mod
from core.economy_v6 import contribution_margin_pct
from storage import agent_log
from storage import db as dbm
from storage import snapshot as snap

RUN_RESULT_STATUSES = {"finished", "stopped", "error"}
LIVE_RUN_RESULT_STATUSES = {"running", "paused", "draining"}
FRAMEWORK_LABELS = {
    "react": "React",
    "react_160k_compact_30k": "React",
    "hermes": "Hermes",
    "rule_based": "Rule-based",
    "auto_seed": "Auto Seed",
    "human": "Human",
    "none": "None",
}
RUN_COLOR_PALETTE = [
    "#2F9E44", "#168AAD", "#F08C00", "#1971C2", "#E03131",
    "#7048E8", "#C2255C", "#5C940D", "#0B7285", "#495057",
    "#CA6702", "#862E9C", "#087F5B", "#364FC7", "#A61E4D",
]
FRAMEWORK_COLOR_BY_KEY = {
    "react": "#2F9E44",
    "react_160k_compact_30k": "#168AAD",
    "hermes": "#1971C2",
    "rule_based": "#F08C00",
    "auto_seed": "#F08C00",
    "human": "#7048E8",
    "none": "#6B7280",
}
MODEL_BOOTSTRAP_KEYS = {"react", "react_160k_compact_30k", "hermes"}
ENV_TOOL_ORIGIN = "merchantbench_env"
ORDER_ANOMALY_STATUSES = (
    "cancelled",
    "stockout",
    "insufficient_balance",
    "settled_refund",
    "settled_only_refund",
    "settled_bad_review",
)
_ORDER_ANOMALY_STATUS_SQL = ",".join(
    f"'{status}'" for status in ORDER_ANOMALY_STATUSES
)
ORDER_ANOMALY_SQL_CONDITION = (
    f"(late_t IS NOT NULL OR current_status IN ({_ORDER_ANOMALY_STATUS_SQL}))"
)
LISTING_ACTION_TOOL_SPECS = [
    {
        "key": "list",
        "label": "List",
        "metric": "listing_action_list_calls",
        "tool": "list_product",
    },
    {
        "key": "delist",
        "label": "Delist",
        "metric": "listing_action_delist_calls",
        "tool": "delist_product",
    },
    {
        "key": "price",
        "label": "Price",
        "metric": "listing_action_price_calls",
        "tool": "adjust_price",
    },
]
SOURCING_TOOL_CALL_SPECS = [
    {
        "key": "market_brief",
        "label": "Market Brief",
        "metric": "market_brief_calls",
        "tool": "market_brief",
    },
    {
        "key": "hot_search_terms",
        "label": "Hot Search Terms",
        "metric": "hot_search_terms_calls",
        "tool": "hot_search_terms",
    },
    {
        "key": "search",
        "label": "Search",
        "metric": "search_products_calls",
        "tool": "search_products",
    },
    {
        "key": "daily_report",
        "label": "Daily Report",
        "metric": "get_daily_report_calls",
        "tool": "get_daily_report",
    },
    {
        "key": "product_detail",
        "label": "Product Detail",
        "metric": "get_product_detail_calls",
        "tool": "get_product_detail",
    },
    {
        "key": "supplier_profile",
        "label": "Supplier Profile",
        "metric": "get_supplier_profile_calls",
        "tool": "get_supplier_profile",
    },
    {
        "key": "supplier_products",
        "label": "Supplier Products",
        "metric": "list_supplier_products_calls",
        "tool": "list_supplier_products",
    },
]
RUNTIME_HEALTH_METRICS = (
    "abnormal_ended_windows",
    "api_failed_attempts",
    "tool_call_failures",
    "retry_exhausted",
    "memory_compactions",
    "skills_evolutions",
)
RUNTIME_HEALTH_COVERAGE_STATES = {"complete", "partial", "unavailable"}
LEADERBOARD_MAX_SERIES_POINTS = 500
_HERMES_API_FAILURE_RE = re.compile(r"API call failed \(attempt (\d+)/(\d+)\)")
_HERMES_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
_HERMES_THREAD_RE = re.compile(r"\bthread=([^\s]+)")
# Legacy listing tools that are no longer exposed in the UI but must still be
# classified as listing/pricing and counted in the aggregate listing_action_calls.
_LEGACY_LISTING_TOOL_NAMES = {"set_promised_ship_hours"}
LISTING_TOOL_NAMES = {spec["tool"] for spec in LISTING_ACTION_TOOL_SPECS} | _LEGACY_LISTING_TOOL_NAMES
LISTING_UI_TOOL_NAMES = {spec["tool"] for spec in LISTING_ACTION_TOOL_SPECS}
LISTING_TOOL_ORDER = {
    spec["tool"]: idx
    for idx, spec in enumerate(LISTING_ACTION_TOOL_SPECS)
}
AVERAGE_PRODUCT_PRICE_KEYS = [
    "avg_listing_sale_price",
    "avg_listing_sale_price_count",
]
AVERAGE_PRODUCT_MARGIN_KEYS = [
    "avg_listing_margin_ratio",
    "avg_listing_margin_ratio_count",
    # Legacy amount metrics are retained so completed runs can be converted to
    # a ratio using their average listing price.
    "avg_listing_margin",
    "avg_listing_margin_count",
    "avg_listing_sale_price",
]
AVERAGE_PRODUCT_RATING_KEYS = [
    "avg_listing_rating",
    "avg_listing_rating_count",
]
TOOL_CATEGORY_SPECS = [
    {
        "key": "sourcing",
        "label": "Sourcing",
        "color": "#168AAD",
        "tools": {
            "market_brief",
            "hot_search_terms",
            "get_daily_report",
            "search_products",
            "get_product_detail",
            "get_supplier_profile",
            "list_supplier_products",
        },
    },
    {
        "key": "listing_pricing",
        "label": "Listing and Pricing",
        "color": "#2F9E44",
        "tools": LISTING_TOOL_NAMES | {
            "review_my_listings",
            "query_my_listings",
        },
    },
    {
        "key": "cash_flow",
        "label": "Cash-Flow",
        "color": "#F08C00",
        "tools": {
            "query_balance",
            "query_cash_pipeline",
            "query_store_performance",
            "query_product_sales_stats",
        },
    },
    {
        "key": "store_state",
        "label": "Store State",
        "color": "#1971C2",
        "tools": {
            "get_store_snapshot",
            "query_platform_rules",
            "query_my_orders",
            "query_open_orders",
            "query_order_updates",
            "query_order_detail",
            "query_supply_chain_anomalies",
            "get_observation",
            "list_tools",
        },
    },
    {
        "key": "memory",
        "label": "Memory",
        "color": "#7048E8",
        "tools": {"read_memory_doc", "write_memory_doc"},
    },
]
TOOL_CATEGORY_BY_NAME = {
    name: spec["key"]
    for spec in TOOL_CATEGORY_SPECS
    for name in spec["tools"]
}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _now_for_elapsed(start_dt: datetime) -> datetime:
    if start_dt.tzinfo is None:
        return datetime.now()
    return datetime.now(timezone.utc)


def _align_elapsed_clocks(start_dt: datetime, end_dt: datetime) -> tuple[datetime, datetime]:
    start_aware = start_dt.tzinfo is not None
    end_aware = end_dt.tzinfo is not None
    if start_aware == end_aware:
        return start_dt, end_dt
    if start_aware:
        start_dt = start_dt.astimezone().replace(tzinfo=None)
    if end_aware:
        end_dt = end_dt.astimezone().replace(tzinfo=None)
    return start_dt, end_dt


def _loads_json(value, fallback):
    try:
        out = json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return fallback
    return out if out is not None else fallback


def _loads_yaml(value, fallback):
    try:
        out = yaml.safe_load(value or "")
    except (TypeError, yaml.YAMLError):
        return fallback
    return out if out is not None else fallback


def _float_or(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _bootstrap_config(row: dict) -> dict:
    cfg = row.get("bootstrap_config")
    if isinstance(cfg, dict):
        return cfg
    return _loads_json(row.get("bootstrap_config_json"), {})


def _stable_run_color(run_id: Optional[str]) -> str:
    digest = hashlib.sha256(str(run_id or "").encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(RUN_COLOR_PALETTE)
    return RUN_COLOR_PALETTE[idx]


def _framework_color(framework_key: Optional[str]) -> str:
    if framework_key in FRAMEWORK_COLOR_BY_KEY:
        return FRAMEWORK_COLOR_BY_KEY[str(framework_key)]
    return _stable_run_color(framework_key)


def _step_hours_for_run(row: dict) -> float:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    run_cfg = (scenario.get("run") or {}) if isinstance(scenario, dict) else {}
    try:
        return float(run_cfg.get("step_hours") or row.get("step_hours") or 1)
    except (TypeError, ValueError):
        return 1.0


def _virtual_start_date_for_run(row: dict) -> Optional[str]:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    run_cfg = (scenario.get("run") or {}) if isinstance(scenario, dict) else {}
    virtual_time = run_cfg.get("virtual_time") or {}
    if not virtual_time.get("enabled") or not virtual_time.get("start_date"):
        return None
    return str(virtual_time["start_date"])


def _activation_period_for_run(row: dict) -> int:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    agent_cfg = (scenario.get("agent") or {}) if isinstance(scenario, dict) else {}
    try:
        period = int(agent_cfg.get("activation_period") or 1)
    except (TypeError, ValueError):
        period = 1
    return max(1, period)


def _horizon_for_run(row: dict) -> Optional[int]:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    run_cfg = (scenario.get("run") or {}) if isinstance(scenario, dict) else {}
    try:
        horizon = int(row.get("horizon") or run_cfg.get("horizon_steps") or 0)
    except (TypeError, ValueError):
        return None
    return horizon if horizon > 0 else None


def _max_active_listings_for_run(row: dict) -> int:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    rules = (scenario.get("platform_rules") or {}) if isinstance(scenario, dict) else {}
    try:
        max_active = int(rules.get("max_active_listings") or 100)
    except (TypeError, ValueError):
        max_active = 100
    return max(1, max_active)


def _shop_rating_visual_config(row: dict) -> tuple[list[float], list[float]]:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    cfg = (scenario.get("shop_rating") or {}) if isinstance(scenario, dict) else {}
    thresholds = [float(v) for v in cfg.get("bucket_thresholds", [])]
    multipliers = [float(v) for v in cfg.get("star_multipliers", [])]
    return thresholds, multipliers


def _listing_rating_config(row: dict) -> dict[str, float]:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    cfg = (scenario.get("listing_rating") or {}) if isinstance(scenario, dict) else {}
    return {
        "initial_rating": float(cfg.get("initial_rating", 4.0)),
        "prior_weight": float(cfg.get("prior_weight", 10.0)),
    }


def _week_for_t(t: int, step_hours: float) -> int:
    elapsed_hours = int(t) * float(step_hours or 1)
    day = int(elapsed_hours // 24) + 1
    return int((day - 1) // 7) + 1


def _virtual_start_date_value(row: dict) -> Optional[date]:
    raw = _virtual_start_date_for_run(row)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _month_for_t(
    t: int,
    step_hours: float,
    virtual_start: Optional[date],
) -> int:
    elapsed_days = int((int(t) * float(step_hours or 1)) // 24)
    if virtual_start is None:
        return elapsed_days // 30 + 1
    current = virtual_start + timedelta(days=elapsed_days)
    return (
        (current.year - virtual_start.year) * 12
        + current.month
        - virtual_start.month
        + 1
    )


def _shift_month(value: date, offset: int) -> date:
    zero_based = value.year * 12 + value.month - 1 + int(offset)
    return date(zero_based // 12, zero_based % 12 + 1, 1)


def _month_bounds_hours(
    month: int,
    virtual_start: Optional[date],
) -> tuple[float, float]:
    if virtual_start is None:
        start_hours = (int(month) - 1) * 30 * 24
        return float(start_hours), float(start_hours + 30 * 24)
    calendar_start = _shift_month(virtual_start, int(month) - 1)
    calendar_end = _shift_month(virtual_start, int(month))
    start = max(virtual_start, calendar_start)
    return (
        float((start - virtual_start).days * 24),
        float((calendar_end - virtual_start).days * 24),
    )


def _month_period_descriptor(
    month: int,
    virtual_start: Optional[date],
) -> dict:
    start_hours, end_hours = _month_bounds_hours(month, virtual_start)
    descriptor = {
        "index": int(month),
        "start_day": int(start_hours // 24),
        "end_day": int(end_hours // 24),
    }
    if virtual_start is not None:
        start = virtual_start + timedelta(days=descriptor["start_day"])
        end = virtual_start + timedelta(days=descriptor["end_day"] - 1)
        descriptor["start_date"] = start.isoformat()
        descriptor["end_date"] = end.isoformat()
    return descriptor


def _initial_capital_for_run(row: dict) -> float:
    scenario = _loads_yaml(row.get("scenario_yaml"), {}) if row else {}
    run_cfg = (scenario.get("run") or {}) if isinstance(scenario, dict) else {}
    initial_cash = _float_or(run_cfg.get("initial_cash"), 2000.0)
    initial_deposit = _float_or(run_cfg.get("initial_deposit"), 1000.0)
    return initial_cash + initial_deposit


def _tool_category_key(name: str) -> str:
    # Historical traces can contain retired environment tools. Keep the UI to
    # the five current business categories while retaining those calls in the
    # operational-state bucket instead of inventing an "Other" category.
    return TOOL_CATEGORY_BY_NAME.get(name, "store_state")


def _tool_category_rows(counts: Counter[str] | dict[str, int]) -> list[dict]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for name, count in counts.items():
        if not count:
            continue
        grouped[_tool_category_key(name)][name] += int(count)

    rows = []
    for spec in TOOL_CATEGORY_SPECS:
        tool_counts = grouped.get(spec["key"], Counter())
        total = sum(tool_counts.values())
        if not total:
            continue
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "color": spec["color"],
            "count": int(total),
            "tools": [
                {"name": name, "count": int(count)}
                for name, count in sorted(
                    tool_counts.items(),
                    key=lambda item: (
                        -item[1],
                        LISTING_TOOL_ORDER.get(item[0], len(LISTING_TOOL_ORDER))
                        if spec["key"] == "listing_pricing"
                        else item[0],
                        item[0],
                    ),
                )
            ],
        })
    return rows


def _framework_label(value: Optional[str]) -> str:
    raw = str(value or "none")
    return FRAMEWORK_LABELS.get(raw, raw.replace("_", " ").title())


def _registered_agent_identity(row: dict, runs_root: Optional[str]) -> dict:
    if not runs_root:
        return {}
    run_id = row.get("run_id")
    if not run_id:
        return {}
    meta = agent_log.read_meta(runs_root, str(run_id))
    agents = meta.get("agents") if isinstance(meta, dict) else []
    if not isinstance(agents, list):
        return {}
    agent = next(
        (
            rec
            for rec in agents
            if isinstance(rec, dict) and rec.get("agent_id") == "agent_0"
        ),
        None,
    )
    if agent is None:
        agent = next((rec for rec in agents if isinstance(rec, dict)), None)
    if agent is None:
        return {}
    return {
        "framework_key": str(agent.get("framework") or "").strip(),
        "model": str(agent.get("model") or "").strip(),
    }


def run_identity(row: dict, runs_root: Optional[str] = None) -> dict:
    framework_key = str(row.get("bootstrap_agent") or "none")
    registered = _registered_agent_identity(row, runs_root)
    if framework_key == "none" and registered.get("framework_key"):
        framework_key = registered["framework_key"]
    framework = _framework_label(framework_key)
    cfg = _bootstrap_config(row)
    model = ""
    if framework_key == "human":
        model = str(cfg.get("human_model") or "").strip()
    elif framework_key == "rule_based":
        model = str(cfg.get("selection_mode") or "").strip()
    elif framework_key in MODEL_BOOTSTRAP_KEYS:
        model = str(cfg.get("react_model") or "").strip()
    if not model:
        model = registered.get("model", "")
    model = model or "—"
    display_label = f"{framework} ({model})" if model != "—" else framework
    return {
        "framework_key": framework_key,
        "framework": framework,
        "model": model,
        "display_label": display_label,
    }


def decorate_run(row: dict, runs_root: Optional[str] = None) -> dict:
    out = dict(row)
    out.update(run_identity(row, runs_root))
    return out


def compute_run_result(
    registry,
    run_id: str,
    agent_id: str = "agent_0",
    *,
    conn,
    row: Optional[dict] = None,
) -> Optional[dict]:
    row = row or dbm.get_run(conn, run_id) or {}
    status = row.get("status") or ""
    if status in LIVE_RUN_RESULT_STATUSES:
        with registry.lock:
            worker = registry.workers.get(run_id)
        thread = getattr(worker, "_thread", None)
        if (
            getattr(worker, "state", None) not in LIVE_RUN_RESULT_STATUSES
            or thread is None
            or not thread.is_alive()
        ):
            return None
    elif status not in RUN_RESULT_STATUSES:
        return None

    metric_lasts = dbm.load_metric_lasts(
        conn,
        run_id,
        agent_id,
        [
            "net_assets",
            "cum_net_profit",
            "cum_gmv",
            "cum_cost",
            "cum_fine",
            "cum_fee",
            "shop_rating_mean",
            "shop_rating_score",
            "shop_reputation_evidence_count",
            "shop_qualified_transaction_count",
            "shop_service_quality_score",
            "public_review_rating",
            "public_review_count",
            "public_review_eligible_count",
            "public_review_response_rate",
            "public_review_full_response_rating",
            "public_review_selection_gap",
            "public_review_quality_gap",
            "public_review_confidence",
            "public_review_quality_multiplier",
            "public_review_reputation_multiplier",
            "public_review_demand_multiplier",
        ],
    )
    net_point = metric_lasts.get("net_assets")
    if net_point is None:
        return None

    profit_point = metric_lasts.get("cum_net_profit")
    gmv_point = metric_lasts.get("cum_gmv")
    cost_point = metric_lasts.get("cum_cost")
    fine_point = metric_lasts.get("cum_fine")
    fee_point = metric_lasts.get("cum_fee")
    rating_point = metric_lasts.get("shop_rating_mean")
    canonical_rating = rating_point is not None
    if rating_point is None:
        rating_point = metric_lasts.get("shop_rating_score")
    net = float(net_point[1])
    profit = (
        float(profit_point[1])
        if profit_point is not None else None
    )
    if profit is None and net is not None:
        profit = float(net) - _initial_capital_for_run(row)
    gmv = float(gmv_point[1]) if gmv_point is not None else None
    cum_cost = float(cost_point[1]) if cost_point is not None else 0.0
    fine = float(fine_point[1]) if fine_point is not None else None
    fee_total = float(fee_point[1]) if fee_point is not None else 0.0
    rating = (
        float(rating_point[1])
        if rating_point is not None else None
    )
    reputation_evidence_point = metric_lasts.get(
        "shop_reputation_evidence_count",
    )
    qualified_transaction_point = metric_lasts.get(
        "shop_qualified_transaction_count",
    )
    service_quality_point = metric_lasts.get("shop_service_quality_score")
    public_review_rating_point = metric_lasts.get("public_review_rating")
    public_review_count_point = metric_lasts.get("public_review_count")
    public_review_eligible_count_point = metric_lasts.get(
        "public_review_eligible_count"
    )
    public_review_response_rate_point = metric_lasts.get(
        "public_review_response_rate",
    )
    public_review_full_response_point = metric_lasts.get(
        "public_review_full_response_rating",
    )
    public_review_selection_gap_point = metric_lasts.get(
        "public_review_selection_gap",
    )
    public_review_quality_gap_point = metric_lasts.get(
        "public_review_quality_gap",
    )
    public_review_confidence_point = metric_lasts.get(
        "public_review_confidence"
    )
    public_review_quality_multiplier_point = metric_lasts.get(
        "public_review_quality_multiplier"
    )
    public_review_reputation_multiplier_point = metric_lasts.get(
        "public_review_reputation_multiplier"
    )
    public_review_demand_multiplier_point = metric_lasts.get(
        "public_review_demand_multiplier"
    )
    orders = _orders_generated_total(conn, run_id, agent_id)
    profit_margin = (
        float(profit) / float(gmv)
        if profit is not None and gmv is not None and abs(float(gmv)) > 1e-12
        else None
    )
    horizon = _horizon_for_run(row)
    active_horizon_clause = " AND t<?" if horizon is not None else ""
    active_listing_params: list = [run_id, agent_id]
    if horizon is not None:
        active_listing_params.append(int(horizon))
    active_listings_row = conn.execute(
        "SELECT AVG(value) AS average"
        " FROM metrics WHERE run_id=? AND agent_id=? AND key='n_active_listings'"
        f"{active_horizon_clause}",
        tuple(active_listing_params),
    ).fetchone()
    average_active_listings = (
        float(active_listings_row["average"])
        if active_listings_row is not None
        and active_listings_row["average"] is not None
        else None
    )
    anomaly_row = conn.execute(
        "SELECT COUNT(*) AS total,"
        f" SUM(CASE WHEN {ORDER_ANOMALY_SQL_CONDITION}"
        " THEN 1 ELSE 0 END) AS anomalies"
        " FROM orders WHERE run_id=? AND agent_id=?",
        (run_id, agent_id),
    ).fetchone()
    persisted_orders = int(anomaly_row["total"] or 0) if anomaly_row is not None else 0
    order_anomaly_rate = (
        int(anomaly_row["anomalies"] or 0) / persisted_orders
        if persisted_orders else None
    )
    agents = {a.agent_id: a for a in dbm.list_agents(conn, run_id)}
    agent = agents.get(agent_id)
    cost = agent_log.read_cost(registry.runs_root, run_id)
    total_cost = cost.get("total") or {}
    started = _parse_iso(row.get("started_at"))
    finished = _parse_iso(row.get("finished_at")) if row.get("finished_at") else None
    if started:
        finished = finished or _now_for_elapsed(started)
        started, finished = _align_elapsed_clocks(started, finished)
        elapsed_ms = int((finished - started).total_seconds() * 1000)
    else:
        elapsed_ms = None

    return {
        "final_net_assets": round(float(net), 2),
        "cum_gmv": round(float(gmv), 2) if gmv is not None else None,
        "net_profit": round(float(profit), 2) if profit is not None else None,
        "net_profit_margin": (
            round(float(profit_margin), 6)
            if profit_margin is not None else None
        ),
        "fee_total": round(float(fee_total), 2),
        "contribution_margin_pct": round(
            contribution_margin_pct(
                float(gmv or 0.0), float(cum_cost), float(fee_total),
            ),
            4,
        ),
        "cum_fine": round(float(fine), 2) if fine is not None else None,
        "cum_orders": int(orders) if orders is not None else None,
        "order_anomaly_rate": (
            round(float(order_anomaly_rate), 6)
            if order_anomaly_rate is not None else None
        ),
        "average_active_listings": (
            round(float(average_active_listings), 4)
            if average_active_listings is not None else None
        ),
        # These trace-derived values are filled by build_charts(). Keeping
        # trace parsing out of compute_run_result preserves the lightweight
        # initial dashboard response; the async leaderboard payload supplies
        # the exact values and persists them in the terminal cache.
        "effective_window_rate": None,
        "total_tool_calls": None,
        "shop_rating_mean": (
            round(float(rating), 4)
            if canonical_rating and rating is not None else None
        ),
        "shop_rating_score": round(float(rating), 4) if rating is not None else None,
        "shop_rating_scale": (
            "1-5" if canonical_rating
            else "0-1" if rating is not None
            else None
        ),
        "reputation_evidence_count": (
            int(reputation_evidence_point[1])
            if reputation_evidence_point is not None else None
        ),
        "qualified_transaction_count": (
            int(qualified_transaction_point[1])
            if qualified_transaction_point is not None else None
        ),
        "service_quality_score": (
            round(float(service_quality_point[1]), 4)
            if service_quality_point is not None else None
        ),
        "public_review_rating": (
            round(float(public_review_rating_point[1]), 4)
            if public_review_rating_point is not None else None
        ),
        "public_review_count": (
            int(public_review_count_point[1])
            if public_review_count_point is not None else None
        ),
        "public_review_eligible_count": (
            int(public_review_eligible_count_point[1])
            if public_review_eligible_count_point is not None else None
        ),
        "public_review_response_rate": (
            round(float(public_review_response_rate_point[1]), 6)
            if public_review_response_rate_point is not None else None
        ),
        "public_review_full_response_rating": (
            round(float(public_review_full_response_point[1]), 4)
            if public_review_full_response_point is not None else None
        ),
        "public_review_selection_gap": (
            round(float(public_review_selection_gap_point[1]), 4)
            if public_review_selection_gap_point is not None else None
        ),
        "public_review_quality_gap": (
            round(float(public_review_quality_gap_point[1]), 4)
            if public_review_quality_gap_point is not None else None
        ),
        "public_review_confidence": (
            round(float(public_review_confidence_point[1]), 6)
            if public_review_confidence_point is not None else None
        ),
        "public_review_quality_multiplier": (
            round(float(public_review_quality_multiplier_point[1]), 6)
            if public_review_quality_multiplier_point is not None else None
        ),
        "public_review_reputation_multiplier": (
            round(float(public_review_reputation_multiplier_point[1]), 6)
            if public_review_reputation_multiplier_point is not None else None
        ),
        "public_review_demand_multiplier": (
            round(float(public_review_demand_multiplier_point[1]), 6)
            if public_review_demand_multiplier_point is not None else None
        ),
        "is_alive": bool(agent.is_alive) if agent is not None else True,
        "died_at_t": agent.died_at_t if agent is not None else None,
        "t": int(row.get("current_t") or 0),
        "n_steps": dbm.count_metric_points(
            conn, run_id, agent_id, "net_assets"
        ),
        "tokens": int(total_cost.get("total", 0) or 0),
        "usd": round(float(total_cost.get("usd", 0.0) or 0.0), 6),
        "turns": int(total_cost.get("turns", 0) or 0),
        "elapsed_ms": elapsed_ms,
        "terminal_status": row.get("status") or "unknown",
    }


def build_run_results(registry, runs: Optional[list[dict]] = None) -> list[dict]:
    rows = runs if runs is not None else registry.list_runs()
    out = []
    for row in rows:
        run_id = row["run_id"]
        try:
            with registry.read_conn_for(run_id) as conn:
                full = dbm.get_run(conn, run_id) or row
                result = compute_run_result(registry, full["run_id"], conn=conn, row=full)
        except KeyError:
            # Run was deleted between list_runs and read_conn_for - expected race
            continue
        except (FileNotFoundError, OSError) as e:
            # File disappeared during race condition - expected
            log.debug("run %s file missing during read: %s", run_id, e)
            continue
        except sqlite3.Error as e:
            # Database errors should be logged, not silently swallowed
            log.warning("database error reading run %s: %s", run_id, e)
            continue
        if result is None:
            continue
        ident = run_identity(full, registry.runs_root)
        out.append({
            "run_id": full["run_id"],
            "name": full.get("name") or full["run_id"],
            "bootstrap_agent": ident["framework_key"],
            "framework": ident["framework"],
            "model": ident["model"],
            "display_label": ident["display_label"],
            "master_seed": full.get("master_seed"),
            "horizon": full.get("horizon"),
            "step_hours": full.get("step_hours"),
            "current_t": full.get("current_t"),
            "status": full.get("status") or "unknown",
            "started_at": full.get("started_at"),
            "finished_at": full.get("finished_at"),
            "result": result,
        })
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out


def build_leaderboard(run_results: list[dict]) -> list[dict]:
    rows = []
    for row in run_results:
        result = row.get("result") or {}
        if not result:
            continue
        rows.append({
            "run_id": row.get("run_id"),
            "name": row.get("name") or row.get("run_id"),
            "framework": row.get("framework") or "none",
            "model": row.get("model") or "—",
            "bootstrap_agent": row.get("bootstrap_agent") or "none",
            "open_url": f"/dashboard?run_id={row.get('run_id')}",
            "master_seed": row.get("master_seed"),
            "horizon": row.get("horizon"),
            "step_hours": row.get("step_hours") or 1,
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "elapsed_ms": result.get("elapsed_ms"),
            "runs": 1,
            "avg_final_net_assets": float(result.get("final_net_assets", 0.0) or 0.0),
            "avg_cum_gmv": float(result.get("cum_gmv", 0.0) or 0.0),
            "avg_net_profit": float(result.get("net_profit", 0.0) or 0.0),
            "avg_net_profit_margin": result.get("net_profit_margin"),
            "avg_cum_fine": float(result.get("cum_fine", 0.0) or 0.0),
            "avg_orders": int(result.get("cum_orders", 0) or 0),
            "avg_order_anomaly_rate": result.get("order_anomaly_rate"),
            "avg_active_listings": result.get("average_active_listings"),
            "avg_effective_window_rate": result.get("effective_window_rate"),
            "avg_total_tool_calls": result.get("total_tool_calls"),
            "avg_shop_rating_score": result.get("shop_rating_score"),
            "shop_rating_scale": result.get("shop_rating_scale") or (
                "1-5" if float(result["shop_rating_score"]) > 1 else "0-1"
            ) if result.get("shop_rating_score") is not None else None,
            "avg_tokens": int(result.get("tokens", 0) or 0),
            "avg_usd": float(result.get("usd", 0.0) or 0.0),
            "avg_t": result.get("t", 0),
            "terminal_status": result.get("terminal_status") or row.get("status") or "unknown",
        })
    rows.sort(key=lambda r: (r["avg_final_net_assets"], r["avg_net_profit"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def _run_results_in_leaderboard_order(run_results: list[dict]) -> tuple[list[dict], dict[str, int]]:
    leaderboard = build_leaderboard(run_results)
    rank_by_run = {
        str(row["run_id"]): int(row["rank"])
        for row in leaderboard
        if row.get("run_id") is not None
    }
    fallback_rank = len(run_results) + 1
    ordered = [
        row for _, row in sorted(
            enumerate(run_results),
            key=lambda item: (
                rank_by_run.get(str(item[1].get("run_id")), fallback_rank),
                item[0],
            ),
        )
    ]
    return ordered, rank_by_run


def _line_payload(
    conn,
    run_id: str,
    agent_id: str,
    key: str,
    *,
    step_hours: float,
) -> list[list]:
    # Day filters and window rankings consume this same series. Preserve the
    # exact end-of-day values instead of using display-only even sampling.
    series = dbm.load_metrics_bulk_daily_lasts(
        conn,
        run_id,
        agent_id,
        [key],
        step_hours=step_hours,
    ).get(key, [])
    return [[int(t), float(v)] for t, v in series]


def _daily_average_metric_payload(
    conn,
    run_id: str,
    agent_id: str,
    key: str,
    *,
    step_hours: float,
    horizon: Optional[int] = None,
) -> list[list]:
    """Return exact per-day averages plus sample counts.

    The third tuple element is a sufficient-statistic weight. It lets arbitrary
    full-day windows and the all-days view reproduce the raw per-step average
    exactly, including a partial final day.
    """
    horizon_clause = " AND t<?" if horizon is not None else ""
    params: list = [run_id, agent_id, key]
    if horizon is not None:
        params.append(int(horizon))
    params.append(float(step_hours or 1))
    rows = conn.execute(
        "SELECT MIN(t) AS first_t, AVG(value) AS average, COUNT(*) AS samples"
        " FROM metrics"
        " WHERE run_id=? AND agent_id=? AND key=?"
        f"{horizon_clause}"
        " GROUP BY CAST((t * ?) / 24 AS INTEGER)"
        " ORDER BY first_t",
        tuple(params),
    ).fetchall()
    return [
        [
            int(row["first_t"]),
            float(row["average"]),
            int(row["samples"]),
        ]
        for row in rows
        if row["average"] is not None and int(row["samples"] or 0) > 0
    ]


def _cum_orders_payload(
    conn,
    run_id: str,
    agent_id: str,
    *,
    step_hours: float,
    day_endpoints: Optional[list[int]] = None,
) -> list[list]:
    if dbm.count_orders_for_run(conn, run_id):
        if day_endpoints is None:
            day_endpoints = [
                int(t)
                for t, _ in dbm.load_metrics_bulk_daily_lasts(
                    conn,
                    run_id,
                    agent_id,
                    ["net_assets"],
                    step_hours=step_hours,
                ).get("net_assets", [])
            ]
        counts = dbm.load_order_counts_by_t(conn, run_id, agent_id)
        out = []
        count_index = 0
        total = 0
        for t in day_endpoints:
            while count_index < len(counts) and counts[count_index][0] <= t:
                total += counts[count_index][1]
                count_index += 1
            out.append([int(t), float(total)])
        return out

    # Legacy fixtures may have metrics but no persisted order rows.
    return [
        [int(t), float(total)]
        for t, total in dbm.load_metric_cumulative_daily_lasts(
            conn,
            run_id,
            "_global",
            "orders_generated",
            step_hours=step_hours,
        )
    ]


def _cum_order_anomalies_payload(
    conn,
    run_id: str,
    agent_id: str,
    *,
    step_hours: float,
    day_endpoints: Optional[list[int]] = None,
) -> list[list]:
    if not dbm.count_orders_for_run(conn, run_id):
        return []
    if day_endpoints is None:
        day_endpoints = [
            int(t)
            for t, _ in dbm.load_metrics_bulk_daily_lasts(
                conn,
                run_id,
                agent_id,
                ["net_assets"],
                step_hours=step_hours,
            ).get("net_assets", [])
        ]
    rows = conn.execute(
        "SELECT order_t, COUNT(*) AS n"
        " FROM orders"
        " WHERE run_id=? AND agent_id=?"
        f" AND {ORDER_ANOMALY_SQL_CONDITION}"
        " GROUP BY order_t ORDER BY order_t",
        (run_id, agent_id),
    ).fetchall()
    out = []
    row_index = 0
    total = 0
    for t in day_endpoints:
        while row_index < len(rows) and int(rows[row_index]["order_t"]) <= t:
            total += int(rows[row_index]["n"] or 0)
            row_index += 1
        out.append([int(t), float(total)])
    return out


def _orders_generated_total(
    conn,
    run_id: str,
    agent_id: str = "agent_0",
) -> Optional[float]:
    count = dbm.count_orders_for_agent(conn, run_id, agent_id)
    if count or dbm.count_orders_for_run(conn, run_id):
        return float(count)
    # Tiny/legacy fixtures may carry only the historical global metric.
    return dbm.sum_metric_values(conn, run_id, "_global", "orders_generated")


def _profit_line_payload(
    conn,
    run_id: str,
    agent_id: str,
    initial_capital: float,
    *,
    step_hours: float,
) -> list[list]:
    profit_series = _line_payload(
        conn,
        run_id,
        agent_id,
        "cum_net_profit",
        step_hours=step_hours,
    )
    if profit_series:
        return profit_series
    return [
        [int(t), float(v) - float(initial_capital)]
        for t, v in dbm.load_metrics_bulk_daily_lasts(
            conn,
            run_id,
            agent_id,
            ["net_assets"],
            step_hours=step_hours,
        ).get("net_assets", [])
    ]


def _average_product_price_metrics_payload(conn, run_id: str, agent_id: str) -> tuple[list[list], list[list]]:
    bulk = dbm.load_metrics_bulk_sampled(
        conn,
        run_id,
        agent_id,
        AVERAGE_PRODUCT_PRICE_KEYS,
        max_points_per_key=LEADERBOARD_MAX_SERIES_POINTS,
    )
    prices = dict(bulk.get("avg_listing_sale_price", []))
    counts = dict(bulk.get("avg_listing_sale_price_count", []))
    data = []
    count_data = []
    for t in sorted(prices):
        count = float(counts.get(t) or 0.0)
        if count <= 0:
            continue
        data.append([int(t), round(float(prices[t]), 4)])
        count_data.append([int(t), count])
    return data, count_data


def _listing_sale_price(listing) -> float:
    if isinstance(listing, dict):
        return float(listing.get("sale_price") or 0.0)
    return float(getattr(listing, "sale_price", 0.0) or 0.0)


def _listing_product_id(listing) -> str:
    if isinstance(listing, dict):
        return str(listing.get("product_id") or "")
    return str(getattr(listing, "product_id", "") or "")


def _average_product_price_point(t: int, listings) -> Optional[tuple[list, list]]:
    prices = [_listing_sale_price(listing) for listing in (listings or [])]
    if not prices:
        return None
    count = float(len(prices))
    avg_price = sum(prices) / count
    return [int(t), round(avg_price, 4)], [int(t), count]


def _sample_steps(steps: list[int], max_points: int = 240) -> list[int]:
    if len(steps) <= max_points:
        return steps
    if max_points <= 1:
        return [steps[-1]]
    last = len(steps) - 1
    return sorted({
        steps[round(i * last / (max_points - 1))]
        for i in range(max_points)
    })


def _average_product_price_snapshot_payload(
    runs_root: str,
    run_id: str,
    agent_id: str,
) -> tuple[list[list], list[list]]:
    snap_dir = os.path.join(snap.run_dir(runs_root, run_id), "env_snapshot")
    if not os.path.isdir(snap_dir):
        return [], []
    steps = []
    for name in os.listdir(snap_dir):
        if not (name.startswith("t_") and name.endswith(".json")):
            continue
        try:
            steps.append(int(name[2:-5]))
        except ValueError:
            continue
    data = []
    counts = []
    for t in _sample_steps(sorted(steps)):
        frame = snap.read_env_snapshot(runs_root, run_id, t) or {}
        agents = frame.get("agents") or []
        agent = next((a for a in agents if a.get("agent_id") == agent_id), None)
        if not agent:
            continue
        point = _average_product_price_point(
            int(frame.get("t", t)),
            agent.get("store_listings") or [],
        )
        if point:
            price_point, count_point = point
            data.append(price_point)
            counts.append(count_point)
    return data, counts


def _average_product_price_current_payload(conn, run_id: str, agent_id: str, row: dict) -> tuple[list[list], list[list]]:
    listings = dbm.list_listings(conn, run_id, agent_id)
    point = _average_product_price_point(
        int(row.get("current_t") or 0),
        listings,
    )
    if not point:
        return [], []
    price_point, count_point = point
    return [price_point], [count_point]


def _average_product_price_payload(
    conn,
    run_id: str,
    agent_id: str,
    *,
    runs_root: str,
    row: dict,
) -> tuple[list[list], list[list]]:
    metrics_data, metrics_counts = _average_product_price_metrics_payload(conn, run_id, agent_id)
    if metrics_data:
        return metrics_data, metrics_counts
    snapshot_data, snapshot_counts = _average_product_price_snapshot_payload(
        runs_root, run_id, agent_id
    )
    if snapshot_data:
        return snapshot_data, snapshot_counts
    return _average_product_price_current_payload(conn, run_id, agent_id, row)


def _average_product_margin_metrics_payload(conn, run_id: str, agent_id: str) -> tuple[list[list], list[list]]:
    bulk = dbm.load_metrics_bulk_sampled(
        conn,
        run_id,
        agent_id,
        AVERAGE_PRODUCT_MARGIN_KEYS,
        max_points_per_key=LEADERBOARD_MAX_SERIES_POINTS,
    )
    ratios = dict(bulk.get("avg_listing_margin_ratio", []))
    ratio_counts = dict(bulk.get("avg_listing_margin_ratio_count", []))
    margins = dict(bulk.get("avg_listing_margin", []))
    counts = dict(bulk.get("avg_listing_margin_count", []))
    prices = dict(bulk.get("avg_listing_sale_price", []))
    data = []
    count_data = []
    for t in sorted(set(ratios) | set(margins)):
        has_ratio = t in ratios
        count = float(
            (ratio_counts.get(t) if has_ratio else counts.get(t)) or 0.0
        )
        if count <= 0:
            continue
        if has_ratio:
            ratio = float(ratios[t])
        else:
            # Older runs stored an average currency margin. Dividing it by the
            # average sale price preserves those histories as a portfolio
            # gross-margin ratio instead of presenting the amount as a percent.
            price = float(prices.get(t) or 0.0)
            if price <= 0:
                continue
            ratio = float(margins[t]) / price
        data.append([int(t), round(ratio, 4)])
        count_data.append([int(t), count])
    return data, count_data


def _product_price_by_id(products) -> dict[str, float]:
    out = {}
    if isinstance(products, dict):
        iterable = products.items()
    else:
        iterable = ((None, product) for product in (products or []))
    for key, product in iterable:
        if isinstance(product, dict):
            product_id = str(product.get("product_id") or key or "")
            price = product.get("price")
        else:
            product_id = str(getattr(product, "product_id", "") or "")
            price = getattr(product, "price", None)
        if not product_id or price is None:
            continue
        out[product_id] = float(price)
    return out


def _average_product_margin_point(
    t: int,
    listings,
    product_prices: dict[str, float],
) -> Optional[tuple[list, list]]:
    margin_ratios = []
    for listing in listings or []:
        product_id = _listing_product_id(listing)
        if product_id not in product_prices:
            continue
        sale_price = _listing_sale_price(listing)
        if sale_price <= 0:
            continue
        margin_ratios.append(
            (sale_price - float(product_prices[product_id])) / sale_price
        )
    if not margin_ratios:
        return None
    count = float(len(margin_ratios))
    avg_margin_ratio = sum(margin_ratios) / count
    return [int(t), round(avg_margin_ratio, 4)], [int(t), count]


def _average_product_margin_current_payload(conn, run_id: str, agent_id: str, row: dict) -> tuple[list[list], list[list]]:
    listings = dbm.list_listings(conn, run_id, agent_id)
    products = _product_price_by_id(dbm.load_products(conn, run_id))
    point = _average_product_margin_point(
        int(row.get("current_t") or 0),
        listings,
        products,
    )
    if not point:
        return [], []
    margin_point, count_point = point
    return [margin_point], [count_point]


def _average_product_margin_payload(
    conn,
    run_id: str,
    agent_id: str,
    *,
    row: dict,
) -> tuple[list[list], list[list]]:
    metrics_data, metrics_counts = _average_product_margin_metrics_payload(conn, run_id, agent_id)
    if metrics_data:
        return metrics_data, metrics_counts
    return _average_product_margin_current_payload(conn, run_id, agent_id, row)


def _average_product_rating_metrics_payload(conn, run_id: str, agent_id: str) -> tuple[list[list], list[list]]:
    bulk = dbm.load_metrics_bulk_sampled(
        conn,
        run_id,
        agent_id,
        AVERAGE_PRODUCT_RATING_KEYS,
        max_points_per_key=LEADERBOARD_MAX_SERIES_POINTS,
    )
    ratings = dict(bulk.get("avg_listing_rating", []))
    counts = dict(bulk.get("avg_listing_rating_count", []))
    data = []
    count_data = []
    for t in sorted(ratings):
        count = float(counts.get(t) or 0.0)
        if count <= 0:
            continue
        data.append([int(t), round(float(ratings[t]), 4)])
        count_data.append([int(t), count])
    return data, count_data


def _listing_rating_value(listing, cfg: dict[str, float]) -> float:
    if isinstance(listing, dict):
        rating_sum = float(listing.get("rating_sum") or 0.0)
        rating_count = float(listing.get("rating_count") or 0)
    else:
        rating_sum = float(getattr(listing, "rating_sum", 0.0) or 0.0)
        rating_count = float(getattr(listing, "rating_count", 0) or 0)
    return lr_mod.compute_listing_rating(
        float(cfg["initial_rating"]),
        rating_sum,
        rating_count,
        float(cfg["prior_weight"]),
    )


def _average_product_rating_point(t: int, listings, cfg: dict[str, float]) -> Optional[tuple[list, list]]:
    ratings = [_listing_rating_value(listing, cfg) for listing in (listings or [])]
    if not ratings:
        return None
    count = float(len(ratings))
    avg_rating = sum(ratings) / count
    return [int(t), round(avg_rating, 4)], [int(t), count]


def _average_product_rating_snapshot_payload(
    runs_root: str,
    run_id: str,
    agent_id: str,
    row: dict,
) -> tuple[list[list], list[list]]:
    snap_dir = os.path.join(snap.run_dir(runs_root, run_id), "env_snapshot")
    if not os.path.isdir(snap_dir):
        return [], []
    steps = []
    for name in os.listdir(snap_dir):
        if not (name.startswith("t_") and name.endswith(".json")):
            continue
        try:
            steps.append(int(name[2:-5]))
        except ValueError:
            continue
    cfg = _listing_rating_config(row)
    data = []
    counts = []
    for t in _sample_steps(sorted(steps)):
        frame = snap.read_env_snapshot(runs_root, run_id, t) or {}
        agents = frame.get("agents") or []
        agent = next((a for a in agents if a.get("agent_id") == agent_id), None)
        if not agent:
            continue
        point = _average_product_rating_point(
            int(frame.get("t", t)),
            agent.get("store_listings") or [],
            cfg,
        )
        if point:
            rating_point, count_point = point
            data.append(rating_point)
            counts.append(count_point)
    return data, counts


def _average_product_rating_current_payload(conn, run_id: str, agent_id: str, row: dict) -> tuple[list[list], list[list]]:
    listings = dbm.list_listings(conn, run_id, agent_id)
    point = _average_product_rating_point(
        int(row.get("current_t") or 0),
        listings,
        _listing_rating_config(row),
    )
    if not point:
        return [], []
    rating_point, count_point = point
    return [rating_point], [count_point]


def _average_product_rating_payload(
    conn,
    run_id: str,
    agent_id: str,
    *,
    runs_root: str,
    row: dict,
) -> tuple[list[list], list[list]]:
    metrics_data, metrics_counts = _average_product_rating_metrics_payload(conn, run_id, agent_id)
    if metrics_data:
        return metrics_data, metrics_counts
    snapshot_data, snapshot_counts = _average_product_rating_snapshot_payload(
        runs_root, run_id, agent_id, row
    )
    if snapshot_data:
        return snapshot_data, snapshot_counts
    return _average_product_rating_current_payload(conn, run_id, agent_id, row)


def _step_from_trace_filename(fname: str) -> Optional[int]:
    if not (fname.startswith("t_") and fname.endswith(".json")):
        return None
    try:
        return int(fname[2:-5])
    except ValueError:
        return None


def _tool_call_origin(msg: dict, call: dict) -> str:
    if call.get("tool_origin") is not None:
        return str(call.get("tool_origin"))
    if msg.get("tool_origin") is not None:
        return str(msg.get("tool_origin"))
    return ENV_TOOL_ORIGIN


def _is_merchantbench_env_tool_call(msg: dict, call: dict) -> bool:
    return _tool_call_origin(msg, call) == ENV_TOOL_ORIGIN


def _tool_result_failed(content) -> bool:
    value = content
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return False
    if isinstance(value, dict):
        if value.get("ok") is False:
            return True
        return any(_tool_result_failed(item) for item in value.values())
    if isinstance(value, list):
        return any(_tool_result_failed(item) for item in value)
    return False


_TOOL_STEP_COUNTS_CACHE_LOCK = threading.Lock()
_TOOL_STEP_COUNTS_CACHE_MAX_ENTRIES = 128
_TOOL_STEP_COUNTS_CACHE: OrderedDict[
    tuple[str, str, str],
    tuple[tuple, list[dict]],
] = OrderedDict()


def _tool_call_step_counts(
    registry,
    run_id: str,
    agent_id: str = "agent_0",
) -> list[dict]:
    by_step_dir = os.path.join(agent_log.agent_dir(registry.runs_root, run_id), "by_step")
    if not os.path.isdir(by_step_dir):
        return []
    trace_files = sorted(
        fname
        for fname in os.listdir(by_step_dir)
        if _step_from_trace_filename(fname) is not None
    )
    if not trace_files:
        return []
    last_path = os.path.join(by_step_dir, trace_files[-1])
    try:
        last_stat = os.stat(last_path)
        signature = (
            len(trace_files),
            trace_files[-1],
            int(last_stat.st_mtime_ns),
            int(last_stat.st_size),
        )
    except OSError:
        signature = (len(trace_files), trace_files[-1], None, None)
    cache_key = (
        os.path.realpath(registry.runs_root),
        str(run_id),
        str(agent_id),
    )
    with _TOOL_STEP_COUNTS_CACHE_LOCK:
        cached = _TOOL_STEP_COUNTS_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            _TOOL_STEP_COUNTS_CACHE.move_to_end(cache_key)
            return cached[1]

    steps = []
    seen_runtime_execution_ids: set[str] = set()
    for fname in trace_files:
        step_t = _step_from_trace_filename(fname)
        path = os.path.join(by_step_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                step_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        counts: Counter[str] = Counter()
        total_tool_calls = 0
        runtime: Counter[str] = Counter()
        runtime_telemetry: set[str] = set()
        saw_assistant = False
        messages = list(step_data.get("messages") or [])
        message_agents = step_data.get("message_agents")
        turns = list(step_data.get("turns") or [])
        has_scoped_turns = any(
            isinstance(turn, dict) and "agent_id" in turn
            for turn in turns
        )
        if (
            isinstance(message_agents, list)
            and len(message_agents) == len(messages)
            and any(owner is not None for owner in message_agents)
        ):
            messages = [
                msg
                for msg, owner in zip(messages, message_agents)
                if str(owner or "") == agent_id
            ]
        elif agent_id != "agent_0" or has_scoped_turns:
            # Missing ownership in a multi-agent trace must fail closed.  Truly
            # legacy single-agent traces are attributed to agent_0 below.
            messages = []
        for msg in messages:
            if msg.get("role") == "tool":
                execution_id = str(msg.get("runtime_execution_id") or "")
                if msg.get("runtime_historical") is True:
                    continue
                if execution_id:
                    if execution_id in seen_runtime_execution_ids:
                        continue
                    seen_runtime_execution_ids.add(execution_id)
                if (
                    str(msg.get("tool_origin") or ENV_TOOL_ORIGIN) == ENV_TOOL_ORIGIN
                    and str(msg.get("name") or "") != "end_of_step"
                    and _tool_result_failed(msg.get("content"))
                ):
                    runtime["tool_call_failures"] += 1
                continue
            if msg.get("role") != "assistant":
                continue
            saw_assistant = True
            for call in msg.get("tool_calls", []):
                if not _is_merchantbench_env_tool_call(msg, call):
                    continue
                name = ((call.get("function") or {}).get("name") or "").strip()
                if not name:
                    continue
                total_tool_calls += 1
                if name != "end_of_step":
                    counts[name] += 1
        if has_scoped_turns:
            turns = [
                turn for turn in turns
                if isinstance(turn, dict)
                and str(turn.get("agent_id") or "") == agent_id
            ]
        elif agent_id != "agent_0":
            turns = []
        for turn in turns:
            context = turn.get("context") if isinstance(turn, dict) else None
            if not isinstance(context, dict):
                continue
            if "compacted" in context:
                runtime_telemetry.add("memory_compactions")
                if context.get("compacted") is True:
                    runtime["memory_compactions"] += 1
            for key in (
                "provider_api_failed_attempts",
                "retry_exhausted",
                "skills_evolutions",
            ):
                if key not in context:
                    continue
                runtime_telemetry.add(key)
                try:
                    runtime[key] += max(0, int(context.get(key) or 0))
                except (TypeError, ValueError):
                    continue
        if saw_assistant:
            steps.append({
                "t": int(step_t),
                "counts": dict(counts),
                "total_tool_calls": int(total_tool_calls),
                "runtime": dict(runtime),
                "runtime_telemetry": sorted(runtime_telemetry),
                "hook_open_wall_ms": int(step_data.get("hook_open_wall_ms") or 0),
                "hook_close_wall_ms": int(step_data.get("hook_close_wall_ms") or 0),
            })
    with _TOOL_STEP_COUNTS_CACHE_LOCK:
        _TOOL_STEP_COUNTS_CACHE[cache_key] = (signature, steps)
        _TOOL_STEP_COUNTS_CACHE.move_to_end(cache_key)
        while (
            len(_TOOL_STEP_COUNTS_CACHE)
            > _TOOL_STEP_COUNTS_CACHE_MAX_ENTRIES
        ):
            _TOOL_STEP_COUNTS_CACHE.popitem(last=False)
    return steps


def _sum_step_counts(step_counts: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for step in step_counts:
        counts.update(step.get("counts") or {})
    return counts


def _daily_tool_call_counts(
    step_counts: list[dict],
    step_hours: float,
) -> list[dict]:
    """Compact outbound tool history to the day granularity used by the UI."""
    hours = float(step_hours or 1)
    by_day: dict[int, Counter[str]] = defaultdict(Counter)
    for step in step_counts:
        day = int(int(step.get("t", 0) or 0) * hours // 24)
        by_day[day].update(step.get("counts") or {})
    return [
        {
            "t": int(round(day * 24 / hours)),
            "counts": dict(by_day[day]),
        }
        for day in sorted(by_day)
    ]


def _tool_call_counts(registry, run_id: str) -> dict[str, int]:
    counts = _sum_step_counts(_tool_call_step_counts(registry, run_id))
    return dict(counts)


def _hook_window_week_counts(
    row: dict,
    step_counts: list[dict],
    step_hours: float,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
) -> Counter[int]:
    try:
        completed_t = int(row.get("current_t") or 0)
    except (TypeError, ValueError):
        completed_t = 0
    observed_next_t = max(
        (int(step.get("t", 0) or 0) + 1 for step in step_counts),
        default=0,
    )
    completed_t = max(completed_t, observed_next_t)
    horizon = _horizon_for_run(row)
    if horizon is not None:
        completed_t = min(completed_t, horizon)

    period = _activation_period_for_run(row)
    windows: Counter[int] = Counter()
    for t in range(0, max(0, completed_t), period):
        windows[bucket_for_t(t, step_hours)] += 1

    traced_windows: Counter[int] = Counter()
    for step in step_counts:
        traced_windows[bucket_for_t(int(step.get("t", 0) or 0), step_hours)] += 1
    for week, count in traced_windows.items():
        windows[week] = max(windows[week], count)
    return windows


def _activity_summary(
    row: dict,
    step_counts: list[dict],
    step_hours: float,
) -> dict:
    window_counts = _hook_window_week_counts(row, step_counts, step_hours)
    available_windows = sum(int(count or 0) for count in window_counts.values())
    effective_windows = sum(
        1
        for step in step_counts
        if _effective_tool_calls_for_step(step) > 0
    )
    total_tool_calls = sum(
        _total_tool_calls_for_step(step)
        for step in step_counts
    )
    return {
        "effective_window_rate": (
            round(min(effective_windows, available_windows) / available_windows, 6)
            if available_windows else None
        ),
        "total_tool_calls": int(total_tool_calls),
    }


def _effective_tool_calls_for_step(step: dict) -> int:
    """Count non-end environment calls used to decide if a hook was active."""
    return sum(
        int(count or 0)
        for count in (step.get("counts") or {}).values()
    )


def _total_tool_calls_for_step(step: dict) -> int:
    """Count all environment calls, including end_of_step."""
    return int(
        step.get(
            "total_tool_calls",
            _effective_tool_calls_for_step(step),
        )
        or 0
    )


def _daily_activity_payload(
    row: dict,
    step_counts: list[dict],
    step_hours: float,
) -> list[dict]:
    def day_bucket(t: int, hours: float) -> int:
        return int(int(t) * float(hours or 1) // 24)

    window_counts = _hook_window_week_counts(
        row,
        step_counts,
        step_hours,
        day_bucket,
    )
    effective: Counter[int] = Counter()
    total_calls: Counter[int] = Counter()
    for step in step_counts:
        day = day_bucket(int(step.get("t", 0) or 0), step_hours)
        effective_call_total = _effective_tool_calls_for_step(step)
        total_calls[day] += _total_tool_calls_for_step(step)
        if effective_call_total > 0:
            effective[day] += 1
    days = sorted(set(window_counts) | set(effective) | set(total_calls))
    return [
        {
            "t": int(round(day * 24 / float(step_hours or 1))),
            "available_windows": int(window_counts[day]),
            "effective_windows": int(min(effective[day], window_counts[day])),
            "total_tool_calls": int(total_calls[day]),
        }
        for day in days
    ]


def _weekly_tool_metrics(
    step_counts: list[dict],
    step_hours: float,
    window_counts: Optional[Counter[int] | dict[int, int]] = None,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
) -> dict[str, dict[int, float]]:
    windows: Counter[int] = Counter(window_counts or {})
    effective: Counter[int] = Counter()
    total_calls: Counter[int] = Counter()
    listing_action_ui_calls: Counter[int] = Counter()
    listing_action_calls: Counter[int] = Counter()
    sourcing_calls: Counter[int] = Counter()
    listing_action_calls_by_metric: dict[str, Counter[int]] = {
        spec["metric"]: Counter()
        for spec in LISTING_ACTION_TOOL_SPECS
    }
    sourcing_calls_by_metric: dict[str, Counter[int]] = {
        spec["metric"]: Counter()
        for spec in SOURCING_TOOL_CALL_SPECS
    }
    listing_action_tool_to_metric = {
        spec["tool"]: spec["metric"] for spec in LISTING_ACTION_TOOL_SPECS
    }
    traced_windows: Counter[int] = Counter()
    for step in step_counts:
        week = bucket_for_t(int(step.get("t", 0) or 0), step_hours)
        counts = Counter(step.get("counts") or {})
        traced_windows[week] += 1
        effective_call_total = _effective_tool_calls_for_step(step)
        call_total = _total_tool_calls_for_step(step)
        if effective_call_total > 0:
            effective[week] += 1
        total_calls[week] += call_total
        for spec in SOURCING_TOOL_CALL_SPECS:
            count = counts.get(spec["tool"], 0)
            sourcing_calls_by_metric[spec["metric"]][week] += count
            sourcing_calls[week] += count
        for tool_name in LISTING_TOOL_NAMES:
            count = counts.get(tool_name, 0)
            if not count:
                continue
            listing_action_calls[week] += count
            metric = listing_action_tool_to_metric.get(tool_name)
            if metric:
                listing_action_ui_calls[week] += count
                listing_action_calls_by_metric[metric][week] += count
    for week, count in traced_windows.items():
        windows[week] = max(windows[week], count)

    metrics = {
        "effective_window_rate": {
            week: (effective[week] / windows[week]) if windows[week] else None
            for week in sorted(windows)
        },
        "total_tool_calls": {week: int(total_calls[week]) for week in sorted(windows)},
        "sourcing_calls": {week: int(sourcing_calls[week]) for week in sorted(windows)},
        "listing_action_ui_calls": {
            week: int(listing_action_ui_calls[week])
            for week in sorted(windows)
        },
        "listing_action_calls": {
            week: int(listing_action_calls[week])
            for week in sorted(windows)
        },
    }
    for metric, values in listing_action_calls_by_metric.items():
        metrics[metric] = {week: int(values[week]) for week in sorted(windows)}
    for metric, values in sourcing_calls_by_metric.items():
        metrics[metric] = {week: int(values[week]) for week in sorted(windows)}
    return metrics


def _runtime_health_metadata(
    registry,
    run_id: str,
    agent_id: str = "agent_0",
) -> tuple[int, dict[str, str]]:
    meta = agent_log.read_meta(registry.runs_root, run_id)
    for row in meta.get("agents", []) if isinstance(meta, dict) else []:
        if str(row.get("agent_id") or "") != agent_id:
            continue
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        try:
            version = max(0, int(extra.get("runtime_health_version") or 0))
        except (TypeError, ValueError):
            version = 0
        raw_capabilities = extra.get("runtime_health_capabilities")
        capabilities = {}
        if isinstance(raw_capabilities, dict):
            for key, value in raw_capabilities.items():
                state = str(value or "").strip().lower()
                if state in {"reported", "not_applicable"}:
                    capabilities[str(key)] = state
        return version, capabilities
    return 0, {}

@lru_cache(maxsize=64)
def _parsed_hermes_runtime_log(
    log_path: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[
    tuple[tuple[int, str, int, int, bool], ...],
    Optional[int],
    Optional[int],
]:
    events: list[tuple[int, str, int, int, bool]] = []
    first_wall_ms: Optional[int] = None
    last_wall_ms: Optional[int] = None
    last_api_failure: Optional[tuple[int, int, int, bool]] = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            timestamp_match = _HERMES_LOG_TIMESTAMP_RE.match(line)
            if not timestamp_match:
                continue
            try:
                wall_ms = int(
                    datetime.strptime(
                        timestamp_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                    ).timestamp() * 1000
                )
            except ValueError:
                continue
            first_wall_ms = (
                wall_ms if first_wall_ms is None else min(first_wall_ms, wall_ms)
            )
            last_wall_ms = (
                wall_ms if last_wall_ms is None else max(last_wall_ms, wall_ms)
            )
            attempt_match = _HERMES_API_FAILURE_RE.search(line)
            streaming_failure = "Streaming failed after partial delivery, not retrying:" in line
            nonretryable_failure = "Non-retryable client error:" in line
            skill_evolution = (
                "agent.tool_executor: tool skill_manage completed" in line
            )
            if (
                not attempt_match
                and not streaming_failure
                and not nonretryable_failure
                and not skill_evolution
            ):
                continue
            if skill_evolution:
                events.append((wall_ms, "skills_evolution", 0, 0, False))
            elif attempt_match:
                attempt, limit = map(int, attempt_match.groups())
                thread_match = _HERMES_THREAD_RE.search(line)
                # Explicitly exclude checkpoint-review/background failures
                # from business-window termination counts. Older log lines
                # without a thread field are treated as foreground.
                foreground = (
                    thread_match is None
                    or thread_match.group(1).startswith("MainThread:")
                )
                events.append((wall_ms, "api_failure", attempt, limit, foreground))
                last_api_failure = (wall_ms, attempt, limit, foreground)
                if foreground and limit > 1 and attempt >= limit:
                    events.append((wall_ms, "provider_terminal", attempt, limit, True))
            elif streaming_failure:
                thread_match = _HERMES_THREAD_RE.search(line)
                foreground = (
                    thread_match is None
                    or thread_match.group(1).startswith("MainThread:")
                )
                events.append((wall_ms, "api_failure", 0, 0, foreground))
                if foreground:
                    events.append((wall_ms, "provider_terminal", 0, 0, True))
            elif last_api_failure is not None:
                failure_wall_ms, attempt, limit, foreground = last_api_failure
                if foreground and wall_ms - failure_wall_ms <= 5_000:
                    events.append((wall_ms, "provider_terminal", attempt, limit, True))
    return tuple(events), first_wall_ms, last_wall_ms


def _hermes_runtime_log_metrics(
    registry,
    run_id: str,
    step_counts: list[dict],
    step_hours: float,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
) -> Optional[dict]:
    log_path = os.path.join(
        agent_log.agent_dir(registry.runs_root, run_id),
        "hermes_home",
        "logs",
        "agent.log",
    )
    if not os.path.exists(log_path):
        return None
    anchors = sorted(
        (
            int(step.get("hook_open_wall_ms") or 0),
            int(step.get("t") or 0),
        )
        for step in step_counts
        if int(step.get("hook_open_wall_ms") or 0) > 0
    )
    if not anchors:
        return None
    opened_values = [row[0] for row in anchors]
    api_failures: Counter[int] = Counter()
    exhausted: Counter[int] = Counter()
    skills_evolutions: Counter[int] = Counter()
    abnormal_window_steps: set[int] = set()
    try:
        stat = os.stat(log_path)
        events, first_log_wall_ms, last_log_wall_ms = _parsed_hermes_runtime_log(
            log_path, stat.st_mtime_ns, stat.st_size
        )
        if first_log_wall_ms is None or last_log_wall_ms is None:
            return None
        for wall_ms, event_type, attempt, limit, foreground in events:
            anchor_idx = max(0, bisect_right(opened_values, wall_ms) - 1)
            step_t = anchors[anchor_idx][1]
            week = bucket_for_t(step_t, step_hours)
            if event_type == "skills_evolution":
                skills_evolutions[week] += 1
                continue
            if event_type == "provider_terminal":
                if foreground:
                    abnormal_window_steps.add(step_t)
                continue
            api_failures[week] += 1
            if limit > 1 and attempt >= limit:
                exhausted[week] += 1
    except OSError:
        return None
    coverage_weeks = {
        bucket_for_t(int(step.get("t") or 0), step_hours)
        for step in step_counts
        if (
            int(step.get("hook_open_wall_ms") or 0) > 0
            and first_log_wall_ms
            <= int(
                step.get("hook_close_wall_ms")
                or step.get("hook_open_wall_ms")
                or 0
            )
            and last_log_wall_ms >= int(step.get("hook_open_wall_ms") or 0)
        )
    }
    coverage_weeks.update(api_failures)
    coverage_weeks.update(exhausted)
    coverage_weeks.update(skills_evolutions)
    coverage_weeks.update(
        bucket_for_t(step_t, step_hours) for step_t in abnormal_window_steps
    )
    return {
        "api_failed_attempts": api_failures,
        "retry_exhausted": exhausted,
        "skills_evolutions": skills_evolutions,
        "abnormal_window_steps": abnormal_window_steps,
        # A legacy text log proves that some telemetry exists in these weeks,
        # but without an append-only start marker it cannot prove completeness.
        "coverage_weeks": coverage_weeks,
    }


def _weekly_runtime_health(
    registry,
    run_id: str,
    step_counts: list[dict],
    step_hours: float,
    window_counts: Counter[int] | dict[int, int],
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
    bucket_bounds_hours: Optional[Callable[[int], tuple[float, float]]] = None,
) -> tuple[
    dict[str, dict[int, Optional[float]]],
    dict[str, dict[int, str]],
]:
    expected_windows = Counter({
        int(week): max(0, int(count or 0))
        for week, count in dict(window_counts).items()
    })
    target_weeks = sorted(expected_windows)
    by_metric: dict[str, Counter[int]] = {
        key: Counter() for key in RUNTIME_HEALTH_METRICS
    }
    merchantbench_api_failures: Counter[int] = Counter()
    abnormal_window_steps: set[int] = set()
    telemetry_first_t: dict[str, int] = {}
    traced_windows: Counter[int] = Counter()
    for step in step_counts:
        step_t = int(step.get("t", 0) or 0)
        week = bucket_for_t(step_t, step_hours)
        traced_windows[week] += 1
        runtime = step.get("runtime") if isinstance(step.get("runtime"), dict) else {}
        by_metric["tool_call_failures"][week] += int(
            runtime.get("tool_call_failures", 0) or 0
        )
        by_metric["memory_compactions"][week] += int(
            runtime.get("memory_compactions", 0) or 0
        )
        step_telemetry = step.get("runtime_telemetry")
        observed_telemetry: set[str] = set()
        if isinstance(step_telemetry, list):
            observed_telemetry.update(str(key) for key in step_telemetry)
        # Compatibility with traces written before runtime_telemetry existed.
        observed_telemetry.update(
            key for key in (
                "provider_api_failed_attempts",
                "retry_exhausted",
                "skills_evolutions",
            )
            if key in runtime
        )
        for key in observed_telemetry:
            telemetry_first_t[key] = min(
                telemetry_first_t.get(key, step_t),
                step_t,
            )
        provider_failures = int(runtime.get("provider_api_failed_attempts", 0) or 0)
        by_metric["api_failed_attempts"][week] += provider_failures
        retry_exhausted = int(runtime.get("retry_exhausted", 0) or 0)
        by_metric["retry_exhausted"][week] += retry_exhausted
        if retry_exhausted > 0:
            abnormal_window_steps.add(step_t)
        by_metric["skills_evolutions"][week] += int(
            runtime.get("skills_evolutions", 0) or 0
        )

    runtime_events = agent_log.read_runtime_events(registry.runs_root, run_id)
    telemetry_started_t: Optional[int] = None
    any_telemetry_start_marker = False
    if isinstance(runtime_events, dict):
        for event in runtime_events.get("events", []):
            if not isinstance(event, dict):
                continue
            if event.get("event_type") == "runtime_telemetry_started":
                any_telemetry_start_marker = True
            if str(event.get("agent_id") or "agent_0") != "agent_0":
                continue
            if event.get("event_type") == "runtime_telemetry_started":
                try:
                    marker_t = max(0, int(event.get("t") or 0))
                except (TypeError, ValueError):
                    continue
                telemetry_started_t = (
                    marker_t
                    if telemetry_started_t is None
                    else min(telemetry_started_t, marker_t)
                )
                continue
            if event.get("event_type") != "merchantbench_api_failed_attempt":
                continue
            week = bucket_for_t(int(event.get("t", 0) or 0), step_hours)
            by_metric["api_failed_attempts"][week] += 1
            merchantbench_api_failures[week] += 1
            payload = (
                event.get("payload")
                if isinstance(event.get("payload"), dict)
                else {}
            )
            try:
                status = int(payload.get("status") or 0)
            except (TypeError, ValueError):
                status = 0
            if status == 425:
                abnormal_window_steps.add(int(event.get("t", 0) or 0))

    provider_telemetry_seen = "provider_api_failed_attempts" in telemetry_first_t
    retry_telemetry_seen = "retry_exhausted" in telemetry_first_t
    skills_context_seen = "skills_evolutions" in telemetry_first_t
    hermes_log_metrics = _hermes_runtime_log_metrics(
        registry, run_id, step_counts, step_hours, bucket_for_t
    )
    if hermes_log_metrics is not None:
        if not provider_telemetry_seen:
            by_metric["api_failed_attempts"].update(
                hermes_log_metrics["api_failed_attempts"]
            )
        if not retry_telemetry_seen:
            by_metric["retry_exhausted"].update(
                hermes_log_metrics["retry_exhausted"]
            )
        if not skills_context_seen:
            by_metric["skills_evolutions"].update(
                hermes_log_metrics["skills_evolutions"]
            )
        abnormal_window_steps.update(
            hermes_log_metrics["abnormal_window_steps"]
        )

    for step_t in abnormal_window_steps:
        by_metric["abnormal_ended_windows"][bucket_for_t(step_t, step_hours)] += 1
    for week in target_weeks:
        # An expected hook with no agent trace means the agent did not
        # participate in that business window (timeout, process exit, etc.).
        by_metric["abnormal_ended_windows"][week] += max(
            0, int(expected_windows[week]) - int(traced_windows[week])
        )

    version, capabilities = _runtime_health_metadata(registry, run_id)

    def capability_state(metric: str) -> Optional[str]:
        if capabilities:
            return capabilities.get(metric)
        return "reported" if version >= 1 else None

    coverage_rank = {"unavailable": 0, "partial": 1, "complete": 2}

    def coverage_from_start(week: int, start_t: Optional[int]) -> str:
        if start_t is None:
            return "unavailable"
        start_hours = int(start_t) * float(step_hours or 1)
        if bucket_bounds_hours is None:
            week_start_hours = (int(week) - 1) * 7 * 24
            week_end_hours = int(week) * 7 * 24
        else:
            week_start_hours, week_end_hours = bucket_bounds_hours(int(week))
        if start_hours <= week_start_hours:
            return "complete"
        if start_hours < week_end_hours:
            return "partial"
        return "unavailable"

    def merge_coverage(*states: str) -> str:
        return max(states, key=lambda state: coverage_rank[state])

    def cap_coverage(*states: str) -> str:
        return min(states, key=lambda state: coverage_rank[state])

    def trace_coverage(week: int) -> str:
        expected = int(expected_windows[week])
        traced = int(traced_windows[week])
        if expected <= 0:
            return "partial" if traced else "unavailable"
        if traced >= expected:
            return "complete"
        if traced:
            return "partial"
        return "unavailable"

    def reported_coverage_from_start(week: int, start_t: Optional[int]) -> str:
        return cap_coverage(
            coverage_from_start(week, start_t),
            trace_coverage(week),
        )

    def declared_coverage(metric: str, week: int) -> str:
        state = capability_state(metric)
        if state not in {"reported", "not_applicable"}:
            return "unavailable"
        if telemetry_started_t is None:
            return "partial"
        marker_coverage = coverage_from_start(week, telemetry_started_t)
        if state == "not_applicable":
            return marker_coverage
        return cap_coverage(marker_coverage, trace_coverage(week))

    coverage: dict[str, dict[int, str]] = {
        metric: {} for metric in RUNTIME_HEALTH_METRICS
    }
    merchantbench_source_seen = (
        telemetry_started_t is not None
        or bool(merchantbench_api_failures)
        or (
            isinstance(runtime_events, dict)
            and not any_telemetry_start_marker
        )
    )
    hermes_coverage_weeks = (
        set(hermes_log_metrics.get("coverage_weeks") or set())
        if hermes_log_metrics is not None
        else set()
    )
    for week in target_weeks:
        if merchantbench_source_seen:
            merchantbench_coverage = (
                coverage_from_start(week, telemetry_started_t)
                if telemetry_started_t is not None
                else "partial"
            )
            if (
                merchantbench_coverage == "unavailable"
                and merchantbench_api_failures[week] > 0
            ):
                merchantbench_coverage = "partial"
        else:
            merchantbench_coverage = "unavailable"
        provider_coverage = merge_coverage(
            declared_coverage("provider_api_failed_attempts", week),
            reported_coverage_from_start(
                week, telemetry_first_t.get("provider_api_failed_attempts")
            ),
            "partial" if week in hermes_coverage_weeks else "unavailable",
        )
        if merchantbench_coverage == "complete" and provider_coverage == "complete":
            api_coverage = "complete"
        elif merchantbench_coverage == "unavailable" and provider_coverage == "unavailable":
            api_coverage = "unavailable"
        else:
            api_coverage = "partial"
        coverage["api_failed_attempts"][week] = api_coverage
        # Missing hooks are known from the expected schedule, while legacy
        # provider-terminal reasons are reconstructed from best-effort logs.
        # Keep this metric partial until every termination path is emitted as
        # structured per-window telemetry.
        coverage["abnormal_ended_windows"][week] = (
            "partial" if expected_windows[week] > 0 else "unavailable"
        )
        coverage["tool_call_failures"][week] = trace_coverage(week)
        coverage["retry_exhausted"][week] = merge_coverage(
            declared_coverage("retry_exhausted", week),
            reported_coverage_from_start(
                week, telemetry_first_t.get("retry_exhausted")
            ),
            "partial" if week in hermes_coverage_weeks else "unavailable",
        )
        coverage["memory_compactions"][week] = merge_coverage(
            declared_coverage("memory_compactions", week),
            reported_coverage_from_start(
                week, telemetry_first_t.get("memory_compactions")
            ),
        )
        coverage["skills_evolutions"][week] = merge_coverage(
            declared_coverage("skills_evolutions", week),
            reported_coverage_from_start(
                week, telemetry_first_t.get("skills_evolutions")
            ),
            "partial" if week in hermes_coverage_weeks else "unavailable",
        )
    assert all(
        set(states.values()) <= RUNTIME_HEALTH_COVERAGE_STATES
        for states in coverage.values()
    )
    out: dict[str, dict[int, Optional[float]]] = {}
    for metric in RUNTIME_HEALTH_METRICS:
        out[metric] = {
            week: (
                int(by_metric[metric][week])
                if coverage[metric][week] != "unavailable"
                else None
            )
            for week in target_weeks
        }
    return out, coverage


def _weekly_cumulative_deltas(
    conn,
    run_id: str,
    agent_id: str,
    key: str,
    step_hours: float,
    initial_previous: float = 0.0,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
) -> dict[int, float]:
    by_week: dict[int, float] = {}
    for t, value in dbm.load_metric_series(conn, run_id, agent_id, key):
        by_week[bucket_for_t(int(t), step_hours)] = float(value)

    out: dict[int, float] = {}
    previous = float(initial_previous)
    for week in sorted(by_week):
        current = by_week[week]
        out[week] = current - previous
        previous = current
    return out


def _weekly_profit_deltas(
    conn,
    run_id: str,
    agent_id: str,
    step_hours: float,
    initial_capital: float,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
) -> dict[int, float]:
    profit = _weekly_cumulative_deltas(
        conn,
        run_id,
        agent_id,
        "cum_net_profit",
        step_hours,
        bucket_for_t=bucket_for_t,
    )
    if profit:
        return profit
    return _weekly_cumulative_deltas(
        conn,
        run_id,
        agent_id,
        "net_assets",
        step_hours,
        initial_previous=initial_capital,
        bucket_for_t=bucket_for_t,
    )


def _shelf_metric_source(conn, run_id: str, agent_id: str) -> dict:
    """Read shelf events and order product ids once for all period grains."""
    rows = conn.execute(
        "SELECT rowid AS event_order, t, event_type, entity_id"
        " FROM events"
        " WHERE run_id=? AND agent_id=?"
        " AND event_type IN ('agent_list_product', 'agent_delist_product')"
        " ORDER BY t ASC, rowid ASC",
        (run_id, agent_id),
    ).fetchall()
    seen_list_events = {
        str(row["entity_id"])
        for row in rows
        if str(row["event_type"]) == "agent_list_product" and row["entity_id"]
    }
    synthetic_rows = []
    current_rows = conn.execute(
        "SELECT product_id, COALESCE(first_listed_at, listed_at, 0) AS listed_t"
        " FROM store_listings"
        " WHERE run_id=? AND agent_id=?",
        (run_id, agent_id),
    ).fetchall()
    for row in current_rows:
        product_id = str(row["product_id"] or "")
        if not product_id or product_id in seen_list_events:
            continue
        synthetic_rows.append({
            "event_order": -1,
            "t": int(row["listed_t"] or 0),
            "event_type": "agent_list_product",
            "entity_id": product_id,
        })

    events = sorted(
        [dict(row) for row in rows] + synthetic_rows,
        key=lambda row: (
            int(row["t"] or 0),
            int(row["event_order"] or 0),
            str(row["entity_id"] or ""),
        ),
    )
    order_rows = conn.execute(
        "SELECT order_t, product_id"
        " FROM orders"
        " WHERE run_id=? AND agent_id=?",
        (run_id, agent_id),
    ).fetchall()
    return {"events": events, "order_rows": order_rows}


def _weekly_shelf_metrics(
    conn,
    run_id: str,
    agent_id: str,
    step_hours: float,
    weeks: list[int] | set[int],
    shelf_capacity: int,
    bucket_for_t: Callable[[int, float], int] = _week_for_t,
    *,
    source: Optional[dict] = None,
) -> dict[str, dict[int, float]]:
    target_weeks = sorted(int(week) for week in weeks)
    if not target_weeks:
        return {
            "shelf_product_count": {},
            "shelf_utilization_rate": {},
            "sell_through_capacity_rate": {},
            "sell_through_active_shelf_rate": {},
            "sell_through_rate": {},
            "weekly_new_unique_products": {},
        }

    source = source or _shelf_metric_source(conn, run_id, agent_id)
    events = source["events"]
    active: set[str] = set()
    event_idx = 0
    shelf_counts: dict[int, float] = {}
    for week in target_weeks:
        while event_idx < len(events):
            event = events[event_idx]
            event_week = bucket_for_t(int(event["t"] or 0), step_hours)
            if event_week > week:
                break
            product_id = str(event["entity_id"] or "")
            if product_id:
                if str(event["event_type"]) == "agent_delist_product":
                    active.discard(product_id)
                else:
                    active.add(product_id)
            event_idx += 1
        shelf_counts[week] = float(len(active))

    sold_by_week: dict[int, set[str]] = defaultdict(set)
    target_week_set = set(target_weeks)
    for row in source["order_rows"]:
        product_id = str(row["product_id"] or "")
        if not product_id:
            continue
        week = bucket_for_t(int(row["order_t"] or 0), step_hours)
        if week in target_week_set:
            sold_by_week[week].add(product_id)

    capacity = max(1, int(shelf_capacity or 1))
    first_listed_week: dict[str, int] = {}
    for event in events:
        if str(event.get("event_type") or "") != "agent_list_product":
            continue
        product_id = str(event.get("entity_id") or "")
        if not product_id or product_id in first_listed_week:
            continue
        first_listed_week[product_id] = bucket_for_t(
            int(event.get("t") or 0), step_hours
        )
    new_products_by_week = Counter(first_listed_week.values())
    weekly_new_unique_products = {
        week: int(new_products_by_week[week])
        for week in target_weeks
    }
    shelf_utilization = {
        week: shelf_counts[week] / capacity
        for week in target_weeks
    }
    sell_through_capacity = {
        week: len(sold_by_week[week]) / capacity
        for week in target_weeks
    }
    sell_through_active = {
        week: (len(sold_by_week[week]) / shelf_counts[week]) if shelf_counts[week] else None
        for week in target_weeks
    }
    return {
        "shelf_product_count": shelf_counts,
        "shelf_utilization_rate": shelf_utilization,
        "sell_through_capacity_rate": sell_through_capacity,
        "sell_through_active_shelf_rate": sell_through_active,
        "sell_through_rate": sell_through_active,
        "weekly_new_unique_products": weekly_new_unique_products,
    }


PERIOD_RATE_METRICS = {
    "effective_window_rate",
    "shelf_utilization_rate",
    "sell_through_capacity_rate",
    "sell_through_active_shelf_rate",
    "sell_through_rate",
}


def _period_chart_payload(
    *,
    index_key: str,
    indices: list[int],
    metric_keys: list[str],
    period_runs: list[
        tuple[
            dict,
            dict[str, dict[int, Optional[float]]],
            dict[str, dict[int, str]],
        ]
    ],
    start_date: Optional[str],
    descriptors: Optional[list[dict]] = None,
) -> dict:
    payload = {
        index_key: indices,
        "start_date": start_date,
        "metrics": {key: [] for key in metric_keys},
        "listing_action_tools": [
            {"key": "all", "label": "All Actions", "metric": "listing_action_calls"},
            *[
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "metric": spec["metric"],
                }
                for spec in LISTING_ACTION_TOOL_SPECS
            ],
        ],
        "sourcing_tools": [
            {"key": "all", "label": "All Sourcing", "metric": "sourcing_calls"},
            *[
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "metric": spec["metric"],
                }
                for spec in SOURCING_TOOL_CALL_SPECS
            ],
        ],
    }
    if descriptors is not None:
        payload["periods"] = descriptors
    for meta, metrics, runtime_coverage in period_runs:
        for key in metric_keys:
            values = []
            for index in indices:
                value = metrics.get(key, {}).get(index)
                if value is None:
                    values.append(None)
                elif key in PERIOD_RATE_METRICS:
                    values.append(round(float(value), 4))
                elif key.endswith("_calls") or key in {
                    "weekly_new_unique_products",
                    *RUNTIME_HEALTH_METRICS,
                }:
                    values.append(int(value))
                else:
                    values.append(round(float(value), 2))
            row = {
                **meta,
                "values": values,
            }
            if key in RUNTIME_HEALTH_METRICS:
                metric_coverage = runtime_coverage.get(key, {})
                coverage_values = [
                    metric_coverage.get(index, "unavailable")
                    for index in indices
                ]
                active_values = [
                    index in metrics.get("total_tool_calls", {})
                    for index in indices
                ]
                active_coverage = [
                    coverage_values[idx]
                    for idx, active in enumerate(active_values)
                    if active
                ]
                if active_coverage and all(
                    state == "complete" for state in active_coverage
                ):
                    aggregate_coverage = "complete"
                elif active_coverage and any(
                    state != "unavailable" for state in active_coverage
                ):
                    aggregate_coverage = "partial"
                else:
                    aggregate_coverage = "unavailable"
                row["coverage"] = aggregate_coverage
                row["coverage_values"] = coverage_values
                row["active_values"] = active_values
                row["available"] = aggregate_coverage == "complete"
            payload["metrics"][key].append(row)
    return payload


def build_charts(registry, run_results: list[dict]) -> dict:
    ordered_run_results, rank_by_run = _run_results_in_leaderboard_order(run_results)
    runs_meta = []
    net_assets = []
    net_assets_cost = []
    cum_gmv = []
    cum_net_profit = []
    cum_fine = []
    cum_orders = []
    cum_order_anomalies = []
    active_listings = []
    shop_rating_score = []
    average_product_price = []
    average_product_margin = []
    average_product_rating = []
    tool_call_runs = []
    tool_totals: Counter[str] = Counter()
    weekly_runs: list[
        tuple[
            dict,
            dict[str, dict[int, Optional[float]]],
            dict[str, dict[int, str]],
        ]
    ] = []
    monthly_runs: list[
        tuple[
            dict,
            dict[str, dict[int, Optional[float]]],
            dict[str, dict[int, str]],
        ]
    ] = []
    all_weeks: set[int] = set()
    all_months: set[int] = set()
    for row in ordered_run_results:
        result = row.get("result") or {}
        fallback_model = row.get("model") or "—"
        fallback_framework = row.get("framework") or "None"
        label = row.get("display_label") or (
            f"{fallback_framework} ({fallback_model})"
            if fallback_model != "—"
            else fallback_framework
        )
        run_id = row.get("run_id")
        framework_key = row.get("bootstrap_agent") or "none"
        framework_color = _framework_color(framework_key)
        model = row.get("model") or "—"
        meta = {
            "run_id": run_id,
            "rank": rank_by_run.get(str(run_id)) if run_id is not None else None,
            "label": label,
            "framework": row.get("framework") or "none",
            "model": model,
            "axis_label": model if model != "—" else (row.get("framework") or "none"),
            "bootstrap_agent": framework_key,
            "framework_color": framework_color,
            "color": framework_color,
            "step_hours": 1.0,  # Default, will be updated if run_id exists
            "virtual_start_date": None,
        }
        runs_meta.append(meta)
        if run_id:
            try:
                with registry.read_conn_for(run_id) as conn:
                    full_row = dbm.get_run(conn, run_id) or {}
                    step_hours = _step_hours_for_run(full_row or {})
                    initial_capital = _initial_capital_for_run(full_row or {})
                    shelf_capacity = _max_active_listings_for_run(full_row or {})
                    virtual_start = _virtual_start_date_value(full_row or {})
                    meta["step_hours"] = step_hours
                    meta["virtual_start_date"] = (
                        virtual_start.isoformat() if virtual_start else None
                    )
                    series = _line_payload(
                        conn,
                        run_id,
                        "agent_0",
                        "net_assets",
                        step_hours=step_hours,
                    )
                    if series:
                        net_assets.append({
                            **meta,
                            "data": series,
                        })
                    for key, target in (
                        ("cum_gmv", cum_gmv),
                        ("cum_fine", cum_fine),
                    ):
                        metric_series = _line_payload(
                            conn,
                            run_id,
                            "agent_0",
                            key,
                            step_hours=step_hours,
                        )
                        if metric_series:
                            target.append({
                                **meta,
                                "data": metric_series,
                            })
                    profit_series = _profit_line_payload(
                        conn,
                        run_id,
                        "agent_0",
                        initial_capital,
                        step_hours=step_hours,
                    )
                    if profit_series:
                        cum_net_profit.append({
                            **meta,
                            "data": profit_series,
                        })
                    orders_series = _cum_orders_payload(
                        conn,
                        run_id,
                        "agent_0",
                        step_hours=step_hours,
                        day_endpoints=[
                            int(point[0]) for point in series
                        ],
                    )
                    if orders_series:
                        cum_orders.append({
                            **meta,
                            "data": orders_series,
                        })
                    anomaly_series = _cum_order_anomalies_payload(
                        conn,
                        run_id,
                        "agent_0",
                        step_hours=step_hours,
                        day_endpoints=[
                            int(point[0]) for point in series
                        ],
                    )
                    if anomaly_series:
                        cum_order_anomalies.append({
                            **meta,
                            "data": anomaly_series,
                        })
                    active_listing_series = _daily_average_metric_payload(
                        conn,
                        run_id,
                        "agent_0",
                        "n_active_listings",
                        step_hours=step_hours,
                        horizon=_horizon_for_run(full_row or {}),
                    )
                    if active_listing_series:
                        active_listings.append({
                            **meta,
                            "data": active_listing_series,
                        })
                    shop_series = _line_payload(
                        conn,
                        run_id,
                        "agent_0",
                        "shop_rating_mean",
                        step_hours=step_hours,
                    )
                    shop_rating_scale = "1-5" if shop_series else "0-1"
                    if not shop_series:
                        shop_series = _line_payload(
                            conn,
                            run_id,
                            "agent_0",
                            "shop_rating_score",
                            step_hours=step_hours,
                        )
                    if shop_series:
                        thresholds, star_multipliers = _shop_rating_visual_config(full_row or {})
                        shop_rating_score.append({
                            **meta,
                            "data": shop_series,
                            "stars": _line_payload(
                                conn,
                                run_id,
                                "agent_0",
                                "shop_rating_stars",
                                step_hours=step_hours,
                            ),
                            "thresholds": thresholds,
                            "star_multipliers": star_multipliers,
                            "rating_scale": shop_rating_scale,
                        })
                    avg_price_data, avg_price_counts = _average_product_price_payload(
                        conn,
                        run_id,
                        "agent_0",
                        runs_root=registry.runs_root,
                        row=full_row or {},
                    )
                    if avg_price_data:
                        average_product_price.append({
                            **meta,
                            "data": avg_price_data,
                            "counts": avg_price_counts,
                        })
                    avg_margin_data, avg_margin_counts = _average_product_margin_payload(
                        conn,
                        run_id,
                        "agent_0",
                        row=full_row or {},
                    )
                    if avg_margin_data:
                        average_product_margin.append({
                            **meta,
                            "data": avg_margin_data,
                            "counts": avg_margin_counts,
                        })
                    avg_rating_data, avg_rating_counts = _average_product_rating_payload(
                        conn,
                        run_id,
                        "agent_0",
                        runs_root=registry.runs_root,
                        row=full_row or {},
                    )
                    if avg_rating_data:
                        average_product_rating.append({
                            **meta,
                            "data": avg_rating_data,
                            "counts": avg_rating_counts,
                        })
                    step_counts = _tool_call_step_counts(registry, run_id)
                    activity_summary = _activity_summary(
                        full_row or {},
                        step_counts,
                        step_hours,
                    )
                    result.update(activity_summary)
                    tool_counts = dict(_sum_step_counts(step_counts))
                    window_counts = _hook_window_week_counts(full_row or {}, step_counts, step_hours)
                    weekly_metrics = _weekly_tool_metrics(step_counts, step_hours, window_counts)
                    shelf_source = _shelf_metric_source(conn, run_id, "agent_0")
                    weekly_metrics.update(
                        _weekly_shelf_metrics(
                            conn,
                            run_id,
                            "agent_0",
                            step_hours,
                            set(window_counts.keys()) | set(weekly_metrics.get("total_tool_calls", {}).keys()),
                            shelf_capacity,
                            source=shelf_source,
                        )
                    )
                    runtime_metrics, runtime_coverage = _weekly_runtime_health(
                        registry,
                        run_id,
                        step_counts,
                        step_hours,
                        window_counts,
                    )
                    weekly_metrics.update(runtime_metrics)
                    weekly_metrics["weekly_gmv"] = _weekly_cumulative_deltas(
                        conn, run_id, "agent_0", "cum_gmv", step_hours
                    )
                    weekly_metrics["weekly_profit"] = _weekly_profit_deltas(
                        conn, run_id, "agent_0", step_hours, initial_capital
                    )
                    for values in weekly_metrics.values():
                        all_weeks.update(values.keys())
                    weekly_runs.append((meta, weekly_metrics, runtime_coverage))

                    def month_bucket(t: int, hours: float) -> int:
                        return _month_for_t(t, hours, virtual_start)

                    def month_bounds(month: int) -> tuple[float, float]:
                        return _month_bounds_hours(month, virtual_start)

                    monthly_window_counts = _hook_window_week_counts(
                        full_row or {},
                        step_counts,
                        step_hours,
                        month_bucket,
                    )
                    monthly_metrics = _weekly_tool_metrics(
                        step_counts,
                        step_hours,
                        monthly_window_counts,
                        month_bucket,
                    )
                    monthly_metrics.update(
                        _weekly_shelf_metrics(
                            conn,
                            run_id,
                            "agent_0",
                            step_hours,
                            set(monthly_window_counts.keys())
                            | set(monthly_metrics.get("total_tool_calls", {}).keys()),
                            shelf_capacity,
                            month_bucket,
                            source=shelf_source,
                        )
                    )
                    monthly_runtime_metrics, monthly_runtime_coverage = (
                        _weekly_runtime_health(
                            registry,
                            run_id,
                            step_counts,
                            step_hours,
                            monthly_window_counts,
                            month_bucket,
                            month_bounds,
                        )
                    )
                    monthly_metrics.update(monthly_runtime_metrics)
                    monthly_metrics["weekly_gmv"] = _weekly_cumulative_deltas(
                        conn,
                        run_id,
                        "agent_0",
                        "cum_gmv",
                        step_hours,
                        bucket_for_t=month_bucket,
                    )
                    monthly_metrics["weekly_profit"] = _weekly_profit_deltas(
                        conn,
                        run_id,
                        "agent_0",
                        step_hours,
                        initial_capital,
                        month_bucket,
                    )
                    for values in monthly_metrics.values():
                        all_months.update(values.keys())
                    monthly_runs.append(
                        (meta, monthly_metrics, monthly_runtime_coverage)
                    )
                    if tool_counts:
                        tool_totals.update(tool_counts)
                    tool_call_runs.append({
                        **meta,
                        "counts": tool_counts,
                        "by_step": _daily_tool_call_counts(
                            step_counts,
                            step_hours,
                        ),
                        "activity_by_day": _daily_activity_payload(
                            full_row or {},
                            step_counts,
                            step_hours,
                        ),
                        "categories": _tool_category_rows(tool_counts),
                        "total": activity_summary["total_tool_calls"],
                    })
            except KeyError as e:
                # Run was deleted between list_runs and per-run DB access,
                # Skip this run's chart data rather than crashing the dashboard.
                log.debug("skipping chart data for run %s: %s", run_id, e)
            except (OSError, sqlite3.Error) as e:
                # DB corruption or filesystem errors should be visible in
                # production logs even though the dashboard can keep rendering.
                log.warning("skipping chart data for run %s: %s", run_id, e)
        if result:
            net_assets_cost.append({
                **meta,
                "final_net_assets": float(result.get("final_net_assets", 0.0) or 0.0),
                "usd": float(result.get("usd", 0.0) or 0.0),
                "cumulative_orders": int(result.get("cum_orders", 0) or 0),
                "cumulative_net_profit": float(result.get("net_profit", 0.0) or 0.0),
            })
    weeks = sorted(all_weeks)
    weekly_metric_keys = [
        "effective_window_rate",
        "total_tool_calls",
        "sourcing_calls",
        "listing_action_ui_calls",
        "listing_action_calls",
        *[spec["metric"] for spec in LISTING_ACTION_TOOL_SPECS],
        *[spec["metric"] for spec in SOURCING_TOOL_CALL_SPECS],
        "shelf_product_count",
        "shelf_utilization_rate",
        "sell_through_capacity_rate",
        "sell_through_active_shelf_rate",
        "sell_through_rate",
        "weekly_new_unique_products",
        "weekly_gmv",
        "weekly_profit",
        *RUNTIME_HEALTH_METRICS,
    ]
    virtual_start_dates = {
        str(meta.get("virtual_start_date"))
        for meta in runs_meta
        if meta.get("virtual_start_date")
    }
    common_virtual_start_date = (
        next(iter(virtual_start_dates))
        if runs_meta
        and all(meta.get("virtual_start_date") for meta in runs_meta)
        and len(virtual_start_dates) == 1
        else None
    )
    weekly_payload = _period_chart_payload(
        index_key="weeks",
        indices=weeks,
        metric_keys=weekly_metric_keys,
        period_runs=weekly_runs,
        start_date=common_virtual_start_date,
    )
    months = sorted(all_months)
    common_start_value = (
        date.fromisoformat(common_virtual_start_date)
        if common_virtual_start_date
        else None
    )
    monthly_payload = _period_chart_payload(
        index_key="months",
        indices=months,
        metric_keys=weekly_metric_keys,
        period_runs=monthly_runs,
        start_date=common_virtual_start_date,
        descriptors=[
            _month_period_descriptor(month, common_start_value)
            for month in months
        ],
    )
    return {
        "runs": runs_meta,
        "net_assets": net_assets,
        "net_assets_cost": net_assets_cost,
        "cum_gmv": cum_gmv,
        "cum_net_profit": cum_net_profit,
        "cum_fine": cum_fine,
        "cum_orders": cum_orders,
        "cum_order_anomalies": cum_order_anomalies,
        "active_listings": active_listings,
        "shop_rating_score": shop_rating_score,
        "average_product_price": average_product_price,
        "average_product_margin": average_product_margin,
        "average_product_rating": average_product_rating,
        "tool_calls": {
            "tools": [
                name for name, _ in sorted(
                    tool_totals.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "runs": tool_call_runs,
            "categories": _tool_category_rows(tool_totals),
            "category_specs": [
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "color": spec["color"],
                }
                for spec in TOOL_CATEGORY_SPECS
            ],
            "tool_category_map": TOOL_CATEGORY_BY_NAME,
            "listing_tools": sorted(LISTING_UI_TOOL_NAMES),
        },
        "weekly": weekly_payload,
        "monthly": monthly_payload,
    }


def merge_chart_payloads(
    payloads: list[dict],
    run_results: list[dict],
) -> dict:
    """Merge cached chart groups and restore global leaderboard ordering.

    This lets the dashboard reuse immutable terminal-run analytics while only
    rebuilding the much smaller set of live runs.
    """
    payloads = [payload for payload in payloads if payload]
    ordered_results, rank_by_run = _run_results_in_leaderboard_order(run_results)
    run_order = {
        str(row.get("run_id")): index
        for index, row in enumerate(ordered_results)
    }
    allowed = set(run_order)

    def ranked_rows(rows: list[dict]) -> list[dict]:
        by_run = {}
        for row in rows:
            run_id = str(row.get("run_id"))
            if run_id not in allowed:
                continue
            by_run[run_id] = {
                **row,
                "rank": rank_by_run.get(run_id),
            }
        return [
            by_run[run_id]
            for run_id in sorted(by_run, key=run_order.get)
        ]

    simple_keys = [
        "runs",
        "net_assets",
        "net_assets_cost",
        "cum_gmv",
        "cum_net_profit",
        "cum_fine",
        "cum_orders",
        "cum_order_anomalies",
        "active_listings",
        "shop_rating_score",
        "average_product_price",
        "average_product_margin",
        "average_product_rating",
    ]
    merged = {
        key: ranked_rows([
            row
            for payload in payloads
            for row in (payload.get(key) or [])
        ])
        for key in simple_keys
    }

    tool_runs = ranked_rows([
        row
        for payload in payloads
        for row in ((payload.get("tool_calls") or {}).get("runs") or [])
    ])
    tool_totals: Counter[str] = Counter()
    for row in tool_runs:
        tool_totals.update({
            str(name): int(count)
            for name, count in (row.get("counts") or {}).items()
        })
    merged["tool_calls"] = {
        "tools": [
            name for name, _ in sorted(
                tool_totals.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "runs": tool_runs,
        "categories": _tool_category_rows(tool_totals),
        "category_specs": [
            {
                "key": spec["key"],
                "label": spec["label"],
                "color": spec["color"],
            }
            for spec in TOOL_CATEGORY_SPECS
        ],
        "tool_category_map": TOOL_CATEGORY_BY_NAME,
        "listing_tools": sorted(LISTING_UI_TOOL_NAMES),
    }

    def merge_period(name: str, index_key: str) -> dict:
        sources = [
            payload.get(name) or {}
            for payload in payloads
            if payload.get(name)
        ]
        indices = sorted({
            int(index)
            for source in sources
            for index in (source.get(index_key) or [])
        })
        start_dates = {
            str(source.get("start_date"))
            for source in sources
            if source.get("start_date")
        }
        start_date = (
            next(iter(start_dates))
            if sources
            and all(source.get("start_date") for source in sources)
            and len(start_dates) == 1
            else None
        )
        out = {
            index_key: indices,
            "start_date": start_date,
            "metrics": {},
            "listing_action_tools": [
                {"key": "all", "label": "All Actions", "metric": "listing_action_calls"},
                *[
                    {
                        "key": spec["key"],
                        "label": spec["label"],
                        "metric": spec["metric"],
                    }
                    for spec in LISTING_ACTION_TOOL_SPECS
                ],
            ],
            "sourcing_tools": [
                {"key": "all", "label": "All Sourcing", "metric": "sourcing_calls"},
                *[
                    {
                        "key": spec["key"],
                        "label": spec["label"],
                        "metric": spec["metric"],
                    }
                    for spec in SOURCING_TOOL_CALL_SPECS
                ],
            ],
        }
        metric_keys = []
        for source in sources:
            for key in (source.get("metrics") or {}):
                if key not in metric_keys:
                    metric_keys.append(key)
        for key in metric_keys:
            remapped_rows = []
            for source in sources:
                source_indices = [
                    int(index) for index in (source.get(index_key) or [])
                ]
                for row in ((source.get("metrics") or {}).get(key) or []):
                    values_by_index = dict(zip(
                        source_indices,
                        row.get("values") or [],
                    ))
                    remapped = {
                        **row,
                        "values": [values_by_index.get(index) for index in indices],
                    }
                    for aux_key, default in (
                        ("coverage_values", "unavailable"),
                        ("active_values", False),
                    ):
                        if aux_key not in row:
                            continue
                        aux_by_index = dict(zip(
                            source_indices,
                            row.get(aux_key) or [],
                        ))
                        remapped[aux_key] = [
                            aux_by_index.get(index, default)
                            for index in indices
                        ]
                    remapped_rows.append(remapped)
            out["metrics"][key] = ranked_rows(remapped_rows)
        if name == "monthly":
            try:
                common_start = (
                    date.fromisoformat(start_date)
                    if start_date else None
                )
            except ValueError:
                common_start = None
            out["periods"] = [
                _month_period_descriptor(index, common_start)
                for index in indices
            ]
        return out

    merged["weekly"] = merge_period("weekly", "weeks")
    merged["monthly"] = merge_period("monthly", "months")
    return merged
