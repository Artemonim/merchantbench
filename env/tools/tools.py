"""Agent-facing tool layer. Pure Python functions; HTTP routes are thin wrappers.

Each tool function returns a JSON-serializable dict. Tools that mutate the environment
take `env` + `agent_id` and operate under env.lock.

Catalog tools expose only public marketplace fields: market_brief series,
hot_search_terms keyword discovery, search_products catalog retrieval, product
details, and supplier pages.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import date, timedelta
from typing import Any, Optional, get_args

from compat import LEGACY_MEMORY_VERSION_MARKERS, MEMORY_VERSION_MARKER
from core import listing_rating as lr_mod
from core import sim_time
from core.demand import MIN_SALE_PRICE
from core.economy_v6 import EconomyV6, public_return_rate
from core.entities import Cash, EventLog, Order, OrderStatus, StoreListing
from core.inventory import effective_quantity
from core.simulator import Environment
from data import daily_reports
from storage import agent_log
from storage import db as dbm
from tools.hot_search import HotSearchIndex
from tools.table import compact_table

_ORDER_STATUS_SET = set(get_args(OrderStatus))
_OPEN_ORDER_STATUSES = ("ordered", "late", "shipped", "delivered")
_ORDER_STATUS_ORDER = tuple(get_args(OrderStatus))
PAGE_MAX = 1_000_000
_DIRECT_PRODUCT_PENALTY_EVENTS = (
    "order_stockout_violation",
    "order_insufficient_balance_violation",
)
_PENALTY_EVENT_TYPES = (
    "order_late",
    "order_cancelled",
    "order_settled_refund",
    "order_settled_only_refund",
    "order_settled_bad_review",
    *_DIRECT_PRODUCT_PENALTY_EVENTS,
)
_PUBLIC_PRODUCT_COLUMNS = (
    "product_id",
    "name",
    "quantity",
    "price",
    "supplier_id",
    "supplier_name",
    "supplier_ship_hours",
    "logistics_hours",
    "category",
    "historical_avg_rating",
    "shop_rating",
    "supplier_age_years",
)
_V6_PUBLIC_PRODUCT_COLUMNS = (
    "return_rate",
    "return_buyer_rate",
)
_ORDER_FEE_COLUMNS = (
    "commission_amount",
    "logistics_fee",
    "reverse_logistics_fee",
)
_LISTING_COLUMNS = (
    "product_id",
    "name",
    "sale_price",
    "supplier_price",
    "supplier_ship_hours",
    "supplier_logistics_hours",
    "procured_orders",
    "cum_gross_profit",
    "cum_net_profit",
    "cum_fine",
    "listing_rating",
)
_ORDER_SUMMARY_COLUMNS = (
    "order_id",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "current_status",
    "order_time",
    "status_age_hours",
    "expected_delivery_time",
    "delivered_time",
    "sale_price",
    "purchase_price",
    "total_penalty",
    "net_profit",
    "profit_finalized",
)
_ORDER_UPDATE_COLUMNS = (
    "order_id",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "previous_status",
    "current_status",
    "order_time",
    "status_age_hours",
    "expected_delivery_time",
    "delivered_time",
    "sale_price",
    "purchase_price",
    "total_penalty",
    "net_profit",
    "profit_finalized",
)
_MY_ORDER_COLUMNS = (
    "order_id",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "order_time",
    "sale_price",
    "purchase_price",
    "current_status",
    "supplier_ship_hours",
    "supplier_logistics_hours",
    "actual_logistics_hours",
    "realized_revenue",
    "realized_cost",
    "total_penalty",
    "net_profit",
    "profit_finalized",
)
_PRODUCT_SALES_COLUMNS = (
    "product_id",
    "name",
    "category",
    "supplier_id",
    "current_sale_price",
    "current_supplier_price",
    "current_gross_margin_rate",
    "orders",
    "gmv",
    "gross_profit",
    "net_profit",
    "fine",
    "late_count",
    "stockout_count",
    "insufficient_balance_count",
    "refund_count",
    "bad_review_count",
)
_REVIEW_LISTING_COLUMNS = (
    "product_id",
    "name",
    "listing_age_days",
    "days_without_sales",
    "procured_orders",
    "fine",
    "open_orders",
    "listing_rating",
)


# ---------- 虚拟时间转换 (agent-facing surface) ----------
# Internal storage uses raw integer ticks (env.t, Order.order_t, etc.). The
# agent must only see virtual {day, hour}. These helpers do the conversion at
# the tool boundary; entities.py and the simulator core are unchanged.


def _step_hours(env: Environment) -> int:
    return int(env.scenario["run"]["step_hours"])


def t_to_dh(t: int, step_hours: int) -> dict:
    """Convert raw tick to {day, hour}. day is 1-indexed, hour is 0-23."""
    return sim_time.legacy_day_hour(t, step_hours)


def t_to_dh_optional(t: Optional[int], step_hours: int) -> Optional[dict]:
    return None if t is None else t_to_dh(t, step_hours)


def t_to_agent_time(env: Environment, t: int) -> dict:
    return sim_time.time_view(env.scenario, t, _step_hours(env))


def t_to_agent_time_optional(env: Environment, t: Optional[int]) -> Optional[dict]:
    return None if t is None else t_to_agent_time(env, t)


def _compact_agent_time(env: Environment, t: Optional[int]) -> Optional[str]:
    if t is None:
        return None
    view = t_to_agent_time(env, int(t))
    return f"D{int(view['day'])}H{int(view['hour'])}"


def _round_money(value: Any) -> float:
    return round(float(value), 2)


def _economy_v6(env: Environment) -> EconomyV6:
    eco = getattr(env, "economy_v6", None)
    if eco is not None:
        return eco
    return EconomyV6.from_scenario(getattr(env, "scenario", None))


def _economy_v6_enabled(env: Environment) -> bool:
    return bool(_economy_v6(env).enabled)


def _public_product_columns(env: Environment) -> tuple[str, ...]:
    if _economy_v6_enabled(env):
        return _PUBLIC_PRODUCT_COLUMNS + _V6_PUBLIC_PRODUCT_COLUMNS
    return _PUBLIC_PRODUCT_COLUMNS


def _with_order_fee_columns(
    env: Environment,
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    if not _economy_v6_enabled(env):
        return columns
    cols = list(columns)
    insert_at = cols.index("net_profit") if "net_profit" in cols else len(cols)
    for offset, name in enumerate(_ORDER_FEE_COLUMNS):
        if name not in cols:
            cols.insert(insert_at + offset, name)
    return tuple(cols)


def _order_amount(source: Any, key: str) -> float:
    if isinstance(source, Order):
        return float(getattr(source, key, 0.0) or 0.0)
    try:
        value = source[key]
    except (KeyError, IndexError, TypeError):
        return 0.0
    return float(value or 0.0)


def _order_net_profit_money(source: Any) -> float:
    """Fee-aware order P&L matching ``Order.net_profit``."""
    if isinstance(source, Order):
        return _round_money(source.net_profit)
    return _round_money(
        _order_amount(source, "realized_revenue")
        - _order_amount(source, "realized_cost")
        - _order_amount(source, "total_penalty")
        - _order_amount(source, "commission_amount")
        - _order_amount(source, "logistics_fee")
        - _order_amount(source, "reverse_logistics_fee")
    )


def _order_fee_fields(env: Environment, source: Any) -> dict[str, float]:
    if not _economy_v6_enabled(env):
        return {}
    return {
        "commission_amount": _round_money(
            _order_amount(source, "commission_amount"),
        ),
        "logistics_fee": _round_money(_order_amount(source, "logistics_fee")),
        "reverse_logistics_fee": _round_money(
            _order_amount(source, "reverse_logistics_fee"),
        ),
    }


def _cash_to_agent_dict(cash: Cash) -> dict:
    return {
        "balance": _round_money(cash.balance),
        "deposit_pool": _round_money(cash.deposit_pool),
        "in_transit": _round_money(cash.in_transit),
        "receivable": _round_money(cash.receivable),
        "cumulative_fine": _round_money(cash.cumulative_fine),
    }


def _coerce_statuses(
    value: Any, *, allowed: tuple[str, ...], default: tuple[str, ...]
) -> tuple[Optional[dict], list[str]]:
    if value is None or value == []:
        return None, list(default)
    if isinstance(value, str):
        statuses = [value]
    elif isinstance(value, list):
        statuses = [str(v) for v in value]
    else:
        return {"ok": False, "error": "statuses must be a list of strings"}, []
    invalid = [s for s in statuses if s not in allowed]
    if invalid:
        return {
            "ok": False,
            "error": "invalid statuses; must be one of: " + ", ".join(allowed),
        }, []
    return None, statuses


def _product_fines_by_event_time(
    env: Environment,
    agent_id: str,
    t_from: int,
    t_to: int,
    *,
    product_ids: Optional[set[str]] = None,
) -> dict[str, float]:
    """Aggregate penalties by the event timestamp that charged them.

    Order.total_penalty is cumulative and is therefore unsuitable for interval
    attribution. Lifecycle penalty events identify an order; immediate
    procurement failures identify the product directly.
    """
    event_qmarks = ",".join("?" for _ in _PENALTY_EVENT_TYPES)
    rows = env.conn.execute(
        f"SELECT e.event_type, e.entity_id, e.payload, o.product_id"
        f" FROM events e LEFT JOIN orders o"
        f" ON o.run_id=e.run_id AND o.order_id=e.entity_id"
        f" WHERE e.run_id=? AND e.agent_id=? AND e.t BETWEEN ? AND ?"
        f" AND e.event_type IN ({event_qmarks})",
        (
            env.run_id,
            agent_id,
            int(t_from),
            int(t_to),
            *_PENALTY_EVENT_TYPES,
        ),
    ).fetchall()
    selected = {str(pid) for pid in product_ids} if product_ids is not None else None
    out: dict[str, float] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if row["event_type"] in _DIRECT_PRODUCT_PENALTY_EVENTS:
            product_id = payload.get("product_id") or row["entity_id"]
        else:
            product_id = row["product_id"] or payload.get("product_id")
        if not product_id:
            continue
        product_id = str(product_id)
        if selected is not None and product_id not in selected:
            continue
        try:
            penalty = float(payload.get("penalty") or 0.0)
        except (TypeError, ValueError):
            penalty = 0.0
        out[product_id] = out.get(product_id, 0.0) + penalty
    return {pid: round(value, 2) for pid, value in out.items()}


def _status_logs_for_orders(env: Environment, order_ids: list[str]) -> dict[str, list[dict]]:
    if not order_ids:
        return {}
    qmarks = ",".join("?" for _ in order_ids)
    rows = env.conn.execute(
        f"SELECT order_id, t, status FROM order_status"
        f" WHERE run_id=? AND order_id IN ({qmarks})"
        f" ORDER BY order_id ASC, t ASC, {dbm._status_rank_sql()} ASC",
        (env.run_id, *order_ids),
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["order_id"], []).append(
            {
                "t": int(row["t"]),
                "status": row["status"],
            }
        )
    return out


def _current_status_t(order_row: Any, log_rows: list[dict]) -> int:
    status = order_row["current_status"]
    for row in reversed(log_rows):
        if row["status"] == status:
            return int(row["t"])
    if status == "late" and order_row["late_t"] is not None:
        return int(order_row["late_t"])
    if status == "shipped" and order_row["shipped_t"] is not None:
        return int(order_row["shipped_t"])
    if status == "delivered" and order_row["delivered_t"] is not None:
        return int(order_row["delivered_t"])
    if status.startswith("settled_") and order_row["settled_t"] is not None:
        return int(order_row["settled_t"])
    return int(order_row["order_t"])


def _order_field(order: Any, key: str) -> Any:
    try:
        return order[key]
    except (KeyError, TypeError):
        return getattr(order, key)


def _expected_delivery_t_for_order(
    env: Environment,
    order_row: Any,
) -> Optional[int]:
    status = str(_order_field(order_row, "current_status") or "")
    if status in ("stockout", "insufficient_balance"):
        return None
    shipped_t = _order_field(order_row, "shipped_t")
    actual_logistics = int(_order_field(order_row, "actual_logistics_hours") or 0)
    if shipped_t is not None and actual_logistics > 0:
        return int(shipped_t) + actual_logistics
    purchase_t = _order_field(order_row, "purchase_t")
    product_id = _order_field(order_row, "product_id")
    product = env.products.get(product_id)
    supplier_ship = int(
        _order_field(order_row, "supplier_ship_hours")
        or _order_field(order_row, "actual_ship_hours")
        or (product.supplier_ship_hours if product else 0)
        or 0
    )
    logistics = int(
        _order_field(order_row, "actual_logistics_hours") or (product.logistics_hours if product else 0) or 0
    )
    if purchase_t is not None and supplier_ship > 0 and logistics > 0:
        return int(purchase_t) + supplier_ship + logistics
    promised_delivery_t = _order_field(order_row, "promised_delivery_t")
    if promised_delivery_t is not None:
        return int(promised_delivery_t)
    return None


def _compact_order_row(env: Environment, order_row: Any, log_rows: list[dict]) -> dict:
    status_t = _current_status_t(order_row, log_rows)
    status_age_h = max(0, (int(env.t) - status_t) * _step_hours(env))
    product = env.products.get(order_row["product_id"])
    net_profit = _order_net_profit_money(order_row)
    profit_finalized = order_row["settled_t"] is not None
    row = {
        "order_id": order_row["order_id"],
        "product_id": order_row["product_id"],
        "product_name": product.name if product else "",
        "supplier_id": order_row["supplier_id"],
        "supplier_name": product.supplier_name if product else "",
        "current_status": order_row["current_status"],
        "order_time": t_to_agent_time(env, order_row["order_t"]),
        "status_age_hours": status_age_h,
        "expected_delivery_time": t_to_agent_time_optional(
            env,
            _expected_delivery_t_for_order(env, order_row),
        ),
        "delivered_time": t_to_agent_time_optional(
            env,
            order_row["delivered_t"],
        ),
        "sale_price": _round_money(order_row["sale_price"]),
        "purchase_price": _round_money(order_row["purchase_price"]),
        "total_penalty": _round_money(order_row["total_penalty"] or 0.0),
        **_order_fee_fields(env, order_row),
        "net_profit": net_profit,
        "profit_finalized": profit_finalized,
    }
    return row


def _previous_status(log_rows: list[dict], first_t: int, first_status: str) -> Optional[str]:
    prev = None
    for row in log_rows:
        if row["t"] == first_t and row["status"] == first_status:
            return prev
        prev = row["status"]
    return prev


def day_range_to_t(
    day_from: Optional[int], day_to: Optional[int], step_hours: int
) -> tuple[Optional[int], Optional[int]]:
    """Inclusive day range -> inclusive raw-tick range. day 1 = ticks
    [0, steps_per_day-1]. day_to = N covers up to and including the last
    tick of day N."""
    sh = int(step_hours)
    steps_per_day = max(1, 24 // sh)
    t_from = (int(day_from) - 1) * steps_per_day if day_from is not None else None
    t_to = int(day_to) * steps_per_day - 1 if day_to is not None else None
    return t_from, t_to


def _current_virtual_day(env: Environment) -> int:
    return int(sim_time.legacy_day_hour(env.t, _step_hours(env))["day"])


def _coerce_optional_day_range(
    env: Environment,
    day_from: Optional[Any],
    day_to: Optional[Any],
) -> tuple[Optional[dict], Optional[int], Optional[int]]:
    parsed: dict[str, Optional[int]] = {"day_from": None, "day_to": None}
    for name, value in (("day_from", day_from), ("day_to", day_to)):
        if value is None:
            continue
        try:
            day = int(value)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"{name} must be an integer"}, None, None
        if day < 1:
            return {"ok": False, "error": f"{name} must be >= 1"}, None, None
        parsed[name] = day

    df = parsed["day_from"]
    dt = parsed["day_to"]
    if df is not None and dt is not None and df > dt:
        return {"ok": False, "error": "day_from must be <= day_to"}, None, None

    current_day = _current_virtual_day(env)
    if df is not None and df > current_day:
        return (
            {
                "ok": False,
                "error": f"day_from cannot exceed current day ({current_day})",
                "current_day": current_day,
            },
            None,
            None,
        )
    if dt is not None and dt > current_day:
        return (
            {
                "ok": False,
                "error": f"day_to cannot exceed current day ({current_day})",
                "current_day": current_day,
            },
            None,
            None,
        )

    return None, df, dt


# ---------- 选品 ----------

_VISIBLE_PRODUCT_KEYS = {
    "product_id",
    "name",
    "quantity",
    "price",
    "supplier_id",
    "supplier_name",
    "supplier_ship_hours",
    "logistics_hours",
    "category",
    "historical_avg_rating",
    "shop_rating",
    "supplier_age_years",
}


def _public_product(p, current_t: int | None = None, env: Environment | None = None) -> dict:
    out = {k: v for k, v in p.visible().items() if k in _VISIBLE_PRODUCT_KEYS}
    if current_t is not None:
        out["quantity"] = effective_quantity(p, current_t)
    if env is not None and _economy_v6_enabled(env):
        out["return_rate"] = public_return_rate(
            getattr(p, "refund_rate", 0.0),
            getattr(p, "only_refund_rate", 0.0),
        )
        out["return_buyer_rate"] = float(p.return_buyer_rate)
    return out


def _market_sales_last_n(env: Environment, category: str, days: int) -> tuple[list[float], float]:
    products = [p for p in env.products.values() if p.category == category]
    if not products:
        return [0.0 for _ in range(days)], 0.0
    step_hours = int(env.scenario["run"]["step_hours"])
    small_share = float((env.scenario.get("data") or {}).get("small_share", 1.0))
    latest_completed_idx = (sim_time.curve_day_index(env.scenario, env.t, step_hours) - 1) % 365
    daily = []
    for d in range(days - 1, -1, -1):
        total = 0.0
        day_idx = (latest_completed_idx - d) % 365
        for p in products:
            total += p.market_curve[day_idx]
        daily.append(total * small_share)
    daily_avg = sum(daily) / days if days > 0 else 0.0
    return daily, daily_avg


def _market_gmv_last_n(env: Environment, category: str, days: int) -> tuple[list[float], float]:
    """Daily GMV series for *category* over the last *days* days.

    GMV per product per day = market_curve[day] * small_share * ref_price.
    ref_price is stable across supplier price-change events, so historical
    market GMV is not rewritten by a current procurement-cost change.
    Returns (daily_series, daily_avg).
    """
    products = [p for p in env.products.values() if p.category == category]
    if not products:
        return [0.0 for _ in range(days)], 0.0
    step_hours = int(env.scenario["run"]["step_hours"])
    small_share = float((env.scenario.get("data") or {}).get("small_share", 1.0))
    latest_completed_idx = (sim_time.curve_day_index(env.scenario, env.t, step_hours) - 1) % 365
    daily = []
    for d in range(days - 1, -1, -1):
        total = 0.0
        day_idx = (latest_completed_idx - d) % 365
        for p in products:
            total += p.market_curve[day_idx] * p.ref_price
        daily.append(total * small_share)
    daily_avg = sum(daily) / days if days > 0 else 0.0
    return daily, daily_avg


def _category_avg_price(env: Environment, category: str) -> float:
    """Mean supplier price across all products in *category*."""
    prices = [p.price for p in env.products.values() if p.category == category]
    return sum(prices) / len(prices) if prices else 0.0


def market_brief(env: Environment, window_days: int = 7) -> dict:
    try:
        days = int(window_days)
    except (TypeError, ValueError):
        return {"ok": False, "error": "window_days must be 7 or 30"}
    if days not in (7, 30):
        return {"ok": False, "error": "window_days must be 7 or 30"}

    categories = []
    for category in sorted({p.category for p in env.products.values()}):
        daily, daily_avg = _market_sales_last_n(env, category, days)
        gmv_daily, gmv_avg = _market_gmv_last_n(env, category, days)
        avg_price = _category_avg_price(env, category)
        categories.append(
            {
                "category": category,
                "total_sales": [round(value) for value in daily],
                "daily_avg_sales": round(daily_avg),
                "total_gmv": [round(value, 2) for value in gmv_daily],
                "daily_avg_gmv": round(gmv_avg, 2),
                "avg_price": round(avg_price, 2),
            }
        )
    return {"ok": True, "window_days": days, "categories": categories}


def hot_search_terms(env: Environment, category: Optional[str] = None, window_days: int = 7) -> dict:
    try:
        days = int(window_days)
    except (TypeError, ValueError):
        return {"ok": False, "error": "window_days must be 7 or 30"}
    if days not in (7, 30):
        return {"ok": False, "error": "window_days must be 7 or 30"}
    # Normalize category: None means all categories, but empty string is also
    # treated as "no filter" to avoid silent coercion surprises.
    if category is not None:
        category = str(category)
        if not category:
            category = None

    step_hours = int(env.scenario["run"]["step_hours"])
    small_share = float((env.scenario.get("data") or {}).get("small_share", 1.0))
    latest_completed_idx = (sim_time.curve_day_index(env.scenario, env.t, step_hours) - 1) % 365
    index = getattr(env, "_hot_search_index", None)
    if index is None:
        index = HotSearchIndex(env.products.values())
        env._hot_search_index = index
    trends = index.rank(
        category=category,
        window_days=days,
        today_idx=latest_completed_idx,
        small_share=small_share,
    )
    view = sim_time.time_view(env.scenario, env.t, step_hours)
    date_value = view.get("datetime")
    return {
        "day": view["day"],
        "date": date_value[:10] if date_value else None,
        "window_days": days,
        "category": category,
        "trends": compact_table(
            [
                {
                    "rank": row.rank,
                    "keyword": row.keyword,
                    "category": row.category,
                    "trend": row.trend,
                    "change_pct": row.change_pct,
                    "rank_change": row.rank_change,
                }
                for row in trends
            ],
            ("rank", "keyword", "category", "trend", "change_pct", "rank_change"),
        ),
    }


def _daily_report_date_text(env: Environment) -> str:
    step_hours = int(env.scenario["run"]["step_hours"])
    view = sim_time.time_view(env.scenario, env.t, step_hours)
    return str(view.get("datetime", ""))[:10]


def daily_report_notice_available(env: Environment, agent_id: str) -> bool:
    """Whether today's report exists and this agent has not read it yet."""
    date_text = _daily_report_date_text(env)
    if not date_text:
        return False
    read_dates = getattr(env, "daily_report_read_date_by_agent", {})
    if read_dates.get(agent_id) == date_text:
        return False
    data_cfg = env.scenario.get("data") or {}
    report_dir = daily_reports.resolve_report_dir(data_cfg.get("daily_report_dir"))
    report_date = date.fromisoformat(date_text)
    return os.path.isfile(daily_reports.report_path(report_dir, report_date))


def get_daily_report(env: Environment, agent_id: Optional[str] = None) -> dict:
    date_text = _daily_report_date_text(env)
    if not date_text:
        return {"ok": False, "error": "daily report requires virtual_time.start_date"}
    report_date = date.fromisoformat(date_text)
    data_cfg = env.scenario.get("data") or {}
    report_dir = daily_reports.resolve_report_dir(data_cfg.get("daily_report_dir"))
    content = daily_reports.read_report(report_dir, report_date)
    if content is None:
        return {"ok": False, "error": f"daily report not found for {date_text}"}
    result = {
        "ok": True,
        "report_date": report_date.isoformat(),
        "data_as_of": (report_date - timedelta(days=1)).isoformat(),
        "content": content,
    }
    if agent_id is not None:
        # A successful tool read suppresses only this agent's notice for this
        # simulation date. Persist it with the observation cursor so a process
        # restart does not reintroduce the same notice.
        with env.turn_lock:
            read_dates = getattr(env, "daily_report_read_date_by_agent", None)
            if read_dates is None:
                read_dates = {}
                env.daily_report_read_date_by_agent = read_dates
            read_dates[agent_id] = date_text
            getattr(env, "observation_cache_by_agent_step", {}).pop((agent_id, int(env.t)), None)
            agent_log.persist_observation_state(
                env.runs_root,
                env.run_id,
                env.last_observation_step_by_agent,
                windows_by_agent_step=env.observation_window_by_agent_step,
                daily_report_read_dates_by_agent=read_dates,
            )
    return result


def _optional_float(name: str, value: Optional[Any]) -> tuple[Optional[dict], Optional[float]]:
    if value is None:
        return None, None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"{name} must be a number"}, None
    if not math.isfinite(number):
        return {"ok": False, "error": f"{name} must be a finite number"}, None
    return None, number


def _optional_int(name: str, value: Optional[Any]) -> tuple[Optional[dict], Optional[int]]:
    if value is None:
        return None, None
    try:
        return None, int(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"{name} must be an integer"}, None


def _page_args(
    page: int = 1,
    page_size: int = 20,
    *,
    maximum: int = 50,
) -> tuple[Optional[dict], int, int]:
    try:
        page_i = int(page)
        page_size_i = int(page_size)
    except (TypeError, ValueError):
        return {"ok": False, "error": "page and page_size must be integers"}, 1, 20
    if page_i < 1:
        return {"ok": False, "error": "page must be >= 1"}, page_i, page_size_i
    if page_i > PAGE_MAX:
        return (
            {
                "ok": False,
                "error": f"page must be <= {PAGE_MAX}",
            },
            page_i,
            page_size_i,
        )
    if page_size_i < 1 or page_size_i > maximum:
        return (
            {
                "ok": False,
                "error": f"page_size must be in [1, {maximum}]",
            },
            page_i,
            page_size_i,
        )
    return None, page_i, page_size_i


def search_products(
    env: Environment,
    query: str = "",
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    supplier_rating_min: Optional[float] = None,
    historical_rating_min: Optional[float] = None,
    logistics_hours_max: Optional[int] = None,
    supplier_ship_hours_max: Optional[int] = None,
    delivery_hours_max: Optional[int] = None,
    quantity_min: Optional[int] = None,
    sort_by: str = "relevance",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    query_text = str(query or "")
    if re.fullmatch(r"[\u4e00-\u9fff]", query_text.strip()):
        return {
            "ok": False,
            "error": "单字搜索不支持，请至少输入两个汉字",
        }
    err, page_i, page_size_i = _page_args(page, page_size)
    if err is not None:
        return err
    sort_by = str(sort_by or "relevance")
    allowed_sorts = {
        "relevance",
        "price_asc",
        "price_desc",
        "rating",
        "supplier_rating",
        "logistics_speed",
    }
    if sort_by not in allowed_sorts:
        return {"ok": False, "error": f"sort_by must be one of {sorted(allowed_sorts)}"}

    numeric_args = [
        ("price_min", price_min, _optional_float),
        ("price_max", price_max, _optional_float),
        ("supplier_rating_min", supplier_rating_min, _optional_float),
        ("historical_rating_min", historical_rating_min, _optional_float),
        ("logistics_hours_max", logistics_hours_max, _optional_int),
        ("supplier_ship_hours_max", supplier_ship_hours_max, _optional_int),
        ("delivery_hours_max", delivery_hours_max, _optional_int),
        ("quantity_min", quantity_min, _optional_int),
    ]
    parsed: dict[str, Any] = {}
    for name, value, parser in numeric_args:
        err, parsed_value = parser(name, value)
        if err is not None:
            return err
        parsed[name] = parsed_value

    offset = (page_i - 1) * page_size_i
    overlay_dirty = bool(getattr(env, "_catalog_sql_overlay_dirty", False))
    rows = dbm.search_products_sql(
        env.conn,
        env.run_id,
        query=query_text,
        filters=parsed,
        sort_by=sort_by,
        limit=(offset + page_size_i + 1) if overlay_dirty else (page_size_i + 1),
        offset=0 if overlay_dirty else offset,
        current_t=env.t,
    )

    def _visible_items(sql_rows: list[dict]) -> tuple[list[dict], bool]:
        visible = []
        saw_stale = False
        for row in sql_rows:
            p = env.products.get(row["product_id"])
            if p is not None:
                if not p.is_listed_by_supplier:
                    saw_stale = True
                    continue
                visible.append(_public_product(p, env.t, env=env))
            else:
                visible.append(row)
        return visible, saw_stale

    visible_rows, saw_stale = _visible_items(rows)
    if saw_stale and not overlay_dirty:
        env._catalog_sql_overlay_dirty = True
        rows = dbm.search_products_sql(
            env.conn,
            env.run_id,
            query=query_text,
            filters=parsed,
            sort_by=sort_by,
            limit=offset + page_size_i + 1,
            offset=0,
            current_t=env.t,
        )
        visible_rows, _ = _visible_items(rows)

    page_start = offset if (overlay_dirty or saw_stale) else 0
    page_window = visible_rows[page_start : page_start + page_size_i + 1]
    items = []
    for item in page_window[:page_size_i]:
        items.append(item)
    has_next = len(page_window) > page_size_i
    return {
        "query": query_text,
        "sort_by": sort_by,
        "page": page_i,
        "page_size": page_size_i,
        "has_next": has_next,
        "items": compact_table(items, _public_product_columns(env)),
    }


def get_product_detail(env: Environment, agent_id: str, product_id: str) -> Optional[dict]:
    p = env.products.get(product_id)
    if p is None:
        return None
    if not p.is_listed_by_supplier:
        listing = dbm.get_listing(
            env.conn,
            env.run_id,
            agent_id,
            product_id,
        )
        if listing is None:
            return None
    return {
        **_public_product(p, env.t, env=env),
        "supplier_available": bool(p.is_listed_by_supplier),
    }


def get_supplier_profile(env: Environment, supplier_id: str) -> Optional[dict]:
    row = dbm.get_supplier_profile_sql(env.conn, env.run_id, supplier_id)
    if row is None:
        return None
    return {
        "supplier_id": row["supplier_id"],
        "supplier_name": row["supplier_name"],
        "shop_rating": row["shop_rating"],
        "return_buyer_rate": row["return_buyer_rate"],
        "supplier_age_years": row["supplier_age_years"],
        "product_count": int(row["product_count"] or 0),
    }


def list_supplier_products(env: Environment, supplier_id: str, page: int = 1, page_size: int = 20) -> dict:
    err, page_i, page_size_i = _page_args(page, page_size)
    if err is not None:
        return err
    start = (page_i - 1) * page_size_i
    target = start + page_size_i + 1
    visible = []
    sql_offset = 0
    fetch_limit = target
    while len(visible) < target:
        rows = dbm.list_supplier_products_sql(
            env.conn,
            env.run_id,
            supplier_id,
            limit=fetch_limit,
            offset=sql_offset,
            current_t=env.t,
        )
        sql_offset += len(rows)
        for row in rows:
            product = env.products.get(row["product_id"])
            if product is not None:
                if not product.is_listed_by_supplier:
                    continue
                row = _public_product(product, env.t, env=env)
            visible.append(row)
        if len(rows) < fetch_limit:
            break
        fetch_limit = max(page_size_i + 1, target - len(visible))
    page_window = visible[start:target]
    return {
        "supplier_id": supplier_id,
        "page": page_i,
        "page_size": page_size_i,
        "has_next": len(page_window) > page_size_i,
        "items": compact_table(
            page_window[:page_size_i],
            _public_product_columns(env),
        ),
    }


# ---------- 经营 (per-agent) ----------


def _alive_guard(env: Environment, agent_id: str) -> Optional[dict]:
    """Return an error dict if the agent is unknown or dead, else None."""
    st = env.agents.get(agent_id)
    if st is None:
        return {"ok": False, "error": f"unknown agent {agent_id}"}
    if not st.is_alive:
        return {
            "ok": False,
            "error": f"agent {agent_id} is dead (deposit exhausted)",
            "died_at": t_to_agent_time_optional(env, st.died_at_t),
        }
    return None


def _ship_promise_bounds(rules: dict) -> tuple[int, int]:
    default_h = int(rules.get("default_promised_ship_hours", 48))
    return default_h, default_h


def _write_agent_listing_event(
    env: Environment,
    agent_id: str,
    event_type: str,
    product_id: str,
    payload: dict,
) -> None:
    dbm.write_events(
        env.conn,
        env.run_id,
        [
            EventLog(
                t=env.t,
                event_type=event_type,
                entity_id=product_id,
                agent_id=agent_id,
                payload=payload,
            )
        ],
    )


_BATCH_MUTATION_MAX_ITEMS = 100
_LIST_PRODUCT_RESULT_COLUMNS = (
    "product_id",
    "ok",
    "error",
    "supplier_ship_hours",
)
_MUTATION_RESULT_COLUMNS = ("product_id", "ok", "error")


def _batch_items_error(items: Any) -> Optional[dict]:
    if not isinstance(items, list):
        return {"ok": False, "error": "items must be a list"}
    if len(items) < 1:
        return {"ok": False, "error": "items must contain at least 1 item"}
    if len(items) > _BATCH_MUTATION_MAX_ITEMS:
        return {
            "ok": False,
            "error": f"items must contain at most {_BATCH_MUTATION_MAX_ITEMS} items",
        }
    return None


def _unknown_item_args(tool_name: str, item: dict, allowed: set[str]) -> Optional[str]:
    unknown = sorted(set(item) - allowed)
    if not unknown:
        return None
    return f"unknown arguments for {tool_name} item: {', '.join(unknown)}"


def _list_product_locked(env: Environment, agent_id: str, product_id: str, sale_price: float) -> dict:
    guard = _alive_guard(env, agent_id)
    if guard is not None:
        return guard
    product = env.products.get(product_id)
    if product is None:
        return {"ok": False, "error": "unknown product"}
    try:
        sale_price_f = float(sale_price)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"sale_price must be at least {MIN_SALE_PRICE:.2f}"}
    if not math.isfinite(sale_price_f) or sale_price_f < MIN_SALE_PRICE:
        return {"ok": False, "error": f"sale_price must be at least {MIN_SALE_PRICE:.2f}"}
    rules = env.scenario["platform_rules"]
    # Preserve accumulators: check the active listing first, fall back to
    # orders table so that delist+relist keeps historical sales and rating data.
    existing = dbm.get_listing(env.conn, env.run_id, agent_id, product_id)
    if existing is None and not product.is_listed_by_supplier:
        return {
            "ok": False,
            "error": "supplier product is not currently available",
        }
    max_active = int(rules.get("max_active_listings", 100))
    if existing is None:
        active_count = len(dbm.list_listings(env.conn, env.run_id, agent_id))
        if active_count >= max_active:
            return {
                "ok": False,
                "error": f"max_active_listings={max_active} reached",
                "active_listings": active_count,
                "max_active_listings": max_active,
            }
    if existing:
        init_sales = existing.cum_sales
        init_revenue = existing.cum_revenue
        first_listed_at = existing.first_listed_at
        init_normal_count = existing.normal_count
        init_bad_review_count = existing.bad_review_count
        init_rating_sum = existing.rating_sum
        init_rating_count = existing.rating_count
    else:
        row = env.conn.execute(
            "SELECT"
            " COALESCE(SUM(CASE WHEN current_status NOT IN"
            "   ('stockout','insufficient_balance') THEN 1 ELSE 0 END), 0) AS n,"
            " COALESCE(SUM(CASE WHEN current_status NOT IN"
            "   ('stockout','insufficient_balance') THEN sale_price ELSE 0 END), 0) AS rev"
            " FROM orders WHERE run_id=? AND agent_id=? AND product_id=?",
            (env.run_id, agent_id, product_id),
        ).fetchone()
        init_sales = int(row["n"])
        init_revenue = float(row["rev"])
        first_listed_at = env.t
        counts_row = env.conn.execute(
            "SELECT"
            " SUM(CASE WHEN current_status='settled_normal' THEN 1 ELSE 0 END) AS nc,"
            " SUM(CASE WHEN current_status='settled_bad_review' THEN 1 ELSE 0 END) AS brc"
            " FROM orders WHERE run_id=? AND agent_id=? AND product_id=?",
            (env.run_id, agent_id, product_id),
        ).fetchone()
        init_normal_count = int(counts_row["nc"] or 0)
        init_bad_review_count = int(counts_row["brc"] or 0)
        if env._uses_order_outcome_rating():
            cutoff_t = env._last_completed_day_cutoff()
            rows = [
                row
                for row in dbm.load_order_rating_rows(
                    env.conn,
                    env.run_id,
                    agent_id,
                    cutoff_t,
                )
                if row[0] == product_id
            ]
            scores, weights = env._rating_outcome_cfg()
            lr_cfg = env.scenario.get("listing_rating") or {}
            evidence = lr_mod.rebuild_evidence(
                rows,
                cutoff_t=cutoff_t,
                step_hours=_step_hours(env),
                half_life_days=float(lr_cfg["half_life_days"]),
                scores=scores,
                weights=weights,
            ).get(product_id, (0.0, 0.0, 0))
            init_rating_sum = float(evidence[0])
            init_rating_count = float(evidence[1])
        else:
            rating_rows = env.conn.execute(
                "SELECT current_status, late_t"
                " FROM orders WHERE run_id=? AND agent_id=? AND product_id=?"
                " AND settled_t IS NOT NULL",
                (env.run_id, agent_id, product_id),
            ).fetchall()
            lr_cfg = env.scenario.get("listing_rating") or {}
            scores = {key: lr_cfg[key] for key in lr_mod.DEFAULT_OUTCOME_SCORES if key in lr_cfg}
            outcome_scores = [
                score
                for score in (
                    lr_mod.score_for_order_outcome(
                        row["current_status"],
                        row["late_t"],
                        scores,
                    )
                    for row in rating_rows
                )
                if score is not None
            ]
            init_rating_sum = float(sum(outcome_scores))
            init_rating_count = float(len(outcome_scores))
    listing = StoreListing(
        product_id=product_id,
        agent_id=agent_id,
        sale_price=sale_price_f,
        listed_at=existing.listed_at if existing is not None else env.t,
        first_listed_at=first_listed_at,
        cum_sales=init_sales,
        cum_revenue=init_revenue,
        normal_count=init_normal_count,
        bad_review_count=init_bad_review_count,
        rating_sum=init_rating_sum,
        rating_count=init_rating_count,
    )
    dbm.upsert_listing(env.conn, env.run_id, agent_id, listing)
    env.agents[agent_id].listings[product_id] = listing
    was_already = existing is not None
    _write_agent_listing_event(
        env,
        agent_id,
        "agent_list_product",
        product_id,
        {
            "sale_price": sale_price_f,
            "was_already_listed": was_already,
        },
    )
    result = {
        "ok": True,
        "supplier_ship_hours": int(product.supplier_ship_hours),
    }
    return result


def list_product(env: Environment, agent_id: str, items: list[dict]) -> dict:
    err = _batch_items_error(items)
    if err is not None:
        return err
    rows = []
    with env.lock:
        for item in items:
            if not isinstance(item, dict):
                rows.append(
                    {
                        "product_id": None,
                        "ok": False,
                        "error": "item must be an object",
                        "supplier_ship_hours": None,
                    }
                )
                continue
            product_id_i = item.get("product_id")
            unknown = _unknown_item_args(
                "list_product",
                item,
                {"product_id", "sale_price"},
            )
            if unknown is not None:
                rows.append(
                    {
                        "product_id": product_id_i,
                        "ok": False,
                        "error": unknown,
                        "supplier_ship_hours": None,
                    }
                )
                continue
            result = _list_product_locked(
                env,
                agent_id,
                product_id_i,
                item.get("sale_price"),
            )
            rows.append(
                {
                    "product_id": product_id_i,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                    "supplier_ship_hours": result.get("supplier_ship_hours"),
                }
            )
    return {
        "ok": all(row["ok"] for row in rows),
        "items": compact_table(rows, _LIST_PRODUCT_RESULT_COLUMNS),
    }


def _delist_product_locked(env: Environment, agent_id: str, product_id: str) -> dict:
    guard = _alive_guard(env, agent_id)
    if guard is not None:
        return guard
    existing = dbm.get_listing(env.conn, env.run_id, agent_id, product_id)
    if not existing:
        return {"ok": False, "error": "not currently listed"}
    _write_agent_listing_event(
        env,
        agent_id,
        "agent_delist_product",
        product_id,
        {
            "sale_price": existing.sale_price,
        },
    )
    dbm.delete_listing(env.conn, env.run_id, agent_id, product_id)
    env.agents[agent_id].listings.pop(product_id, None)
    return {"ok": True}


def delist_product(env: Environment, agent_id: str, items: list[dict]) -> dict:
    err = _batch_items_error(items)
    if err is not None:
        return err
    rows = []
    with env.lock:
        for item in items:
            if not isinstance(item, dict):
                rows.append({"product_id": None, "ok": False, "error": "item must be an object"})
                continue
            product_id_i = item.get("product_id")
            unknown = _unknown_item_args("delist_product", item, {"product_id"})
            if unknown is not None:
                rows.append(
                    {
                        "product_id": product_id_i,
                        "ok": False,
                        "error": unknown,
                    }
                )
                continue
            result = _delist_product_locked(env, agent_id, product_id_i)
            rows.append(
                {
                    "product_id": product_id_i,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                }
            )
    return {
        "ok": all(row["ok"] for row in rows),
        "items": compact_table(rows, _MUTATION_RESULT_COLUMNS),
    }


def _adjust_price_locked(env: Environment, agent_id: str, product_id: str, new_price: float) -> dict:
    guard = _alive_guard(env, agent_id)
    if guard is not None:
        return guard
    existing = dbm.get_listing(env.conn, env.run_id, agent_id, product_id)
    if not existing:
        return {"ok": False, "error": "not currently listed"}
    try:
        new_price_f = float(new_price)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"new_price must be at least {MIN_SALE_PRICE:.2f}"}
    if not math.isfinite(new_price_f) or new_price_f < MIN_SALE_PRICE:
        return {"ok": False, "error": f"new_price must be at least {MIN_SALE_PRICE:.2f}"}
    old_price = float(existing.sale_price)
    existing.sale_price = new_price_f
    dbm.upsert_listing(env.conn, env.run_id, agent_id, existing)
    if product_id in env.agents[agent_id].listings:
        env.agents[agent_id].listings[product_id].sale_price = new_price_f
    _write_agent_listing_event(
        env,
        agent_id,
        "agent_adjust_price",
        product_id,
        {
            "old_price": old_price,
            "new_price": new_price_f,
        },
    )
    return {"ok": True}


def adjust_price(env: Environment, agent_id: str, items: list[dict]) -> dict:
    err = _batch_items_error(items)
    if err is not None:
        return err
    rows = []
    with env.lock:
        for item in items:
            if not isinstance(item, dict):
                rows.append({"product_id": None, "ok": False, "error": "item must be an object"})
                continue
            product_id_i = item.get("product_id")
            unknown = _unknown_item_args("adjust_price", item, {"product_id", "new_price"})
            if unknown is not None:
                rows.append(
                    {
                        "product_id": product_id_i,
                        "ok": False,
                        "error": unknown,
                    }
                )
                continue
            result = _adjust_price_locked(env, agent_id, product_id_i, item.get("new_price"))
            rows.append(
                {
                    "product_id": product_id_i,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                }
            )
    return {
        "ok": all(row["ok"] for row in rows),
        "items": compact_table(rows, _MUTATION_RESULT_COLUMNS),
    }


def query_my_listings(env: Environment, agent_id: str) -> dict:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        per_listing_pnl = {}
        for r in env.conn.execute(
            "SELECT product_id,"
            " COALESCE(SUM(CASE WHEN current_status NOT IN"
            "   ('stockout','insufficient_balance')"
            "   THEN sale_price - purchase_price ELSE 0 END), 0) AS gross_profit,"
            " COALESCE(SUM(CASE WHEN settled_t IS NOT NULL"
            f"   THEN {dbm.order_net_profit_sql()} ELSE 0 END), 0) AS net_profit,"
            " COALESCE(SUM(total_penalty), 0) AS fine"
            " FROM orders WHERE run_id=? AND agent_id=?"
            " GROUP BY product_id",
            (env.run_id, agent_id),
        ).fetchall():
            per_listing_pnl[r["product_id"]] = {
                "cum_gross_profit": round(float(r["gross_profit"]), 2),
                "cum_net_profit": round(float(r["net_profit"]), 2),
                "cum_fine": round(float(r["fine"]), 2),
            }
        lr_cfg = env.scenario.get("listing_rating")
        initial_rating = float(lr_cfg.get("initial_rating", 4.0)) if lr_cfg else 4.0
        prior_weight = float(lr_cfg["prior_weight"]) if lr_cfg else 10.0
        out = []
        for l in dbm.list_listings(env.conn, env.run_id, agent_id):
            p = env.products.get(l.product_id)
            pnl = per_listing_pnl.get(l.product_id, {})
            lr = lr_mod.compute_listing_rating(
                initial_rating,
                l.rating_sum,
                l.rating_count,
                prior_weight,
            )
            out.append(
                {
                    "product_id": l.product_id,
                    "name": p.name if p else "",
                    "sale_price": l.sale_price,
                    "supplier_price": p.price if p else None,
                    "supplier_ship_hours": p.supplier_ship_hours if p else None,
                    "supplier_logistics_hours": p.logistics_hours if p else None,
                    "procured_orders": l.cum_sales,
                    "cum_gross_profit": pnl.get("cum_gross_profit", 0.0),
                    "cum_net_profit": pnl.get("cum_net_profit", 0.0),
                    "cum_fine": pnl.get("cum_fine", 0.0),
                    "listing_rating": round(lr, 2),
                }
            )
        return compact_table(out, _LISTING_COLUMNS)


def review_my_listings(
    env: Environment, agent_id: str, sort_by: str = "listing_age_days", window_days: int = 7
) -> dict:
    """Per-listing health review: age, sales velocity, penalty exposure,
    and fulfillment backlog. Surfaces products needing attention."""
    try:
        days = int(window_days)
    except (TypeError, ValueError):
        return {"ok": False, "error": "window_days must be 7 or 30"}
    if days not in (7, 30):
        return {"ok": False, "error": "window_days must be 7 or 30"}

    allowed_sort = {
        "listing_age_days",
        "days_without_sales",
        "fine",
        "procured_orders",
    }
    if not isinstance(sort_by, str) or sort_by not in allowed_sort:
        return {"ok": False, "error": f"sort_by must be one of {sorted(allowed_sort)}"}

    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard

        sh = _step_hours(env)
        # Rolling window in ticks.
        window_start_hour = max(0, int(env.t) * sh - days * 24)
        t_win_from = (window_start_hour + sh - 1) // sh
        t_win_to = int(env.t)

        listings = dbm.list_listings(env.conn, env.run_id, agent_id)
        if not listings:
            return {
                "sort_by": sort_by,
                "window_days": days,
                **compact_table([], _REVIEW_LISTING_COLUMNS),
            }

        pids = [l.product_id for l in listings]
        qmarks = ",".join("?" for _ in pids)
        base_params: tuple = (env.run_id, agent_id, *pids)

        # Successfully procured orders in the rolling window. Immediate
        # stockout/insufficient-balance failures remain order records but never
        # entered procurement, so they do not count toward listing velocity.
        o_win_rows = env.conn.execute(
            f"SELECT o.product_id, COUNT(*) AS n FROM orders o"
            " JOIN store_listings l"
            " ON l.run_id=o.run_id AND l.agent_id=o.agent_id"
            " AND l.product_id=o.product_id"
            f" WHERE o.run_id=? AND o.agent_id=? AND o.product_id IN ({qmarks})"
            # Demand for env.t is generated before the agent hook.  An order
            # created at the exact relist step therefore belongs to the prior
            # activation, not the listing created later in that hook.
            " AND o.order_t>l.listed_at"
            " AND o.order_t BETWEEN ? AND ?"
            " AND o.current_status NOT IN ('stockout','insufficient_balance')"
            " GROUP BY o.product_id",
            (*base_params, t_win_from, t_win_to),
        ).fetchall()
        procured_orders = {r["product_id"]: int(r["n"]) for r in o_win_rows}

        fine_win = _product_fines_by_event_time(
            env,
            agent_id,
            t_win_from,
            t_win_to,
            product_ids=set(pids),
        )

        # Successful sales since the current listing activation. Relisting starts
        # a fresh health window; failed order attempts do not reset no-sale days.
        sale_day_rows = env.conn.execute(
            f"SELECT o.product_id, MAX(o.order_t) AS max_t FROM orders o"
            " JOIN store_listings l"
            " ON l.run_id=o.run_id AND l.agent_id=o.agent_id"
            " AND l.product_id=o.product_id"
            f" WHERE o.run_id=? AND o.agent_id=? AND o.product_id IN ({qmarks})"
            " AND o.order_t>l.listed_at"
            " AND o.current_status NOT IN ('stockout','insufficient_balance')"
            " GROUP BY o.product_id",
            base_params,
        ).fetchall()
        last_sale_t = {row["product_id"]: int(row["max_t"]) for row in sale_day_rows if row["max_t"] is not None}

        # Open orders per product.
        open_rows = env.conn.execute(
            f"SELECT product_id, COUNT(*) AS n FROM orders"
            f" WHERE run_id=? AND agent_id=? AND product_id IN ({qmarks})"
            f" AND current_status IN ({','.join('?' for _ in _OPEN_ORDER_STATUSES)})"
            " GROUP BY product_id",
            (*base_params, *_OPEN_ORDER_STATUSES),
        ).fetchall()
        open_orders = {r["product_id"]: int(r["n"]) for r in open_rows}

        # Assemble rows.
        lr_cfg = env.scenario.get("listing_rating")
        initial_rating = float(lr_cfg.get("initial_rating", 4.0)) if lr_cfg else 4.0
        prior_weight = float(lr_cfg["prior_weight"]) if lr_cfg else 10.0
        out = []
        for l in listings:
            latest_sale_t = last_sale_t.get(l.product_id)
            listing_age_days = max(
                0,
                ((int(env.t) - int(l.listed_at)) * sh) // 24,
            )
            no_sale_anchor_t = latest_sale_t if latest_sale_t is not None else int(l.listed_at)
            days_without_sales = max(
                0,
                ((int(env.t) - no_sale_anchor_t) * sh) // 24,
            )
            p = env.products.get(l.product_id)
            lr = lr_mod.compute_listing_rating(
                initial_rating,
                l.rating_sum,
                l.rating_count,
                prior_weight,
            )
            out.append(
                {
                    "product_id": l.product_id,
                    "name": p.name if p else "",
                    "listing_age_days": listing_age_days,
                    "days_without_sales": days_without_sales,
                    "procured_orders": procured_orders.get(l.product_id, 0),
                    "fine": fine_win.get(l.product_id, 0.0),
                    "open_orders": open_orders.get(l.product_id, 0),
                    "listing_rating": round(lr, 2),
                }
            )

        # Sort descending for age/no-sale/fine, ascending for procured orders.
        reverse = sort_by != "procured_orders"
        out.sort(key=lambda r: (-r[sort_by] if reverse else r[sort_by], str(r["product_id"])))
        return {
            "sort_by": sort_by,
            "window_days": days,
            **compact_table(out, _REVIEW_LISTING_COLUMNS),
        }


def query_balance(env: Environment, agent_id: str) -> dict:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        return _cash_to_agent_dict(env.agents[agent_id].cash)


def get_store_snapshot(env: Environment, agent_id: str) -> dict:
    from tools.observation import build_store_snapshot, current_or_cached_change_window

    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        window = current_or_cached_change_window(env, agent_id)
        return build_store_snapshot(env, agent_id, window=window)


def query_platform_rules(env: Environment) -> dict:
    from tools.observation import compose_system_brief

    return {"system_prompt": compose_system_brief(env)["system_prompt"]}


# ---------- 订单 ----------


def query_open_orders(
    env: Environment, agent_id: str, statuses: Optional[list[str]] = None, page: int = 1, page_size: int = 20
) -> dict:
    err, selected = _coerce_statuses(
        statuses,
        allowed=_OPEN_ORDER_STATUSES,
        default=_OPEN_ORDER_STATUSES,
    )
    if err is not None:
        return err
    err, page_i, page_size_i = _page_args(
        page,
        page_size,
        maximum=50,
    )
    if err is not None:
        return err
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        qmarks = ",".join("?" for _ in selected)
        params: list[Any] = [env.run_id, agent_id, *selected]
        count_rows = env.conn.execute(
            "SELECT current_status, COUNT(*) AS n FROM orders"
            f" WHERE run_id=? AND agent_id=? AND current_status IN ({qmarks})"
            " GROUP BY current_status",
            tuple(params),
        ).fetchall()
        counts_raw = {row["current_status"]: int(row["n"] or 0) for row in count_rows}
        by_status = {status: counts_raw.get(status, 0) for status in _OPEN_ORDER_STATUSES}
        total = sum(counts_raw.values())
        offset = (page_i - 1) * page_size_i
        rows = env.conn.execute(
            "SELECT * FROM orders"
            f" WHERE run_id=? AND agent_id=? AND current_status IN ({qmarks})"
            " ORDER BY CASE current_status"
            " WHEN 'late' THEN 0"
            " WHEN 'delivered' THEN 1"
            " WHEN 'shipped' THEN 2"
            " WHEN 'ordered' THEN 3"
            " ELSE 4 END,"
            " order_t DESC, order_id ASC"
            " LIMIT ? OFFSET ?",
            tuple([*params, page_size_i + 1, offset]),
        ).fetchall()
        page_rows = rows[:page_size_i]
        order_ids = [row["order_id"] for row in page_rows]
        logs = _status_logs_for_orders(env, order_ids)
        has_next = len(rows) > page_size_i
        return {
            "tick": t_to_agent_time(env, env.t),
            "total_count": total,
            "by_status": by_status,
            "page": page_i,
            "page_size": page_size_i,
            "has_next": has_next,
            "orders": compact_table(
                [_compact_order_row(env, row, logs.get(row["order_id"], [])) for row in page_rows],
                _with_order_fee_columns(env, _ORDER_SUMMARY_COLUMNS),
            ),
        }


def query_order_updates(
    env: Environment,
    agent_id: str,
    statuses: Optional[list[str]] = None,
    include_ordered: bool = True,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    from tools.observation import current_or_cached_change_window

    err, selected = _coerce_statuses(
        statuses,
        allowed=_ORDER_STATUS_ORDER,
        default=_ORDER_STATUS_ORDER,
    )
    if err is not None:
        return err
    err, page_i, page_size_i = _page_args(
        page,
        page_size,
        maximum=100,
    )
    if err is not None:
        return err
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        window = current_or_cached_change_window(env, agent_id)
        if window is None:
            return {
                "window": None,
                "total_count": 0,
                "by_current": {status: 0 for status in _ORDER_STATUS_ORDER},
                "page": page_i,
                "page_size": page_size_i,
                "has_next": False,
                "orders": compact_table(
                    [],
                    _with_order_fee_columns(env, _ORDER_UPDATE_COLUMNS),
                ),
            }
        t_from, t_to = window
        parts = [
            "os.run_id=?",
            "o.agent_id=?",
            "os.t>=?",
            "os.t<=?",
        ]
        params: list[Any] = [env.run_id, agent_id, int(t_from), int(t_to)]
        if not bool(include_ordered):
            parts.append("os.status!='ordered'")
        transition_rows = env.conn.execute(
            "SELECT os.order_id, os.t, os.status FROM order_status os"
            " JOIN orders o ON o.run_id=os.run_id AND o.order_id=os.order_id"
            f" WHERE {' AND '.join(parts)}"
            f" ORDER BY os.t ASC, {dbm._status_rank_sql('os.status')} ASC",
            tuple(params),
        ).fetchall()
        if not transition_rows:
            return {
                "window": {
                    "from": _compact_agent_time(env, t_from),
                    "to": _compact_agent_time(env, t_to),
                },
                "total_count": 0,
                "by_current": {status: 0 for status in _ORDER_STATUS_ORDER},
                "page": page_i,
                "page_size": page_size_i,
                "has_next": False,
                "orders": compact_table(
                    [],
                    _with_order_fee_columns(env, _ORDER_UPDATE_COLUMNS),
                ),
            }

        transitions_by_id: dict[str, list[dict]] = {}
        latest_transition_t: dict[str, int] = {}
        for row in transition_rows:
            oid = row["order_id"]
            transition = {"t": int(row["t"]), "status": row["status"]}
            transitions_by_id.setdefault(oid, []).append(transition)
            latest_transition_t[oid] = int(row["t"])

        order_ids = list(transitions_by_id)
        qmarks = ",".join("?" for _ in order_ids)
        order_rows = env.conn.execute(
            f"SELECT * FROM orders WHERE run_id=? AND agent_id=? AND order_id IN ({qmarks})",
            (env.run_id, agent_id, *order_ids),
        ).fetchall()
        selected_set = set(selected)
        orders_by_id = {row["order_id"]: row for row in order_rows if row["current_status"] in selected_set}
        filtered_ids = sorted(
            orders_by_id,
            key=lambda oid: (-latest_transition_t.get(oid, -1), oid),
        )
        by_current: dict[str, int] = {status: 0 for status in _ORDER_STATUS_ORDER}
        for oid in filtered_ids:
            status = orders_by_id[oid]["current_status"]
            by_current[status] = by_current.get(status, 0) + 1

        offset = (page_i - 1) * page_size_i
        page_ids = filtered_ids[offset : offset + page_size_i]
        logs = _status_logs_for_orders(env, page_ids)
        out_rows = []
        for oid in page_ids:
            order_row = orders_by_id[oid]
            compact = _compact_order_row(env, order_row, logs.get(oid, []))
            transitions = transitions_by_id[oid]
            first = transitions[0]
            previous = _previous_status(logs.get(oid, []), first["t"], first["status"])
            update_columns = _with_order_fee_columns(env, _ORDER_UPDATE_COLUMNS)
            compact = {
                **compact,
                "previous_status": previous,
            }
            out_rows.append({key: compact[key] for key in update_columns})

        total = len(filtered_ids)
        has_next = offset + page_size_i < total
        return {
            "window": {
                "from": _compact_agent_time(env, t_from),
                "to": _compact_agent_time(env, t_to),
            },
            "total_count": total,
            "by_current": by_current,
            "page": page_i,
            "page_size": page_size_i,
            "has_next": has_next,
            "orders": compact_table(
                out_rows,
                _with_order_fee_columns(env, _ORDER_UPDATE_COLUMNS),
            ),
        }


def query_my_orders(
    env: Environment,
    agent_id: str,
    status: Optional[str] = None,
    product_id: Optional[str] = None,
    supplier_id: Optional[str] = None,
    day_from: Optional[int] = None,
    day_to: Optional[int] = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        err, page_i, page_size_i = _page_args(page, page_size)
        if err is not None:
            return err
        sh = _step_hours(env)
        range_err, day_from_i, day_to_i = _coerce_optional_day_range(
            env,
            day_from,
            day_to,
        )
        if range_err is not None:
            return range_err
        t_from, t_to = day_range_to_t(day_from_i, day_to_i, sh)
        q = "SELECT * FROM orders WHERE run_id = ? AND agent_id = ?"
        params: list[Any] = [env.run_id, agent_id]
        if status:
            if status not in _ORDER_STATUS_SET:
                allowed = ", ".join(sorted(_ORDER_STATUS_SET))
                return {"ok": False, "error": f"invalid status; must be one of: {allowed}"}
            q += " AND current_status = ?"
            params.append(status)
        if product_id:
            q += " AND product_id = ?"
            params.append(product_id)
        if supplier_id:
            q += " AND supplier_id = ?"
            params.append(supplier_id)
        if t_from is not None:
            q += " AND order_t >= ?"
            params.append(t_from)
        if t_to is not None:
            q += " AND order_t <= ?"
            params.append(t_to)
        count_row = env.conn.execute(
            f"SELECT COUNT(*) AS n FROM ({q})",
            tuple(params),
        ).fetchone()
        total_count = int(count_row["n"] or 0) if count_row else 0
        offset = (page_i - 1) * page_size_i
        q += " ORDER BY order_t DESC, order_id DESC LIMIT ? OFFSET ?"
        params.extend([page_size_i + 1, offset])
        rows = env.conn.execute(q, tuple(params)).fetchall()
        out = []
        for r in rows[:page_size_i]:
            p = env.products.get(r["product_id"])
            net_profit = _order_net_profit_money(r)
            profit_finalized = r["settled_t"] is not None
            out.append(
                {
                    "order_id": r["order_id"],
                    "product_id": r["product_id"],
                    "product_name": p.name if p else "",
                    "supplier_id": r["supplier_id"],
                    "supplier_name": p.supplier_name if p else "",
                    "order_time": t_to_agent_time(env, r["order_t"]),
                    "sale_price": r["sale_price"],
                    "purchase_price": r["purchase_price"],
                    "current_status": r["current_status"],
                    "supplier_ship_hours": (r["supplier_ship_hours"] or (p.supplier_ship_hours if p else None)),
                    "supplier_logistics_hours": p.logistics_hours if p else None,
                    "actual_logistics_hours": (
                        r["actual_logistics_hours"]
                        if r["delivered_t"] is not None and r["actual_logistics_hours"] > 0
                        else None
                    ),
                    "realized_revenue": r["realized_revenue"],
                    "realized_cost": r["realized_cost"],
                    "total_penalty": r["total_penalty"],
                    **_order_fee_fields(env, r),
                    "net_profit": net_profit,
                    "profit_finalized": profit_finalized,
                }
            )
        orders = compact_table(out, _with_order_fee_columns(env, _MY_ORDER_COLUMNS))
        return {
            "page": page_i,
            "page_size": page_size_i,
            "total_count": total_count,
            "has_next": len(rows) > page_size_i,
            "orders": orders,
        }


def _serialize_order_for_agent(env: Environment, o: Order) -> dict:
    """Agent-facing order view. Mirrors Order.visible() but converts every
    raw-tick field (order_t / late_t / status_log[].t) into agent-facing time."""
    product = env.products.get(o.product_id)
    status_t = max((int(row.t) for row in o.status_log), default=int(o.order_t))
    net_profit = _round_money(o.net_profit)
    profit_finalized = o.settled_t is not None
    return {
        "order_id": o.order_id,
        "product_id": o.product_id,
        "product_name": product.name if product else "",
        "supplier_id": o.supplier_id,
        "supplier_name": product.supplier_name if product else "",
        "order_time": t_to_agent_time(env, o.order_t),
        "status_age_hours": max(
            0,
            (int(env.t) - status_t) * _step_hours(env),
        ),
        "expected_delivery_time": t_to_agent_time_optional(
            env,
            _expected_delivery_t_for_order(env, o),
        ),
        "delivered_time": t_to_agent_time_optional(env, o.delivered_t),
        "sale_price": o.sale_price,
        "purchase_price": o.purchase_price,
        "current_status": o.current_status,
        "supplier_ship_hours": (o.supplier_ship_hours or (product.supplier_ship_hours if product else None)),
        "supplier_logistics_hours": product.logistics_hours if product else None,
        "actual_logistics_hours": (
            o.actual_logistics_hours if o.delivered_t is not None and o.actual_logistics_hours > 0 else None
        ),
        "late_time": t_to_agent_time_optional(env, o.late_t),
        "realized_revenue": o.realized_revenue,
        "realized_cost": o.realized_cost,
        "total_penalty": o.total_penalty,
        **_order_fee_fields(env, o),
        "net_profit": net_profit,
        "profit_finalized": profit_finalized,
    }


def query_order_detail(env: Environment, agent_id: str, order_id: str) -> Optional[dict]:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        o = dbm.load_order(env.conn, env.run_id, order_id)
        if not o or o.agent_id != agent_id:
            return None
        view = _serialize_order_for_agent(env, o)
        view["status_log"] = [{"time": t_to_agent_time(env, s.t), "status": s.status} for s in o.status_log]
        return view


def _coerce_day_range(env: Environment, day_from: Any, day_to: Any) -> tuple[Optional[dict], int, int, int, int]:
    try:
        df = int(day_from)
        dt = int(day_to)
    except (TypeError, ValueError):
        return {"ok": False, "error": "day_from and day_to must be integers"}, 0, 0, 0, 0
    if df < 1 or dt < 1:
        return {"ok": False, "error": "day_from and day_to must be >= 1"}, 0, 0, 0, 0
    if df > dt:
        return {"ok": False, "error": "day_from must be <= day_to"}, 0, 0, 0, 0
    current_day = _current_virtual_day(env)
    if dt > current_day:
        return (
            {
                "ok": False,
                "error": f"day_to cannot exceed current day ({current_day})",
                "current_day": current_day,
            },
            0,
            0,
            0,
            0,
        )
    sh = _step_hours(env)
    t_from, t_to = day_range_to_t(df, dt, sh)
    return None, df, dt, int(t_from or 0), int(t_to or 0)


def _bucket_days(day_from: int, day_to: int, level: str) -> list[tuple[str, int, int]]:
    if level == "day":
        return [(f"D{day}", day, day) for day in range(day_from, day_to + 1)]
    out = []
    idx = 1
    day = day_from
    while day <= day_to:
        end = min(day_to, day + 6)
        out.append((f"W{idx}", day, end))
        idx += 1
        day = end + 1
    return out


def _last_value_at(series: list[tuple[int, float]], t_to: int) -> float:
    value = 0.0
    for t_step, point in series:
        if int(t_step) > int(t_to):
            break
        value = float(point)
    return value


def query_store_performance(env: Environment, agent_id: str, day_from: int, day_to: int, level: str = "day") -> dict:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        err, df, dt, _t_from, t_to = _coerce_day_range(env, day_from, day_to)
        if err is not None:
            return err
        level = str(level or "day")
        if level not in ("day", "week"):
            return {"ok": False, "error": "level must be 'day' or 'week'"}
        sh = _step_hours(env)
        steps_per_day = max(1, 24 // sh)
        buckets = _bucket_days(df, dt, level)
        metric_keys = [
            "cum_gmv",
            "cum_cost",
            "cum_gross_profit",
            "cum_net_profit",
            "cum_fine",
            "cum_fee",
            "net_assets",
        ]
        series = dbm.load_metrics_bulk(env.conn, env.run_id, agent_id, metric_keys, None, t_to)
        out = {
            "day_from": df,
            "day_to": dt,
            "level": level,
            "bucket_label": [],
            "bucket_day_from": [],
            "bucket_day_to": [],
            "cum_orders": [],
            "cum_gmv": [],
            "cum_cost": [],
            "cum_gross_profit": [],
            "cum_net_profit": [],
            "cum_fine": [],
            "cum_fee": [],
            "net_assets": [],
        }
        for label, start_day, end_day in buckets:
            bucket_t_to = end_day * steps_per_day - 1
            order_row = env.conn.execute(
                "SELECT COUNT(*) AS n FROM orders WHERE run_id=? AND agent_id=? AND order_t<=?",
                (env.run_id, agent_id, int(bucket_t_to)),
            ).fetchone()
            out["bucket_label"].append(label)
            out["bucket_day_from"].append(start_day)
            out["bucket_day_to"].append(end_day)
            out["cum_orders"].append(int(order_row["n"] or 0) if order_row else 0)
            for key in metric_keys:
                out[key].append(round(_last_value_at(series.get(key, []), bucket_t_to), 2))
        return out


def query_product_sales_stats(
    env: Environment, agent_id: str, day_from: int, day_to: int, sort_by: str = "net_profit", limit: int = 10
) -> dict:
    with env.lock:
        err, df, dt, t_from, t_to = _coerce_day_range(env, day_from, day_to)
        if err is not None:
            return err
        sort_by = str(sort_by or "net_profit")
        allowed = {"orders", "gmv", "gross_profit", "net_profit", "fine"}
        if sort_by not in allowed:
            return {"ok": False, "error": "sort_by must be one of orders, gmv, gross_profit, net_profit, fine"}
        try:
            lim = int(limit)
        except (TypeError, ValueError):
            return {"ok": False, "error": "limit must be an integer"}
        lim = max(1, min(lim, 100))
        rows = env.conn.execute(
            "SELECT o.product_id, o.supplier_id, p.name AS name,"
            " p.category AS category,"
            " p.price AS current_supplier_price,"
            " SUM(CASE WHEN o.order_t>=? AND o.order_t<=? THEN 1 ELSE 0 END) AS orders,"
            " SUM(CASE WHEN o.order_t>=? AND o.order_t<=?"
            "   AND o.current_status NOT IN ('stockout','insufficient_balance')"
            "   THEN COALESCE(o.sale_price, 0.0) ELSE 0 END) AS gmv,"
            " SUM(CASE WHEN o.order_t>=? AND o.order_t<=?"
            "   AND o.current_status NOT IN ('stockout','insufficient_balance')"
            "   THEN COALESCE(o.sale_price, 0.0) - COALESCE(o.purchase_price, 0.0)"
            "   ELSE 0 END) AS gross_profit,"
            " SUM(CASE WHEN o.settled_t IS NOT NULL AND o.settled_t>=? AND o.settled_t<=?"
            f"   THEN {dbm.order_net_profit_sql('o')}"
            "   ELSE 0 END) AS net_profit,"
            " SUM(CASE WHEN o.late_t IS NOT NULL AND o.late_t>=? AND o.late_t<=? THEN 1 ELSE 0 END) AS late_count,"
            " SUM(CASE WHEN o.order_t>=? AND o.order_t<=? AND o.current_status='stockout' THEN 1 ELSE 0 END) AS stockout_count,"
            " SUM(CASE WHEN o.order_t>=? AND o.order_t<=? AND o.current_status='insufficient_balance' THEN 1 ELSE 0 END) AS insufficient_balance_count,"
            " SUM(CASE WHEN o.settled_t IS NOT NULL AND o.settled_t>=? AND o.settled_t<=?"
            "   AND o.current_status IN ('settled_refund','settled_only_refund') THEN 1 ELSE 0 END) AS refund_count,"
            " SUM(CASE WHEN o.settled_t IS NOT NULL AND o.settled_t>=? AND o.settled_t<=?"
            "   AND o.current_status='settled_bad_review' THEN 1 ELSE 0 END) AS bad_review_count"
            " FROM orders o"
            " LEFT JOIN products p ON p.run_id=o.run_id AND p.product_id=o.product_id"
            " WHERE o.run_id=? AND o.agent_id=?"
            " AND ((o.order_t>=? AND o.order_t<=?)"
            "   OR (o.settled_t IS NOT NULL AND o.settled_t>=? AND o.settled_t<=?)"
            "   OR (o.late_t IS NOT NULL AND o.late_t>=? AND o.late_t<=?))"
            " GROUP BY o.product_id, o.supplier_id, p.name, p.category, p.price",
            (
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                env.run_id,
                agent_id,
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
                int(t_from),
                int(t_to),
            ),
        ).fetchall()
        listing_by_pid = {listing.product_id: listing for listing in dbm.list_listings(env.conn, env.run_id, agent_id)}
        fine_by_product = _product_fines_by_event_time(
            env,
            agent_id,
            t_from,
            t_to,
        )
        items = []
        for row in rows:
            n = int(row["orders"] or 0)
            listing = listing_by_pid.get(row["product_id"])
            current_sale_price = float(listing.sale_price) if listing is not None else None
            current_supplier_price = (
                float(row["current_supplier_price"]) if row["current_supplier_price"] is not None else None
            )
            current_gross_margin_rate = None
            if current_sale_price is not None and current_supplier_price is not None and current_sale_price > 0:
                current_gross_margin_rate = round(
                    (current_sale_price - current_supplier_price) / current_sale_price,
                    4,
                )
            items.append(
                {
                    "product_id": row["product_id"],
                    "name": row["name"] or "",
                    "category": row["category"] or "",
                    "supplier_id": row["supplier_id"] or "",
                    "current_sale_price": current_sale_price,
                    "current_supplier_price": current_supplier_price,
                    "current_gross_margin_rate": current_gross_margin_rate,
                    "orders": n,
                    "gmv": round(float(row["gmv"] or 0.0), 2),
                    "gross_profit": round(float(row["gross_profit"] or 0.0), 2),
                    "net_profit": round(float(row["net_profit"] or 0.0), 2),
                    "fine": fine_by_product.get(str(row["product_id"]), 0.0),
                    "late_count": int(row["late_count"] or 0),
                    "stockout_count": int(row["stockout_count"] or 0),
                    "insufficient_balance_count": int(row["insufficient_balance_count"] or 0),
                    "refund_count": int(row["refund_count"] or 0),
                    "bad_review_count": int(row["bad_review_count"] or 0),
                }
            )
        items.sort(key=lambda item: (-float(item[sort_by]), str(item["product_id"])))
        return {
            "day_from": df,
            "day_to": dt,
            "sort_by": sort_by,
            "limit": lim,
            "items": compact_table(items[:lim], _PRODUCT_SALES_COLUMNS),
        }


def query_cash_pipeline(env: Environment, agent_id: str, window_days: int = 7) -> dict:
    with env.lock:
        guard = _alive_guard(env, agent_id)
        if guard:
            return guard
        try:
            days = int(window_days)
        except (TypeError, ValueError):
            return {"ok": False, "error": "window_days must be one of: 1, 3, 7, 14"}
        if days not in (1, 3, 7, 14):
            return {"ok": False, "error": "window_days must be one of: 1, 3, 7, 14"}

        cash = env.agents[agent_id].cash
        cash_now = {
            "balance": _round_money(cash.balance),
            "deposit_pool": _round_money(cash.deposit_pool),
            "in_transit": _round_money(cash.in_transit),
            "receivable": _round_money(cash.receivable),
            "net_assets": _round_money(cash.balance + cash.deposit_pool + cash.in_transit + cash.receivable),
        }

        sh = _step_hours(env)
        steps_per_day = max(1, 24 // sh)
        settlement_cfg = env.scenario.get("settlement") or {}
        normal_delay_hours = int(settlement_cfg.get("normal_delay_hours", 168))
        age_cutoff_t = int(env.t) - days * steps_per_day

        # Only report facts that are already observable: which orders are
        # currently delivered/unsettled and how long their receivables have
        # been outstanding.  Do not use settlement_delay_steps or
        # preset_anomaly here: both are pre-sampled future outcomes hidden from
        # the agent, and refund paths do not necessarily credit sale_price.
        receivable = env.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(sale_price), 0.0) AS amount,"
            " SUM(CASE WHEN delivered_t>=? THEN 1 ELSE 0 END) AS recent_n,"
            " COALESCE(SUM(CASE WHEN delivered_t>=? THEN sale_price"
            " ELSE 0 END), 0.0) AS recent_amount,"
            " SUM(CASE WHEN delivered_t<? THEN 1 ELSE 0 END) AS older_n,"
            " COALESCE(SUM(CASE WHEN delivered_t<? THEN sale_price"
            " ELSE 0 END), 0.0) AS older_amount"
            " FROM orders"
            " WHERE run_id=? AND agent_id=?"
            " AND current_status='delivered'"
            " AND delivered_t IS NOT NULL"
            " AND settled_t IS NULL",
            (
                age_cutoff_t,
                age_cutoff_t,
                age_cutoff_t,
                age_cutoff_t,
                env.run_id,
                agent_id,
            ),
        ).fetchone()

        qmarks = ",".join("?" for _ in _OPEN_ORDER_STATUSES)
        open_rows = env.conn.execute(
            "SELECT current_status, COUNT(*) AS n,"
            " COALESCE(SUM(purchase_price), 0.0) AS purchase_cost"
            " FROM orders"
            f" WHERE run_id=? AND agent_id=? AND current_status IN ({qmarks})"
            " GROUP BY current_status",
            (env.run_id, agent_id, *_OPEN_ORDER_STATUSES),
        ).fetchall()
        by_status = {status: 0 for status in _OPEN_ORDER_STATUSES}
        count = 0
        purchase_cost = 0.0
        for row in open_rows:
            status = row["current_status"]
            n = int(row["n"] or 0)
            by_status[status] = n
            count += n
            purchase_cost += float(row["purchase_cost"] or 0.0)

        return {
            "window_days": days,
            "cash_now": cash_now,
            "receivable_aging": {
                "total": {
                    "amount": _round_money(receivable["amount"] if receivable else 0.0),
                    "order_count": int(receivable["n"] or 0) if receivable else 0,
                },
                "delivered_within_window": {
                    "amount": _round_money(receivable["recent_amount"] if receivable else 0.0),
                    "order_count": (int(receivable["recent_n"] or 0) if receivable else 0),
                },
                "delivered_before_window": {
                    "amount": _round_money(receivable["older_amount"] if receivable else 0.0),
                    "order_count": (int(receivable["older_n"] or 0) if receivable else 0),
                },
            },
            "settlement_policy": {
                "max_resolution_hours_after_delivery": normal_delay_hours,
                "exact_timing_known": False,
                "full_sale_proceeds_guaranteed": False,
            },
            "open_orders": {
                "count": count,
                "purchase_cost": _round_money(purchase_cost),
                "by_status": by_status,
            },
        }


def query_supply_chain_anomalies(env: Environment, agent_id: str, mode: str = "new") -> dict:
    from tools.observation import supply_chain_anomalies

    with env.lock:
        return supply_chain_anomalies(env, agent_id, mode)


# ---------- 记忆文档 ----------


def _memory_doc_path(env: Environment, agent_id: str) -> str:
    safe_agent_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_id or "agent")
    safe_agent_id = safe_agent_id.strip("._") or "agent"
    return os.path.join(env.runs_root, env.run_id, "agent", "memory", f"{safe_agent_id}.md")


def _memory_history_path(env: Environment, agent_id: str) -> str:
    safe_agent_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_id or "agent")
    safe_agent_id = safe_agent_id.strip("._") or "agent"
    return os.path.join(env.runs_root, env.run_id, "agent", "memory", f"{safe_agent_id}.history.md")


def _append_memory_history(env: Environment, agent_id: str, content: str) -> None:
    path = _memory_history_path(env, agent_id)
    version = 1
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = f.read()
            version += history.count(MEMORY_VERSION_MARKER)
            version += sum(history.count(marker) for marker in LEGACY_MEMORY_VERSION_MARKERS)
    wall_ms = int(time.time() * 1000)
    size = len(content.encode("utf-8"))
    with open(path, "a", encoding="utf-8") as f:
        if os.path.getsize(path) > 0:
            f.write("\n\n")
        f.write(f"{MEMORY_VERSION_MARKER}{version} -->\n")
        f.write(f"## Memory version {version}\n")
        f.write(f"- step: {env.t}\n")
        f.write(f"- wall_ms: {wall_ms}\n")
        f.write(f"- bytes: {size}\n\n")
        f.write(content)
        if content and not content.endswith("\n"):
            f.write("\n")


def read_memory_doc(env: Environment, agent_id: str) -> dict:
    """Read this agent's Markdown scratchpad for the current run."""
    path = _memory_doc_path(env, agent_id)
    if not os.path.exists(path):
        return {"ok": True, "content": "", "bytes": 0}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"ok": True, "content": content, "bytes": len(content.encode("utf-8"))}


_MEMORY_DOC_MAX_BYTES = 256 * 1024  # 256 KiB


def write_memory_doc(env: Environment, agent_id: str, content: str) -> dict:
    """Overwrite this agent's Markdown scratchpad for the current run."""
    if not isinstance(content, str):
        content = str(content)
    size = len(content.encode("utf-8"))
    if size > _MEMORY_DOC_MAX_BYTES:
        return {"ok": False, "error": f"content too large ({size} bytes, max {_MEMORY_DOC_MAX_BYTES})"}
    path = _memory_doc_path(env, agent_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    _append_memory_history(env, agent_id, content)
    return {"ok": True, "bytes": size}


# ---------- 结束 ----------


def end_of_step_result(env: Environment) -> dict:
    return {"ok": True, "time": t_to_agent_time(env, env.t)}


def end_of_step(env: Environment) -> dict:
    env.hook_event.set()
    return end_of_step_result(env)
