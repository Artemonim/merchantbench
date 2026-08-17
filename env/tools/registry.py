"""Single source of truth for the agent-facing tool surface.

Each ToolSpec carries the OpenAI tool-calling schema, examples, and a pointer
to the actual handler function in tools.tools. This registry is consumed by:

  * GET /runs/<rid>/tools/schema  (returned as OpenAI tool-calling list)
  * tools/observation.list_tools  (OpenAI tool-calling schema list for
    hook-gated self-discovery through /act)
  * tests/test_registry.py asserting handler signatures match parameter schema

The registry is metadata only — it does NOT generate per-tool HTTP routes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_args

from core.entities import OrderStatus
from tools import tools as t


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict                   # JSON Schema, OpenAI-compatible
    examples: list[dict] = field(default_factory=list)
    method: str = "POST"               # "GET" | "POST"
    path_suffix: str = ""              # e.g. "adjust_price" (under /tools/)
    handler: Optional[Callable[..., Any]] = None
    mutating: bool = False             # mutating tools support idempotency_key

    def openai_schema(self) -> dict:
        """OpenAI tool-calling format: {type: "function", function: {...}}."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------- helpers for schema construction ----------

def _obj(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_NO_PARAMS = _obj({}, [])

_PRODUCT_ID = {"type": "string", "description": "Product ID, e.g. 'p_017'."}
_SALE_PRICE = {
    "type": "number",
    "description": "Listing price in store currency, must be > 0.",
}
_ORDER_STATUS_ENUM = list(get_args(OrderStatus))
_BATCH_MAX_ITEMS = 100


def _batch_array(item_properties: dict, required: list[str]) -> dict:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": _BATCH_MAX_ITEMS,
        "items": _obj(item_properties, required),
    }


# ---------- registry ----------

REGISTRY: list[ToolSpec] = [
    # ---- 选品 (catalog reads — env-only, no per-agent state mutation) ----
    ToolSpec(
        name="market_brief",
        description="Return public first-level category market summary for the last 7 or 30 "
                    "fully completed virtual days, "
                    "scaled by the scenario's small_share. "
                    "total_sales / total_gmv are daily series for the requested window; "
                    "daily_avg_sales / daily_avg_gmv are their scalar averages. "
                    "Historical GMV uses each product's stable reference price, so current "
                    "supplier price changes do not rewrite past market values. "
                    "avg_price is the mean supplier price across products in the category. "
                    "This is category-level only.",
        parameters=_obj({
            "window_days": {"type": "integer", "enum": [7, 30], "default": 7},
        }, []),
        examples=[{"window_days": 7}],
        method="GET", path_suffix="market_brief",
        handler=t.market_brief, mutating=False,
    ),
    ToolSpec(
        name="hot_search_terms",
        description="Return demand proxy hot-search trends through the last fully completed "
                    "virtual day for an optional first-level category. "
                    "The result includes day/date metadata and a compact trend table with rank, "
                    "keyword, category, trend, trailing-window change_pct, and rank_change.",
        parameters=_obj({
            "category": {"type": "string", "description": "Optional first-level category filter."},
            "window_days": {"type": "integer", "enum": [7, 30], "default": 7},
        }, []),
        examples=[{"category": "cleaning", "window_days": 7}],
        method="GET", path_suffix="hot_search_terms",
        handler=t.hot_search_terms, mutating=False,
    ),
    ToolSpec(
        name="get_daily_report",
        description="Return the merchant business opportunity daily report published for the current "
                    "simulation date. The response includes report_date, data_as_of (the previous "
                    "simulation date), and Markdown content with recent market news, category momentum, "
                    "and keyword opportunity signals. ",
        parameters=_NO_PARAMS,
        examples=[{}],
        method="GET", path_suffix="get_daily_report",
        handler=t.get_daily_report, mutating=False,
    ),
    ToolSpec(
        name="search_products",
        description="Paginated keyword search over the public supplier catalog. "
                    "A Chinese keyword query must contain at least two characters; "
                    "a single CJK character returns a validation error. "
                    "Every non-empty query uses one FTS5 match set; sort_by only changes "
                    "the ordering of those matches. Relevance ranks by BM25, while price, "
                    "rating, supplier rating, and logistics modes use BM25 as a tie-breaker. "
                    "Supports optional filters over visible product/supplier fields. Never "
                    "uses hidden demand, margin, risk, or future-state signals. items is "
                    "returned as a compact table: {columns, rows, count}.",
        parameters=_obj({
            "query": {"type": "string", "default": "",
                      "description": "Keyword query over visible product, category, and supplier text. Chinese queries require at least two characters; empty string browses visible catalog products."},
            "price_min": {"type": "number", "description": "Optional minimum supplier price filter."},
            "price_max": {"type": "number", "description": "Optional maximum supplier price filter."},
            "supplier_rating_min": {"type": "number", "description": "Optional minimum supplier shop rating."},
            "historical_rating_min": {"type": "number", "description": "Optional minimum product historical average rating."},
            "logistics_hours_max": {"type": "integer", "description": "Optional maximum logistics hours."},
            "supplier_ship_hours_max": {"type": "integer", "description": "Optional maximum current supplier ship hours."},
            "delivery_hours_max": {"type": "integer", "description": "Optional maximum total delivery hours (ship + logistics combined)."},
            "quantity_min": {"type": "integer", "description": "Optional minimum currently available quantity."},
            "sort_by": {
                "type": "string",
                "enum": [
                    "relevance",
                    "price_asc",
                    "price_desc",
                    "rating",
                    "supplier_rating",
                    "logistics_speed",
                ],
                "default": "relevance",
            },
            "page": {"type": "integer", "minimum": 1, "maximum": t.PAGE_MAX, "default": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        }, []),
        examples=[{"query": "earbud", "page": 1, "page_size": 20}],
        method="GET", path_suffix="search_products",
        handler=t.search_products, mutating=False,
    ),
    ToolSpec(
        name="get_product_detail",
        description="Fetch visible product detail by product_id. Includes trust signals: "
                    "historical_avg_rating (per product, 1-5), shop_rating (per supplier, 1-5, "
                    "shared across all products of same supplier_id), supplier_age_years "
                    "(per supplier, years). Supplier-delisted products remain visible only "
                    "to an agent that currently has the product on its own shelf; "
                    "supplier_available reports whether new procurement is possible.",
        parameters=_obj({"product_id": _PRODUCT_ID}, ["product_id"]),
        examples=[{"product_id": "p_017"}],
        method="GET", path_suffix="get_product_detail",
        handler=t.get_product_detail, mutating=False,
    ),
    ToolSpec(
        name="get_supplier_profile",
        description="Fetch public supplier profile by supplier_id: supplier name, shop rating, "
                    "return buyer rate, supplier age, and currently visible product count.",
        parameters=_obj({"supplier_id": {"type": "string"}}, ["supplier_id"]),
        examples=[{"supplier_id": "sup_0001"}],
        method="GET", path_suffix="get_supplier_profile",
        handler=t.get_supplier_profile, mutating=False,
    ),
    ToolSpec(
        name="list_supplier_products",
        description="Paginated list of currently visible products from one supplier. "
                    "Each row contains the same public fields as get_product_detail. "
                    "items is returned as a compact table: {columns, rows, count}.",
        parameters=_obj({
            "supplier_id": {"type": "string"},
            "page": {"type": "integer", "minimum": 1, "maximum": t.PAGE_MAX, "default": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        }, ["supplier_id"]),
        examples=[{"supplier_id": "sup_0001", "page": 1, "page_size": 20}],
        method="GET", path_suffix="list_supplier_products",
        handler=t.list_supplier_products, mutating=False,
    ),

    # ---- 经营 (mutating, per-agent) ----
    ToolSpec(
        name="list_product",
        description=(
            "Batch put products on this agent's shelf at sale_price, subject to "
            "max_active_listings. Each item requires product_id and sale_price."
        ),
        parameters=_obj({
            "items": _batch_array({
                "product_id": _PRODUCT_ID,
                "sale_price": _SALE_PRICE,
            }, ["product_id", "sale_price"]),
        }, ["items"]),
        examples=[{"items": [{"product_id": "p_017", "sale_price": 41.5}]}],
        method="POST", path_suffix="list_product",
        handler=t.list_product, mutating=True,
    ),
    ToolSpec(
        name="delist_product",
        description="Batch remove products from this agent's shelf.",
        parameters=_obj({
            "items": _batch_array({"product_id": _PRODUCT_ID}, ["product_id"]),
        }, ["items"]),
        examples=[{"items": [{"product_id": "p_017"}]}],
        method="POST", path_suffix="delist_product",
        handler=t.delist_product, mutating=True,
    ),
    ToolSpec(
        name="adjust_price",
        description="Batch change selling prices for already-listed products.",
        parameters=_obj({
            "items": _batch_array({"product_id": _PRODUCT_ID, "new_price": _SALE_PRICE},
                                  ["product_id", "new_price"]),
        }, ["items"]),
        examples=[{"items": [{"product_id": "p_017", "new_price": 43.0}]}],
        method="POST", path_suffix="adjust_price",
        handler=t.adjust_price, mutating=True,
    ),

    # ---- 自查 (per-agent reads) ----
    ToolSpec(
        name="review_my_listings",
        description=(
            "Per-listing health review for this agent's currently-listed products. "
            "Returns product name, full 24-hour listing age, procured-order "
            "velocity, penalty exposure, fulfillment backlog, and "
            "days_without_sales. Relisting starts a fresh no-sale window; "
            "stockout/insufficient-balance orders are not successful sales."
        ),
        parameters=_obj({
            "sort_by": {
                "type": "string",
                "enum": [
                    "listing_age_days",
                    "days_without_sales",
                    "fine",
                    "procured_orders",
                ],
                "default": "listing_age_days",
                "description": "Sort order for the result table.",
            },
            "window_days": {
                "type": "integer",
                "enum": [7, 30],
                "default": 7,
                "description": "Rolling window (days) for procured orders and fines.",
            },
        }, []),
        examples=[{}, {"sort_by": "days_without_sales"}, {"window_days": 30}],
        method="GET", path_suffix="review_my_listings",
        handler=t.review_my_listings, mutating=False,
    ),
    ToolSpec(
        name="query_my_listings",
        description=(
            "List this agent's currently-listed products with sale_price, supplier_price, "
            "procured_orders, cum_gross_profit, cum_net_profit, and cum_fine "
            "accumulators. stockout/insufficient-balance orders are excluded "
            "from procured_orders. Returns a compact table: {columns, rows, count}."
        ),
        parameters=_NO_PARAMS, examples=[{}],
        method="GET", path_suffix="query_my_listings",
        handler=t.query_my_listings, mutating=False,
    ),
    ToolSpec(
        name="query_balance",
        description=(
            "Cash breakdown for this agent. balance = usable cash for procurement "
            "and the first source for every fine; deposit_pool = locked fulfillment "
            "guarantee unavailable for procurement and used only when balance cannot "
            "cover a fine. Cash credits restore deposit_pool to its target before any "
            "remainder enters balance. Exhausting deposit_pool permanently closes the "
            "shop; balance reaching zero alone does not. in_transit = procurement cost "
            "tied up in ordered/shipped goods; receivable = delivered sales awaiting "
            "settlement; cumulative_fine = fines already deducted. Net assets are "
            "balance + deposit_pool + in_transit + receivable."
        ),
        parameters=_NO_PARAMS, examples=[{}],
        method="GET", path_suffix="query_balance",
        handler=t.query_balance, mutating=False,
    ),
    ToolSpec(
        name="get_store_snapshot",
        description=(
            "Compact structured store snapshot matching the observation: order "
            "changes plus current_status totals, supply/listing risks, full cash "
            "fields, and shop rating. Preferred over broad order/listing scans "
            "for routine store state checks."
        ),
        parameters=_NO_PARAMS, examples=[{}],
        method="GET", path_suffix="get_store_snapshot",
        handler=t.get_store_snapshot, mutating=False,
    ),
    ToolSpec(
        name="query_platform_rules",
        description="Platform rules, penalty amounts, initial capital, shop closure rule, and role/goals brief.",
        parameters=_NO_PARAMS, examples=[{}],
        method="GET", path_suffix="query_platform_rules",
        handler=t.query_platform_rules, mutating=False,
    ),
    ToolSpec(
        name="query_my_orders",
        description=(
            "Paginated historical order search for this agent's orders, with filters by "
            "exact current_status, product, supplier, and order day range. "
            "Returns canonical order/product/supplier identity, logistics, and "
            "accounting fields. net_profit is the current realized P&L and may "
            "still change until profit_finalized=true. "
            "Use it for retrospective analysis. orders is returned as a compact "
            "table: {columns, rows, count}."
        ),
        parameters=_obj({
            "status": {
                "type": "string",
                "enum": _ORDER_STATUS_ENUM,
                "description": (
                    "Optional exact current_status filter. Use one of the supported "
                    "order statuses; aggregate names like refund or settled are not "
                    "valid filters."
                ),
            },
            "product_id": {"type": "string"},
            "supplier_id": {"type": "string"},
            "day_from": {"type": "integer",
                          "description": "Optional. 1-indexed virtual day, inclusive."},
            "day_to": {"type": "integer",
                        "description": "Optional. 1-indexed virtual day, inclusive."},
            "page": {"type": "integer", "default": 1, "minimum": 1, "maximum": t.PAGE_MAX},
            "page_size": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
        }, []),
        examples=[{"status": "delivered", "page": 1, "page_size": 20}],
        method="GET", path_suffix="query_my_orders",
        handler=t.query_my_orders, mutating=False,
    ),
    ToolSpec(
        name="query_open_orders",
        description=(
            "Paginated view of this agent's active non-terminal orders: ordered, "
            "late, shipped, and delivered. Use for routine fulfillment, "
            "delivery, receivable, and penalty follow-up. Returns canonical "
            "order fields including product/supplier names, expected/actual "
            "delivery timing, current "
            "net profit, and whether profit is finalized."
        ),
        parameters=_obj({
            "statuses": {
                "type": "array",
                "items": {"type": "string", "enum": ["ordered", "late", "shipped", "delivered"]},
                "description": "Optional current_status filter. Defaults to all active non-terminal statuses.",
            },
            "page": {"type": "integer", "minimum": 1, "maximum": t.PAGE_MAX, "default": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        }, []),
        examples=[{"statuses": ["late", "delivered"], "page": 1, "page_size": 20}],
        method="GET", path_suffix="query_open_orders",
        handler=t.query_open_orders, mutating=False,
    ),
    ToolSpec(
        name="query_order_updates",
        description=(
            "Paginated list of this agent's orders whose status changed since "
            "the last observation window. Coalesces multiple transitions per "
            "order into previous_status/current_status and returns canonical "
            "order fields including product/supplier names, expected/actual "
            "delivery timing, current "
            "net profit, and whether profit is finalized."
        ),
        parameters=_obj({
            "statuses": {
                "type": "array",
                "items": {"type": "string", "enum": _ORDER_STATUS_ENUM},
                "description": "Optional current_status filter after transition coalescing.",
            },
            "include_ordered": {
                "type": "boolean",
                "default": True,
                "description": "Whether newly-created ordered rows are included.",
            },
            "page": {"type": "integer", "minimum": 1, "maximum": t.PAGE_MAX, "default": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
        }, []),
        examples=[{"statuses": ["late", "shipped"], "include_ordered": True,
                   "page": 1, "page_size": 50}],
        method="GET", path_suffix="query_order_updates",
        handler=t.query_order_updates, mutating=False,
    ),
    ToolSpec(
        name="query_order_detail",
        description=(
            "Full detail for one order, including status_log timeline, "
            "product/supplier names, expected/actual delivery timing, logistics "
            "fields, realized accounting, "
            "penalties, net_profit, and profit_finalized."
        ),
        parameters=_obj({"order_id": {"type": "string"}}, ["order_id"]),
        examples=[{"order_id": "o_00042"}],
        method="GET", path_suffix="query_order_detail",
        handler=t.query_order_detail, mutating=False,
    ),
    ToolSpec(
        name="query_supply_chain_anomalies",
        description=(
            "Compact supplier/listing anomaly drilldown. mode='new' returns "
            "supplier anomaly events since the last observation plus affected "
            "listing rows; mode='now' returns currently abnormal listing rows. "
            "Event payloads only expose product_id, name, event_type, before, "
            "and after; recovery timing is hidden. listings is returned as a "
            "compact table: {columns, rows, count}."
        ),
        parameters=_obj({
            "mode": {
                "type": "string",
                "enum": ["new", "now"],
                "default": "new",
            },
        }, []),
        examples=[{"mode": "new"}, {"mode": "now"}],
        method="GET", path_suffix="query_supply_chain_anomalies",
        handler=t.query_supply_chain_anomalies, mutating=False,
    ),
    ToolSpec(
        name="query_store_performance",
        description=(
            "Columnar cumulative store performance over a virtual day range, "
            "bucketed by day or week. Returns compact arrays for orders, GMV, "
            "cost, gross profit, net profit, fines, fees, and net_assets."
        ),
        parameters=_obj({
            "day_from": {"type": "integer", "description": "1-indexed virtual day, inclusive."},
            "day_to": {"type": "integer", "description": "1-indexed virtual day, inclusive."},
            "level": {"type": "string", "enum": ["day", "week"], "default": "day"},
        }, ["day_from", "day_to"]),
        examples=[{"day_from": 3, "day_to": 7, "level": "day"}],
        method="GET", path_suffix="query_store_performance",
        handler=t.query_store_performance, mutating=False,
    ),
    ToolSpec(
        name="query_product_sales_stats",
        description=(
            "Rank products over one virtual day interval by orders, GMV, gross "
            "profit, net profit, or fines. Returns top limit rows with order "
            "count, P&L, anomaly counts, and current product/listing context. "
            "orders includes every order record, while GMV/gross profit exclude "
            "stockout and insufficient-balance failures. fine is attributed to "
            "the penalty event time; net_profit is attributed to settlement time. "
            "items is returned as a compact table: {columns, rows, count}."
        ),
        parameters=_obj({
            "day_from": {"type": "integer", "description": "1-indexed virtual day, inclusive."},
            "day_to": {"type": "integer", "description": "1-indexed virtual day, inclusive."},
            "sort_by": {
                "type": "string",
                "enum": ["orders", "gmv", "gross_profit", "net_profit", "fine"],
                "default": "net_profit",
            },
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
        }, ["day_from", "day_to"]),
        examples=[{"day_from": 3, "day_to": 7, "sort_by": "net_profit", "limit": 10}],
        method="GET", path_suffix="query_product_sales_stats",
        handler=t.query_product_sales_stats, mutating=False,
    ),
    ToolSpec(
        name="query_cash_pipeline",
        description=(
            "Compact read-only cash safety check over known current orders. "
            "Returns current cash, receivable aging inside/outside the selected "
            "lookback window, the public settlement policy, and active "
            "open_orders cost/status totals. It does not expose exact future "
            "settlement timing and does not assume all receivables become full "
            "sale proceeds."
        ),
        parameters=_obj({
            "window_days": {"type": "integer", "enum": [1, 3, 7, 14], "default": 7},
        }, []),
        examples=[{"window_days": 7}],
        method="GET", path_suffix="query_cash_pipeline",
        handler=t.query_cash_pipeline, mutating=False,
    ),
    ToolSpec(
        name="read_memory_doc",
        description="Read this agent's run-local Markdown scratchpad. "
                    "Use it to recover long-term notes, plans, supplier/product "
                    "decisions, and open follow-ups that may have fallen out of "
                    "the context window. Returns empty content if no document exists.",
        parameters=_NO_PARAMS, examples=[{}],
        method="GET", path_suffix="read_memory_doc",
        handler=t.read_memory_doc, mutating=False,
    ),
    ToolSpec(
        name="write_memory_doc",
        description="Overwrite this agent's run-local Markdown scratchpad. "
                    "Keep it concise and structured: current strategy, product "
                    "notes, open tasks, and decisions worth remembering. "
                    "Maximum size: 256 KiB (UTF-8 encoded).",
        parameters=_obj({
            "content": {
                "type": "string",
                "description": "Full Markdown content to store for this agent (max 256 KiB).",
            },
        }, ["content"]),
        examples=[{"content": "# Memory\n\n- Recheck listings tomorrow.\n"}],
        method="POST", path_suffix="write_memory_doc",
        handler=t.write_memory_doc, mutating=True,
    ),

    # ---- 控制 (global hook release) ----
    ToolSpec(
        name="end_of_step",
        description="Release the per-step hook. Call once you're done acting for this tick.",
        parameters=_NO_PARAMS, examples=[{}],
        method="POST", path_suffix="end_of_step",
        handler=t.end_of_step, mutating=False,
    ),

    # ---- 观测聚合 (added by tools/observation.py at import time to avoid cycle) ----
    # See observation.register_observation_tool() — appended below.
]


# ---------- lookup helpers ----------

_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in REGISTRY}


def get(name: str) -> Optional[ToolSpec]:
    return _BY_NAME.get(name)


def all_specs(denylist: Optional[list[str]] = None) -> list[ToolSpec]:
    if denylist is None:
        return list(REGISTRY)
    s = set(denylist)
    return [spec for spec in REGISTRY if spec.name not in s]


_CHINESE_PRODUCT_NAME_NOTE = (
    " NOTE: Product names are in Chinese; use Chinese keywords."
)


def catalog_uses_chinese_product_names(scenario: Any) -> bool:
    """Return True when search_products should hint at Chinese keywords.

    ``data.source == private_real`` historically meant a Chinese-named
    catalog. Olist v6 and other English pools reuse that source key, so the
    hint follows sqlite meta threaded onto the scenario (``dataset_id`` /
    ``source_label``), an explicit ``product_name_language`` flag, or an
    Olist path in the pool field.
    """
    if not isinstance(scenario, dict):
        return False
    data = scenario.get("data")
    if not isinstance(data, dict):
        return False
    if str(data.get("source", "") or "") != "private_real":
        return False
    language = str(data.get("product_name_language") or "").strip().lower()
    if language:
        return language in {"zh", "zh-cn", "zh_cn", "chinese"}
    dataset_id = str(data.get("dataset_id") or "").lower()
    source_label = str(data.get("source_label") or "").lower()
    pool = str(
        data.get("catalog_pool_path")
        or data.get("private_real_db_path")
        or ""
    ).replace("\\", "/").lower()
    # * English Olist pools keep English names despite data.source=private_real.
    if (
        dataset_id.startswith("olist")
        or "olist" in source_label
        or "olist" in pool
    ):
        return False
    return True


def parameters_for_env(spec: ToolSpec, env: Any = None) -> dict:
    return copy.deepcopy(spec.parameters)


def openai_schema_for_env(spec: ToolSpec, env: Any = None) -> dict:
    desc = spec.description
    if env is not None and spec.name == "search_products":
        if catalog_uses_chinese_product_names(getattr(env, "scenario", None)):
            desc += _CHINESE_PRODUCT_NAME_NOTE
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": desc,
            "parameters": parameters_for_env(spec, env),
        },
    }


def openai_schema_dump(denylist: Optional[list[str]] = None, env: Any = None) -> list[dict]:
    if env is None:
        return [spec.openai_schema() for spec in all_specs(denylist)]
    return [openai_schema_for_env(spec, env) for spec in all_specs(denylist)]


def append(spec: ToolSpec) -> None:
    """Used by observation.py to register get_observation after import."""
    if spec.name in _BY_NAME:
        return
    REGISTRY.append(spec)
    _BY_NAME[spec.name] = spec
