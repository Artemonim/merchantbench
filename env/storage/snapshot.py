"""JSON snapshot writer/reader.

Layout: {runs_root}/{run_id}/{env_snapshot,meta.json}

Snapshot contains a list of agents, each with its own listings + cash.
The Product pool is still global. Orders include agent_id and can be filtered
on the frontend per agent.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from core.entities import EventLog, Order, Product
from core.inventory import effective_quantity

if TYPE_CHECKING:
    from core.simulator import AgentState


def run_dir(runs_root: str, run_id: str) -> str:
    return os.path.join(runs_root, run_id)


def ensure_run_layout(runs_root: str, run_id: str) -> str:
    base = run_dir(runs_root, run_id)
    os.makedirs(os.path.join(base, "env_snapshot"), exist_ok=True)
    os.makedirs(os.path.join(base, "env_checkpoint"), exist_ok=True)
    return base


def write_meta(runs_root: str, run_id: str, meta: dict) -> None:
    base = ensure_run_layout(runs_root, run_id)
    with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_meta(runs_root: str, run_id: str) -> dict | None:
    p = os.path.join(run_dir(runs_root, run_id), "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _step_path(base: str, kind: str, t: int) -> str:
    return os.path.join(base, kind, f"t_{t:05d}.json")


def _checkpoint_path(base: str, t: int) -> str:
    return os.path.join(base, "env_checkpoint", f"t_{t:05d}.json.gz")


def product_mutable_state(
    p: Product,
    *,
    quantity: int | None = None,
    current_t: int | None = None,
) -> dict[str, Any]:
    if quantity is None and current_t is not None:
        quantity = effective_quantity(p, current_t)
    return {
        "quantity": int(p.quantity if quantity is None else quantity),
        "quantity_updated_t": int(getattr(p, "quantity_updated_t", 0) or 0),
        "price": p.price,
        "supplier_ship_hours": int(p.supplier_ship_hours),
        "is_listed_by_supplier": bool(p.is_listed_by_supplier),
        "delist_recover_t": p.delist_recover_t,
        "price_recover_t": p.price_recover_t,
        "timeout_active": bool(p.timeout_active),
        "timeout_recover_t": p.timeout_recover_t,
    }


def _agents_blob(agents: list["AgentState"]) -> list[dict]:
    out = []
    for a in agents:
        out.append(
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "cash": a.cash.to_dict(),
                "store_listings": [asdict(l) for l in a.listings.values()],
                "n_good": float(getattr(a, "n_good", 0.0)),
                "n_bad": float(getattr(a, "n_bad", 0.0)),
                "shop_rating_sum": float(getattr(a, "shop_rating_sum", 0.0)),
                "shop_rating_weight": float(getattr(a, "shop_rating_weight", 0.0)),
                "shop_rating_order_count": int(getattr(a, "shop_rating_order_count", 0)),
                "shop_rating_published_t": int(getattr(a, "shop_rating_published_t", 0)),
                "public_review_sum": float(getattr(a, "public_review_sum", 0.0)),
                "public_review_count": int(getattr(a, "public_review_count", 0)),
                "public_review_eligible_sum": float(getattr(a, "public_review_eligible_sum", 0.0)),
                "public_review_eligible_count": int(getattr(a, "public_review_eligible_count", 0)),
                "is_alive": a.is_alive,
                "died_at_t": a.died_at_t,
            }
        )
    return out


def write_env_snapshot(
    runs_root: str,
    run_id: str,
    t: int,
    products: list[Product],
    agents: list["AgentState"],
    orders: list[Order],
    events_this_step: list[EventLog],
    survival_state: dict,
) -> str:
    base = ensure_run_layout(runs_root, run_id)
    agents_blob = []
    for a in agents:
        agents_blob.append(
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "cash": a.cash.to_dict(),
                "store_listings": [asdict(l) for l in a.listings.values()],
                "n_good": float(getattr(a, "n_good", 0.0)),
                "n_bad": float(getattr(a, "n_bad", 0.0)),
                "shop_rating_sum": float(getattr(a, "shop_rating_sum", 0.0)),
                "shop_rating_weight": float(getattr(a, "shop_rating_weight", 0.0)),
                "shop_rating_order_count": int(getattr(a, "shop_rating_order_count", 0)),
                "shop_rating_published_t": int(getattr(a, "shop_rating_published_t", 0)),
                "public_review_sum": float(getattr(a, "public_review_sum", 0.0)),
                "public_review_count": int(getattr(a, "public_review_count", 0)),
                "public_review_eligible_sum": float(getattr(a, "public_review_eligible_sum", 0.0)),
                "public_review_eligible_count": int(getattr(a, "public_review_eligible_count", 0)),
                "is_alive": a.is_alive,
                "died_at_t": a.died_at_t,
            }
        )
    snap_obj: dict[str, Any] = {
        "t": t,
        "products": [asdict(p) for p in products],
        "agents": agents_blob,
        "orders": [{**asdict(o), "status_log": [asdict(s) for s in o.status_log]} for o in orders],
        "events_this_step": [asdict(e) for e in events_this_step],
        "survival_state": survival_state,
    }
    path = _step_path(base, "env_snapshot", t)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap_obj, f, ensure_ascii=False, default=str)
    return path


def write_env_delta_snapshot(
    runs_root: str,
    run_id: str,
    t: int,
    *,
    dirty_products: list[Product] | Any,
    agents: list["AgentState"],
    mutated_orders: list[Order],
    events_this_step: list[EventLog],
    survival_state: dict,
    current_t: int | None = None,
) -> str:
    base = ensure_run_layout(runs_root, run_id)
    snap_obj: dict[str, Any] = {
        "kind": "delta",
        "t": t,
        "products_delta": {p.product_id: product_mutable_state(p, current_t=current_t) for p in dirty_products},
        # Backward-compatible aliases for older dashboard smoke paths. These
        # contain only delta content, not the historical full products/orders.
        "products": {p.product_id: product_mutable_state(p, current_t=current_t) for p in dirty_products},
        "agents": _agents_blob(agents),
        "orders_delta": [{**asdict(o), "status_log": [asdict(s) for s in o.status_log]} for o in mutated_orders],
        "orders": [{**asdict(o), "status_log": [asdict(s) for s in o.status_log]} for o in mutated_orders],
        "events_this_step": [asdict(e) for e in events_this_step],
        "survival_state": survival_state,
    }
    path = _step_path(base, "env_snapshot", t)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap_obj, f, ensure_ascii=False, default=str, separators=(",", ":"))
    return path


def write_env_checkpoint(
    runs_root: str,
    run_id: str,
    t: int,
    products: list[Product],
    current_t: int | None = None,
) -> str:
    base = ensure_run_layout(runs_root, run_id)
    obj = {
        "kind": "checkpoint",
        "t": t,
        "products": {
            p.product_id: product_mutable_state(p, current_t=t if current_t is None else current_t) for p in products
        },
    }
    path = _checkpoint_path(base, t)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return path


def read_env_checkpoint(runs_root: str, run_id: str, t: int) -> dict | None:
    p = _checkpoint_path(run_dir(runs_root, run_id), t)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_env_snapshot(runs_root: str, run_id: str, t: int) -> dict | None:
    p = _step_path(run_dir(runs_root, run_id), "env_snapshot", t)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
