"""Discrete-event scheduler for supplier-side product state.

The old runtime polled every product every tick. This module keeps only the next
due supplier event per active channel and schedules the next event after each
transition. Random schedules are deterministic for the new runtime because all
sampling is keyed by master_seed, product_id, event_type, and sequence number.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable

from core.entities import EventLog, Product
from core.inventory import materialize_quantity
from core.rng import derive_rng

STOCHASTIC_EVENT_TYPES = ("price_change", "supplier_delist", "supplier_timeout")
RECOVERY_EVENT_TYPES = ("price_recover", "supplier_relist", "supplier_timeout_end")


@dataclass(frozen=True)
class SupplierEvent:
    due_t: int
    product_id: str
    event_type: str
    seq: int = 0
    payload: dict | None = None

    def to_row(self) -> dict:
        return {
            "due_t": int(self.due_t),
            "product_id": self.product_id,
            "event_type": self.event_type,
            "seq": int(self.seq),
            "payload": self.payload or {},
        }

    @classmethod
    def from_row(cls, row) -> "SupplierEvent":
        payload = row.get("payload", {}) if isinstance(row, dict) else row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload or "{}")
        return cls(
            due_t=int(row["due_t"]),
            product_id=str(row["product_id"]),
            event_type=str(row["event_type"]),
            seq=int(row["seq"]),
            payload=payload or {},
        )


def _rate_for(product: Product, event_type: str) -> float:
    if event_type == "price_change":
        return float(product.price_change_rate)
    if event_type == "supplier_delist":
        return float(product.supplier_delist_rate)
    if event_type == "supplier_timeout":
        return float(product.timeout_rate)
    raise ValueError(f"unknown stochastic supplier event {event_type!r}")


def _next_due_t(master_seed: int, product_id: str, event_type: str, seq: int, earliest_t: int, p: float) -> int | None:
    if p <= 0:
        return None
    if p >= 1:
        return int(earliest_t)
    rng = derive_rng(master_seed, "supplier_event_wait", product_id, event_type, seq)
    u = min(max(float(rng.random()), 1e-12), 1.0 - 1e-12)
    wait = int(math.floor(math.log(1.0 - u) / math.log(1.0 - p)))
    return int(earliest_t) + wait


def schedule_stochastic_event(
    product: Product, event_type: str, *, master_seed: int, earliest_t: int, seq: int = 0, horizon: int | None = None
) -> SupplierEvent | None:
    due_t = _next_due_t(master_seed, product.product_id, event_type, seq, earliest_t, _rate_for(product, event_type))
    if due_t is None:
        return None
    if horizon is not None and due_t >= int(horizon):
        return None
    return SupplierEvent(due_t=due_t, product_id=product.product_id, event_type=event_type, seq=seq)


def initial_supplier_events(
    products: Iterable[Product], master_seed: int, start_t: int, horizon: int | None = None
) -> list[SupplierEvent]:
    events: list[SupplierEvent] = []
    for p in products:
        if p.price_recover_t is not None:
            events.append(SupplierEvent(p.price_recover_t, p.product_id, "price_recover", 0))
        elif p.is_listed_by_supplier and abs(p.price - p.base_price) < 1e-9:
            ev = schedule_stochastic_event(
                p,
                "price_change",
                master_seed=master_seed,
                earliest_t=start_t,
                seq=0,
                horizon=horizon,
            )
            if ev is not None:
                events.append(ev)
        if p.delist_recover_t is not None:
            events.append(SupplierEvent(p.delist_recover_t, p.product_id, "supplier_relist", 0))
        elif p.is_listed_by_supplier:
            ev = schedule_stochastic_event(
                p,
                "supplier_delist",
                master_seed=master_seed,
                earliest_t=start_t,
                seq=0,
                horizon=horizon,
            )
            if ev is not None:
                events.append(ev)
        if p.timeout_recover_t is not None:
            events.append(SupplierEvent(p.timeout_recover_t, p.product_id, "supplier_timeout_end", 0))
        elif p.is_listed_by_supplier and not p.timeout_active:
            ev = schedule_stochastic_event(
                p,
                "supplier_timeout",
                master_seed=master_seed,
                earliest_t=start_t,
                seq=0,
                horizon=horizon,
            )
            if ev is not None:
                events.append(ev)
    return events


def _payload_rng(master_seed: int, event: SupplierEvent):
    return derive_rng(master_seed, "supplier_event_payload", event.product_id, event.event_type, event.seq)


def _schedule_after(
    product: Product, event_type: str, event: SupplierEvent, t: int, master_seed: int, horizon: int | None
) -> SupplierEvent | None:
    return schedule_stochastic_event(
        product,
        event_type,
        master_seed=master_seed,
        earliest_t=t + 1,
        seq=event.seq + 1,
        horizon=horizon,
    )


def apply_due_events(
    products_by_id: dict[str, Product],
    due_events: Iterable[SupplierEvent],
    t: int,
    master_seed: int,
    sup_cfg: dict,
    horizon: int | None = None,
) -> tuple[list[EventLog], set[str], list[SupplierEvent], list[tuple[str, str]]]:
    logs: list[EventLog] = []
    dirty: set[str] = set()
    followups: list[SupplierEvent] = []
    cancel_pending: list[tuple[str, str]] = []
    recover_lo, recover_hi = sup_cfg["recover_steps"]
    factor_lo, factor_hi = sup_cfg["price_adjust_factor"]
    delay_lo, delay_hi = sup_cfg.get("timeout_delay_hours", [24, 96])

    for event in sorted(due_events, key=lambda e: (e.due_t, e.product_id, e.event_type, e.seq)):
        p = products_by_id.get(event.product_id)
        if p is None:
            continue

        if event.event_type == "price_change":
            if not p.is_listed_by_supplier or p.price_recover_t is not None or abs(p.price - p.base_price) >= 1e-9:
                continue
            rng = _payload_rng(master_seed, event)
            factor = float(rng.uniform(factor_lo, factor_hi))
            duration = int(rng.integers(recover_lo, recover_hi + 1))
            p.price = round(p.base_price * factor, 2)
            p.price_recover_t = t + duration
            dirty.add(p.product_id)
            logs.append(
                EventLog(
                    t=t,
                    event_type="price_change",
                    entity_id=p.product_id,
                    payload={
                        "new_price": p.price,
                        "ref_price": p.ref_price,
                        "base_price": p.base_price,
                        "factor": factor,
                        "recover_t": p.price_recover_t,
                    },
                )
            )
            followups.append(SupplierEvent(p.price_recover_t, p.product_id, "price_recover", event.seq))

        elif event.event_type == "price_recover":
            if p.price_recover_t is not None and t >= p.price_recover_t:
                old_price = p.price
                p.price = p.base_price
                p.price_recover_t = None
                dirty.add(p.product_id)
                logs.append(
                    EventLog(
                        t=t,
                        event_type="price_recover",
                        entity_id=p.product_id,
                        payload={
                            "from": old_price,
                            "to": p.base_price,
                            "ref_price": p.ref_price,
                            "base_price": p.base_price,
                        },
                    )
                )
                ev = _schedule_after(p, "price_change", event, t, master_seed, horizon)
                if ev is not None and p.is_listed_by_supplier:
                    followups.append(ev)

        elif event.event_type == "supplier_delist":
            if not p.is_listed_by_supplier:
                continue
            materialize_quantity(p, t)
            rng = _payload_rng(master_seed, event)
            duration = int(rng.integers(recover_lo, recover_hi + 1))
            p.is_listed_by_supplier = False
            p.delist_recover_t = t + duration
            dirty.add(p.product_id)
            cancel_pending.extend(
                [
                    (p.product_id, "price_change"),
                    (p.product_id, "supplier_timeout"),
                    (p.product_id, "supplier_delist"),
                ]
            )
            logs.append(
                EventLog(
                    t=t, event_type="supplier_delist", entity_id=p.product_id, payload={"recover_t": p.delist_recover_t}
                )
            )
            followups.append(SupplierEvent(p.delist_recover_t, p.product_id, "supplier_relist", event.seq))

        elif event.event_type == "supplier_relist":
            if p.delist_recover_t is not None and t >= p.delist_recover_t:
                p.is_listed_by_supplier = True
                p.delist_recover_t = None
                p.quantity_updated_t = t
                dirty.add(p.product_id)
                logs.append(EventLog(t=t, event_type="supplier_relist", entity_id=p.product_id, payload={}))
                for event_type in STOCHASTIC_EVENT_TYPES:
                    if event_type == "price_change" and p.price_recover_t is not None:
                        continue
                    if event_type == "supplier_timeout" and p.timeout_active:
                        continue
                    ev = schedule_stochastic_event(
                        p,
                        event_type,
                        master_seed=master_seed,
                        earliest_t=t + 1,
                        seq=event.seq + 1,
                        horizon=horizon,
                    )
                    if ev is not None:
                        followups.append(ev)

        elif event.event_type == "supplier_timeout":
            if not p.is_listed_by_supplier or p.timeout_active:
                continue
            rng = _payload_rng(master_seed, event)
            duration = int(rng.integers(recover_lo, recover_hi + 1))
            delay = int(rng.integers(delay_lo, max(delay_lo + 1, delay_hi + 1)))
            before_ship = int(p.supplier_ship_hours)
            p.timeout_active = True
            p.timeout_recover_t = t + duration
            p.supplier_ship_hours = int(p.base_ship_hours) + delay
            dirty.add(p.product_id)
            logs.append(
                EventLog(
                    t=t,
                    event_type="supplier_timeout",
                    entity_id=p.product_id,
                    payload={
                        "recover_t": p.timeout_recover_t,
                        "before_supplier_ship_hours": before_ship,
                        "after_supplier_ship_hours": int(p.supplier_ship_hours),
                    },
                )
            )
            followups.append(SupplierEvent(p.timeout_recover_t, p.product_id, "supplier_timeout_end", event.seq))

        elif event.event_type == "supplier_timeout_end":
            if p.timeout_recover_t is not None and t >= p.timeout_recover_t:
                before_ship = int(p.supplier_ship_hours)
                p.timeout_active = False
                p.timeout_recover_t = None
                p.supplier_ship_hours = int(p.base_ship_hours)
                dirty.add(p.product_id)
                logs.append(
                    EventLog(
                        t=t,
                        event_type="supplier_timeout_end",
                        entity_id=p.product_id,
                        payload={
                            "before_supplier_ship_hours": before_ship,
                            "after_supplier_ship_hours": int(p.supplier_ship_hours),
                        },
                    )
                )
                ev = _schedule_after(p, "supplier_timeout", event, t, master_seed, horizon)
                if ev is not None and p.is_listed_by_supplier:
                    followups.append(ev)

    return logs, dirty, followups, cancel_pending
