"""Per-step supplier-side state updates: inventory, price change, delist, recovery.

Operates in-memory on Product objects; caller persists changes.
"""

from __future__ import annotations

from typing import Iterable

from core.entities import EventLog, Product
from core.rng import derive_rng


def update_products(
    products: Iterable[Product],
    t: int,
    master_seed: int,
    sup_cfg: dict,
) -> list[EventLog]:
    events: list[EventLog] = []
    recover_lo, recover_hi = sup_cfg["recover_steps"]
    factor_lo, factor_hi = sup_cfg["price_adjust_factor"]
    delay_lo, delay_hi = sup_cfg.get("timeout_delay_hours", [24, 96])

    for p in products:
        # 1) inventory increment
        if p.is_listed_by_supplier and p.quantity < p.max_quantity:
            new_q = min(p.quantity + p.hourly_increment, p.max_quantity)
            if new_q != p.quantity:
                events.append(
                    EventLog(
                        t=t,
                        event_type="inventory_inc",
                        entity_id=p.product_id,
                        payload={"from": p.quantity, "to": new_q},
                    )
                )
                p.quantity = new_q

        # 2) price change events (only fire on original procurement-price items)
        if p.price_recover_t is None and abs(p.price - p.base_price) < 1e-9 and p.is_listed_by_supplier:
            rng = derive_rng(master_seed, "price_change", p.product_id, t)
            u_fire = float(rng.random())
            if u_fire < p.price_change_rate:
                factor = float(rng.uniform(factor_lo, factor_hi))
                duration = int(rng.integers(recover_lo, recover_hi + 1))
                new_price = round(p.base_price * factor, 2)
                p.price = new_price
                p.price_recover_t = t + duration
                events.append(
                    EventLog(
                        t=t,
                        event_type="price_change",
                        entity_id=p.product_id,
                        payload={
                            "new_price": new_price,
                            "ref_price": p.ref_price,
                            "base_price": p.base_price,
                            "factor": factor,
                            "recover_t": p.price_recover_t,
                        },
                    )
                )

        # 3) delist events
        if p.is_listed_by_supplier:
            rng = derive_rng(master_seed, "delist", p.product_id, t)
            u_fire = float(rng.random())
            if u_fire < p.supplier_delist_rate:
                duration = int(rng.integers(recover_lo, recover_hi + 1))
                p.is_listed_by_supplier = False
                p.delist_recover_t = t + duration
                events.append(
                    EventLog(
                        t=t,
                        event_type="supplier_delist",
                        entity_id=p.product_id,
                        payload={"recover_t": p.delist_recover_t},
                    )
                )

        # 4) recovery: price
        if p.price_recover_t is not None and t >= p.price_recover_t and abs(p.price - p.base_price) > 1e-9:
            old_price = p.price
            p.price = p.base_price
            p.price_recover_t = None
            events.append(
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

        # 5) recovery: delist
        if p.delist_recover_t is not None and t >= p.delist_recover_t and not p.is_listed_by_supplier:
            p.is_listed_by_supplier = True
            p.delist_recover_t = None
            events.append(EventLog(t=t, event_type="supplier_relist", entity_id=p.product_id, payload={}))

        # 6) supplier-side shipping delay: while active, new orders snapshot a longer
        # supplier_ship_hours; existing order snapshots are not changed.
        if not p.timeout_active and p.is_listed_by_supplier:
            rng = derive_rng(master_seed, "timeout", p.product_id, t)
            u_fire = float(rng.random())
            if u_fire < p.timeout_rate:
                duration = int(rng.integers(recover_lo, recover_hi + 1))
                delay = int(rng.integers(delay_lo, max(delay_lo + 1, delay_hi + 1)))
                before_ship = int(p.supplier_ship_hours)
                p.timeout_active = True
                p.timeout_recover_t = t + duration
                p.supplier_ship_hours = int(p.base_ship_hours) + delay
                events.append(
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

        # 7) recovery: timeout
        if p.timeout_recover_t is not None and t >= p.timeout_recover_t and p.timeout_active:
            before_ship = int(p.supplier_ship_hours)
            p.timeout_active = False
            p.timeout_recover_t = None
            p.supplier_ship_hours = int(p.base_ship_hours)
            events.append(
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

    return events
