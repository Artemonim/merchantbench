"""Lazy supplier inventory helpers.

Supplier stock naturally replenishes over time, but large catalogs cannot afford
to materialize that increment for every product on every tick. These helpers expose
the effective stock at a given timestep and only write back to the Product when
the caller needs a concrete state transition, such as procurement or checkpoint.
"""
from __future__ import annotations

from core.entities import Product


def effective_quantity(product: Product, t: int) -> int:
    if not product.is_listed_by_supplier:
        return int(product.quantity)
    elapsed = max(0, int(t) - int(getattr(product, "quantity_updated_t", 0) or 0))
    return int(min(product.max_quantity, product.quantity + elapsed * product.hourly_increment))


def materialize_quantity(product: Product, t: int) -> int:
    qty = effective_quantity(product, t)
    product.quantity = qty
    product.quantity_updated_t = int(t)
    return qty


def consume_quantity(product: Product, t: int, n: int = 1) -> bool:
    qty = materialize_quantity(product, t)
    if qty < int(n):
        return False
    product.quantity = qty - int(n)
    product.quantity_updated_t = int(t)
    return True
