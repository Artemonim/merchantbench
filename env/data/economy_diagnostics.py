"""Offline CES listing-day diagnostics for synthetic-economy ablations.

These helpers evaluate catalog demand and gross profit at a chosen sale
price with lifecycle=1 and rating=1. They do not start the simulator.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, TypedDict

import numpy as np

from core.demand import MIN_SALE_PRICE

# * Matches agent.baselines.auto_seed.DEFAULT_MARKUP (rule_based sale = markup * cost).
RULE_BASED_DEFAULT_MARKUP = 2.00
TEN_X_COST_MULTIPLIER = 10.0
APPLIANCES_CATEGORY = "appliances"


class CatalogEconomyAggregates(TypedDict):
    """Mean CES listing-day metrics over a catalog (or a category slice)."""

    n_products: int
    share_eps_lt_1: float
    mean_margin_at_ref: float
    mean_q_day_at_ref: float
    mean_q_day_at_rule_markup: float
    mean_gross_day_at_ref: float
    mean_gross_day_at_rule_markup: float
    mean_gross_day_at_10x_cost: float


def listing_day_demand_at_sale(
    product: Any,
    sale_price: float,
    small_share: float,
) -> float:
    """Return CES listing-day demand at ``sale_price``.

    Hourly weights sum to 1, so a day at lifecycle=1 and rating=1 is
    ``mean(market_curve) * small_share * (sale / ref) ** (-ε)``.

    Args:
        product: Catalog product with ``market_curve``, ``ref_price``,
            and ``elasticity``.
        sale_price: Merchant listing price.
        small_share: Shop share of market demand.

    Returns:
        Expected units per listing-day, or ``0.0`` when the CES inputs
        are invalid (``sale < 0.01``, ``ref <= 0``, ``ε < 0``, or
        non-finite).
    """
    try:
        sale = float(sale_price)
        ref = float(product.ref_price)
        elasticity = float(product.elasticity)
        share = float(small_share)
        curve_mean = float(np.mean(product.market_curve))
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(sale)
        or sale < MIN_SALE_PRICE
        or not math.isfinite(ref)
        or ref <= 0.0
        or not math.isfinite(elasticity)
        or elasticity < 0.0
        or not math.isfinite(share)
        or share < 0.0
        or not math.isfinite(curve_mean)
        or curve_mean <= 0.0
    ):
        return 0.0
    scale = curve_mean * share
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    # * Same CES as core.demand.expected_demand, collapsed over a day.
    try:
        log_demand = math.log(scale) - elasticity * (math.log(sale) - math.log(ref))
    except (OverflowError, ValueError):
        return 0.0
    if not math.isfinite(log_demand):
        return 0.0
    try:
        demand = math.exp(log_demand)
    except OverflowError:
        return 0.0
    if not math.isfinite(demand) or demand <= 0.0:
        return 0.0
    return float(demand)


def listing_day_gross_at_sale(
    product: Any,
    sale_price: float,
    small_share: float,
) -> float:
    """Return expected listing-day gross profit at ``sale_price``.

    Gross is ``q_day * (sale - cost)`` where ``cost`` is ``product.price``.

    Args:
        product: Catalog product with ``price`` plus CES demand fields.
        sale_price: Merchant listing price.
        small_share: Shop share of market demand.

    Returns:
        Expected gross profit per listing-day, or ``0.0`` when demand or
        cost is invalid.
    """
    try:
        sale = float(sale_price)
        cost = float(product.price)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sale) or not math.isfinite(cost):
        return 0.0
    demand = listing_day_demand_at_sale(product, sale, small_share)
    gross = demand * (sale - cost)
    if not math.isfinite(gross):
        return 0.0
    return float(gross)


def catalog_economy_aggregates(
    products: Iterable[Any],
    small_share: float,
    *,
    rule_markup: float = RULE_BASED_DEFAULT_MARKUP,
    category: str | None = None,
) -> CatalogEconomyAggregates:
    """Return catalog-level CES listing-day aggregates.

    Metrics use lifecycle=1 and rating=1. Rule-based markup prices at
    ``rule_markup * cost``. The 10× probe prices at
    ``TEN_X_COST_MULTIPLIER * cost``.

    Args:
        products: Catalog products.
        small_share: Shop share of market demand.
        rule_markup: Multiplier applied to cost for the rule_based probe.
        category: If set, restrict to this ``product.category``.

    Returns:
        Mapping of share / mean demand / mean gross metrics.

    Raises:
        ValueError: If the (filtered) catalog is empty, or ``rule_markup``
            is not finite and positive.
    """
    markup = float(rule_markup)
    if not math.isfinite(markup) or markup <= 0.0:
        raise ValueError(f"rule_markup must be positive and finite, got {rule_markup!r}")
    catalog = _select_products(products, category)
    n = len(catalog)
    share = float(small_share)

    eps_lt_1 = 0.0
    margins: list[float] = []
    q_at_ref: list[float] = []
    q_at_markup: list[float] = []
    gross_at_ref: list[float] = []
    gross_at_markup: list[float] = []
    gross_at_10x: list[float] = []
    for product in catalog:
        elasticity = _finite_float(getattr(product, "elasticity", None))
        if elasticity is not None and elasticity < 1.0:
            eps_lt_1 += 1.0
        ref = _finite_float(getattr(product, "ref_price", None))
        cost = _finite_float(getattr(product, "price", None))
        if ref is not None and ref > 0.0 and cost is not None:
            margins.append((ref - cost) / ref)
        else:
            margins.append(0.0)

        q_ref = listing_day_demand_at_sale(product, ref if ref is not None else 0.0, share)
        q_at_ref.append(q_ref)
        gross_at_ref.append(
            listing_day_gross_at_sale(product, ref if ref is not None else 0.0, share)
        )

        sale_markup = (cost * markup) if cost is not None else 0.0
        q_at_markup.append(listing_day_demand_at_sale(product, sale_markup, share))
        gross_at_markup.append(listing_day_gross_at_sale(product, sale_markup, share))

        sale_10x = (cost * TEN_X_COST_MULTIPLIER) if cost is not None else 0.0
        gross_at_10x.append(listing_day_gross_at_sale(product, sale_10x, share))

    return {
        "n_products": n,
        "share_eps_lt_1": eps_lt_1 / float(n),
        "mean_margin_at_ref": float(np.mean(margins)),
        "mean_q_day_at_ref": float(np.mean(q_at_ref)),
        "mean_q_day_at_rule_markup": float(np.mean(q_at_markup)),
        "mean_gross_day_at_ref": float(np.mean(gross_at_ref)),
        "mean_gross_day_at_rule_markup": float(np.mean(gross_at_markup)),
        "mean_gross_day_at_10x_cost": float(np.mean(gross_at_10x)),
    }


def _select_products(products: Iterable[Any], category: str | None) -> list[Any]:
    """Return the catalog, optionally filtered by category."""
    catalog = list(products)
    if category is not None:
        catalog = [product for product in catalog if product.category == category]
    if not catalog:
        if category is None:
            raise ValueError("products must be non-empty")
        raise ValueError(f"no products in category {category!r}")
    return catalog


def _finite_float(value: Any) -> float | None:
    """Parse a finite float, or return ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
