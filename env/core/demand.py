"""Demand prediction + order generation.

q_h(p) = market_curve[t mod 365]
       * small_share
       * hour_share[category, hour_of_day]
       * (price / ref_price) ** (-elasticity)
       * lifecycle_factor(days_since_first_listed)

For each (agent, listed product) pair at step t: draw n ~ Poisson(q_h). Emit n potential orders
with preset anomaly + anomaly time. Orders carry agent_id so the simulator's auto-purchase
phase + order_manager can route cash mutations + listing sale bumps back to the right merchant.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from core.entities import AnomalyKind, Order, Product, StoreListing
from core.rng import derive_rng


# Agent-facing prices use two-decimal store currency.  Keep the same floor in
# the tool schema/handlers, and fail closed here for legacy or manually-corrupt
# listings that bypassed those entry points.
MIN_SALE_PRICE = 0.01

# This is a numerical/resource fallback, not normal demand calibration. Default
# scenarios operate far below it. It keeps an extreme discount or malformed
# private data from feeding an unbounded value into the Poisson sampler.
MAX_EXPECTED_DEMAND_PER_LISTING_STEP = 1_000.0


def _hour_of_day(t: int, step_hours: int) -> int:
    return int((t * step_hours) % 24)


def _day_index(t: int, step_hours: int, day_offset: int = 0) -> int:
    return int((t * step_hours) // 24) + int(day_offset)


def lifecycle_factor(
    t: int,
    first_listed_at: int,
    step_hours: int,
    start: float = 0.1,
    ramp_days: int = 10,
    decay_rate: float = 0.04,
    floor: float = 0.10,
) -> float:
    """Listing lifecycle multiplier: linear ramp then exponential decay.

    days <= ramp_days: start + (1-start) * days/ramp_days
    days >  ramp_days: floor + (1-floor) * exp(-decay_rate * (days - ramp_days))
    """
    hours_since = max(0, t - first_listed_at) * step_hours
    days = hours_since / 24.0
    if days <= ramp_days:
        return start + (1.0 - start) * days / ramp_days
    return floor + (1.0 - floor) * math.exp(-decay_rate * (days - ramp_days))


def expected_demand(
    product: Product,
    listing: StoreListing,
    hourly_w: np.ndarray,
    t: int,
    step_hours: int,
    small_share: float,
    day_offset: int = 0,
    lifecycle_cfg: Optional[dict] = None,
) -> float:
    h = _hour_of_day(t, step_hours)
    day = _day_index(t, step_hours, day_offset) % 365
    try:
        sale_price = float(listing.sale_price)
        ref = float(product.ref_price)
        elasticity = float(product.elasticity)
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(sale_price)
        or sale_price < MIN_SALE_PRICE
        or not math.isfinite(ref)
        or ref <= 0.0
        or not math.isfinite(elasticity)
        or elasticity < 0.0
    ):
        return 0.0

    # Compute the same multiplicative formula in log space so no intermediate
    # price ratio or exponent can underflow/overflow before the final fallback.
    try:
        scale = float(
            product.market_curve[day]
            * small_share
            * hourly_w[h]
        )
    except (IndexError, OverflowError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    log_demand = (
        math.log(scale)
        - elasticity * (math.log(sale_price) - math.log(ref))
    )
    if lifecycle_cfg is not None:
        try:
            lf = lifecycle_factor(
                t, listing.first_listed_at, step_hours,
                start=lifecycle_cfg["start"],
                ramp_days=lifecycle_cfg["ramp_days"],
                decay_rate=lifecycle_cfg["decay_rate"],
                floor=lifecycle_cfg["floor"],
            )
        except (OverflowError, TypeError, ValueError):
            return 0.0
        if not math.isfinite(lf) or lf <= 0.0:
            return 0.0
        log_demand += math.log(lf)
    if math.isnan(log_demand):
        return 0.0
    if log_demand >= math.log(MAX_EXPECTED_DEMAND_PER_LISTING_STEP):
        return MAX_EXPECTED_DEMAND_PER_LISTING_STEP
    try:
        base = math.exp(log_demand)
    except OverflowError:
        return MAX_EXPECTED_DEMAND_PER_LISTING_STEP
    return base if math.isfinite(base) and base > 0.0 else 0.0


def generate_orders_for_step(
    listed_triples: Iterable[tuple[Product, StoreListing, str]],
    hourly_dist: dict[str, np.ndarray],
    t: int,
    step_hours: int,
    small_share: float,
    master_seed: int,
    rating_factors: Optional[dict[str, float]] = None,
    day_offset: int = 0,
    normal_delay_hours: int = 168,
    lifecycle_cfg: Optional[dict] = None,
) -> list[Order]:
    """`listed_triples` yields (product, listing, agent_id). Each agent's listings
    drive an independent Poisson stream via per-(agent, product, t) RNG seed.

    `rating_factors` maps agent_id → demand multiplier from shop rating
    (1.0 = neutral / 5★ shops boost, 1★ shops penalized). Missing agent_id
    defaults to 1.0. None disables rating entirely.

    Orders are emitted regardless of supplier delist / stock status — downstream
    auto-purchase enforces those checks and may convert the order into a
    platform-rule violation (stockout / insufficient balance).
    """
    normal_delay_steps = max(1, int(normal_delay_hours / step_hours))
    out: list[Order] = []
    for product, listing, agent_id in listed_triples:
        w = hourly_dist.get(product.category)
        if w is None:
            continue
        q = expected_demand(
            product, listing, w, t, step_hours, small_share,
            day_offset=day_offset,
            lifecycle_cfg=lifecycle_cfg,
        )
        if rating_factors is not None:
            q *= rating_factors.get(agent_id, 1.0)
        if not math.isfinite(q) or q <= 0:
            continue
        q = min(q, MAX_EXPECTED_DEMAND_PER_LISTING_STEP)
        arrival_rng = derive_rng(master_seed, "arrival", agent_id, product.product_id, t)
        n = int(arrival_rng.poisson(q))
        if n <= 0:
            continue
        for i in range(n):
            order = _make_order(product, listing, agent_id, t, i, master_seed, normal_delay_steps)
            out.append(order)
    return out


def _make_order(
    product: Product,
    listing: StoreListing,
    agent_id: str,
    t: int,
    i: int,
    master_seed: int,
    normal_delay_steps: int,
) -> Order:
    order_id = f"{agent_id}-{product.product_id}-t{t}-{i}"
    promised = t + int(product.supplier_ship_hours) + product.logistics_hours

    a_rng = derive_rng(master_seed, "anomaly_type", agent_id, product.product_id, t, i)
    u_type = float(a_rng.random())
    kind = _pick_anomaly(u_type, product)

    at_rng = derive_rng(master_seed, "anomaly_time", agent_id, product.product_id, t, i)
    if kind == "cancel":
        offset = int(at_rng.integers(1, max(2, int(product.supplier_ship_hours) + 1)))
        preset_anomaly_t = t + offset
    elif kind in ("refund", "only_refund"):
        # refund / only_refund now lands randomly within 0-7 days after delivery,
        # same window as the normal/bad_review settlement period.
        offset = int(at_rng.integers(0, normal_delay_steps + 1))
        preset_anomaly_t = t + int(product.supplier_ship_hours) + product.logistics_hours + offset
    else:
        # "normal" and "bad_review" both settle on the normal account-period clock;
        # the bad_review fine is applied at that same settle time, so no separate offset.
        preset_anomaly_t = -1

    sd_rng = derive_rng(master_seed, "settlement_delay", agent_id, product.product_id, t, i)
    if kind in ("normal", "bad_review"):
        settlement_delay_steps = int(sd_rng.integers(0, normal_delay_steps + 1))
    else:
        settlement_delay_steps = -1

    order = Order(
        order_id=order_id,
        product_id=product.product_id,
        supplier_id=product.supplier_id,
        agent_id=agent_id,
        order_t=t,
        promised_delivery_t=promised,
        sale_price=listing.sale_price,
        purchase_price=product.price,
        supplier_ship_hours=int(product.supplier_ship_hours),
        preset_anomaly=kind,
        preset_anomaly_t=preset_anomaly_t,
        settlement_delay_steps=settlement_delay_steps,
    )
    return order


def _pick_anomaly(u: float, product: Product) -> AnomalyKind:
    probs = [product.cancel_rate, product.refund_rate, product.only_refund_rate, product.bad_review_rate]
    labels = ["cancel", "refund", "only_refund", "bad_review"]
    cum = 0.0
    for p, lab in zip(probs, labels):
        cum += p
        if u < cum:
            return lab  # type: ignore[return-value]
    return "normal"
