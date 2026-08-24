"""Order lifecycle transitions per timestep.

Orders are auto-purchased at creation by `simulator._auto_purchase_new_orders` and
enter the state machine at `ordered`.

State machine:
  ordered      -> (t - purchase_t > default_promised_ship_hours)        -> late
  ordered/late -> (t >= purchase_t + supplier_ship_hours)               -> shipped
  shipped + preset cancel & t >= preset_anomaly_t                       -> cancelled (terminal, penalty)
  shipped/late + t >= shipped_t + actual_logistics_hours                -> delivered
  delivered + normal & t >= delivered_t + 168h                          -> settled_normal
  delivered + bad_review & t >= delivered_t + 168h                      -> settled_bad_review (revenue collected + fine)
  delivered + refund / only_refund & t >= preset_anomaly_t              -> settled_refund / settled_only_refund

The `late` state is intermediate: the order still ships, arrives, and follows
the normal delivered→settled_* path. Only the late-entry incurs the timeout
penalty.

Cash mutations route by order.agent_id into the per-agent Cash object;
listing bumps route into the per-(agent_id, product_id) StoreListing.
"""

from __future__ import annotations

from typing import Iterable

from core.economy_v6 import EconomyV6
from core.entities import Cash, EventLog, Order, OrderStatusRow, Product, StoreListing


def _credit_cash(cash: Cash, amount: float, initial_deposit: float) -> None:
    """Restore the initial guarantee amount, then credit remaining usable cash."""
    amount = max(0.0, float(amount))
    target = max(0.0, float(initial_deposit))
    refill = min(amount, max(0.0, target - cash.deposit_pool))
    cash.deposit_pool += refill
    amount -= refill
    cash.balance += amount


def _debit_balance_then_deposit(cash: Cash, amount: float) -> None:
    """Take ``amount`` from usable cash first, then from the locked guarantee."""
    amount = max(0.0, float(amount))
    take_bal = min(max(0.0, cash.balance), amount)
    cash.balance = max(0.0, cash.balance - take_bal)
    remainder = amount - take_bal
    take_dep = min(max(0.0, cash.deposit_pool), remainder)
    cash.deposit_pool = max(0.0, cash.deposit_pool - take_dep)
    if remainder > 0.0 and cash.deposit_pool <= 0.0:
        # `_check_death_for` consumes this transient marker later in the same
        # simulator step. It deliberately survives any later cash credit.
        setattr(cash, "_guarantee_exhausted", True)


def _apply_penalty(
    cash: Cash,
    penalty: float,
) -> None:
    """Deduct a fine from balance first, then from the locked guarantee."""
    penalty = max(0.0, float(penalty))
    cash.cumulative_fine += penalty
    _debit_balance_then_deposit(cash, penalty)


def _apply_fee(cash: Cash, amount: float) -> None:
    """Debit a non-fine fee from balance first, then from the locked guarantee.

    Unlike `_apply_penalty`, this does not increment `cumulative_fine`.
    """
    _debit_balance_then_deposit(cash, amount)


def _penalty_amount(platform_rules: dict, kind: str, sale_price: float) -> float:
    """Resolve a penalty as a fixed amount when configured, else legacy ratio."""
    amount_key = f"{kind}_penalty_amount"
    if amount_key in platform_rules:
        return float(platform_rules[amount_key])
    return float(sale_price) * float(platform_rules[f"{kind}_penalty_ratio"])


def step_orders(
    orders: Iterable[Order],
    products_by_id: dict[str, Product],
    listings_by_key: dict[tuple[str, str], StoreListing],
    cash_by_agent: dict[str, Cash],
    t: int,
    step_hours: int,
    settlement_cfg: dict,
    platform_rules: dict,
    initial_deposit: float = 0.0,
    sup_cfg: dict | None = None,
    master_seed: int | None = None,
    economy: EconomyV6 | None = None,
) -> tuple[list[EventLog], list[Order], list[OrderStatusRow], dict]:
    """Mutate orders + per-agent cash + per-agent listings in-place.
    Returns (events, mutated_orders, new_status_rows, daily_delta).

    listings_by_key keyed by (agent_id, product_id).
    settlement_cfg supplies settlement timing (normal_delay_hours).
    platform_rules supplies penalty amounts or legacy ratios.
    economy supplies v6 take-rate / refund-haircut / reverse-fulfillment flags.
    sup_cfg + master_seed are accepted for compatibility with older callers; supplier
    shipping delays are already represented in product/order supplier_ship_hours.
    """
    events: list[EventLog] = []
    mutated: list[Order] = []
    new_status: list[OrderStatusRow] = []
    daily_delta: dict[int, dict] = {}
    eco = economy if economy is not None else EconomyV6()
    normal_delay_steps = max(1, int(settlement_cfg["normal_delay_hours"] / step_hours))
    # Platform-wide late detection threshold.
    default_promised = int(platform_rules.get("default_promised_ship_hours", 48))

    def _add_status(o: Order, status: str) -> None:
        row = OrderStatusRow(t=t, status=status)
        o.status_log.append(row)
        new_status.append(OrderStatusRow(t=t, status=status))
        o.current_status = status

    def _day(step_t: int) -> int:
        return int((step_t * step_hours) // 24)

    def _bump_day(day: int, key: str, val: float) -> None:
        d = daily_delta.setdefault(day, {"gmv": 0.0, "anomaly_count": 0, "fine_total": 0.0})
        if key == "anomaly_count":
            d[key] += int(val)
        else:
            d[key] += float(val)

    def _ensure_supplier_ship(o: Order, product: Product) -> int:
        if o.supplier_ship_hours > 0:
            supplier_ship = int(o.supplier_ship_hours)
        elif o.actual_ship_hours > 0:
            supplier_ship = int(o.actual_ship_hours)
            o.supplier_ship_hours = supplier_ship
        else:
            supplier_ship = int(product.supplier_ship_hours)
            o.supplier_ship_hours = supplier_ship
        if o.purchase_t is None or t < o.purchase_t + supplier_ship:
            return 0
        o.actual_ship_hours = supplier_ship
        o.actual_logistics_hours = int(product.logistics_hours)
        return supplier_ship

    def _order_time_payload(o: Order, product: Product | None) -> dict:
        return {
            "supplier_ship_hours": (
                int(o.supplier_ship_hours)
                if o.supplier_ship_hours > 0
                else int(product.supplier_ship_hours)
                if product
                else None
            ),
            "supplier_logistics_hours": int(product.logistics_hours) if product else None,
            "actual_logistics_hours": (
                int(o.actual_logistics_hours) if o.delivered_t is not None and o.actual_logistics_hours > 0 else None
            ),
        }

    def _category_for(o: Order) -> str:
        product = products_by_id.get(o.product_id)
        return product.category if product is not None else ""

    def _credit_sale(o: Order, cash: Cash) -> None:
        """Clear receivable and credit sticker GMV minus take-rate when enabled."""
        cash.receivable -= o.sale_price
        if eco.take_rate_enabled:
            o.commission_amount = round(o.sale_price * eco.take_rate(_category_for(o)), 2)
            _credit_cash(cash, o.sale_price - o.commission_amount, initial_deposit)
        else:
            _credit_cash(cash, o.sale_price, initial_deposit)
        o.realized_revenue = o.sale_price

    for o in orders:
        cash = cash_by_agent.get(o.agent_id)
        if cash is None:
            continue
        listings_by_key.get((o.agent_id, o.product_id))
        s = o.current_status

        if s in ("ordered", "late"):
            product = products_by_id.get(o.product_id)
            if product is None:
                continue
            if o.purchase_t is not None:
                supplier_ship = _ensure_supplier_ship(o, product)
                changed = False
                if s == "ordered" and t - o.purchase_t > default_promised:
                    penalty = _penalty_amount(platform_rules, "timeout", o.sale_price)
                    _apply_penalty(cash, penalty)
                    o.total_penalty += penalty
                    o.late_t = t
                    _add_status(o, "late")
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_late",
                            entity_id=o.order_id,
                            payload={"penalty": penalty, **_order_time_payload(o, product)},
                            agent_id=o.agent_id,
                        )
                    )
                    _bump_day(_day(t), "anomaly_count", 1)
                    _bump_day(_day(t), "fine_total", penalty)
                    changed = True
                if supplier_ship > 0 and t >= o.purchase_t + supplier_ship:
                    o.actual_logistics_hours = int(product.logistics_hours)
                    _add_status(o, "shipped")
                    o.shipped_t = t
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_shipped",
                            entity_id=o.order_id,
                            payload=_order_time_payload(o, product),
                            agent_id=o.agent_id,
                        )
                    )
                    changed = True
                if changed:
                    mutated.append(o)
                    continue
                if o.supplier_ship_hours > 0:
                    mutated.append(o)
                    continue

        if s in ("shipped", "late"):
            product = products_by_id.get(o.product_id)
            if product is None:
                continue

            if o.preset_anomaly == "cancel" and t >= o.preset_anomaly_t and s == "shipped":
                penalty = _penalty_amount(platform_rules, "cancel", o.sale_price)
                cash.in_transit -= o.purchase_price
                _credit_cash(cash, o.purchase_price, initial_deposit)
                _apply_penalty(cash, penalty)
                o.total_penalty += penalty
                # Cost was credited back to cash above; mark realized_cost as 0
                # so per-order net_profit reflects the penalty and any sunk outbound F.
                o.realized_cost = 0.0
                o.settled_t = t
                _add_status(o, "cancelled")
                events.append(
                    EventLog(
                        t=t,
                        event_type="order_cancelled",
                        entity_id=o.order_id,
                        payload={"penalty": penalty, **_order_time_payload(o, product)},
                        agent_id=o.agent_id,
                    )
                )
                _bump_day(_day(t), "anomaly_count", 1)
                _bump_day(_day(t), "fine_total", penalty)
                mutated.append(o)
                continue

            if o.shipped_t is not None and o.actual_logistics_hours > 0 and t >= o.shipped_t + o.actual_logistics_hours:
                o.delivered_t = t
                cash.in_transit -= o.purchase_price
                cash.receivable += o.sale_price
                _add_status(o, "delivered")
                events.append(
                    EventLog(
                        t=t,
                        event_type="order_delivered",
                        entity_id=o.order_id,
                        payload={
                            "receivable": o.sale_price,
                            "was_late": o.late_t is not None,
                            **_order_time_payload(o, product),
                        },
                        agent_id=o.agent_id,
                    )
                )
                mutated.append(o)
                continue

        if s == "delivered":
            if o.preset_anomaly == "normal":
                delay_steps = o.settlement_delay_steps if o.settlement_delay_steps >= 0 else normal_delay_steps
                if o.delivered_t is not None and t >= o.delivered_t + delay_steps:
                    _credit_sale(o, cash)
                    o.settled_t = t
                    _add_status(o, "settled_normal")
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_settled_normal",
                            entity_id=o.order_id,
                            payload={
                                "revenue": o.sale_price,
                                **_order_time_payload(o, products_by_id.get(o.product_id)),
                            },
                            agent_id=o.agent_id,
                        )
                    )
                    mutated.append(o)
                    continue
            elif o.preset_anomaly == "bad_review":
                delay_steps = o.settlement_delay_steps if o.settlement_delay_steps >= 0 else normal_delay_steps
                if o.delivered_t is not None and t >= o.delivered_t + delay_steps:
                    # Customer pays normally, then the bad-review fine is charged.
                    _credit_sale(o, cash)
                    penalty = _penalty_amount(platform_rules, "bad_review", o.sale_price)
                    _apply_penalty(cash, penalty)
                    o.total_penalty += penalty
                    o.settled_t = t
                    _add_status(o, "settled_bad_review")
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_settled_bad_review",
                            entity_id=o.order_id,
                            payload={
                                "penalty": penalty,
                                "revenue": o.sale_price,
                                **_order_time_payload(o, products_by_id.get(o.product_id)),
                            },
                            agent_id=o.agent_id,
                        )
                    )
                    _bump_day(_day(t), "anomaly_count", 1)
                    _bump_day(_day(t), "fine_total", penalty)
                    mutated.append(o)
                    continue
            elif o.preset_anomaly == "refund":
                if t >= o.preset_anomaly_t:
                    penalty = _penalty_amount(platform_rules, "refund", o.sale_price)
                    cash.receivable -= o.sale_price
                    if eco.refund_enabled:
                        recovery_rate = eco.cost_recovery_rate()
                        recovered = round(o.purchase_price * recovery_rate, 2)
                        _credit_cash(cash, recovered, initial_deposit)
                        o.cost_recovery_rate = recovery_rate
                        o.realized_cost = round(o.purchase_price - recovered, 2)
                        o.realized_revenue = 0.0
                        if eco.reverse_fulfillment:
                            fee = round(eco.fulfillment_fee(_category_for(o)), 2)
                            _apply_fee(cash, fee)
                            o.reverse_logistics_fee = fee
                        o.refund_loss = round((o.purchase_price - recovered) + o.reverse_logistics_fee, 2)
                    else:
                        # Goods returned → procurement cost is recovered as resellable inventory.
                        _credit_cash(cash, o.purchase_price, initial_deposit)
                        o.realized_cost = 0.0
                    _apply_penalty(cash, penalty)
                    o.total_penalty += penalty
                    o.settled_t = t
                    _add_status(o, "settled_refund")
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_settled_refund",
                            entity_id=o.order_id,
                            payload={"penalty": penalty, **_order_time_payload(o, products_by_id.get(o.product_id))},
                            agent_id=o.agent_id,
                        )
                    )
                    _bump_day(_day(t), "anomaly_count", 1)
                    _bump_day(_day(t), "fine_total", penalty)
                    mutated.append(o)
                    continue
            elif o.preset_anomaly == "only_refund":
                if t >= o.preset_anomaly_t:
                    penalty = _penalty_amount(platform_rules, "only_refund", o.sale_price)
                    cash.receivable -= o.sale_price
                    _apply_penalty(cash, penalty)
                    o.total_penalty += penalty
                    o.settled_t = t
                    _add_status(o, "settled_only_refund")
                    events.append(
                        EventLog(
                            t=t,
                            event_type="order_settled_only_refund",
                            entity_id=o.order_id,
                            payload={"penalty": penalty, **_order_time_payload(o, products_by_id.get(o.product_id))},
                            agent_id=o.agent_id,
                        )
                    )
                    _bump_day(_day(t), "anomaly_count", 1)
                    _bump_day(_day(t), "fine_total", penalty)
                    mutated.append(o)
                    continue

    return events, mutated, new_status, daily_delta
