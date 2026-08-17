import pytest

from core.economy_v6 import EconomyV6
from core.entities import Cash, Order, OrderStatusRow, Product, StoreListing
from core.order_manager import _apply_fee, _apply_penalty, _credit_cash, step_orders
from core.simulator import AgentState, Environment
from storage import db as dbm


SETTLEMENT = {
    "normal_delay_hours": 168,
}

PLATFORM_RULES = {
    "cancel_penalty_amount": 0.0,
    "refund_penalty_amount": 8.0,
    "only_refund_penalty_amount": 0.0,
    "bad_review_penalty_amount": 5.0,
    "timeout_penalty_amount": 3.0,
    "stockout_penalty_amount": 5.0,
    "insufficient_balance_penalty_amount": 5.0,
    "max_promised_ship_hours": 48,
    "default_promised_ship_hours": 48,
}

SUP_CFG = {"timeout_delay_hours": [24, 96]}


def _mkproduct(ship=1, logi=1, category="electronics") -> Product:
    return Product(
        product_id="P0", name="x", quantity=10,
        price=100.0, ref_price=100.0, supplier_id="s", supplier_name="S",
        ship_hours=ship, logistics_hours=logi, category=category,
        historical_avg_rating=4.5, shop_rating=4.5,
        return_buyer_rate=0.18, supplier_age_years=3.5,
        cancel_rate=0.0, refund_rate=0.0, only_refund_rate=0.0,
        bad_review_rate=0.0,
        max_quantity=100, hourly_increment=5,
        timeout_rate=0.0, price_change_rate=0.0, supplier_delist_rate=0.0,
        elasticity=1.0, market_curve=[1.0] * 365,
    )


def _mkorder(product: Product, listing: StoreListing, cash: Cash,
             anomaly="normal", anomaly_t=-1, t: int = 0,
             settlement_delay_steps: int = -1) -> Order:
    """Construct an order already auto-purchased at t=t, mirroring simulator._auto_purchase_new_orders."""
    cash.balance -= product.price
    cash.in_transit += product.price
    product.quantity -= 1
    listing.cum_sales += 1
    listing.cum_revenue += listing.sale_price
    supplier_ship_hours = int(product.supplier_ship_hours)
    o = Order(order_id="O0", product_id="P0", supplier_id="s",
              agent_id="agent_0",
              order_t=t, promised_delivery_t=t + supplier_ship_hours + product.logistics_hours,
              sale_price=listing.sale_price, purchase_price=product.price,
              supplier_ship_hours=supplier_ship_hours,
              preset_anomaly=anomaly, preset_anomaly_t=anomaly_t,
              settlement_delay_steps=settlement_delay_steps,
              current_status="ordered", purchase_t=t)
    o.actual_ship_hours = supplier_ship_hours
    o.status_log.append(OrderStatusRow(t=t, status="ordered"))
    return o


def _step(orders, products, listing, cash, t, master_seed=42, economy=None):
    return step_orders(
        orders, {"P0": products},
        {("agent_0", "P0"): listing},
        {"agent_0": cash},
        t=t, step_hours=1, settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
        sup_cfg=SUP_CFG, master_seed=master_seed, economy=economy,
    )


def test_penalty_uses_balance_before_guarantee_and_clamps_both_pools():
    cash = Cash(balance=3.0, deposit_pool=10.0)

    _apply_penalty(cash, 8.0)

    assert cash.balance == 0.0
    assert cash.deposit_pool == 5.0
    assert cash.cumulative_fine == 8.0

    _apply_penalty(cash, 20.0)

    assert cash.balance == 0.0
    assert cash.deposit_pool == 0.0
    assert cash.cumulative_fine == 28.0
    assert getattr(cash, "_guarantee_exhausted") is True


def test_cash_credit_refills_guarantee_before_balance_without_changing_value():
    cash = Cash(balance=7.0, deposit_pool=990.0, in_transit=2.0, receivable=1.0)
    before = cash.balance + cash.deposit_pool + cash.in_transit + cash.receivable

    _credit_cash(cash, 6.0, 1000.0)
    assert cash.deposit_pool == 996.0
    assert cash.balance == 7.0

    _credit_cash(cash, 9.0, 1000.0)
    assert cash.deposit_pool == 1000.0
    assert cash.balance == 12.0
    after = cash.balance + cash.deposit_pool + cash.in_transit + cash.receivable
    assert after == before + 15.0


@pytest.mark.parametrize(
    ("anomaly", "terminal_t", "expected_balance"),
    [
        ("normal", 170, 1010.0),
        ("cancel", 2, 990.0),
        ("refund", 10, 982.0),
        ("bad_review", 170, 1005.0),
    ],
)
def test_all_order_cash_credit_paths_refill_guarantee_first(
    anomaly, terminal_t, expected_balance,
):
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=990.0)
    anomaly_t = 2 if anomaly == "cancel" else 10 if anomaly == "refund" else -1
    order = _mkorder(p, listing, cash, anomaly=anomaly, anomaly_t=anomaly_t)
    order.realized_cost = 100.0

    for t in (1, 2):
        step_orders(
            [order], {"P0": p}, {("agent_0", "P0"): listing},
            {"agent_0": cash}, t=t, step_hours=1,
            settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
            initial_deposit=1000.0,
        )
    if terminal_t > 2:
        step_orders(
            [order], {"P0": p}, {("agent_0", "P0"): listing},
            {"agent_0": cash}, t=terminal_t, step_hours=1,
            settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
            initial_deposit=1000.0,
        )

    assert cash.deposit_pool == 1000.0
    assert cash.balance == expected_balance


def test_full_normal_lifecycle():
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)
    assert order.current_status == "ordered"
    assert cash.balance == 900.0 and cash.in_transit == 100.0

    _step([order], p, listing, cash, t=1)
    assert order.current_status == "shipped"
    _step([order], p, listing, cash, t=2)
    assert order.current_status == "delivered"
    assert cash.receivable == 100.0 and cash.in_transit == 0.0
    _step([order], p, listing, cash, t=2 + 168)
    assert order.current_status == "settled_normal"
    assert cash.balance == 1000.0 and cash.receivable == 0.0


def test_cancel_lifecycle_has_no_extra_penalty():
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="cancel", anomaly_t=2)
    order.realized_cost = 100.0  # mirror simulator's auto-purchase debit
    _step([order], p, listing, cash, t=1)
    assert order.current_status == "shipped"
    _step([order], p, listing, cash, t=2)
    assert order.current_status == "cancelled"
    assert cash.cumulative_fine == 0.0
    assert cash.balance == 1000.0
    # cost was credited back on cancel -> realized_cost zeroed, no extra penalty.
    assert order.realized_cost == 0.0
    assert order.net_profit == 0.0


def test_refund_after_delivery():
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    order.realized_cost = 100.0  # mirror simulator's auto-purchase debit
    _step([order], p, listing, cash, t=1)
    _step([order], p, listing, cash, t=2)
    assert order.current_status == "delivered"
    _step([order], p, listing, cash, t=10)
    assert order.current_status == "settled_refund"
    assert cash.cumulative_fine == 8.0
    # Goods returned: cost credited back, only the fixed 8 penalty hits cash net.
    # balance trace: 900 (after purchase) → 900 (delivered, in_transit→receivable) →
    #                 900 + 100 (cost return) - 8 (penalty) = 992.
    assert abs(cash.balance - 992.0) < 1e-6
    assert order.realized_cost == 0.0
    assert order.realized_revenue == 0.0
    assert order.net_profit == -8.0


def test_only_refund_does_not_recover_cost():
    """only_refund: customer keeps the goods. Cost is permanently lost; no extra penalty."""
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=500.0)
    order = _mkorder(p, listing, cash, anomaly="only_refund", anomaly_t=10)
    order.realized_cost = 100.0
    _step([order], p, listing, cash, t=1)
    _step([order], p, listing, cash, t=2)
    assert order.current_status == "delivered"
    _step([order], p, listing, cash, t=10)
    assert order.current_status == "settled_only_refund"
    assert cash.cumulative_fine == 0.0
    # No cost recovery → realized_cost stays at purchase_price
    assert order.realized_cost == 100.0
    assert order.realized_revenue == 0.0
    assert order.net_profit == -100.0  # 0 - 100 cost - 0 penalty


def test_bad_review_normal_settle_plus_fine():
    """bad_review: order delivers and settles like normal (revenue collected on 168h account
    period), then a fine is deducted from balance first. Status: settled_bad_review.
    Net cash effect for sale_price=120, cost=100, fixed bad_review penalty=5:
      balance: 1000 → 900 (purchase) → 900 → 900 → ... → 900 + 120 (revenue) - 5 (fine) = 1015
      cumulative_fine: +5
      realized_revenue=120, realized_cost=100, total_penalty=5 → net_profit=15
    """
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=500.0)
    rules = dict(PLATFORM_RULES)
    order = _mkorder(p, listing, cash, anomaly="bad_review", anomaly_t=-1)
    order.realized_cost = 100.0  # mirror simulator's auto-purchase debit

    # shipped at t=1, delivered at t=2
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    assert order.current_status == "delivered"
    assert cash.receivable == 120.0

    # settles on the normal 168h account period; fine is applied alongside
    events, mutated, _new_status, daily = step_orders(
        [order], {"P0": p}, {("agent_0", "P0"): listing},
        {"agent_0": cash}, t=2 + 168, step_hours=1,
        settlement_cfg=SETTLEMENT, platform_rules=rules,
        sup_cfg=SUP_CFG, master_seed=42,
    )
    assert order.current_status == "settled_bad_review"
    assert cash.receivable == 0.0
    # balance: 900 (after purchase) + 120 (revenue) - 5 (fine) = 1015
    assert abs(cash.balance - 1015.0) < 1e-6
    assert cash.deposit_pool == 500.0  # fine routed to balance only, deposit untouched
    assert abs(cash.cumulative_fine - 5.0) < 1e-6
    assert order.realized_revenue == 120.0
    assert order.realized_cost == 100.0  # cost is real — customer kept and paid for the goods
    assert abs(order.total_penalty - 5.0) < 1e-6
    assert abs(order.net_profit - 15.0) < 1e-6

    # event + daily aggregates (GMV accrues at procurement, not settlement)
    assert any(e.event_type == "order_settled_bad_review" for e in events)
    day = ((2 + 168) * 1) // 24
    assert daily[day]["anomaly_count"] == 1
    assert abs(daily[day]["fine_total"] - 5.0) < 1e-6


def test_bad_review_fine_overflows_balance_to_deposit():
    """bad_review with balance-source: if the post-revenue balance is too small to absorb
    the fine, the overflow lands on deposit_pool (mirroring the existing refund overflow path)."""
    rules = {**PLATFORM_RULES, "bad_review_penalty_amount": 180.0}
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=10.0, deposit_pool=500.0)
    order = _mkorder(p, listing, cash, anomaly="bad_review", anomaly_t=-1)
    # _mkorder subtracts 100 (purchase) → balance = -90
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    # at settle: receivable -=120, balance +=120 → balance = -90+120 = 30
    # then penalty 180 from balance first: balance 30 → 0, overflow 150 to deposit: 500-150 = 350
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2 + 168, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    assert order.current_status == "settled_bad_review"
    assert abs(cash.balance - 0.0) < 1e-6
    assert abs(cash.deposit_pool - 350.0) < 1e-6
    assert abs(cash.cumulative_fine - 180.0) < 1e-6


def test_slow_logistics_after_on_time_ship_does_not_trigger_late():
    """Timeout penalties are based on shipping, not delivery."""
    p = _mkproduct(ship=2, logi=100)
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    _step([order], p, listing, cash, t=2)
    assert order.current_status == "shipped"
    assert order.actual_ship_hours == 2
    assert order.actual_logistics_hours == 100

    _step([order], p, listing, cash, t=50)
    assert order.current_status == "shipped"
    assert cash.cumulative_fine == 0.0

    _step([order], p, listing, cash, t=102)
    assert order.current_status == "delivered"
    assert cash.cumulative_fine == 0.0


def test_supplier_shipping_delay_snapshot_extends_ship_time_not_logistics():
    """Supplier shipping delay affects the ordered -> shipped interval only."""
    p = _mkproduct(ship=1, logi=5)
    p.supplier_ship_hours = 3
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg={"timeout_delay_hours": [2, 2]}, master_seed=42)
    assert order.current_status == "ordered"
    assert order.supplier_ship_hours == 3
    assert order.actual_logistics_hours == 0

    _step([order], p, listing, cash, t=3)
    assert order.current_status == "shipped"
    assert order.shipped_t == 3
    deliver_t = order.shipped_t + order.actual_logistics_hours
    _step([order], p, listing, cash, t=deliver_t)
    assert order.current_status == "delivered"
    assert cash.cumulative_fine == 0.0


def test_supplier_shipping_delay_only_affects_new_order_snapshots():
    """Supplier shipping delay changes current supplier_ship_hours for future orders only."""
    p = _mkproduct(ship=2, logi=5)
    p.base_ship_hours = 2
    p.supplier_ship_hours = 2
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)

    before_delay = _mkorder(p, listing, cash, t=0)
    assert before_delay.supplier_ship_hours == 2

    p.supplier_ship_hours = 10
    during_delay = _mkorder(p, listing, cash, t=1)
    assert during_delay.supplier_ship_hours == 10
    assert before_delay.supplier_ship_hours == 2

    _step([before_delay], p, listing, cash, t=2)
    assert before_delay.current_status == "shipped"
    _step([during_delay], p, listing, cash, t=10)
    assert during_delay.current_status == "ordered"
    _step([during_delay], p, listing, cash, t=11)
    assert during_delay.current_status == "shipped"


def test_late_triggers_when_actual_ship_exceeds_platform_default():
    """When supplier ship time exceeds platform default (48h), order goes late, then still ships and delivers."""
    p = _mkproduct(ship=50, logi=1)
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    events, _mutated, _new_status, _daily = step_orders(
        [order], {"P0": p}, {("agent_0", "P0"): listing},
        {"agent_0": cash}, t=49, step_hours=1,
        settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
        sup_cfg=SUP_CFG, master_seed=42,
    )
    assert order.current_status == "late"
    assert order.supplier_ship_hours == 50
    assert order.late_t == 49
    assert cash.cumulative_fine == 3.0
    late_payload = next(e.payload for e in events if e.event_type == "order_late")
    assert late_payload["supplier_ship_hours"] == 50
    assert late_payload["supplier_logistics_hours"] == 1
    assert "actual_ship_hours" not in late_payload
    assert late_payload["actual_logistics_hours"] is None
    assert "promised_logistics_hours" not in late_payload

    events, _mutated, _new_status, _daily = step_orders(
        [order], {"P0": p}, {("agent_0", "P0"): listing},
        {"agent_0": cash}, t=50, step_hours=1,
        settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
        sup_cfg=SUP_CFG, master_seed=42,
    )
    assert order.current_status == "shipped"
    assert cash.cumulative_fine == 3.0
    shipped_payload = next(e.payload for e in events if e.event_type == "order_shipped")
    assert shipped_payload["supplier_ship_hours"] == 50
    assert shipped_payload["supplier_logistics_hours"] == 1
    assert "actual_ship_hours" not in shipped_payload
    assert shipped_payload["actual_logistics_hours"] is None
    assert "promised_logistics_hours" not in shipped_payload
    events, _mutated, _new_status, _daily = _step([order], p, listing, cash, t=51)
    assert order.current_status == "delivered"
    assert cash.cumulative_fine == 3.0
    delivered_payload = next(e.payload for e in events if e.event_type == "order_delivered")
    assert delivered_payload["supplier_ship_hours"] == 50
    assert delivered_payload["supplier_logistics_hours"] == 1
    assert "actual_ship_hours" not in delivered_payload
    assert delivered_payload["actual_logistics_hours"] == 1


def test_late_and_ship_same_tick_keeps_ship_time_consistent():
    """When the late threshold and actual ship time meet on one tick, ship on that tick."""
    p = _mkproduct(ship=50, logi=1)
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    events, _mutated, new_status, _daily = step_orders(
        [order], {"P0": p}, {("agent_0", "P0"): listing},
        {"agent_0": cash}, t=50, step_hours=1,
        settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
        sup_cfg=SUP_CFG, master_seed=42,
    )

    assert order.current_status == "shipped"
    assert order.late_t == 50
    assert order.shipped_t == 50
    assert order.actual_ship_hours == 50
    assert order.shipped_t - order.purchase_t == order.actual_ship_hours
    assert [row.status for row in new_status] == ["late", "shipped"]
    assert [e.event_type for e in events] == ["order_late", "order_shipped"]
    assert cash.cumulative_fine == 3.0


def test_late_order_ignores_later_supplier_shipping_delay():
    """A late-but-unshipped order keeps the supplier_ship_hours snapshotted at purchase."""
    p = _mkproduct(ship=60, logi=1)
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=49, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg={"timeout_delay_hours": [5, 5]}, master_seed=42)
    assert order.current_status == "late"
    assert order.supplier_ship_hours == 60

    p.supplier_ship_hours = 70
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=50, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg={"timeout_delay_hours": [5, 5]}, master_seed=42)
    assert order.current_status == "late"
    assert order.shipped_t is None
    assert order.supplier_ship_hours == 60

    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=60, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg={"timeout_delay_hours": [5, 5]}, master_seed=42)
    assert order.current_status == "shipped"
    assert order.shipped_t == 60
    assert order.supplier_ship_hours == 60
    assert cash.cumulative_fine == 3.0


def test_supplier_shipping_delay_no_late_when_under_platform_default():
    """A longer supplier_ship_hours does not cause a penalty when under platform default (48h)."""
    p = _mkproduct(ship=1, logi=5)
    p.supplier_ship_hours = 3
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)

    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg={"timeout_delay_hours": [2, 2]}, master_seed=42)
    assert order.current_status == "ordered"
    _step([order], p, listing, cash, t=3)
    assert order.current_status == "shipped"
    _step([order], p, listing, cash, t=8)
    assert order.current_status == "delivered"
    assert cash.cumulative_fine == 0.0


def test_late_to_delivered_to_settled_normal():
    """Full intermediate-state lifecycle: ordered → late → shipped → delivered → settled_normal."""
    p = _mkproduct(ship=50, logi=1)
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)
    _step([order], p, listing, cash, t=49)
    assert order.current_status == "late"
    _step([order], p, listing, cash, t=50)
    assert order.current_status == "shipped"
    _step([order], p, listing, cash, t=51)
    assert order.current_status == "delivered"
    _step([order], p, listing, cash, t=51 + 168)
    assert order.current_status == "settled_normal"
    # late fee 3 + settled_normal revenue 100 net.
    assert abs(cash.balance - (1000 - 100 - 3 + 100)) < 1e-6
    assert cash.cumulative_fine == 3.0


def test_refund_penalty_uses_balance_before_guarantee():
    rules = dict(PLATFORM_RULES)
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=3.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=10, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    # Refund recovers purchase cost to balance=1000; fixed 8 penalty hits balance.
    assert cash.deposit_pool == 3.0
    assert abs(cash.balance - 992.0) < 1e-6
    assert cash.cumulative_fine == 8.0


def test_refund_penalty_preserves_full_guarantee_when_balance_is_sufficient():
    rules = dict(PLATFORM_RULES)
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=1000.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=10, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    assert cash.deposit_pool == 1000.0
    assert cash.balance == 992.0
    assert cash.cumulative_fine == 8.0


def _auto_buy_inline(o: Order, p: Product, listing: StoreListing, cash: Cash, t: int) -> None:
    cash.balance -= p.price
    cash.in_transit += p.price
    p.quantity -= 1
    listing.cum_sales += 1
    listing.cum_revenue += listing.sale_price
    o.current_status = "ordered"
    o.purchase_t = t
    o.status_log.append(OrderStatusRow(t=t, status="ordered"))


def test_refund_penalty_preserves_partial_guarantee_when_balance_is_sufficient():
    rules = dict(PLATFORM_RULES)
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=500.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=10, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    # refund recovers purchase cost to balance=1000; fixed 8 penalty hits balance first.
    assert cash.deposit_pool == 500.0
    assert abs(cash.balance - 992.0) < 1e-6
    assert cash.cumulative_fine == 8.0


def test_refund_penalty_overflows_to_guarantee_after_cost_recovery():
    """refund now credits back purchase_price, so even with low balance the penalty
    can be absorbed by the recovered cost. Verifies overflow logic still works
    when penalty exceeds available balance."""
    # Use a fixed penalty amount that forces overflow even after cost recovery.
    rules = {**PLATFORM_RULES, "refund_penalty_amount": 150.0}
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=10.0, deposit_pool=500.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    # _mkorder takes 100 → balance=-90 from initial 10. We've gone slightly negative; that's fine for the test.
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=1, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=2, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    # at t=10 refund: receivable -= 100. balance += 100 (cost recovery): -90+100 = 10
    # penalty = 150 (balance source): take 10 from balance → 0, remainder 140 to deposit: 500-140 = 360
    step_orders([order], {"P0": p}, {("agent_0", "P0"): listing},
                {"agent_0": cash}, t=10, step_hours=1,
                settlement_cfg=SETTLEMENT, platform_rules=rules,
                sup_cfg=SUP_CFG, master_seed=42)
    assert order.current_status == "settled_refund"
    assert abs(cash.balance - 0.0) < 1e-6
    assert abs(cash.deposit_pool - 360.0) < 1e-6
    assert cash.cumulative_fine == 150.0
    assert order.realized_cost == 0.0


def test_cancel_total_penalty_is_zero():
    """Cancel recovers cost and does not add an extra penalty."""
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="cancel", anomaly_t=2)
    order.realized_cost = 100.0  # mirror simulator's auto-purchase debit
    _step([order], p, listing, cash, t=1)
    _step([order], p, listing, cash, t=2)
    assert order.current_status == "cancelled"
    assert order.total_penalty == 0.0
    assert order.realized_revenue == 0.0
    assert order.realized_cost == 0.0  # cost credited back on cancel
    assert order.net_profit == 0.0


def test_normal_settlement_sets_realized_revenue():
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)
    order.realized_cost = 100.0  # purchase_price
    _step([order], p, listing, cash, t=1)
    _step([order], p, listing, cash, t=2)
    _step([order], p, listing, cash, t=2 + 168)
    assert order.current_status == "settled_normal"
    assert order.realized_revenue == 120.0
    assert order.realized_cost == 100.0
    assert order.total_penalty == 0.0
    assert order.net_profit == 20.0  # margin




def test_orders_routed_by_agent_id():
    """Two agents listing same product → cash mutations stay separate."""
    p = _mkproduct()
    listing_a = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    listing_b = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_1")
    cash_a = Cash(balance=1000.0)
    cash_b = Cash(balance=1000.0)

    o_a = Order(order_id="O_A", product_id="P0", supplier_id="s",
                agent_id="agent_0", order_t=0, promised_delivery_t=2,
                sale_price=100.0, purchase_price=100.0)
    o_b = Order(order_id="O_B", product_id="P0", supplier_id="s",
                agent_id="agent_1", order_t=0, promised_delivery_t=2,
                sale_price=120.0, purchase_price=100.0)

    _auto_buy_inline(o_a, p, listing_a, cash_a, t=0)
    _auto_buy_inline(o_b, p, listing_b, cash_b, t=0)
    assert cash_a.balance == 900.0
    assert cash_b.balance == 900.0

    step_orders([o_a, o_b], {"P0": p},
                {("agent_0", "P0"): listing_a, ("agent_1", "P0"): listing_b},
                {"agent_0": cash_a, "agent_1": cash_b},
                t=1, step_hours=1, settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg=SUP_CFG, master_seed=42)
    step_orders([o_a, o_b], {"P0": p},
                {("agent_0", "P0"): listing_a, ("agent_1", "P0"): listing_b},
                {"agent_0": cash_a, "agent_1": cash_b},
                t=2, step_hours=1, settlement_cfg=SETTLEMENT, platform_rules=PLATFORM_RULES,
                sup_cfg=SUP_CFG, master_seed=42)
    assert o_a.current_status == "delivered"
    assert o_b.current_status == "delivered"
    assert cash_a.receivable == 100.0
    assert cash_b.receivable == 120.0


def test_normal_and_bad_review_use_random_settlement():
    """Both normal and bad_review orders respect explicit settlement_delay_steps."""
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    good = _mkorder(p, listing, cash, settlement_delay_steps=0)
    bad = _mkorder(p, listing, cash, anomaly="bad_review", settlement_delay_steps=168)
    good.realized_cost = 100.0
    bad.realized_cost = 100.0

    _step([good, bad], p, listing, cash, t=1)
    _step([good, bad], p, listing, cash, t=2)
    assert good.current_status == "delivered"
    assert bad.current_status == "delivered"
    _step([good, bad], p, listing, cash, t=3)
    assert good.current_status == "settled_normal"
    assert bad.current_status == "delivered"


def _v6_economy(
    *,
    take=False,
    fulfillment=False,
    refund=False,
    reverse=True,
    take_default=0.08,
    take_by=None,
    fee_default=8.0,
    fee_by=None,
    recovery=0.85,
) -> EconomyV6:
    return EconomyV6.from_scenario({
        "economy_v6": {
            "enabled": True,
            "take_rate": {
                "enabled": take,
                "default": take_default,
                "by_category": take_by or {"womenswear": 0.10, "appliances": 0.05},
            },
            "fulfillment": {
                "enabled": fulfillment,
                "default_fee": fee_default,
                "by_category": fee_by or {"womenswear": 6.0, "appliances": 15.0},
            },
            "refund": {
                "enabled": refund,
                "cost_recovery_rate": recovery,
                "reverse_fulfillment": reverse,
            },
        }
    })


def test_economy_v6_master_off_ignores_nested_flags():
    eco = EconomyV6.from_scenario({
        "economy_v6": {
            "enabled": False,
            "take_rate": {"enabled": True, "default": 0.08},
            "fulfillment": {"enabled": True, "default_fee": 8.0},
            "refund": {
                "enabled": True,
                "cost_recovery_rate": 0.85,
                "reverse_fulfillment": True,
            },
        }
    })
    assert eco.take_rate_enabled is False
    assert eco.fulfillment_enabled is False
    assert eco.refund_enabled is False
    assert eco.reverse_fulfillment is False
    assert eco.cost_recovery_rate() == 1.0


def test_apply_fee_overflows_to_deposit_without_cumulative_fine():
    cash = Cash(balance=3.0, deposit_pool=10.0)

    _apply_fee(cash, 8.0)

    assert cash.balance == 0.0
    assert cash.deposit_pool == 5.0
    assert cash.cumulative_fine == 0.0


def test_take_rate_credits_sale_minus_commission():
    """Take-rate keeps sticker GMV and credits sale * (1-τ)."""
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash)
    order.realized_cost = 100.0
    eco = _v6_economy(take=True)

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    assert order.current_status == "delivered"
    assert cash.receivable == 120.0
    _step([order], p, listing, cash, t=2 + 168, economy=eco)

    assert order.current_status == "settled_normal"
    assert order.realized_revenue == 120.0
    assert order.commission_amount == 12.0
    assert order.total_penalty == 0.0
    assert order.net_profit == 8.0
    assert cash.receivable == 0.0
    assert abs(cash.balance - 1008.0) < 1e-6


def test_take_rate_also_applies_on_bad_review_settlement():
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="bad_review")
    order.realized_cost = 100.0
    eco = _v6_economy(take=True)

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    _step([order], p, listing, cash, t=2 + 168, economy=eco)

    assert order.current_status == "settled_bad_review"
    assert order.commission_amount == 12.0
    assert order.realized_revenue == 120.0
    assert order.total_penalty == 5.0
    assert order.net_profit == 3.0
    assert cash.cumulative_fine == 5.0
    assert abs(cash.balance - 1003.0) < 1e-6


def test_refund_v6_recovers_partial_cost_and_keeps_unrecovered_cogs():
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    order.realized_cost = 100.0
    eco = _v6_economy(refund=True, reverse=False)

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    _step([order], p, listing, cash, t=10, economy=eco)

    assert order.current_status == "settled_refund"
    assert order.cost_recovery_rate == 0.85
    assert order.realized_cost == 15.0
    assert order.realized_revenue == 0.0
    assert order.reverse_logistics_fee == 0.0
    assert order.refund_loss == 15.0
    assert order.total_penalty == 8.0
    assert cash.cumulative_fine == 8.0
    assert order.net_profit == -23.0
    assert abs(cash.balance - 977.0) < 1e-6


def test_refund_reverse_fulfillment_is_not_in_total_penalty():
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    order.realized_cost = 100.0
    eco = _v6_economy(refund=True, reverse=True)

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    _step([order], p, listing, cash, t=10, economy=eco)

    assert order.current_status == "settled_refund"
    assert order.realized_cost == 15.0
    assert order.reverse_logistics_fee == 6.0
    assert order.refund_loss == 21.0
    assert order.total_penalty == 8.0
    assert cash.cumulative_fine == 8.0
    assert order.net_profit == -29.0
    assert abs(cash.balance - 971.0) < 1e-6


def test_refund_reverse_fulfillment_overflows_deposit_without_extra_fine():
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0, deposit_pool=500.0)
    order = _mkorder(p, listing, cash, anomaly="refund", anomaly_t=10)
    order.realized_cost = 100.0
    eco = _v6_economy(refund=True, reverse=True, fee_by={"womenswear": 100.0})

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    cash.balance = 2.0
    _step([order], p, listing, cash, t=10, economy=eco)

    # recovered 85 → balance 87; reverse F 100 takes 87 + 13 deposit;
    # refund fine 8 then takes deposit. Fine stays 8 RMB only.
    assert order.total_penalty == 8.0
    assert cash.cumulative_fine == 8.0
    assert order.reverse_logistics_fee == 100.0
    assert abs(cash.balance - 0.0) < 1e-6
    assert abs(cash.deposit_pool - 479.0) < 1e-6


def test_cancel_does_not_refund_outbound_logistics_fee():
    p = _mkproduct()
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="cancel", anomaly_t=2)
    order.realized_cost = 100.0
    order.logistics_fee = 6.0

    _step([order], p, listing, cash, t=1)
    _step([order], p, listing, cash, t=2)

    assert order.current_status == "cancelled"
    assert order.realized_cost == 0.0
    assert order.logistics_fee == 6.0
    assert order.net_profit == -6.0
    assert cash.balance == 1000.0


def test_only_refund_does_not_add_v6_dispute_or_commission():
    p = _mkproduct(category="womenswear")
    listing = StoreListing(product_id="P0", sale_price=100.0, agent_id="agent_0")
    cash = Cash(balance=1000.0)
    order = _mkorder(p, listing, cash, anomaly="only_refund", anomaly_t=10)
    order.realized_cost = 100.0
    order.logistics_fee = 6.0
    eco = _v6_economy(take=True, refund=True, reverse=True)

    _step([order], p, listing, cash, t=1, economy=eco)
    _step([order], p, listing, cash, t=2, economy=eco)
    _step([order], p, listing, cash, t=10, economy=eco)

    assert order.current_status == "settled_only_refund"
    assert order.commission_amount == 0.0
    assert order.reverse_logistics_fee == 0.0
    assert order.realized_cost == 100.0
    assert order.total_penalty == 0.0
    assert order.net_profit == -106.0


def _purchase_env(tmp_path, economy_block, balance=1000.0, category="womenswear",
                  quantity=10):
    conn = dbm.open_db(str(tmp_path / "state.db"))
    run_id = "econ-v6"
    product = _mkproduct(category=category)
    product.quantity = quantity
    listing = StoreListing(product_id="P0", sale_price=120.0, agent_id="agent_0")
    dbm.upsert_listing(conn, run_id, "agent_0", listing)
    state = AgentState(
        agent_id="agent_0", name="Agent",
        cash=Cash(balance, 500.0),
        listings={"P0": listing},
    )
    scenario = {
        "run": {"step_hours": 1, "horizon_steps": 100, "master_seed": 42},
        "data": {"small_share": 0.01},
        "settlement": {"normal_delay_hours": 168},
        "platform_rules": PLATFORM_RULES,
        "supplier_ranges": {},
        "economy_v6": economy_block,
    }
    env = Environment(run_id, conn, scenario, str(tmp_path), [product], {}, {"agent_0": state})
    env._step_revenue = {aid: 0.0 for aid in env.agents}
    env._step_cost = {aid: 0.0 for aid in env.agents}
    return env, product


def _candidate_order() -> Order:
    return Order(
        order_id="O-v6",
        product_id="P0",
        supplier_id="s",
        agent_id="agent_0",
        order_t=0,
        promised_delivery_t=2,
        sale_price=120.0,
        purchase_price=100.0,
    )


def test_fulfillment_fee_is_extra_cash_at_purchase_not_in_transit(tmp_path):
    env, _product = _purchase_env(tmp_path, {
        "enabled": True,
        "take_rate": {"enabled": False, "default": 0.08, "by_category": {}},
        "fulfillment": {
            "enabled": True,
            "default_fee": 8.0,
            "by_category": {"womenswear": 6.0},
        },
        "refund": {"enabled": False, "cost_recovery_rate": 0.85, "reverse_fulfillment": True},
    })
    events = []
    kept = env._auto_purchase_new_orders([_candidate_order()], PLATFORM_RULES, events)
    st = env.agents["agent_0"]

    assert len(kept) == 1
    assert kept[0].current_status == "ordered"
    assert kept[0].logistics_fee == 6.0
    assert st.cash.balance == 894.0
    assert st.cash.in_transit == 100.0


def test_fulfillment_insufficient_balance_includes_fee(tmp_path):
    env, _product = _purchase_env(tmp_path, {
        "enabled": True,
        "take_rate": {"enabled": False, "default": 0.08, "by_category": {}},
        "fulfillment": {
            "enabled": True,
            "default_fee": 8.0,
            "by_category": {"womenswear": 6.0},
        },
        "refund": {"enabled": False, "cost_recovery_rate": 0.85, "reverse_fulfillment": True},
    }, balance=103.0)
    events = []
    kept = env._auto_purchase_new_orders([_candidate_order()], PLATFORM_RULES, events)
    st = env.agents["agent_0"]

    assert len(kept) == 1
    assert kept[0].current_status == "insufficient_balance"
    assert kept[0].logistics_fee == 0.0
    assert kept[0].total_penalty == 5.0
    assert st.cash.in_transit == 0.0
    assert abs(st.cash.balance - 98.0) < 1e-6


def test_stockout_does_not_charge_fulfillment_fee(tmp_path):
    env, _product = _purchase_env(tmp_path, {
        "enabled": True,
        "take_rate": {"enabled": False, "default": 0.08, "by_category": {}},
        "fulfillment": {
            "enabled": True,
            "default_fee": 8.0,
            "by_category": {"womenswear": 6.0},
        },
        "refund": {"enabled": False, "cost_recovery_rate": 0.85, "reverse_fulfillment": True},
    }, quantity=0)
    events = []
    kept = env._auto_purchase_new_orders([_candidate_order()], PLATFORM_RULES, events)
    st = env.agents["agent_0"]

    assert kept[0].current_status == "stockout"
    assert kept[0].logistics_fee == 0.0
    assert st.cash.balance == 995.0
    assert st.cash.in_transit == 0.0
