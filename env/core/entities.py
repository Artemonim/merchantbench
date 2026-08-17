"""Pure data containers for the simulation. No business logic here."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


AnomalyKind = Literal["normal", "cancel", "refund", "only_refund", "bad_review"]

OrderStatus = Literal[
    "ordered",
    "shipped",
    "late",                # intermediate: actual ship time exceeded merchant's promise
    "stockout",            # terminal: auto-purchase rejected — supplier delisted or out of stock
    "insufficient_balance",# terminal: auto-purchase rejected — merchant cash < required procurement cash
    "cancelled",
    "delivered",
    "settled_normal",
    "settled_refund",
    "settled_only_refund",
    "settled_bad_review",  # terminal: delivered + settled normally with an extra fine for the bad review
]


@dataclass
class Product:
    product_id: str
    name: str
    quantity: int
    price: float
    ref_price: float
    supplier_id: str
    supplier_name: str
    ship_hours: int
    logistics_hours: int
    category: str

    # Trust signals shown to the agent alongside operational fields.
    # historical_avg_rating is per-product; the other three are per-supplier
    # and MUST be identical across every product sharing the same supplier_id
    # (enforced by Environment.__init__ on load).
    historical_avg_rating: float
    shop_rating: float
    return_buyer_rate: float
    supplier_age_years: float

    cancel_rate: float
    refund_rate: float
    only_refund_rate: float
    bad_review_rate: float

    max_quantity: int
    hourly_increment: int
    timeout_rate: float
    price_change_rate: float
    supplier_delist_rate: float

    elasticity: float
    market_curve: list[float]
    base_price: float = 0.0
    base_ship_hours: Optional[int] = None
    supplier_ship_hours: Optional[int] = None
    quantity_updated_t: int = 0

    is_listed_by_supplier: bool = True
    delist_recover_t: Optional[int] = None
    price_recover_t: Optional[int] = None
    timeout_active: bool = False
    timeout_recover_t: Optional[int] = None

    def __post_init__(self) -> None:
        if self.base_price <= 0:
            self.base_price = float(self.price)
        if self.base_ship_hours is None:
            self.base_ship_hours = int(self.ship_hours)
        if self.supplier_ship_hours is None:
            self.supplier_ship_hours = int(self.base_ship_hours)

    def visible(self) -> dict:
        """Agent-visible fields only."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "quantity": self.quantity,
            "price": self.price,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "supplier_ship_hours": int(self.supplier_ship_hours),
            "logistics_hours": self.logistics_hours,
            "category": self.category,
            "historical_avg_rating": self.historical_avg_rating,
            "shop_rating": self.shop_rating,
            "supplier_age_years": self.supplier_age_years,
        }


@dataclass
class OrderStatusRow:
    t: int
    status: OrderStatus


@dataclass
class Order:
    order_id: str
    product_id: str
    supplier_id: str
    order_t: int
    promised_delivery_t: int
    sale_price: float
    purchase_price: float
    agent_id: str = "agent_0"

    current_status: OrderStatus = "ordered"
    purchase_t: Optional[int] = None
    shipped_t: Optional[int] = None
    delivered_t: Optional[int] = None
    settled_t: Optional[int] = None

    preset_anomaly: AnomalyKind = "normal"
    preset_anomaly_t: int = 0
    # Supplier ordered -> shipped duration, snapshotted at procurement time.
    supplier_ship_hours: int = 0
    # Legacy DB/dashboard compatibility for runs created before supplier_ship_hours.
    # Active state-machine logic mirrors this from supplier_ship_hours only.
    actual_ship_hours: int = 0
    # Legacy DB/dashboard compatibility for runs created before the ship SLA
    # rename. Active tool/state-machine logic does not read or write this.
    promised_logistics_hours: int = 0
    # Actual transit hours used only for delivery scheduling.
    actual_logistics_hours: int = 0
    late_t: Optional[int] = None

    # Per-order settlement delay for normal/bad_review orders.
    # -1 means "unset / legacy", fallback to normal_delay_steps at runtime.
    settlement_delay_steps: int = -1

    # Per-order P&L attribution. realized_cost set at procurement (purchase_price);
    # realized_revenue set on settled_normal / settled_bad_review credit; total_penalty
    # accumulates across every penalty event applied to this order
    # (cancel / late / refund / only_refund / bad_review).
    realized_revenue: float = 0.0
    realized_cost: float = 0.0
    total_penalty: float = 0.0
    # * v6 cash fees. total_penalty stays platform-violation fines only.
    commission_amount: float = 0.0
    logistics_fee: float = 0.0
    reverse_logistics_fee: float = 0.0
    cost_recovery_rate: float = 1.0
    # * Diagnostic: unrecovered COGS + reverse F. Not subtracted again in net_profit.
    refund_loss: float = 0.0

    status_log: list[OrderStatusRow] = field(default_factory=list)

    @property
    def net_profit(self) -> float:
        return (
            self.realized_revenue
            - self.realized_cost
            - self.total_penalty
            - self.commission_amount
            - self.logistics_fee
            - self.reverse_logistics_fee
        )

    def visible(self) -> dict:
        return {
            "order_id": self.order_id,
            "product_id": self.product_id,
            "supplier_id": self.supplier_id,
            "agent_id": self.agent_id,
            "order_t": self.order_t,
            "promised_delivery_t": self.promised_delivery_t,
            "sale_price": self.sale_price,
            "purchase_price": self.purchase_price,
            "current_status": self.current_status,
            "supplier_ship_hours": self.supplier_ship_hours,
            "promised_logistics_hours": self.promised_logistics_hours,
            "actual_logistics_hours": self.actual_logistics_hours,
            "late_t": self.late_t,
            "realized_revenue": self.realized_revenue,
            "realized_cost": self.realized_cost,
            "total_penalty": self.total_penalty,
            "net_profit": self.net_profit,
        }


@dataclass
class Cash:
    balance: float
    deposit_pool: float = 0.0
    in_transit: float = 0.0
    receivable: float = 0.0
    cumulative_fine: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StoreListing:
    product_id: str
    sale_price: float
    agent_id: str = "agent_0"
    cum_sales: int = 0
    cum_revenue: float = 0.0
    listed_at: int = 0
    first_listed_at: int = 0
    normal_count: int = 0
    bad_review_count: int = 0
    rating_sum: float = 0.0
    # Effective rating evidence weight. It is fractional after daily decay.
    rating_count: float = 0.0
    # Legacy DB/dashboard compatibility for historical run snapshots.
    promised_logistics_hours: Optional[int] = None


@dataclass
class Agent:
    agent_id: str
    name: str
    created_at: str = ""
    is_alive: bool = True
    died_at_t: Optional[int] = None


@dataclass
class EventLog:
    t: int
    event_type: str
    entity_id: str
    payload: dict
    agent_id: str = ""  # empty means global event (supplier/order lifecycle)
