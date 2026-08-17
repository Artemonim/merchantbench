"""Environment.step orchestrator.

Per-agent demand -> supplier/order transition -> commit -> hook -> snapshot.

Designed to be called from web.runner which manages the hook blocking.
The Environment owns one supplier pool (global) plus N independent agent shop pools.
Each agent has its own Cash, its own StoreListings, its own order stream.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from core import demand as demand_mod
from core import listing_rating as lr_mod
from core import order_manager as om
from core import product_manager as pm
from core import public_reviews as public_reviews_mod
from core import rating as rating_mod
from core import sim_time
from core import supplier_scheduler
from core.economy_v6 import EconomyV6
from core.inventory import consume_quantity, effective_quantity
from core.entities import Cash, EventLog, Order, OrderStatusRow, Product, StoreListing
from storage import db as dbm
from storage import snapshot as snap


# Event types that feed the Beta-Binomial good/bad counters. Canonical
# definitions live in core.rating so the rehydrate path can replay them
# off the events table without going through the simulator.
_RATING_GOOD_EVENTS = rating_mod.RATING_GOOD_EVENT_TYPES
_RATING_BAD_EVENTS = rating_mod.RATING_BAD_EVENT_TYPES
@dataclass
class StepResult:
    t: int
    new_orders: int
    state_transitions: int
    events: int


@dataclass
class AgentState:
    agent_id: str
    name: str
    cash: Cash
    listings: dict[str, StoreListing] = field(default_factory=dict)
    is_alive: bool = True
    died_at_t: Optional[int] = None
    # Beta-Binomial counters for shop rating. Floats so the per-step decay
    # can produce fractional values. Rating itself is derived from these +
    # the scenario's prior (see core.rating.posterior_mean).
    n_good: float = 0.0
    n_bad: float = 0.0
    # * Order-outcome shop evidence is published at completed-day boundaries.
    shop_rating_sum: float = 0.0
    shop_rating_weight: float = 0.0
    # * Raw lifetime qualified-transaction evidence never decays.
    shop_rating_order_count: int = 0
    shop_rating_published_t: int = 0
    # * Public reviews are deterministic buyer-visible evidence in v4.
    public_review_sum: float = 0.0
    public_review_count: int = 0
    public_review_eligible_sum: float = 0.0
    public_review_eligible_count: int = 0


class Environment:
    """One Environment instance per run. Holds in-memory copies of mutable state
    (products + per-agent listings + per-agent cash). Orders live in SQLite."""

    def __init__(
        self,
        run_id: str,
        conn,
        scenario: dict,
        runs_root: str,
        products: list[Product],
        hourly_dist: dict,
        agents: dict[str, AgentState],
    ):
        self.run_id = run_id
        self.conn = conn
        self.scenario = scenario
        self.economy_v6 = EconomyV6.from_scenario(scenario)
        self.runs_root = runs_root
        self.products: dict[str, Product] = {p.product_id: p for p in products}
        # Supplier-level trust signals (shop_rating / return_buyer_rate /
        # supplier_age_years) are denormalised onto every Product row but must
        # agree for every product sharing the same supplier_id. Fail fast on
        # any drift so generator bugs and scenario-injected products surface
        # at construction rather than as silent inconsistencies in the UI.
        _by_sup: dict[str, Product] = {}
        for p in products:
            first = _by_sup.setdefault(p.supplier_id, p)
            if (first.shop_rating != p.shop_rating
                    or first.return_buyer_rate != p.return_buyer_rate
                    or first.supplier_age_years != p.supplier_age_years):
                raise ValueError(
                    f"supplier {p.supplier_id} trust-signal drift: "
                    f"{first.product_id} has "
                    f"(shop_rating={first.shop_rating}, "
                    f"return_buyer_rate={first.return_buyer_rate}, "
                    f"supplier_age_years={first.supplier_age_years}) but "
                    f"{p.product_id} has "
                    f"(shop_rating={p.shop_rating}, "
                    f"return_buyer_rate={p.return_buyer_rate}, "
                    f"supplier_age_years={p.supplier_age_years})"
                )
        self.hourly_dist = hourly_dist
        self.agents: dict[str, AgentState] = agents
        self.t: int = 0
        self.lock = threading.RLock()
        self.step_lock = threading.Lock()
        # hook event - set when agent calls end_of_step or when timer fires
        self.hook_event = threading.Event()
        # hook_cond - notifies observation long-poll waiters when the hook
        # opens. Wrap hook_open writes in `with self.hook_cond:` and call
        # notify_all() so waiters wake up.
        self.hook_cond = threading.Condition()
        self.hook_open: bool = False
        # Set when the run finishes (env.t >= horizon). notify_all on
        # hook_cond too, so long-poll waiters can return 410.
        self.finished: bool = False
        self.drain_started_t: Optional[int] = None
        from storage import agent_log as _agent_log
        # Per-step protocol log. Drained at phase 7.
        self.observation_packet: Optional[dict] = None
        self.last_observation_step_by_agent: dict[str, int] = (
            _agent_log.load_observation_state(runs_root, run_id)
        )
        self.observation_cache_by_agent_step: dict[tuple[str, int], dict] = {}
        self.observation_window_by_agent_step: dict[
            tuple[str, int], Optional[tuple[int, int]]
        ] = _agent_log.load_observation_windows(runs_root, run_id)
        self.daily_report_read_date_by_agent: dict[str, str] = (
            _agent_log.load_daily_report_read_dates(runs_root, run_id)
        )
        self.messages_buffer: list[dict] = []
        self.messages_buffer_agents: list[Optional[str]] = []
        self.turns_meta: list[dict] = []
        self.observation_packet_by_agent: dict[str, dict] = {}
        self.messages_buffer_by_agent: dict[str, list[dict]] = {}
        self.turns_meta_by_agent: dict[str, list[dict]] = {}
        self.brief_served_by_agent: set[str] = set()
        self.active_hook_agents: set[str] = set()
        self.hook_done_agents: set[str] = set()
        self.turn_lock = threading.Lock()
        self.last_hook_open_wall_ms: int = 0
        self.last_hook_close_wall_ms: int = 0
        self.last_turn_wall_ms: int = 0
        # idempotency cache (per-run LRU); persisted to runs/<rid>/agent/idem_cache.json
        cap = int((scenario.get("agent", {}) or {}).get("idem_cache_cap", 256))
        self.idem_cache = _agent_log.load_idem(runs_root, run_id, cap=cap)
        # Dirty flag: a failed immediate persist leaves this set so a retry or
        # the hook boundary can durably flush the successful result.
        self._idem_dirty: bool = False
        # Hot search index: lazily built on first hot_search_terms call.
        # Built from product titles at init time; demand/listing state is
        # read live at query time so delisted products are filtered correctly.
        self._hot_search_index = None
        # Turn observers: called after each /act. Used by RunWorker for SSE.
        self.turn_listeners: list[Callable[[dict], None]] = []
        self._dirty_product_ids: set[str] = set()
        self._product_metric_contrib: dict[str, tuple[int, float, int]] = {}
        self._supplier_metric_state = {
            "product_avail_count": 0,
            "total_supplier_price": 0.0,
            "total_supplier_qty": 0,
        }
        self._recompute_supplier_metrics()

    # ----- agent management -----

    def add_agent(self, agent_id: str, name: str, initial_cash: float,
                  initial_deposit: Optional[float] = None) -> AgentState:
        if agent_id in self.agents:
            return self.agents[agent_id]
        if initial_deposit is None:
            initial_deposit = float(self.scenario["run"].get("initial_deposit", 1000.0))
        cash = Cash(balance=float(initial_cash), deposit_pool=float(initial_deposit))
        st = AgentState(agent_id=agent_id, name=name, cash=cash)
        self.agents[agent_id] = st
        dbm.write_cash_log(self.conn, self.run_id, agent_id, -1, st.cash)
        return st

    def _ensure_agent_protocol_state(self, agent_id: str) -> None:
        self.messages_buffer_by_agent.setdefault(agent_id, [])
        self.turns_meta_by_agent.setdefault(agent_id, [])

    def _live_agent_ids(self) -> set[str]:
        return {aid for aid, st in self.agents.items() if st.is_alive}

    def mark_agent_step_done(self, agent_id: str) -> bool:
        """Record one agent's EOS and return True only when the hook can close."""
        live_agents = self._live_agent_ids()
        if not self.active_hook_agents:
            self.active_hook_agents = set(live_agents)
        else:
            self.active_hook_agents &= live_agents
        if agent_id in self.active_hook_agents:
            self.hook_done_agents.add(agent_id)
        return bool(self.active_hook_agents) and self.active_hook_agents <= self.hook_done_agents

    def _combined_agent_messages_and_turns(self) -> tuple[list[dict], list[dict], list[Optional[str]]]:
        messages: list[dict] = list(self.messages_buffer)
        message_agents: list[Optional[str]] = list(self.messages_buffer_agents)
        turns: list[dict] = list(self.turns_meta)
        if not messages and self.messages_buffer:
            messages = list(self.messages_buffer)
            message_agents = list(self.messages_buffer_agents)
        if not turns and self.turns_meta:
            turns = list(self.turns_meta)
        if not turns:
            for aid in sorted(self.agents):
                for turn in self.turns_meta_by_agent.get(aid, []):
                    turns.append({**turn, "agent_id": aid})
        if len(message_agents) != len(messages):
            message_agents = [None] * len(messages)
        return messages, turns, message_agents

    def _record_agent_message(self, agent_id: str, message: dict) -> dict:
        clean = {k: v for k, v in message.items() if k != "agent_id"}
        self.messages_buffer_by_agent.setdefault(agent_id, []).append(clean)
        self.messages_buffer.append(clean)
        self.messages_buffer_agents.append(agent_id)
        return clean

    def _check_death_for(self, agent_id: str, t: int) -> Optional[EventLog]:
        """Permanently close an agent whose guarantee has been exhausted."""
        st = self.agents.get(agent_id)
        if st is None or not st.is_alive:
            return None
        guarantee_exhausted = bool(getattr(st.cash, "_guarantee_exhausted", False))
        should_close = st.cash.deposit_pool <= 0.0 or guarantee_exhausted
        if should_close:
            st.is_alive = False
            st.died_at_t = t
            dbm.mark_agent_dead(self.conn, self.run_id, agent_id, t)
            return EventLog(t=t, event_type="agent_died",
                            entity_id=agent_id, agent_id=agent_id,
                            payload={"deposit_pool": st.cash.deposit_pool,
                                     "balance": st.cash.balance,
                                     "cumulative_fine": st.cash.cumulative_fine})
        return None

    def get_cash(self, agent_id: str) -> Optional[Cash]:
        st = self.agents.get(agent_id)
        return st.cash if st else None

    def reload_listings(self, agent_id: str) -> None:
        rows = dbm.list_listings(self.conn, self.run_id, agent_id)
        self.agents[agent_id].listings = {l.product_id: l for l in rows}

    def reload_all_listings(self) -> None:
        for aid in self.agents:
            self.reload_listings(aid)

    # ----- step -----

    def step(self, hook_blocker: Optional[Callable[[], None]] = None,
             drain: bool = False) -> StepResult:
        with self.step_lock:
            try:
                return self._step_impl(hook_blocker=hook_blocker, drain=drain)
            except Exception:
                # _step_impl uses explicit environment/finalization
                # transactions; if anything threw mid-transaction, clear it so
                # the next step can BEGIN cleanly on this shared connection.
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def _step_impl(self, hook_blocker: Optional[Callable[[], None]] = None,
                   drain: bool = False) -> StepResult:
        run_row = dbm.get_run(self.conn, self.run_id) or {}
        pending_hook_t = run_row.get("pending_hook_t")
        if pending_hook_t is not None:
            if int(pending_hook_t) != int(self.t):
                raise RuntimeError(
                    "durable step boundary mismatch: "
                    f"current_t={self.t}, pending_hook_t={pending_hook_t}"
                )
            return self._resume_committed_step(
                hook_blocker=hook_blocker,
                drain=drain,
                hook_already_closed=bool(run_row.get("pending_hook_closed")),
            )

        with self.lock:
            run_cfg = self.scenario["run"]
            step_hours = int(run_cfg["step_hours"])
            small_share = float(self.scenario["data"]["small_share"])
            master_seed = int(run_cfg["master_seed"])
            settlement_cfg = self.scenario["settlement"]
            platform_rules = self.scenario["platform_rules"]
            sup_cfg = self.scenario["supplier_ranges"]

            events: list[EventLog] = []
            self._dirty_product_ids = set()
            candidate_orders: list[Order] = []
            new_orders: list[Order] = []
            self._step_revenue = {aid: 0.0 for aid in self.agents}
            self._step_cost = {aid: 0.0 for aid in self.agents}

            # Complete and commit the environment transition for t before the
            # agent observes it.  The hook below is therefore a clean
            # state/action boundary: observations include all effects through
            # t, while agent mutations affect demand from t + 1 onward.
            self.conn.execute("BEGIN")

            # 1) demand → new orders (per agent). Drain steps intentionally
            # skip demand so the operating horizon closes the order book while
            # existing orders continue through the lifecycle.
            if not drain:
                self.reload_all_listings()
                triples: list[tuple[Product, StoreListing, str]] = []
                for st in self.agents.values():
                    for l in st.listings.values():
                        p = self.products.get(l.product_id)
                        if p is not None:
                            triples.append((p, l, st.agent_id))
                rating_factors = self._compute_rating_factors()
                lifecycle_cfg = self.scenario.get("lifecycle")
                candidate_orders = demand_mod.generate_orders_for_step(
                    triples, self.hourly_dist, self.t, step_hours,
                    small_share, master_seed,
                    rating_factors=rating_factors,
                    day_offset=sim_time.demand_day_offset(self.scenario),
                    normal_delay_hours=int(settlement_cfg["normal_delay_hours"]),
                    lifecycle_cfg=lifecycle_cfg,
                )
                new_orders = self._auto_purchase_new_orders(
                    candidate_orders, platform_rules, events,
                )
                if new_orders:
                    dbm.insert_orders(self.conn, self.run_id, new_orders)
                    for o in new_orders:
                        product = self.products.get(o.product_id)
                        events.append(EventLog(t=self.t, event_type="order_created",
                                               entity_id=o.order_id, agent_id=o.agent_id,
                                               payload={"product_id": o.product_id,
                                                        "sale_price": o.sale_price,
                                                        "purchase_price": o.purchase_price,
                                                        "supplier_ship_hours": o.supplier_ship_hours or None,
                                                        "supplier_logistics_hours": product.logistics_hours if product else None,
                                                        "actual_logistics_hours": None}))
                # Violations during auto-purchase can drain deposit and kill an agent
                for aid in {o.agent_id for o in candidate_orders}:
                    death_evt = self._check_death_for(aid, self.t)
                    if death_evt is not None:
                        events.append(death_evt)

            # 2) product manager (global supplier pool)
            due_rows = dbm.load_supplier_events_due(self.conn, self.run_id, self.t)
            if due_rows:
                dbm.delete_supplier_events_due(self.conn, self.run_id, self.t)
            due_events = [supplier_scheduler.SupplierEvent.from_row(r) for r in due_rows]
            prod_events, product_dirty, followups, cancel_pending = supplier_scheduler.apply_due_events(
                self.products, due_events, self.t, master_seed, sup_cfg,
                horizon=int(run_cfg["horizon_steps"]),
            )
            for product_id, event_type in cancel_pending:
                dbm.delete_pending_supplier_events(self.conn, self.run_id, product_id, [event_type])
            if followups:
                dbm.insert_supplier_events(self.conn, self.run_id, [e.to_row() for e in followups])
            self._dirty_product_ids.update(product_dirty)
            dirty_products = [
                self.products[pid] for pid in self._dirty_product_ids
                if pid in self.products
            ]
            dbm.upsert_product_states(self.conn, self.run_id, dirty_products)
            self._update_supplier_metrics_for_dirty(self._dirty_product_ids)
            events.extend(prod_events)

            # 3) order manager — pull active orders, route by agent_id
            self.reload_all_listings()
            normal_delay_steps = max(1, int(settlement_cfg["normal_delay_hours"] / step_hours))
            default_promised = int(platform_rules.get("default_promised_ship_hours", 48))
            active = dbm.load_due_orders(self.conn, self.run_id, self.t, normal_delay_steps, default_promised)
            cash_by_agent = {aid: st.cash for aid, st in self.agents.items()}
            listings_by_key: dict[tuple[str, str], StoreListing] = {}
            for st in self.agents.values():
                for pid, l in st.listings.items():
                    listings_by_key[(st.agent_id, pid)] = l
            om_events, mutated, new_status, daily_delta = om.step_orders(
                active, self.products, listings_by_key, cash_by_agent, self.t,
                step_hours, settlement_cfg, platform_rules,
                initial_deposit=float(run_cfg.get("initial_deposit", 1000.0)),
                sup_cfg=sup_cfg, master_seed=master_seed,
                economy=self.economy_v6,
            )
            for o in mutated:
                dbm.update_order_state(self.conn, self.run_id, o)
            for o in mutated:
                for row in o.status_log:
                    if row.t == self.t:
                        dbm.insert_status_row(self.conn, self.run_id, o.order_id, row)
            # persist any per-listing accumulator changes (cum_sales/cum_revenue updated in auto-purchase)
            for st in self.agents.values():
                for l in st.listings.values():
                    dbm.upsert_listing(self.conn, self.run_id, st.agent_id, l)
            events.extend(om_events)
            # check death for any agent whose deposit was exhausted this step
            mutated_agents = {o.agent_id for o in mutated}
            for aid in mutated_agents:
                death_evt = self._check_death_for(aid, self.t)
                if death_evt is not None:
                    events.append(death_evt)
            # GMV accrues at procurement (auto-purchase), not at settlement.
            step_day = int((self.t * step_hours) // 24)
            step_gmv = sum(self._step_revenue.values())
            if step_gmv > 0:
                d = daily_delta.setdefault(step_day, {"gmv": 0.0, "anomaly_count": 0, "fine_total": 0.0})
                d["gmv"] += step_gmv
            for day, delta in daily_delta.items():
                dbm.upsert_daily_aggregate(self.conn, self.run_id, day,
                                           delta["gmv"], delta["anomaly_count"],
                                           delta["fine_total"])

            # * Legacy scenarios keep their event-level hourly rating update.
            # * Order-outcome models publish product and shop evidence daily.
            ratings_published = False
            if self._uses_order_outcome_rating():
                cutoff_t = self._completed_day_cutoff_after_step()
                if cutoff_t is not None:
                    ratings_published = self._publish_daily_ratings(cutoff_t)
            else:
                self._update_ratings(events)
                self._update_listing_ratings([*new_orders, *mutated])

            # 4) persist per-agent cash + events
            for aid, st in self.agents.items():
                dbm.write_cash_log(self.conn, self.run_id, aid, self.t, st.cash)
            dbm.write_events(self.conn, self.run_id, events)

            # 5) write metrics (global + per-agent)
            self._maybe_recompute_supplier_metrics_for_checkpoint()
            self._write_metrics(
                events,
                new_orders,
                mutated,
                write_rating_metrics=(
                    not self._uses_order_outcome_rating()
                    or self.t == 0
                    or ratings_published
                ),
            )

            # The agent must never observe a half-written step.  Commit all
            # orders, events, cash and metrics before opening the hook.  The
            # durable marker lets a stopped/restarted worker resume this hook
            # without replaying demand, procurement, or supplier events.
            dbm.mark_step_transition_committed(self.conn, self.run_id, self.t)
            self.conn.execute("COMMIT")

        return self._run_hook_and_finalize(
            hook_blocker=hook_blocker,
            drain=drain,
            run_cfg=run_cfg,
            events=events,
            new_orders=new_orders,
            mutated=mutated,
            dirty_products=dirty_products,
            hook_already_closed=False,
            step_t=self.t,
        )

    def _resume_committed_step(
        self,
        *,
        hook_blocker: Optional[Callable[[], None]],
        drain: bool,
        hook_already_closed: bool,
    ) -> StepResult:
        """Resume a hook whose environment transition was already committed."""
        with self.lock:
            event_rows = [
                row
                for row in dbm.load_events_at(self.conn, self.run_id, self.t)
                if row["event_type"] not in {
                    "agent_list_product",
                    "agent_delist_product",
                    "agent_adjust_price",
                }
            ]
            events = [
                EventLog(
                    t=int(row["t"]),
                    event_type=str(row["event_type"]),
                    entity_id=str(row["entity_id"]),
                    agent_id=row.get("agent_id") or "",
                    payload=dict(row.get("payload") or {}),
                )
                for row in event_rows
            ]

            new_order_ids = [
                str(row["order_id"])
                for row in self.conn.execute(
                    "SELECT order_id FROM orders WHERE run_id=? AND order_t=?",
                    (self.run_id, self.t),
                ).fetchall()
            ]
            mutated_order_ids = [
                str(row["order_id"])
                for row in self.conn.execute(
                    "SELECT DISTINCT o.order_id FROM order_status s"
                    " JOIN orders o ON o.run_id=s.run_id AND o.order_id=s.order_id"
                    " WHERE s.run_id=? AND s.t=? AND o.order_t<>?",
                    (self.run_id, self.t, self.t),
                ).fetchall()
            ]

            def load_orders(order_ids: list[str]) -> list[Order]:
                loaded: list[Order] = []
                for order_id in order_ids:
                    order = dbm.load_order(self.conn, self.run_id, order_id)
                    if order is None:
                        continue
                    order.status_log = dbm.load_status_log(
                        self.conn, self.run_id, order_id
                    )
                    loaded.append(order)
                return loaded

            new_orders = load_orders(new_order_ids)
            mutated = load_orders(mutated_order_ids)
            dirty_ids = {order.product_id for order in new_orders}
            for event in events:
                if event.entity_id in self.products:
                    dirty_ids.add(event.entity_id)
                product_id = event.payload.get("product_id")
                if product_id in self.products:
                    dirty_ids.add(str(product_id))
            dirty_products = [
                self.products[product_id]
                for product_id in dirty_ids
                if product_id in self.products
            ]

        return self._run_hook_and_finalize(
            hook_blocker=hook_blocker,
            drain=drain,
            run_cfg=self.scenario["run"],
            events=events,
            new_orders=new_orders,
            mutated=mutated,
            dirty_products=dirty_products,
            hook_already_closed=hook_already_closed,
            step_t=self.t,
        )

    def _run_hook_and_finalize(
        self,
        *,
        hook_blocker: Optional[Callable[[], None]],
        drain: bool,
        run_cfg: dict,
        events: list[EventLog],
        new_orders: list[Order],
        mutated: list[Order],
        dirty_products: list[Product],
        hook_already_closed: bool,
        step_t: int,
    ) -> StepResult:
        # 6) hook — the agent observes the complete state at t and chooses
        # actions for subsequent demand.  When activation_period > 1, off-ticks
        # still execute the full environment transition but do not open a hook.
        drain = bool(drain or step_t >= int(run_cfg["horizon_steps"]))
        period = int((self.scenario.get("agent") or {}).get("activation_period") or 1)
        if period < 1:
            period = 1
        agent_active = (not drain and step_t % period == 0)

        from storage import agent_log
        now_ms = agent_log.now_ms
        persisted_hook_trace: Optional[dict] = None
        if hook_already_closed:
            # A previous process completed the hook and failed only during
            # snapshot/trace finalization. Do not invite a second action at t.
            persisted_hook_trace = agent_log.read_step_index(
                self.runs_root, self.run_id, step_t
            )
            recovered_wall_ms = now_ms()
            if self.last_hook_open_wall_ms <= 0:
                self.last_hook_open_wall_ms = int(
                    (persisted_hook_trace or {}).get("hook_open_wall_ms")
                    or recovered_wall_ms
                )
            if self.last_turn_wall_ms <= 0:
                self.last_turn_wall_ms = self.last_hook_open_wall_ms
            if self.last_hook_close_wall_ms <= 0:
                self.last_hook_close_wall_ms = int(
                    (persisted_hook_trace or {}).get("hook_close_wall_ms")
                    or self.last_hook_open_wall_ms
                )
        else:
            with self.turn_lock:
                self.active_hook_agents = self._live_agent_ids() if agent_active else set()
                self.hook_done_agents = set()
            self.last_hook_open_wall_ms = now_ms()
            self.last_turn_wall_ms = self.last_hook_open_wall_ms
            if agent_active and self.active_hook_agents:
                with self.lock:
                    self.hook_event.clear()
                    with self.hook_cond:
                        self.hook_open = True
                        self.hook_cond.notify_all()  # wake observation long-polls
                try:
                    if hook_blocker is not None:
                        hook_blocker()
                finally:
                    with self.lock:
                        with self.hook_cond:
                            self.hook_open = False
                            self.hook_cond.notify_all()  # wake waiters for 408/410
                    self.last_hook_close_wall_ms = now_ms()
            else:
                self.last_hook_close_wall_ms = self.last_hook_open_wall_ms

            # Persist the complete hook artifacts before closing its durable
            # boundary. A restart after the marker is set can then finish the
            # snapshot and cost aggregation without losing the agent turn.
            with self.turn_lock:
                boundary_messages, boundary_turns, boundary_agents = (
                    self._combined_agent_messages_and_turns()
                )
                boundary_has_observation = (
                    bool(self.observation_packet_by_agent)
                    or self.observation_packet is not None
                )
            if self._idem_dirty:
                agent_log.persist_idem(
                    self.runs_root, self.run_id, self.idem_cache
                )
                self._idem_dirty = False
            if boundary_has_observation or boundary_messages:
                agent_log.write_step_index(
                    self.runs_root,
                    self.run_id,
                    step_t,
                    boundary_messages,
                    boundary_turns,
                    message_agents=boundary_agents,
                    hook_open_wall_ms=self.last_hook_open_wall_ms,
                    hook_close_wall_ms=self.last_hook_close_wall_ms,
                )
            dbm.mark_step_hook_closed(self.conn, self.run_id, step_t)

        with self.lock:
            # 7) Agent tools persist mutations during the hook. Reload listings
            # so the end-of-step snapshot records the state entering t + 1,
            # then atomically finalize the run clock and trace metadata.
            self.conn.execute("BEGIN")
            self.reload_all_listings()

            # Snapshot (single global file containing all agents).
            survival = self._survival_state()
            snapshot_orders = list(new_orders)
            seen_oids = {o.order_id for o in snapshot_orders}
            snapshot_orders.extend(o for o in mutated if o.order_id not in seen_oids)
            snap.write_env_delta_snapshot(
                self.runs_root, self.run_id, step_t,
                dirty_products=dirty_products,
                agents=list(self.agents.values()),
                mutated_orders=snapshot_orders,
                events_this_step=events,
                survival_state=survival,
                current_t=step_t,
            )
            interval = int(run_cfg.get("checkpoint_interval_steps", 168) or 0)
            if interval > 0 and (step_t % interval == 0 or (
                not drain and step_t + 1 >= int(run_cfg["horizon_steps"])
            )):
                snap.write_env_checkpoint(
                    self.runs_root, self.run_id, step_t, list(self.products.values()),
                    current_t=step_t,
                )
            # Finalize agent trace for this step.
            agent_cfg = self.scenario.get("agent", {}) or {}
            with self.turn_lock:
                messages, turns_meta, message_agents = self._combined_agent_messages_and_turns()
                has_obs = bool(self.observation_packet_by_agent) or self.observation_packet is not None
                self.observation_packet = None
                self.messages_buffer = []
                self.messages_buffer_agents = []
                self.turns_meta = []
                self.observation_packet_by_agent = {}
                self.messages_buffer_by_agent = {}
                self.turns_meta_by_agent = {}
                self.active_hook_agents = set()
                self.hook_done_agents = set()
            if hook_already_closed and persisted_hook_trace is not None:
                messages = list(persisted_hook_trace.get("messages") or [])
                turns_meta = list(persisted_hook_trace.get("turns") or [])
                message_agents = list(
                    persisted_hook_trace.get("message_agents")
                    or [None] * len(messages)
                )
            env_step_ms = max(0, self.last_hook_close_wall_ms - self.last_hook_open_wall_ms)
            has_activity = bool(has_obs or messages)
            if has_activity:
                agent_log.write_step_index(
                    self.runs_root, self.run_id, step_t,
                    messages, turns_meta,
                    message_agents=message_agents,
                    hook_open_wall_ms=self.last_hook_open_wall_ms,
                    hook_close_wall_ms=self.last_hook_close_wall_ms,
                )
                agent_log.update_cost(
                    self.runs_root, self.run_id, step_t, turns_meta,
                    pricing=agent_cfg.get("cost_pricing"),
                    env_step_ms=env_step_ms,
                )
            if self._idem_dirty:
                agent_log.persist_idem(self.runs_root, self.run_id, self.idem_cache)
                self._idem_dirty = False
            dbm.finish_committed_step(self.conn, self.run_id, step_t + 1)
            self.conn.execute("COMMIT")

            self.t = step_t + 1

            return StepResult(t=step_t, new_orders=len(new_orders),
                              state_transitions=len(mutated), events=len(events))

    # ----- auto-purchase -----

    def _auto_purchase_new_orders(self, candidates: list[Order],
                                  platform_rules: dict,
                                  events: list[EventLog]) -> list[Order]:
        """For each customer-side order: try to procure from supplier at the
        supplier's current `price`. Outcomes:
          * fulfillable     → debit purchase_price, dec supplier qty, bump
                              listing accumulators, enter state machine at `ordered`
          * stockout        → supplier delisted or qty==0: emit violation event
                              + apply the stockout penalty
          * insufficient $  → agent balance < required cash (purchase_price,
                              plus fulfillment fee when v6 fulfillment is on):
                              emit violation event + apply the insufficient-balance
                              penalty
        Failed orders are dropped (not persisted into DB).
        """
        from core.order_manager import _apply_penalty, _penalty_amount
        kept: list[Order] = []

        def _listing_ship_promise(listing: StoreListing, product: Product) -> int:
            # Use product's supplier_ship_hours directly (no per-listing promise)
            return int(product.supplier_ship_hours) if product.supplier_ship_hours else 0

        for o in candidates:
            product = self.products.get(o.product_id)
            if product is None:
                continue
            st = self.agents.get(o.agent_id)
            if st is None or not st.is_alive:
                continue
            listing = st.listings.get(o.product_id)
            if listing is None:
                continue
            supplier_ship = _listing_ship_promise(listing, product)
            o.supplier_ship_hours = supplier_ship
            # supplier-side stockout (delisted or empty inventory) → platform violation.
            # The order never gets procured but is still persisted in the orders table
            # so the merchant's per-order P&L surfaces the penalty (net_profit = -penalty).
            available_qty = effective_quantity(product, self.t)
            if (not product.is_listed_by_supplier) or available_qty <= 0:
                penalty = _penalty_amount(platform_rules, "stockout", o.sale_price)
                _apply_penalty(st.cash, penalty)
                reason = "supplier_delisted" if not product.is_listed_by_supplier else "out_of_stock"
                events.append(EventLog(t=self.t, event_type="order_stockout_violation",
                                       entity_id=o.product_id, agent_id=o.agent_id,
                                       payload={"order_id": o.order_id,
                                                "reason": reason,
                                                "penalty": penalty,
                                                "sale_price": o.sale_price,
                                                "supplier_ship_hours": product.supplier_ship_hours,
                                                "supplier_logistics_hours": product.logistics_hours,
                                                "actual_logistics_hours": None}))
                o.total_penalty = penalty
                o.current_status = "stockout"
                o.purchase_t = self.t
                o.settled_t = self.t
                o.status_log.append(OrderStatusRow(t=self.t, status="stockout"))
                kept.append(o)
                death_evt = self._check_death_for(o.agent_id, self.t)
                if death_evt is not None:
                    events.append(death_evt)
                continue
            # merchant-side: not enough cash to fund the procurement → platform violation.
            # Same persistence treatment as stockout — the order row records the penalty.
            # Deposit cannot pay procurement. Fulfillment fee is required cash when enabled.
            fulfillment_fee = (
                round(self.economy_v6.fulfillment_fee(product.category), 2)
                if self.economy_v6.fulfillment_enabled
                else 0.0
            )
            required_cash = o.purchase_price + fulfillment_fee
            if st.cash.balance < required_cash:
                penalty = _penalty_amount(platform_rules, "insufficient_balance", o.sale_price)
                _apply_penalty(st.cash, penalty)
                events.append(EventLog(t=self.t, event_type="order_insufficient_balance_violation",
                                       entity_id=o.product_id, agent_id=o.agent_id,
                                       payload={"order_id": o.order_id,
                                                "purchase_price": o.purchase_price,
                                                "balance": st.cash.balance,
                                                "penalty": penalty,
                                                "sale_price": o.sale_price,
                                                "supplier_ship_hours": product.supplier_ship_hours,
                                                "supplier_logistics_hours": product.logistics_hours,
                                                "actual_logistics_hours": None}))
                o.total_penalty = penalty
                o.current_status = "insufficient_balance"
                o.purchase_t = self.t
                o.settled_t = self.t
                o.status_log.append(OrderStatusRow(t=self.t, status="insufficient_balance"))
                kept.append(o)
                death_evt = self._check_death_for(o.agent_id, self.t)
                if death_evt is not None:
                    events.append(death_evt)
                continue
            if not consume_quantity(product, self.t, 1):
                penalty = _penalty_amount(platform_rules, "stockout", o.sale_price)
                _apply_penalty(st.cash, penalty)
                events.append(EventLog(t=self.t, event_type="order_stockout_violation",
                                       entity_id=o.product_id, agent_id=o.agent_id,
                                       payload={"order_id": o.order_id,
                                                "reason": "out_of_stock",
                                                "penalty": penalty,
                                                "sale_price": o.sale_price,
                                                "supplier_ship_hours": product.supplier_ship_hours,
                                                "supplier_logistics_hours": product.logistics_hours,
                                                "actual_logistics_hours": None}))
                o.total_penalty = penalty
                o.current_status = "stockout"
                o.purchase_t = self.t
                o.settled_t = self.t
                o.status_log.append(OrderStatusRow(t=self.t, status="stockout"))
                kept.append(o)
                death_evt = self._check_death_for(o.agent_id, self.t)
                if death_evt is not None:
                    events.append(death_evt)
                continue

            self._dirty_product_ids.add(product.product_id)
            st.cash.balance -= o.purchase_price + fulfillment_fee
            st.cash.in_transit += o.purchase_price
            o.logistics_fee = fulfillment_fee
            listing.cum_sales += 1
            listing.cum_revenue += o.sale_price
            # Persist the listing's new accumulators immediately. The order-
            # manager reload_all_listings() later in this environment transition
            # would otherwise revert the in-memory increment.
            dbm.upsert_listing(self.conn, self.run_id, st.agent_id, listing)
            o.realized_cost = o.purchase_price
            o.current_status = "ordered"
            o.purchase_t = self.t
            o.supplier_ship_hours = int(product.supplier_ship_hours)
            o.actual_ship_hours = o.supplier_ship_hours
            o.status_log.append(OrderStatusRow(t=self.t, status="ordered"))
            self._step_revenue[o.agent_id] = self._step_revenue.get(o.agent_id, 0.0) + o.sale_price
            self._step_cost[o.agent_id] = self._step_cost.get(o.agent_id, 0.0) + o.purchase_price
            kept.append(o)
        return kept

    # ----- shop rating -----

    def _rating_cfg(self) -> Optional[dict]:
        """Return the shop_rating block if enabled, else None.

        None means rating is disabled entirely — no multiplier applied to demand,
        no metrics written, no observation field emitted.
        """
        cfg = self.scenario.get("shop_rating") or {}
        if not cfg.get("enabled", False):
            return None
        return cfg

    def _rating_model(self) -> str:
        cfg = self._rating_cfg()
        if cfg is None:
            return "disabled"
        return str(cfg.get("model") or "beta_event_v1")

    def _uses_order_outcome_rating(self) -> bool:
        return self._rating_model() in lr_mod.ORDER_OUTCOME_RATING_MODELS

    def _uses_reputation_volume(self) -> bool:
        return self._rating_model() == lr_mod.REPUTATION_VOLUME_RATING_MODEL

    def _uses_public_review_demand(self) -> bool:
        return self._rating_model() == lr_mod.PUBLIC_REVIEW_RATING_MODEL

    def _public_reviews_cfg(self) -> Optional[dict]:
        """Return the public-review policy owned by v4 rating semantics."""
        cfg = self.scenario.get("public_reviews") or {}
        if not self._uses_public_review_demand():
            return None
        if not cfg.get("enabled", False):
            raise ValueError(
                "order_outcome_v4 requires public_reviews.enabled=true"
            )
        return cfg

    def _public_reviews_agent_visible(self) -> bool:
        """Return whether public reviews belong to the agent contract."""
        cfg = self._public_reviews_cfg()
        return cfg is not None

    def _rating_outcome_cfg(self) -> tuple[dict, dict]:
        cfg = self.scenario.get("rating_outcomes") or {}
        scores = {
            key: cfg[key]
            for key in lr_mod.DEFAULT_OUTCOME_SCORES
            if key in cfg
        }
        weights = {
            key: cfg[key]
            for key in lr_mod.DEFAULT_OUTCOME_WEIGHTS
            if key in cfg
        }
        return scores, weights

    def _shop_rating_value(self, st: AgentState) -> float:
        cfg = self._rating_cfg()
        if cfg is None:
            return 4.0
        if self._uses_order_outcome_rating():
            default_prior_weight = (
                0.0
                if self._rating_model() in {
                    lr_mod.REPUTATION_VOLUME_RATING_MODEL,
                    lr_mod.PUBLIC_REVIEW_RATING_MODEL,
                }
                else 20.0
            )
            return lr_mod.compute_listing_rating(
                float(cfg.get("initial_rating", 4.0)),
                st.shop_rating_sum,
                st.shop_rating_weight,
                float(cfg.get("prior_weight", default_prior_weight)),
            )
        return rating_mod.posterior_mean(
            st.n_good,
            st.n_bad,
            float(cfg["prior_good"]),
            float(cfg["prior_bad"]),
        )

    def _shop_rating_updated_through_step(self, st: AgentState) -> int:
        """Return the last included step, not the exclusive publish cutoff."""
        cutoff_t = int(st.shop_rating_published_t)
        if cutoff_t <= 0:
            return 0
        return max(0, min(int(self.t), cutoff_t) - 1)

    def _shop_rating_state(self, st: AgentState) -> dict:
        """Return current quality, trust, and combined demand signals."""
        cfg = self._rating_cfg()
        if cfg is None:
            return {}
        service_score = self._shop_rating_value(st)
        service_stars = rating_mod.stars_from_score(
            service_score, cfg["bucket_thresholds"],
        )
        service_quality_multiplier = rating_mod.multiplier_from_stars(
            service_stars, cfg["star_multipliers"],
        )
        score = service_score
        stars = service_stars
        quality_multiplier = service_quality_multiplier
        reputation_multiplier = 1.0
        demand_source = "service_quality"
        if self._uses_reputation_volume():
            reputation_multiplier = lr_mod.reputation_volume_multiplier(
                st.shop_rating_order_count,
                cfg.get("reputation_volume"),
            )
            demand_source = "service_quality_and_transaction_volume"
        elif self._uses_public_review_demand():
            public_reviews = self._public_review_state(st)
            if public_reviews is None:
                raise ValueError(
                    "order_outcome_v4 requires an enabled public review state"
                )
            score = (
                float(public_reviews["rating"])
                if public_reviews["rating"] is not None else None
            )
            stars = (
                int(public_reviews["stars"])
                if public_reviews["stars"] is not None else None
            )
            quality_multiplier = float(
                public_reviews["quality_multiplier"]
            )
            reputation_multiplier = float(
                public_reviews["reputation_multiplier"]
            )
            demand_source = "public_reviews"
        return {
            "score": score,
            "stars": float(stars) if stars is not None else None,
            "quality_multiplier": quality_multiplier,
            "reputation_multiplier": reputation_multiplier,
            "demand_multiplier": quality_multiplier * reputation_multiplier,
            "service_quality_score": service_score,
            "service_quality_stars": float(service_stars),
            "service_quality_multiplier": service_quality_multiplier,
            "rating_available": (
                score is not None
                if self._uses_public_review_demand() else True
            ),
            "demand_source": demand_source,
        }

    def _public_review_state(self, st: AgentState) -> Optional[dict]:
        """Return buyer-visible public reputation and comparison signals."""
        cfg = self._public_reviews_cfg()
        if cfg is None:
            return None
        resolved = public_reviews_mod.resolve_public_review_config(cfg)
        review_count = int(st.public_review_count)
        eligible_count = int(st.public_review_eligible_count)
        rating = (
            float(st.public_review_sum) / review_count
            if review_count > 0 else None
        )
        full_response_rating = (
            float(st.public_review_eligible_sum) / eligible_count
            if eligible_count > 0 else None
        )
        quality_score = self._shop_rating_value(st)
        state = {
            "model": resolved["model"],
            "rating": rating,
            "count": review_count,
            "eligible_count": eligible_count,
            "response_rate": (
                review_count / eligible_count if eligible_count > 0 else 0.0
            ),
            "full_response_rating": full_response_rating,
            "selection_gap": (
                rating - full_response_rating
                if rating is not None and full_response_rating is not None
                else None
            ),
            "quality_gap": (
                rating - quality_score if rating is not None else None
            ),
            "affects_demand": self._uses_public_review_demand(),
        }
        if self._uses_public_review_demand():
            shop_cfg = self._rating_cfg()
            if shop_cfg is None:
                raise ValueError("order_outcome_v4 requires shop_rating")
            state.update(public_reviews_mod.public_review_demand_factors(
                rating,
                review_count,
                bucket_thresholds=list(shop_cfg["bucket_thresholds"]),
                star_multipliers=list(shop_cfg["star_multipliers"]),
                config=cfg,
            ))
        return state

    def _compute_rating_factors(self) -> Optional[dict[str, float]]:
        """Per-agent demand multiplier dict for this step's order generation.

        Returns None when rating is disabled (so demand.generate_orders_for_step
        applies no multiplier). Dead agents still get a factor — their listings
        won't trigger demand anyway because auto_purchase rejects them.
        """
        cfg = self._rating_cfg()
        if cfg is None:
            return None
        out: dict[str, float] = {}
        for aid, st in self.agents.items():
            out[aid] = self._shop_rating_state(st)["demand_multiplier"]
        return out

    def _listing_rating_cfg(self) -> Optional[dict]:
        cfg = self.scenario.get("listing_rating")
        if not cfg:
            return None
        return cfg

    def _listing_rating_metric_values(self, st: AgentState) -> dict[str, float]:
        cfg = self._listing_rating_cfg()
        if cfg is None:
            return {}
        total = len(st.listings)
        prior_weight = float(cfg.get("prior_weight", 10.0))
        initial_rating = float(cfg.get("initial_rating", 4.0))
        rating_total = sum(
            lr_mod.compute_listing_rating(
                initial_rating,
                listing.rating_sum,
                listing.rating_count,
                prior_weight,
            )
            for listing in st.listings.values()
        )
        return {
            "avg_listing_rating": (rating_total / total) if total else 0.0,
            "avg_listing_rating_count": float(total),
        }

    def _shop_rating_metric_values(self, st: AgentState) -> dict[str, float]:
        cfg = self._rating_cfg()
        if cfg is None:
            return {}
        rating_state = self._shop_rating_state(st)
        score = rating_state["score"]
        stars = rating_state["stars"]
        out = {
            "shop_quality_multiplier": rating_state["quality_multiplier"],
            "shop_reputation_multiplier": rating_state["reputation_multiplier"],
            "shop_demand_multiplier": rating_state["demand_multiplier"],
        }
        if score is not None and stars is not None:
            out.update({
                "shop_rating_score": float(score),
                "shop_rating_stars": float(stars),
            })
        if self._uses_order_outcome_rating():
            reputation_evidence_count = st.shop_rating_order_count
            if self._uses_public_review_demand():
                reputation_evidence_count = st.public_review_count
            out.update({
                "shop_rating_order_count": float(st.shop_rating_order_count),
                "shop_qualified_transaction_count": float(
                    st.shop_rating_order_count
                ),
                "shop_reputation_evidence_count": float(
                    reputation_evidence_count
                ),
            })
            if score is not None:
                out["shop_rating_mean"] = float(score)
            if self._uses_public_review_demand():
                out.update({
                    "shop_service_quality_score": float(
                        rating_state["service_quality_score"]
                    ),
                    "shop_service_quality_stars": float(
                        rating_state["service_quality_stars"]
                    ),
                    "shop_service_quality_multiplier": float(
                        rating_state["service_quality_multiplier"]
                    ),
                })
        else:
            out.update({
                "shop_n_good_effective": st.n_good,
                "shop_n_bad_effective": st.n_bad,
            })
        public_reviews = self._public_review_state(st)
        if public_reviews is not None:
            out.update({
                "public_review_count": float(public_reviews["count"]),
                "public_review_eligible_count": float(
                    public_reviews["eligible_count"],
                ),
                "public_review_response_rate": float(
                    public_reviews["response_rate"],
                ),
            })
            optional_metrics = {
                "public_review_rating": public_reviews["rating"],
                "public_review_full_response_rating": (
                    public_reviews["full_response_rating"]
                ),
                "public_review_selection_gap": public_reviews["selection_gap"],
                "public_review_quality_gap": public_reviews["quality_gap"],
                "public_review_confidence": public_reviews.get("confidence"),
                "public_review_raw_quality_multiplier": public_reviews.get(
                    "raw_quality_multiplier"
                ),
                "public_review_quality_multiplier": public_reviews.get(
                    "quality_multiplier"
                ),
                "public_review_reputation_multiplier": public_reviews.get(
                    "reputation_multiplier"
                ),
                "public_review_demand_multiplier": public_reviews.get(
                    "demand_multiplier"
                ),
            }
            out.update({
                key: float(value)
                for key, value in optional_metrics.items()
                if value is not None
            })
        return out

    def _write_current_rating_metrics(self, t: int) -> None:
        for aid, st in self.agents.items():
            kv = self._shop_rating_metric_values(st)
            kv.update(self._listing_rating_metric_values(st))
            dbm.write_metrics(self.conn, self.run_id, aid, int(t), kv)

    def _listing_price_metric_values(
        self, st: AgentState, *, include_rating: bool = True,
    ) -> dict[str, float]:
        total = len(st.listings)
        price_total = sum(float(listing.sale_price) for listing in st.listings.values())
        margins = [
            float(listing.sale_price) - float(self.products[listing.product_id].price)
            for listing in st.listings.values()
            if listing.product_id in self.products
        ]
        margin_ratios = [
            (
                float(listing.sale_price) - float(self.products[listing.product_id].price)
            ) / float(listing.sale_price)
            for listing in st.listings.values()
            if listing.product_id in self.products and float(listing.sale_price) > 0
        ]
        out = {
            "avg_listing_sale_price": (price_total / total) if total else 0.0,
            "avg_listing_sale_price_count": float(total),
            "avg_listing_margin": (sum(margins) / len(margins)) if margins else 0.0,
            "avg_listing_margin_count": float(len(margins)),
            "avg_listing_margin_ratio": (
                sum(margin_ratios) / len(margin_ratios)
            ) if margin_ratios else 0.0,
            "avg_listing_margin_ratio_count": float(len(margin_ratios)),
        }
        if include_rating:
            out.update(self._listing_rating_metric_values(st))
        return out

    def _update_ratings(self, events: list[EventLog]) -> None:
        """Apply gentle decay to every agent's counters, then fold in this
        step's good/bad events. No-op if rating is disabled.
        """
        cfg = self._rating_cfg()
        if cfg is None or self._uses_order_outcome_rating():
            return
        decay = float(cfg["decay"])
        # 1) Decay all agents (so an idle agent's rating still drifts toward
        # the prior over many steps, matching real-platform "fading reputation").
        for st in self.agents.values():
            st.n_good = rating_mod.apply_decay(st.n_good, decay)
            st.n_bad = rating_mod.apply_decay(st.n_bad, decay)
        # 2) Fold in this step's events.
        for e in events:
            aid = e.agent_id
            if not aid or aid not in self.agents:
                continue
            if e.event_type in _RATING_GOOD_EVENTS:
                self.agents[aid].n_good += 1.0
            elif e.event_type in _RATING_BAD_EVENTS:
                self.agents[aid].n_bad += 1.0

    def _update_listing_ratings(self, mutated: list[Order]) -> None:
        """Fold one final-experience score per terminal order into listings."""
        if self._uses_order_outcome_rating():
            return
        cfg = self._listing_rating_cfg()
        if cfg is None:
            return
        scores = {
            key: cfg[key]
            for key in (
                "normal_score",
                "late_score",
                "refund_score",
                "only_refund_score",
                "bad_review_score",
                "stockout_score",
            )
            if key in cfg
        }
        settled_this_step: dict[str, tuple[str, float]] = {}
        for o in mutated:
            if o.settled_t != self.t:
                continue
            score = lr_mod.score_for_order_outcome(o.current_status, o.late_t, scores)
            if score is None:
                continue
            settled_this_step[o.order_id] = (o.current_status, score)
        if not settled_this_step:
            return
        for o in mutated:
            if o.order_id not in settled_this_step:
                continue
            st = self.agents.get(o.agent_id)
            if st is None:
                continue
            listing = st.listings.get(o.product_id)
            if listing is None:
                continue
            status, score = settled_this_step[o.order_id]
            if status == "settled_normal":
                listing.normal_count += 1
            elif status == "settled_bad_review":
                listing.bad_review_count += 1
            listing.rating_sum += score
            listing.rating_count += 1
            dbm.upsert_listing(self.conn, self.run_id, o.agent_id, listing)

    def _completed_day_cutoff_after_step(self) -> Optional[int]:
        step_hours = int(self.scenario["run"]["step_hours"])
        next_t = self.t + 1
        if (next_t * step_hours) % 24 != 0:
            return None
        return next_t

    def _last_completed_day_cutoff(self) -> int:
        step_hours = int(self.scenario["run"]["step_hours"])
        completed_hours = (self.t * step_hours // 24) * 24
        return completed_hours // step_hours

    def _publish_daily_ratings(self, cutoff_t: int) -> bool:
        """Rebuild and publish order-outcome evidence through ``cutoff_t``."""
        if not self._uses_order_outcome_rating():
            return False
        shop_cfg = self._rating_cfg()
        product_cfg = self._listing_rating_cfg()
        if shop_cfg is None or product_cfg is None:
            return False
        scores, weights = self._rating_outcome_cfg()
        public_review_cfg = self._public_reviews_cfg()
        step_hours = int(self.scenario["run"]["step_hours"])
        master_seed = int(self.scenario["run"]["master_seed"])
        for aid, st in self.agents.items():
            feedback_rows = dbm.load_order_feedback_rows(
                self.conn, self.run_id, aid, cutoff_t,
            )
            rows = [
                (product_id, status, late_t, settled_t)
                for _, product_id, status, late_t, settled_t in feedback_rows
            ]
            product_evidence = lr_mod.rebuild_evidence(
                rows,
                cutoff_t=cutoff_t,
                step_hours=step_hours,
                half_life_days=float(product_cfg["half_life_days"]),
                scores=scores,
                weights=weights,
            )
            shop_rows = [
                (aid, status, late_t, settled_t)
                for _, status, late_t, settled_t in rows
            ]
            shop_evidence = lr_mod.rebuild_evidence(
                shop_rows,
                cutoff_t=cutoff_t,
                step_hours=step_hours,
                half_life_days=float(shop_cfg["half_life_days"]),
                scores=scores,
                weights=weights,
            ).get(aid, (0.0, 0.0, 0))
            st.shop_rating_sum = float(shop_evidence[0])
            st.shop_rating_weight = float(shop_evidence[1])
            st.shop_rating_order_count = int(shop_evidence[2])
            st.shop_rating_published_t = int(cutoff_t)
            if public_review_cfg is not None:
                public_review_evidence = (
                    public_reviews_mod.rebuild_public_review_evidence(
                        [
                            (order_id, status, late_t)
                            for order_id, _, status, late_t, _ in feedback_rows
                        ],
                        master_seed=master_seed,
                        agent_id=aid,
                        config=public_review_cfg,
                        scores=scores,
                    )
                )
                st.public_review_sum = float(
                    public_review_evidence.review_score_sum,
                )
                st.public_review_count = int(public_review_evidence.review_count)
                st.public_review_eligible_sum = float(
                    public_review_evidence.eligible_score_sum,
                )
                st.public_review_eligible_count = int(
                    public_review_evidence.eligible_count,
                )
            for pid, listing in st.listings.items():
                evidence = product_evidence.get(pid, (0.0, 0.0, 0))
                listing.rating_sum = float(evidence[0])
                listing.rating_count = float(evidence[1])
                dbm.upsert_listing(self.conn, self.run_id, aid, listing)
        return True

    def publish_terminal_ratings(self) -> bool:
        """Close a final partial virtual day and persist its rating point."""
        if not self._uses_order_outcome_rating() or self.t <= 0:
            return False
        step_hours = int(self.scenario["run"]["step_hours"])
        elapsed_hours = self.t * step_hours
        cutoff_hours = ((elapsed_hours + 23) // 24) * 24
        cutoff_t = cutoff_hours // step_hours
        if all(
            st.shop_rating_published_t >= cutoff_t
            for st in self.agents.values()
        ):
            return False
        if not self._publish_daily_ratings(cutoff_t):
            return False
        self._write_current_rating_metrics(self.t - 1)
        return True

    def restore_rating_state(self, *, include_terminal_partial_day: bool = False) -> None:
        """Restore the configured rating model after a process restart.

        Order-outcome models rebuild product/shop evidence from terminal orders;
        legacy runs rebuild their Beta counters from events. No-op when ratings
        are disabled.
        """
        cfg = self._rating_cfg()
        if cfg is None:
            return
        if self._uses_order_outcome_rating():
            cutoff_t = self._last_completed_day_cutoff()
            if include_terminal_partial_day and self.t > cutoff_t:
                step_hours = int(self.scenario["run"]["step_hours"])
                cutoff_hours = ((self.t * step_hours + 23) // 24) * 24
                cutoff_t = cutoff_hours // step_hours
            if cutoff_t > 0:
                self._publish_daily_ratings(cutoff_t)
            return
        decay = float(cfg["decay"])
        types = list(_RATING_GOOD_EVENTS | _RATING_BAD_EVENTS)
        rating_events = dbm.load_rating_events(self.conn, self.run_id, types)
        counters = rating_mod.rebuild_counters(
            rating_events, self.t, decay,
            _RATING_GOOD_EVENTS, _RATING_BAD_EVENTS,
        )
        for aid, (n_good, n_bad) in counters.items():
            st = self.agents.get(aid)
            if st is None:
                continue
            st.n_good = n_good
            st.n_bad = n_bad

    # ----- metrics aggregation -----

    def _product_metric_contribution(self, p: Product) -> tuple[int, float, int]:
        if not p.is_listed_by_supplier:
            return (0, 0.0, 0)
        return (1, float(p.price), int(effective_quantity(p, self.t)))

    def _recompute_supplier_metrics(self) -> None:
        contrib: dict[str, tuple[int, float, int]] = {}
        product_avail = 0
        total_price = 0.0
        total_qty = 0
        for p in self.products.values():
            c = self._product_metric_contribution(p)
            contrib[p.product_id] = c
            product_avail += c[0]
            total_price += c[1]
            total_qty += c[2]
        self._product_metric_contrib = contrib
        self._supplier_metric_state = {
            "product_avail_count": product_avail,
            "total_supplier_price": total_price,
            "total_supplier_qty": total_qty,
        }

    def _update_supplier_metrics_for_dirty(self, dirty_product_ids: set[str]) -> None:
        if not dirty_product_ids:
            return
        state = self._supplier_metric_state
        for pid in dirty_product_ids:
            p = self.products.get(pid)
            if p is None:
                continue
            old = self._product_metric_contrib.get(pid, (0, 0.0, 0))
            new = self._product_metric_contribution(p)
            self._product_metric_contrib[pid] = new
            state["product_avail_count"] += new[0] - old[0]
            state["total_supplier_price"] += new[1] - old[1]
            state["total_supplier_qty"] += new[2] - old[2]

    def _maybe_recompute_supplier_metrics_for_checkpoint(self) -> None:
        interval = int(self.scenario["run"].get("checkpoint_interval_steps", 168) or 0)
        horizon = int(self.scenario["run"]["horizon_steps"])
        if interval > 0 and (self.t % interval == 0 or self.t + 1 >= horizon):
            self._recompute_supplier_metrics()

    def _write_metrics(
        self,
        events: list[EventLog],
        new_orders: list[Order],
        mutated: list[Order],
        *,
        write_rating_metrics: bool = True,
    ) -> None:
        t = self.t

        # Global (_global): supplier pool + order flow
        supplier_state = self._supplier_metric_state
        product_avail = int(supplier_state.get("product_avail_count", 0))
        total_price = float(supplier_state.get("total_supplier_price", 0.0))
        mean_price = (total_price / product_avail) if product_avail else 0.0
        total_qty = int(supplier_state.get("total_supplier_qty", 0))
        delist_events = sum(1 for e in events if e.event_type == "supplier_delist")
        relist_events = sum(1 for e in events if e.event_type == "supplier_relist")
        price_changes = sum(1 for e in events if e.event_type == "price_change")
        supplier_timeout_events = sum(1 for e in events if e.event_type == "supplier_timeout")
        supplier_timeout_end_events = sum(1 for e in events if e.event_type == "supplier_timeout_end")
        orders_generated = len(new_orders)

        global_kv = {
            "product_avail_count": product_avail,
            "mean_supplier_price": mean_price,
            "total_supplier_qty": total_qty,
            "delist_events": delist_events,
            "relist_events": relist_events,
            "price_changes": price_changes,
            "supplier_timeout_events": supplier_timeout_events,
            "supplier_timeout_end_events": supplier_timeout_end_events,
            "orders_generated": orders_generated,
        }
        dbm.write_metrics(self.conn, self.run_id, "_global", t, global_kv)

        # Per-agent metrics.
        # cum_gmv: gross merchandise volume — sum of sale_price for all successfully
        #   procured orders (DB query, never decremented on refund/cancel, survives
        #   listing removal).
        # cum_cost: total procurement cost for all successfully procured orders
        #   (DB query on purchase_price, never decremented on refund/cancel).
        # cum_gross_profit: booked gross profit for all successfully procured orders
        #   (cum_gmv - cum_cost), never decremented on refund/cancel.
        # cum_net_profit: matched economic profit, summed only over orders that have
        #   reached a terminal status (settled_t IS NOT NULL). For each settled order:
        #   realized_revenue − realized_cost − total_penalty − commission_amount −
        #   logistics_fee − reverse_logistics_fee (matches Order.net_profit). Fee
        #   columns are 0 until charged / when v6 flags are off.
        # cum_fee: SUM(commission_amount + logistics_fee + reverse_logistics_fee)
        #   over all orders for this agent. Fees stay 0 until charged. Distinct
        #   from cum_fine (platform-violation penalties). _step_cost stays
        #   purchase COGS only; fulfillment F lives here, not in step_cost.
        # revenue_rate: per-step gross revenue accumulated during auto-purchase.
        step_revenue = getattr(self, "_step_revenue", {})
        for aid, st in self.agents.items():
            n_listings = len(st.listings)
            revenue_rate = float(step_revenue.get(aid, 0.0))
            sums_row = self.conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN current_status NOT IN"
                "   ('stockout','insufficient_balance')"
                "   THEN purchase_price ELSE 0 END), 0) AS cost,"
                " COALESCE(SUM(CASE WHEN current_status NOT IN"
                "   ('stockout','insufficient_balance')"
                "   THEN sale_price ELSE 0 END), 0) AS gmv,"
                " COALESCE(SUM(CASE WHEN settled_t IS NOT NULL"
                f"   THEN {dbm.order_net_profit_sql()} ELSE 0 END), 0)"
                "   AS net_profit,"
                f" COALESCE(SUM({dbm.order_fee_total_sql()}), 0) AS fee_total"
                " FROM orders"
                " WHERE run_id=? AND agent_id=?",
                (self.run_id, aid),
            ).fetchone()
            cum_cost = float(sums_row["cost"] or 0.0)
            cum_gmv = float(sums_row["gmv"] or 0.0)
            cum_fine = st.cash.cumulative_fine
            cum_gross_profit = cum_gmv - cum_cost
            cum_net_profit = float(sums_row["net_profit"] or 0.0)
            cum_fee = float(sums_row["fee_total"] or 0.0)
            net_assets = (st.cash.balance + st.cash.receivable
                          + st.cash.in_transit + st.cash.deposit_pool)
            kv = {
                "balance": st.cash.balance,
                "deposit_pool": st.cash.deposit_pool,
                "in_transit": st.cash.in_transit,
                "receivable": st.cash.receivable,
                "cum_fine": cum_fine,
                "n_active_listings": n_listings,
                "cum_gmv": cum_gmv,
                "revenue_rate": revenue_rate,
                "cum_cost": cum_cost,
                "cum_gross_profit": cum_gross_profit,
                "cum_net_profit": cum_net_profit,
                "cum_fee": cum_fee,
                "net_assets": net_assets,
            }
            if write_rating_metrics:
                kv.update(self._shop_rating_metric_values(st))
            kv.update(self._listing_price_metric_values(
                st, include_rating=write_rating_metrics,
            ))
            dbm.write_metrics(self.conn, self.run_id, aid, t, kv)

    def _survival_state(self) -> dict:
        """Run-level liveness: alive = at least one agent still has deposit_pool > 0.
        Dead agents stay dead and their actions are blocked. Once no agent remains
        alive, the runner stops normal simulation and only drains existing orders.
        """
        alive_agents = [aid for aid, st in self.agents.items() if st.is_alive]
        is_alive = bool(alive_agents)
        return {
            "is_alive": is_alive,
            "alive_agents": alive_agents,
            "dead_agents": [{"agent_id": aid, "died_at_t": st.died_at_t}
                            for aid, st in self.agents.items() if not st.is_alive],
            "days_alive": int((self.t * int(self.scenario["run"]["step_hours"])) // 24),
            "bankruptcy_step": None if is_alive else self.t,
        }

    def record_act(self, agent_id: str, assistant_msg: dict, tool_msgs: list[dict],
                   token_usage: Optional[dict] = None,
                   trace_msgs: Optional[list[dict]] = None,
                   recorded_messages: Optional[list[dict]] = None,
                   context: Optional[dict] = None,
                   ignore_turn_quota: bool = False) -> dict:
        """Record one /act turn: assistant message + tool result messages.

        Called by the /act route. Atomically writes by_step for live visibility.
        Returns {"ok": True, "turn_idx": N} or error.

        ``ignore_turn_quota`` is for pure ``end_of_step`` releases after the
        per-step action budget is exhausted.
        """
        from storage import agent_log
        max_turns = int((self.scenario.get("agent", {}) or {}).get("max_turns_per_step", 0))
        token_usage = agent_log.normalize_token_usage(token_usage)
        messages_for_origin = (
            recorded_messages
            if recorded_messages is not None
            else [*(trace_msgs or []), assistant_msg, *tool_msgs]
        )
        tool_origins: dict[str, int] = {}
        message_origins: dict[str, int] = {}
        tool_call_origins: dict[str, int] = {}
        tool_result_origins: dict[str, int] = {}
        for msg in messages_for_origin:
            origin = msg.get("tool_origin") if isinstance(msg, dict) else None
            if origin:
                origin = str(origin)
                message_origins[origin] = message_origins.get(origin, 0) + 1
                tool_origins[origin] = tool_origins.get(origin, 0) + 1
                if msg.get("role") == "tool":
                    tool_result_origins[origin] = tool_result_origins.get(origin, 0) + 1
            if isinstance(msg, dict):
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("tool_origin"):
                        origin = str(tc["tool_origin"])
                        tool_call_origins[origin] = tool_call_origins.get(origin, 0) + 1
                        tool_origins[origin] = tool_origins.get(origin, 0) + 1
        with self.turn_lock:
            self._ensure_agent_protocol_state(agent_id)
            agent_turns = self.turns_meta_by_agent[agent_id]
            turn_idx = len(agent_turns)
            if max_turns and turn_idx >= max_turns and not ignore_turn_quota:
                return {"ok": False, "error": f"max_turns_per_step={max_turns} reached"}
            wall_ms = agent_log.now_ms()
            if recorded_messages is not None:
                for msg in recorded_messages:
                    self._record_agent_message(agent_id, msg)
            else:
                for msg in trace_msgs or []:
                    self._record_agent_message(agent_id, msg)
                self._record_agent_message(agent_id, assistant_msg)
                for msg in tool_msgs:
                    self._record_agent_message(agent_id, msg)
            turn_meta = {
                "turn_idx": turn_idx,
                "token_usage": token_usage or {},
                "tool_origins": tool_origins,
                "message_origins": message_origins,
                "tool_call_origins": tool_call_origins,
                "tool_result_origins": tool_result_origins,
                "received_at_wall_ms": wall_ms,
            }
            if context:
                turn_meta["context"] = context
            agent_turns.append(turn_meta)
            global_turn_meta = {
                "turn_idx": turn_idx,
                "agent_id": agent_id,
                "token_usage": token_usage or {},
                "tool_origins": tool_origins,
                "message_origins": message_origins,
                "tool_call_origins": tool_call_origins,
                "tool_result_origins": tool_result_origins,
                "received_at_wall_ms": wall_ms,
            }
            if context:
                global_turn_meta["context"] = context
            self.turns_meta.append(global_turn_meta)
            self.last_turn_wall_ms = wall_ms
            messages_snapshot, turns_snapshot, message_agents_snapshot = self._combined_agent_messages_and_turns()
        agent_log.write_step_live(
            self.runs_root, self.run_id, self.t,
            messages_snapshot, turns_snapshot,
            message_agents=message_agents_snapshot,
            hook_open_wall_ms=self.last_hook_open_wall_ms,
        )
        if self.turn_listeners:
            import logging as _log
            event = {
                "agent_id": agent_id,
                "turn_idx": turn_idx,
                "t": self.t,
                "received_at_wall_ms": wall_ms,
            }
            for fn in list(self.turn_listeners):
                try:
                    fn(event)
                except Exception:
                    _log.getLogger(__name__).exception("turn listener failed")
        return {"ok": True, "turn_idx": turn_idx}

    def with_idempotency(self, key: Optional[str], fn: Callable[[], dict],
                         fingerprint: Optional[dict] = None) -> dict:
        """Execute fn(), caching its result by `key`. On a cache hit the
        returned dict carries `_idempotent_replay: True`. Falsy key bypasses
        the cache entirely.

        Successful mutations are persisted before their result is returned.
        This keeps the idempotency record on the same acknowledgement side of
        a crash boundary as the mutation itself.
        """
        if not key:
            return fn()
        hit = self.idem_cache.get(key)
        if hit is not None:
            if self._idem_dirty:
                from storage import agent_log
                agent_log.persist_idem(
                    self.runs_root, self.run_id, self.idem_cache
                )
                self._idem_dirty = False
            if isinstance(hit, dict) and "__idem_result" in hit:
                if hit.get("__idem_fingerprint") != fingerprint:
                    return {
                        "ok": False,
                        "error": "idempotency_conflict",
                        "_http_status": 409,
                    }
                cached = hit.get("__idem_result")
                if isinstance(cached, dict):
                    return {**cached, "_idempotent_replay": True}
                return {"ok": False, "error": "invalid idempotency cache entry"}
            return {**hit, "_idempotent_replay": True}
        result = fn()
        # Only cache successful results — failures are typically transient and
        # the agent should be free to retry without the cache returning the error.
        if isinstance(result, dict) and result.get("ok", True) is not False:
            self.idem_cache.put(key, {
                "__idem_fingerprint": fingerprint,
                "__idem_result": result,
            })
            self._idem_dirty = True
            from storage import agent_log
            agent_log.persist_idem(
                self.runs_root, self.run_id, self.idem_cache
            )
            self._idem_dirty = False
        return result
