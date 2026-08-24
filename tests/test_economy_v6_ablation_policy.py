"""In-process 1-day rule_based-markup policy for economy v6 fee overlays.

Uses ``create_app`` / ``/runs`` / ``/step`` like ``tests/test_synth_v5_ablation_policy.py``.
Does not start Hermes, a long-lived env server, or ``scripts/run_batch.py``.

A 1-day horizon charges fulfillment F at purchase. Take-rate and refund
haircuts wait for delivered+settlement (~168h) and are covered by
``tests/test_order_manager.py``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
from core.entities import StoreListing
from data.economy_diagnostics import RULE_BASED_DEFAULT_MARKUP
from storage import db as dbm
from web.app import create_app
from web.runner import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "env" / "scenarios"
ABLATIONS_DIR = SCENARIOS_DIR / "ablations"
POLICY_N_PRODUCTS = 100
POLICY_N_LISTINGS = 10
POLICY_HORIZON_STEPS = 24
POLICY_SEED = 42
# * Enough cash to book 2×cost volume plus fulfillment F without
# * insufficient_balance drowning the booked-only comparison.
POLICY_INITIAL_CASH = 100000.0


def _policy_scenario(path: Path, *, seed: int) -> dict[str, Any]:
    """Return a 1-day, 100-product copy of an ablation overlay."""
    scenario = copy.deepcopy(load_scenario(str(path)))
    scenario["run"]["max_hook_seconds"] = 0.1
    scenario["run"]["horizon_steps"] = POLICY_HORIZON_STEPS
    scenario["run"]["master_seed"] = int(seed)
    scenario["run"]["initial_cash"] = POLICY_INITIAL_CASH
    scenario["data"]["num_products"] = POLICY_N_PRODUCTS
    # * Skip agent hooks: this test seeds listings directly, like smoke.
    scenario.setdefault("agent", {})["tool_denylist"] = []
    scenario["agent"]["activation_period"] = 10**9
    scenario.pop("lifecycle", None)
    return scenario


def _preseed_rule_markup_listings(env, n: int = POLICY_N_LISTINGS) -> list[str]:
    """List the first ``n`` product_ids at rule_based ``DEFAULT_MARKUP``."""
    ranked = sorted(env.products.values(), key=lambda product: product.product_id)
    listed_ids: list[str] = []
    for product in ranked[:n]:
        sale_price = round(float(product.price) * RULE_BASED_DEFAULT_MARKUP, 2)
        listing = StoreListing(
            product_id=product.product_id,
            agent_id="agent_0",
            sale_price=sale_price,
            listed_at=0,
        )
        dbm.upsert_listing(env.conn, env.run_id, "agent_0", listing)
        env.agents["agent_0"].listings[product.product_id] = listing
        listed_ids.append(product.product_id)
    return listed_ids


def _booked_fee_metrics(env) -> dict[str, float]:
    """Return booked GMV/gross/net plus fulfillment cash diagnostics."""
    conn = env.conn
    run_id = env.run_id
    booked_gmv, booked_gross, booked_count, booked_net, fee_sum, commission_sum = conn.execute(
        f"""
            SELECT COALESCE(SUM(sale_price), 0),
                   COALESCE(SUM(sale_price - purchase_price), 0),
                   COUNT(*),
                   COALESCE(SUM({dbm.order_net_profit_sql()}), 0),
                   COALESCE(SUM(logistics_fee), 0),
                   COALESCE(SUM(commission_amount), 0)
            FROM orders
            WHERE run_id = ?
              AND current_status NOT IN ('stockout', 'insufficient_balance')
            """,
        (run_id,),
    ).fetchone()
    rows = conn.execute(
        """
        SELECT product_id, logistics_fee
        FROM orders
        WHERE run_id = ?
          AND current_status NOT IN ('stockout', 'insufficient_balance')
        """,
        (run_id,),
    ).fetchall()
    expected_fulfillment = 0.0
    for product_id, _fee in rows:
        product = env.products[product_id]
        if env.economy_v6.fulfillment_enabled:
            expected_fulfillment += env.economy_v6.fulfillment_fee(product.category)
    cash = env.agents["agent_0"].cash
    return {
        "booked_gmv": float(booked_gmv),
        "booked_gross": float(booked_gross),
        "booked_count": int(booked_count),
        "booked_net": float(booked_net),
        "logistics_fee_sum": float(fee_sum),
        "commission_sum": float(commission_sum),
        "expected_fulfillment": float(expected_fulfillment),
        "cash_balance": float(cash.balance),
        "cash_in_transit": float(cash.in_transit),
    }


def _simulate_markup_policy_day(
    scenario_path: Path,
    tmp_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Create a run, list 10 products at 2×cost, step one sim day, return metrics."""
    scenario = _policy_scenario(scenario_path, seed=seed)
    app = create_app(
        db_path=os.path.join(tmp_dir, "test.db"),
        runs_root=os.path.join(tmp_dir, "runs"),
    )
    try:
        with app.test_client() as client:
            response = client.post("/runs", json={"scenario": scenario})
            assert response.status_code == 200, response.get_data(as_text=True)
            run_id = response.get_json()["run_id"]
            env = app.registry._require(run_id)
            listed_ids = _preseed_rule_markup_listings(env)
            assert listed_ids == [f"P{idx:05d}" for idx in range(POLICY_N_LISTINGS)]
            for _ in range(POLICY_HORIZON_STEPS):
                step = client.post(f"/runs/{run_id}/step")
                assert step.status_code == 200, step.get_data(as_text=True)
            metrics = _booked_fee_metrics(env)
            metrics["run_id"] = run_id
            metrics["fulfillment_enabled"] = bool(env.economy_v6.fulfillment_enabled)
            metrics["take_rate_enabled"] = bool(env.economy_v6.take_rate_enabled)
            return metrics
    finally:
        app.registry.shutdown()


@pytest.fixture(scope="module")
def v6_policy_day(tmp_path_factory):
    """Run the 1-day markup policy for v5 both, fees_only, and refund_only."""
    axes = {
        "v5_both": ABLATIONS_DIR / "both.yaml",
        "fees_only": ABLATIONS_DIR / "economy_v6_fees_only.yaml",
        "refund_only": ABLATIONS_DIR / "economy_v6_refund_only.yaml",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, path in axes.items():
        tmp = tmp_path_factory.mktemp(f"{name}_{POLICY_SEED}")
        results[name] = _simulate_markup_policy_day(path, tmp, seed=POLICY_SEED)
    return results


def test_fees_only_keeps_booked_gmv_and_charges_fulfillment_at_purchase(
    v6_policy_day,
):
    v5 = v6_policy_day["v5_both"]
    fees = v6_policy_day["fees_only"]
    refund = v6_policy_day["refund_only"]

    assert v5["booked_count"] > 0
    assert fees["booked_count"] > 0
    # * Demand is unchanged: arrival Poisson does not read fee flags.
    assert fees["booked_gmv"] == pytest.approx(v5["booked_gmv"], rel=0.15, abs=1.0)
    assert fees["booked_gross"] == pytest.approx(v5["booked_gross"], rel=0.15, abs=1.0)

    assert v5["fulfillment_enabled"] is False
    assert v5["logistics_fee_sum"] == pytest.approx(0.0)
    assert v5["commission_sum"] == pytest.approx(0.0)

    assert fees["fulfillment_enabled"] is True
    assert fees["take_rate_enabled"] is True
    assert fees["expected_fulfillment"] > 0.0
    assert fees["logistics_fee_sum"] == pytest.approx(fees["expected_fulfillment"])
    # * Take-rate is not charged until settlement (~168h). 1-day stays at 0.
    assert fees["commission_sum"] == pytest.approx(0.0)

    # * Fee-aware net drops by F at purchase, not by τ·GMV, in a 1-day window.
    assert fees["booked_net"] == pytest.approx(v5["booked_net"] - fees["logistics_fee_sum"], abs=1.0)
    assert fees["booked_net"] < v5["booked_net"]
    assert fees["cash_balance"] == pytest.approx(v5["cash_balance"] - fees["logistics_fee_sum"], abs=1.0)

    assert refund["fulfillment_enabled"] is False
    assert refund["logistics_fee_sum"] == pytest.approx(0.0)
    assert refund["booked_gmv"] == pytest.approx(v5["booked_gmv"], rel=0.15, abs=1.0)
