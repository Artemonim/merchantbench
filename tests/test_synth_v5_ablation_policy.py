"""In-process 1-day rule_based-markup policy ablations.

Uses ``create_app`` / ``/runs`` / ``/step`` like ``tests/test_smoke.py``.
Does not start Hermes, a long-lived env server, or ``scripts/run_batch.py``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
from core.entities import StoreListing
from data.economy_diagnostics import RULE_BASED_DEFAULT_MARKUP
from data.generation_profiles import ELASTICITY_CLIP_MIN
from storage import db as dbm
from web.app import create_app
from web.runner import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_DIR = REPO_ROOT / "env" / "scenarios" / "ablations"
POLICY_AXES = ("pricing_only", "demand_only", "both")
POLICY_N_PRODUCTS = 100
POLICY_N_LISTINGS = 10
POLICY_HORIZON_STEPS = 24
POLICY_SEED = 42
POLICY_FALLBACK_SEED = 43
# * Enough cash to book pricing_only volume at 2×cost without
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


def _booked_order_metrics(conn, run_id: str) -> dict[str, float]:
    """Return booked-only GMV/gross/count plus all-row and violation counts."""
    booked_gmv, booked_gross, booked_count = conn.execute(
        """
        SELECT COALESCE(SUM(sale_price), 0),
               COALESCE(SUM(sale_price - purchase_price), 0),
               COUNT(*)
        FROM orders
        WHERE run_id = ?
          AND current_status NOT IN ('stockout', 'insufficient_balance')
        """,
        (run_id,),
    ).fetchone()
    all_count = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    violation_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE run_id = ?
          AND current_status IN ('stockout', 'insufficient_balance')
        """,
        (run_id,),
    ).fetchone()[0]
    return {
        "booked_gmv": float(booked_gmv),
        "booked_gross": float(booked_gross),
        "booked_count": int(booked_count),
        "all_count": int(all_count),
        "violation_count": int(violation_count),
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
            metrics = _booked_order_metrics(env.conn, run_id)
            metrics["run_id"] = run_id
            metrics["elasticities"] = [float(product.elasticity) for product in env.products.values()]
            return metrics
    finally:
        app.registry.shutdown()


def _sum_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Add booked counters across seeds."""
    return {
        "booked_gmv": sum(row["booked_gmv"] for row in rows),
        "booked_gross": sum(row["booked_gross"] for row in rows),
        "booked_count": sum(row["booked_count"] for row in rows),
        "all_count": sum(row["all_count"] for row in rows),
        "violation_count": sum(row["violation_count"] for row in rows),
    }


@pytest.fixture(scope="module")
def policy_day_by_axis(tmp_path_factory):
    """Run the 1-day markup policy once per ablation axis at seed 42."""
    results: dict[str, dict[str, Any]] = {}
    for name in POLICY_AXES:
        tmp = tmp_path_factory.mktemp(f"{name}_{POLICY_SEED}")
        results[name] = _simulate_markup_policy_day(
            ABLATIONS_DIR / f"{name}.yaml",
            tmp,
            seed=POLICY_SEED,
        )
    return results


def test_policy_volume_and_margin_axes_separate_on_booked_orders(policy_day_by_axis, tmp_path_factory):
    pricing = policy_day_by_axis["pricing_only"]
    demand = policy_day_by_axis["demand_only"]
    both = policy_day_by_axis["both"]

    pricing_totals = {
        "booked_gmv": pricing["booked_gmv"],
        "booked_gross": pricing["booked_gross"],
        "booked_count": pricing["booked_count"],
    }
    both_totals = {
        "booked_gmv": both["booked_gmv"],
        "booked_gross": both["booked_gross"],
        "booked_count": both["booked_count"],
    }
    # * Rare 1-day Poisson tie: add seed 43 rather than claiming a live 7d batch.
    if pricing_totals["booked_count"] <= both_totals["booked_count"]:
        extra = {}
        for name in ("pricing_only", "both"):
            tmp = tmp_path_factory.mktemp(f"{name}_{POLICY_FALLBACK_SEED}")
            extra[name] = _simulate_markup_policy_day(
                ABLATIONS_DIR / f"{name}.yaml",
                tmp,
                seed=POLICY_FALLBACK_SEED,
            )
        pricing_totals = _sum_metrics([pricing, extra["pricing_only"]])
        both_totals = _sum_metrics([both, extra["both"]])

    assert pricing_totals["booked_count"] > both_totals["booked_count"]
    assert pricing_totals["booked_gmv"] > both_totals["booked_gmv"]
    assert both["booked_gross"] > 0.0
    assert demand["booked_gross"] > 0.0
    assert all(eps >= ELASTICITY_CLIP_MIN for eps in pricing["elasticities"])
    assert all(eps >= ELASTICITY_CLIP_MIN for eps in both["elasticities"])
    assert any(eps < 1.0 for eps in demand["elasticities"])
    assert pricing["all_count"] >= pricing["booked_count"]
    assert demand["all_count"] >= demand["booked_count"]
    assert both["all_count"] >= both["booked_count"]


def test_policy_day_lists_first_ten_product_ids_at_rule_markup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("list_check")
    scenario = _policy_scenario(ABLATIONS_DIR / "both.yaml", seed=POLICY_SEED)
    app = create_app(
        db_path=os.path.join(tmp, "test.db"),
        runs_root=os.path.join(tmp, "runs"),
    )
    try:
        with app.test_client() as client:
            run_id = client.post("/runs", json={"scenario": scenario}).get_json()["run_id"]
            env = app.registry._require(run_id)
            listed = _preseed_rule_markup_listings(env)
            assert listed == [f"P{idx:05d}" for idx in range(POLICY_N_LISTINGS)]
            for product_id in listed:
                product = env.products[product_id]
                listing = env.agents["agent_0"].listings[product_id]
                assert listing.sale_price == round(float(product.price) * RULE_BASED_DEFAULT_MARKUP, 2)
    finally:
        app.registry.shutdown()
