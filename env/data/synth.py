"""Synthetic data generator. All randomness from derive_rng(channel='data_gen').

Produces:
- list[Product]
- hourly_dist: {category: 24h ndarray, sums to 1}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.entities import Product
from core.rng import derive_rng
from data.generation_profiles import (
    DEFAULT_PRICING_MODEL,
    PRICING_MODEL_LEGACY_ANCHOR_AT_COST,
    PRICING_MODEL_MARGIN_CONSISTENT_V1,
    apply_risk_trust_coupling,
    build_hourly_dist_for_categories,
    cost_and_elasticity_from_margin,
    generation_params_from_scenario,
    normalize_supplier_ranges,
    rand_range,
    resolve_base_demand_range,
    risk_trust_coupling_enabled,
    sample_elasticity,
    sample_operational_fields,
    sample_product_rating,
    sample_retail_margin,
    sample_risk_event_fields,
    sample_supplier_profile_maps,
)
from data.product_titles import (
    LEGACY_ADJECTIVES,
    LEGACY_FALLBACK_NOUNS,
    LEGACY_NOUN_POOLS,
    generate_title,
    parse_title_typo_rate,
)


def _market_curve(rng: np.random.Generator, base: float) -> list[float]:
    """365-day curve with a weekly cycle + seasonal trend + noise. Non-negative."""
    days = np.arange(365)
    seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * days / 365)
    weekly = 1.0 + 0.2 * np.sin(2 * np.pi * days / 7)
    noise = rng.normal(1.0, 0.1, size=365)
    curve = base * seasonal * weekly * np.maximum(noise, 0.3)
    return [float(max(0.0, x)) for x in curve]


# Safe defaults for the trust-signal profile ranges. Older scenarios that
# don't declare these sections still produce valid Products. Tune in scenario
# YAML via `supplier_profile_ranges` / `product_profile_ranges`.
_DEFAULT_SUPPLIER_PROFILE_RANGES = {
    "shop_rating": [3.5, 5.0],
    "return_buyer_rate": [0.05, 0.30],
    "supplier_age_years": [0.25, 10.0],
}
_DEFAULT_PRODUCT_PROFILE_RANGES = {
    "historical_avg_rating": [3.5, 5.0],
}


def _resolve_pricing_model(profile_params: dict[str, Any]) -> str:
    """Return a known catalog pricing model, defaulting to v5."""
    raw = profile_params.get("pricing_model", DEFAULT_PRICING_MODEL)
    if raw is None or raw == "":
        raw = DEFAULT_PRICING_MODEL
    model = str(raw)
    if model not in {
        PRICING_MODEL_MARGIN_CONSISTENT_V1,
        PRICING_MODEL_LEGACY_ANCHOR_AT_COST,
    }:
        raise ValueError(f"unknown pricing_model: {model!r}")
    return model


def generate(scenario: dict[str, Any]) -> tuple[list[Product], dict[str, np.ndarray]]:
    master_seed = int(scenario["run"]["master_seed"])
    data_cfg = scenario["data"]
    risk_cfg = scenario["risk_ranges"]
    sup_cfg = scenario["supplier_ranges"]
    sup_prof_cfg = {**_DEFAULT_SUPPLIER_PROFILE_RANGES,
                    **scenario.get("supplier_profile_ranges", {})}
    prod_prof_cfg = {**_DEFAULT_PRODUCT_PROFILE_RANGES,
                     **scenario.get("product_profile_ranges", {})}
    profile_params = generation_params_from_scenario(scenario)
    pricing_model = _resolve_pricing_model(profile_params)
    demand_lo, demand_hi = resolve_base_demand_range(profile_params)

    sup_cfg = normalize_supplier_ranges(sup_cfg, scenario.get("platform_rules", {}))

    n_products = int(data_cfg["num_products"])
    n_suppliers = int(data_cfg["num_suppliers"])
    categories = list(data_cfg["category_pool"])[: int(data_cfg["num_categories"])]
    typo_rate = parse_title_typo_rate(
        data_cfg.get("title_typo_rate", 0.0),
        field="data.title_typo_rate",
    )

    supplier_names = [f"sup_{i:04d}" for i in range(n_suppliers)]
    supplier_display = [f"Supplier#{i:04d}" for i in range(n_suppliers)]

    # Supplier-level trust signals: sampled ONCE per supplier and then shared
    # across every product owned by that supplier. Use a dedicated rng so the
    # values stay deterministic regardless of how many products / which order.
    sup_profile_rng = derive_rng(master_seed, "data_gen", "supplier_profile")
    shop_rating_by_sup, return_buyer_by_sup, age_by_sup = sample_supplier_profile_maps(
        supplier_names,
        sup_profile_rng,
        sup_prof_cfg,
    )

    products: list[Product] = []
    for pid_idx in range(n_products):
        rng = derive_rng(master_seed, "data_gen", "product", pid_idx)
        cat = categories[pid_idx % len(categories)]
        sup_idx = int(rng.integers(0, n_suppliers))
        # * Dummy draws keep this stream bitwise-compatible with the
        # * pre-generator catalog (legacy name-pool integer bounds).
        noun_pool = LEGACY_NOUN_POOLS.get(cat, LEGACY_FALLBACK_NOUNS)
        _ = noun_pool[int(rng.integers(0, len(noun_pool)))]
        _ = LEGACY_ADJECTIVES[int(rng.integers(0, len(LEGACY_ADJECTIVES)))]
        # * Titles use derive_rng(..., "product_title", pid_idx), not this stream.

        ref_price = rand_range(rng, *sup_cfg["ref_price"])
        base_demand = rand_range(rng, demand_lo, demand_hi)
        operational = sample_operational_fields(rng, sup_cfg)
        risk_event = sample_risk_event_fields(rng, risk_cfg, sup_cfg)

        sup_name = supplier_names[sup_idx]
        # * RNG order: ref_price, base_demand, operational, risk, rating,
        # * then exactly one elasticity-or-margin draw, then market_curve.
        historical_avg_rating = sample_product_rating(rng, prod_prof_cfg)
        if pricing_model == PRICING_MODEL_MARGIN_CONSISTENT_V1:
            margin = sample_retail_margin(rng, cat, profile_params)
            cost, elasticity = cost_and_elasticity_from_margin(ref_price, margin)
            price = cost
        else:
            price = ref_price
            elasticity = sample_elasticity(
                rng, cat, profile_params, sup_cfg["elasticity"]
            )
        title_rng = derive_rng(master_seed, "data_gen", "product_title", pid_idx)
        product = Product(
            product_id=f"P{pid_idx:05d}",
            name=generate_title(cat, title_rng, typo_rate=typo_rate),
            quantity=operational["quantity"],
            price=price,
            ref_price=ref_price,
            supplier_id=sup_name,
            supplier_name=supplier_display[sup_idx],
            ship_hours=operational["ship_hours"],
            logistics_hours=operational["logistics_hours"],
            category=cat,
            historical_avg_rating=historical_avg_rating,
            shop_rating=shop_rating_by_sup[sup_name],
            return_buyer_rate=return_buyer_by_sup[sup_name],
            supplier_age_years=age_by_sup[sup_name],
            cancel_rate=risk_event["cancel_rate"],
            refund_rate=risk_event["refund_rate"],
            only_refund_rate=risk_event["only_refund_rate"],
            bad_review_rate=risk_event["bad_review_rate"],
            max_quantity=operational["max_quantity"],
            hourly_increment=operational["hourly_increment"],
            timeout_rate=risk_event["timeout_rate"],
            price_change_rate=risk_event["price_change_rate"],
            supplier_delist_rate=risk_event["supplier_delist_rate"],
            elasticity=elasticity,
            market_curve=_market_curve(rng, base=base_demand),
        )
        products.append(product)

    hourly_dist = build_hourly_dist_for_categories(
        categories,
        seed=master_seed,
        params=profile_params,
    )
    # * Risk↔trust coupling is a deterministic post-process. It must not
    # * consume RNG or sit between v5 prefix draws (operational / risk /
    # * rating / elasticity / market_curve).
    if risk_trust_coupling_enabled(profile_params):
        for product in products:
            apply_risk_trust_coupling(
                product,
                risk_ranges=risk_cfg,
                supplier_ranges=sup_cfg,
                supplier_profile_ranges=sup_prof_cfg,
                product_profile_ranges=prod_prof_cfg,
            )
    return products, hourly_dist
