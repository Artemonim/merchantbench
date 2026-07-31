"""Scenario difficulty helpers.

`difficulty_rate` is a run-local multiplier map for probability fields. It
does not mutate the source private dataset; it only adjusts the Product objects
copied into a run.
"""
from __future__ import annotations

import math
from typing import Iterable

from core.entities import Product


RATE_FIELDS = (
    "cancel_rate",
    "refund_rate",
    "only_refund_rate",
    "bad_review_rate",
    "timeout_rate",
    "price_change_rate",
    "supplier_delist_rate",
)


def apply_difficulty_rate(products: Iterable[Product], scenario: dict) -> dict[str, float]:
    """Apply top-level ``difficulty_rate`` multipliers to run-local products.

    Missing fields default to 1.0. Values must be non-negative finite numbers.
    Effective probabilities are clamped to 1.0 because probabilities cannot
    exceed 100%.
    """
    if "difficulty_rate" not in scenario or scenario.get("difficulty_rate") is None:
        return {}
    raw_cfg = scenario["difficulty_rate"]
    if not isinstance(raw_cfg, dict):
        raise ValueError("difficulty_rate must be a mapping of probability field to multiplier")
    if not raw_cfg:
        return {}

    allowed = set(RATE_FIELDS)
    multipliers: dict[str, float] = {}
    for field, raw_value in raw_cfg.items():
        if field not in allowed:
            allowed_text = ", ".join(RATE_FIELDS)
            raise ValueError(f"unknown difficulty_rate.{field}; allowed fields: {allowed_text}")
        try:
            multiplier = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"difficulty_rate.{field} must be a non-negative finite number"
            ) from exc
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError(
                f"difficulty_rate.{field} must be a non-negative finite number"
            )
        multipliers[field] = multiplier

    for product in products:
        for field, multiplier in multipliers.items():
            base_rate = float(getattr(product, field))
            setattr(product, field, min(base_rate * multiplier, 1.0))

    return multipliers
