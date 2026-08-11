"""Shared downstream order-rating math for products and shops.

rating =
  (prior_weight * initial_rating + decayed_weighted_score_sum)
  / (prior_weight + decayed_effective_weight)

Every terminal order contributes exactly once. Product and shop ratings use
the same outcome score/weight mapping; they only differ in aggregation scope,
prior strength, and evidence half-life.

The v3 shop policy combines that recent-quality score with a separate lifetime
rated-order count. The count never decays and maps to a bounded trust multiplier
with diminishing returns.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from typing import Optional


DEFAULT_OUTCOME_SCORES = {
    "normal_score": 4.5,
    "late_score": 3.0,
    "refund_score": 2.0,
    "only_refund_score": 1.5,
    "bad_review_score": 1.0,
    "stockout_score": 1.0,
}

DEFAULT_OUTCOME_WEIGHTS = {
    "normal_weight": 1.0,
    "late_weight": 1.0,
    "refund_weight": 1.0,
    "only_refund_weight": 2.0,
    "bad_review_weight": 2.0,
    "stockout_weight": 3.0,
}

DEFAULT_REPUTATION_VOLUME_CONFIG = {
    "min_multiplier": 0.80,
    "max_multiplier": 1.00,
    "half_saturation_orders": 20.0,
}


def _clamp_rating(value: float) -> float:
    return max(1.0, min(5.0, float(value)))


def compute_listing_rating(
    initial_rating: float,
    rating_sum: float,
    rating_count: float,
    prior_weight: float,
) -> float:
    denominator = prior_weight + rating_count
    if denominator <= 0:
        return _clamp_rating(initial_rating)
    numerator = prior_weight * initial_rating + rating_sum
    raw = numerator / denominator
    return _clamp_rating(raw)


def outcome_key_for_order(
    current_status: str,
    late_t: Optional[int] = None,
) -> Optional[str]:
    """Return the canonical outcome prefix for one terminal order."""
    if current_status == "settled_normal":
        return "late" if late_t is not None else "normal"
    elif current_status == "settled_refund":
        return "refund"
    elif current_status == "settled_only_refund":
        return "only_refund"
    elif current_status == "settled_bad_review":
        return "bad_review"
    elif current_status == "stockout":
        return "stockout"
    return None


def score_for_order_outcome(
    current_status: str,
    late_t: Optional[int] = None,
    scores: Optional[dict] = None,
) -> Optional[float]:
    """Return one final-experience score for an eligible terminal order."""
    outcome = outcome_key_for_order(current_status, late_t)
    if outcome is None:
        return None
    cfg = {**DEFAULT_OUTCOME_SCORES, **(scores or {})}
    return _clamp_rating(float(cfg[f"{outcome}_score"]))


def outcome_contribution(
    current_status: str,
    late_t: Optional[int] = None,
    scores: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> Optional[tuple[float, float]]:
    """Return ``(weighted_score, effective_weight)`` for one order."""
    outcome = outcome_key_for_order(current_status, late_t)
    if outcome is None:
        return None
    score_cfg = {**DEFAULT_OUTCOME_SCORES, **(scores or {})}
    weight_cfg = {**DEFAULT_OUTCOME_WEIGHTS, **(weights or {})}
    score = _clamp_rating(float(score_cfg[f"{outcome}_score"]))
    weight = float(weight_cfg[f"{outcome}_weight"])
    if weight <= 0:
        raise ValueError(f"{outcome}_weight must be positive")
    return score * weight, weight


def decay_from_half_life(half_life_days: float) -> float:
    """Return the per-day exponential decay for a positive half-life."""
    half_life = float(half_life_days)
    if half_life <= 0:
        raise ValueError("half_life_days must be positive")
    return 2.0 ** (-1.0 / half_life)


def resolve_reputation_volume_config(
    config: Optional[dict] = None,
) -> dict[str, float]:
    """Return validated diminishing-return trust multiplier settings."""
    resolved = {
        key: float((config or {}).get(key, default))
        for key, default in DEFAULT_REPUTATION_VOLUME_CONFIG.items()
    }
    if not all(math.isfinite(value) for value in resolved.values()):
        raise ValueError("reputation_volume values must be finite")
    if resolved["min_multiplier"] < 0:
        raise ValueError("reputation_volume.min_multiplier must be non-negative")
    if resolved["max_multiplier"] < resolved["min_multiplier"]:
        raise ValueError(
            "reputation_volume.max_multiplier must be greater than or equal to "
            "min_multiplier"
        )
    if resolved["half_saturation_orders"] <= 0:
        raise ValueError(
            "reputation_volume.half_saturation_orders must be positive"
        )
    return resolved


def reputation_volume_multiplier(
    rated_order_count: float,
    config: Optional[dict] = None,
) -> float:
    """Map lifetime rating volume to a bounded, saturating trust multiplier.

    ``half_saturation_orders`` is the lifetime count at which half of the gap
    between the minimum and maximum multiplier has been earned.
    """
    count = float(rated_order_count)
    if not math.isfinite(count) or count < 0:
        raise ValueError("rated_order_count must be finite and non-negative")
    resolved = resolve_reputation_volume_config(config)
    minimum = resolved["min_multiplier"]
    maximum = resolved["max_multiplier"]
    half_saturation = resolved["half_saturation_orders"]
    saturation = count / (count + half_saturation)
    return minimum + (maximum - minimum) * saturation


def rebuild_evidence(
    rows: Iterable[tuple[Hashable, str, Optional[int], int]],
    *,
    cutoff_t: int,
    step_hours: int,
    half_life_days: float,
    scores: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> dict[Hashable, tuple[float, float, int]]:
    """Rebuild decayed evidence through an exclusive completed-day cutoff.

    Each row is ``(group_key, current_status, late_t, settled_t)``.  The
    returned tuple is ``(weighted_score_sum, effective_weight, raw_order_count)``.
    Orders in the current incomplete day or after ``cutoff_t`` are excluded.
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    cutoff_hours = int(cutoff_t) * int(step_hours)
    completed_days = cutoff_hours // 24
    if completed_days <= 0:
        return {}
    completed_cutoff_t = (completed_days * 24) // int(step_hours)
    decay = decay_from_half_life(half_life_days)
    out: dict[Hashable, list[float]] = {}
    for group_key, current_status, late_t, settled_t in rows:
        settled_t_i = int(settled_t)
        if settled_t_i < 0 or settled_t_i >= completed_cutoff_t:
            continue
        contribution = outcome_contribution(
            current_status, late_t, scores=scores, weights=weights,
        )
        if contribution is None:
            continue
        event_day = (settled_t_i * int(step_hours)) // 24
        age_days = (completed_days - 1) - event_day
        if age_days < 0:
            continue
        weighted_score, effective_weight = contribution
        factor = decay ** age_days
        slot = out.setdefault(group_key, [0.0, 0.0, 0.0])
        slot[0] += weighted_score * factor
        slot[1] += effective_weight * factor
        slot[2] += 1.0
    return {
        key: (values[0], values[1], int(values[2]))
        for key, values in out.items()
    }
