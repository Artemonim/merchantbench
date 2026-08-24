"""Deterministic public-review sampling and buyer-perception math.

Every eligible terminal order contributes to an all-response counterfactual,
while a seeded self-selection model decides whether that experience becomes
public. The v4 shop-rating policy owns this stream and uses the resulting public
rating and count for demand.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Optional

from core import listing_rating as lr_mod
from core import rating as rating_mod
from core.rng import derive_rng

PUBLIC_REVIEW_MODEL = "self_selection_v1"
DEFAULT_PROBABILITY_BY_STAR = (0.30, 0.18, 0.08, 0.06, 0.12)
DEFAULT_DEMAND_CONFIG = {
    "min_trust_multiplier": 0.80,
    "max_trust_multiplier": 1.00,
    "half_saturation_reviews": 20.0,
}


@dataclass(frozen=True)
class PublicReviewEvidence:
    """Lifetime public reviews and their full-response counterfactual."""

    review_score_sum: float = 0.0
    review_count: int = 0
    eligible_score_sum: float = 0.0
    eligible_count: int = 0


def resolve_public_review_config(config: Optional[dict] = None) -> dict:
    """Return validated public-review sampling settings.

    Args:
        config: Optional scenario ``public_reviews`` block.

    Returns:
        A normalized model name and five probabilities indexed by star rating.

    Raises:
        ValueError: If the model or probabilities are invalid.
    """
    source = config or {}
    model = str(source.get("model") or PUBLIC_REVIEW_MODEL)
    if model != PUBLIC_REVIEW_MODEL:
        raise ValueError(f"public_reviews.model must be {PUBLIC_REVIEW_MODEL!r}")
    raw_probabilities = source.get(
        "probability_by_star",
        DEFAULT_PROBABILITY_BY_STAR,
    )
    if not isinstance(raw_probabilities, (list, tuple)):
        raise ValueError("public_reviews.probability_by_star must be a list")
    probabilities = tuple(float(value) for value in raw_probabilities)
    if len(probabilities) != 5:
        raise ValueError("public_reviews.probability_by_star must contain five values")
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError("public_reviews.probability_by_star values must be finite")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("public_reviews.probability_by_star values must be within [0, 1]")
    return {"model": model, "probability_by_star": probabilities}


def resolve_public_review_demand_config(
    config: Optional[dict] = None,
) -> dict[str, float]:
    """Return validated buyer trust and rating-confidence settings."""
    demand_cfg = (config or {}).get("demand") or {}
    if not isinstance(demand_cfg, dict):
        raise ValueError("public_reviews.demand must be a mapping")
    resolved = {key: float(demand_cfg.get(key, default)) for key, default in DEFAULT_DEMAND_CONFIG.items()}
    if not all(math.isfinite(value) for value in resolved.values()):
        raise ValueError("public_reviews.demand values must be finite")
    if resolved["min_trust_multiplier"] < 0:
        raise ValueError("public_reviews.demand.min_trust_multiplier must be non-negative")
    if resolved["max_trust_multiplier"] < resolved["min_trust_multiplier"]:
        raise ValueError(
            "public_reviews.demand.max_trust_multiplier must be greater than or equal to min_trust_multiplier"
        )
    if resolved["half_saturation_reviews"] <= 0:
        raise ValueError("public_reviews.demand.half_saturation_reviews must be positive")
    return resolved


def public_review_demand_factors(
    rating: Optional[float],
    review_count: int,
    *,
    bucket_thresholds: list[float],
    star_multipliers: list[float],
    config: Optional[dict] = None,
) -> dict[str, Optional[float]]:
    """Map public rating evidence to confidence-adjusted buyer demand.

    Rating confidence and seller-volume trust share a saturating evidence
    curve. Confidence shrinks the raw star effect toward neutral, preventing a
    single extreme review from receiving the same weight as an established
    public history.
    """
    count = int(review_count)
    if count < 0:
        raise ValueError("review_count must be non-negative")
    resolved = resolve_public_review_demand_config(config)
    half_saturation = resolved["half_saturation_reviews"]
    confidence = count / (count + half_saturation)
    public_stars: Optional[int] = None
    raw_quality_multiplier = 1.0
    if count > 0:
        if rating is None or not math.isfinite(float(rating)):
            raise ValueError("rating must be finite when review_count is positive")
        public_stars = rating_mod.stars_from_score(
            float(rating),
            bucket_thresholds,
        )
        raw_quality_multiplier = rating_mod.multiplier_from_stars(
            public_stars,
            star_multipliers,
        )
    quality_multiplier = 1.0 + confidence * (raw_quality_multiplier - 1.0)
    trust_multiplier = (
        resolved["min_trust_multiplier"]
        + (resolved["max_trust_multiplier"] - resolved["min_trust_multiplier"]) * confidence
    )
    return {
        "stars": float(public_stars) if public_stars is not None else None,
        "confidence": confidence,
        "raw_quality_multiplier": raw_quality_multiplier,
        "quality_multiplier": quality_multiplier,
        "reputation_multiplier": trust_multiplier,
        "demand_multiplier": quality_multiplier * trust_multiplier,
    }


def star_for_order_outcome(
    current_status: str,
    late_t: Optional[int] = None,
    scores: Optional[dict] = None,
) -> Optional[int]:
    """Map an eligible terminal experience score to a public 1–5 star value."""
    score = lr_mod.score_for_order_outcome(current_status, late_t, scores)
    if score is None:
        return None
    return max(1, min(5, int(math.floor(float(score) + 0.5))))


def leaves_public_review(
    *,
    master_seed: int,
    agent_id: str,
    order_id: Hashable,
    current_status: str,
    stars: int,
    config: Optional[dict] = None,
) -> bool:
    """Return a stable public-review decision for one eligible order.

    Existing ``settled_bad_review`` outcomes are public by definition. Other
    experiences use an independent per-order RNG stream, so enabling this
    observation channel cannot advance demand or anomaly RNG state.
    """
    resolved = resolve_public_review_config(config)
    return _leaves_public_review_resolved(
        master_seed=master_seed,
        agent_id=agent_id,
        order_id=order_id,
        current_status=current_status,
        stars=stars,
        resolved=resolved,
    )


def _leaves_public_review_resolved(
    *,
    master_seed: int,
    agent_id: str,
    order_id: Hashable,
    current_status: str,
    stars: int,
    resolved: dict,
) -> bool:
    """Apply one review decision using prevalidated sampling settings."""
    star_value = int(stars)
    if star_value < 1 or star_value > 5:
        raise ValueError("stars must be within [1, 5]")
    if current_status == "settled_bad_review":
        return True
    probability = resolved["probability_by_star"][star_value - 1]
    rng = derive_rng(
        int(master_seed),
        "public_review",
        resolved["model"],
        agent_id,
        order_id,
    )
    return float(rng.random()) < probability


def rebuild_public_review_evidence(
    rows: Iterable[tuple[Hashable, str, Optional[int]]],
    *,
    master_seed: int,
    agent_id: str,
    config: Optional[dict] = None,
    scores: Optional[dict] = None,
) -> PublicReviewEvidence:
    """Rebuild lifetime sampled reviews and the all-response benchmark.

    Args:
        rows: ``(order_id, current_status, late_t)`` terminal-order facts.
        master_seed: Scenario seed used for independent per-order sampling.
        agent_id: Merchant identity included in RNG derivation.
        config: Optional public-review sampling configuration.
        scores: Optional terminal-outcome score overrides.

    Returns:
        Lifetime sampled and eligible score/count aggregates.
    """
    resolved = resolve_public_review_config(config)
    review_score_sum = 0.0
    review_count = 0
    eligible_score_sum = 0.0
    eligible_count = 0
    for order_id, current_status, late_t in rows:
        stars = star_for_order_outcome(current_status, late_t, scores)
        if stars is None:
            continue
        eligible_score_sum += float(stars)
        eligible_count += 1
        if _leaves_public_review_resolved(
            master_seed=master_seed,
            agent_id=agent_id,
            order_id=order_id,
            current_status=current_status,
            stars=stars,
            resolved=resolved,
        ):
            review_score_sum += float(stars)
            review_count += 1
    return PublicReviewEvidence(
        review_score_sum=review_score_sum,
        review_count=review_count,
        eligible_score_sum=eligible_score_sum,
        eligible_count=eligible_count,
    )
