import pytest

from core import public_reviews as reviews


@pytest.mark.parametrize(
    ("status", "late_t", "expected"),
    [
        ("settled_normal", None, 5),
        ("settled_normal", 7, 3),
        ("settled_refund", None, 2),
        ("settled_only_refund", None, 2),
        ("settled_bad_review", None, 1),
        ("stockout", None, 1),
        ("cancelled", None, None),
        ("insufficient_balance", None, None),
    ],
)
def test_star_for_order_outcome_uses_nearest_discrete_star(
    status, late_t, expected,
):
    assert reviews.star_for_order_outcome(status, late_t) == expected


def test_rebuild_tracks_forced_bad_review_and_all_response_counterfactual():
    rows = [
        ("normal", "settled_normal", None),
        ("late", "settled_normal", 4),
        ("refund", "settled_refund", None),
        ("bad", "settled_bad_review", None),
        ("stockout", "stockout", None),
        ("cancelled", "cancelled", None),
        ("cash", "insufficient_balance", None),
    ]
    evidence = reviews.rebuild_public_review_evidence(
        rows,
        master_seed=42,
        agent_id="agent_0",
        config={"probability_by_star": [0, 0, 0, 0, 0]},
    )

    assert evidence.review_score_sum == 1.0
    assert evidence.review_count == 1
    assert evidence.eligible_score_sum == 12.0
    assert evidence.eligible_count == 5


def test_rebuild_with_full_response_preserves_every_eligible_experience():
    evidence = reviews.rebuild_public_review_evidence(
        [
            ("normal", "settled_normal", None),
            ("late", "settled_normal", 4),
            ("refund", "settled_refund", None),
            ("bad", "settled_bad_review", None),
            ("stockout", "stockout", None),
        ],
        master_seed=42,
        agent_id="agent_0",
        config={"probability_by_star": [1, 1, 1, 1, 1]},
    )

    assert evidence.review_score_sum == evidence.eligible_score_sum == 12.0
    assert evidence.review_count == evidence.eligible_count == 5


def test_review_sampling_is_stable_per_seed_agent_and_order():
    kwargs = {
        "master_seed": 42,
        "agent_id": "agent_0",
        "order_id": "agent_0-p1-t4-0",
        "current_status": "settled_normal",
        "stars": 5,
        "config": {"probability_by_star": [0.5] * 5},
    }

    assert reviews.leaves_public_review(**kwargs) == reviews.leaves_public_review(
        **kwargs,
    )
    decisions = {
        reviews.leaves_public_review(
            **{**kwargs, "order_id": f"order-{index}"},
        )
        for index in range(20)
    }
    assert decisions == {False, True}


def test_default_sampling_has_extremity_bias_and_negative_asymmetry():
    probabilities = reviews.resolve_public_review_config()[
        "probability_by_star"
    ]

    assert probabilities[0] > probabilities[4] > probabilities[2]
    assert probabilities[4] > probabilities[3]


def test_v4_demand_starts_with_neutral_quality_and_cold_start_trust():
    factors = reviews.public_review_demand_factors(
        None,
        0,
        bucket_thresholds=[2.5, 3.3, 3.8, 4.2],
        star_multipliers=[0.1, 0.35, 0.8, 1.0, 1.12],
    )

    assert factors == {
        "stars": None,
        "confidence": 0.0,
        "raw_quality_multiplier": 1.0,
        "quality_multiplier": 1.0,
        "reputation_multiplier": 0.8,
        "demand_multiplier": 0.8,
    }


def test_v4_confidence_shrinks_single_extreme_reviews_toward_neutral():
    common = {
        "review_count": 1,
        "bucket_thresholds": [2.5, 3.3, 3.8, 4.2],
        "star_multipliers": [0.1, 0.35, 0.8, 1.0, 1.12],
    }

    positive = reviews.public_review_demand_factors(5.0, **common)
    negative = reviews.public_review_demand_factors(1.0, **common)

    confidence = 1 / 21
    trust = 0.8 + 0.2 * confidence
    assert positive["confidence"] == pytest.approx(confidence)
    assert positive["quality_multiplier"] == pytest.approx(
        1.0 + 0.12 * confidence
    )
    assert negative["quality_multiplier"] == pytest.approx(
        1.0 - 0.9 * confidence
    )
    assert positive["reputation_multiplier"] == pytest.approx(trust)
    assert negative["reputation_multiplier"] == pytest.approx(trust)


def test_v4_review_confidence_has_configured_half_saturation():
    factors = reviews.public_review_demand_factors(
        5.0,
        20,
        bucket_thresholds=[2.5, 3.3, 3.8, 4.2],
        star_multipliers=[0.1, 0.35, 0.8, 1.0, 1.12],
    )

    assert factors["confidence"] == pytest.approx(0.5)
    assert factors["quality_multiplier"] == pytest.approx(1.06)
    assert factors["reputation_multiplier"] == pytest.approx(0.9)
    assert factors["demand_multiplier"] == pytest.approx(0.954)


@pytest.mark.parametrize(
    "config",
    [
        {"model": "unknown"},
        {"probability_by_star": "0.1,0.2"},
        {"probability_by_star": [0.1] * 4},
        {"probability_by_star": [0.1, 0.1, float("nan"), 0.1, 0.1]},
        {"probability_by_star": [0.1, 0.1, 1.1, 0.1, 0.1]},
    ],
)
def test_public_review_config_rejects_invalid_settings(config):
    with pytest.raises(ValueError):
        reviews.resolve_public_review_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"demand": "invalid"},
        {"demand": {"min_trust_multiplier": -0.1}},
        {
            "demand": {
                "min_trust_multiplier": 1.0,
                "max_trust_multiplier": 0.9,
            },
        },
        {"demand": {"half_saturation_reviews": 0}},
    ],
)
def test_public_review_demand_config_rejects_invalid_settings(config):
    with pytest.raises(ValueError):
        reviews.resolve_public_review_demand_config(config)


def test_review_decision_rejects_invalid_star():
    with pytest.raises(ValueError, match="stars"):
        reviews.leaves_public_review(
            master_seed=42,
            agent_id="agent_0",
            order_id="order",
            current_status="settled_normal",
            stars=0,
        )
