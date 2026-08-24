"""Unit tests for core.rating — Beta-Binomial posterior + bucketing + decay."""

import pytest
from core import rating

_DEFAULT_THRESHOLDS = [0.5, 0.7, 0.85, 0.95]
_DEFAULT_MULTIPLIERS = [0.4, 0.7, 1.0, 1.2, 1.5]
_PRIOR_GOOD = 500.0
_PRIOR_BAD = 50.0


def test_prior_alone_gives_4star_start():
    """With (n_good=0, n_bad=0) and prior (500, 50), a new shop starts at
    500/550 = 0.909, which buckets to 4 stars (multiplier 1.2×)."""
    score = rating.posterior_mean(0.0, 0.0, _PRIOR_GOOD, _PRIOR_BAD)
    assert abs(score - 500 / 550) < 1e-9
    stars = rating.stars_from_score(score, _DEFAULT_THRESHOLDS)
    assert stars == 4
    mult = rating.multiplier_from_stars(stars, _DEFAULT_MULTIPLIERS)
    assert mult == 1.2


def test_volume_aware_more_orders_higher_confidence():
    """Same 100% good ratio: more orders should score strictly higher because
    the prior gets diluted more. With the larger (500, 50) prior it takes more
    observations to reach 5 stars."""
    low_vol = rating.posterior_mean(5.0, 0.0, _PRIOR_GOOD, _PRIOR_BAD)
    high_vol = rating.posterior_mean(1000.0, 0.0, _PRIOR_GOOD, _PRIOR_BAD)
    assert high_vol > low_vol
    # The 5-order shop is still close to prior (4 stars); the 1000-order shop
    # crosses into 5-star territory.
    assert rating.stars_from_score(low_vol, _DEFAULT_THRESHOLDS) == 4
    assert rating.stars_from_score(high_vol, _DEFAULT_THRESHOLDS) == 5


def test_posterior_monotone_with_good_count():
    """Adding a good event must strictly increase rating; adding a bad event
    must strictly decrease it."""
    base = rating.posterior_mean(10.0, 5.0, _PRIOR_GOOD, _PRIOR_BAD)
    plus_good = rating.posterior_mean(11.0, 5.0, _PRIOR_GOOD, _PRIOR_BAD)
    plus_bad = rating.posterior_mean(10.0, 6.0, _PRIOR_GOOD, _PRIOR_BAD)
    assert plus_good > base > plus_bad


def test_decay_preserves_ratio():
    """Applying the same decay to both counts must keep the posterior ratio
    in [0, 1] valid and approaching the prior as counts shrink."""
    n_good = rating.apply_decay(50.0, 0.999)
    n_bad = rating.apply_decay(10.0, 0.999)
    assert n_good == pytest.approx(50.0 * 0.999)
    assert n_bad == pytest.approx(10.0 * 0.999)
    # After many decay applications, both → 0, posterior → prior mean.
    for _ in range(20_000):
        n_good = rating.apply_decay(n_good, 0.999)
        n_bad = rating.apply_decay(n_bad, 0.999)
    score = rating.posterior_mean(n_good, n_bad, _PRIOR_GOOD, _PRIOR_BAD)
    prior_mean = _PRIOR_GOOD / (_PRIOR_GOOD + _PRIOR_BAD)
    assert abs(score - prior_mean) < 1e-6


def test_bucket_thresholds_and_multiplier_table():
    """Spot-check star bucketing at the boundaries and the multiplier lookup."""
    th = [0.5, 0.7, 0.85, 0.95]
    # Below first threshold → 1 star
    assert rating.stars_from_score(0.0, th) == 1
    assert rating.stars_from_score(0.49, th) == 1
    # Exact threshold belongs to the higher bucket
    assert rating.stars_from_score(0.5, th) == 2
    assert rating.stars_from_score(0.7, th) == 3
    assert rating.stars_from_score(0.85, th) == 4
    assert rating.stars_from_score(0.95, th) == 5
    assert rating.stars_from_score(1.0, th) == 5
    # Multiplier lookup
    mults = [0.4, 0.7, 1.0, 1.2, 1.5]
    for s, m in zip([1, 2, 3, 4, 5], mults):
        assert rating.multiplier_from_stars(s, mults) == m
    # Out-of-range stars clamp to the table.
    assert rating.multiplier_from_stars(0, mults) == 0.4
    assert rating.multiplier_from_stars(99, mults) == 1.5


def test_rebuild_counters_empty_events():
    out = rating.rebuild_counters([], env_t=10, decay=0.999)
    assert out == {}


def test_rebuild_counters_single_good_event_decay_weighted():
    # One settled_normal at step 0; current_t = 10 means last completed
    # step = 9 → exponent = 9 - 0 = 9 decays applied.
    events = [("agent_0", "order_settled_normal", 0)]
    out = rating.rebuild_counters(events, env_t=10, decay=0.999)
    assert "agent_0" in out
    n_good, n_bad = out["agent_0"]
    assert n_good == pytest.approx(0.999**9)
    assert n_bad == 0.0


def test_rebuild_counters_event_at_last_completed_step_no_decay():
    # Event fires at the last completed step (env_t - 1 = 4) → weight is
    # decay**0 = 1.0, matching the live update where a step's events are
    # added after that step's decay.
    events = [("agent_0", "order_settled_normal", 4)]
    out = rating.rebuild_counters(events, env_t=5, decay=0.5)
    assert out["agent_0"] == (1.0, 0.0)


def test_rebuild_counters_decay_one_is_plain_count():
    events = [
        ("agent_0", "order_settled_normal", 1),
        ("agent_0", "order_settled_normal", 2),
        ("agent_0", "order_late", 3),
        ("agent_0", "order_settled_refund", 4),
    ]
    out = rating.rebuild_counters(events, env_t=10, decay=1.0)
    n_good, n_bad = out["agent_0"]
    assert n_good == 2.0
    assert n_bad == 2.0


def test_rebuild_counters_ignores_cancelled_orders():
    events = [
        ("agent_0", "order_settled_normal", 1),
        ("agent_0", "order_cancelled", 2),
    ]
    out = rating.rebuild_counters(events, env_t=5, decay=1.0)
    assert out["agent_0"] == (1.0, 0.0)


def test_rebuild_counters_counts_stockout_as_bad_signal():
    events = [
        ("agent_0", "order_settled_normal", 1),
        ("agent_0", "order_stockout_violation", 2),
    ]
    out = rating.rebuild_counters(events, env_t=5, decay=1.0)
    assert out["agent_0"] == (1.0, 1.0)


def test_rebuild_counters_groups_by_agent():
    events = [
        ("agent_0", "order_settled_normal", 0),
        ("agent_1", "order_settled_refund", 0),
        ("agent_0", "order_late", 1),
        ("agent_1", "order_settled_normal", 2),
    ]
    out = rating.rebuild_counters(events, env_t=3, decay=0.9)
    # agent_0: good at t=0 (decay**2), bad at t=1 (decay**1)
    a0_good, a0_bad = out["agent_0"]
    assert a0_good == pytest.approx(0.9**2)
    assert a0_bad == pytest.approx(0.9)
    # agent_1: bad at t=0 (decay**2), good at t=2 (decay**0)
    a1_good, a1_bad = out["agent_1"]
    assert a1_good == pytest.approx(1.0)
    assert a1_bad == pytest.approx(0.9**2)


def test_rebuild_counters_skips_future_and_negative_steps():
    # env_t=5 → only t in [0, 4] should count. t=5 is "future" relative
    # to runs.current_t (defensive); t=-1 is negative.
    events = [
        ("agent_0", "order_settled_normal", 5),  # future, skip
        ("agent_0", "order_settled_normal", -1),  # negative, skip
        ("agent_0", "order_settled_normal", 4),  # valid: weight 1.0
    ]
    out = rating.rebuild_counters(events, env_t=5, decay=0.5)
    assert out["agent_0"] == (1.0, 0.0)


def test_rebuild_counters_unknown_event_type_ignored():
    # An event type that's not in good or bad sets is silently dropped.
    events = [
        ("agent_0", "order_settled_normal", 1),
        ("agent_0", "order_shipped", 2),  # not a rating signal
    ]
    out = rating.rebuild_counters(events, env_t=5, decay=1.0)
    assert out["agent_0"] == (1.0, 0.0)


def test_rebuild_counters_matches_live_update_for_two_steps():
    # End-to-end equivalence between rebuild and live _update_ratings logic:
    # simulate two steps by hand and check we get the same answer.
    decay = 0.9
    # Step 0: one good event.
    n_good, n_bad = 0.0, 0.0
    n_good *= decay
    n_bad *= decay  # decay first (no-op on zeros)
    n_good += 1.0  # good event at t=0
    # Step 1: one bad event.
    n_good *= decay
    n_bad *= decay
    n_bad += 1.0  # bad event at t=1
    # After step 1 completes, current_t = 2 → env_t = 2.
    live_good, live_bad = n_good, n_bad
    # Now rebuild from the same event log:
    events = [
        ("agent_0", "order_settled_normal", 0),
        ("agent_0", "order_late", 1),
    ]
    rebuilt = rating.rebuild_counters(events, env_t=2, decay=decay)
    rebuilt_good, rebuilt_bad = rebuilt["agent_0"]
    assert rebuilt_good == pytest.approx(live_good)
    assert rebuilt_bad == pytest.approx(live_bad)
