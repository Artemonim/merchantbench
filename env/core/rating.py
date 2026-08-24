"""Shop rating math: Beta-Binomial posterior + star bucketing + demand multiplier.

Pure functions only. The simulator owns the per-agent `n_good` / `n_bad`
counters and decay scheduling; this module just turns counts into a score,
a score into stars, and stars into a demand multiplier.

Why Beta-Binomial: real-platform-style rating that
  (a) is volume-aware — 100 good orders beats 5 good orders despite the
      same 100% ratio,
  (b) is small-sample robust — the Beta prior absorbs one-off noise,
  (c) is step_hours invariant — all signals are per-event, not per-step.
"""

from __future__ import annotations

from typing import Iterable

# Canonical mapping from event types in the events log to good/bad rating
# signals. The simulator's _update_ratings folds these into n_good / n_bad
# per step; the rehydrate path replays them off the events table. Keeping
# the sets here means there is exactly one place that decides which
# transitions matter for shop rating.
RATING_GOOD_EVENT_TYPES = frozenset({"order_settled_normal"})
RATING_BAD_EVENT_TYPES = frozenset(
    {
        "order_late",
        "order_settled_refund",
        "order_settled_only_refund",
        "order_settled_bad_review",
        "order_stockout_violation",
    }
)


def posterior_mean(n_good: float, n_bad: float, prior_good: float, prior_bad: float) -> float:
    """Beta-Binomial posterior mean. Returns rating in [0, 1].

    With (n_good=0, n_bad=0) the result is the prior mean
    prior_good / (prior_good + prior_bad). Increasing good_count drives it
    toward 1; increasing bad_count drives it toward 0.
    """
    denom = n_good + n_bad + prior_good + prior_bad
    return (n_good + prior_good) / denom


def stars_from_score(score: float, thresholds: list[float]) -> int:
    """Bucket score to 1..5 stars by ascending thresholds of length 4.

    thresholds = [t1, t2, t3, t4] means:
        score < t1            → 1 star
        t1 <= score < t2      → 2 stars
        t2 <= score < t3      → 3 stars
        t3 <= score < t4      → 4 stars
        score >= t4           → 5 stars
    """
    stars = 1
    for t in thresholds:
        if score >= t:
            stars += 1
        else:
            break
    return stars


def multiplier_from_stars(stars: int, multipliers: list[float]) -> float:
    """Look up demand multiplier; stars is 1-indexed, list is 0-indexed."""
    idx = max(0, min(len(multipliers) - 1, stars - 1))
    return float(multipliers[idx])


def apply_decay(n: float, decay: float) -> float:
    """One-step gentle decay; returns n * decay."""
    return n * decay


def rebuild_counters(
    rating_events: Iterable[tuple[str, str, int]],
    env_t: int,
    decay: float,
    good_event_types: Iterable[str] = RATING_GOOD_EVENT_TYPES,
    bad_event_types: Iterable[str] = RATING_BAD_EVENT_TYPES,
) -> dict[str, tuple[float, float]]:
    """Reconstruct (n_good, n_bad) per agent from the persisted event log.

    `rating_events` yields (agent_id, event_type, t_e). `env_t` is the next
    step the environment will run (i.e. `runs.current_t`); the last
    completed step is therefore `env_t - 1`. An event at step `t_e` was
    decayed once per step in `(t_e, env_t - 1]`, so its contribution at
    rehydrate time is `decay ** (env_t - 1 - t_e)`. This mirrors
    `core.simulator._update_ratings`, which decays first then adds the
    step's events with weight 1.

    Events with `t_e >= env_t` are skipped defensively — a clean stop puts
    `runs.current_t` strictly past every committed event, but we don't
    want a stray future-step row to corrupt the rebuild.
    """
    good_set = frozenset(good_event_types)
    bad_set = frozenset(bad_event_types)
    out: dict[str, list[float]] = {}
    last_completed = env_t - 1
    for agent_id, event_type, t_e in rating_events:
        if t_e > last_completed or t_e < 0:
            continue
        weight = decay ** (last_completed - t_e)
        slot = out.setdefault(agent_id, [0.0, 0.0])
        if event_type in good_set:
            slot[0] += weight
        elif event_type in bad_set:
            slot[1] += weight
    return {aid: (g, b) for aid, (g, b) in out.items()}
