"""Hosted-evaluation result metric.

Headline metric: final net_assets. The result intentionally reports the
env's wealth metric directly instead of converting it into a separate
derived metric.

Inputs come from the env's merchant-section endpoint
(/runs/<run_id>/agents/<aid>/sections/merchant). We pull the LAST point
of the `net_assets` time series rather than the live cash dict because
net_assets is the env's authoritative wealth number. It is recorded as
balance + deposit_pool + in_transit + receivable; fines have already
reduced balance/deposit_pool when applied, so cumulative_fine is not
subtracted again.
"""

from __future__ import annotations

from typing import Any, Optional


def _last(series: list[list]) -> Optional[float]:
    """Return the last value of a [[t, v], ...] series, or None if empty."""
    if not series:
        return None
    last = series[-1]
    if len(last) < 2:
        return None
    try:
        return float(last[1])
    except (TypeError, ValueError):
        return None


def compute(merchant_section: dict) -> dict[str, Any]:
    """Compute the hosted-eval result metrics from a merchant-section
    payload.

    Returns a dict with:
      - score: float — compatibility alias for final_net_assets.
      - final_net_assets, net_profit, shop_rating_mean: raw env values
                for transparency on the leaderboard page.
      - is_alive, died_at_t: agent survival flag.
      - n_steps: number of metric points logged (sanity check that the
                run actually executed).
    """
    series = merchant_section.get("series") or {}
    net = _last(series.get("net_assets") or [])
    profit = _last(series.get("cum_net_profit") or [])
    rating_mean = _last(series.get("shop_rating_mean") or [])
    rating = rating_mean
    if rating is None:
        rating = _last(series.get("shop_rating_score") or [])

    if net is None:
        # Fall back to live cash dict — this happens when net_assets
        # series wasn't populated (very early failure).
        cash = merchant_section.get("cash") or {}
        net = (
            float(cash.get("balance", 0))
            + float(cash.get("deposit_pool", 0))
            + float(cash.get("in_transit", 0))
            + float(cash.get("receivable", 0))
        )

    return {
        "score": round(float(net), 2),
        "final_net_assets": round(float(net), 2),
        "net_profit": round(float(profit), 2) if profit is not None else None,
        "shop_rating_mean": (round(float(rating_mean), 4) if rating_mean is not None else None),
        "shop_rating_score": round(float(rating), 4) if rating is not None else None,
        "shop_rating_scale": ("1-5" if rating_mean is not None else "0-1" if rating is not None else None),
        "is_alive": bool(merchant_section.get("is_alive", True)),
        "died_at_t": merchant_section.get("died_at_t"),
        "n_steps": len((series.get("net_assets") or [])),
    }
