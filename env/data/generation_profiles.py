"""Catalog generation helpers."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Iterable

import numpy as np
import yaml
from core.rng import derive_rng

# * Match the private-real pipeline spirit without importing that builder.
ELASTICITY_CLIP_MIN = 1.10
ELASTICITY_CLIP_MAX = 6.00
REF_PRICE_CAP_RATIO = 2.0
MAX_RETAIL_MARGIN = 1.0 - 1.0 / REF_PRICE_CAP_RATIO
MIN_RETAIL_MARGIN = 1.0 / ELASTICITY_CLIP_MAX

PRICING_MODEL_MARGIN_CONSISTENT_V1 = "margin_consistent_v1"
PRICING_MODEL_LEGACY_ANCHOR_AT_COST = "legacy_anchor_at_cost"
DEFAULT_PRICING_MODEL = PRICING_MODEL_MARGIN_CONSISTENT_V1

# * Legacy U(1, 50) amplitude, kept for ablation overlays.
LEGACY_BASE_DEMAND_RANGE = (1.0, 50.0)
# * Same min/max skew as legacy, shifted so mean(base) * small_share = 0.52.
CALIBRATED_BASE_DEMAND_RANGE = (0.02, 1.02)
# * 26 paper shop-level orders/day / 50 active listings.
TARGET_LISTING_DAY_DEMAND_AT_REF = 0.52
TARGET_SHOP_DAY_ORDERS_AT_REF = 26.0
TARGET_ACTIVE_LISTINGS = 50

# * Gated risk↔trust post-process. Off by default so the v5 catalog prefix
# * stays bit-identical. Weights are fractions of each configured range span.
RISK_TRUST_COUPLING_KEY = "risk_trust_coupling"
RISK_TRUST_SHOP_RATE_WEIGHT = 1.0
RISK_TRUST_HIST_RATE_WEIGHT = 0.25
RISK_TRUST_SHOP_LOGISTICS_WEIGHT = 0.6


_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_SCENARIO_PATH = os.path.join(_ENV_ROOT, "scenarios", "default.yaml")

_DEFAULT_HOUR_SHAPE = [
    0.18,
    0.12,
    0.10,
    0.10,
    0.14,
    0.24,
    0.42,
    0.65,
    0.82,
    0.96,
    1.05,
    1.10,
    1.00,
    0.92,
    0.88,
    0.94,
    1.05,
    1.18,
    1.30,
    1.22,
    1.05,
    0.78,
    0.52,
    0.32,
]


def load_default_generation_params(path: str | None = None) -> dict[str, Any]:
    """Load generation parameters from a scenario YAML."""
    resolved = path or DEFAULT_SCENARIO_PATH
    with open(resolved, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("generation params YAML must be a mapping")
    return _coerce_generation_params(data)


def generation_params_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build generation params from a full scenario mapping."""
    params = copy.deepcopy(scenario.get("generation_params") or {})
    if "risk_ranges" in scenario:
        params["risk_ranges"] = copy.deepcopy(scenario["risk_ranges"])
    if "supplier_ranges" in scenario:
        params["supplier_ranges"] = copy.deepcopy(scenario["supplier_ranges"])
    return params


def dump_params_yaml(params: dict[str, Any], path: str) -> None:
    """Persist generation params as YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)


def params_to_yaml(params: dict[str, Any]) -> str:
    return yaml.safe_dump(params, allow_unicode=True, sort_keys=False)


def params_from_yaml(text: str) -> dict[str, Any]:
    params = yaml.safe_load(text) or {}
    if not isinstance(params, dict):
        raise ValueError("params YAML must be a mapping")
    return _coerce_generation_params(params)


def translate_category(source_category: str, params: dict[str, Any]) -> str | None:
    mapping = params.get("category_mapping") or {}
    value = mapping.get(source_category)
    return str(value) if value else None


def category_pool(params: dict[str, Any]) -> list[str]:
    return [str(v) for v in (params.get("category_mapping") or {}).values()]


def normalize_supplier_ranges(
    supplier_ranges: dict[str, Any],
    platform_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy supplier ranges without applying merchant promise caps.

    Merchant promises now apply only to shipping time and do not constrain the
    supplier's operational ship/logistics distributions.
    """
    return copy.deepcopy(supplier_ranges)


def rand_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(float(lo), float(hi)))


def resolve_base_demand_range(
    params: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Return the per-product ``base_demand`` uniform sampling range.

    Reads ``base_demand`` from ``params``, or from nested
    ``generation_params`` when a full scenario mapping is passed. A
    missing key uses ``CALIBRATED_BASE_DEMAND_RANGE``.

    Args:
        params: Generation params, a scenario mapping, or ``None``.

    Returns:
        Pair ``(lo, hi)`` consumed by ``rand_range``.

    Raises:
        ValueError: If the configured range is non-finite, ``lo <= 0``,
            or ``hi <= lo``.
    """
    raw = _lookup_base_demand(params)
    if raw is None:
        return CALIBRATED_BASE_DEMAND_RANGE
    return _validate_base_demand_range(raw)


def mean_listing_day_demand_at_ref(
    products: Iterable[Any],
    small_share: float,
) -> float:
    """Return mean listing-day demand at reference price.

    Hourly weights sum to 1, so a day at ``sale_price = ref_price`` with
    lifecycle and rating factors equal to 1 sums to
    ``mean(market_curve) * small_share`` per product.

    Args:
        products: Catalog products that expose ``market_curve``.
        small_share: Shop share of market demand.

    Returns:
        Mean over products of ``mean(market_curve) * small_share``.

    Raises:
        ValueError: If ``products`` is empty.
    """
    catalog = list(products)
    if not catalog:
        raise ValueError("products must be non-empty")
    share = float(small_share)
    per_listing = [float(np.mean(product.market_curve)) * share for product in catalog]
    return float(np.mean(per_listing))


def expected_shop_day_orders_at_ref(
    products: Iterable[Any],
    small_share: float,
    n_listings: int,
) -> float:
    """Return expected shop-level daily orders at reference price.

    Scales ``mean_listing_day_demand_at_ref`` by ``n_listings``. Default
    calibration targets ``TARGET_SHOP_DAY_ORDERS_AT_REF`` when
    ``n_listings`` is ``TARGET_ACTIVE_LISTINGS`` and ``small_share`` is 1.

    Args:
        products: Catalog products that expose ``market_curve``.
        small_share: Shop share of market demand.
        n_listings: Number of simultaneously listed products.

    Returns:
        Expected shop-level orders per day at ``sale_price = ref_price``.

    Raises:
        ValueError: If ``products`` is empty or ``n_listings`` is negative.
    """
    listings = int(n_listings)
    if listings < 0:
        raise ValueError(f"n_listings must be non-negative, got {n_listings!r}")
    return mean_listing_day_demand_at_ref(products, small_share) * float(listings)


def sample_operational_fields(
    rng: np.random.Generator,
    supplier_ranges: dict[str, Any],
) -> dict[str, int]:
    max_quantity = int(rng.integers(*_int_range(supplier_ranges["max_quantity"])))
    quantity = min(int(rng.integers(20, 400)), max_quantity)
    return {
        "quantity": quantity,
        "max_quantity": max_quantity,
        "hourly_increment": int(rng.integers(*_int_range(supplier_ranges["hourly_increment"]))),
        "ship_hours": int(rng.integers(*_int_range(supplier_ranges["ship_hours"]))),
        "logistics_hours": int(rng.integers(*_int_range(supplier_ranges["logistics_hours"]))),
    }


def sample_risk_event_fields(
    rng: np.random.Generator,
    risk_ranges: dict[str, Any],
    supplier_ranges: dict[str, Any],
) -> dict[str, float]:
    return {
        "cancel_rate": rand_range(rng, *risk_ranges["cancel_rate"]),
        "refund_rate": rand_range(rng, *risk_ranges["refund_rate"]),
        "only_refund_rate": rand_range(rng, *risk_ranges["only_refund_rate"]),
        "bad_review_rate": rand_range(rng, *risk_ranges["bad_review_rate"]),
        "timeout_rate": rand_range(rng, *supplier_ranges["timeout_rate"]),
        "price_change_rate": rand_range(rng, *supplier_ranges["price_change_rate"]),
        "supplier_delist_rate": rand_range(rng, *supplier_ranges["supplier_delist_rate"]),
    }


def sample_supplier_profile_maps(
    supplier_names: Iterable[str],
    rng: np.random.Generator,
    supplier_profile_ranges: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    names = list(supplier_names)
    shop_rating_by_sup = {name: rand_range(rng, *supplier_profile_ranges["shop_rating"]) for name in names}
    return_buyer_by_sup = {name: rand_range(rng, *supplier_profile_ranges["return_buyer_rate"]) for name in names}
    age_by_sup = {name: rand_range(rng, *supplier_profile_ranges["supplier_age_years"]) for name in names}
    return shop_rating_by_sup, return_buyer_by_sup, age_by_sup


def sample_product_rating(
    rng: np.random.Generator,
    product_profile_ranges: dict[str, Any],
) -> float:
    return rand_range(rng, *product_profile_ranges["historical_avg_rating"])


def risk_trust_coupling_enabled(params: dict[str, Any] | None) -> bool:
    """Return whether gated risk↔trust post-process is on.

    Missing or empty ``risk_trust_coupling`` is off so older scenarios keep
    the independent v5 risk draws.

    Args:
        params: Generation params or ``None``.

    Returns:
        True only when the flag is an explicit truthy value.
    """
    if not isinstance(params, dict):
        return False
    raw = params.get(RISK_TRUST_COUPLING_KEY, False)
    if raw is None or raw == "":
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def apply_risk_trust_coupling(
    product: Any,
    *,
    risk_ranges: dict[str, Any],
    supplier_ranges: dict[str, Any],
    supplier_profile_ranges: dict[str, Any],
    product_profile_ranges: dict[str, Any],
) -> None:
    """Bias per-product risk and logistics using public trust signals.

    Deterministic post-process: no RNG. Lower ``shop_rating`` raises
    ``refund_rate`` / ``only_refund_rate`` / ``bad_review_rate`` and
    ``logistics_hours``. Higher ``historical_avg_rating`` slightly lowers
    the refund-like rates. Supplier profile fields are not modified.
    Rates stay in ``[0, 1]`` and inside the configured ranges.

    Args:
        product: Catalog product mutated in place.
        risk_ranges: Scenario ``risk_ranges`` used to clamp refund-like rates.
        supplier_ranges: Scenario ``supplier_ranges`` used to clamp hours.
        supplier_profile_ranges: Range used to normalize ``shop_rating``.
        product_profile_ranges: Range used to normalize historical rating.

    Raises:
        ValueError: If a required range is missing or invalid.
    """
    shop_lo, shop_hi = _float_range_pair(supplier_profile_ranges["shop_rating"], "shop_rating")
    hist_lo, hist_hi = _float_range_pair(
        product_profile_ranges["historical_avg_rating"],
        "historical_avg_rating",
    )
    shop_trust = _unit_position(float(product.shop_rating), shop_lo, shop_hi)
    hist_trust = _unit_position(float(product.historical_avg_rating), hist_lo, hist_hi)
    # * Positive shop_risk means a below-midpoint supplier rating.
    shop_risk = 0.5 - shop_trust
    hist_relief = hist_trust - 0.5
    rate_delta = shop_risk * RISK_TRUST_SHOP_RATE_WEIGHT - hist_relief * RISK_TRUST_HIST_RATE_WEIGHT
    for field in ("refund_rate", "only_refund_rate", "bad_review_rate"):
        lo, hi = _float_range_pair(risk_ranges[field], field)
        setattr(
            product,
            field,
            _shift_rate(float(getattr(product, field)), lo, hi, rate_delta),
        )

    hours_lo, hours_hi_excl = _int_range(supplier_ranges["logistics_hours"])
    hours_hi = hours_hi_excl - 1
    hour_span = float(hours_hi - hours_lo)
    delta_hours = shop_risk * RISK_TRUST_SHOP_LOGISTICS_WEIGHT * hour_span
    new_hours = int(round(float(product.logistics_hours) + delta_hours))
    product.logistics_hours = int(_clamp(new_hours, hours_lo, hours_hi))


def _float_range_pair(raw: Any, label: str) -> tuple[float, float]:
    """Parse a ``[lo, hi]`` float range."""
    try:
        sequence = list(raw)
        lo = float(sequence[0])
        hi = float(sequence[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{label} must be a [lo, hi] pair, got {raw!r}") from exc
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"{label} range must be finite, got {raw!r}")
    if hi < lo:
        raise ValueError(f"{label} range must satisfy lo <= hi, got {raw!r}")
    return lo, hi


def _unit_position(value: float, lo: float, hi: float) -> float:
    """Map ``value`` onto ``[0, 1]`` for ``[lo, hi]``; 0.5 when the span is empty."""
    if hi <= lo:
        return 0.5
    return _clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)


def _shift_rate(old: float, lo: float, hi: float, delta_units: float) -> float:
    """Shift ``old`` by ``delta_units`` of ``[lo, hi]``; clamp to the range and ``[0, 1]``."""
    shifted = float(old) + float(delta_units) * (hi - lo)
    return _clamp(_clamp(shifted, lo, hi), 0.0, 1.0)


def sample_elasticity(
    rng: np.random.Generator,
    category: str,
    params: dict[str, Any] | None = None,
    fallback_range: list[float] | tuple[float, float] | None = None,
) -> float:
    params = params or {}
    profile = (params.get("categories") or {}).get(category) or {}
    elasticity = profile.get("elasticity")
    if isinstance(elasticity, dict):
        mean = float(elasticity["mean"])
        jitter = float(elasticity.get("jitter", 0.0))
        lo = float(elasticity.get("min", mean - jitter))
        hi = float(elasticity.get("max", mean + jitter))
        value = mean + rand_range(rng, -jitter, jitter)
        return _clamp(value, lo, hi)
    if fallback_range is not None:
        return rand_range(rng, *fallback_range)
    return 1.3


def sample_retail_margin(
    rng: np.random.Generator,
    category: str,
    params: dict[str, Any] | None = None,
) -> float:
    """Sample a target retail margin for ``category``.

    Draws ``mean + U(-jitter, jitter)``, then clamps to the category
    profile ``min``/``max`` and to
    ``[MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN]``.

    Args:
        rng: Catalog-generation RNG stream.
        category: Simulator category key.
        params: Generation params containing ``categories.<cat>.retail_margin``.

    Returns:
        Clamped retail margin in ``[MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN]``.

    Raises:
        ValueError: If the category has no ``retail_margin`` profile.
    """
    params = params or {}
    profile = (params.get("categories") or {}).get(category) or {}
    retail_margin = profile.get("retail_margin")
    if not isinstance(retail_margin, dict):
        raise ValueError(f"category {category!r} is missing retail_margin profile")
    mean = float(retail_margin["mean"])
    jitter = float(retail_margin.get("jitter", 0.0))
    lo = float(retail_margin.get("min", mean - jitter))
    hi = float(retail_margin.get("max", mean + jitter))
    value = mean + rand_range(rng, -jitter, jitter)
    value = _clamp(value, lo, hi)
    return _clamp(value, MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN)


def cost_and_elasticity_from_margin(
    ref_price: float,
    margin: float,
) -> tuple[float, float]:
    """Derive supplier cost and CES elasticity from a target retail margin.

    Constant-elasticity optimum is ``p* = ε/(ε-1) * cost``. Setting ``p*``
    equal to ``ref_price`` gives ``ε = ref / (ref - cost) = 1 / margin`` and
    ``cost = ref * (1 - margin)``. Incoming ``margin`` is clamped to
    ``[MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN]`` before that derivation so a
    raw draw of ``0.99`` cannot yield ``ε≈1.1`` / a 91% margin. Elasticity
    is then clipped to ``[ELASTICITY_CLIP_MIN, ELASTICITY_CLIP_MAX]`` and
    cost is recomputed so the CES identity still holds. Cost is strictly
    between 0 and ``ref_price``.

    Args:
        ref_price: Consumer reference price and theoretical CES optimum.
        margin: Target retail margin ``(ref - cost) / ref``.

    Returns:
        Tuple of ``(cost, elasticity)``.

    Raises:
        ValueError: If ``ref_price`` is not positive and finite, or ``margin``
            is not finite, or the derived cost is not in ``(0, ref_price)``.
    """
    ref = float(ref_price)
    sampled_margin = float(margin)
    if not math.isfinite(ref) or ref <= 0.0:
        raise ValueError(f"ref_price must be positive and finite, got {ref_price!r}")
    if not math.isfinite(sampled_margin):
        raise ValueError(f"margin must be finite, got {margin!r}")
    # * Clamp first so m=0.99 cannot produce ε≈1.1 / 91% retail margin.
    sampled_margin = _clamp(sampled_margin, MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN)

    cost = ref * (1.0 - sampled_margin)
    # * ε = ref / (ref - cost) is defined only while cost stays in (0, ref).
    if 0.0 < cost < ref:
        raw_elasticity = ref / (ref - cost)
    elif sampled_margin <= 0.0:
        raw_elasticity = ELASTICITY_CLIP_MAX
    else:
        raw_elasticity = ELASTICITY_CLIP_MIN
    elasticity = _clamp(raw_elasticity, ELASTICITY_CLIP_MIN, ELASTICITY_CLIP_MAX)
    # * Recompute cost after the elasticity clip so p* stays at ref_price.
    cost = ref * (1.0 - 1.0 / elasticity)
    if not (0.0 < cost < ref):
        raise ValueError(f"derived cost must be in (0, ref_price), got cost={cost!r} ref_price={ref!r}")
    return float(cost), float(elasticity)


def hourly_dist_for_category(
    category: str,
    rng: np.random.Generator | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    params = params or {}
    profile = (params.get("categories") or {}).get(category) or {}
    hour_shape = _float_array(profile.get("hour_shape", _DEFAULT_HOUR_SHAPE), 24, "hour_shape")
    jitter_range = profile.get("hourly_jitter", params.get("hourly_jitter"))
    if rng is not None and jitter_range:
        hour_shape = hour_shape * rng.uniform(
            float(jitter_range[0]),
            float(jitter_range[1]),
            size=hour_shape.shape,
        )
    total = float(hour_shape.sum())
    if total <= 0:
        raise ValueError(f"hourly_dist for {category!r} must have positive weight")
    return hour_shape / total


def build_hourly_dist_for_categories(
    categories: Iterable[str],
    *,
    seed: int,
    params: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    params = params or {}
    out: dict[str, np.ndarray] = {}
    for category in categories:
        rng = derive_rng(int(seed), "data_gen", "hourly_dist", category)
        out[str(category)] = hourly_dist_for_category(str(category), rng, params)
    return out


def hourly_dist_rows(
    hourly_dist: dict[str, np.ndarray],
) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    for category, weights in hourly_dist.items():
        rows = []
        for hour in range(24):
            rows.append((hour, float(weights[hour])))
        out[category] = rows
    return out


def resample_periodic_curve(values: list[float], target_len: int = 365) -> list[float]:
    if len(values) == target_len:
        return [float(v) for v in values]
    if len(values) <= 1:
        raise ValueError("market curve must contain at least two points")
    source = np.asarray(values, dtype=float)
    source_sum = float(source.sum())
    if not np.isfinite(source_sum) or source_sum <= 0:
        raise ValueError("market curve must have positive finite total")
    x_old = np.arange(len(source) + 1, dtype=float)
    y_old = np.concatenate([source, source[:1]])
    x_new = np.linspace(0.0, float(len(source)), target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, y_old)
    resampled = np.maximum(resampled, 0.0)
    if float(resampled.sum()) <= 0:
        raise ValueError("resampled market curve must have positive total")
    return [float(v) for v in resampled]


def _int_range(values: Any) -> tuple[int, int]:
    lo, hi = values
    lo_i = int(lo)
    hi_i = int(hi)
    if hi_i <= lo_i:
        raise ValueError(f"invalid integer range: {values!r}")
    return lo_i, hi_i


def _float_array(values: Any, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (expected_len,):
        raise ValueError(f"{label} must have length {expected_len}")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{label} must contain finite non-negative values")
    return arr


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _lookup_base_demand(params: dict[str, Any] | None) -> Any:
    """Return the raw ``base_demand`` value, or ``None`` if unset."""
    if not isinstance(params, dict):
        return None
    if "base_demand" in params:
        value = params["base_demand"]
        if value is None or value == "":
            return None
        return value
    nested = params.get("generation_params")
    if isinstance(nested, dict) and "base_demand" in nested:
        value = nested["base_demand"]
        if value is None or value == "":
            return None
        return value
    return None


def _validate_base_demand_range(raw: Any) -> tuple[float, float]:
    """Parse and validate a ``[lo, hi]`` base_demand range."""
    try:
        sequence = list(raw)
    except TypeError as exc:
        raise ValueError(f"base_demand must be a [lo, hi] pair, got {raw!r}") from exc
    if len(sequence) != 2:
        raise ValueError(f"base_demand must be a [lo, hi] pair, got {raw!r}")
    try:
        lo_f = float(sequence[0])
        hi_f = float(sequence[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"base_demand must be a [lo, hi] pair, got {raw!r}") from exc
    if not math.isfinite(lo_f) or not math.isfinite(hi_f):
        raise ValueError(f"base_demand range must be finite, got {raw!r}")
    if lo_f <= 0.0 or hi_f <= lo_f:
        raise ValueError(f"base_demand range must satisfy 0 < lo < hi, got lo={lo_f!r} hi={hi_f!r}")
    return lo_f, hi_f


def _coerce_generation_params(data: dict[str, Any]) -> dict[str, Any]:
    if "generation_params" in data or "risk_ranges" in data or "supplier_ranges" in data:
        return generation_params_from_scenario(data)
    return copy.deepcopy(data)
