"""Catalog generation helpers."""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import numpy as np
import yaml

from core.rng import derive_rng


_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_SCENARIO_PATH = os.path.join(_ENV_ROOT, "scenarios", "default.yaml")

_DEFAULT_HOUR_SHAPE = [
    0.18, 0.12, 0.10, 0.10, 0.14, 0.24, 0.42, 0.65,
    0.82, 0.96, 1.05, 1.10, 1.00, 0.92, 0.88, 0.94,
    1.05, 1.18, 1.30, 1.22, 1.05, 0.78, 0.52, 0.32,
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
    shop_rating_by_sup = {
        name: rand_range(rng, *supplier_profile_ranges["shop_rating"])
        for name in names
    }
    return_buyer_by_sup = {
        name: rand_range(rng, *supplier_profile_ranges["return_buyer_rate"])
        for name in names
    }
    age_by_sup = {
        name: rand_range(rng, *supplier_profile_ranges["supplier_age_years"])
        for name in names
    }
    return shop_rating_by_sup, return_buyer_by_sup, age_by_sup


def sample_product_rating(
    rng: np.random.Generator,
    product_profile_ranges: dict[str, Any],
) -> float:
    return rand_range(rng, *product_profile_ranges["historical_avg_rating"])


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


def _coerce_generation_params(data: dict[str, Any]) -> dict[str, Any]:
    if "generation_params" in data or "risk_ranges" in data or "supplier_ranges" in data:
        return generation_params_from_scenario(data)
    return copy.deepcopy(data)
