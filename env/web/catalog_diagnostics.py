"""Dashboard-only catalog diagnostics for the supplier pool.

This module intentionally reads hidden product fields such as market_curve and
risk rates. It is wired only into dashboard routes, not agent tools.
"""
from __future__ import annotations

import math
import json
import os
import tempfile
from collections import Counter, defaultdict
from typing import Any

from core.entities import Product
from core.simulator import Environment


ORDER_RISK_COMPONENTS = ("cancel_rate", "refund_rate", "only_refund_rate", "bad_review_rate")
SUPPLY_RISK_COMPONENTS = ("timeout_rate", "price_change_rate", "supplier_delist_rate")
RISK_COMPONENTS = ORDER_RISK_COMPONENTS + SUPPLY_RISK_COMPONENTS
CATALOG_DIAGNOSTICS_SCHEMA_VERSION = 1
MATERIALIZED_SAMPLE_SIZE = 2000
MATERIALIZED_TOP_N = 100
CATALOG_DIAGNOSTICS_FILENAME = "catalog_diagnostics.json"
CATALOG_DIAGNOSTICS_STATUS_FILENAME = "catalog_diagnostics_status.json"
SUPPLIER_RISK_PERIOD_STEPS = 24
_CATALOG_DIAGNOSTICS_STATUSES = {"pending", "ready", "failed"}


def build_catalog_diagnostics_artifact(
    env: Environment,
    *,
    run_meta: dict[str, Any] | None = None,
    legacy_initial_quantity_fallback: bool = False,
) -> dict[str, Any]:
    products = _products(env)
    categories = sorted({p.category for p in products})
    source_meta = {
        key: (run_meta or {}).get(key)
        for key in (
            "run_id",
            "started_at",
            "difficulty_rate",
            "data_source",
            "dataset_id",
            "dataset_rows",
            "dataset_sha256",
        )
        if (run_meta or {}).get(key) is not None
    }
    return {
        "schema_version": CATALOG_DIAGNOSTICS_SCHEMA_VERSION,
        "basis": "initial_catalog",
        "parameters": {
            "sample_size": MATERIALIZED_SAMPLE_SIZE,
            "top_n": MATERIALIZED_TOP_N,
            "refund_penalty_amount": _penalty_amount(env, "refund"),
            "bad_review_penalty_amount": _penalty_amount(env, "bad_review"),
            "step_hours": int((env.scenario.get("run") or {}).get("step_hours", 1)),
            "supplier_risk_period_steps": SUPPLIER_RISK_PERIOD_STEPS,
        },
        "metadata": {
            **source_meta,
            "legacy_initial_quantity_fallback": bool(legacy_initial_quantity_fallback),
        },
        "diagnostics": build_catalog_diagnostics(
            env,
            run_meta=run_meta,
            sample_size=MATERIALIZED_SAMPLE_SIZE,
            top_n=MATERIALIZED_TOP_N,
        ),
        "category_bands": {
            category: _category_curve_band(
                [product for product in products if product.category == category]
            )
            for category in categories
        },
    }


def write_catalog_diagnostics_artifact(path: str, artifact: dict[str, Any]) -> None:
    _write_json_atomic(path, artifact, CATALOG_DIAGNOSTICS_FILENAME)


def write_catalog_diagnostics_status(
    path: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if status not in _CATALOG_DIAGNOSTICS_STATUSES:
        raise ValueError(f"unknown catalog diagnostics status: {status!r}")
    payload = {"status": status}
    if error:
        payload["error"] = str(error)[:500]
    _write_json_atomic(path, payload, CATALOG_DIAGNOSTICS_STATUS_FILENAME)


def read_catalog_diagnostics_status(path: str) -> dict[str, str] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") not in _CATALOG_DIAGNOSTICS_STATUSES:
        return None
    result = {"status": str(payload["status"])}
    if payload.get("error"):
        result["error"] = str(payload["error"])
    return result


def _write_json_atomic(path: str, payload: dict[str, Any], filename: str) -> None:
    directory = os.path.dirname(path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def read_catalog_diagnostics_artifact(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        return None
    if (
        artifact.get("schema_version") != CATALOG_DIAGNOSTICS_SCHEMA_VERSION
        or artifact.get("basis") != "initial_catalog"
        or "diagnostics" not in artifact
    ):
        return None
    return artifact


def diagnostics_from_artifact(
    artifact: dict[str, Any], *, sample_size: int, top_n: int
) -> dict[str, Any]:
    payload = dict(artifact["diagnostics"])
    candidates = list(payload.get("scatter_points") or [])
    payload["scatter_points"] = _scatter_points(
        candidates, min(sample_size, len(candidates))
    )
    payload["outliers"] = {
        key: list(rows)[:top_n]
        for key, rows in (payload.get("outliers") or {}).items()
    }
    return payload


def build_materialized_product_diagnostics(
    product: Product,
    *,
    small_share: float,
    category_band: dict[str, list[list[float]]],
    refund_penalty: float = 8.0,
    bad_review_penalty: float = 5.0,
) -> dict[str, Any]:
    metric = _metrics(
        [product],
        small_share,
        refund_penalty=refund_penalty,
        bad_review_penalty=bad_review_penalty,
    )[0]
    return {
        "product": _product_summary(metric),
        "market_curve": _series365(product.market_curve),
        "category_band": category_band,
    }


def build_catalog_diagnostics(
    env: Environment,
    *,
    run_meta: dict[str, Any] | None = None,
    sample_size: int = 1200,
    top_n: int = 20,
) -> dict[str, Any]:
    products = _products(env)
    small_share = _small_share(env)
    metrics = _metrics(
        products,
        small_share,
        refund_penalty=_penalty_amount(env, "refund"),
        bad_review_penalty=_penalty_amount(env, "bad_review"),
    )
    by_category = _group_metrics(metrics)
    categories = sorted(by_category)
    hourly_dist = getattr(env, "hourly_dist", {}) or {}

    return {
        "kpis": _kpis(env, products, categories, hourly_dist, run_meta),
        "category_summary": [
            _category_summary(category, by_category[category])
            for category in categories
        ],
        "distributions": {
            "price": _histogram_by_category(
                categories, by_category, metrics, "price", scale="log", bin_count=16
            ),
            "demand365": _histogram_by_category(
                categories, by_category, metrics, "demand365",
                scale="log",
                bin_count=16,
            ),
            "profit365": _profit_histograms_by_metric(categories, by_category, metrics),
            "order_anomaly": _risk_histograms_by_event(
                categories, by_category, metrics, "order_risk", ORDER_RISK_COMPONENTS
            ),
            "supplier_anomaly": _risk_histograms_by_event(
                categories,
                by_category,
                metrics,
                "supply_risk",
                SUPPLY_RISK_COMPONENTS,
                period_steps=SUPPLIER_RISK_PERIOD_STEPS,
            ),
        },
        "hourly_heatmap": _hourly_heatmap(categories, hourly_dist),
        "market_series": _market_series(products, small_share),
        "risk_by_category": _risk_by_category(categories, by_category),
        "scatter_points": _scatter_points(metrics, max(1, sample_size)),
        "outliers": _outliers(metrics, max(1, top_n)),
    }


def build_product_diagnostics(env: Environment, product_id: str) -> dict[str, Any] | None:
    product = env.products.get(product_id)
    if product is None:
        return None

    products = _products(env)
    small_share = _small_share(env)
    metric_by_id = {m["product_id"]: m for m in _metrics(
        products,
        small_share,
        refund_penalty=_penalty_amount(env, "refund"),
        bad_review_penalty=_penalty_amount(env, "bad_review"),
    )}
    metric = metric_by_id[product_id]
    category_products = [p for p in products if p.category == product.category]

    return {
        "product": _product_summary(metric),
        "market_curve": _series365(product.market_curve),
        "category_band": _category_curve_band(category_products),
    }


def _products(env: Environment) -> list[Product]:
    return sorted(env.products.values(), key=lambda p: (p.category, p.product_id))


def _small_share(env: Environment) -> float:
    return float((env.scenario.get("data") or {}).get("small_share", 1.0))


def _penalty_amount(env: Environment, kind: str) -> float:
    rules = env.scenario.get("platform_rules") or {}
    amount_key = f"{kind}_penalty_amount"
    if amount_key in rules:
        return float(rules[amount_key])
    return 0.0


def _metrics(products: list[Product], small_share: float, *,
             refund_penalty: float = 8.0,
             bad_review_penalty: float = 5.0) -> list[dict[str, Any]]:
    raw_rows = []
    for p in products:
        curve_sum = float(sum(p.market_curve))
        curve365_sum = float(sum(p.market_curve[:365]))
        curve_mean = curve_sum / max(1, len(p.market_curve))
        curve_peak = float(max(p.market_curve)) if p.market_curve else 0.0
        peak_to_mean = curve_peak / curve_mean if curve_mean > 0 else 0.0
        arrival_hours = float(p.supplier_ship_hours + p.logistics_hours)
        risk_values = {key: float(getattr(p, key)) for key in RISK_COMPONENTS}
        order_risk = sum(risk_values[key] for key in ORDER_RISK_COMPONENTS)
        supply_risk = sum(risk_values[key] for key in SUPPLY_RISK_COMPONENTS)
        risk_sum = order_risk
        inventory_pressure = 1.0 - min(1.0, max(0.0, p.quantity / max(1, p.max_quantity)))
        margin_ratio = (p.ref_price - p.price) / p.ref_price if p.ref_price else 0.0
        opportunity_gmv365 = curve_sum * small_share * p.ref_price
        demand365 = curve365_sum * small_share
        gross_profit365 = curve365_sum * small_share * (p.ref_price - p.price)
        cum_fine365 = (
            curve365_sum
            * small_share
            * (
                risk_values["refund_rate"] * refund_penalty
                + risk_values["bad_review_rate"] * bad_review_penalty
            )
        )
        net_profit365 = (
            curve365_sum
            * small_share
            * _expected_unit_profit_at_ref(
                float(p.price),
                float(p.ref_price),
                risk_values,
                refund_penalty=refund_penalty,
                bad_review_penalty=bad_review_penalty,
            )
        )
        row = {
            "product_id": p.product_id,
            "name": p.name,
            "category": p.category,
            "supplier_id": p.supplier_id,
            "supplier_name": p.supplier_name,
            "price": float(p.price),
            "ref_price": float(p.ref_price),
            "margin_ratio": margin_ratio,
            "quantity": int(p.quantity),
            "max_quantity": int(p.max_quantity),
            "arrival_hours": arrival_hours,
            "curve_sum": curve_sum,
            "demand365": demand365,
            "curve_peak": curve_peak,
            "peak_to_mean": peak_to_mean,
            "opportunity_gmv365": opportunity_gmv365,
            "gross_profit365": gross_profit365,
            "cum_fine365": cum_fine365,
            "net_profit365": net_profit365,
            "order_risk": order_risk,
            "supply_risk": supply_risk,
            "risk_sum": risk_sum,
            **risk_values,
            "inventory_pressure": inventory_pressure,
            "elasticity": float(p.elasticity),
        }
        raw_rows.append(row)
    return raw_rows


def _group_metrics(metrics: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[str(row["category"])].append(row)
    return grouped


def _kpis(
    env: Environment,
    products: list[Product],
    categories: list[str],
    hourly_dist: dict[str, Any],
    run_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    data_cfg = env.scenario.get("data") or {}
    meta = run_meta or {}
    source_meta = {
        key: meta.get(key)
        for key in ("data_source", "dataset_id", "dataset_rows", "dataset_sha256")
        if meta.get(key) is not None
    }
    return {
        "product_count": len(products),
        "supplier_count": len({p.supplier_id for p in products}),
        "category_count": len(categories),
        "hourly_dist_category_count": len(hourly_dist),
        "small_share": _small_share(env),
        "data_source": data_cfg.get("source", source_meta.get("data_source", "synthetic")),
        "source_meta": source_meta,
    }


def _category_summary(category: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    suppliers = Counter(str(r["supplier_id"]) for r in rows)
    n = len(rows)
    return {
        "category": category,
        "product_count": n,
        "supplier_count": len(suppliers),
        "top_supplier_share": _r((max(suppliers.values()) / n) if n else 0.0),
        "demand365": _r(sum(float(r["demand365"]) for r in rows)),
        "opportunity_gmv365": _r(sum(float(r["opportunity_gmv365"]) for r in rows)),
        "median_risk_sum": _r(_quantile([float(r["risk_sum"]) for r in rows], 0.50)),
        "p90_risk_sum": _r(_quantile([float(r["risk_sum"]) for r in rows], 0.90)),
        "median_order_risk": _r(_quantile([float(r["order_risk"]) for r in rows], 0.50)),
        "median_supply_risk": _r(_quantile([float(r["supply_risk"]) for r in rows], 0.50)),
    }


def _histogram_by_category(
    categories: list[str],
    by_category: dict[str, list[dict[str, Any]]],
    metrics: list[dict[str, Any]],
    field: str,
    *,
    zero_bucket: bool = False,
    scale: str = "linear",
    bin_count: int = 24,
) -> dict[str, Any]:
    return {
        "field": field,
        "scale": scale,
        "categories": ["All", *categories],
        "all": _histogram(
            [float(row[field]) for row in metrics],
            zero_bucket=zero_bucket,
            scale=scale,
            bin_count=bin_count,
        ),
        "by_category": {
            category: _histogram(
                [float(row[field]) for row in by_category[category]],
                zero_bucket=zero_bucket,
                scale=scale,
                bin_count=bin_count,
            )
            for category in categories
        },
    }


def _hourly_heatmap(categories: list[str], hourly_dist: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for category in categories:
        weights = hourly_dist.get(category)
        if weights is None:
            continue
        for hour, weight in enumerate(list(weights)):
            rows.append({"category": category, "hour": hour, "w": _r(float(weight), 6)})
    return {"categories": categories, "hours": list(range(24)), "rows": rows}


def _market_series(products: list[Product], small_share: float) -> dict[str, list[list[float]]]:
    demand = [0.0] * 365
    gmv = [0.0] * 365
    for p in products:
        for day, value in enumerate(p.market_curve[:365]):
            expected = float(value) * small_share
            demand[day] += expected
            gmv[day] += expected * p.ref_price
    return {
        "demand": [[day, _r(value)] for day, value in enumerate(demand)],
        "gmv": [[day, _r(value)] for day, value in enumerate(gmv)],
    }


def _risk_by_category(
    categories: list[str],
    by_category: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for category in categories:
        rows = by_category[category]
        n = max(1, len(rows))
        out.append({
            "category": category,
            **{
                key: _r(sum(float(r[key]) for r in rows) / n)
                for key in RISK_COMPONENTS
            },
            "order_risk": _r(sum(float(r["order_risk"]) for r in rows) / n),
            "supply_risk": _r(sum(float(r["supply_risk"]) for r in rows) / n),
        })
    return out


def _risk_histograms_by_event(
    categories: list[str],
    by_category: dict[str, list[dict[str, Any]]],
    metrics: list[dict[str, Any]],
    total_key: str,
    components: tuple[str, ...],
    *,
    period_steps: int = 1,
) -> dict[str, Any]:
    labels = {
        "All": "All",
        "cancel_rate": "cancel",
        "refund_rate": "refund",
        "only_refund_rate": "only refund",
        "bad_review_rate": "bad review",
        "timeout_rate": "timeout",
        "price_change_rate": "price change",
        "supplier_delist_rate": "delist",
    }
    event_fields = {"All": total_key, **{key: key for key in components}}
    events = list(event_fields)

    def event_values(rows: list[dict[str, Any]], event: str, field: str) -> list[float]:
        if period_steps <= 1:
            return [float(row[field]) for row in rows]
        if event == "All":
            return [
                _period_any_probability([float(row[key]) for key in components], period_steps)
                for row in rows
            ]
        return [
            _period_probability(float(row[field]), period_steps)
            for row in rows
        ]

    def event_histograms(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for event, field in event_fields.items():
            values = event_values(rows, event, field)
            out[event] = _histogram(
                values,
                bin_count=20,
                trim_quantile=None,
                fixed_range=(0.0, _focused_rate_upper(values)),
            )
            out[event]["focused_range"] = True
        return out

    return {
        "scale": "percent",
        "period_steps": period_steps,
        "categories": ["All", *categories],
        "events": events,
        "event_labels": {event: labels.get(event, event) for event in events},
        "all": event_histograms(metrics),
        "by_category": {
            category: event_histograms(by_category[category])
            for category in categories
        },
    }


def _profit_histograms_by_metric(
    categories: list[str],
    by_category: dict[str, list[dict[str, Any]]],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_fields = ("gross_profit365", "net_profit365", "cum_fine365")
    metric_labels = {
        "gross_profit365": "gross profit365",
        "net_profit365": "net profit365",
        "cum_fine365": "cum fine365",
    }

    def histograms(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            field: _histogram(
                [float(row[field]) for row in rows],
                zero_bucket=field == "cum_fine365",
                scale="log" if field == "cum_fine365" else "signed_log",
                bin_count=18,
            )
            for field in metric_fields
        }

    return {
        "scale": "profit",
        "categories": ["All", *categories],
        "metrics": list(metric_fields),
        "metric_labels": metric_labels,
        "all": histograms(metrics),
        "by_category": {
            category: histograms(by_category[category])
            for category in categories
        },
    }


def _scatter_points(metrics: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if not metrics:
        return []
    if sample_size >= len(metrics):
        return [_point(r) for r in sorted(metrics, key=lambda r: str(r["product_id"]))]

    selected: dict[str, dict[str, Any]] = {}
    protected_capacity = min(sample_size, max(2, min(80, sample_size // 5)))
    protected_groups = [
        sorted(metrics, key=lambda r: (float(r["net_profit365"]), str(r["product_id"])), reverse=True),
        sorted(metrics, key=lambda r: (float(r["order_risk"]), str(r["product_id"])), reverse=True),
        sorted(metrics, key=lambda r: (float(r["net_profit365"]), float(r["order_risk"]), str(r["product_id"]))),
        sorted(metrics, key=lambda r: (-float(r["order_risk"]), float(r["net_profit365"]), str(r["product_id"]))),
    ]
    per_group = max(1, protected_capacity // len(protected_groups))
    for group in protected_groups:
        for row in group[:per_group]:
            if len(selected) >= protected_capacity:
                break
            selected[str(row["product_id"])] = row
        if len(selected) >= protected_capacity:
            break

    remaining_count = sample_size - len(selected)
    if remaining_count > 0:
        rest = [
            r for r in sorted(metrics, key=lambda r: (float(r["gross_profit365"]), str(r["product_id"])))
            if str(r["product_id"]) not in selected
        ]
        bin_count = min(40, max(1, int(remaining_count ** 0.5)))
        base_quota = max(1, remaining_count // bin_count)
        remainder = remaining_count - base_quota * bin_count
        for idx in range(bin_count):
            if len(selected) >= sample_size:
                break
            start = round(idx * len(rest) / bin_count)
            end = round((idx + 1) * len(rest) / bin_count)
            bucket = sorted(rest[start:end], key=lambda r: (float(r["order_risk"]), str(r["product_id"])))
            quota = base_quota + (1 if idx < remainder else 0)
            for row in _stable_pick(bucket, quota):
                if len(selected) >= sample_size:
                    break
                selected[str(row["product_id"])] = row

    if len(selected) < sample_size:
        rest = [r for r in sorted(metrics, key=lambda r: str(r["product_id"]))
                if str(r["product_id"]) not in selected]
        for row in _stable_pick(rest, sample_size - len(selected)):
            selected[str(row["product_id"])] = row

    return [
        _point(r)
        for r in sorted(
            selected.values(),
            key=lambda r: (
                float(r["order_risk"]),
                float(r["net_profit365"]),
                str(r["product_id"]),
            ),
        )
    ]


def _outliers(metrics: list[dict[str, Any]], top_n: int) -> dict[str, list[dict[str, Any]]]:
    low_margin_high_demand = sorted(
        metrics,
        key=lambda r: (float(r["margin_ratio"]), -float(r["opportunity_gmv365"]), str(r["product_id"])),
    )
    return {
        "high_gmv": [
            _point(r) for r in sorted(metrics, key=lambda r: (-float(r["opportunity_gmv365"]), str(r["product_id"])))[:top_n]
        ],
        "high_risk": [
            _point(r) for r in sorted(metrics, key=lambda r: (-float(r["risk_sum"]), str(r["product_id"])))[:top_n]
        ],
        "extreme_arrival": [
            _point(r) for r in sorted(metrics, key=lambda r: (-float(r["arrival_hours"]), str(r["product_id"])))[:top_n]
        ],
        "low_margin_high_demand": [_point(r) for r in low_margin_high_demand[:top_n]],
    }


def _point(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": row["product_id"],
        "name": row["name"],
        "category": row["category"],
        "supplier_id": row["supplier_id"],
        "price": _r(row["price"]),
        "ref_price": _r(row["ref_price"]),
        "margin_ratio": _r(row["margin_ratio"]),
        "arrival_hours": _r(row["arrival_hours"]),
        "demand365": _r(row["demand365"]),
        "opportunity_gmv365": _r(row["opportunity_gmv365"]),
        "gross_profit365": _r(row["gross_profit365"]),
        "cum_fine365": _r(row["cum_fine365"]),
        "net_profit365": _r(row["net_profit365"]),
        "order_risk": _r(row["order_risk"]),
        "supply_risk": _r(row["supply_risk"]),
        "risk_sum": _r(row["risk_sum"]),
        "peak_to_mean": _r(row["peak_to_mean"]),
        "elasticity": _r(row["elasticity"]),
    }


def _expected_unit_profit_at_ref(cost: float, ref_price: float, rates: dict[str, float],
                                 *, refund_penalty: float = 8.0,
                                 bad_review_penalty: float = 5.0) -> float:
    margin = float(ref_price) - float(cost)
    cancel = max(0.0, min(1.0, float(rates.get("cancel_rate", 0.0))))
    refund = max(0.0, min(1.0, float(rates.get("refund_rate", 0.0))))
    only_refund = max(0.0, min(1.0, float(rates.get("only_refund_rate", 0.0))))
    bad_review = max(0.0, min(1.0, float(rates.get("bad_review_rate", 0.0))))
    normal = max(0.0, 1.0 - cancel - refund - only_refund - bad_review)
    return (
        normal * margin
        + cancel * 0.0
        + refund * -refund_penalty
        + only_refund * -float(cost)
        + bad_review * (margin - bad_review_penalty)
    )


def _product_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **_point(row),
        "supplier_name": row["supplier_name"],
        "quantity": row["quantity"],
        "max_quantity": row["max_quantity"],
        "curve_sum": _r(row["curve_sum"]),
        "curve_peak": _r(row["curve_peak"]),
    }


def _category_curve_band(products: list[Product]) -> dict[str, list[list[float]]]:
    bands = {"p10": [], "p50": [], "p90": []}
    for day in range(365):
        values = [float(p.market_curve[day]) for p in products if len(p.market_curve) > day]
        bands["p10"].append([day, _r(_quantile(values, 0.10))])
        bands["p50"].append([day, _r(_quantile(values, 0.50))])
        bands["p90"].append([day, _r(_quantile(values, 0.90))])
    return bands


def _series365(values: list[float]) -> list[list[float]]:
    return [[day, _r(float(value))] for day, value in enumerate(values[:365])]


def _histogram(
    values: list[float],
    *,
    zero_bucket: bool = False,
    bin_count: int = 24,
    trim_quantile: float | None = 0.99,
    scale: str = "linear",
    fixed_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        empty = {
            "count": 0,
            "zero_count": 0,
            "positive_count": 0,
            "trimmed_count": 0,
            "trimmed_removed": 0,
            "trim_range": [0.0, 0.0],
            "bins": [],
            "trimmed_bins": [],
        }
        return empty

    if trim_quantile is None:
        lo = fixed_range[0] if fixed_range else min(clean)
        hi = fixed_range[1] if fixed_range else max(clean)
        trimmed = clean
    else:
        lo = _quantile(clean, 1.0 - trim_quantile)
        hi = _quantile(clean, trim_quantile)
        trimmed = [v for v in clean if lo <= v <= hi]
    if not trimmed:
        trimmed = clean
        lo = min(clean)
        hi = max(clean)

    return {
        "count": len(clean),
        "scale": scale,
        "zero_count": sum(1 for v in clean if v == 0.0),
        "positive_count": sum(1 for v in clean if v > 0.0),
        "trimmed_count": len(trimmed),
        "trimmed_removed": len(clean) - len(trimmed),
        "trim_range": [_r(lo), _r(hi)],
        "range": [_r(fixed_range[0]), _r(fixed_range[1])] if fixed_range else [_r(min(clean)), _r(max(clean))],
        "bins": _histogram_bins(
            clean,
            zero_bucket=zero_bucket,
            bin_count=bin_count,
            scale=scale,
            fixed_range=fixed_range,
        ),
        "nonzero_bins": _histogram_bins(
            [v for v in clean if v != 0.0],
            zero_bucket=False,
            bin_count=bin_count,
            scale=scale,
            fixed_range=fixed_range,
        ),
        "trimmed_bins": _histogram_bins(
            trimmed,
            zero_bucket=zero_bucket,
            bin_count=bin_count,
            scale=scale,
            fixed_range=fixed_range,
        ),
    }


def _histogram_bins(
    values: list[float],
    *,
    zero_bucket: bool,
    bin_count: int,
    scale: str,
    fixed_range: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    if not values:
        return []

    bins: list[dict[str, Any]] = []
    remaining = list(values)
    total = len(values)
    if zero_bucket:
        zero_count = sum(1 for v in remaining if v == 0.0)
        if zero_count:
            bins.append({
                "lo": 0.0,
                "hi": 0.0,
                "count": zero_count,
                "share": _r(zero_count / total, 6),
                "zero": True,
            })
            remaining = [v for v in remaining if v != 0.0]
    if not remaining:
        return bins

    lo = fixed_range[0] if fixed_range else min(remaining)
    hi = fixed_range[1] if fixed_range else max(remaining)
    if lo == hi:
        bins.append({
            "lo": _r(lo),
            "hi": _r(hi),
            "count": len(remaining),
            "share": _r(len(remaining) / total, 6),
            "zero": False,
        })
        return bins

    count = max(1, int(bin_count))
    counts = [0] * count
    if scale == "log":
        positive = [v for v in remaining if v > 0.0]
        if not positive:
            return bins
        log_lo = _safe_log10(max(min(positive), 1e-9))
        log_hi = _safe_log10(max(max(positive), 1e-9))
        if log_lo == log_hi:
            bins.append({
                "lo": _r(min(positive)),
                "hi": _r(max(positive)),
                "count": len(positive),
                "share": _r(len(positive) / total, 6),
                "zero": False,
            })
            return bins
        width = (log_hi - log_lo) / count
        for value in positive:
            idx = min(count - 1, int((_safe_log10(value) - log_lo) / width))
            counts[idx] += 1
        edges = [10 ** (log_lo + idx * width) for idx in range(count + 1)]
    elif scale == "signed_log":
        log_lo = _signed_log10(lo)
        log_hi = _signed_log10(hi)
        if log_lo == log_hi:
            bins.append({
                "lo": _r(lo),
                "hi": _r(hi),
                "count": len(remaining),
                "share": _r(len(remaining) / total, 6),
                "zero": False,
            })
            return bins
        width = (log_hi - log_lo) / count
        for value in remaining:
            idx = min(count - 1, max(0, int((_signed_log10(value) - log_lo) / width)))
            counts[idx] += 1
        edges = [_signed_exp10(log_lo + idx * width) for idx in range(count + 1)]
    else:
        width = (hi - lo) / count
        for value in remaining:
            idx = min(count - 1, max(0, int((value - lo) / width)))
            counts[idx] += 1
        edges = [lo + idx * width for idx in range(count + 1)]

    for idx, n in enumerate(counts):
        bin_lo = edges[idx]
        bin_hi = hi if idx == count - 1 and scale not in {"log", "signed_log"} else edges[idx + 1]
        bins.append({
            "lo": _r(bin_lo),
            "hi": _r(bin_hi),
            "count": n,
            "share": _r(n / total, 6),
            "zero": False,
        })
    return bins


def _safe_log10(value: float) -> float:
    return math.log10(max(float(value), 1e-9))


def _signed_log10(value: float) -> float:
    number = float(value)
    if number == 0.0:
        return 0.0
    return math.copysign(math.log10(1.0 + abs(number)), number)


def _signed_exp10(value: float) -> float:
    number = float(value)
    if number == 0.0:
        return 0.0
    return math.copysign((10 ** abs(number)) - 1.0, number)


def _focused_rate_upper(values: list[float]) -> float:
    clean = [max(0.0, min(1.0, float(v))) for v in values if v is not None]
    if not clean:
        return 0.05
    top = max(clean)
    if top <= 0.0:
        return 0.05
    if top <= 0.05:
        step = 0.005
    elif top <= 0.20:
        step = 0.01
    else:
        step = 0.05
    return min(1.0, max(step, math.ceil(top / step) * step))


def _period_probability(rate: float, steps: int) -> float:
    clean = max(0.0, min(1.0, float(rate)))
    return 1.0 - ((1.0 - clean) ** max(1, int(steps)))


def _period_any_probability(rates: list[float], steps: int) -> float:
    no_event_per_step = 1.0
    for rate in rates:
        no_event_per_step *= 1.0 - max(0.0, min(1.0, float(rate)))
    return 1.0 - (no_event_per_step ** max(1, int(steps)))


def _stable_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    step = (len(rows) - 1) / (count - 1)
    return [rows[round(i * step)] for i in range(count)]


def _quantile(values: list[float], q: float) -> float:
    clean = sorted(float(v) for v in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _r(value: Any, ndigits: int = 4) -> float:
    return round(float(value), ndigits)
