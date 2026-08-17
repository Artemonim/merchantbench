"""Loader for offline private_real catalog datasets.

Runs read a prebuilt SQLite dataset DB and copy its rows into the normal
per-run DB. Oversized pools are subsampled at run start with
``derive_rng(master_seed, "data_gen", "catalog_subsample")``.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Any

import numpy as np

from core.entities import Product
from core.rng import derive_rng


class PrivateRealDataError(ValueError):
    """Raised when the private_real dataset DB is missing or invalid."""


_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_PRIVATE_REAL_DB_PATH = os.path.join(
    _HERE, "private_data", "private_real_1k.sqlite"
)


def resolve_dataset_path(path: str | None = None) -> str:
    if not path:
        return DEFAULT_PRIVATE_REAL_DB_PATH
    if os.path.isabs(path):
        return path
    return os.path.join(_ENV_ROOT, path)


def dataset_available(path: str | None = None) -> bool:
    resolved = resolve_dataset_path(path)
    if not os.path.isfile(resolved):
        return False
    try:
        with _connect_readonly(resolved) as conn:
            conn.execute("SELECT 1 FROM dataset_meta LIMIT 1").fetchone()
            conn.execute("SELECT 1 FROM products LIMIT 1").fetchone()
            _require_hourly_dist_schema(conn)
            conn.execute("SELECT 1 FROM hourly_dist LIMIT 1").fetchone()
    except (sqlite3.Error, PrivateRealDataError):
        return False
    return True


def load_dataset(path: str | None = None) -> tuple[list[Product], dict[str, np.ndarray], dict[str, str]]:
    resolved = resolve_dataset_path(path)
    if not os.path.isfile(resolved):
        raise PrivateRealDataError(f"private_real dataset DB not found: {resolved}")

    try:
        with _connect_readonly(resolved) as conn:
            meta = _load_meta(conn)
            products = _load_products(conn)
            _require_hourly_dist_schema(conn)
            hourly_dist = _load_hourly_dist(conn)
    except sqlite3.Error as exc:
        raise PrivateRealDataError(f"private_real dataset DB is unreadable: {exc}") from exc

    _validate_dataset(products, hourly_dist)
    meta.setdefault("data_source", "private_real")
    meta.setdefault("dataset_rows", str(len(products)))
    return products, hourly_dist, meta


def subsample_catalog(
    products: list[Product],
    hourly_dist: dict[str, np.ndarray],
    num_products: int,
    master_seed: int,
) -> tuple[list[Product], dict[str, np.ndarray]]:
    """Return a seed-stable prefix of a shuffled catalog.

    Shuffles product IDs once with
    ``derive_rng(master_seed, "data_gen", "catalog_subsample")`` and keeps
    the first ``num_products``. The same seed therefore yields the same
    assortment, and a smaller ``N`` is a prefix of a larger ``N``.
    Synthetic catalogs that already have ``len <= num_products`` are
    returned unchanged (no second shuffle).

    Args:
        products: Pool loaded from the private_real dataset.
        hourly_dist: Per-category 24h weights for the full pool.
        num_products: Requested assortment size.
        master_seed: Run ``master_seed``.

    Returns:
        Subsampled products in shuffle order and hourly_dist restricted
        to remaining categories.

    Raises:
        ValueError: If ``num_products`` is not a positive integer.
    """
    try:
        keep_n = int(num_products)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"num_products must be a positive integer, got {num_products!r}"
        ) from exc
    if keep_n <= 0:
        raise ValueError(
            f"num_products must be a positive integer, got {num_products!r}"
        )
    if len(products) <= keep_n:
        return products, hourly_dist

    ordered_ids = [product.product_id for product in products]
    rng = derive_rng(int(master_seed), "data_gen", "catalog_subsample")
    rng.shuffle(ordered_ids)
    kept_ids = ordered_ids[:keep_n]
    by_id = {product.product_id: product for product in products}
    sampled = [by_id[product_id] for product_id in kept_ids]
    remaining = {product.category for product in sampled}
    filtered_hourly = {
        category: hourly_dist[category]
        for category in remaining
        if category in hourly_dist
    }
    return sampled, filtered_hourly


def _connect_readonly(path: str) -> sqlite3.Connection:
    uri = "file:" + os.path.abspath(path) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM dataset_meta").fetchall()
    return {str(r["key"]): str(r["value"]) for r in rows}


def _load_products(conn: sqlite3.Connection) -> list[Product]:
    rows = conn.execute("SELECT * FROM products ORDER BY category, product_id").fetchall()
    out: list[Product] = []
    for r in rows:
        columns = set(r.keys())
        try:
            curve = json.loads(r["market_curve"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise PrivateRealDataError(
                f"invalid market_curve for product {r['product_id']!r}"
            ) from exc
        out.append(Product(
            product_id=str(r["product_id"]),
            name=str(r["name"]),
            quantity=int(r["quantity"]),
            price=float(r["price"]),
            ref_price=float(r["ref_price"]),
            base_price=float(r["base_price"]) if "base_price" in columns and r["base_price"] is not None else float(r["price"]),
            supplier_id=str(r["supplier_id"]),
            supplier_name=str(r["supplier_name"]),
            ship_hours=int(r["ship_hours"]),
            logistics_hours=int(r["logistics_hours"]),
            category=str(r["category"]),
            historical_avg_rating=float(r["historical_avg_rating"]),
            shop_rating=float(r["shop_rating"]),
            return_buyer_rate=float(r["return_buyer_rate"]),
            supplier_age_years=float(r["supplier_age_years"]),
            cancel_rate=float(r["cancel_rate"]),
            refund_rate=float(r["refund_rate"]),
            only_refund_rate=float(r["only_refund_rate"]),
            bad_review_rate=float(r["bad_review_rate"]),
            max_quantity=int(r["max_quantity"]),
            hourly_increment=int(r["hourly_increment"]),
            timeout_rate=float(r["timeout_rate"]),
            price_change_rate=float(r["price_change_rate"]),
            supplier_delist_rate=float(r["supplier_delist_rate"]),
            elasticity=float(r["elasticity"]),
            market_curve=[float(x) for x in curve],
        ))
    return out


def _load_hourly_dist(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    rows = conn.execute(
        "SELECT category, hour, w FROM hourly_dist ORDER BY category, hour"
    ).fetchall()
    out: dict[str, np.ndarray] = {}
    for r in rows:
        cat = str(r["category"])
        hour = int(r["hour"])
        if hour < 0 or hour > 23:
            raise PrivateRealDataError(f"invalid hourly_dist slot for category {cat!r}")
        out.setdefault(cat, np.zeros(24, dtype=float))[hour] = float(r["w"])
    return out


def _require_hourly_dist_schema(conn: sqlite3.Connection) -> None:
    columns = [str(r["name"]) for r in conn.execute("PRAGMA table_info(hourly_dist)")]
    if columns != ["category", "hour", "w"]:
        raise PrivateRealDataError("hourly_dist schema must be (category, hour, w)")


def _validate_dataset(products: list[Product], hourly_dist: dict[str, np.ndarray]) -> None:
    if not products:
        raise PrivateRealDataError("private_real dataset has no products")
    seen_ids: set[str] = set()
    supplier_profiles: dict[str, tuple[float, float, float]] = {}
    categories = {p.category for p in products}

    for p in products:
        if not p.product_id or not p.name or not p.category:
            raise PrivateRealDataError("product_id, name, and category must be non-empty")
        if p.product_id in seen_ids:
            raise PrivateRealDataError(f"duplicate product_id: {p.product_id}")
        seen_ids.add(p.product_id)
        _require_finite("price", p.price, p.product_id, lo=1e-9)
        _require_finite("ref_price", p.ref_price, p.product_id, lo=1e-9)
        _require_finite("base_price", p.base_price, p.product_id, lo=1e-9)
        if p.quantity < 0 or p.max_quantity <= 0 or p.quantity > p.max_quantity:
            raise PrivateRealDataError(f"invalid quantity bounds for {p.product_id}")
        if p.hourly_increment < 0:
            raise PrivateRealDataError(f"hourly_increment must be >= 0 for {p.product_id}")
        if p.ship_hours < 1 or not 1 <= p.logistics_hours <= 72:
            raise PrivateRealDataError(f"invalid logistics fields for {p.product_id}")
        for field in ("cancel_rate", "refund_rate", "only_refund_rate",
                      "bad_review_rate", "timeout_rate", "price_change_rate",
                      "supplier_delist_rate"):
            value = float(getattr(p, field))
            _require_finite(field, value, p.product_id, lo=0.0, hi=1.0)
        _require_finite("elasticity", p.elasticity, p.product_id, lo=0.5, hi=6.0)
        _require_finite("historical_avg_rating", p.historical_avg_rating,
                        p.product_id, lo=1.0, hi=5.0)
        _require_finite("shop_rating", p.shop_rating, p.product_id, lo=1.0, hi=5.0)
        _require_finite("return_buyer_rate", p.return_buyer_rate,
                        p.product_id, lo=0.0, hi=1.0)
        _require_finite("supplier_age_years", p.supplier_age_years,
                        p.product_id, lo=0.0)
        if len(p.market_curve) != 365 or any((not math.isfinite(x) or x < 0) for x in p.market_curve):
            raise PrivateRealDataError(f"market_curve must be 365 non-negative floats for {p.product_id}")
        profile = (round(p.shop_rating, 8), round(p.return_buyer_rate, 8),
                   round(p.supplier_age_years, 8))
        old = supplier_profiles.setdefault(p.supplier_id, profile)
        if old != profile:
            raise PrivateRealDataError(f"supplier profile drift for {p.supplier_id}")

    missing = categories - set(hourly_dist)
    if missing:
        raise PrivateRealDataError(f"missing hourly_dist categories: {sorted(missing)}")
    for category in categories:
        w = hourly_dist[category]
        if w.shape != (24,):
            raise PrivateRealDataError(f"hourly_dist for {category!r} must be 24h")
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise PrivateRealDataError(f"hourly_dist for {category!r} has invalid weights")
        if not np.isclose(float(w.sum()), 1.0, atol=1e-6):
            raise PrivateRealDataError(
                f"hourly_dist for {category!r} must sum to 1"
            )


def _require_finite(
    field: str,
    value: float,
    product_id: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> None:
    if not math.isfinite(value):
        raise PrivateRealDataError(f"{field} must be finite for {product_id}")
    if lo is not None and value < lo:
        raise PrivateRealDataError(f"{field} must be >= {lo} for {product_id}")
    if hi is not None and value > hi:
        raise PrivateRealDataError(f"{field} must be <= {hi} for {product_id}")
