"""Build an offline private_real dataset SQLite DB from CSV extracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable

from core.rng import derive_rng
from data.generation_profiles import (
    build_hourly_dist_for_categories,
    hourly_dist_rows,
    load_default_generation_params,
    normalize_supplier_ranges,
    rand_range,
    resample_periodic_curve,
    sample_operational_fields,
    sample_risk_event_fields,
    translate_category,
)

SUPPLIER_CAP = 50
MAX_REF_PRICE_RATIO = 5.0
REF_PRICE_CAP_RATIO = 2.0
OPPORTUNITY_PROFIT_CAP_QUANTILE = 0.95
EXPECTED_NET_PROFIT_CAP_365 = 50_000.0
EXPECTED_NET_PROFIT_TOP1000_P95_CAP_365 = 40_000.0
EXPECTED_NET_PROFIT_TOP_N = 1000
REFUND_PENALTY_AMOUNT = 8.0
BAD_REVIEW_PENALTY_AMOUNT = 5.0
ELASTICITY_CLIP_MIN = 1.10
ELASTICITY_CLIP_MAX = 6.00
RISK_CALIBRATION_VERSION = "profit_density_v3"
SAMPLING_STRATEGY = "stratified_v1"
CATEGORY_ALLOCATION_BALANCED = "balanced"
CATEGORY_ALLOCATION_CAPACITY = "capacity"
CATEGORY_ALLOCATION_SOURCE = "source"
STRATUM_RATIOS = {
    "good": 0.05,
    "safe": 0.25,
    "trap": 0.15,
    "mediocre": 0.45,
    "minefield": 0.10,
}
RISK_GROUP_BOUNDS = {
    "low": (0.05, 0.30),
    "medium": (0.18, 0.60),
    "high": (0.40, 0.90),
}
RISK_GROUP_BETA_SHAPES = {
    "low": (2.0, 2.8),
    "medium": (2.2, 2.2),
    "high": (2.3, 1.9),
}
RISK_DENSITY_ANCHORS = (
    (0.00, {"low": 0.70, "medium": 0.25, "high": 0.05}),
    (0.50, {"low": 0.50, "medium": 0.38, "high": 0.12}),
    (0.80, {"low": 0.36, "medium": 0.44, "high": 0.20}),
    (0.95, {"low": 0.22, "medium": 0.43, "high": 0.35}),
    (0.99, {"low": 0.10, "medium": 0.32, "high": 0.58}),
    (0.998, {"low": 0.06, "medium": 0.24, "high": 0.70}),
    (1.00, {"low": 0.05, "medium": 0.25, "high": 0.70}),
)
STRATUM_RISK_DENSITY_BIAS = {
    "good": {"low": 1.30, "medium": 1.00, "high": 0.75},
    "safe": {"low": 1.15, "medium": 1.00, "high": 0.85},
    "mediocre": {"low": 1.08, "medium": 1.02, "high": 0.90},
    "trap": {"low": 0.75, "medium": 1.00, "high": 1.35},
    "minefield": {"low": 0.55, "medium": 0.85, "high": 1.75},
}
PRODUCT_COLS = (
    "product_id",
    "name",
    "quantity",
    "price",
    "base_price",
    "ref_price",
    "raw_ref_price",
    "supplier_id",
    "supplier_name",
    "ship_hours",
    "logistics_hours",
    "category",
    "historical_avg_rating",
    "shop_rating",
    "return_buyer_rate",
    "supplier_age_years",
    "cancel_rate",
    "refund_rate",
    "only_refund_rate",
    "bad_review_rate",
    "max_quantity",
    "hourly_increment",
    "timeout_rate",
    "price_change_rate",
    "supplier_delist_rate",
    "elasticity",
    "market_curve",
    "stratum",
    "good_rate_source",
    "pt_rate_source",
)


def build_private_real_db(
    *,
    bench_csv: str,
    output_db: str,
    target_rows: int = 10000,
    seed: int = 42,
    supplier_cap: int = SUPPLIER_CAP,
    params_path: str | None = None,
    params: dict | None = None,
    exact_streaming: bool = False,
    progress_every: int = 0,
    category_allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> dict[str, str]:
    if exact_streaming:
        products, hourly_dist, meta = prepare_private_real_dataset_streaming(
            bench_csv=bench_csv,
            target_rows=target_rows,
            seed=seed,
            supplier_cap=supplier_cap,
            params_path=params_path,
            params=params,
            progress_every=progress_every,
            category_allocation=category_allocation,
        )
    else:
        products, hourly_dist, meta = prepare_private_real_dataset(
            bench_csv=bench_csv,
            target_rows=target_rows,
            seed=seed,
            supplier_cap=supplier_cap,
            params_path=params_path,
            params=params,
            category_allocation=category_allocation,
        )
    _write_db(output_db, products, hourly_dist, meta)
    return meta


def prepare_private_real_dataset(
    *,
    bench_csv: str,
    target_rows: int = 10000,
    seed: int = 42,
    supplier_cap: int = SUPPLIER_CAP,
    params_path: str | None = None,
    params: dict | None = None,
    category_allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> tuple[list[dict], dict[str, list[tuple[int, float]]], dict[str, str]]:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")

    profile_params = params or load_default_generation_params(params_path)
    item_bounds = profile_params.get("supplier_item_count") or {}
    supplier_min = int(item_bounds.get("min", 5))
    supplier_max = int(item_bounds.get("max", supplier_cap))
    supplier_cap = min(int(supplier_cap), supplier_max)
    supplier_cap = _effective_supplier_selection_cap(target_rows, supplier_cap)

    base_rows = _read_bench_rows(bench_csv, profile_params)
    return prepare_private_real_dataset_from_rows(
        base_rows,
        bench_csv=bench_csv,
        target_rows=target_rows,
        seed=seed,
        supplier_cap=supplier_cap,
        params=profile_params,
        supplier_min=supplier_min,
        supplier_max=supplier_max,
        category_allocation=category_allocation,
    )


def prepare_private_real_dataset_from_rows(
    base_rows: list[dict],
    *,
    bench_csv: str,
    target_rows: int = 10000,
    seed: int = 42,
    supplier_cap: int = SUPPLIER_CAP,
    params: dict,
    supplier_min: int | None = None,
    supplier_max: int | None = None,
    category_allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> tuple[list[dict], dict[str, list[tuple[int, float]]], dict[str, str]]:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    item_bounds = params.get("supplier_item_count") or {}
    supplier_min = int(supplier_min if supplier_min is not None else item_bounds.get("min", 5))
    supplier_max = int(supplier_max if supplier_max is not None else item_bounds.get("max", supplier_cap))
    supplier_cap = min(int(supplier_cap), supplier_max)
    supplier_cap = _effective_supplier_selection_cap(target_rows, supplier_cap)
    selection_supplier_min = min(supplier_min, target_rows)

    raw_rows = list(base_rows)
    if not raw_rows:
        raise ValueError("bench_csv has no valid rows after data quality filtering")
    categories = _select_categories(raw_rows)
    category_counts = Counter(row["_category"] for row in raw_rows if row["_category"] in categories)
    category_quotas = _category_quota_counts(
        target_rows,
        categories,
        category_counts,
        allocation=category_allocation,
    )

    rng = random.Random(seed)
    selected = _sample_rows(raw_rows, categories, category_quotas, rng, supplier_cap, selection_supplier_min)
    products = _build_products(selected, seed, params)
    hourly_dist = _build_hourly_dist(categories, seed, params)
    meta = _metadata(
        bench_csv,
        products,
        hourly_dist,
        target_rows,
        supplier_min=supplier_min,
        supplier_max=supplier_max,
        supplier_selection_cap=supplier_cap,
        params=params,
        category_allocation=category_allocation,
    )
    return products, hourly_dist, meta


def _effective_supplier_selection_cap(target_rows: int, supplier_cap: int) -> int:
    return int(supplier_cap)


def _category_quota_counts(
    target_rows: int,
    categories: list[str],
    category_counts: Counter,
    *,
    allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> dict[str, int]:
    if not categories:
        raise ValueError("no categories found in bench_csv")
    if target_rows < len(categories):
        raise ValueError("target_rows is smaller than the selected category count")
    available = {category: int(category_counts.get(category, 0)) for category in categories}
    if allocation == CATEGORY_ALLOCATION_BALANCED:
        base = target_rows // len(categories)
        remainder = target_rows - base * len(categories)
        quotas = {category: base + (1 if idx < remainder else 0) for idx, category in enumerate(categories)}
        for category, quota in quotas.items():
            if available[category] < quota:
                raise ValueError(f"category {category!r} has {available[category]} candidates, need {quota}")
        return quotas
    if allocation == CATEGORY_ALLOCATION_SOURCE:
        return _proportional_quota_counts(target_rows, categories, available)
    if allocation != CATEGORY_ALLOCATION_CAPACITY:
        raise ValueError(
            f"category_allocation must be {CATEGORY_ALLOCATION_BALANCED!r} "
            f"{CATEGORY_ALLOCATION_SOURCE!r}, or {CATEGORY_ALLOCATION_CAPACITY!r}"
        )

    if sum(available.values()) < target_rows:
        raise ValueError(f"only {sum(available.values())} candidates available, need {target_rows}")
    base = target_rows // len(categories)
    quotas = {category: min(available[category], base) for category in categories}
    remaining = target_rows - sum(quotas.values())
    while remaining > 0:
        eligible = [category for category in categories if quotas[category] < available[category]]
        if not eligible:
            raise ValueError(f"only {sum(quotas.values())} candidates available, need {target_rows}")
        eligible = sorted(
            eligible,
            key=lambda category: (-(available[category] - quotas[category]), category),
        )
        for category in eligible:
            if remaining <= 0:
                break
            if quotas[category] >= available[category]:
                continue
            quotas[category] += 1
            remaining -= 1
    return quotas


def _proportional_quota_counts(
    target_rows: int,
    categories: list[str],
    available: dict[str, int],
) -> dict[str, int]:
    total_available = sum(available.values())
    if total_available < target_rows:
        raise ValueError(f"only {total_available} candidates available, need {target_rows}")
    raw = {category: (available[category] / total_available) * target_rows for category in categories}
    quotas = {category: min(available[category], int(math.floor(raw[category]))) for category in categories}
    remaining = target_rows - sum(quotas.values())
    while remaining > 0:
        eligible = [category for category in categories if quotas[category] < available[category]]
        if not eligible:
            raise ValueError(f"only {sum(quotas.values())} candidates available, need {target_rows}")
        eligible = sorted(
            eligible,
            key=lambda category: (
                raw[category] - math.floor(raw[category]),
                available[category] - quotas[category],
                category,
            ),
            reverse=True,
        )
        for category in eligible:
            if remaining <= 0:
                break
            quotas[category] += 1
            remaining -= 1
    return quotas


def prepare_private_real_dataset_streaming(
    *,
    bench_csv: str,
    target_rows: int = 10000,
    seed: int = 42,
    supplier_cap: int = SUPPLIER_CAP,
    params_path: str | None = None,
    params: dict | None = None,
    progress_every: int = 0,
    category_allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> tuple[list[dict], dict[str, list[tuple[int, float]]], dict[str, str]]:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")

    profile_params = params or load_default_generation_params(params_path)
    item_bounds = profile_params.get("supplier_item_count") or {}
    supplier_min = int(item_bounds.get("min", 5))
    supplier_max = int(item_bounds.get("max", supplier_cap))
    supplier_cap = min(int(supplier_cap), supplier_max)
    supplier_cap = _effective_supplier_selection_cap(target_rows, supplier_cap)
    selection_supplier_min = min(supplier_min, target_rows)

    hard_cap = _streaming_price_hard_cap(
        bench_csv,
        profile_params,
        progress_every=progress_every,
    )
    with tempfile.NamedTemporaryFile(prefix="private_real_candidates_", suffix=".sqlite", delete=False) as tmp:
        candidates_db = tmp.name
    try:
        supplier_counts, category_counts = _streaming_collect_positive_candidates(
            bench_csv,
            candidates_db,
            profile_params,
            hard_cap,
            supplier_min=supplier_min,
            supplier_max=supplier_max,
            progress_every=progress_every,
        )
        categories = _rank_candidate_categories(category_counts)
        category_quotas = _category_quota_counts(
            target_rows,
            categories,
            category_counts,
            allocation=category_allocation,
        )

        rng = random.Random(seed)
        selected_rows, curve_cap_by_category = _streaming_select_candidates(
            candidates_db,
            supplier_counts,
            categories,
            category_quotas,
            rng,
            selection_supplier_min,
            supplier_cap,
        )
        expected_rows = sum(category_quotas.values())
        if len(selected_rows) != expected_rows:
            raise ValueError(f"selected {len(selected_rows)} rows, expected {expected_rows}")
        full_rows = _streaming_load_selected_rows(
            bench_csv,
            selected_rows,
            curve_cap_by_category,
            profile_params,
            hard_cap,
            progress_every=progress_every,
        )
    finally:
        try:
            os.remove(candidates_db)
        except FileNotFoundError:
            pass

    products = _build_products(full_rows, seed, profile_params)
    hourly_dist = _build_hourly_dist(categories, seed, profile_params)
    meta = _metadata(
        bench_csv,
        products,
        hourly_dist,
        target_rows,
        supplier_min=supplier_min,
        supplier_max=supplier_max,
        supplier_selection_cap=supplier_cap,
        params=profile_params,
        build_mode="exact_streaming",
        category_allocation=category_allocation,
    )
    meta["price_hard_cap"] = str(hard_cap)
    return products, hourly_dist, meta


def _streaming_price_hard_cap(
    bench_csv: str,
    params: dict,
    *,
    progress_every: int = 0,
) -> float:
    prices = array("d")
    with open(bench_csv, newline="", encoding="utf-8") as f:
        for row_idx, raw_row in enumerate(csv.DictReader(f), start=1):
            row = _parse_bench_row_light(raw_row, params)
            if row is not None:
                prices.append(float(row["_price"]))
                prices.append(float(row["_ref_price"]))
            _progress("price_cap", row_idx, progress_every)
    if not prices:
        return 0.0
    cap = _value_quantile(sorted(prices), 0.995)
    return max(cap, 1.0) * 2.0


def _streaming_collect_positive_candidates(
    bench_csv: str,
    candidates_db: str,
    params: dict,
    hard_cap: float,
    *,
    supplier_min: int,
    supplier_max: int,
    progress_every: int = 0,
) -> tuple[Counter, Counter]:
    conn = sqlite3.connect(candidates_db)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            "CREATE TABLE candidates("
            "row_idx INTEGER PRIMARY KEY,"
            "category TEXT NOT NULL,"
            "item_id TEXT NOT NULL,"
            "member_id TEXT NOT NULL,"
            "curve_sum REAL NOT NULL,"
            "price REAL NOT NULL,"
            "ref_price REAL NOT NULL,"
            "good_rate REAL NOT NULL,"
            "pt_rate REAL NOT NULL,"
            "good_rate_source TEXT NOT NULL,"
            "pt_rate_source TEXT NOT NULL)"
        )
        supplier_counts: Counter = Counter()
        positive_rows: list[tuple] = []
        with open(bench_csv, newline="", encoding="utf-8") as f:
            for row_idx, raw_row in enumerate(csv.DictReader(f), start=1):
                row = _parse_bench_row_light(raw_row, params)
                if row is None:
                    _progress("collect_candidates", row_idx, progress_every)
                    continue
                if row["_price"] > hard_cap or row["_ref_price"] > hard_cap:
                    _progress("collect_candidates", row_idx, progress_every)
                    continue
                supplier_id = _supplier_id(row)
                supplier_counts[supplier_id] += 1
                positive_rows.append(
                    (
                        row_idx,
                        row["_category"],
                        row["item_id"],
                        supplier_id,
                        row["_curve_sum"],
                        row["_price"],
                        row["_ref_price"],
                        row["_good_rate"],
                        row["_pt_rate"],
                        row["_good_rate_source"],
                        row["_pt_rate_source"],
                    )
                )
                if len(positive_rows) >= 20000:
                    conn.executemany(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        positive_rows,
                    )
                    positive_rows.clear()
                _progress("collect_candidates", row_idx, progress_every)
        if positive_rows:
            conn.executemany(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                positive_rows,
            )
        conn.execute("CREATE INDEX idx_candidates_category ON candidates(category)")
        conn.commit()

        category_counts: Counter = Counter()
        for (category,) in conn.execute("SELECT category FROM candidates"):
            category_counts[str(category)] += 1
        return supplier_counts, category_counts
    finally:
        conn.close()


def _rank_candidate_categories(category_counts: Counter) -> list[str]:
    ranked = sorted(category_counts, key=lambda c: (-category_counts[c], c))
    if not ranked:
        raise ValueError("bench_csv has no valid rows after data quality filtering")
    return ranked[:10]


def _streaming_select_candidates(
    candidates_db: str,
    supplier_counts: Counter,
    categories: list[str],
    category_quotas: dict[str, int],
    rng: random.Random,
    supplier_min: int,
    supplier_cap: int,
) -> tuple[dict[int, str], dict[str, float]]:
    conn = sqlite3.connect(candidates_db)
    conn.row_factory = sqlite3.Row
    try:
        selected_by_row_idx: dict[int, str] = {}
        opportunity_cap_by_category: dict[str, float] = {}
        selection_supplier_counts: dict[str, int] = defaultdict(int)
        category_supplier_counts = _category_supplier_counts(conn, categories)
        selected_supplier_counts: Counter = Counter()
        for counts in category_supplier_counts.values():
            selected_supplier_counts.update(counts)
        selection_order, capacity_stats = _category_selection_order(
            conn,
            selected_supplier_counts,
            categories,
            supplier_min,
            supplier_cap,
        )
        future_supplier_counts = Counter(selected_supplier_counts)
        selected_row_info: dict[int, tuple[str, str]] = {}
        for category in selection_order:
            per_category = int(category_quotas.get(category, 0))
            if per_category <= 0:
                continue
            future_supplier_counts.subtract(category_supplier_counts.get(category, Counter()))
            _drop_non_positive_counts(future_supplier_counts)
            candidates = []
            for r in conn.execute(
                "SELECT * FROM candidates WHERE category=? ORDER BY row_idx",
                (category,),
            ):
                supplier_id = str(r["member_id"])
                if selected_supplier_counts[supplier_id] < supplier_min:
                    continue
                candidates.append(
                    {
                        "_source_row_idx": int(r["row_idx"]),
                        "_category": str(r["category"]),
                        "item_id": str(r["item_id"]),
                        "member_id": supplier_id,
                        "_curve_sum": float(r["curve_sum"]),
                        "_price": float(r["price"]),
                        "_ref_price": float(r["ref_price"]),
                        "_good_rate": float(r["good_rate"]),
                        "_pt_rate": float(r["pt_rate"]),
                        "_good_rate_source": str(r["good_rate_source"]),
                        "_pt_rate_source": str(r["pt_rate_source"]),
                    }
                )
            if len(candidates) < per_category:
                raise ValueError(f"category {category!r} has {len(candidates)} candidates, need {per_category}")
            opportunity_cap_by_category[category] = _opportunity_profit_cap(
                candidates,
                OPPORTUNITY_PROFIT_CAP_QUANTILE,
            )
            capped = _cap_market_curves_by_category(candidates, quantile=OPPORTUNITY_PROFIT_CAP_QUANTILE)
            category_supplier_min = _effective_category_supplier_min(capped, supplier_min, per_category)
            chosen = _assign_stratified_sample(
                capped,
                per_category,
                rng,
                selection_supplier_counts,
                supplier_cap,
                category_supplier_min,
                supplier_min,
                selected_supplier_counts,
                future_supplier_counts,
            )
            if len(chosen) < per_category:
                remaining_capacity = _remaining_supplier_capacity(
                    candidates,
                    selection_supplier_counts,
                    supplier_cap,
                )
                stats = capacity_stats.get(category, {})
                raise ValueError(
                    f"category {category!r} cannot satisfy supplier cap with {per_category} rows; "
                    f"selected={len(chosen)}, remaining_supplier_capacity={remaining_capacity}, "
                    f"initial_supplier_capacity={stats.get('supplier_capacity')}, "
                    f"distinct_suppliers={stats.get('distinct_suppliers')}"
                )
            for row in chosen[:per_category]:
                row_idx = int(row["_source_row_idx"])
                supplier_id = _supplier_id(row)
                selected_by_row_idx[row_idx] = str(row["_stratum"])
                selected_row_info[row_idx] = (str(row["_category"]), supplier_id)
        _repair_open_supplier_blocks(
            conn,
            selected_by_row_idx,
            selected_row_info,
            selection_supplier_counts,
            set(categories),
            supplier_min,
        )
        _validate_supplier_count_values(selection_supplier_counts, supplier_min, supplier_cap)
        return selected_by_row_idx, opportunity_cap_by_category
    finally:
        conn.close()


def _category_selection_order(
    conn: sqlite3.Connection,
    supplier_counts: Counter,
    categories: list[str],
    supplier_min: int,
    supplier_cap: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    stats: dict[str, dict[str, int]] = {}
    for category in categories:
        by_supplier: Counter = Counter()
        for (supplier_id,) in conn.execute(
            "SELECT member_id FROM candidates WHERE category=?",
            (category,),
        ):
            supplier = str(supplier_id)
            if supplier_counts[supplier] >= supplier_min:
                by_supplier[supplier] += 1
        supplier_capacity = sum(min(count, supplier_cap) for count in by_supplier.values())
        stats[category] = {
            "candidate_count": sum(by_supplier.values()),
            "distinct_suppliers": len(by_supplier),
            "supplier_capacity": supplier_capacity,
        }
    ordered = sorted(
        categories,
        key=lambda category: (
            stats[category]["supplier_capacity"],
            stats[category]["distinct_suppliers"],
            stats[category]["candidate_count"],
            category,
        ),
    )
    return ordered, stats


def _category_supplier_counts(conn: sqlite3.Connection, categories: list[str]) -> dict[str, Counter]:
    out = {category: Counter() for category in categories}
    for category in categories:
        for (supplier_id,) in conn.execute(
            "SELECT member_id FROM candidates WHERE category=?",
            (category,),
        ):
            out[category][str(supplier_id)] += 1
    return out


def _drop_non_positive_counts(counts: Counter) -> None:
    for key in [key for key, value in counts.items() if value <= 0]:
        del counts[key]


def _remaining_supplier_capacity(
    candidates: list[dict],
    supplier_counts: dict[str, int],
    supplier_cap: int,
) -> int:
    by_supplier: Counter = Counter(str(row["member_id"]) for row in candidates)
    capacity = 0
    for supplier, count in by_supplier.items():
        remaining = max(0, supplier_cap - int(supplier_counts.get(supplier, 0)))
        capacity += min(count, remaining)
    return capacity


def _streaming_load_selected_rows(
    bench_csv: str,
    selected_by_row_idx: dict[int, str],
    curve_cap_by_category: dict[str, float],
    params: dict,
    hard_cap: float,
    *,
    progress_every: int = 0,
) -> list[dict]:
    out = []
    remaining = set(selected_by_row_idx)
    with open(bench_csv, newline="", encoding="utf-8") as f:
        for row_idx, raw_row in enumerate(csv.DictReader(f), start=1):
            if row_idx not in remaining:
                _progress("load_selected", row_idx, progress_every)
                continue
            row = _parse_bench_row(raw_row, params)
            if row is None:
                raise ValueError(f"selected row {row_idx} no longer passes parsing")
            if row["_price"] > hard_cap or row["_ref_price"] > hard_cap:
                raise ValueError(f"selected row {row_idx} no longer passes price cap")
            _apply_market_curve_cap(row, curve_cap_by_category.get(row["_category"], 0.0))
            row["_stratum"] = selected_by_row_idx[row_idx]
            out.append(row)
            remaining.remove(row_idx)
            if not remaining:
                break
            _progress("load_selected", row_idx, progress_every)
    if remaining:
        raise ValueError(f"could not reload {len(remaining)} selected rows from bench_csv")
    return out


def _read_bench_rows(path: str, params: dict | None = None) -> list[dict]:
    profile_params = params or load_default_generation_params()
    parsed: list[dict] = []
    prices: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw_row in csv.DictReader(f):
            row = _parse_bench_row(raw_row, profile_params)
            if row is None:
                continue
            parsed.append(row)
            prices.extend([row["_price"], row["_ref_price"]])

    if not parsed:
        return []
    cap = _quantile(prices, 0.995)
    hard_cap = max(cap, 1.0) * 2.0
    return [r for r in parsed if r["_price"] <= hard_cap and r["_ref_price"] <= hard_cap]


def _parse_bench_row_light(raw_row: dict, params: dict) -> dict | None:
    source_category = _required_text(raw_row.get("cate_level1_name"))
    category = translate_category(source_category or "", params)
    item_id = _required_text(raw_row.get("item_id"))
    title = _required_text(raw_row.get("title"))
    member_id = _required_text(raw_row.get("member_id"))
    company_name = _required_text(raw_row.get("company_name"))
    if not all([source_category, category, item_id, title, member_id, company_name]):
        return None
    if not _product_id_base(item_id):
        return None

    price = _num(raw_row.get("reserve_price"))
    ref_price = _num(raw_row.get("ref_price"))
    if price is None or ref_price is None or price <= 0 or ref_price <= 0:
        return None
    if ref_price < price:
        return None
    if ref_price / price >= MAX_REF_PRICE_RATIO:
        return None

    order_cnt = _required_nonnegative(raw_row.get("order_cnt"))
    satisfied = _required_rate(raw_row.get("satisfied_rate_std_001"))
    repeat_rate = _required_rate(raw_row.get("e_repeat_rate_6m_001_slr"))
    age_years = _required_nonnegative(raw_row.get("tp_service_y_cnt"))
    if any(v is None for v in (order_cnt, satisfied, repeat_rate, age_years)):
        return None

    good_rate, good_rate_source = _quality_rate(
        raw_row.get("good_rate"),
        raw_row=raw_row,
        category=category,
        order_cnt=order_cnt,
        satisfied=satisfied,
        repeat_rate=repeat_rate,
    )
    pt_rate, pt_rate_source = _problem_transaction_rate(
        raw_row.get("pt_rate"),
        raw_row=raw_row,
        category=category,
        good_rate=good_rate,
        satisfied=satisfied,
        repeat_rate=repeat_rate,
    )
    curve_sum = _parse_daily_curve_sum(raw_row.get("converted_order_cnt_str"))
    if curve_sum is None:
        return None

    return {
        "item_id": item_id,
        "member_id": member_id,
        "_category": category,
        "_price": price,
        "_ref_price": ref_price,
        "_curve_sum": curve_sum,
        "_good_rate": good_rate,
        "_pt_rate": pt_rate,
        "_good_rate_source": good_rate_source,
        "_pt_rate_source": pt_rate_source,
    }


def _parse_bench_row(raw_row: dict, params: dict) -> dict | None:
    source_category = _required_text(raw_row.get("cate_level1_name"))
    category = translate_category(source_category or "", params)
    item_id = _required_text(raw_row.get("item_id"))
    title = _required_text(raw_row.get("title"))
    member_id = _required_text(raw_row.get("member_id"))
    company_name = _required_text(raw_row.get("company_name"))
    if not all([source_category, category, item_id, title, member_id, company_name]):
        return None
    if not _product_id_base(item_id):
        return None

    price = _num(raw_row.get("reserve_price"))
    ref_price = _num(raw_row.get("ref_price"))
    if price is None or ref_price is None or price <= 0 or ref_price <= 0:
        return None
    if ref_price < price:
        return None
    if ref_price / price >= MAX_REF_PRICE_RATIO:
        return None

    order_cnt = _required_nonnegative(raw_row.get("order_cnt"))
    satisfied = _required_rate(raw_row.get("satisfied_rate_std_001"))
    repeat_rate = _required_rate(raw_row.get("e_repeat_rate_6m_001_slr"))
    age_years = _required_nonnegative(raw_row.get("tp_service_y_cnt"))
    if any(v is None for v in (order_cnt, satisfied, repeat_rate, age_years)):
        return None

    good_rate, good_rate_source = _quality_rate(
        raw_row.get("good_rate"),
        raw_row=raw_row,
        category=category,
        order_cnt=order_cnt,
        satisfied=satisfied,
        repeat_rate=repeat_rate,
    )
    pt_rate, pt_rate_source = _problem_transaction_rate(
        raw_row.get("pt_rate"),
        raw_row=raw_row,
        category=category,
        good_rate=good_rate,
        satisfied=satisfied,
        repeat_rate=repeat_rate,
    )

    curve = _parse_daily_curve(raw_row.get("converted_order_cnt_str"))
    if curve is None:
        return None

    row = dict(raw_row)
    row["cate_level1_name"] = source_category
    row["item_id"] = item_id
    row["title"] = title
    row["member_id"] = member_id
    row["company_name"] = company_name
    row["_source_category"] = source_category
    row["_category"] = category
    row["_price"] = price
    row["_ref_price"] = ref_price
    row["_order_cnt"] = order_cnt
    row["_curve"] = curve
    row["_curve_sum"] = sum(curve)
    row["_good_rate"] = good_rate
    row["_pt_rate"] = pt_rate
    row["_good_rate_source"] = good_rate_source
    row["_pt_rate_source"] = pt_rate_source
    row["_satisfied"] = satisfied
    row["_repeat_rate"] = repeat_rate
    row["_age_years"] = age_years
    return row


def _filter_supplier_item_counts(rows: list[dict], min_count: int, max_count: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[_supplier_id(row)] += 1
    return [row for row in rows if counts[_supplier_id(row)] >= min_count]


def _filter_positive_market_curves(rows: list[dict]) -> list[dict]:
    return [row for row in rows if float(row.get("_curve_sum", 0.0)) > 0.0]


def _select_categories(rows: list[dict]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["_category"]] += 1
    ranked = sorted(counts, key=lambda c: (-counts[c], c))
    if not ranked:
        raise ValueError("no categories found in bench_csv")
    return ranked[:10]


def _sample_rows(
    rows: list[dict],
    categories: list[str],
    category_quotas: dict[str, int],
    rng: random.Random,
    supplier_cap: int,
    supplier_min: int = 1,
) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    eligible_rows: list[dict] = []
    for row in rows:
        category = row["_category"]
        if category in categories:
            row = dict(row)
            eligible_rows.append(row)
            by_category[category].append(row)

    supplier_candidate_counts = Counter(_supplier_id(row) for row in eligible_rows)
    future_supplier_counts = Counter(supplier_candidate_counts)
    supplier_counts: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for category in categories:
        per_category = int(category_quotas.get(category, 0))
        if per_category <= 0:
            continue
        future_supplier_counts.subtract(Counter(_supplier_id(row) for row in by_category.get(category, [])))
        _drop_non_positive_counts(future_supplier_counts)
        candidates = _cap_market_curves_by_category(
            by_category.get(category, []),
            quantile=OPPORTUNITY_PROFIT_CAP_QUANTILE,
        )
        if len(candidates) < per_category:
            raise ValueError(f"category {category!r} has {len(candidates)} candidates, need {per_category}")
        category_supplier_min = _effective_category_supplier_min(candidates, supplier_min, per_category)
        chosen = _assign_stratified_sample(
            candidates,
            per_category,
            rng,
            supplier_counts,
            supplier_cap,
            category_supplier_min,
            supplier_min,
            supplier_candidate_counts,
            future_supplier_counts,
        )
        if len(chosen) < per_category:
            raise ValueError(f"category {category!r} cannot satisfy supplier cap with {per_category} rows")
        selected.extend(chosen[:per_category])
    _validate_final_supplier_counts(selected, supplier_min, supplier_cap)
    return selected


def _cap_market_curves_by_category(rows: list[dict], quantile: float = OPPORTUNITY_PROFIT_CAP_QUANTILE) -> list[dict]:
    cap = _opportunity_profit_cap(rows, quantile)
    if cap <= 0:
        return [dict(r) for r in rows]
    out = []
    for row in rows:
        copied = dict(row)
        opportunity = _opportunity_profit_365(copied)
        if opportunity > cap:
            scale = cap / opportunity
            if "_curve" in copied:
                copied["_curve"] = [float(v) * scale for v in copied["_curve"]]
                copied["_curve_sum"] = sum(copied["_curve"])
            else:
                copied["_curve_sum"] = float(copied["_curve_sum"]) * scale
        out.append(copied)
    return out


def gross_opportunity_profit_365(row: dict) -> float:
    return _opportunity_profit_365(row)


def _opportunity_profit_365(row: dict) -> float:
    price = float(row["_price"])
    effective_ref_price = min(float(row["_ref_price"]), price * REF_PRICE_CAP_RATIO)
    unit_margin = max(effective_ref_price - price, 0.0)
    return max(float(row.get("_curve_sum", 0.0)), 0.0) * unit_margin


def _opportunity_profit_cap(rows: list[dict], quantile: float = OPPORTUNITY_PROFIT_CAP_QUANTILE) -> float:
    opportunities = [_opportunity_profit_365(r) for r in rows]
    positive = [value for value in opportunities if value > 0.0]
    return _quantile(positive, quantile)


def _apply_market_curve_cap(row: dict, opportunity_cap: float) -> None:
    opportunity = _opportunity_profit_365(row)
    if opportunity_cap <= 0.0 or opportunity <= opportunity_cap:
        return
    scale = opportunity_cap / opportunity
    row["_curve"] = [float(v) * scale for v in row["_curve"]]
    row["_curve_sum"] = sum(row["_curve"])


def expected_unit_profit_at_ref(
    cost: float,
    ref_price: float,
    rates: dict[str, float],
    *,
    refund_penalty_amount: float = REFUND_PENALTY_AMOUNT,
    bad_review_penalty_amount: float = BAD_REVIEW_PENALTY_AMOUNT,
) -> float:
    margin = float(ref_price) - float(cost)
    cancel = _clamp(float(rates.get("cancel_rate", 0.0)), 0.0, 1.0)
    refund = _clamp(float(rates.get("refund_rate", 0.0)), 0.0, 1.0)
    only_refund = _clamp(float(rates.get("only_refund_rate", 0.0)), 0.0, 1.0)
    bad_review = _clamp(float(rates.get("bad_review_rate", 0.0)), 0.0, 1.0)
    normal = max(0.0, 1.0 - cancel - refund - only_refund - bad_review)
    return (
        normal * margin
        + cancel * 0.0
        + refund * -float(refund_penalty_amount)
        + only_refund * -float(cost)
        + bad_review * (margin - float(bad_review_penalty_amount))
    )


def expected_net_profit_365_at_ref(row: dict, rates: dict[str, float]) -> float:
    price = float(row["_price"])
    effective_ref_price = min(float(row["_ref_price"]), price * REF_PRICE_CAP_RATIO)
    unit_profit = expected_unit_profit_at_ref(price, effective_ref_price, rates)
    return max(float(row.get("_curve_sum", 0.0)), 0.0) * unit_profit


def _apply_expected_net_profit_cap(
    row: dict,
    rates: dict[str, float],
    cap: float = EXPECTED_NET_PROFIT_CAP_365,
) -> None:
    expected_net = expected_net_profit_365_at_ref(row, rates)
    if cap <= 0.0 or expected_net <= cap:
        return
    scale = cap / expected_net
    if "_curve" in row:
        row["_curve"] = [float(v) * scale for v in row["_curve"]]
        row["_curve_sum"] = sum(row["_curve"])
    else:
        row["_curve_sum"] = float(row.get("_curve_sum", 0.0)) * scale


def _quota_counts(total: int, ratios: dict[str, float] = STRATUM_RATIOS) -> dict[str, int]:
    raw = {name: max(0.0, float(ratio)) * total for name, ratio in ratios.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = total - sum(quotas.values())
    remainders = sorted(
        ratios,
        key=lambda name: (raw[name] - quotas[name], ratios[name], name),
        reverse=True,
    )
    for name in remainders[:remaining]:
        quotas[name] += 1
    return quotas


def _assign_stratified_sample(
    candidates: list[dict],
    per_category: int,
    rng: random.Random,
    supplier_counts: dict[str, int],
    supplier_cap: int,
    supplier_min: int = 1,
    supplier_final_min: int | None = None,
    supplier_candidate_counts: Counter | None = None,
    supplier_future_counts: Counter | None = None,
) -> list[dict]:
    supplier_final_min = supplier_min if supplier_final_min is None else supplier_final_min
    quotas = _quota_counts(per_category)
    remaining_quotas = dict(quotas)
    pools = _stratum_candidate_pools(candidates)
    weighted_pools = {stratum: _sample_weighted_without_replacement(pool, rng) for stratum, pool in pools.items()}
    weighted_supplier_indexes = {stratum: _supplier_pool_index(pool) for stratum, pool in weighted_pools.items()}
    pool_cursors = {stratum: 0 for stratum in pools}
    fallback_pools: dict[str, list[list[dict]]] = {}
    fallback_supplier_indexes: dict[str, list[dict[str, list[dict]]]] = {}
    fallback_cursors: dict[str, list[int]] = {}
    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    _close_open_supplier_blocks(
        candidates,
        per_category,
        remaining_quotas,
        chosen,
        chosen_ids,
        supplier_counts,
        supplier_cap,
        supplier_final_min,
    )

    for stratum, quota in quotas.items():
        while quota > 0 and remaining_quotas.get(stratum, 0) > 0 and len(chosen) < per_category:
            added, pool_cursors[stratum] = _take_from_pool(
                weighted_pools[stratum],
                weighted_supplier_indexes[stratum],
                pool_cursors[stratum],
                stratum,
                per_category,
                remaining_quotas,
                chosen,
                chosen_ids,
                supplier_counts,
                supplier_cap,
                supplier_min,
                supplier_final_min,
                supplier_candidate_counts,
                supplier_future_counts,
            )
            if added:
                continue
            if stratum not in fallback_pools:
                fallback_pools[stratum] = _fallback_pools(candidates, stratum)
                fallback_supplier_indexes[stratum] = [_supplier_pool_index(pool) for pool in fallback_pools[stratum]]
                fallback_cursors[stratum] = [0] * len(fallback_pools[stratum])
            for idx, fallback_pool in enumerate(fallback_pools[stratum]):
                added, fallback_cursors[stratum][idx] = _take_from_pool(
                    fallback_pool,
                    fallback_supplier_indexes[stratum][idx],
                    fallback_cursors[stratum][idx],
                    stratum,
                    per_category,
                    remaining_quotas,
                    chosen,
                    chosen_ids,
                    supplier_counts,
                    supplier_cap,
                    supplier_min,
                    supplier_final_min,
                    supplier_candidate_counts,
                    supplier_future_counts,
                )
                if added:
                    break
            if not added:
                break

    if len(chosen) < per_category:
        fill_pool = _sample_weighted_without_replacement(candidates, rng)
        fill_supplier_index = _supplier_pool_index(fill_pool)
        fill_cursor = 0
        while len(chosen) < per_category:
            preferred = _next_remaining_stratum(remaining_quotas)
            added, fill_cursor = _take_from_pool(
                fill_pool,
                fill_supplier_index,
                fill_cursor,
                preferred,
                per_category,
                remaining_quotas,
                chosen,
                chosen_ids,
                supplier_counts,
                supplier_cap,
                supplier_min,
                supplier_final_min,
                supplier_candidate_counts,
                supplier_future_counts,
            )
            if not added:
                break

    return chosen


def _stratum_candidate_pools(candidates: list[dict]) -> dict[str, list[dict]]:
    curve_sums = sorted(float(r["_curve_sum"]) for r in candidates)
    positive_curve_sums = sorted(v for v in curve_sums if v > 0.0)
    pos_q20 = _value_quantile(positive_curve_sums, 0.20)
    pos_q25 = _value_quantile(positive_curve_sums, 0.25)
    pos_q60 = _value_quantile(positive_curve_sums, 0.60)
    pos_q70 = _value_quantile(positive_curve_sums, 0.70)
    pos_q75 = _value_quantile(positive_curve_sums, 0.75)
    pos_q95 = _value_quantile(positive_curve_sums, 0.95)
    low_threshold = pos_q20 if positive_curve_sums else 0.0

    pools = {name: [] for name in STRATUM_RATIOS}
    for row in candidates:
        curve_sum = float(row["_curve_sum"])
        rating = _visible_rating(row)
        source_risk = _source_risk(row)
        has_demand = curve_sum > 0.0
        if has_demand and pos_q60 <= curve_sum <= pos_q95 and rating >= 4.5 and source_risk < 0.08:
            pools["good"].append(row)
        if has_demand and pos_q25 <= curve_sum <= pos_q75 and source_risk < 0.12:
            pools["safe"].append(row)
        if has_demand and curve_sum >= pos_q70 and rating >= 4.5:
            pools["trap"].append(row)
        if curve_sum <= low_threshold and source_risk < 0.10:
            pools["mediocre"].append(row)
        if source_risk >= 0.20 or rating <= 3.6:
            pools["minefield"].append(row)
    return pools


def _fallback_pools(candidates: list[dict], stratum: str) -> list[list[dict]]:
    by_curve = sorted(candidates, key=lambda r: float(r["_curve_sum"]))
    positive = [r for r in candidates if float(r["_curve_sum"]) > 0.0]
    list(reversed(by_curve))
    high_curve_positive = sorted(positive, key=lambda r: float(r["_curve_sum"]), reverse=True)
    sorted(candidates, key=lambda r: (_visible_rating(r), float(r["_curve_sum"])), reverse=True)
    high_rating_positive = sorted(positive, key=lambda r: (_visible_rating(r), float(r["_curve_sum"])), reverse=True)
    low_risk = [r for r in candidates if _source_risk(r) < 0.12]
    low_risk_positive = [r for r in positive if _source_risk(r) < 0.12]
    high_risk = [r for r in candidates if _source_risk(r) >= 0.12 or _visible_rating(r) <= 3.8]
    if stratum == "good":
        return [high_rating_positive, low_risk_positive, high_curve_positive, positive]
    if stratum == "safe":
        return [low_risk_positive, positive]
    if stratum == "trap":
        return [high_rating_positive, high_curve_positive, positive]
    if stratum == "mediocre":
        return [by_curve, low_risk, candidates]
    if stratum == "minefield":
        return [high_risk, candidates]
    return [candidates]


def _sample_weighted_without_replacement(pool: list[dict], rng: random.Random) -> list[dict]:
    decorated = []
    for row in pool:
        weight = max(math.sqrt(max(float(row.get("_curve_sum", 0.0)), 0.0)), 1e-6)
        key = rng.random() ** (1.0 / weight)
        decorated.append((key, row))
    return [row for _, row in sorted(decorated, key=lambda x: x[0], reverse=True)]


def _supplier_pool_index(pool_rows: list[dict]) -> dict[str, list[dict]]:
    by_supplier: dict[str, list[dict]] = defaultdict(list)
    for row in pool_rows:
        by_supplier[_supplier_id(row)].append(row)
    return by_supplier


def _max_supplier_count(rows: list[dict]) -> int:
    if not rows:
        return 0
    return max(Counter(_supplier_id(row) for row in rows).values())


def _effective_category_supplier_min(rows: list[dict], desired_min: int, target_rows: int) -> int:
    counts = list(Counter(_supplier_id(row) for row in rows).values())
    upper = min(desired_min, target_rows, max(counts, default=0))
    if target_rows > 1000:
        return upper
    for candidate_min in range(upper, 0, -1):
        reachable = {0}
        for count in counts:
            capped = min(count, target_rows)
            options = range(candidate_min, capped + 1)
            reachable |= {total + option for total in reachable for option in options if total + option <= target_rows}
            if target_rows in reachable:
                return candidate_min
    return 1


def _close_open_supplier_blocks(
    candidates: list[dict],
    category_target_total: int,
    remaining_quotas: dict[str, int],
    chosen: list[dict],
    chosen_ids: set[str],
    supplier_counts: dict[str, int],
    supplier_cap: int,
    supplier_final_min: int,
) -> None:
    if supplier_final_min <= 1:
        return
    supplier_pool_index = _supplier_pool_index(candidates)
    open_suppliers = sorted(
        (supplier for supplier, count in supplier_counts.items() if 0 < int(count) < supplier_final_min),
        key=lambda supplier: (supplier_final_min - int(supplier_counts.get(supplier, 0)), supplier),
        reverse=True,
    )
    for supplier in open_suppliers:
        current = int(supplier_counts.get(supplier, 0))
        needed = supplier_final_min - current
        if needed <= 0:
            continue
        remaining_slots = category_target_total - len(chosen)
        if remaining_slots <= 0:
            return
        available_rows = []
        for row in supplier_pool_index.get(supplier, []):
            row_key = str(row.get("item_id") or id(row))
            if row_key not in chosen_ids:
                available_rows.append(row)
        take_count = min(needed, len(available_rows), remaining_slots, supplier_cap - current)
        if take_count <= 0:
            continue
        for block_row in available_rows[:take_count]:
            preferred = _next_remaining_stratum(remaining_quotas)
            label = _consume_stratum_quota(preferred, remaining_quotas, block_row)
            copied = dict(block_row)
            copied["_stratum"] = label
            chosen.append(copied)
            chosen_ids.add(str(block_row.get("item_id") or id(block_row)))
            supplier_counts[supplier] += 1


def _take_from_pool(
    pool_rows: list[dict],
    supplier_pool_index: dict[str, list[dict]],
    start_idx: int,
    stratum: str,
    category_target_total: int,
    remaining_quotas: dict[str, int],
    chosen: list[dict],
    chosen_ids: set[str],
    supplier_counts: dict[str, int],
    supplier_cap: int,
    supplier_min: int = 1,
    supplier_final_min: int | None = None,
    supplier_candidate_counts: Counter | None = None,
    supplier_future_counts: Counter | None = None,
) -> tuple[int, int]:
    supplier_final_min = supplier_min if supplier_final_min is None else supplier_final_min
    idx = start_idx
    while idx < len(pool_rows):
        row = pool_rows[idx]
        idx += 1
        if len(chosen) >= category_target_total:
            break
        row_key = str(row.get("item_id") or id(row))
        if row_key in chosen_ids:
            continue
        supplier = _supplier_id(row)
        block = _supplier_selection_block(
            supplier_pool_index,
            supplier,
            chosen_ids,
            supplier_counts,
            supplier_cap,
            supplier_min,
            supplier_final_min,
            category_target_total - len(chosen),
            supplier_candidate_counts,
            supplier_future_counts,
        )
        if not block:
            continue
        for block_row in block:
            label = _consume_stratum_quota(stratum, remaining_quotas, block_row)
            copied = dict(block_row)
            copied["_stratum"] = label
            chosen.append(copied)
            chosen_ids.add(str(block_row.get("item_id") or id(block_row)))
            supplier_counts[supplier] += 1
        return len(block), idx
    return 0, idx


def _consume_stratum_quota(preferred: str, remaining_quotas: dict[str, int], row: dict) -> str:
    allowed = None
    if float(row.get("_curve_sum", 0.0)) <= 0.0:
        allowed = {"mediocre", "minefield"}
    if (allowed is None or preferred in allowed) and remaining_quotas.get(preferred, 0) > 0:
        remaining_quotas[preferred] -= 1
        return preferred
    available = [
        (remaining, stratum)
        for stratum, remaining in remaining_quotas.items()
        if remaining > 0 and (allowed is None or stratum in allowed)
    ]
    if not available:
        if allowed is not None:
            return "mediocre"
        return preferred
    _, stratum = max(available)
    remaining_quotas[stratum] -= 1
    return stratum


def _next_remaining_stratum(remaining_quotas: dict[str, int]) -> str:
    available = [(remaining, stratum) for stratum, remaining in remaining_quotas.items() if remaining > 0]
    if not available:
        return "mediocre"
    _, stratum = max(available)
    return stratum


def _supplier_selection_block(
    supplier_pool_index: dict[str, list[dict]],
    supplier: str,
    chosen_ids: set[str],
    supplier_counts: dict[str, int],
    supplier_cap: int,
    supplier_min: int,
    supplier_final_min: int,
    remaining_category_slots: int,
    supplier_candidate_counts: Counter | None = None,
    supplier_future_counts: Counter | None = None,
) -> list[dict]:
    current = int(supplier_counts.get(supplier, 0))
    if current >= supplier_cap:
        return []
    candidate_total = int((supplier_candidate_counts or {}).get(supplier, 0))
    if candidate_total <= 0:
        candidate_total = current + len(supplier_pool_index.get(supplier, []))
    if candidate_total < supplier_final_min:
        return []

    available_rows: list[dict] = []
    seen: set[str] = set()
    for row in supplier_pool_index.get(supplier, []):
        row_key = str(row.get("item_id") or id(row))
        if row_key in chosen_ids or row_key in seen:
            continue
        available_rows.append(row)
        seen.add(row_key)
    available = len(available_rows)
    if available <= 0 or remaining_category_slots <= 0:
        return []

    if current <= 0:
        future = int((supplier_future_counts or {}).get(supplier, 0))
        if (
            supplier_future_counts is not None
            and available < supplier_final_min
            and available + future < supplier_final_min
        ):
            return []
        needed = supplier_min
    elif current < supplier_final_min:
        needed = supplier_final_min - current
    else:
        needed = 1
    needed = min(needed, available, remaining_category_slots, supplier_cap - current)
    if needed <= 0:
        return []

    return available_rows[:needed]


def _repair_open_supplier_blocks(
    conn: sqlite3.Connection,
    selected_by_row_idx: dict[int, str],
    selected_row_info: dict[int, tuple[str, str]],
    supplier_counts: dict[str, int],
    selected_categories: set[str],
    supplier_min: int,
) -> None:
    while True:
        small_suppliers = [
            (supplier, int(count)) for supplier, count in supplier_counts.items() if 0 < int(count) < supplier_min
        ]
        if not small_suppliers:
            return
        progress = False
        for supplier, count in sorted(small_suppliers, key=lambda item: (item[1], item[0])):
            needed = supplier_min - count
            if needed <= 0:
                continue
            for candidate in conn.execute(
                "SELECT row_idx, category FROM candidates WHERE member_id=? ORDER BY category, row_idx",
                (supplier,),
            ):
                if needed <= 0:
                    break
                row_idx = int(candidate["row_idx"])
                category = str(candidate["category"])
                if category not in selected_categories or row_idx in selected_by_row_idx:
                    continue
                donor = _find_supplier_repair_donor(
                    selected_row_info,
                    supplier_counts,
                    category,
                    supplier,
                    supplier_min,
                )
                if donor is None:
                    continue
                donor_row_idx, donor_supplier = donor
                replacement_stratum = selected_by_row_idx[donor_row_idx]
                del selected_by_row_idx[donor_row_idx]
                del selected_row_info[donor_row_idx]
                supplier_counts[donor_supplier] -= 1

                selected_by_row_idx[row_idx] = replacement_stratum
                selected_row_info[row_idx] = (category, supplier)
                supplier_counts[supplier] = int(supplier_counts.get(supplier, 0)) + 1
                needed -= 1
                progress = True
        if not progress:
            return


def _find_supplier_repair_donor(
    selected_row_info: dict[int, tuple[str, str]],
    supplier_counts: dict[str, int],
    category: str,
    target_supplier: str,
    supplier_min: int,
) -> tuple[int, str] | None:
    candidates = [
        (int(supplier_counts.get(supplier, 0)), row_idx, supplier)
        for row_idx, (row_category, supplier) in selected_row_info.items()
        if row_category == category
        and supplier != target_supplier
        and int(supplier_counts.get(supplier, 0)) > supplier_min
    ]
    if not candidates:
        return None
    _, row_idx, supplier = max(candidates)
    return row_idx, supplier


def _validate_final_supplier_counts(rows: list[dict], supplier_min: int, supplier_cap: int) -> None:
    counts = Counter(_supplier_id(row) for row in rows)
    _validate_supplier_count_values(counts, supplier_min, supplier_cap)


def _validate_supplier_count_values(counts: dict[str, int], supplier_min: int, supplier_cap: int) -> None:
    if not counts:
        return
    too_small = {supplier: count for supplier, count in counts.items() if count < supplier_min}
    too_large = {supplier: count for supplier, count in counts.items() if count > supplier_cap}
    if too_small or too_large:
        raise ValueError(
            "final supplier block counts violate bounds: "
            f"min={supplier_min}, cap={supplier_cap}, "
            f"too_small={dict(sorted(too_small.items())[:5])}, "
            f"too_large={dict(sorted(too_large.items())[:5])}"
        )


def _visible_rating(row: dict) -> float:
    return _clamp(1.0 + float(row["_good_rate"]) * 4.0, 1.0, 5.0)


def _source_risk(row: dict) -> float:
    return _clamp(float(row.get("_pt_rate", 0.0)) + (1.0 - float(row.get("_good_rate", 1.0))), 0.0, 1.0)


def _build_products(rows: list[dict], seed: int, params: dict) -> list[dict]:
    profiles = _supplier_profiles(rows)
    risk_ranges = params["risk_ranges"]
    supplier_ranges = normalize_supplier_ranges(
        params["supplier_ranges"],
        {"max_promised_ship_hours": 48},
    )
    used_ids: dict[str, int] = defaultdict(int)
    products = []
    for idx, row in enumerate(rows):
        row = dict(row)
        if "_curve" in row:
            row["_curve"] = [float(v) for v in row["_curve"]]
        supplier_id = _supplier_id(row)
        profile = profiles[supplier_id]
        product_id = _product_id(str(row.get("item_id") or "item"), used_ids)
        rng = derive_rng(seed, "data_gen", "private_real_product", idx, product_id)
        operational = sample_operational_fields(rng, supplier_ranges)
        risk_event = sample_risk_event_fields(rng, risk_ranges, supplier_ranges)
        stratum = str(row.get("_stratum") or "safe")
        base_price = float(row["_price"])
        raw_ref_price = float(row["_ref_price"])
        ref_price = min(raw_ref_price, base_price * REF_PRICE_CAP_RATIO)
        stratum_risk = _calibrated_risk_profile(
            rng,
            stratum,
            price=base_price,
            ref_price=ref_price,
            opportunity_rank=1.0,
        )
        products.append(
            {
                "product_id": product_id,
                "name": _clean_name(str(row["title"])),
                "quantity": operational["quantity"],
                "price": round(base_price, 4),
                "base_price": round(base_price, 4),
                "ref_price": round(ref_price, 4),
                "raw_ref_price": round(raw_ref_price, 4),
                "supplier_id": supplier_id,
                "supplier_name": profile["supplier_name"],
                "ship_hours": operational["ship_hours"],
                "logistics_hours": operational["logistics_hours"],
                "category": row["_category"],
                "historical_avg_rating": round(_visible_rating(row), 4),
                "shop_rating": profile["shop_rating"],
                "return_buyer_rate": profile["return_buyer_rate"],
                "supplier_age_years": profile["supplier_age_years"],
                "cancel_rate": round(stratum_risk["cancel_rate"], 6),
                "refund_rate": round(stratum_risk["refund_rate"], 6),
                "only_refund_rate": round(stratum_risk["only_refund_rate"], 6),
                "bad_review_rate": round(stratum_risk["bad_review_rate"], 6),
                "max_quantity": operational["max_quantity"],
                "hourly_increment": operational["hourly_increment"],
                "timeout_rate": round(risk_event["timeout_rate"], 6),
                "price_change_rate": round(risk_event["price_change_rate"], 6),
                "supplier_delist_rate": round(risk_event["supplier_delist_rate"], 6),
                "elasticity": round(_ref_implied_elasticity(base_price, ref_price), 4),
                "market_curve": json.dumps(_market_curve(row), ensure_ascii=False),
                "stratum": stratum,
                "good_rate_source": str(row.get("_good_rate_source") or "raw"),
                "pt_rate_source": str(row.get("_pt_rate_source") or "raw"),
            }
        )
    _apply_final_profit_controls(products, seed)
    return sorted(products, key=lambda r: (r["category"], r["product_id"]))


def _opportunity_ranks(rows: list[dict]) -> dict[int, float]:
    positive = [
        (gross_opportunity_profit_365(row), idx)
        for idx, row in enumerate(rows)
        if gross_opportunity_profit_365(row) > 0.0
    ]
    positive.sort(key=lambda item: (-item[0], item[1]))
    n = max(1, len(rows))
    ranks: dict[int, float] = {}
    group_start = 0
    while group_start < len(positive):
        opportunity = positive[group_start][0]
        group_end = group_start + 1
        while group_end < len(positive) and positive[group_end][0] == opportunity:
            group_end += 1
        rank_value = (group_start + 0.5) / n
        for _, idx in positive[group_start:group_end]:
            ranks[idx] = rank_value
        group_start = group_end
    return ranks


def _apply_final_profit_risk_overlay(products: list[dict], seed: int) -> None:
    _apply_final_profit_density_calibration(products, seed)


def _apply_final_profit_density_calibration(products: list[dict], seed: int) -> None:
    rank_rows = []
    for product in products:
        rank_rows.append(
            {
                "_price": float(product["price"]),
                "_ref_price": float(product["ref_price"]),
                "_curve_sum": sum(json.loads(str(product["market_curve"]))),
            }
        )
    ranks = _opportunity_ranks(rank_rows)
    for idx, product in enumerate(products):
        opportunity_rank = ranks.get(idx, 1.0)
        rng = derive_rng(
            seed,
            "data_gen",
            "private_real_final_profit_density",
            product["product_id"],
        )
        stratum = str(product.get("stratum") or "safe")
        probabilities = _risk_group_probabilities(opportunity_rank, stratum)
        risk_group = _sample_risk_group(rng, probabilities)
        target = _sample_order_risk_for_group(rng, risk_group)
        raised = _split_order_risk_components(
            rng,
            target,
            float(product["price"]),
        )
        for field in ("cancel_rate", "refund_rate", "only_refund_rate", "bad_review_rate"):
            product[field] = round(raised[field], 6)


def _apply_final_profit_controls(products: list[dict], seed: int) -> None:
    _apply_final_profit_density_calibration(products, seed)
    _apply_expected_net_profit_portfolio_cap(products)


def _risk_group_probabilities(opportunity_rank: float, stratum: str) -> dict[str, float]:
    profit_percentile = 1.0 - _clamp(float(opportunity_rank), 0.0, 1.0)
    base = _interpolated_risk_group_probabilities(profit_percentile)
    bias = STRATUM_RISK_DENSITY_BIAS.get(stratum, STRATUM_RISK_DENSITY_BIAS["safe"])
    weighted = {group: max(0.0, base[group] * float(bias.get(group, 1.0))) for group in RISK_GROUP_BOUNDS}
    return _normalize_probabilities(weighted)


def _interpolated_risk_group_probabilities(profit_percentile: float) -> dict[str, float]:
    percentile = _clamp(float(profit_percentile), 0.0, 1.0)
    anchors = RISK_DENSITY_ANCHORS
    if percentile <= anchors[0][0]:
        return dict(anchors[0][1])
    for idx in range(1, len(anchors)):
        lo_percentile, lo_probs = anchors[idx - 1]
        hi_percentile, hi_probs = anchors[idx]
        if percentile <= hi_percentile:
            span = max(hi_percentile - lo_percentile, 1e-9)
            weight = (percentile - lo_percentile) / span
            return {
                group: float(lo_probs[group]) + (float(hi_probs[group]) - float(lo_probs[group])) * weight
                for group in RISK_GROUP_BOUNDS
            }
    return dict(anchors[-1][1])


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in probabilities.values())
    if total <= 0.0:
        return {"low": 1.0 / 3.0, "medium": 1.0 / 3.0, "high": 1.0 / 3.0}
    return {group: max(0.0, float(probabilities.get(group, 0.0))) / total for group in RISK_GROUP_BOUNDS}


def _sample_risk_group(rng, probabilities: dict[str, float]) -> str:
    draw = float(rng.random())
    cumulative = 0.0
    for group in ("low", "medium", "high"):
        cumulative += float(probabilities[group])
        if draw <= cumulative:
            return group
    return "high"


def _sample_order_risk_for_group(rng, risk_group: str) -> float:
    lo, hi = RISK_GROUP_BOUNDS.get(risk_group, RISK_GROUP_BOUNDS["medium"])
    alpha, beta = RISK_GROUP_BETA_SHAPES.get(risk_group, RISK_GROUP_BETA_SHAPES["medium"])
    value = float(lo) + (float(hi) - float(lo)) * float(rng.beta(alpha, beta))
    return _clamp(value, 0.0, 0.90)


def _split_order_risk_components(rng, order_risk: float, price: float) -> dict[str, float]:
    target = _clamp(float(order_risk), 0.0, 0.90)
    cancel_rate = min(0.12, target * rand_range(rng, 0.05, 0.12))
    only_refund_cap = _only_refund_cap_for_price(price)
    if float(price) <= 10.0:
        only_refund_share = rand_range(rng, 0.08, 0.22)
    elif float(price) <= 50.0:
        only_refund_share = rand_range(rng, 0.03, 0.10)
    else:
        only_refund_share = rand_range(rng, 0.005, 0.035)
    only_refund_rate = min(only_refund_cap, target * only_refund_share)
    remaining = max(0.0, target - cancel_rate - only_refund_rate)
    refund_share = rand_range(rng, 0.48, 0.62)
    refund_rate = remaining * refund_share
    bad_review_rate = remaining - refund_rate
    return _cap_order_risk_total(
        {
            "cancel_rate": cancel_rate,
            "refund_rate": refund_rate,
            "only_refund_rate": only_refund_rate,
            "bad_review_rate": bad_review_rate,
        },
        0.90,
    )


def _apply_expected_net_profit_portfolio_cap(
    products: list[dict],
    *,
    top_n: int = EXPECTED_NET_PROFIT_TOP_N,
    p95_cap: float = EXPECTED_NET_PROFIT_TOP1000_P95_CAP_365,
    max_cap: float = EXPECTED_NET_PROFIT_CAP_365,
    max_iterations: int | None = None,
) -> None:
    if not products:
        return
    top_n = max(1, min(int(top_n), len(products)))
    p95_index = int(round((top_n - 1) * 0.95))
    allowed_above_p95_cap = max(0, top_n - (p95_index + 1))
    max_iterations = len(products) + 1 if max_iterations is None else max_iterations
    for product in products:
        _cap_product_expected_net_profit(product, max_cap)

    for _ in range(max_iterations):
        top_products = sorted(
            products,
            key=lambda product: (_product_gross_opportunity_profit_365(product), str(product["product_id"])),
            reverse=True,
        )[:top_n]
        above_cap = [
            product for product in top_products if _product_expected_net_profit_365_at_ref(product) > p95_cap + 1e-6
        ]
        if len(above_cap) <= allowed_above_p95_cap:
            return
        above_cap.sort(
            key=lambda product: (
                _product_expected_net_profit_365_at_ref(product),
                _product_gross_opportunity_profit_365(product),
                str(product["product_id"]),
            ),
            reverse=True,
        )
        changed = False
        for product in above_cap[allowed_above_p95_cap:]:
            changed = _cap_product_expected_net_profit(product, p95_cap) or changed
        if not changed:
            break
    raise ValueError(
        f"expected_net_profit top portfolio cap did not converge: top_n={top_n}, p95_cap={p95_cap}, max_cap={max_cap}"
    )


def _cap_product_expected_net_profit(product: dict, cap: float) -> bool:
    expected_net = _product_expected_net_profit_365_at_ref(product)
    if cap <= 0.0 or expected_net <= cap:
        return False
    scale = float(cap) / expected_net
    curve = [float(value) * scale for value in json.loads(str(product["market_curve"]))]
    product["market_curve"] = json.dumps(curve, ensure_ascii=False)
    return True


def _product_gross_opportunity_profit_365(product: dict) -> float:
    curve_sum = sum(json.loads(str(product["market_curve"])))
    return curve_sum * max(float(product["ref_price"]) - float(product["price"]), 0.0)


def _product_expected_net_profit_365_at_ref(product: dict) -> float:
    row = {
        "_curve_sum": sum(json.loads(str(product["market_curve"]))),
        "_price": float(product["price"]),
        "_ref_price": float(product["ref_price"]),
    }
    rates = {
        "cancel_rate": float(product["cancel_rate"]),
        "refund_rate": float(product["refund_rate"]),
        "only_refund_rate": float(product["only_refund_rate"]),
        "bad_review_rate": float(product["bad_review_rate"]),
    }
    return expected_net_profit_365_at_ref(row, rates)


def _ref_implied_elasticity(cost: float, ref_price: float) -> float:
    denominator = max(float(ref_price) - float(cost), 1e-9)
    raw = float(ref_price) / denominator
    return _clamp(raw, ELASTICITY_CLIP_MIN, ELASTICITY_CLIP_MAX)


def _calibrated_risk_profile(
    rng,
    stratum: str,
    *,
    price: float,
    ref_price: float,
    opportunity_rank: float,
) -> dict[str, float]:
    _ = (ref_price, opportunity_rank)
    out = _risk_profile_for_stratum(rng, stratum)
    out["only_refund_rate"] = min(out["only_refund_rate"], _only_refund_cap_for_price(price))
    return _cap_order_risk_total(out, 0.90)


def _only_refund_cap_for_price(price: float) -> float:
    price = float(price)
    if price <= 10.0:
        return 0.18
    if price <= 50.0:
        return 0.06
    return 0.025


def _cap_order_risk_total(rates: dict[str, float], cap: float) -> dict[str, float]:
    out = dict(rates)
    total = _order_risk(out)
    if total <= cap or total <= 0.0:
        return out
    scale = float(cap) / total
    for field in ("cancel_rate", "refund_rate", "only_refund_rate", "bad_review_rate"):
        out[field] = float(out.get(field, 0.0)) * scale
    return out


def _order_risk(rates: dict[str, float]) -> float:
    return sum(
        float(rates.get(field, 0.0))
        for field in (
            "cancel_rate",
            "refund_rate",
            "only_refund_rate",
            "bad_review_rate",
        )
    )


def _risk_profile_for_stratum(rng, stratum: str) -> dict[str, float]:
    ranges = {
        "good": {
            "cancel_rate": (0.010, 0.025),
            "refund_rate": (0.025, 0.050),
            "only_refund_rate": (0.004, 0.015),
            "bad_review_rate": (0.035, 0.070),
        },
        "safe": {
            "cancel_rate": (0.020, 0.040),
            "refund_rate": (0.045, 0.085),
            "only_refund_rate": (0.006, 0.025),
            "bad_review_rate": (0.065, 0.110),
        },
        "trap": {
            "cancel_rate": (0.040, 0.080),
            "refund_rate": (0.160, 0.240),
            "only_refund_rate": (0.012, 0.040),
            "bad_review_rate": (0.170, 0.260),
        },
        "mediocre": {
            "cancel_rate": (0.015, 0.035),
            "refund_rate": (0.040, 0.080),
            "only_refund_rate": (0.004, 0.020),
            "bad_review_rate": (0.055, 0.105),
        },
        "minefield": {
            "cancel_rate": (0.100, 0.180),
            "refund_rate": (0.260, 0.380),
            "only_refund_rate": (0.040, 0.120),
            "bad_review_rate": (0.270, 0.370),
        },
    }
    selected = ranges.get(stratum, ranges["safe"])
    out = {name: rand_range(rng, *bounds) for name, bounds in selected.items()}
    return out


def _supplier_profiles(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_supplier_id(row)].append(row)
    out = {}
    for supplier_id, sup_rows in grouped.items():
        shop_rating = _avg(1.0 + float(r["_satisfied"]) * 4.0 for r in sup_rows)
        return_buyer = _avg(float(r["_repeat_rate"]) for r in sup_rows)
        age = _avg(float(r["_age_years"]) for r in sup_rows)
        out[supplier_id] = {
            "supplier_name": _canonical_supplier_name(sup_rows),
            "shop_rating": round(_clamp(shop_rating, 1.0, 5.0), 4),
            "return_buyer_rate": round(_clamp(return_buyer, 0.0, 1.0), 6),
            "supplier_age_years": round(max(0.0, age), 4),
        }
    return out


def _canonical_supplier_name(rows: list[dict]) -> str:
    counts = Counter(_clean_name(str(row["company_name"]), limit=80) for row in rows)
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def _market_curve(row: dict) -> list[float]:
    return [round(float(x), 4) for x in row["_curve"]]


def _build_hourly_dist(
    categories: list[str],
    seed: int,
    params: dict,
) -> dict[str, list[tuple[int, float]]]:
    return hourly_dist_rows(build_hourly_dist_for_categories(categories, seed=seed, params=params))


def _metadata(
    bench_csv: str,
    products: list[dict],
    hourly_dist: dict[str, list[tuple[int, float]]],
    target_rows: int,
    *,
    supplier_min: int,
    supplier_max: int,
    supplier_selection_cap: int,
    params: dict,
    build_mode: str = "in_memory",
    category_allocation: str = CATEGORY_ALLOCATION_BALANCED,
) -> dict[str, str]:
    categories = sorted({p["category"] for p in products})
    stratum_counts = Counter(str(p.get("stratum") or "unknown") for p in products)
    good_rate_source_counts = Counter(str(p.get("good_rate_source") or "unknown") for p in products)
    pt_rate_source_counts = Counter(str(p.get("pt_rate_source") or "unknown") for p in products)
    quality_imputation_counts = {
        "good_rate": dict(sorted(good_rate_source_counts.items())),
        "pt_rate": dict(sorted(pt_rate_source_counts.items())),
    }
    row_count = max(1, len(products))
    quality_imputation_rates = {
        "good_rate": round(good_rate_source_counts.get("imputed", 0) / row_count, 6),
        "pt_rate": round(pt_rate_source_counts.get("imputed", 0) / row_count, 6),
    }
    expected_net_summary = _expected_net_profit_summary(products)
    profile = {
        "build_mode": build_mode,
        "category_allocation": category_allocation,
        "sampling_strategy": SAMPLING_STRATEGY,
        "supplier_selection_cap": supplier_selection_cap,
        "supplier_block_scope": "global_final_export",
        "stratum_ratios": STRATUM_RATIOS,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "opportunity_profit_cap_quantile": OPPORTUNITY_PROFIT_CAP_QUANTILE,
        "opportunity_profit_cap_metric": "expected_orders_365_at_ref * max(ref_price - cost, 0)",
        "expected_net_profit_cap_365": EXPECTED_NET_PROFIT_CAP_365,
        "expected_net_profit_top_n": EXPECTED_NET_PROFIT_TOP_N,
        "expected_net_profit_top1000_p95_cap_365": EXPECTED_NET_PROFIT_TOP1000_P95_CAP_365,
        "expected_net_profit_summary": expected_net_summary,
        "risk_calibration_version": RISK_CALIBRATION_VERSION,
        "risk_target_metric": "profit_percentile_density_refund_bad_review_dominant",
        "elasticity_source": "ref_price_implied_clipped",
        "elasticity_clip_min": ELASTICITY_CLIP_MIN,
        "elasticity_clip_max": ELASTICITY_CLIP_MAX,
        "ref_price_cap_ratio": REF_PRICE_CAP_RATIO,
        "quality_imputation_counts": quality_imputation_counts,
        "quality_imputation_rates": quality_imputation_rates,
    }
    payload = json.dumps(
        {"products": products, "hourly_dist": hourly_dist},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "data_source": "private_real",
        "dataset_id": f"private_real_{len(products)}",
        "dataset_rows": str(len(products)),
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "target_rows": str(target_rows),
        "build_mode": build_mode,
        "category_allocation": category_allocation,
        "categories_json": json.dumps(categories, ensure_ascii=False),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bench_csv": os.path.basename(bench_csv),
        "supplier_item_count_min": str(supplier_min),
        "supplier_item_count_max": str(supplier_max),
        "supplier_selection_cap": str(supplier_selection_cap),
        "supplier_block_scope": "global_final_export",
        "category_mapping_json": json.dumps(params.get("category_mapping", {}), ensure_ascii=False, sort_keys=True),
        "sampling_strategy": SAMPLING_STRATEGY,
        "opportunity_profit_cap_quantile": str(OPPORTUNITY_PROFIT_CAP_QUANTILE),
        "opportunity_profit_cap_metric": "expected_orders_365_at_ref * max(ref_price - cost, 0)",
        "expected_net_profit_cap_365": str(EXPECTED_NET_PROFIT_CAP_365),
        "expected_net_profit_top_n": str(EXPECTED_NET_PROFIT_TOP_N),
        "expected_net_profit_top1000_p95_cap_365": str(EXPECTED_NET_PROFIT_TOP1000_P95_CAP_365),
        "expected_net_profit_summary_json": json.dumps(expected_net_summary, ensure_ascii=False, sort_keys=True),
        "risk_calibration_version": RISK_CALIBRATION_VERSION,
        "risk_target_metric": "profit_percentile_density_refund_bad_review_dominant",
        "elasticity_source": "ref_price_implied_clipped",
        "elasticity_clip_min": str(ELASTICITY_CLIP_MIN),
        "elasticity_clip_max": str(ELASTICITY_CLIP_MAX),
        "ref_price_cap_ratio": str(REF_PRICE_CAP_RATIO),
        "stratum_counts_json": json.dumps(dict(sorted(stratum_counts.items())), ensure_ascii=False, sort_keys=True),
        "quality_imputation_counts_json": json.dumps(quality_imputation_counts, ensure_ascii=False, sort_keys=True),
        "quality_imputation_rates_json": json.dumps(quality_imputation_rates, ensure_ascii=False, sort_keys=True),
        "profile_json": json.dumps(profile, ensure_ascii=False, sort_keys=True),
    }


def _expected_net_profit_summary(products: list[dict]) -> dict[str, float]:
    values = [_product_expected_net_profit_365_at_ref(product) for product in products]
    return {
        "min": round(_value_quantile(sorted(values), 0.0), 6),
        "p50": round(_value_quantile(sorted(values), 0.50), 6),
        "p95": round(_value_quantile(sorted(values), 0.95), 6),
        "p99": round(_value_quantile(sorted(values), 0.99), 6),
        "max": round(_value_quantile(sorted(values), 1.0), 6),
    }


def _write_db(
    output_db: str,
    products: list[dict],
    hourly_dist: dict[str, list[tuple[int, float]]],
    meta: dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(output_db) or ".", exist_ok=True)
    if os.path.exists(output_db):
        os.remove(output_db)
    conn = sqlite3.connect(output_db)
    try:
        conn.execute("CREATE TABLE dataset_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE products("
            "product_id TEXT PRIMARY KEY, name TEXT NOT NULL, quantity INTEGER NOT NULL,"
            "price REAL NOT NULL, base_price REAL NOT NULL,"
            "ref_price REAL NOT NULL, raw_ref_price REAL NOT NULL,"
            "supplier_id TEXT NOT NULL,"
            "supplier_name TEXT NOT NULL, ship_hours INTEGER NOT NULL,"
            "logistics_hours INTEGER NOT NULL, category TEXT NOT NULL,"
            "historical_avg_rating REAL NOT NULL, shop_rating REAL NOT NULL,"
            "return_buyer_rate REAL NOT NULL, supplier_age_years REAL NOT NULL,"
            "cancel_rate REAL NOT NULL, refund_rate REAL NOT NULL,"
            "only_refund_rate REAL NOT NULL, bad_review_rate REAL NOT NULL,"
            "max_quantity INTEGER NOT NULL, hourly_increment INTEGER NOT NULL,"
            "timeout_rate REAL NOT NULL, price_change_rate REAL NOT NULL,"
            "supplier_delist_rate REAL NOT NULL, elasticity REAL NOT NULL,"
            "market_curve TEXT NOT NULL, stratum TEXT NOT NULL,"
            "good_rate_source TEXT NOT NULL, pt_rate_source TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE hourly_dist("
            "category TEXT NOT NULL, hour INTEGER NOT NULL,"
            "w REAL NOT NULL, PRIMARY KEY(category, hour))"
        )
        conn.executemany(
            "INSERT INTO dataset_meta(key, value) VALUES (?, ?)",
            sorted(meta.items()),
        )
        cols = ",".join(PRODUCT_COLS)
        placeholders = ",".join("?" for _ in PRODUCT_COLS)
        conn.executemany(
            f"INSERT INTO products({cols}) VALUES ({placeholders})",
            [tuple(p[c] for c in PRODUCT_COLS) for p in products],
        )
        dist_rows = []
        for category, weights in hourly_dist.items():
            dist_rows.extend((category, h, w) for h, w in weights)
        conn.executemany(
            "INSERT INTO hourly_dist(category, hour, w) VALUES (?, ?, ?)",
            dist_rows,
        )
        conn.commit()
    finally:
        conn.close()


def _supplier_id(row: dict) -> str:
    return str(row["member_id"]).strip()


def _product_id(raw: str, used: dict[str, int]) -> str:
    base = _product_id_base(raw)
    used[base] += 1
    if used[base] == 1:
        return base
    suffix = f"_{used[base]}"
    return base[: 32 - len(suffix)] + suffix


def _product_id_base(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")[:32]


def _clean_name(value: str, limit: int = 200) -> str:
    return " ".join(str(value).split())[:limit]


def _required_text(value) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None


def _required_nonnegative(value) -> float | None:
    number = _num(value)
    if number is None or number < 0:
        return None
    return number


def _required_rate(value) -> float | None:
    number = _num(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _parse_daily_curve_sum(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    out = []
    total = 0.0
    for part in text.split("-"):
        number = _num(part)
        if number is None or number < 0:
            return None
        out.append(number)
        total += number
    if len(out) == 362:
        if total <= 0:
            return 0.0
        try:
            return sum(resample_periodic_curve(out, 365))
        except ValueError:
            return None
    if len(out) != 365:
        return None
    return total


def _quality_rate(
    value,
    *,
    raw_row: dict,
    category: str,
    order_cnt: float,
    satisfied: float,
    repeat_rate: float,
) -> tuple[float, str]:
    raw = _optional_rate(value)
    if raw is not None:
        return raw, "raw"

    order_signal = min(math.log1p(max(order_cnt, 0.0)) / math.log1p(10000.0), 1.0)
    jitter = (_stable_unit("good_rate", category, raw_row.get("item_id"), raw_row.get("member_id")) - 0.5) * 0.05
    imputed = 0.78 + satisfied * 0.15 + repeat_rate * 0.04 + order_signal * 0.02 + jitter
    return round(_clamp(imputed, 0.60, 0.995), 6), "imputed"


def _problem_transaction_rate(
    value,
    *,
    raw_row: dict,
    category: str,
    good_rate: float,
    satisfied: float,
    repeat_rate: float,
) -> tuple[float, str]:
    raw = _optional_rate(value)
    if raw is not None:
        return raw, "raw"

    fulfill_rate = _optional_rate(raw_row.get("lgt_fulfill_got_rate_30d"))
    quality_gap = 1.0 - good_rate
    seller_gap = 1.0 - satisfied
    fulfill_gap = 1.0 - (fulfill_rate if fulfill_rate is not None else satisfied)
    repeat_gap = max(0.0, 0.20 - repeat_rate)
    base = 0.008 + quality_gap * 0.30 + seller_gap * 0.08 + fulfill_gap * 0.08 + repeat_gap * 0.03
    tail = _stable_unit("pt_rate_tail", category, raw_row.get("item_id"), raw_row.get("member_id"))
    tail_jitter = _stable_unit("pt_rate_jitter", raw_row.get("item_id"), raw_row.get("title"))
    if tail >= 0.985:
        imputed = 0.30 + tail_jitter * 0.18
    elif tail >= 0.94:
        imputed = max(base, 0.16 + tail_jitter * 0.14)
    elif tail >= 0.82:
        imputed = max(base, 0.05 + tail_jitter * 0.08)
    else:
        imputed = base + (tail_jitter - 0.5) * 0.025
    return round(_clamp(imputed, 0.0, 0.50), 6), "imputed"


def _optional_rate(value) -> float | None:
    number = _num(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _stable_unit(*parts) -> float:
    text = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def _parse_daily_curve(value) -> list[float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    out = []
    for part in text.split("-"):
        number = _num(part)
        if number is None or number < 0:
            return None
        out.append(number)
    if len(out) == 362:
        if sum(out) <= 0:
            out = [0.0] * 365
        else:
            try:
                out = resample_periodic_curve(out, 365)
            except ValueError:
                return None
    if len(out) != 365:
        return None
    return out


def _progress(phase: str, row_idx: int, progress_every: int) -> None:
    if progress_every > 0 and row_idx % progress_every == 0:
        print(json.dumps({"phase": phase, "rows_seen": row_idx}), flush=True)


def _num(value, default=None):
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    return _value_quantile(vals, q)


def _value_quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[max(0, min(len(sorted_values) - 1, idx))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private_real SQLite dataset DB")
    parser.add_argument("--bench-csv", required=True)
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-yaml", default=None)
    parser.add_argument("--exact-streaming", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--category-allocation",
        choices=[
            CATEGORY_ALLOCATION_BALANCED,
            CATEGORY_ALLOCATION_SOURCE,
            CATEGORY_ALLOCATION_CAPACITY,
        ],
        default=CATEGORY_ALLOCATION_BALANCED,
    )
    args = parser.parse_args()
    meta = build_private_real_db(
        bench_csv=args.bench_csv,
        output_db=args.output_db,
        target_rows=args.target_rows,
        seed=args.seed,
        params_path=args.params_yaml,
        exact_streaming=args.exact_streaming,
        progress_every=args.progress_every,
        category_allocation=args.category_allocation,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
