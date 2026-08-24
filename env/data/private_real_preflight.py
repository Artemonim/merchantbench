"""Preflight checks for private_real SQLite catalog artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Any

from data import private_real
from data.private_real import PrivateRealDataError


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _resolve_preflight_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    return private_real.resolve_dataset_path(path)


def preflight_dataset(path: str, expected_sha: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    meta: dict[str, str] = {}
    products_count = 0
    resolved = _resolve_preflight_path(path)
    if not os.path.isfile(resolved):
        return {"ok": False, "path": resolved, "errors": [f"DB not found: {resolved}"]}

    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(resolved)}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        for table in ("dataset_meta", "products", "hourly_dist"):
            cols = _table_columns(conn, table)
            if not cols:
                errors.append(f"missing table: {table}")
        if errors:
            return {"ok": False, "path": resolved, "errors": errors}

        meta = {str(r["key"]): str(r["value"]) for r in conn.execute("SELECT key, value FROM dataset_meta").fetchall()}
        products_count = int(conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"])
        declared_rows = int(meta.get("dataset_rows", products_count))
        if declared_rows != products_count:
            errors.append(f"dataset_rows={declared_rows} does not match products count={products_count}")
        if expected_sha and meta.get("dataset_sha256") != expected_sha:
            errors.append(f"dataset_sha256 mismatch: expected {expected_sha}, got {meta.get('dataset_sha256', '')}")

        min_len, max_len = conn.execute(
            "SELECT MIN(json_array_length(market_curve)) AS mn,"
            " MAX(json_array_length(market_curve)) AS mx FROM products"
        ).fetchone()
        if min_len != 365 or max_len != 365:
            errors.append(f"market_curve length range must be 365..365, got {min_len}..{max_len}")

        bad_curve = conn.execute(
            "SELECT product_id FROM products"
            " WHERE EXISTS (SELECT 1 FROM json_each(products.market_curve)"
            " WHERE json_each.value < 0 OR json_type(json_each.value) NOT IN ('integer','real'))"
            " LIMIT 1"
        ).fetchone()
        if bad_curve:
            errors.append(f"invalid market_curve values for {bad_curve['product_id']}")

        hourly_rows = conn.execute(
            "SELECT category, COUNT(*) AS n, SUM(w) AS s, MIN(w) AS mn, MAX(w) AS mx FROM hourly_dist GROUP BY category"
        ).fetchall()
        for row in hourly_rows:
            if row["n"] != 24:
                errors.append(f"hourly_dist category {row['category']} has {row['n']} rows")
            if row["mn"] is None or row["mn"] < 0 or abs(float(row["s"]) - 1.0) > 1e-6:
                errors.append(f"hourly_dist category {row['category']} must be non-negative and sum to 1")

        drift = conn.execute(
            "SELECT supplier_id FROM products GROUP BY supplier_id"
            " HAVING COUNT(DISTINCT ROUND(shop_rating, 8) || '|' ||"
            " ROUND(return_buyer_rate, 8) || '|' || ROUND(supplier_age_years, 8)) > 1"
            " LIMIT 1"
        ).fetchone()
        if drift:
            errors.append(f"supplier profile drift for {drift['supplier_id']}")

        try:
            private_real.load_dataset(resolved)
        except PrivateRealDataError as exc:
            errors.append(str(exc))
    except (sqlite3.Error, ValueError, TypeError) as exc:
        errors.append(f"unreadable or invalid DB: {exc}")

    return {
        "ok": not errors,
        "path": resolved,
        "errors": errors,
        "meta": meta,
        "products_count": products_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a private_real SQLite dataset artifact.")
    parser.add_argument("--db", required=True, help="Path to private_real SQLite DB")
    parser.add_argument("--expected-sha", default=None, help="Expected dataset_sha256 metadata value")
    args = parser.parse_args()
    result = preflight_dataset(args.db, args.expected_sha)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
