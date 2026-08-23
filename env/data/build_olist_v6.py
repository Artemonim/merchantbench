"""Build a private_real SQLite catalog from the Olist Brazilian E-Commerce CSVs.

Observed listing prices stay in ``ref_price`` (consumer WTP). Cost and
elasticity come from ``cost_and_elasticity_from_margin`` and YAML retail
margins. Market curves are real daily order counts, tiled or resampled
to 365 days — never the synthetic seasonal sine.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable

from core.rng import derive_rng
from data.build_private_real_db import _write_db
from data.generation_profiles import (
    CALIBRATED_BASE_DEMAND_RANGE,
    apply_risk_trust_coupling,
    build_hourly_dist_for_categories,
    cost_and_elasticity_from_margin,
    hourly_dist_rows,
    load_default_generation_params,
    normalize_supplier_ranges,
    resample_periodic_curve,
    sample_operational_fields,
    sample_retail_margin,
    sample_risk_event_fields,
)
from data.product_titles import generate_title, parse_title_typo_rate

# * Empty / missing English category is junk; unknown names use this pool key.
FALLBACK_CATEGORY = "home_goods"
EMPTY_CURVE_FLOOR = float(CALIBRATED_BASE_DEMAND_RANGE[0])
DEFAULT_RATING = 4.0
MIN_EMPIRICAL_ORDERS = 3
MIN_EMPIRICAL_REVIEWS = 3
BUILD_SEED_CHANNEL = "olist_v6_product"
USER_AGENT = "MerchantBench-olist-v6/1.0"
REQUIRED_CSV_NAMES = (
    "olist_products_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
)
OPTIONAL_CSV_NAMES = ("olist_customers_dataset.csv",)
REQUIRED_HEADERS = {
    "olist_products_dataset.csv": ("product_id", "product_category_name"),
    "olist_order_items_dataset.csv": (
        "order_id",
        "product_id",
        "seller_id",
        "price",
    ),
    "olist_orders_dataset.csv": (
        "order_id",
        "order_status",
        "order_purchase_timestamp",
    ),
    "olist_order_reviews_dataset.csv": ("order_id", "review_score"),
    "olist_sellers_dataset.csv": ("seller_id",),
    "product_category_name_translation.csv": (
        "product_category_name",
        "product_category_name_english",
    ),
    "olist_customers_dataset.csv": ("customer_id", "customer_unique_id"),
}
# * Try HuggingFace first, then public GitHub / jsDelivr mirrors. No Kaggle token.
CSV_MIRRORS = (
    "https://huggingface.co/datasets/debs-b/ecommerce-brazil/resolve/main/{name}",
    "https://raw.githubusercontent.com/Kaaykun/OlistAnalysis/master/data/csv/{name}",
    "https://cdn.jsdelivr.net/gh/Kaaykun/OlistAnalysis@master/data/csv/{name}",
    (
        "https://raw.githubusercontent.com/mohamedyounis10/"
        "Olist-brazilian-ecommerce-analytics/main/Datasets/{name}"
    ),
)
UCI_ONLINE_RETAIL_II_ZIP = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_OUTPUT_DB = os.path.join("data", "private_data", "olist_v6.sqlite")
DEFAULT_CSV_DIR = os.path.join("data", "private_data", "olist_csv")
_DEFAULT_SUPPLIER_PROFILE = {
    "shop_rating": [3.5, 5.0],
    "return_buyer_rate": [0.05, 0.30],
    "supplier_age_years": [0.25, 10.0],
}
_DEFAULT_PRODUCT_PROFILE = {
    "historical_avg_rating": [3.5, 5.0],
}

# * Official English names (plus known Olist typos) -> default.yaml category_pool.
OLIST_EN_TO_POOL = {
    "computers_accessories": "office",
    "computers": "office",
    "stationery": "office",
    "office_furniture": "office",
    "books_general_interest": "office",
    "books_imported": "office",
    "books_technical": "office",
    "industry_commerce_and_business": "office",
    "tablets_printing_image": "office",
    "agro_industry_and_commerce": "office",
    "fashion_female_clothing": "womenswear",
    "fashio_female_clothing": "womenswear",
    "fashion_underwear_beach": "womenswear",
    "fashion_shoes": "womenswear",
    "fashion_male_clothing": "womenswear",
    "fashion_childrens_clothes": "womenswear",
    "pet_shop": "pet_garden",
    "garden_tools": "pet_garden",
    "costruction_tools_garden": "pet_garden",
    "flowers": "pet_garden",
    "home_appliances": "appliances",
    "home_appliances_2": "appliances",
    "small_appliances": "appliances",
    "small_appliances_home_oven_and_coffee": "appliances",
    "air_conditioning": "appliances",
    "electronics": "appliances",
    "telephony": "appliances",
    "fixed_telephony": "appliances",
    "audio": "appliances",
    "furniture_decor": "home_decor",
    "bed_bath_table": "home_decor",
    "home_confort": "home_decor",
    "home_comfort_2": "home_decor",
    "christmas_supplies": "home_decor",
    "art": "home_decor",
    "arts_and_craftmanship": "home_decor",
    "furniture_living_room": "home_decor",
    "furniture_bedroom": "home_decor",
    "furniture_mattress_and_upholstery": "home_decor",
    "housewares": "home_goods",
    "food": "home_goods",
    "drinks": "home_goods",
    "food_drink": "home_goods",
    "la_cuisine": "home_goods",
    "kitchen_dining_laundry_garden_furniture": "home_goods",
    "market_place": "home_goods",
    "perfumery": "home_goods",
    "health_beauty": "home_goods",
    "cine_photo": "home_goods",
    "music": "home_goods",
    "cds_dvds_musicals": "home_goods",
    "security_and_services": "home_goods",
    "party_supplies": "home_goods",
    "auto": "home_goods",
    "construction_tools_construction": "home_goods",
    "construction_tools_lights": "home_goods",
    "costruction_tools_tools": "home_goods",
    "home_construction": "home_goods",
    "diapers_and_hygiene": "cleaning",
    "construction_tools_safety": "cleaning",
    "signaling_and_security": "cleaning",
    "toys": "toys",
    "baby": "toys",
    "consoles_games": "toys",
    "cool_stuff": "toys",
    "fashion_bags_accessories": "bags",
    "luggage_accessories": "bags",
    "watches_gifts": "bags",
    "sports_leisure": "sports",
    "fashion_sport": "sports",
    "musical_instruments": "sports",
}


def map_olist_category(english_name: str) -> str:
    """Map an Olist English category onto ``default.yaml`` ``category_pool``.

    Empty names are rejected by the caller. Unknown non-empty names use
    ``FALLBACK_CATEGORY`` (``home_goods``).
    """
    key = _norm_category_key(english_name)
    if not key:
        raise ValueError("english category name is empty")
    return OLIST_EN_TO_POOL.get(key, FALLBACK_CATEGORY)


def prepare_olist_v6_from_tables(
    tables: dict[str, list[dict[str, Any]]],
    *,
    seed: int = 42,
    params: dict[str, Any] | None = None,
    params_path: str | None = None,
    source_label: str = "inline",
    typo_rate: float = 0.0,
) -> tuple[list[dict], dict[str, list[tuple[int, float]]], dict[str, str]]:
    """Map Olist-shaped tables onto the private_real product schema.

    Args:
        tables: Keys ``products``, ``order_items``, ``orders``, ``reviews``,
            ``sellers``, ``translations``; optional ``customers``.
        seed: Build-time RNG seed for operational gaps.
        params: Preloaded generation params. Overrides ``params_path``.
        params_path: Scenario YAML used when ``params`` is omitted.
        source_label: Recorded in dataset metadata.
        typo_rate: Probability of one character-level title typo.

    Returns:
        Product row dicts, hourly_dist rows, and metadata.

    Raises:
        ValueError: If required tables are missing or no SKU survives filters.
    """
    title_typo_rate = parse_title_typo_rate(typo_rate, field="typo_rate")
    profile = _load_build_params(params, params_path)
    products_rows = tables.get("products") or []
    items_rows = tables.get("order_items") or []
    orders_rows = tables.get("orders") or []
    reviews_rows = tables.get("reviews") or []
    sellers_rows = tables.get("sellers") or []
    translations_rows = tables.get("translations") or []
    customers_rows = tables.get("customers") or []
    if not products_rows or not items_rows:
        raise ValueError("Olist tables must include products and order_items")

    translations = _translation_map(translations_rows)
    orders = {_text(row.get("order_id")): row for row in orders_rows if _text(row.get("order_id"))}
    customers = {
        _text(row.get("customer_id")): _text(row.get("customer_unique_id"))
        for row in customers_rows
        if _text(row.get("customer_id")) and _text(row.get("customer_unique_id"))
    }
    seller_city = {
        _text(row.get("seller_id")): _text(row.get("seller_city")) or ""
        for row in sellers_rows
        if _text(row.get("seller_id"))
    }

    items_by_sku: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    items_by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items_rows:
        product_id = _text(row.get("product_id"))
        seller_id = _text(row.get("seller_id"))
        order_id = _text(row.get("order_id"))
        price = _positive_float(row.get("price"))
        if not product_id or not seller_id or not order_id or price is None:
            continue
        payload = dict(row)
        payload["_price"] = price
        items_by_sku[(product_id, seller_id)].append(payload)
        items_by_order[order_id].append(payload)

    product_meta = {}
    for row in products_rows:
        product_id = _text(row.get("product_id"))
        if not product_id:
            continue
        product_meta[product_id] = row

    reviews_by_order: dict[str, list[float]] = defaultdict(list)
    for row in reviews_rows:
        order_id = _text(row.get("order_id"))
        score = _review_score(row.get("review_score"))
        if order_id and score is not None:
            reviews_by_order[order_id].append(score)

    sku_ids = _assign_sku_ids(items_by_sku)
    calendar = _global_calendar(orders, items_by_sku)
    seller_stats = _seller_stats(
        items_by_sku,
        orders,
        reviews_by_order,
        customers,
        seller_city,
        profile,
    )

    products: list[dict] = []
    dropped = {"empty_category": 0, "bad_price": 0, "unknown_product": 0}
    for idx, ((raw_product_id, seller_id), sku_items) in enumerate(
        sorted(items_by_sku.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        source = product_meta.get(raw_product_id)
        if source is None:
            dropped["unknown_product"] += 1
            continue
        portuguese = _text(source.get("product_category_name"))
        if not portuguese:
            dropped["empty_category"] += 1
            continue
        english = translations.get(portuguese) or portuguese
        if not _norm_category_key(english):
            dropped["empty_category"] += 1
            continue
        prices = [float(item["_price"]) for item in sku_items]
        if not prices:
            dropped["bad_price"] += 1
            continue
        ref_price = float(statistics.median(prices))
        if ref_price <= 0.0:
            dropped["bad_price"] += 1
            continue
        category = map_olist_category(english)
        sku_id = sku_ids[(raw_product_id, seller_id)]
        products.append(
            _build_sku_row(
                idx=idx,
                sku_id=sku_id,
                raw_product_id=raw_product_id,
                seller_id=seller_id,
                english_name=english,
                category=category,
                ref_price=ref_price,
                sku_items=sku_items,
                orders=orders,
                reviews_by_order=reviews_by_order,
                seller_profile=seller_stats[seller_id],
                calendar=calendar,
                seed=seed,
                params=profile,
                typo_rate=title_typo_rate,
            )
        )

    if not products:
        raise ValueError(
            "no Olist SKUs survived filters "
            f"(dropped={dropped})"
        )

    categories = sorted({row["category"] for row in products})
    hourly_dist = hourly_dist_rows(
        build_hourly_dist_for_categories(categories, seed=seed, params=profile)
    )
    meta = _metadata(
        products,
        hourly_dist,
        seed=seed,
        source_label=source_label,
        dropped=dropped,
        used_customers=bool(customers),
    )
    return products, hourly_dist, meta


def build_olist_v6(
    *,
    output_db: str,
    csv_dir: str,
    seed: int = 42,
    params_path: str | None = None,
    skip_download: bool = False,
    min_skus: int = 1000,
    typo_rate: float = 0.0,
) -> dict[str, str]:
    """Download (if needed), map, and write the Olist v6 catalog SQLite DB.

    Args:
        output_db: Destination sqlite path.
        csv_dir: Directory for official CSVs (created if missing).
        seed: Build-time RNG seed.
        params_path: Optional scenario YAML for margins and ranges.
        skip_download: Reuse CSVs already in ``csv_dir``.
        min_skus: Fail if fewer SKUs survive filters.
        typo_rate: Probability of one character-level title typo.

    Returns:
        Dataset metadata written into ``dataset_meta``.

    Raises:
        FileNotFoundError: If required CSVs are missing and download is off.
        ValueError: If the mapped catalog is smaller than ``min_skus``.
        RuntimeError: If every public mirror fails (see the message for URLs).
    """
    paths = ensure_olist_csvs(csv_dir, skip_download=skip_download)
    tables = {
        "products": _read_csv(paths["olist_products_dataset.csv"]),
        "order_items": _read_csv(paths["olist_order_items_dataset.csv"]),
        "orders": _read_csv(paths["olist_orders_dataset.csv"]),
        "reviews": _read_csv(paths["olist_order_reviews_dataset.csv"]),
        "sellers": _read_csv(paths["olist_sellers_dataset.csv"]),
        "translations": _read_csv(paths["product_category_name_translation.csv"]),
    }
    customers_path = paths.get("olist_customers_dataset.csv")
    if customers_path:
        tables["customers"] = _read_csv(customers_path)
    products, hourly_dist, meta = prepare_olist_v6_from_tables(
        tables,
        seed=seed,
        params_path=params_path,
        source_label="olist_csv",
        typo_rate=typo_rate,
    )
    if len(products) < int(min_skus):
        raise ValueError(
            f"only {len(products)} SKUs after filters, need >= {min_skus}"
        )
    write_olist_v6_db(output_db, products, hourly_dist, meta)
    return meta


def write_olist_v6_db(
    output_db: str,
    products: list[dict],
    hourly_dist: dict[str, list[tuple[int, float]]],
    meta: dict[str, str],
) -> None:
    """Write product rows using the existing private_real sqlite schema."""
    _write_db(output_db, products, hourly_dist, meta)


def ensure_olist_csvs(
    csv_dir: str,
    *,
    skip_download: bool = False,
) -> dict[str, str]:
    """Return local paths for required (and optional) Olist CSVs.

    Existing files with a valid header are kept. Missing required files
    are fetched from ``CSV_MIRRORS``. Optional customers is best-effort.

    Args:
        csv_dir: Destination directory.
        skip_download: Do not hit the network.

    Returns:
        Map of filename to absolute path.

    Raises:
        FileNotFoundError: Required file missing and download skipped.
        RuntimeError: All mirrors failed; UCI fallback also unusable.
    """
    os.makedirs(csv_dir, exist_ok=True)
    paths: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_CSV_NAMES:
        dest = os.path.join(csv_dir, name)
        if _csv_looks_valid(dest, REQUIRED_HEADERS[name]):
            paths[name] = dest
            continue
        if skip_download:
            missing.append(name)
            continue
        if _download_named_csv(name, dest):
            paths[name] = dest
        else:
            missing.append(name)

    if missing:
        _raise_download_blocker(csv_dir, missing)

    for name in OPTIONAL_CSV_NAMES:
        dest = os.path.join(csv_dir, name)
        if _csv_looks_valid(dest, REQUIRED_HEADERS[name]):
            paths[name] = dest
            continue
        if skip_download:
            continue
        if _download_named_csv(name, dest):
            paths[name] = dest
    return paths


def _download_named_csv(name: str, dest: str) -> bool:
    errors: list[str] = []
    for template in CSV_MIRRORS:
        url = template.format(name=name)
        try:
            _http_download(url, dest)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            errors.append(f"{url} ({exc})")
            if os.path.exists(dest):
                os.remove(dest)
            continue
        if _csv_looks_valid(dest, REQUIRED_HEADERS[name]):
            print(f"downloaded {name} from {url}", file=sys.stderr)
            return True
        errors.append(f"{url} (invalid header or empty body)")
        if os.path.exists(dest):
            os.remove(dest)
    print(
        "failed {0}: {1}".format(name, " | ".join(errors)),
        file=sys.stderr,
    )
    return False


def _raise_download_blocker(csv_dir: str, missing: list[str]) -> None:
    """Explain the Olist failure and the UCI fallback outcome, then stop."""
    tried = [template.format(name=missing[0]) for template in CSV_MIRRORS]
    uci_note = _try_uci_fallback_note(csv_dir)
    raise RuntimeError(
        "Olist CSVs are missing and every public mirror failed. "
        f"missing={missing} csv_dir={csv_dir} "
        f"tried={tried} "
        "No Kaggle token was used; HuggingFace may return 401 without "
        "HF_TOKEN, which is why GitHub mirrors are tried next. "
        f"{uci_note}"
    )


def _try_uci_fallback_note(csv_dir: str) -> str:
    """Attempt the UCI Online Retail II zip and describe why it cannot replace Olist here."""
    dest = os.path.join(csv_dir, "online_retail_ii.zip")
    try:
        _http_download(UCI_ONLINE_RETAIL_II_ZIP, dest)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        return (
            f"UCI Online Retail II fallback also failed: {UCI_ONLINE_RETAIL_II_ZIP} "
            f"({exc}). No secret is configured for that URL."
        )
    size = os.path.getsize(dest) if os.path.isfile(dest) else 0
    return (
        f"UCI Online Retail II zip downloaded to {dest} ({size} bytes) from "
        f"{UCI_ONLINE_RETAIL_II_ZIP}, but this builder maps Olist CSVs only. "
        "The zip is typically XLSX and the env has no openpyxl dependency. "
        "Stopping rather than inventing a fake catalog."
    )


def _http_download(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise OSError("empty HTTP body")
    tmp_path = dest + ".part"
    with open(tmp_path, "wb") as handle:
        handle.write(payload)
    os.replace(tmp_path, dest)


def _csv_looks_valid(path: str, required: Iterable[str]) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 32:
        return False
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, UnicodeError, csv.Error):
        return False
    have = {str(col).strip() for col in header}
    return set(required).issubset(have)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_build_params(
    params: dict[str, Any] | None,
    params_path: str | None,
) -> dict[str, Any]:
    if params is not None:
        merged = copy.deepcopy(params)
    else:
        merged = load_default_generation_params(params_path)
    if "risk_ranges" not in merged or "supplier_ranges" not in merged:
        defaults = load_default_generation_params()
        merged.setdefault("risk_ranges", copy.deepcopy(defaults["risk_ranges"]))
        merged.setdefault(
            "supplier_ranges", copy.deepcopy(defaults["supplier_ranges"])
        )
        merged.setdefault("categories", copy.deepcopy(defaults.get("categories")))
        merged.setdefault(
            "hourly_jitter", copy.deepcopy(defaults.get("hourly_jitter"))
        )
    merged.setdefault(
        "supplier_profile_ranges", copy.deepcopy(_DEFAULT_SUPPLIER_PROFILE)
    )
    merged.setdefault(
        "product_profile_ranges", copy.deepcopy(_DEFAULT_PRODUCT_PROFILE)
    )
    merged["supplier_ranges"] = normalize_supplier_ranges(
        merged["supplier_ranges"]
    )
    return merged


def _translation_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        portuguese = _text(row.get("product_category_name"))
        english = _text(row.get("product_category_name_english"))
        if portuguese and english:
            out[portuguese] = english
    return out


def _assign_sku_ids(
    items_by_sku: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], str]:
    sellers_by_product: dict[str, set[str]] = defaultdict(set)
    for product_id, seller_id in items_by_sku:
        sellers_by_product[product_id].add(seller_id)
    assigned: dict[tuple[str, str], str] = {}
    for product_id, seller_id in items_by_sku:
        if len(sellers_by_product[product_id]) == 1:
            assigned[(product_id, seller_id)] = product_id
        else:
            assigned[(product_id, seller_id)] = f"{product_id}_{seller_id[:8]}"
    return assigned


def _global_calendar(
    orders: dict[str, dict[str, Any]],
    items_by_sku: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[datetime, datetime] | None:
    stamps: list[datetime] = []
    for sku_items in items_by_sku.values():
        for item in sku_items:
            order = orders.get(_text(item.get("order_id")))
            if not order:
                continue
            stamp = _parse_dt(order.get("order_purchase_timestamp"))
            if stamp is not None:
                stamps.append(stamp)
    if not stamps:
        return None
    return min(stamps), max(stamps)


def _seller_stats(
    items_by_sku: dict[tuple[str, str], list[dict[str, Any]]],
    orders: dict[str, dict[str, Any]],
    reviews_by_order: dict[str, list[float]],
    customers: dict[str, str],
    seller_city: dict[str, str],
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scores: dict[str, list[float]] = defaultdict(list)
    stamps: dict[str, list[datetime]] = defaultdict(list)
    buyer_orders: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (product_id, seller_id), sku_items in items_by_sku.items():
        del product_id
        seen_orders: set[str] = set()
        for item in sku_items:
            order_id = _text(item.get("order_id"))
            if not order_id or order_id in seen_orders:
                continue
            seen_orders.add(order_id)
            scores[seller_id].extend(reviews_by_order.get(order_id, []))
            order = orders.get(order_id)
            if not order:
                continue
            stamp = _parse_dt(order.get("order_purchase_timestamp"))
            if stamp is not None:
                stamps[seller_id].append(stamp)
            customer_key = _seller_customer_key(order, customers)
            if customer_key:
                buyer_orders[seller_id][customer_key] += 1

    profile_ranges = params["supplier_profile_ranges"]
    out: dict[str, dict[str, Any]] = {}
    seller_ids = {seller_id for _, seller_id in items_by_sku}
    for seller_id in seller_ids:
        rating = _mean_or_default(scores[seller_id], DEFAULT_RATING)
        rating = _clamp(rating, 1.0, 5.0)
        age = _supplier_age_years(stamps[seller_id])
        if buyer_orders[seller_id]:
            buyers = buyer_orders[seller_id]
            repeats = sum(1 for count in buyers.values() if count >= 2)
            return_buyer = repeats / float(len(buyers))
        else:
            return_buyer = _return_buyer_from_rating(rating, profile_ranges)
        city = seller_city.get(seller_id) or ""
        display = f"Seller {city.title()}" if city else f"Seller {seller_id[:8]}"
        out[seller_id] = {
            "shop_rating": round(rating, 4),
            "return_buyer_rate": round(_clamp(return_buyer, 0.0, 1.0), 4),
            "supplier_age_years": round(max(0.0, age), 4),
            "supplier_name": display[:80],
        }
    return out


def _seller_customer_key(order: dict[str, Any], customers: dict[str, str]) -> str:
    customer_id = _text(order.get("customer_id"))
    if customer_id and customer_id in customers:
        return customers[customer_id]
    return ""


def _build_sku_row(
    *,
    idx: int,
    sku_id: str,
    raw_product_id: str,
    seller_id: str,
    english_name: str,
    category: str,
    ref_price: float,
    sku_items: list[dict[str, Any]],
    orders: dict[str, dict[str, Any]],
    reviews_by_order: dict[str, list[float]],
    seller_profile: dict[str, Any],
    calendar: tuple[datetime, datetime] | None,
    seed: int,
    params: dict[str, Any],
    typo_rate: float = 0.0,
) -> dict[str, Any]:
    rng = derive_rng(int(seed), "data_gen", BUILD_SEED_CHANNEL, idx, sku_id)
    supplier_ranges = params["supplier_ranges"]
    risk_ranges = params["risk_ranges"]
    operational = sample_operational_fields(rng, supplier_ranges)
    risk_event = sample_risk_event_fields(rng, risk_ranges, supplier_ranges)
    margin = sample_retail_margin(rng, category, params)
    cost, elasticity = cost_and_elasticity_from_margin(ref_price, margin)

    order_rows = _sku_orders(sku_items, orders)
    review_scores = _sku_review_scores(order_rows, reviews_by_order)
    hist_rating = _mean_or_default(review_scores, DEFAULT_RATING)
    hist_rating = _clamp(hist_rating, 1.0, 5.0)

    ship_hours, logistics_hours, have_logistics = _delivery_hours(
        order_rows, operational, supplier_ranges
    )
    rates, rates_empirical = _sku_rates(order_rows, review_scores, risk_event)
    if not rates_empirical:
        _apply_rating_bias(
            rates,
            logistics_hours=logistics_hours,
            have_logistics=have_logistics,
            shop_rating=float(seller_profile["shop_rating"]),
            historical_avg_rating=hist_rating,
            params=params,
        )
        if not have_logistics:
            logistics_hours = int(rates.pop("_logistics_hours"))
        else:
            rates.pop("_logistics_hours", None)
    else:
        rates.pop("_logistics_hours", None)

    timestamps = [
        stamp
        for stamp in (
            _parse_dt(order.get("order_purchase_timestamp")) for order in order_rows
        )
        if stamp is not None
    ]
    curve = market_curve_from_timestamps(timestamps, calendar)
    # * Title RNG is a trailing independent stream; it must not precede
    # * operational / risk / margin draws on the product generator.
    title_rng = derive_rng(
        int(seed), "data_gen", BUILD_SEED_CHANNEL, "title", idx, sku_id
    )
    title_category = category or _humanize_category(english_name)
    name = generate_title(title_category, title_rng, typo_rate=typo_rate)
    return {
        "product_id": sku_id,
        "name": name[:200],
        "quantity": int(operational["quantity"]),
        "price": round(float(cost), 4),
        "base_price": round(float(cost), 4),
        "ref_price": round(float(ref_price), 4),
        "raw_ref_price": round(float(ref_price), 4),
        "supplier_id": seller_id,
        "supplier_name": str(seller_profile["supplier_name"]),
        "ship_hours": int(ship_hours),
        "logistics_hours": int(logistics_hours),
        "category": category,
        "historical_avg_rating": round(hist_rating, 4),
        "shop_rating": float(seller_profile["shop_rating"]),
        "return_buyer_rate": float(seller_profile["return_buyer_rate"]),
        "supplier_age_years": float(seller_profile["supplier_age_years"]),
        "cancel_rate": round(float(rates["cancel_rate"]), 6),
        "refund_rate": round(float(rates["refund_rate"]), 6),
        "only_refund_rate": round(float(rates["only_refund_rate"]), 6),
        "bad_review_rate": round(float(rates["bad_review_rate"]), 6),
        "max_quantity": int(operational["max_quantity"]),
        "hourly_increment": int(operational["hourly_increment"]),
        "timeout_rate": round(float(risk_event["timeout_rate"]), 6),
        "price_change_rate": round(float(risk_event["price_change_rate"]), 6),
        "supplier_delist_rate": round(float(risk_event["supplier_delist_rate"]), 6),
        "elasticity": round(float(elasticity), 4),
        "market_curve": json.dumps(curve, ensure_ascii=False),
        "stratum": "olist_v6",
        "good_rate_source": "olist_review" if review_scores else "imputed",
        "pt_rate_source": "olist_status" if rates_empirical else "imputed",
    }


def _sku_orders(
    sku_items: list[dict[str, Any]],
    orders: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in sku_items:
        order_id = _text(item.get("order_id"))
        if not order_id or order_id in seen:
            continue
        seen.add(order_id)
        order = orders.get(order_id)
        if order is not None:
            rows.append(order)
    return rows


def _sku_review_scores(
    order_rows: list[dict[str, Any]],
    reviews_by_order: dict[str, list[float]],
) -> list[float]:
    scores: list[float] = []
    for order in order_rows:
        scores.extend(reviews_by_order.get(_text(order.get("order_id")), []))
    return scores


def _delivery_hours(
    order_rows: list[dict[str, Any]],
    operational: dict[str, int],
    supplier_ranges: dict[str, Any],
) -> tuple[int, int, bool]:
    ship_lo, ship_hi = _closed_hours(supplier_ranges["ship_hours"], exclusive_hi=False)
    log_lo, log_hi = _closed_hours(
        supplier_ranges["logistics_hours"], exclusive_hi=True
    )
    log_hi = min(log_hi, 72)
    ship_samples: list[float] = []
    log_samples: list[float] = []
    for order in order_rows:
        approved = _parse_dt(order.get("order_approved_at"))
        carrier = _parse_dt(order.get("order_delivered_carrier_date"))
        delivered = _parse_dt(order.get("order_delivered_customer_date"))
        ship = _hours_between(approved, carrier)
        logistics = _hours_between(carrier, delivered)
        if ship is None and logistics is None and approved and delivered:
            total = _hours_between(approved, delivered)
            if total is not None:
                ship = total * 0.25
                logistics = total - ship
        if ship is not None:
            ship_samples.append(ship)
        if logistics is not None:
            log_samples.append(logistics)
    ship_hours = int(operational["ship_hours"])
    logistics_hours = int(operational["logistics_hours"])
    have_logistics = False
    if ship_samples:
        ship_hours = _clamp_int(statistics.median(ship_samples), ship_lo, ship_hi)
    if log_samples:
        logistics_hours = _clamp_int(statistics.median(log_samples), log_lo, log_hi)
        have_logistics = True
    ship_hours = max(1, int(ship_hours))
    logistics_hours = int(_clamp(logistics_hours, 1, 72))
    return ship_hours, logistics_hours, have_logistics


def _sku_rates(
    order_rows: list[dict[str, Any]],
    review_scores: list[float],
    risk_event: dict[str, float],
) -> tuple[dict[str, float], bool]:
    n_orders = len(order_rows)
    n_reviews = len(review_scores)
    cancel = sum(
        1 for order in order_rows if _text(order.get("order_status")) == "canceled"
    )
    unavailable = sum(
        1 for order in order_rows if _text(order.get("order_status")) == "unavailable"
    )
    bad = sum(1 for score in review_scores if score <= 2.0)
    ones = sum(1 for score in review_scores if score <= 1.0)
    empirical = n_orders >= MIN_EMPIRICAL_ORDERS or n_reviews >= MIN_EMPIRICAL_REVIEWS
    if empirical:
        cancel_rate = (cancel / float(n_orders)) if n_orders else 0.0
        refund_rate = (unavailable / float(n_orders)) if n_orders else 0.0
        if n_reviews:
            # * Low-star reviews are the public return-like signal when refunds are absent.
            refund_rate = max(refund_rate, bad / float(n_reviews) * 0.5)
            bad_review_rate = bad / float(n_reviews)
            only_refund_rate = ones / float(n_reviews) * 0.25
        else:
            bad_review_rate = 0.0
            only_refund_rate = (unavailable / float(n_orders)) * 0.25 if n_orders else 0.0
        rates = {
            "cancel_rate": _clamp(cancel_rate, 0.0, 1.0),
            "refund_rate": _clamp(refund_rate, 0.0, 1.0),
            "only_refund_rate": _clamp(only_refund_rate, 0.0, 1.0),
            "bad_review_rate": _clamp(bad_review_rate, 0.0, 1.0),
        }
        return rates, True
    return {
        "cancel_rate": float(risk_event["cancel_rate"]),
        "refund_rate": float(risk_event["refund_rate"]),
        "only_refund_rate": float(risk_event["only_refund_rate"]),
        "bad_review_rate": float(risk_event["bad_review_rate"]),
    }, False


def _apply_rating_bias(
    rates: dict[str, float],
    *,
    logistics_hours: int,
    have_logistics: bool,
    shop_rating: float,
    historical_avg_rating: float,
    params: dict[str, Any],
) -> None:
    """Bias imputed rates the same way as ``apply_risk_trust_coupling``."""
    ns = SimpleNamespace(
        shop_rating=shop_rating,
        historical_avg_rating=historical_avg_rating,
        refund_rate=float(rates["refund_rate"]),
        only_refund_rate=float(rates["only_refund_rate"]),
        bad_review_rate=float(rates["bad_review_rate"]),
        logistics_hours=int(logistics_hours),
    )
    apply_risk_trust_coupling(
        ns,
        risk_ranges=params["risk_ranges"],
        supplier_ranges=params["supplier_ranges"],
        supplier_profile_ranges=params["supplier_profile_ranges"],
        product_profile_ranges=params["product_profile_ranges"],
    )
    rates["refund_rate"] = float(ns.refund_rate)
    rates["only_refund_rate"] = float(ns.only_refund_rate)
    rates["bad_review_rate"] = float(ns.bad_review_rate)
    if not have_logistics:
        rates["_logistics_hours"] = int(ns.logistics_hours)


def market_curve_from_timestamps(
    timestamps: list[datetime],
    calendar: tuple[datetime, datetime] | None,
) -> list[float]:
    """Build a 365-day non-negative curve from purchase timestamps.

    Daily counts use the shared catalog calendar when available. Shorter
    histories are tiled; longer ones go through ``resample_periodic_curve``.
    Empty history becomes a constant ``EMPTY_CURVE_FLOOR`` — not a sine.
    Amplitude is then scaled into ``CALIBRATED_BASE_DEMAND_RANGE``.
    """
    if not timestamps:
        return [EMPTY_CURVE_FLOOR] * 365
    if calendar is None:
        start, end = min(timestamps), max(timestamps)
    else:
        start, end = calendar
    counts = _daily_counts(timestamps, start, end)
    return _fit_and_scale_curve(counts)


def _daily_counts(
    timestamps: list[datetime],
    start: datetime,
    end: datetime,
) -> list[float]:
    n_days = (end.date() - start.date()).days + 1
    n_days = max(1, n_days)
    counts = [0.0] * n_days
    origin = start.date()
    for stamp in timestamps:
        index = (stamp.date() - origin).days
        if 0 <= index < n_days:
            counts[index] += 1.0
    return counts


def _fit_and_scale_curve(values: list[float], target_len: int = 365) -> list[float]:
    cleaned = [max(0.0, float(value)) for value in values]
    if not cleaned or sum(cleaned) <= 0.0:
        return [EMPTY_CURVE_FLOOR] * target_len
    if len(cleaned) == target_len:
        fitted = cleaned
    elif len(cleaned) == 1:
        fitted = cleaned * target_len
    elif len(cleaned) < target_len:
        # * Tile the observed daily series; do not replace it with a seasonal sine.
        fitted = _tile_series(cleaned, target_len)
    else:
        fitted = _resample_or_tile(cleaned, target_len)
    return _scale_curve(fitted)


def _tile_series(values: list[float], target_len: int) -> list[float]:
    reps = (target_len + len(values) - 1) // len(values)
    return (list(values) * reps)[:target_len]


def _occupied_window(values: list[float]) -> list[float]:
    nonzero = [index for index, value in enumerate(values) if value > 0.0]
    if not nonzero:
        return []
    return list(values[nonzero[0] : nonzero[-1] + 1])


def _resample_or_tile(cleaned: list[float], target_len: int) -> list[float]:
    """Resample a long real series; tile the occupied window if interp goes flat.

    Isolated spikes on a multi-year calendar can resample to a non-positive
    total. Tiling the first-to-last positive day keeps the observed shape
    instead of substituting the synthetic sine.
    """
    try:
        return resample_periodic_curve(cleaned, target_len)
    except ValueError:
        occupied = _occupied_window(cleaned)
        if not occupied:
            return [EMPTY_CURVE_FLOOR] * target_len
        if len(occupied) == 1:
            return occupied * target_len
        if len(occupied) < target_len:
            return _tile_series(occupied, target_len)
        step = len(occupied) / float(target_len)
        return [occupied[min(len(occupied) - 1, int(index * step))] for index in range(target_len)]


def _scale_curve(values: list[float]) -> list[float]:
    mean = sum(values) / float(len(values))
    if mean <= 0.0:
        return [EMPTY_CURVE_FLOOR] * len(values)
    lo, hi = CALIBRATED_BASE_DEMAND_RANGE
    target = min(hi, max(lo, mean))
    factor = target / mean
    return [round(value * factor, 4) for value in values]


def _metadata(
    products: list[dict],
    hourly_dist: dict[str, list[tuple[int, float]]],
    *,
    seed: int,
    source_label: str,
    dropped: dict[str, int],
    used_customers: bool,
) -> dict[str, str]:
    categories = sorted({row["category"] for row in products})
    payload = json.dumps(
        {"products": products, "hourly_dist": hourly_dist},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "data_source": "private_real",
        "dataset_id": f"olist_v6_{len(products)}",
        "dataset_rows": str(len(products)),
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "source_label": source_label,
        "build_seed": str(seed),
        "categories_json": json.dumps(categories, ensure_ascii=False),
        "fallback_category": FALLBACK_CATEGORY,
        "empty_curve_policy": (
            f"constant {EMPTY_CURVE_FLOOR} floor; no synthetic seasonal sine"
        ),
        "ref_price_policy": "observed_listing_price",
        "elasticity_source": "cost_and_elasticity_from_margin",
        "dropped_json": json.dumps(dropped, sort_keys=True),
        "used_customers": "1" if used_customers else "0",
        "license": "CC BY-NC-SA 4.0",
        "attribution": "Olist Brazilian E-Commerce Public Dataset",
    }


def _return_buyer_from_rating(shop_rating: float, ranges: dict[str, Any]) -> float:
    lo = float(ranges["return_buyer_rate"][0])
    hi = float(ranges["return_buyer_rate"][1])
    position = _clamp((float(shop_rating) - 1.0) / 4.0, 0.0, 1.0)
    return lo + position * (hi - lo)


def _supplier_age_years(stamps: list[datetime]) -> float:
    if len(stamps) < 2:
        return 0.0
    delta = max(stamps) - min(stamps)
    return max(0.0, delta.total_seconds() / (365.25 * 24.0 * 3600.0))


def _closed_hours(raw: Any, *, exclusive_hi: bool) -> tuple[int, int]:
    lo = int(raw[0])
    hi = int(raw[1])
    if exclusive_hi:
        hi -= 1
    if hi < lo:
        raise ValueError(f"invalid hour range {raw!r}")
    return lo, hi


def _hours_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    hours = (end - start).total_seconds() / 3600.0
    if hours < 0.0:
        return None
    return hours


def _parse_dt(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _review_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not (1.0 <= score <= 5.0):
        return None
    return score


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0.0:
        return None
    return number


def _mean_or_default(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    return float(sum(values) / float(len(values)))


def _humanize_category(english_name: str) -> str:
    return _norm_category_key(english_name).replace("_", " ").title() or "Product"


def _norm_category_key(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", "_").split())


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, round(float(value)))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Olist CSVs and build the v6 private_real catalog DB"
    )
    parser.add_argument(
        "--output-db",
        default=DEFAULT_OUTPUT_DB,
        help="SQLite destination (default: data/private_data/olist_v6.sqlite)",
    )
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_CSV_DIR,
        help="CSV cache directory (default: data/private_data/olist_csv)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-yaml", default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse CSVs already in --csv-dir",
    )
    parser.add_argument("--min-skus", type=int, default=1000)
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Allow catalogs smaller than --min-skus (for fixtures)",
    )
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=0.0,
        help="Probability of one character-level typo in each product title (0-1)",
    )
    args = parser.parse_args()
    min_skus = 1 if args.allow_small else int(args.min_skus)
    typo_rate = parse_title_typo_rate(args.typo_rate, field="--typo-rate")
    meta = build_olist_v6(
        output_db=args.output_db,
        csv_dir=args.csv_dir,
        seed=args.seed,
        params_path=args.params_yaml,
        skip_download=args.skip_download,
        min_skus=min_skus,
        typo_rate=typo_rate,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
