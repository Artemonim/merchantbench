"""Inline Olist-shaped rows for CI-safe v6 catalog tests."""
from __future__ import annotations

from datetime import datetime, timedelta

from data.build_olist_v6 import prepare_olist_v6_from_tables, write_olist_v6_db
from data.generation_profiles import load_default_generation_params


N_VALID_PRODUCTS = 24
SELLER_A = "seller_aaa000000000000000000000000"
SELLER_B = "seller_bbb000000000000000000000000"


def inline_olist_tables(
    *,
    n_valid: int = N_VALID_PRODUCTS,
    include_junk: bool = True,
) -> dict[str, list[dict[str, str]]]:
    """Return Olist-shaped tables with two sellers and dated orders.

    Args:
        n_valid: Number of priced, categorized products (split across sellers).
        include_junk: Append an empty-category row and a zero-price row.

    Returns:
        Tables accepted by ``prepare_olist_v6_from_tables``.
    """
    if n_valid < 2:
        raise ValueError("n_valid must be at least 2")
    n_a = n_valid // 2
    n_b = n_valid - n_a
    products: list[dict[str, str]] = []
    items: list[dict[str, str]] = []
    orders: list[dict[str, str]] = []
    reviews: list[dict[str, str]] = []
    customers: list[dict[str, str]] = []
    origin = datetime(2017, 10, 1, 10, 0, 0)

    def _add_product(index: int, seller_id: str, portuguese: str, price: str) -> None:
        product_id = f"prod{index:04d}{'0' * 24}"
        products.append({
            "product_id": product_id,
            "product_category_name": portuguese,
        })
        n_orders = 2 + (index % 3)
        for order_offset in range(n_orders):
            order_id = f"ord{index:04d}{order_offset:02d}"
            day = origin + timedelta(days=index + order_offset * 3)
            status = "canceled" if order_offset == 0 and index % 7 == 0 else "delivered"
            approved = day + timedelta(hours=1)
            carrier = approved + timedelta(hours=6 + (index % 5))
            delivered = carrier + timedelta(hours=18 + (index % 10))
            customer_id = f"cust{index % 8:02d}{order_offset:02d}"
            unique_id = f"person{index % 6:02d}"
            orders.append({
                "order_id": order_id,
                "customer_id": customer_id,
                "order_status": status,
                "order_purchase_timestamp": day.isoformat(sep=" "),
                "order_approved_at": approved.isoformat(sep=" "),
                "order_delivered_carrier_date": carrier.isoformat(sep=" "),
                "order_delivered_customer_date": delivered.isoformat(sep=" "),
            })
            items.append({
                "order_id": order_id,
                "order_item_id": "1",
                "product_id": product_id,
                "seller_id": seller_id,
                "price": price,
                "freight_value": "12.5",
            })
            score = str(2 + (index + order_offset) % 4)
            reviews.append({
                "review_id": f"rev{index:04d}{order_offset:02d}",
                "order_id": order_id,
                "review_score": score,
            })
            customers.append({
                "customer_id": customer_id,
                "customer_unique_id": unique_id,
            })

    for index in range(n_a):
        _add_product(index, SELLER_A, "utilidades_domesticas", str(40 + index))
    for index in range(n_a, n_valid):
        _add_product(index, SELLER_B, "esporte_lazer", str(55 + index))

    if include_junk:
        products.append({
            "product_id": "prodjunkempty000000000000000000",
            "product_category_name": "",
        })
        items.append({
            "order_id": "ordjunk00",
            "product_id": "prodjunkempty000000000000000000",
            "seller_id": SELLER_A,
            "price": "10.0",
        })
        products.append({
            "product_id": "prodjunkzero0000000000000000000",
            "product_category_name": "utilidades_domesticas",
        })
        items.append({
            "order_id": "ordjunk01",
            "product_id": "prodjunkzero0000000000000000000",
            "seller_id": SELLER_B,
            "price": "0",
        })

    return {
        "products": products,
        "order_items": items,
        "orders": orders,
        "reviews": reviews,
        "sellers": [
            {
                "seller_id": SELLER_A,
                "seller_city": "sao paulo",
                "seller_state": "SP",
            },
            {
                "seller_id": SELLER_B,
                "seller_city": "rio de janeiro",
                "seller_state": "RJ",
            },
        ],
        "translations": [
            {
                "product_category_name": "utilidades_domesticas",
                "product_category_name_english": "housewares",
            },
            {
                "product_category_name": "esporte_lazer",
                "product_category_name_english": "sports_leisure",
            },
        ],
        "customers": customers,
    }


def write_olist_v6_fixture_db(path: str, *, n_valid: int = N_VALID_PRODUCTS) -> str:
    """Write a tiny private_real sqlite from inline Olist rows.

    Args:
        path: Destination sqlite path.
        n_valid: Number of valid products to keep after junk filters.

    Returns:
        The destination path.
    """
    params = load_default_generation_params()
    products, hourly_dist, meta = prepare_olist_v6_from_tables(
        inline_olist_tables(n_valid=n_valid),
        seed=42,
        params=params,
        source_label="test_fixture",
    )
    write_olist_v6_db(path, products, hourly_dist, meta)
    return path
