"""Small synthetic CSV fixture for testing the optional SQLite catalog loader."""

from __future__ import annotations

import csv
import os

BENCH_FIELDS = [
    "cate_level1_id",
    "cate_level1_name",
    "cate_id",
    "cate_name",
    "cate_id_tb",
    "item_id",
    "member_id",
    "title",
    "pict_url",
    "reserve_price",
    "ref_price",
    "order_cnt",
    "buyer_cnt",
    "tp_service_y_cnt",
    "e_repeat_rate_6m_001_slr",
    "satisfied_rate_std_001",
    "lgt_fulfill_got_rate_30d",
    "company_name",
    "good_rate",
    "pt_rate",
    "converted_order_cnt_str",
    "ds",
]


def _series(base: float) -> str:
    values = [round(base + (index % 7) * 0.1, 3) for index in range(365)]
    return "-".join(str(value) for value in values)


def _bench_row(category_index: int, row_index: int, stratum: str) -> dict:
    category_names = ["办公、文化", "女装"]
    item_id = f"ITEM-{category_index}-{row_index:03d}"
    member_id = f"member-{row_index % 3}"
    sales_by_stratum = {
        "good": 900,
        "safe": 420,
        "trap": 820,
        "mediocre": 20,
        "minefield": 120,
    }
    quality_by_stratum = {
        "good": (0.99, 0.995, 0.0),
        "safe": (0.97, 0.985, 0.01),
        "trap": (0.99, 0.70, 0.35),
        "mediocre": (0.98, 0.99, 0.0),
        "minefield": (0.65, 0.55, 0.45),
    }
    sales = sales_by_stratum[stratum]
    good_rate, fulfill_rate, pt_rate = quality_by_stratum[stratum]
    return {
        "cate_level1_id": f"L{category_index}",
        "cate_level1_name": category_names[category_index],
        "cate_id": f"C{category_index}-{row_index}",
        "cate_name": f"Leaf {category_index}-{row_index}",
        "cate_id_tb": f"500{category_index}",
        "item_id": item_id,
        "member_id": member_id,
        "title": f"{category_names[category_index]} 商品 {row_index}",
        "pict_url": "",
        "reserve_price": str(10 + row_index),
        "ref_price": str(12 + row_index),
        "order_cnt": str(sales),
        "buyer_cnt": str(max(1, sales // 100)),
        "tp_service_y_cnt": str(1 + row_index % 8),
        "e_repeat_rate_6m_001_slr": "0.25",
        "satisfied_rate_std_001": str(good_rate),
        "lgt_fulfill_got_rate_30d": str(fulfill_rate),
        "company_name": f"Supplier {member_id}",
        "good_rate": str(good_rate),
        "pt_rate": str(pt_rate),
        "converted_order_cnt_str": _series(1.0 + row_index),
        "ds": "20260610",
    }


def write_fixture_csv(directory: str) -> str:
    bench_path = os.path.join(directory, "bench.csv")
    strata = [
        "good",
        "safe",
        "safe",
        "safe",
        "safe",
        "trap",
        "trap",
        "mediocre",
        "minefield",
        "safe",
    ]
    with open(bench_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BENCH_FIELDS)
        writer.writeheader()
        for category_index in range(2):
            for row_index, stratum in enumerate(strata):
                writer.writerow(_bench_row(category_index, row_index, stratum))
    return bench_path
