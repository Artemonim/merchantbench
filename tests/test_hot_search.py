from types import SimpleNamespace


def _product(product_id, name, category, current, previous, *, listed=True):
    curve = [0.0] * 365
    curve[7:14] = [float(previous)] * 7
    curve[14:21] = [float(current)] * 7
    return SimpleNamespace(
        product_id=product_id,
        name=name,
        category=category,
        market_curve=curve,
        is_listed_by_supplier=listed,
    )


def test_extract_query_phrases_keeps_complete_queries_not_standalone_noise():
    from tools.hot_search import extract_query_phrases

    terms = extract_query_phrases("厂家批发USB充电风扇大容量家用")

    assert "usb充电风扇" in terms
    assert "usb" not in terms
    assert "usb充电" not in terms
    assert "大容量" not in terms
    assert "家用" not in terms
    assert all("厂家" not in term and "批发" not in term for term in terms)
    assert "usb充电" not in extract_query_phrases("USB充电")


def test_rank_hot_search_uses_previous_equal_window_for_trend():
    from tools.hot_search import HotSearchIndex

    products = [
        *[
            _product(f"fan-{idx}", "手持电风扇", "appliances", 100, 20)
            for idx in range(4)
        ],
        *[
            _product(f"box-{idx}", "厨房收纳盒", "appliances", 50, 50)
            for idx in range(4)
        ],
    ]

    trends = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20
    )

    fan = next(row for row in trends if "电风扇" in row.keyword)
    box = next(row for row in trends if "收纳盒" in row.keyword)
    assert fan.trend == "surging"
    assert fan.change_pct == 400.0
    assert box.trend == "stable"
    assert box.change_pct == 0.0
    assert fan.rank < box.rank
    assert fan.rank_change == 1


def test_rank_hot_search_ignores_future_curve_and_unlisted_products():
    from tools.hot_search import HotSearchIndex

    listed = [
        _product(f"fan-{idx}", "桌面电风扇", "appliances", 30, 10)
        for idx in range(2)
    ]
    unlisted = _product(
        "fan-unlisted", "桌面电风扇", "appliances", 10000, 10000, listed=False
    )
    products = [*listed, unlisted]
    index = HotSearchIndex(products)

    before = index.rank(category="appliances", window_days=7, today_idx=20)
    for product in products:
        product.market_curve[21:40] = [1_000_000.0] * 19
    after = index.rank(category="appliances", window_days=7, today_idx=20)

    assert after == before
    assert any("电风扇" in row.keyword for row in after)


def test_rank_hot_search_suppresses_near_duplicate_phrases():
    from tools.hot_search import HotSearchIndex

    products = [
        _product(f"fan-{idx}", "手持电风扇", "appliances", 100, 20)
        for idx in range(4)
    ]

    trends = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20
    )
    fan_terms = [row.keyword for row in trends if "电风扇" in row.keyword]

    assert fan_terms == ["手持电风扇"]


def test_rank_hot_search_rejects_tiny_support_even_with_huge_demand():
    from tools.hot_search import HotSearchIndex

    products = [
        *[
            _product(f"trusted-{idx}", "可信电风扇", "appliances", 100, 50)
            for idx in range(5)
        ],
        *[
            _product(f"rare-{idx}", "稀有异常词", "appliances", 100000, 1)
            for idx in range(2)
        ],
        *[
            _product(f"ordinary-{idx}", "普通厨房用品", "appliances", 1, 1)
            for idx in range(93)
        ],
    ]

    trends = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20
    )
    keywords = {row.keyword for row in trends}

    assert any("电风扇" in keyword for keyword in keywords)
    assert "稀有异常词" not in keywords


def test_rank_change_counts_terms_that_dropped_to_zero_current_demand():
    from tools.hot_search import HotSearchIndex

    products = [
        *[
            _product(f"fan-{idx}", "可信电风扇", "appliances", 100, 20)
            for idx in range(4)
        ],
        *[
            _product(f"box-{idx}", "厨房收纳盒", "appliances", 50, 50)
            for idx in range(4)
        ],
        *[
            _product(f"old-{idx}", "过季保暖用品", "appliances", 0, 1000)
            for idx in range(4)
        ],
    ]

    trends = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20
    )
    fan = next(row for row in trends if "电风扇" in row.keyword)

    assert fan.rank == 1
    assert fan.rank_change == 2


def test_trend_labels_low_demand_new_keyword_as_new_not_stable():
    from tools.hot_search import _trend

    # previous=0, current=5 over 7 days: a genuinely new keyword with low
    # demand should be labeled "new", not "stable".
    label, change = _trend(current=5.0, previous=0.0, days=7)
    assert label == "new"
    assert change is None

    # previous=0, current=0.5 over 7 days: too small to register.
    label, change = _trend(current=0.5, previous=0.0, days=7)
    assert label == "stable"
    assert change == 0.0


def test_trend_new_label_preserves_change_pct_when_previous_positive():
    from tools.hot_search import _trend

    # previous=6, current=50 over 7 days: surging keyword with low base.
    # Should be "new" (previous/days < 1.0) but change_pct should be preserved.
    label, change = _trend(current=50.0, previous=6.0, days=7)
    assert label == "new"
    assert change is not None
    assert change > 700.0  # ~733%


def test_rank_applies_small_share_scaling():
    from tools.hot_search import HotSearchIndex

    products = [
        _product(f"fan-{idx}", "手持电风扇", "appliances", 100, 20)
        for idx in range(4)
    ]

    trends_full = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20, small_share=1.0,
    )
    trends_scaled = HotSearchIndex(products).rank(
        category="appliances", window_days=7, today_idx=20, small_share=0.2,
    )

    fan_full = next(row for row in trends_full if "电风扇" in row.keyword)
    fan_scaled = next(row for row in trends_scaled if "电风扇" in row.keyword)
    # Both should produce valid trend labels (not crash or return empty).
    assert fan_full.trend in ("surging", "rising", "stable", "falling", "new")
    assert fan_scaled.trend in ("surging", "rising", "stable", "falling", "new")
    # Scaled demand is lower, so change_pct may differ due to +1.0 smoothing.
    assert fan_full.rank == fan_scaled.rank  # same single keyword → rank 1
