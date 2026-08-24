from eval import scoring


def test_score_is_final_net_assets():
    merchant = {
        "series": {
            "net_assets": [[0, 3000.0], [1, 2500.55]],
            "cum_net_profit": [[1, -499.45]],
            "shop_rating_mean": [[1, 3.91234]],
        },
        "is_alive": True,
    }

    result = scoring.compute(merchant)

    assert result["score"] == 2500.55
    assert result["final_net_assets"] == 2500.55
    assert result["net_profit"] == -499.45
    assert result["shop_rating_mean"] == 3.9123
    assert result["shop_rating_score"] == 3.9123
    assert result["shop_rating_scale"] == "1-5"
    assert result["n_steps"] == 2


def test_score_uses_cash_fallback_as_net_assets_when_series_missing():
    merchant = {
        "series": {},
        "cash": {
            "balance": 10,
            "deposit_pool": 20,
            "in_transit": 30,
            "receivable": 40,
        },
    }

    result = scoring.compute(merchant)

    assert result["score"] == 100.0
    assert result["final_net_assets"] == 100.0
    assert result["n_steps"] == 0


def test_legacy_shop_rating_is_not_reported_as_canonical_mean():
    result = scoring.compute(
        {
            "series": {
                "net_assets": [[1, 100.0]],
                "shop_rating_score": [[1, 0.91]],
            },
        }
    )

    assert result["shop_rating_mean"] is None
    assert result["shop_rating_score"] == 0.91
    assert result["shop_rating_scale"] == "0-1"


def test_missing_shop_rating_has_no_scale():
    result = scoring.compute({"series": {"net_assets": [[1, 100.0]]}})

    assert result["shop_rating_mean"] is None
    assert result["shop_rating_score"] is None
    assert result["shop_rating_scale"] is None
