from core.entities import Product
from core.product_manager import update_products


def _mkproduct(**overrides) -> Product:
    base = dict(
        product_id="P00000",
        name="x",
        quantity=10,
        price=100.0,
        ref_price=100.0,
        supplier_id="s",
        supplier_name="S",
        base_price=100.0,
        ship_hours=1,
        logistics_hours=1,
        category="electronics",
        historical_avg_rating=4.5,
        shop_rating=4.5,
        return_buyer_rate=0.18,
        supplier_age_years=3.5,
        cancel_rate=0.0,
        refund_rate=0.0,
        only_refund_rate=0.0,
        bad_review_rate=0.0,
        max_quantity=100,
        hourly_increment=5,
        timeout_rate=0.0,
        price_change_rate=0.0,
        supplier_delist_rate=0.0,
        elasticity=1.0,
        market_curve=[1.0] * 365,
    )
    base.update(overrides)
    return Product(**base)


SUP_CFG = {
    "recover_steps": [12, 12],
    "price_adjust_factor": [1.2, 1.2],
}


def test_inventory_increments_up_to_max():
    p = _mkproduct(quantity=98, hourly_increment=5, max_quantity=100)
    update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)
    assert p.quantity == 100


def test_price_change_fires_when_probability_is_one():
    p = _mkproduct(price_change_rate=1.0)
    events = update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)
    assert p.price != p.base_price
    assert p.price_recover_t == 12
    event = next(e for e in events if e.event_type == "price_change")
    assert event.payload["ref_price"] == 100.0
    assert event.payload["base_price"] == 100.0


def test_price_recovers_at_recover_t():
    p = _mkproduct(price=120.0, ref_price=160.0, base_price=100.0, price_change_rate=0.0)
    p.price_recover_t = 5
    events = update_products([p], t=5, master_seed=1, sup_cfg=SUP_CFG)
    assert p.price == 100.0
    assert p.price_recover_t is None
    event = next(e for e in events if e.event_type == "price_recover")
    assert event.payload["to"] == 100.0


def test_price_change_uses_base_price_not_ref_price():
    p = _mkproduct(price=80.0, ref_price=160.0, base_price=80.0, price_change_rate=1.0)

    update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)

    assert p.price == 96.0
    assert p.price_recover_t == 12

    update_products([p], t=12, master_seed=1, sup_cfg=SUP_CFG)

    assert p.price == 80.0
    assert p.price_recover_t is None


def test_delist_fires_and_recovers():
    p = _mkproduct(supplier_delist_rate=1.0)
    update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)
    assert p.is_listed_by_supplier is False
    assert p.delist_recover_t == 12
    update_products([p], t=12, master_seed=1, sup_cfg=SUP_CFG)
    assert p.is_listed_by_supplier is True
    assert p.delist_recover_t is None


def test_supplier_timeout_fires_and_recovers():
    p = _mkproduct(timeout_rate=1.0)
    events = update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)
    assert p.timeout_active is True
    assert p.timeout_recover_t == 12
    assert any(e.event_type == "supplier_timeout" for e in events)
    events = update_products([p], t=12, master_seed=1, sup_cfg=SUP_CFG)
    assert p.timeout_active is False
    assert p.timeout_recover_t is None
    assert any(e.event_type == "supplier_timeout_end" for e in events)


def test_supplier_timeout_skipped_when_delisted():
    p = _mkproduct(timeout_rate=1.0, supplier_delist_rate=1.0)
    update_products([p], t=0, master_seed=1, sup_cfg=SUP_CFG)
    # supplier_delist fires first and sets is_listed_by_supplier=False,
    # which blocks the same-step timeout firing.
    assert p.is_listed_by_supplier is False
    assert p.timeout_active is False
