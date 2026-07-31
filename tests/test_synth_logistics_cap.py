"""Supplier logistics_hours are not capped by merchant ship promise rules.

The merchant only promises shipping time. Supplier logistics time still controls
delivery timing, but it is not a timeout-penalty threshold and must not be
clamped by max_promised_ship_hours."""
from data.synth import generate
from web.runner import load_default_scenario


def _default_ship_promise(scenario):
    return int(scenario["platform_rules"]["default_promised_ship_hours"])


def test_supplier_logistics_hours_can_exceed_ship_promise_when_scenario_allows():
    scenario = load_default_scenario()
    scenario["supplier_ranges"]["logistics_hours"] = [80, 120]
    scenario["data"]["num_products"] = 100
    products, _ = generate(scenario)
    cap = _default_ship_promise(scenario)
    assert any(p.logistics_hours > cap for p in products)


def test_supplier_ship_hours_can_exceed_ship_promise_when_scenario_allows():
    """The 48h default is a late-detection threshold, not a supplier operational limit."""
    scenario = load_default_scenario()
    scenario["supplier_ranges"]["ship_hours"] = [60, 90]
    scenario["data"]["num_products"] = 100
    products, _ = generate(scenario)
    cap = _default_ship_promise(scenario)
    assert any(p.ship_hours > cap for p in products)
