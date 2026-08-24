"""Tests for deterministic marketplace-style product titles."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest
from core.rng import derive_rng
from data import synth
from data.product_titles import (
    ATTRIBUTES,
    ATTRIBUTES_BY_CATEGORY,
    BRANDS,
    CATEGORY_NOUNS,
    MATERIALS_BY_CATEGORY,
    PROMO_TOKENS,
    SPECS,
    SPECS_BY_CATEGORY,
    generate_title,
)
from tools.hot_search import extract_query_phrases
from web.runner import load_default_scenario


def _title_rng(seed: int, key: int = 0):
    return derive_rng(int(seed), "data_gen", "product_title", int(key))


def _known_tokens():
    tokens = set()
    for pool in (BRANDS, ATTRIBUTES, PROMO_TOKENS, SPECS):
        tokens.update(word.lower() for word in pool)
    for mapping in (
        ATTRIBUTES_BY_CATEGORY,
        MATERIALS_BY_CATEGORY,
        CATEGORY_NOUNS,
        SPECS_BY_CATEGORY,
    ):
        for pool in mapping.values():
            tokens.update(word.lower() for item in pool for word in item.replace("-", " ").split())
    return tokens


def test_same_seed_repeats_title():
    first = generate_title("kitchen", _title_rng(42, 3), typo_rate=0.0)
    second = generate_title("kitchen", _title_rng(42, 3), typo_rate=0.0)
    assert first == second
    assert first


def test_different_seeds_change_titles():
    left = [generate_title("bags", _title_rng(1, idx), typo_rate=0.0) for idx in range(12)]
    right = [generate_title("bags", _title_rng(2, idx), typo_rate=0.0) for idx in range(12)]
    assert left != right


def test_zero_typo_rate_stays_in_lexicon():
    known = _known_tokens()
    for idx in range(40):
        title = generate_title("electronics", _title_rng(42, idx), typo_rate=0.0)
        for token in title.replace("-", " ").split():
            if any(ch.isdigit() for ch in token):
                continue
            assert token.lower() in known, (title, token)


def test_full_typo_rate_changes_some_titles():
    changed = 0
    for idx in range(24):
        clean = generate_title("office", _title_rng(9, idx), typo_rate=0.0)
        noisy = generate_title("office", _title_rng(9, idx), typo_rate=1.0)
        if noisy != clean:
            changed += 1
    assert changed >= 1


def test_titles_stay_within_catalog_name_limit():
    for category in ("home_decor", "appliances", "womenswear"):
        for idx in range(30):
            clean = generate_title(category, _title_rng(42, idx), typo_rate=0.0)
            noisy = generate_title(category, _title_rng(42, idx), typo_rate=1.0)
            assert 0 < len(clean) <= 200
            assert 0 < len(noisy) <= 200


def test_title_contains_category_noun():
    for category, nouns in CATEGORY_NOUNS.items():
        title = generate_title(category, _title_rng(7, 0), typo_rate=0.0)
        lowered = title.lower()
        assert any(noun in lowered for noun in nouns), (category, title)


def test_hot_search_extracts_repeated_phrases_from_mini_catalog():
    titles = [generate_title("electronics", _title_rng(42, idx), typo_rate=0.0) for idx in range(16)]
    support = Counter()
    for title in titles:
        for phrase in extract_query_phrases(title):
            support[phrase] += 1
    assert support
    assert any(count >= 2 for count in support.values())


def test_invalid_typo_rate_raises():
    rng = _title_rng(1, 0)
    with pytest.raises(ValueError, match="typo_rate"):
        generate_title("sports", rng, typo_rate=-0.1)
    with pytest.raises(ValueError, match="typo_rate"):
        generate_title("sports", _title_rng(1, 1), typo_rate=1.5)


def _numeric_catalog_fields(product):
    return {
        "product_id": product.product_id,
        "price": product.price,
        "ref_price": product.ref_price,
        "elasticity": product.elasticity,
        "quantity": product.quantity,
        "ship_hours": product.ship_hours,
        "logistics_hours": product.logistics_hours,
        "max_quantity": product.max_quantity,
        "hourly_increment": product.hourly_increment,
        "market_curve": list(product.market_curve),
        "historical_avg_rating": product.historical_avg_rating,
        "cancel_rate": product.cancel_rate,
        "refund_rate": product.refund_rate,
        "only_refund_rate": product.only_refund_rate,
        "bad_review_rate": product.bad_review_rate,
        "timeout_rate": product.timeout_rate,
        "price_change_rate": product.price_change_rate,
        "supplier_delist_rate": product.supplier_delist_rate,
    }


def test_constant_title_monkeypatch_preserves_numeric_catalog_fields(monkeypatch):
    scenario = deepcopy(load_default_scenario())
    scenario["data"]["num_products"] = 32
    baseline, _ = synth.generate(deepcopy(scenario))
    monkeypatch.setattr(synth, "generate_title", lambda *_args, **_kwargs: "Fixed Marketplace Title")
    patched, _ = synth.generate(deepcopy(scenario))
    assert len(baseline) == len(patched)
    for left, right in zip(baseline, patched):
        assert left.name != right.name
        assert _numeric_catalog_fields(left) == _numeric_catalog_fields(right)
