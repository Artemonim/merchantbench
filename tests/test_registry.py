"""Registry sanity: every spec maps to a real handler, parameters schema is
valid JSON Schema-ish (object/properties/required), and the OpenAI format is
emitted correctly."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import registry
import tools.observation  # noqa: F401 — side-effect: registers get_observation


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_TOOLS = {
    "market_brief", "hot_search_terms", "search_products", "get_product_detail",
    "get_daily_report",
    "get_supplier_profile", "list_supplier_products",
    "list_product",
    "delist_product", "adjust_price",
    "query_my_listings", "review_my_listings", "query_balance", "query_platform_rules",
    "query_open_orders", "query_order_updates",
    "query_my_orders", "query_order_detail", "get_store_snapshot",
    "query_supply_chain_anomalies", "query_store_performance",
    "query_product_sales_stats", "query_cash_pipeline",
    "read_memory_doc", "write_memory_doc",
    "end_of_step", "get_observation", "list_tools",
}


def test_registry_contains_all_expected_tools():
    names = {s.name for s in registry.REGISTRY}
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS
    assert not missing, f"missing tools: {missing}"
    assert not extra, f"unexpected tools: {extra}"


def test_human_playground_does_not_expand_agent_tool_surface():
    assert registry.get("query_product_sales_trend") is None
    listings = registry.get("query_my_listings")
    rules = registry.get("query_platform_rules")
    assert listings is not None
    assert listings.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert rules is not None
    assert rules.parameters == listings.parameters


def test_every_spec_has_callable_handler():
    for s in registry.REGISTRY:
        assert callable(s.handler), f"{s.name} handler not callable"


def test_parameters_is_object_with_required_list():
    for s in registry.REGISTRY:
        p = s.parameters
        assert p["type"] == "object", f"{s.name} parameters.type != 'object'"
        assert isinstance(p["properties"], dict)
        assert isinstance(p["required"], list)


def test_required_keys_appear_in_properties():
    for s in registry.REGISTRY:
        props = set(s.parameters["properties"].keys())
        for req in s.parameters["required"]:
            assert req in props, f"{s.name}: required '{req}' missing from properties"


def test_openai_schema_shape():
    for s in registry.REGISTRY:
        sch = s.openai_schema()
        assert sch["type"] == "function"
        assert sch["function"]["name"] == s.name
        assert sch["function"]["description"] == s.description
        assert sch["function"]["parameters"] is s.parameters


def _search_products_schema_description(data):
    env = SimpleNamespace(scenario={"data": data})
    spec = registry.get("search_products")
    assert spec is not None
    return registry.openai_schema_for_env(spec, env)["function"]["description"]


def test_search_products_schema_chinese_note_follows_catalog_meta():
    """Olist/English private_real catalogs must not ask the agent for Chinese keywords."""
    spec = registry.get("search_products")
    assert spec is not None
    chinese_note = registry._CHINESE_PRODUCT_NAME_NOTE.strip()
    assert chinese_note not in spec.description

    assert chinese_note not in _search_products_schema_description(
        {"source": "synthetic"}
    )
    assert chinese_note in _search_products_schema_description(
        {"source": "private_real"}
    )
    assert chinese_note not in _search_products_schema_description({
        "source": "private_real",
        "dataset_id": "olist_v6_33838",
        "source_label": "olist_csv",
    })
    assert chinese_note not in _search_products_schema_description({
        "source": "private_real",
        "private_real_db_path": "data/private_data/olist_v6.sqlite",
    })
    assert chinese_note not in _search_products_schema_description({
        "source": "private_real",
        "product_name_language": "en",
    })
    assert chinese_note in _search_products_schema_description({
        "source": "private_real",
        "dataset_id": "olist_v6_33838",
        "product_name_language": "zh",
    })


def test_mutating_subset_matches_handler_signature():
    """Mutating tools must take agent_id as their second arg (after env)."""
    mutating = {"list_product",
                "delist_product", "adjust_price", "write_memory_doc"}
    for s in registry.REGISTRY:
        if s.name not in mutating:
            continue
        sig = inspect.signature(s.handler)
        params = list(sig.parameters)
        assert params[:2] == ["env", "agent_id"], f"{s.name} signature drift"
        assert s.mutating, f"{s.name} should be marked mutating=True"


def test_list_product_schema_has_no_ship_promise():
    """promised_ship_hours has been removed; list_product only takes product_id and sale_price."""
    list_product = registry.get("list_product")
    assert list_product is not None
    item_props = list_product.parameters["properties"]["items"]["items"]["properties"]
    assert "promised_ship_hours" not in item_props
    assert "promised_logistics_hours" not in item_props
    assert "product_id" in item_props
    assert "sale_price" in item_props
    assert registry.get("set_promised_ship_hours") is None
    assert registry.get("set_promised_logistics_hours") is None


def test_listing_price_schema_keeps_server_side_floor_out_of_public_schema():
    for tool_name, price_field in (
        ("list_product", "sale_price"),
        ("adjust_price", "new_price"),
    ):
        spec = registry.get(tool_name)
        assert spec is not None
        item_props = spec.parameters["properties"]["items"]["items"]["properties"]
        assert "minimum" not in item_props[price_field]


def test_long_design_doc_matches_current_agent_tool_protocol():
    design_doc = REPO_ROOT / "项目详细设计.md"
    if not design_doc.exists():
        pytest.skip("internal long-form design document is not in the artifact")
    text = design_doc.read_text(encoding="utf-8")

    stale_snippets = [
        "set_promised_ship_hours",
        "promised_ship_hours?}]",
        'promised_ship_hours\\": 24',
        "如 `sale_price`、`promised_ship_hours`",
        "各自承诺发货时长",
        "含 `sale_price`、`promised_ship_hours`",
        "`promised_ship_hours`、`supplier_ship_hours`、`supplier_logistics_hours`、累计销量 / 营收",
        "FTS/BM25",
        "每个 mutating 工具的 `tool_call.id` 自动作为 idempotency key",
        "schema（`parameters` / `examples` / `method` / `path_suffix` /",
    ]
    for snippet in stale_snippets:
        assert snippet not in text


def test_query_balance_description_explains_cash_fields():
    spec = registry.get("query_balance")
    assert spec is not None
    desc = spec.description
    assert "balance" in desc
    assert "deposit_pool" in desc
    assert "in_transit" in desc
    assert "receivable" in desc
    assert "cumulative_fine" in desc
    assert "procurement" in desc.lower() or "采购" in desc
    assert "settlement" in desc.lower() or "结算" in desc


def test_aggregate_tool_descriptions_and_cash_pipeline_schema():
    performance = registry.get("query_store_performance")
    product_stats = registry.get("query_product_sales_stats")
    cash_pipeline = registry.get("query_cash_pipeline")

    assert performance is not None
    assert product_stats is not None
    assert cash_pipeline is not None

    assert "net_assets" in performance.description
    assert "current" in product_stats.description
    assert "cash safety" in cash_pipeline.description.lower()
    assert "exact future settlement" in cash_pipeline.description.lower()
    assert "receivable_due" not in cash_pipeline.description

    window = cash_pipeline.parameters["properties"]["window_days"]
    assert window["enum"] == [1, 3, 7, 14]
    assert window["default"] == 7
    assert cash_pipeline.parameters["required"] == []


def test_order_tool_descriptions_have_distinct_positioning():
    open_orders = registry.get("query_open_orders")
    updates = registry.get("query_order_updates")
    history = registry.get("query_my_orders")
    detail = registry.get("query_order_detail")

    assert open_orders is not None
    assert updates is not None
    assert history is not None
    assert detail is not None

    assert "active non-terminal orders" in open_orders.description
    assert "routine fulfillment" in open_orders.description
    assert "status changed since the last observation window" in updates.description
    assert "Coalesces multiple transitions" in updates.description
    assert "Paginated historical order search" in history.description
    assert "retrospective analysis" in history.description
    assert "Full detail for one order" in detail.description
    assert "status_log timeline" in detail.description


def test_order_tools_use_one_pagination_contract_without_legacy_limit():
    for name in ("query_my_orders", "query_open_orders", "query_order_updates"):
        spec = registry.get(name)
        assert spec is not None
        properties = spec.parameters["properties"]
        assert "page" in properties
        assert "page_size" in properties
        assert "limit" not in properties


def test_listing_review_uses_full_day_procured_orders_fields():
    spec = registry.get("review_my_listings")
    assert spec is not None
    sort_by = spec.parameters["properties"]["sort_by"]
    assert sort_by["default"] == "listing_age_days"
    assert sort_by["enum"] == [
        "listing_age_days",
        "days_without_sales",
        "fine",
        "procured_orders",
    ]


def test_get_by_name_and_denylist():
    assert registry.get("adjust_price") is not None
    assert registry.get("nonexistent") is None
    subset = registry.all_specs(["adjust_price", "query_balance"])
    names = {s.name for s in subset}
    assert "adjust_price" not in names
    assert "query_balance" not in names
    assert len(names) == len(registry.REGISTRY) - 2


def test_examples_match_required_params():
    """Every example must contain all required keys (so the openai schema dump
    can be fed directly into a prompt without surprising the agent)."""
    for s in registry.REGISTRY:
        req = set(s.parameters["required"])
        if not s.examples:
            continue
        for ex in s.examples:
            missing = req - set(ex.keys())
            assert not missing, f"{s.name} example missing required keys: {missing}"
