import sys

import requests

import agent.baselines.auto_seed as auto_seed_module
from agent.baselines.auto_seed import AutoSeedAgent


def _table(columns, rows):
    return {"columns": columns, "rows": rows, "count": len(rows)}


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response.url = "http://merchantbench.test/act"
    return requests.HTTPError(f"{status_code} error", response=response)


class _FakeAutoSeedClient:
    def __init__(self):
        self.observation_calls = 0
        self.act_calls = 0
        self.register_kwargs = None

    def register(self, **kwargs):
        self.register_kwargs = kwargs
        return {"ok": True}

    def observation(self):
        self.observation_calls += 1
        if self.observation_calls == 1:
            return {
                "brief": {"system_prompt": "rules", "language": "en"},
                "tick": {"day": 1, "hour": 0},
                "text": "first hook",
            }
        raise _http_error(410)

    def act(self, assistant_msg, token_usage=None):
        self.act_calls += 1
        raise _http_error(425)


class _ScriptedAutoSeed(AutoSeedAgent):
    def __init__(self, responses):
        self.base = "http://merchantbench.test"
        self.run_id = "run_1"
        self.agent_id = "agent_0"
        self.seed_count = 3
        self.markup = 2.0
        self.cash_low_watermark = 500.0
        self.system_prompt = None
        self.language = "en"
        self._seeded = True
        self._last_refresh_day = None
        self._listed_product_ids = set()
        self.calls = []
        self._responses = list(responses)

    def _act(self, thought, tool_calls_spec):
        self.calls.extend(tool_calls_spec)
        if not self._responses:
            return [{} for _ in tool_calls_spec]
        return self._responses.pop(0)


def test_auto_seed_declares_complete_runtime_health_coverage():
    agent = AutoSeedAgent.__new__(AutoSeedAgent)
    agent.client = _FakeAutoSeedClient()
    agent.seed_count = 3
    agent.markup = 2.0
    agent.cash_low_watermark = 500.0

    agent.register()

    extra = agent.client.register_kwargs["extra"]
    assert extra["runtime_health_version"] == 1
    assert set(extra["runtime_health_capabilities"].values()) == {"not_applicable"}


def test_auto_seed_reobserves_after_stale_act_error():
    agent = AutoSeedAgent.__new__(AutoSeedAgent)
    agent.client = _FakeAutoSeedClient()
    agent.system_prompt = None
    agent.language = "en"
    agent._seeded = False
    agent.seed_count = 50
    agent.markup = 2.0
    agent.cash_low_watermark = 500.0
    agent._last_refresh_day = None
    agent._listed_product_ids = set()

    agent.run(max_steps=10, verbose=False)

    assert agent.client.observation_calls == 2
    assert agent.client.act_calls == 1


def test_auto_seed_delists_everything_when_cash_is_low():
    agent = _ScriptedAutoSeed(
        [
            [{"balance": 499.99}],
            [_table(["product_id"], [["p1"], ["p2"]])],
            [{"ok": True}],
            [{"ok": True}],
        ]
    )

    agent._drive_step({"tick": {"day": 3, "hour": 0}}, verbose=False)

    assert ("delist_product", {"items": [{"product_id": "p1"}, {"product_id": "p2"}]}) in agent.calls
    assert not any(name == "get_daily_report" for name, _ in agent.calls)
    assert agent.calls[-1] == ("end_of_step", {})


def test_auto_seed_handles_current_supply_risks_and_refills_random_shelf(monkeypatch):
    monkeypatch.setattr(auto_seed_module, "RANDOM_SEARCH_PAGES", 1)
    monkeypatch.setattr(auto_seed_module, "RANDOM_PAGE_UPPER_BOUND", 1)
    monkeypatch.setattr(auto_seed_module, "RANDOM_FALLBACK_PAGES", 0)
    listings = _table(
        ["product_id", "sale_price", "supplier_price"],
        [
            ["price-up", 10.0, 7.0],
            ["price-down", 20.0, 4.0],
            ["supplier-delisted", 10.0, 5.0],
            ["timeout-active", 10.0, 5.0],
            ["stable", 12.0, 6.0],
        ],
    )
    anomaly_listings = _table(
        ["product_id", "supplier_listed", "supplier_ship_hours", "supplier_price"],
        [
            ["supplier-delisted", False, 12, 5.0],
            ["timeout-active", True, 72, 5.0],
            ["price-up", True, 12, 7.0],
        ],
    )
    replacements = _table(
        ["product_id", "price"],
        [
            ["timeout-active", 5.0],
            ["replacement-1", 3.0],
            ["replacement-2", 4.0],
        ],
    )
    agent = _ScriptedAutoSeed(
        [
            [listings],
            [{"events": [], "listings": anomaly_listings}],
            [{"ok": True}],
            [{"ok": True}],
            [{"items": replacements}],
            [{"ok": True}],
            [{"ok": True}],
        ]
    )
    agent.selection_mode = "random"
    agent.selection_seed = 42
    agent.seed_count = 5
    agent._last_refresh_day = 3

    agent._drive_step({"tick": {"day": 3, "hour": 12}}, verbose=False)

    assert [name for name, _args in agent.calls] == [
        "query_my_listings",
        "query_supply_chain_anomalies",
        "delist_product",
        "adjust_price",
        "search_products",
        "list_product",
        "end_of_step",
    ]
    assert [args for name, args in agent.calls if name == "query_supply_chain_anomalies"] == [{"mode": "now"}]
    assert (
        "delist_product",
        {
            "items": [
                {"product_id": "supplier-delisted"},
                {"product_id": "timeout-active"},
            ]
        },
    ) in agent.calls
    assert (
        "adjust_price",
        {
            "items": [
                {"product_id": "price-up", "new_price": 14.0},
                {"product_id": "price-down", "new_price": 8.0},
            ]
        },
    ) in agent.calls
    assert (
        "list_product",
        {
            "items": [
                {"product_id": "replacement-1", "sale_price": 6.0},
                {"product_id": "replacement-2", "sale_price": 8.0},
            ]
        },
    ) in agent.calls
    assert agent.calls[-1] == ("end_of_step", {})


def test_auto_seed_uses_consecutive_days_without_sales_for_stale_listings():
    review = _table(
        [
            "product_id",
            "listing_age_days",
            "days_without_sales",
            "procured_orders",
        ],
        [
            ["exact-boundary", 20, 7, 1],
            ["recent-sale", 20, 6, 0],
            ["never-sold", 9, 9, 0],
        ],
    )
    agent = _ScriptedAutoSeed([[review]])

    stale_ids = agent._stale_listing_ids()

    assert stale_ids == ["exact-boundary", "never-sold"]
    assert agent.calls == [
        (
            "review_my_listings",
            {
                "sort_by": "days_without_sales",
                "window_days": 7,
            },
        ),
    ]


def test_auto_seed_uses_daily_report_keywords_to_search_and_list_products():
    report = {
        "ok": True,
        "content": (
            "▍异动信号词\n"
            "| 排名 | 关键词 | 搜索量 |\n"
            "| #25(NEW) | 教师节礼物 | 6.4万 |\n"
            '建议补充**午睡枕**与"活页笔记本"。'
        ),
    }
    products = _table(
        [
            "product_id",
            "name",
            "quantity",
            "price",
            "supplier_id",
            "supplier_name",
            "supplier_ship_hours",
            "logistics_hours",
            "category",
            "historical_avg_rating",
            "shop_rating",
            "supplier_age_years",
        ],
        [["p1", "教师节礼物套装", 100, 5.0, "s1", "sup", 12, 24, "office", 4.8, 4.9, 3.0]],
    )
    agent = _ScriptedAutoSeed(
        [
            [{"balance": 2000.0}],
            [_table(["product_id"], [])],
            [{"events": [], "listings": _table(["product_id"], [])}],
            [report],
            [{"items": products}],
            [{"ok": True}],
            [{"ok": True}],
        ]
    )
    agent._seeded = False
    agent.seed_count = 1

    agent._drive_step({"tick": {"day": 1, "hour": 0}}, verbose=False)

    search_calls = [args for name, args in agent.calls if name == "search_products"]
    assert search_calls
    assert search_calls[0]["query"] == "教师节礼物"
    assert ("list_product", {"items": [{"product_id": "p1", "sale_price": 10.0}]}) in agent.calls
    assert agent.calls[-1] == ("end_of_step", {})


def test_auto_seed_prices_products_at_exactly_2x_cost():
    agent = _ScriptedAutoSeed([])

    assert agent._sale_price(0.2) == 0.4
    assert agent._sale_price(5.0) == 10.0


def test_rule_based_random_mode_skips_report_and_all_business_filters(monkeypatch):
    monkeypatch.setattr(auto_seed_module, "RANDOM_SEARCH_PAGES", 1)
    monkeypatch.setattr(auto_seed_module, "RANDOM_PAGE_UPPER_BOUND", 1)
    monkeypatch.setattr(auto_seed_module, "RANDOM_FALLBACK_PAGES", 0)
    products = _table(
        [
            "product_id",
            "quantity",
            "price",
            "supplier_ship_hours",
            "historical_avg_rating",
            "shop_rating",
        ],
        [
            ["low-rating", 100, 5.0, 24, 1.5, 1.0],
            ["too-expensive", 100, 25.0, 24, 5.0, 5.0],
            ["slow-shipping", 100, 6.0, 240, 5.0, 5.0],
            ["low-stock", 1, 7.0, 24, 5.0, 5.0],
        ],
    )
    agent = _ScriptedAutoSeed(
        [
            [{"items": products}],
            [{"ok": True}],
        ]
    )
    agent.selection_mode = "random"
    agent.selection_seed = 42
    agent._listed_product_ids = {
        "low-rating",
        "too-expensive",
        "slow-shipping",
        "low-stock",
    }

    agent._seed_listings(target_count=4)

    search_args = next(args for name, args in agent.calls if name == "search_products")
    assert search_args == {
        "query": "",
        "page": 1,
        "page_size": 50,
        "sort_by": "relevance",
    }
    assert not any(name == "get_daily_report" for name, _ in agent.calls)
    list_args = next(args for name, args in agent.calls if name == "list_product")
    assert {item["product_id"]: item["sale_price"] for item in list_args["items"]} == {
        "low-rating": 10.0,
        "too-expensive": 50.0,
        "slow-shipping": 12.0,
        "low-stock": 14.0,
    }


def test_rule_based_random_mode_skips_stale_listing_review():
    listings = _table(["product_id"], [["p1"], ["p2"], ["p3"]])
    no_anomalies = {
        "events": [],
        "listings": _table(["product_id"], []),
    }
    agent = _ScriptedAutoSeed(
        [
            [listings],
            [no_anomalies],
            [{"ok": True}],
        ]
    )
    agent.selection_mode = "random"
    agent._last_refresh_day = 2

    agent._drive_step({"tick": {"day": 3, "hour": 0}}, verbose=False)

    assert not any(name == "query_balance" for name, _ in agent.calls)
    assert not any(name == "review_my_listings" for name, _ in agent.calls)
    assert not any(name == "delist_product" for name, _ in agent.calls)
    assert agent.calls[-1] == ("end_of_step", {})


def test_rule_based_random_mode_retries_same_day_reproducibly(monkeypatch):
    monkeypatch.setattr(auto_seed_module, "RANDOM_SEARCH_PAGES", 2)
    monkeypatch.setattr(auto_seed_module, "RANDOM_PAGE_UPPER_BOUND", 10)
    monkeypatch.setattr(auto_seed_module, "RANDOM_FALLBACK_PAGES", 0)
    empty = {"items": _table(["product_id"], [])}
    agent = _ScriptedAutoSeed(
        [
            [empty],
            [empty],
            [empty],
            [empty],
        ]
    )
    agent.selection_mode = "random"
    agent.selection_seed = 42

    agent._random_product_picks(set(), 1, selection_day=7)
    first_pages = [args["page"] for name, args in agent.calls if name == "search_products"]
    agent.calls.clear()
    agent._random_product_picks(set(), 1, selection_day=7)
    retry_pages = [args["page"] for name, args in agent.calls if name == "search_products"]

    assert retry_pages == first_pages


def test_auto_seed_cli_passes_http_timeout(monkeypatch):
    seen = {}

    class FakeAgent:
        def __init__(self, base_url, run_id, agent_id, *, seed_count, timeout):
            seen["init"] = {
                "base_url": base_url,
                "run_id": run_id,
                "agent_id": agent_id,
                "seed_count": seed_count,
                "timeout": timeout,
            }

        def run(self, *, max_steps, verbose):
            seen["run"] = {"max_steps": max_steps, "verbose": verbose}

    monkeypatch.setattr(auto_seed_module, "AutoSeedAgent", FakeAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_seed.py",
            "--run-id",
            "run_1",
            "--base-url",
            "http://merchantbench.test",
            "--agent-id",
            "agent_0",
            "--seed-count",
            "7",
            "--max-steps",
            "99",
            "--timeout",
            "120",
            "--quiet",
        ],
    )

    assert auto_seed_module.main() == 0
    assert seen["init"] == {
        "base_url": "http://merchantbench.test",
        "run_id": "run_1",
        "agent_id": "agent_0",
        "seed_count": 7,
        "timeout": 120.0,
    }
    assert seen["run"] == {"max_steps": 99, "verbose": False}


def test_auto_seed_cli_defaults_to_long_http_timeout(monkeypatch):
    seen = {}

    class FakeAgent:
        def __init__(self, base_url, run_id, agent_id, *, seed_count, timeout):
            seen["timeout"] = timeout

        def run(self, *, max_steps, verbose):
            pass

    monkeypatch.setattr(auto_seed_module, "AutoSeedAgent", FakeAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_seed.py",
            "--run-id",
            "run_1",
            "--base-url",
            "http://merchantbench.test",
        ],
    )

    assert auto_seed_module.main() == 0
    assert seen["timeout"] == 600.0
