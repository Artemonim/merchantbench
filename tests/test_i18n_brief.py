"""Language resolution for bilingual brief fields."""
from tools.observation import _resolve_lang, _DEFAULT_ROLE, _DEFAULT_GOALS


def test_default_role_is_bilingual_dict():
    assert isinstance(_DEFAULT_ROLE, dict)
    assert "zh" in _DEFAULT_ROLE and "en" in _DEFAULT_ROLE


def test_default_goals_is_bilingual_dict_of_lists():
    assert isinstance(_DEFAULT_GOALS, dict)
    assert isinstance(_DEFAULT_GOALS["zh"], list) and _DEFAULT_GOALS["zh"]
    assert isinstance(_DEFAULT_GOALS["en"], list) and _DEFAULT_GOALS["en"]


def test_default_goals_use_total_assets_wording():
    assert any("总资产" in g for g in _DEFAULT_GOALS["zh"])
    assert any("total assets" in g.lower() for g in _DEFAULT_GOALS["en"])
    assert not any("cum_profit" in g for g in _DEFAULT_GOALS["zh"])


def test_default_role_does_not_leak_step_or_sim_hour():
    for txt in (_DEFAULT_ROLE["zh"], _DEFAULT_ROLE["en"]):
        assert "sim-hour" not in txt
        assert "simulator" not in txt.lower()
        assert "step" not in txt.lower()


def test_resolve_lang_picks_zh_by_default():
    assert _resolve_lang({"zh": "你好", "en": "hi"}, "zh") == "你好"


def test_resolve_lang_picks_en_when_requested():
    assert _resolve_lang({"zh": "你好", "en": "hi"}, "en") == "hi"


def test_resolve_lang_falls_back_when_requested_key_missing():
    assert _resolve_lang({"zh": "你好"}, "en") == "你好"


def test_resolve_lang_passes_plain_value_through():
    assert _resolve_lang("legacy string", "zh") == "legacy string"
    assert _resolve_lang(["a", "b"], "en") == ["a", "b"]


# ---------- compose_system_brief unit tests ----------

import os
import tempfile

import pytest

from web.app import create_app
from web.runner import load_default_scenario
from tools.observation import compose_system_brief


@pytest.fixture
def app_client():
    tmp = tempfile.mkdtemp()
    app = create_app(db_path=os.path.join(tmp, "test.db"),
                     runs_root=os.path.join(tmp, "runs"))
    with app.test_client() as c:
        yield c


def _make_run(c, agent_overrides=None, run_overrides=None,
              scenario_overrides=None):
    scen = load_default_scenario()
    scen["run"]["horizon_steps"] = 3
    scen["run"]["max_hook_seconds"] = 0.05
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 30
    if run_overrides:
        scen.setdefault("run", {}).update(run_overrides)
    if agent_overrides:
        scen.setdefault("agent", {}).update(agent_overrides)
    if scenario_overrides:
        for section, values in scenario_overrides.items():
            if isinstance(values, dict) and isinstance(scen.get(section), dict):
                scen[section].update(values)
            else:
                scen[section] = values
    return c.post("/runs", json={"scenario": scen}).get_json()["run_id"]


def _get_brief(c, rid):
    with c.application.app_context():
        env = c.application.registry._require(rid)
        return compose_system_brief(env)


def test_brief_defaults_to_en(app_client):
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    assert "MerchantBench store" in brief["role"]
    assert any("total assets" in g.lower() for g in brief["goals"])
    assert "MerchantBench store" in brief["system_prompt"]


def test_brief_honors_explicit_zh(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    assert "你是 MerchantBench" in brief["role"]
    assert any("总资产" in g for g in brief["goals"])
    assert "你是 MerchantBench" in brief["system_prompt"]


def test_brief_honors_explicit_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    assert "MerchantBench store" in brief["role"]
    assert any("total assets" in g.lower() for g in brief["goals"])
    assert "MerchantBench store" in brief["system_prompt"]


def test_brief_detailed_decision_principle_is_opt_in(app_client):
    default_rid = _make_run(app_client, {"language": "en"})
    default_prompt = _get_brief(app_client, default_rid)["system_prompt"]
    assert "Decision principle:" not in default_prompt
    assert "intermediate signals, not standalone objectives" not in default_prompt
    assert "independent demand opportunity" not in default_prompt

    detailed_rid = _make_run(
        app_client,
        {"language": "en", "detailed": True},
    )
    detailed_prompt = _get_brief(app_client, detailed_rid)["system_prompt"]
    assert "Decision principle:" in detailed_prompt
    assert (
        "Margin percentage, rating, and order count are intermediate signals, "
        "not standalone objectives."
    ) in detailed_prompt
    assert (
        "Each active listing generates an independent demand opportunity; "
        "removing one does not redistribute demand to the remaining listings."
    ) in detailed_prompt


def test_brief_detailed_decision_principle_supports_zh(app_client):
    rid = _make_run(
        app_client,
        {"language": "zh", "detailed": True},
    )
    prompt = _get_brief(app_client, rid)["system_prompt"]
    assert "决策原则:" in prompt
    assert "毛利率、评分和订单数是中间信号，不是独立目标" in prompt
    assert "下架某个商品不会将其需求重新分配给其余商品" in prompt


def test_brief_omits_low_price_penalty_heuristic(app_client):
    rid_zh = _make_run(app_client, {"language": "zh"})
    zh_prompt = _get_brief(app_client, rid_zh)["system_prompt"]
    assert "低客单价、低绝对毛利商品更脆弱" not in zh_prompt

    rid_en = _make_run(app_client, {"language": "en"})
    en_prompt = _get_brief(app_client, rid_en)["system_prompt"]
    assert "low-ASP, low-absolute-margin products more fragile" not in en_prompt


def test_brief_drops_http_protocol_details(app_client):
    """Brief must teach the agent the platform rules (penalty amounts,
    deposit-closure, initial capital) and surface tool discovery
    (`list_tools` / `end_of_step`) without leaking HTTP wire details."""
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    assert "conventions" not in brief
    assert "tool_names" not in brief
    ctx = brief.get("context", {})
    assert "max_hook_seconds" not in ctx
    sp = brief["system_prompt"]
    assert "GET /runs" not in sp
    assert "X-Agent-Turn-Id" not in sp
    assert "tools/schema" not in sp
    assert "sim-hour" not in sp


def test_brief_injects_business_period_and_activation_window(app_client):
    rid = _make_run(
        app_client,
        agent_overrides={"language": "zh", "activation_period": 6},
        run_overrides={"horizon_steps": 48, "step_hours": 1},
    )
    brief = _get_brief(app_client, rid)
    ctx = brief["context"]
    assert ctx["horizon_steps"] == 48
    assert ctx["step_hours"] == 1
    assert ctx["horizon_days"] == 2
    assert ctx["activation_period"] == 6
    assert ctx["activation_hours"] == 6
    sp = brief["system_prompt"]
    assert "经营期共 2 天" in sp
    assert "小时离散推进" in sp
    assert "每 6 小时被激活一次" in sp
    assert "之后的销量" in sp
    assert "48 个 step" not in sp
    assert "按本场景 YAML 配置" not in sp
    assert "约每" not in sp


def test_brief_injects_business_period_and_activation_window_en(app_client):
    rid = _make_run(
        app_client,
        agent_overrides={"language": "en", "activation_period": 6},
        run_overrides={"horizon_steps": 48, "step_hours": 1},
    )
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "operates for 2 days" in sp
    assert "discrete steps of 1 hour(s)" in sp
    assert "hourly discrete steps" not in sp
    assert "activated every 6 hours" in sp
    assert "future sales" in sp
    assert "48 steps" not in sp
    assert "about 2 days" not in sp
    assert "Per this scenario YAML" not in sp
    assert "about every" not in sp


def test_create_run_rejects_non_hourly_steps(app_client):
    scen = load_default_scenario()
    scen["run"]["horizon_steps"] = 4
    scen["run"]["max_hook_seconds"] = 0.05
    scen["run"]["step_hours"] = 24
    scen["data"]["source"] = "synthetic"
    scen["data"]["num_products"] = 30

    resp = app_client.post("/runs", json={"scenario": scen})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "run.step_hours must be 1 in this release"


def test_brief_mentions_step_release(app_client):
    """The agent needs to know `end_of_step` is how a step is released."""
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    assert "list_tools" not in brief["system_prompt"]
    assert "end_of_step" in brief["system_prompt"]


def test_brief_mentions_step_release_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "list_tools" not in sp
    assert "end_of_step" in sp
    assert "Operating period" in sp


def test_brief_states_shop_rating_signals_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"].lower()
    assert "shop rating (updated daily)" in sp
    assert "each terminal order counts once" in sp
    assert "normal 4.5×1" in sp
    assert "late but settled 3×1" in sp
    assert "refund-only 1.5×2" in sp
    assert "stockout 1×3" in sp
    assert "cancellations and insufficient-balance failures are excluded" in sp
    assert "starts at 4 with prior weight 20" in sp
    assert "30-day half-life" in sp
    assert "score ranges <2.5, [2.5,3.3), [3.3,3.8), [3.8,4.2), >=4.2" in sp
    assert "order traffic is multiplied by ×0.1, ×0.35, ×0.8, ×1, ×1.2" in sp


def test_brief_states_shop_rating_signals_zh(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "店铺评分（每日更新）" in sp
    assert "每张终态订单只计一次" in sp
    assert "正常 4.5×1" in sp
    assert "超时后结算 3×1" in sp
    assert "仅退款 1.5×2" in sp
    assert "缺货 1×3" in sp
    assert "取消和余额不足不计" in sp
    assert "新店评分 4（先验权重 20）" in sp
    assert "30 天半衰期" in sp
    assert "分数区间 <2.5、[2.5,3.3)、[3.3,3.8)、[3.8,4.2)、≥4.2" in sp
    assert "后续订单流量分别 ×0.1、×0.35、×0.8、×1、×1.2" in sp


def test_brief_reads_exact_shop_rating_rules_from_scenario(app_client):
    rid = _make_run(
        app_client,
        {"language": "en"},
        scenario_overrides={
            "rating_outcomes": {
                "normal_score": 4.6,
                "late_score": 3.1,
                "refund_score": 2.1,
                "only_refund_score": 1.6,
                "bad_review_score": 1.1,
                "stockout_score": 1.2,
                "normal_weight": 1.1,
                "late_weight": 1.2,
                "refund_weight": 1.3,
                "only_refund_weight": 2.1,
                "bad_review_weight": 2.2,
                "stockout_weight": 3.1,
            },
            "shop_rating": {
                "initial_rating": 3.9,
                "prior_weight": 17,
                "half_life_days": 14,
                "bucket_thresholds": [2.4, 3.2, 3.7, 4.1],
                "star_multipliers": [0.12, 0.4, 0.85, 1.05, 1.3],
            },
        },
    )
    sp = _get_brief(app_client, rid)["system_prompt"].lower()
    assert "normal 4.6×1.1" in sp
    assert "late but settled 3.1×1.2" in sp
    assert "refund 2.1×1.3" in sp
    assert "refund-only 1.6×2.1" in sp
    assert "bad review 1.1×2.2" in sp
    assert "stockout 1.2×3.1" in sp
    assert "starts at 3.9 with prior weight 17" in sp
    assert "14-day half-life" in sp
    assert "score ranges <2.4, [2.4,3.2), [3.2,3.7), [3.7,4.1), >=4.1" in sp
    assert "order traffic is multiplied by ×0.12, ×0.4, ×0.85, ×1.05, ×1.3" in sp
    assert "[2.5,3.3)" not in sp


def test_brief_injects_penalty_amounts(app_client):
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    pens = brief["context"]["penalties"]
    for k in ("cancel", "refund", "only_refund", "bad_review", "timeout",
              "stockout", "insufficient_balance"):
        assert k in pens, f"missing {k}"
        assert pens[k]["mode"] == "amount"
        assert pens[k]["amount"] >= 0
        assert "source" not in pens[k]
    assert pens["cancel"]["amount"] == 0
    assert pens["refund"]["amount"] == 8
    assert pens["only_refund"]["amount"] == 0
    assert pens["bad_review"]["amount"] == 5
    assert pens["timeout"]["amount"] == 3
    assert pens["stockout"]["amount"] == 5
    assert pens["insufficient_balance"]["amount"] == 5
    sp = brief["system_prompt"]
    assert "违约罚款" in sp or "Platform penalties" in sp
    assert "8.00" in sp
    assert "保证金" in sp or "deposit" in sp.lower()


def test_brief_states_deposit_closure_rule(app_client):
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "closes it immediately and permanently" in sp
    assert "deposit_pool" in sp
    assert "balance" in sp
    assert "All fines deduct balance first" in sp
    assert "Any cash credit first restores deposit_pool" in sp
    assert "min_deposit_alive" not in brief["context"]
    assert brief["context"]["initial_deposit"] == 1000.0


def test_brief_includes_demand_and_order_lifecycle(app_client):
    rid = _make_run(
        app_client,
        {"language": "zh"},
        scenario_overrides={"lifecycle": {"ramp_days": 9}},
    )
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "日期" in sp
    assert "季节" in sp
    assert "天气" not in sp
    assert "时段" in sp
    assert "商品上架周期" in sp
    assert "价格" in sp
    assert "一级类目热度" not in sp
    assert "动态 listing_rating" not in sp
    assert "listing_rating 影响" not in sp
    assert "影响该商品后续销量" not in sp
    assert "新上架商品初始曝光有限" in sp
    assert "上架满 9 天达到正常水平" in sp
    assert "balance" in sp
    assert "in_transit" in sp
    assert "receivable" in sp
    assert "purchase_price 从 in_transit 移除" in sp
    assert "sale_price 计入 receivable" in sp
    assert "正常订单和差评订单在到货后 0–7 天内结算销售货款" in sp
    assert "仅退款不产生回款且采购成本沉没" in sp
    for status in ("ordered", "late", "shipped", "delivered", "settled_normal"):
        assert status in sp
    assert "期末总资产按" not in sp


def test_brief_includes_supplier_abnormal_events(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "上游供应商异常事件" in sp
    assert "价格变化" in sp
    assert "供应商下架" in sp
    assert "供应商发货超时" in sp
    assert "一段时间后" in sp
    assert "恢复" in sp
    assert "可能是临时状态" in sp
    assert "上游目录商品评分基于历史数据" in sp
    assert "不一定决定未来销售表现" in sp


def test_brief_summarizes_available_actions_zh(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "可用动作:" in sp
    assert "选品: 根据需求、成本、质量信号和供应商可靠性决定卖什么。" in sp
    assert "店铺经营: 管理上架、价格、货架位和资金使用,在增长、毛利和风险之间平衡。" in sp
    assert "上游供应商处理: 应对供应商调价、下架、发货变慢或负毛利风险。" in sp
    assert "下游订单管理: 监控订单异常、应收款、现金和保证金风险。" in sp
    for tool_name in (
        "market_brief",
        "hot_search_terms",
        "search_products",
        "get_product_detail",
        "query_balance",
        "query_cash_pipeline",
    ):
        assert tool_name not in sp


def test_brief_includes_demand_and_order_lifecycle_en(app_client):
    rid = _make_run(
        app_client,
        {"language": "en"},
        run_overrides={"horizon_steps": 3},
        scenario_overrides={"lifecycle": {"ramp_days": 11}},
    )
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "seasonal demand" in sp.lower()
    assert "time of day" in sp.lower()
    assert "weather" not in sp.lower()
    assert "price" in sp.lower()
    assert "first-level category demand" not in sp
    assert "dynamic listing_rating" not in sp
    assert "listing_rating that affects its future sales" not in sp
    assert "sales are affected by date, season, weather, sale price, shop rating, and each product's listing_rating" not in sp.lower()
    assert "New listings have limited initial exposure" in sp
    assert "reaches its normal level 11 days after listing" in sp
    assert "balance" in sp
    assert "in_transit" in sp
    assert "receivable" in sp
    assert "purchase_price leaves in_transit" in sp
    assert "sale_price enters receivable" in sp
    assert "within 0–7 days after delivery" in sp
    assert "refund-only orders produce no cash credit" in sp
    for status in ("ordered", "late", "shipped", "delivered", "settled_normal"):
        assert status in sp
    assert "final net assets" not in sp


def test_brief_summarizes_available_actions_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "Available actions:" in sp
    assert "Sourcing: choose what to sell based on demand, cost, quality signals, and supplier reliability." in sp
    assert "Store operations: manage listings, prices, shelf slots, and cash usage to balance growth, margin, and risk." in sp
    assert "Upstream supplier handling: respond to supplier price changes, delisting, slower shipping, or negative-margin risk." in sp
    assert "Downstream order management: monitor order exceptions, receivables, cash, and deposit risk." in sp
    for tool_name in (
        "market_brief",
        "hot_search_terms",
        "search_products",
        "get_product_detail",
        "query_balance",
        "query_cash_pipeline",
    ):
        assert tool_name not in sp


def test_brief_includes_field_logic_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "Field logic" in sp
    assert "cash.net_assets = balance + deposit_pool + in_transit + receivable" in sp
    assert "order.net_profit = realized_revenue - realized_cost - total_penalty" in sp
    assert "already deducted when applied" in sp
    assert "listing.cum_gross_profit" not in sp
    assert "listing.cum_net_profit" not in sp


def test_brief_includes_field_logic_zh(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"]
    assert "字段口径" in sp
    assert "cash.net_assets = balance + deposit_pool + in_transit + receivable" in sp
    assert "order.net_profit = realized_revenue - realized_cost - total_penalty" in sp
    assert "罚金已在发生时扣除" in sp
    assert "listing.cum_gross_profit" not in sp
    assert "listing.cum_net_profit" not in sp


def test_brief_uses_configured_settlement_delay_in_lifecycle(app_client):
    rid = _make_run(
        app_client,
        {"language": "en"},
        run_overrides={"horizon_steps": 3},
    )
    with app_client.application.app_context():
        env = app_client.application.registry._require(rid)
        env.scenario["settlement"]["normal_delay_hours"] = 48
        brief = compose_system_brief(env)
    sp = brief["system_prompt"]
    assert brief["context"]["normal_settlement_delay_hours"] == 48
    assert "0–2 days" in sp


def test_brief_uses_configured_settlement_delay_in_lifecycle_zh(app_client):
    rid = _make_run(
        app_client,
        {"language": "zh"},
        run_overrides={"horizon_steps": 3},
    )
    with app_client.application.app_context():
        env = app_client.application.registry._require(rid)
        env.scenario["settlement"]["normal_delay_hours"] = 48
        brief = compose_system_brief(env)
    sp = brief["system_prompt"]
    assert brief["context"]["normal_settlement_delay_hours"] == 48
    assert "0–2 天" in sp


def test_brief_includes_supplier_abnormal_events_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    sp = brief["system_prompt"].lower()
    assert "upstream supplier abnormal events" in sp
    assert "price change" in sp
    assert "supplier delist" in sp
    assert "supplier shipping timeout" in sp
    assert "after a period of time" in sp
    assert "recover" in sp
    assert "may be temporary rather than permanent" in sp
    assert "upstream catalog product ratings are based on historical data" in sp
    assert "do not necessarily determine future sales performance" in sp


def test_brief_omits_list_tools_discovery_instruction(app_client):
    for language in ("zh", "en"):
        rid = _make_run(app_client, {"language": language})
        brief = _get_brief(app_client, rid)
        assert "list_tools" not in brief["system_prompt"]


def test_brief_includes_initial_capital(app_client):
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    ctx = brief["context"]
    assert ctx["initial_cash"] > 0
    assert ctx["initial_deposit"] > 0
    sp = brief["system_prompt"]
    assert f"{ctx['initial_cash']:.2f}" in sp
    assert f"{ctx['initial_deposit']:.2f}" in sp
    assert "locked guarantee unavailable for procurement" in sp
    assert "Listing a product is free" not in sp


def test_brief_keeps_ship_promise_constraints(app_client):
    rid = _make_run(app_client)
    brief = _get_brief(app_client, rid)
    assert brief["context"]["default_promised_ship_hours"] == 48
    assert "max_promised_ship_hours" not in brief["context"]
    assert "max_promised_logistics_hours" not in brief["context"]
    assert "default_promised_logistics_hours" not in brief["context"]
    sp = brief["system_prompt"]
    assert "Max promised ship hours" not in sp
    assert "发货承诺上限" not in sp
    assert "if unspecified at listing time" not in sp
    assert "promised_logistics_hours" not in sp


def test_default_scenario_does_not_expose_legacy_logistics_promise_config(app_client):
    scen = load_default_scenario()
    rules = scen["platform_rules"]
    assert "default_promised_ship_hours" in rules
    assert "max_promised_ship_hours" not in rules
    assert "max_promised_logistics_hours" not in rules
    assert "default_promised_logistics_hours" not in rules


def test_brief_includes_active_listing_limit(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    brief = _get_brief(app_client, rid)
    assert brief["context"]["max_active_listings"] == 50
    sp = brief["system_prompt"]
    assert "50" in sp
    assert "上架商品" in sp or "active listings" in sp
    assert "空置货架位会减少商品曝光" in sp
    assert "稀缺经营资源" not in sp
    assert "尽可能多利用店铺位置" not in sp


def test_brief_includes_active_listing_limit_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    brief = _get_brief(app_client, rid)
    assert brief["context"]["max_active_listings"] == 50
    sp = brief["system_prompt"]
    assert "50" in sp
    assert "active listings" in sp
    assert "Empty shelf slots reduce product exposure." in sp
    assert "scarce business resource" not in sp
    assert "use as many shelf slots as practical" not in sp


def test_brief_encourages_all_available_tools_en(app_client):
    rid = _make_run(app_client, {"language": "en"})
    sp = _get_brief(app_client, rid)["system_prompt"]
    assert (
        "Use any available tools and skills, including analysis, automation, and memory tools "
        "when provided, to improve long-run decisions and maximize net_assets."
    ) in sp


def test_brief_encourages_all_available_tools_zh(app_client):
    rid = _make_run(app_client, {"language": "zh"})
    sp = _get_brief(app_client, rid)["system_prompt"]
    assert (
        "请利用所有可用工具，包括在提供时可用的分析、自动化和记忆工具，以及所有可用技能，"
        "持续改进长期经营决策并最大化 net_assets。"
    ) in sp
