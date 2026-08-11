"""Agent system brief + per-step observation packet.

`compose_system_brief` returns the role/goals string the env hands to the
agent at register time. `compose_observation` builds the slim per-step
packet (day/hour + cash + my-listing event counts + order summary) that
the long-poll /observation endpoint serves whenever the hook opens.
"""
from __future__ import annotations

import copy
from typing import Optional, get_args

from core import listing_rating as lr_mod
from core import public_reviews as public_reviews_mod
from core.entities import OrderStatus
from core import sim_time
from core.simulator import Environment
from storage import agent_log
from storage import db as dbm
from tools import registry, tools as t
from tools.table import compact_table


# ---------- agent brief (fetched once at register time) ----------

_DEFAULT_ROLE = {
    "zh": "你是 MerchantBench 平台上一家小店的经营 agent。你会周期性收到当前店铺观测。",
    "en": ("You are the operating agent of a small MerchantBench store. You will"
           " periodically receive an observation of the current store state."),
}

_DEFAULT_GOALS = {
    "zh": ["最大化总资产"],
    "en": ["Maximize total assets"],
}


def _resolve_lang(value, language: str):
    """If value is a {zh, en} dict, pick the matching key (fall back to the
    other language if missing). If value is anything else, pass through as-is
    so existing scenario YAML written as plain strings/lists still works."""
    if isinstance(value, dict) and ("zh" in value or "en" in value):
        if language in value:
            return value[language]
        other = "en" if language == "zh" else "zh"
        return value.get(other)
    return value


def _penalty_spec(rules: dict, kind: str) -> dict:
    spec = {}
    amount_key = f"{kind}_penalty_amount"
    if amount_key in rules:
        spec.update({"mode": "amount", "amount": float(rules[amount_key])})
    else:
        spec.update({"mode": "ratio", "ratio": float(rules[f"{kind}_penalty_ratio"])})
    return spec


def _format_penalty_zh(kind: str, p: dict) -> str:
    if p["mode"] == "amount":
        amount = float(p["amount"])
        if amount == 0.0:
            if kind == "only_refund":
                return "进货价沉没,无额外罚金"
            return "无额外罚金"
        return f"固定 {amount:.2f} 元"
    return f"{p['ratio'] * 100:.1f}%"


def _format_penalty_en(kind: str, p: dict) -> str:
    if p["mode"] == "amount":
        amount = float(p["amount"])
        if amount == 0.0:
            if kind == "only_refund":
                return "purchase cost is lost; no extra fine"
            return "no extra fine"
        return f"fixed {amount:.2f}"
    return f"{p['ratio'] * 100:.1f}%"


def _format_span_days(hours: float) -> str:
    days = float(hours) / 24.0
    if days.is_integer():
        return str(int(days))
    return f"{days:.1f}".rstrip("0").rstrip(".")


def _format_duration_zh(hours: int) -> str:
    if hours % 24 == 0:
        return f"{hours // 24} 天"
    return f"{hours} 小时"


def _format_duration_en(hours: int) -> str:
    if hours % 24 == 0:
        days = hours // 24
        unit = "day" if days == 1 else "days"
        return f"{days} {unit}"
    unit = "hour" if hours == 1 else "hours"
    return f"{hours} {unit}"


def _format_rule_number(value: float) -> str:
    """Render a scenario number compactly without changing its value."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _rating_bucket_ranges(thresholds: list[float], language: str) -> str:
    """Render the exact half-open score ranges configured for star buckets."""
    values = [_format_rule_number(v) for v in thresholds]
    if not values:
        return "all scores" if language == "en" else "全部分数"
    ranges = [f"<{values[0]}"]
    ranges.extend(
        f"[{left},{right})"
        for left, right in zip(values, values[1:])
    )
    ranges.append(f">={values[-1]}" if language == "en" else f"≥{values[-1]}")
    return (", ".join(ranges) if language == "en" else "、".join(ranges))


def _order_outcome_rating_rules(env: Environment) -> dict:
    """Return effective order-outcome rating rules, including defaults."""
    configured = env.scenario.get("rating_outcomes") or {}
    scores = {
        key: float(configured.get(key, default))
        for key, default in lr_mod.DEFAULT_OUTCOME_SCORES.items()
    }
    weights = {
        key: float(configured.get(key, default))
        for key, default in lr_mod.DEFAULT_OUTCOME_WEIGHTS.items()
    }
    return {"scores": scores, "weights": weights}


def _reputation_volume_rules(rating_cfg: dict) -> dict[str, float]:
    """Return effective lifetime-volume trust settings."""
    return lr_mod.resolve_reputation_volume_config(
        rating_cfg.get("reputation_volume"),
    )


def _public_review_rules(review_cfg: dict) -> dict:
    """Return effective deterministic public-review sampling settings."""
    return public_reviews_mod.resolve_public_review_config(review_cfg)


def _public_review_demand_rules(review_cfg: dict) -> dict[str, float]:
    """Return effective public-review demand settings."""
    return public_reviews_mod.resolve_public_review_demand_config(review_cfg)


def _public_review_probability_text(review_cfg: dict, language: str) -> str:
    """Render star-indexed public-review response probabilities."""
    probabilities = _public_review_rules(review_cfg)["probability_by_star"]
    separator = "、" if language == "zh" else ", "
    return separator.join(
        f"{stars}★ {_format_rule_number(probability * 100)}%"
        for stars, probability in enumerate(probabilities, start=1)
    )


_PENALTY_LABELS_ZH = {
    "cancel":               "买家取消订单 (包括运送中)",
    "refund":               "品质退货",
    "only_refund":          "仅退款",
    "bad_review":           "差评",
    "timeout":              "物流超时 (实际发货时长 > 承诺发货时长)",
    "stockout":             "缺货违约 (订单到达但供应商已下架/库存为 0)",
    "insufficient_balance": "资金不足 (订单到达但 balance < 采购款)",
}
_PENALTY_LABELS_EN = {
    "cancel":               "buyer cancel (including in transit)",
    "refund":               "quality return",
    "only_refund":          "refund-only",
    "bad_review":           "bad review",
    "timeout":              "shipping timeout (actual ship time > promised ship time)",
    "stockout":             "stockout violation (order arrives but supplier delisted / qty=0)",
    "insufficient_balance": "insufficient balance (order arrives but balance < purchase price)",
}

_ORDER_STATUS_KEYS = tuple(get_args(OrderStatus))


def compose_system_brief(env: Environment) -> dict:
    """The env tells the agent who it is + what the platform rules are.

    Agents call this once at register time. The returned `system_prompt` is a
    composed string ready to be used as the LLM's system message; structured
    fields (role/goals/context) are also returned for agents that want to
    assemble their own prompt.

    Platform rules — penalty amounts or ratios, cash/guarantee policy,
    guarantee-depletion closure rule, initial capital, and shop-rating effects
    — are injected from the active scenario so the LLM agent can reason about
    expected cost of each action. HTTP wire conventions live in sdk/README.md
    (not in the prompt).
    """
    cfg = _scenario_agent_cfg(env)
    language = (cfg.get("language") or "en").lower()
    if language not in ("zh", "en"):
        language = "en"

    role = _resolve_lang(_DEFAULT_ROLE, language)
    goals = _resolve_lang(_DEFAULT_GOALS, language)

    rules = env.scenario["platform_rules"]
    run_cfg = env.scenario["run"]
    step_hours = int(run_cfg.get("step_hours", 1))
    horizon_steps = int(run_cfg.get("horizon_steps", 0))
    horizon_hours = horizon_steps * step_hours
    horizon_days = _format_span_days(horizon_hours)
    activation_period = int(cfg.get("activation_period") or 1)
    if activation_period < 1:
        activation_period = 1
    activation_hours = activation_period * step_hours
    settlement_cfg = env.scenario.get("settlement") or {}
    normal_settlement_delay_hours = int(settlement_cfg.get("normal_delay_hours", 168))
    normal_settlement_delay_zh = _format_duration_zh(normal_settlement_delay_hours)
    normal_settlement_delay_en = _format_duration_en(normal_settlement_delay_hours)
    lifecycle_cfg = env.scenario.get("lifecycle") or {}
    ramp_days = _format_rule_number(lifecycle_cfg.get("ramp_days", 0))
    start_date = sim_time.start_date(env.scenario)
    initial_cash = float(run_cfg.get("initial_cash", 0.0))
    initial_deposit = float(run_cfg.get("initial_deposit", 1000.0))
    default_ship = int(rules.get("default_promised_ship_hours", 48))

    penalty_keys = ("cancel", "refund", "only_refund", "bad_review", "timeout",
                    "stockout", "insufficient_balance")
    penalties = {}
    for k in penalty_keys:
        penalties[k] = _penalty_spec(rules, k)

    context = {
        "default_promised_ship_hours": default_ship,
        "max_active_listings": int(rules.get("max_active_listings", 100)),
        "horizon_steps": horizon_steps,
        "step_hours": step_hours,
        "horizon_days": int(horizon_days) if horizon_days.isdigit() else float(horizon_days),
        "activation_period": activation_period,
        "activation_hours": activation_hours,
        "normal_settlement_delay_hours": normal_settlement_delay_hours,
        "virtual_start_date": start_date.isoformat() if start_date else None,
        "initial_cash": initial_cash,
        "initial_deposit": initial_deposit,
        "catalog_categories": sorted({p.category for p in env.products.values()}),
        "penalties": penalties,
        "field_logic": {
            "cash.net_assets": "balance + deposit_pool + in_transit + receivable",
            "order.net_profit": "realized_revenue - realized_cost - total_penalty",
            "fines": "deduct balance first, then deposit_pool; already reflected in net_assets",
            "cash_credits": "restore deposit_pool to initial_deposit first, then enter balance",
        },
    }

    rating_cfg = env.scenario.get("shop_rating") or {}
    rating_enabled = bool(rating_cfg.get("enabled", False))

    if language == "zh":
        lines = [role, "", "目标:"]
        for g in goals:
            lines.append(f"  - {g}")
        if env.scenario.get("agent", {}).get("detailed") is True:
            lines.append("")
            lines.append("决策原则:")
            lines.append(
                "  - 最大化最终 net_assets。毛利率、评分和订单数是中间信号，不是独立目标。"
            )
            lines.append(
                "  - 每个在架商品独立产生一个需求机会；下架某个商品不会将其需求重新分配给其余商品。"
            )
        lines.append("")
        lines.append("经营周期:")
        if start_date:
            lines.append(f"  - 经营期共 {horizon_days} 天, 从 {start_date.isoformat()} 开始。")
        else:
            lines.append(f"  - 经营期共 {horizon_days} 天。")
        lines.append(
            f"  - 环境按 {step_hours} 小时离散推进; 你每 {activation_hours} 小时被激活一次并获得行动窗口。"
        )
        lines.append(
            "  - 当前小时的上架、调价、下架等动作只影响之后的销量,不影响当前小时已经生成的订单。"
        )
        lines.append(
            "  - 每步收到观测后,完成判断和动作,最后调用 `end_of_step` 工具释放本步 hook 进入下一步。"
        )
        lines.append("")
        lines.append("初始资金:")
        lines.append(f"  - balance: {initial_cash:.2f},可用于采购。")
        lines.append(f"  - deposit_pool: {initial_deposit:.2f},为锁定的履约保证金,不可用于采购。")
        lines.append("")
        lines.append("可用动作:")
        lines.append(
            "  - 选品: 根据需求、成本、质量信号和供应商可靠性决定卖什么。"
        )
        lines.append(
            "  - 店铺经营: 管理上架、价格、货架位和资金使用,在增长、毛利和风险之间平衡。"
        )
        lines.append(
            "  - 上游供应商处理: 应对供应商调价、下架、发货变慢或负毛利风险。"
        )
        lines.append(
            "  - 下游订单管理: 监控订单异常、应收款、现金和保证金风险。"
        )
        lines.append(
            "  - 请利用所有可用工具，包括在提供时可用的分析、自动化和记忆工具，以及所有可用技能，"
            "持续改进长期经营决策并最大化 net_assets。"
        )
        lines.append("")
        lines.append("需求与销售:")
        lines.append(
            "  - 销售会受日期与季节性需求、时段、销售价格、商品上架周期和店铺评级影响。"
        )
        lines.append(
            f"  - 新上架商品初始曝光有限，流量逐步爬坡，上架满 {ramp_days} 天达到正常水平。"
        )
        lines.append(
            "  - 上游目录商品评分基于历史数据,可作为参考信号,但不一定决定未来销售表现。"
        )
        lines.append("")
        lines.append("上游供应商异常事件:")
        lines.append(
            "  - 供应商侧可能出现价格变化、供应商下架、供应商发货超时三类异常事件。"
        )
        lines.append(
            "  - 这些异常可能是临时状态而非永久异常,通常持续一段时间后恢复或结束; 期间会影响采购成本、可售状态或实际发货时长。"
        )
        lines.append("")
        lines.append("订单生命周期与资金字段:")
        lines.append(
            "  - 客户下单后系统会自动尝试按 supplier_price 采购: balance 立即扣采购成本,in_transit 增加。"
        )
        lines.append(
            "  - 商品送达时,purchase_price 从 in_transit 移除,sale_price 计入 receivable。"
        )
        lines.append(
            f"  - 正常订单和差评订单在到货后 0–{normal_settlement_delay_zh}内结算销售货款; "
            "买家取消和品质退货会退回采购成本; 仅退款不产生回款且采购成本沉没。"
        )
        lines.append(
            f"  - 任何现金回款先将 deposit_pool 补至初始金额 {initial_deposit:.2f}; 余额才进入 balance。"
        )
        lines.append(
            "  - 常见状态生命周期: ordered -> shipped -> delivered -> settled_normal; 若发货超过承诺会先进入 late 后继续流转。"
        )
        lines.append(
            "  - 异常终态/结算态包括 cancelled / settled_refund / settled_only_refund / settled_bad_review / stockout / insufficient_balance。"
        )
        lines.append("")
        lines.append("字段口径:")
        lines.append("  - cash.net_assets = balance + deposit_pool + in_transit + receivable, 是主要总资产口径。")
        lines.append("  - order.net_profit = realized_revenue - realized_cost - total_penalty。")
        lines.append(
            "  - 罚金已在发生时扣除; 不要从 cash.net_assets 中重复扣除。"
        )
        lines.append("")
        lines.append("罚款与关店:")
        lines.append(
            "  - 所有罚款先扣 balance; balance 不足的部分再扣 deposit_pool。"
        )
        lines.append(
            "  - balance 为 0 不会关店; deposit_pool 为 0 时立即且永久关店。"
        )
        lines.append("")
        lines.append("平台违约罚款:")
        for k in penalty_keys:
            p = penalties[k]
            label = _PENALTY_LABELS_ZH[k]
            lines.append(f"  - {label}: {_format_penalty_zh(k, p)}")
        lines.append("")
        lines.append(
            f"上架商品数量限制: 当前店铺最多同时上架 {context['max_active_listings']} 个商品。"
        )
        lines.append("空置货架位会减少商品曝光。")
        if rating_enabled:
            lines.append("")
            rating_model = str(rating_cfg.get("model") or "beta_event_v1")
            if rating_model in lr_mod.ORDER_OUTCOME_RATING_MODELS:
                outcome_rules = _order_outcome_rating_rules(env)
                scores = outcome_rules["scores"]
                weights = outcome_rules["weights"]
                thresholds = [float(v) for v in rating_cfg["bucket_thresholds"]]
                multipliers = [float(v) for v in rating_cfg["star_multipliers"]]
                lines.append("店铺评分（每日更新）:")
                lines.append(
                    "  - 每张终态订单只计一次: "
                    f"正常 {_format_rule_number(scores['normal_score'])}×{_format_rule_number(weights['normal_weight'])}, "
                    f"超时后结算 {_format_rule_number(scores['late_score'])}×{_format_rule_number(weights['late_weight'])}, "
                    f"退款 {_format_rule_number(scores['refund_score'])}×{_format_rule_number(weights['refund_weight'])}, "
                    f"仅退款 {_format_rule_number(scores['only_refund_score'])}×{_format_rule_number(weights['only_refund_weight'])}, "
                    f"差评 {_format_rule_number(scores['bad_review_score'])}×{_format_rule_number(weights['bad_review_weight'])}, "
                    f"缺货 {_format_rule_number(scores['stockout_score'])}×{_format_rule_number(weights['stockout_weight'])}; "
                    "取消和余额不足不计。"
                )
                if rating_model in {
                    lr_mod.REPUTATION_VOLUME_RATING_MODEL,
                    lr_mod.PUBLIC_REVIEW_RATING_MODEL,
                }:
                    lines.append(
                        f"  - 近期质量在无真实证据时显示 {_format_rule_number(rating_cfg.get('initial_rating', 4.0))}"
                        f"（先验权重 {_format_rule_number(rating_cfg.get('prior_weight', 0.0))}）; "
                        f"质量证据按 {_format_rule_number(rating_cfg.get('half_life_days', 180.0))} 天半衰期衰减。"
                    )
                if rating_model == lr_mod.REPUTATION_VOLUME_RATING_MODEL:
                    reputation = _reputation_volume_rules(rating_cfg)
                    lines.append(
                        "  - 终身合格交易证据数不衰减。信誉量乘子从 "
                        f"×{_format_rule_number(reputation['min_multiplier'])} 渐近至 "
                        f"×{_format_rule_number(reputation['max_multiplier'])}; "
                        f"累计 {_format_rule_number(reputation['half_saturation_orders'])} 笔合格交易时获得一半信誉差距。"
                    )
                    lines.append(
                        f"  - 分数区间 {_rating_bucket_ranges(thresholds, 'zh')} 分别对应 "
                        f"1–{len(thresholds) + 1} 星及质量乘子 "
                        + "、".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + "；最终订单流量 = 质量乘子 × 信誉量乘子。"
                    )
                elif rating_model == lr_mod.PUBLIC_REVIEW_RATING_MODEL:
                    lines.append(
                        f"  - 公开评分区间 {_rating_bucket_ranges(thresholds, 'zh')} 分别对应 "
                        f"1–{len(thresholds) + 1} 星及置信度调整前的原始买家乘子 "
                        + "、".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + "。"
                    )
                else:
                    lines.append(
                        f"  - 新店评分 {_format_rule_number(rating_cfg.get('initial_rating', 4.0))}"
                        f"（先验权重 {_format_rule_number(rating_cfg.get('prior_weight', 20.0))}）, "
                        f"历史证据按 {_format_rule_number(rating_cfg.get('half_life_days', 30.0))} 天半衰期衰减。"
                    )
                    lines.append(
                        f"  - 分数区间 {_rating_bucket_ranges(thresholds, 'zh')} 分别对应 "
                        f"1–{len(thresholds) + 1} 星; 后续订单流量分别 "
                        + "、".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + "。"
                    )
                review_cfg = env.scenario.get("public_reviews") or {}
                if (
                    review_cfg.get("enabled", False)
                    and rating_model == lr_mod.PUBLIC_REVIEW_RATING_MODEL
                ):
                    lines.append("公开评价（买家可见并决定订单流量）:")
                    lines.append(
                        "  - 已有 settled_bad_review 按定义必定公开；其他合格交易按星级独立抽样: "
                        f"{_public_review_probability_text(review_cfg, 'zh')}。"
                    )
                    demand = _public_review_demand_rules(review_cfg)
                    lines.append(
                        "  - 公开评分效应按 c=n/(n+h) 向中性 1× 收缩，其中 "
                        f"h={_format_rule_number(demand['half_saturation_reviews'])} 条评价；"
                        f"卖家信任同时从 ×{_format_rule_number(demand['min_trust_multiplier'])} "
                        f"渐近至 ×{_format_rule_number(demand['max_trust_multiplier'])}。"
                    )
                    lines.append(
                        "  - 最终订单流量 = 置信度调整后的公开评分乘子 × 评价量信任乘子；"
                        "近期内部服务质量单独显示，不直接进入 v4 订单流量。"
                    )
                    lines.append(
                        "  - 抽样由订单和主种子确定，不消耗经济 RNG。每次观测的“公开评价”行"
                        "显示公开评分、回复率、全回复基准及选择偏差。"
                    )
            else:
                lines.append(
                    "本店下游店铺评分按每日终态订单结果更新；买家取消不参与店铺评分。"
                    "正常履约贡献较好的评分证据；发货超时、退款、仅退款、差评和 stockout "
                    "会产生较差证据，严重异常影响更大。评分进入 1-5 星离散档位，"
                    "星级越高通常客流越多。"
                )
        lines.append("")
        lines.append("平台一级类目:")
        lines.append("  - " + ", ".join(context["catalog_categories"]))
        if sim_time.virtual_time_enabled(env.scenario):
            lines.append("")
            lines.append("时间显示:")
            _example_dt = sim_time.time_view(env.scenario, t=46, step_hours=1).get("datetime", "")
            lines.append(f"  - 时间默认同时显示仿真时间和日历时间,格式如 `Day 2, Hour 22 ({_example_dt})`; 工具参数仍使用 day/hour。")
    else:
        lines = [role, "", "Goals:"]
        for g in goals:
            lines.append(f"  - {g}")
        if env.scenario.get("agent", {}).get("detailed") is True:
            lines.append("")
            lines.append("Decision principle:")
            lines.append(
                "  - Maximize final net_assets. Margin percentage, rating, and order count "
                "are intermediate signals, not standalone objectives."
            )
            lines.append(
                "  - Each active listing generates an independent demand opportunity; "
                "removing one does not redistribute demand to the remaining listings."
            )
        lines.append("")
        lines.append("Operating period:")
        if start_date:
            lines.append(f"  - The store operates for {horizon_days} days, starting from {start_date.isoformat()}.")
        else:
            lines.append(f"  - The store operates for {horizon_days} days.")
        lines.append(
            f"  - The environment advances in discrete steps of {step_hours} hour(s); you are activated every {activation_hours} hours."
        )
        lines.append(
            "  - Listing, price, and delisting actions in the current hour affect future sales only; they do not affect orders already generated for the current hour."
        )
        lines.append(
            "  - Each step: receive an observation, take your actions, then call `end_of_step` to release the per-step hook and advance."
        )
        lines.append("")
        lines.append("Initial capital:")
        lines.append(f"  - balance: {initial_cash:.2f}, usable for procurement.")
        lines.append(f"  - deposit_pool: {initial_deposit:.2f}, a locked guarantee unavailable for procurement.")
        lines.append("")
        lines.append("Available actions:")
        lines.append(
            "  - Sourcing: choose what to sell based on demand, cost, quality signals, and supplier reliability."
        )
        lines.append(
            "  - Store operations: manage listings, prices, shelf slots, and cash usage to balance growth, margin, and risk."
        )
        lines.append(
            "  - Upstream supplier handling: respond to supplier price changes, delisting, slower shipping, or negative-margin risk."
        )
        lines.append(
            "  - Downstream order management: monitor order exceptions, receivables, cash, and deposit risk."
        )
        lines.append(
            "  - Use any available tools and skills, including analysis, automation, and memory tools "
            "when provided, to improve long-run decisions and maximize net_assets."
        )
        lines.append("")
        lines.append("Demand and sales:")
        lines.append(
            "  - Sales are affected by seasonal demand, time of day, sale price, listing lifecycle, and shop rating."
        )
        lines.append(
            f"  - New listings have limited initial exposure; traffic ramps gradually and reaches its normal level {ramp_days} days after listing."
        )
        lines.append(
            "  - Upstream catalog product ratings are based on historical data. They can be useful reference signals, but they do not necessarily determine future sales performance."
        )
        lines.append("")
        lines.append("Upstream supplier abnormal events:")
        lines.append(
            "  - Supplier-side abnormal events include price change, supplier delist, and supplier shipping timeout."
        )
        lines.append(
            "  - These abnormal states may be temporary rather than permanent; affected suppliers or products usually recover or end after a period of time. While active, they can affect procurement cost, sale availability, or actual ship time."
        )
        lines.append("")
        lines.append("Order lifecycle and cash fields:")
        lines.append(
            "  - When a customer orders, the system automatically tries to procure the product at supplier_price: balance is debited immediately and in_transit increases."
        )
        lines.append(
            "  - At delivery, purchase_price leaves in_transit and sale_price enters receivable."
        )
        lines.append(
            f"  - Normal and bad-review orders settle their sale proceeds within 0–{normal_settlement_delay_en} after delivery; "
            "buyer cancellations and quality returns recover the procurement cost; refund-only orders produce no cash credit and the procurement cost is lost."
        )
        lines.append(
            f"  - Any cash credit first restores deposit_pool to its initial amount of {initial_deposit:.2f}; only the remainder enters balance."
        )
        lines.append(
            "  - Common status lifecycle: ordered -> shipped -> delivered -> settled_normal; if shipping exceeds the promise, the order enters late first and then continues flowing."
        )
        lines.append(
            "  - Abnormal terminal/settlement states include cancelled / settled_refund / settled_only_refund / settled_bad_review / stockout / insufficient_balance."
        )
        lines.append("")
        lines.append("Field logic:")
        lines.append("  - cash.net_assets = balance + deposit_pool + in_transit + receivable; use it as the main total-assets view.")
        lines.append("  - order.net_profit = realized_revenue - realized_cost - total_penalty.")
        lines.append(
            "  - Fines are already deducted when applied; do not subtract them again from cash.net_assets."
        )
        lines.append("")
        lines.append("Penalty and closure:")
        lines.append(
            "  - All fines deduct balance first; any unpaid remainder deducts deposit_pool."
        )
        lines.append(
            "  - balance reaching 0 does not close the shop; deposit_pool reaching 0 closes it immediately and permanently."
        )
        lines.append("")
        lines.append("Platform penalties:")
        for k in penalty_keys:
            p = penalties[k]
            label = _PENALTY_LABELS_EN[k]
            lines.append(f"  - {label}: {_format_penalty_en(k, p)}")
        lines.append("")
        lines.append(
            f"Active listings limit: this shop may have at most {context['max_active_listings']} active listings at once."
        )
        lines.append("Empty shelf slots reduce product exposure.")
        if rating_enabled:
            lines.append("")
            rating_model = str(rating_cfg.get("model") or "beta_event_v1")
            if rating_model in lr_mod.ORDER_OUTCOME_RATING_MODELS:
                outcome_rules = _order_outcome_rating_rules(env)
                scores = outcome_rules["scores"]
                weights = outcome_rules["weights"]
                thresholds = [float(v) for v in rating_cfg["bucket_thresholds"]]
                multipliers = [float(v) for v in rating_cfg["star_multipliers"]]
                lines.append("Shop rating (updated daily):")
                lines.append(
                    "  - Each terminal order counts once: "
                    f"normal {_format_rule_number(scores['normal_score'])}×{_format_rule_number(weights['normal_weight'])}, "
                    f"late but settled {_format_rule_number(scores['late_score'])}×{_format_rule_number(weights['late_weight'])}, "
                    f"refund {_format_rule_number(scores['refund_score'])}×{_format_rule_number(weights['refund_weight'])}, "
                    f"refund-only {_format_rule_number(scores['only_refund_score'])}×{_format_rule_number(weights['only_refund_weight'])}, "
                    f"bad review {_format_rule_number(scores['bad_review_score'])}×{_format_rule_number(weights['bad_review_weight'])}, "
                    f"stockout {_format_rule_number(scores['stockout_score'])}×{_format_rule_number(weights['stockout_weight'])}; "
                    "cancellations and insufficient-balance failures are excluded."
                )
                if rating_model in {
                    lr_mod.REPUTATION_VOLUME_RATING_MODEL,
                    lr_mod.PUBLIC_REVIEW_RATING_MODEL,
                }:
                    lines.append(
                        f"  - Recent quality displays {_format_rule_number(rating_cfg.get('initial_rating', 4.0))} "
                        f"before real evidence, with prior weight {_format_rule_number(rating_cfg.get('prior_weight', 0.0))}; "
                        f"quality evidence decays with a {_format_rule_number(rating_cfg.get('half_life_days', 180.0))}-day half-life."
                    )
                if rating_model == lr_mod.REPUTATION_VOLUME_RATING_MODEL:
                    reputation = _reputation_volume_rules(rating_cfg)
                    lines.append(
                        "  - Lifetime qualified-transaction evidence never decays. Its reputation multiplier rises from "
                        f"×{_format_rule_number(reputation['min_multiplier'])} toward "
                        f"×{_format_rule_number(reputation['max_multiplier'])}, earning half the trust gap at "
                        f"{_format_rule_number(reputation['half_saturation_orders'])} qualified transactions."
                    )
                    lines.append(
                        f"  - Score ranges {_rating_bucket_ranges(thresholds, 'en')} map to "
                        f"1–{len(thresholds) + 1} stars and quality multipliers "
                        + ", ".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + "; final order traffic = quality multiplier × reputation multiplier."
                    )
                elif rating_model == lr_mod.PUBLIC_REVIEW_RATING_MODEL:
                    lines.append(
                        f"  - Public score ranges {_rating_bucket_ranges(thresholds, 'en')} map to "
                        f"1–{len(thresholds) + 1} stars and pre-confidence buyer multipliers "
                        + ", ".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + "."
                    )
                else:
                    lines.append(
                        f"  - A new shop starts at {_format_rule_number(rating_cfg.get('initial_rating', 4.0))} "
                        f"with prior weight {_format_rule_number(rating_cfg.get('prior_weight', 20.0))}; "
                        f"evidence decays with a {_format_rule_number(rating_cfg.get('half_life_days', 30.0))}-day half-life."
                    )
                    lines.append(
                        f"  - Score ranges {_rating_bucket_ranges(thresholds, 'en')} map to "
                        f"1–{len(thresholds) + 1} stars; subsequent order traffic is multiplied by "
                        + ", ".join(f"×{_format_rule_number(v)}" for v in multipliers)
                        + ", respectively."
                    )
                review_cfg = env.scenario.get("public_reviews") or {}
                if (
                    review_cfg.get("enabled", False)
                    and rating_model == lr_mod.PUBLIC_REVIEW_RATING_MODEL
                ):
                    lines.append("Public reviews (buyer-visible and demand-driving):")
                    lines.append(
                        "  - Existing settled_bad_review outcomes are public by definition; other qualified transactions "
                        "respond independently by star: "
                        f"{_public_review_probability_text(review_cfg, 'en')}."
                    )
                    demand = _public_review_demand_rules(review_cfg)
                    lines.append(
                        "  - The public star effect is shrunk toward neutral 1× by confidence c=n/(n+h), "
                        f"with h={_format_rule_number(demand['half_saturation_reviews'])} reviews; seller trust also rises from "
                        f"×{_format_rule_number(demand['min_trust_multiplier'])} toward "
                        f"×{_format_rule_number(demand['max_trust_multiplier'])}."
                    )
                    lines.append(
                        "  - Final order traffic = confidence-adjusted public-rating multiplier × review-volume trust; "
                        "recent internal service quality is reported separately and does not directly enter v4 traffic."
                    )
                    lines.append(
                        "  - Sampling is derived from the order and master seed and does not consume economic RNG. "
                        "The Public reviews line in each observation reports the public rating, response rate, "
                        "all-response benchmark, and selection gap."
                    )
            else:
                lines.append(
                    "Your downstream shop rating is updated daily from final order outcomes. "
                    "Buyer cancellations do not affect shop rating. Normal fulfillment contributes "
                    "better rating evidence; shipping timeouts, refunds, refund-only cases, "
                    "bad reviews, and stockouts contribute worse evidence, with serious failures "
                    "having greater impact. The score enters discrete 1-5 star buckets, and "
                    "higher stars usually bring more customers."
                )
        lines.append("")
        lines.append("First-level marketplace categories:")
        lines.append("  - " + ", ".join(context["catalog_categories"]))
        if sim_time.virtual_time_enabled(env.scenario):
            lines.append("")
            lines.append("Time display:")
            _example_dt_en = sim_time.time_view(env.scenario, t=46, step_hours=1).get("datetime", "")
            lines.append("  - Time is shown as both simulation time and calendar time,"
                         f" for example `Day 2, Hour 22 ({_example_dt_en})`;"
                         " tool arguments still use day/hour.")

    return {
        "system_prompt": "\n".join(lines),
        "role": role,
        "goals": goals,
        "context": context,
        "language": language,
    }


def _scenario_agent_cfg(env: Environment) -> dict:
    return env.scenario.get("agent", {}) or {}


def _denylist(env: Environment) -> Optional[list[str]]:
    cfg = _scenario_agent_cfg(env)
    return cfg.get("tool_denylist")


def _tool_available(env: Environment, tool_name: str) -> bool:
    return tool_name not in set(_denylist(env) or [])


_MY_LISTING_EVENT_BUCKETS = {
    "price_change":             "price_changes",
    "supplier_delist":          "supplier_delists",
    "supplier_timeout":         "timeouts",
    "order_stockout_violation": "stockouts",
}

# Public alias — the set of event types that count as "my listing got hit by
# something abnormal." Both the observation packet's supply counts and the
# `query_supply_chain_anomalies` tool use this single definition.
ABNORMAL_LISTING_EVENT_TYPES = frozenset(_MY_LISTING_EVENT_BUCKETS.keys())
ANOMALY_LISTING_COLUMNS = (
    "product_id",
    "name",
    "sale_price",
    "supplier_price",
    "supplier_ship_hours",
    "supplier_logistics_hours",
    "supplier_listed",
)

def _zero_order_status_counts() -> dict:
    return {"total": 0, **{status: 0 for status in _ORDER_STATUS_KEYS}}

_SUPPLY_EVENT_ZERO = {
    "price_changes": 0,
    "supplier_delists": 0,
    "timeouts": 0,
    "stockouts": 0,
}

_SUPPLY_RISK_ZERO = {
    "supplier_delisted": 0,
    "timeout_risk": 0,
    "price_loss_risk": 0,
}

_NEW_SUPPLY_RISK_ZERO = {
    "supplier_delist": 0,
    "price_change": 0,
    "timeout_risk": 0,
}


def is_event_visible_to_agent(event: dict, agent_id: str) -> bool:
    """Global supplier events are visible to every agent listing that product.
    Agent-specific order events are visible only to the affected agent."""
    return event.get("agent_id") in (None, "", agent_id)


def _new_tick(env: Environment) -> dict:
    tick = sim_time.time_view(env.scenario, env.t, t._step_hours(env))
    tick["step"] = int(env.t)
    return tick


def _new_cash(env: Environment, agent_id: str) -> dict:
    st = env.agents.get(agent_id)
    full = t._cash_to_agent_dict(st.cash) if st is not None else {}
    net_assets = (
        float(full.get("balance", 0.0))
        + float(full.get("deposit_pool", 0.0))
        + float(full.get("in_transit", 0.0))
        + float(full.get("receivable", 0.0))
    )
    return {
        "balance": full.get("balance", 0.0),
        "deposit_pool": full.get("deposit_pool", 0.0),
        "in_transit": full.get("in_transit", 0.0),
        "receivable": full.get("receivable", 0.0),
        "net_assets": round(net_assets, 2),
        "cumulative_fine": full.get("cumulative_fine", 0.0),
    }


def _shop_rating_for(env: Environment, agent_id: str) -> Optional[dict]:
    """Build the agent-visible shop_rating block from the scenario cfg +
    AgentState counters. Returns None when rating is disabled."""
    cfg = env.scenario.get("shop_rating") or {}
    if not cfg.get("enabled", False):
        return None
    st = env.agents.get(agent_id)
    if st is None:
        return None
    rating_state = env._shop_rating_state(st)
    score = rating_state["score"]
    stars = rating_state["stars"]
    out = {
        "model": env._rating_model(),
        "score": round(float(score), 4) if score is not None else None,
        "stars": int(stars) if stars is not None else None,
        "quality_multiplier": round(rating_state["quality_multiplier"], 4),
        "reputation_multiplier": round(
            rating_state["reputation_multiplier"], 4,
        ),
        "demand_multiplier": round(rating_state["demand_multiplier"], 4),
        "rating_available": bool(rating_state["rating_available"]),
        "demand_source": str(rating_state["demand_source"]),
    }
    if env._uses_order_outcome_rating():
        out["rated_order_count"] = st.shop_rating_order_count
        out["qualified_transaction_count"] = st.shop_rating_order_count
        out["reputation_evidence_count"] = (
            st.public_review_count
            if env._uses_public_review_demand()
            else st.shop_rating_order_count
        )
        out["updated_through_step"] = env._shop_rating_updated_through_step(st)
    if env._uses_public_review_demand():
        out.update({
            "service_quality_score": round(
                float(rating_state["service_quality_score"]), 4,
            ),
            "service_quality_stars": int(
                rating_state["service_quality_stars"]
            ),
            "service_quality_multiplier": round(
                float(rating_state["service_quality_multiplier"]), 4,
            ),
        })
    public_reviews = env._public_review_state(st)
    if public_reviews is not None and env._public_reviews_agent_visible():
        public_payload = {
            "model": public_reviews["model"],
            "rating": (
                round(float(public_reviews["rating"]), 4)
                if public_reviews["rating"] is not None else None
            ),
            "count": int(public_reviews["count"]),
            "eligible_count": int(public_reviews["eligible_count"]),
            "response_rate": round(float(public_reviews["response_rate"]), 4),
            "full_response_rating": (
                round(float(public_reviews["full_response_rating"]), 4)
                if public_reviews["full_response_rating"] is not None else None
            ),
            "selection_gap": (
                round(float(public_reviews["selection_gap"]), 4)
                if public_reviews["selection_gap"] is not None else None
            ),
            "quality_gap": (
                round(float(public_reviews["quality_gap"]), 4)
                if public_reviews["quality_gap"] is not None else None
            ),
            "affects_demand": bool(public_reviews["affects_demand"]),
        }
        for key in (
            "stars",
            "confidence",
            "raw_quality_multiplier",
            "quality_multiplier",
            "reputation_multiplier",
            "demand_multiplier",
        ):
            value = public_reviews.get(key)
            if value is not None:
                public_payload[key] = round(float(value), 4)
        out["public_reviews"] = public_payload
    return out


def _observation_cache(env: Environment) -> dict:
    cache = getattr(env, "observation_cache_by_agent_step", None)
    if cache is None:
        cache = {}
        setattr(env, "observation_cache_by_agent_step", cache)
    return cache


def _observation_windows(env: Environment) -> dict:
    windows = getattr(env, "observation_window_by_agent_step", None)
    if windows is None:
        windows = {}
        setattr(env, "observation_window_by_agent_step", windows)
    return windows


def _last_observation_steps(env: Environment) -> dict:
    steps = getattr(env, "last_observation_step_by_agent", None)
    if steps is None:
        steps = {}
        setattr(env, "last_observation_step_by_agent", steps)
    return steps


def observation_change_window(env: Environment, agent_id: str) -> Optional[tuple[int, int]]:
    """Return the inclusive raw-step window for 'since last observation'.

    Every environment transition through env.t is committed before the hook
    opens.  The previous observation therefore already included its own step;
    subsequent windows start at last_observed_step + 1 and include env.t.
    A first observation may summarize all completed changes from step zero.
    """
    last_step = _last_observation_steps(env).get(agent_id)
    t_from = 0 if last_step is None else int(last_step) + 1
    t_to = int(env.t)
    if t_from > t_to:
        return None
    return t_from, t_to


def current_or_cached_change_window(env: Environment, agent_id: str) -> Optional[tuple[int, int]]:
    key = (agent_id, int(env.t))
    windows = _observation_windows(env)
    if key in windows:
        return windows[key]
    return observation_change_window(env, agent_id)


def _load_change_events(env: Environment, window: Optional[tuple[int, int]]) -> list[dict]:
    if window is None:
        return []
    t_from, t_to = window
    event_types = sorted(ABNORMAL_LISTING_EVENT_TYPES)
    return dbm.load_events_range_by_types(
        env.conn,
        env.run_id,
        int(t_from),
        int(t_to),
        event_types,
        limit=100000,
    )


def _current_listings(env: Environment, agent_id: str) -> list:
    return dbm.list_listings(env.conn, env.run_id, agent_id)


def _listing_promise(env: Environment, listing, product) -> int:
    rules = env.scenario["platform_rules"]
    default_h = int(rules.get("default_promised_ship_hours", 48))
    return default_h


def _has_timeout_risk(env: Environment, listing, product) -> bool:
    if product is None:
        return False
    promised = _listing_promise(env, listing, product)
    return int(product.supplier_ship_hours) > promised


def listing_info(env: Environment, listing) -> dict:
    product = env.products.get(listing.product_id)
    return {
        "product_id": listing.product_id,
        "name": product.name if product else "",
        "sale_price": round(float(listing.sale_price), 2),
        "supplier_price": round(float(product.price), 2) if product else None,
        "supplier_ship_hours": product.supplier_ship_hours if product else None,
        "supplier_logistics_hours": product.logistics_hours if product else None,
        "supplier_listed": bool(product.is_listed_by_supplier) if product else False,
    }


def _event_product_id(event: dict) -> str:
    payload = event.get("payload") or {}
    return str(payload.get("product_id") or event.get("entity_id") or "")


def _supply_event_record(env: Environment, event: dict) -> dict:
    product_id = _event_product_id(event)
    product = env.products.get(product_id)
    payload = event.get("payload") or {}
    event_type = event["event_type"]
    before = {}
    after = {}
    if event_type == "price_change":
        old_price = payload.get("old_price", payload.get("base_price"))
        new_price = payload.get("new_price")
        if old_price is not None:
            before["supplier_price"] = round(float(old_price), 2)
        if new_price is not None:
            after["supplier_price"] = round(float(new_price), 2)
    elif event_type == "supplier_delist":
        before["supplier_listed"] = True
        after["supplier_listed"] = False
    elif event_type == "supplier_timeout":
        old_ship = payload.get("before_supplier_ship_hours")
        new_ship = payload.get("after_supplier_ship_hours")
        if old_ship is None and product is not None:
            old_ship = product.base_ship_hours
        if new_ship is None and product is not None:
            new_ship = product.supplier_ship_hours
        if old_ship is not None:
            before["supplier_ship_hours"] = int(old_ship)
        if new_ship is not None:
            after["supplier_ship_hours"] = int(new_ship)
    elif event_type == "order_stockout_violation":
        reason = payload.get("reason")
        if reason is not None:
            before["stockout"] = False
            after["stockout"] = True
    return {
        "product_id": product_id,
        "name": product.name if product else "",
        "event_type": event_type,
        "before": before,
        "after": after,
    }


def supply_chain_anomalies(env: Environment, agent_id: str,
                           mode: str = "new") -> dict:
    mode = str(mode or "new")
    if mode not in ("new", "now"):
        return {"ok": False, "error": "mode must be 'new' or 'now'"}

    listings = _current_listings(env, agent_id)
    listing_by_pid = {l.product_id: l for l in listings}
    my_pids = set(listing_by_pid)
    if mode == "new":
        window = current_or_cached_change_window(env, agent_id)
        events = []
        affected: set[str] = set()
        for event in _load_change_events(env, window):
            if event["event_type"] not in ABNORMAL_LISTING_EVENT_TYPES:
                continue
            if not is_event_visible_to_agent(event, agent_id):
                continue
            product_id = _event_product_id(event)
            if product_id not in my_pids:
                continue
            events.append(_supply_event_record(env, event))
            affected.add(product_id)
        return {
            "mode": mode,
            "events": list(reversed(events)),
            "listings": compact_table([
                listing_info(env, listing_by_pid[pid])
                for pid in sorted(affected)
                if pid in listing_by_pid
            ], ANOMALY_LISTING_COLUMNS),
        }

    abnormal = []
    for listing in listings:
        product = env.products.get(listing.product_id)
        if product is None:
            continue
        if (
            not product.is_listed_by_supplier
            or float(product.price) > float(listing.sale_price)
            or _has_timeout_risk(env, listing, product)
        ):
            abnormal.append(listing_info(env, listing))
    return {
        "mode": mode,
        "events": [],
        "listings": compact_table(abnormal, ANOMALY_LISTING_COLUMNS),
    }


def _supply_event_counts(env: Environment, agent_id: str, events: list[dict],
                         listings: list) -> dict:
    my_pids = {l.product_id for l in listings}
    counts = dict(_SUPPLY_EVENT_ZERO)
    if not my_pids:
        return counts
    for event in events:
        bucket = _MY_LISTING_EVENT_BUCKETS.get(event["event_type"])
        if bucket is None:
            continue
        if not is_event_visible_to_agent(event, agent_id):
            continue
        if _event_product_id(event) not in my_pids:
            continue
        counts[bucket] += 1
    return counts


def _new_supply_risk_counts(env: Environment, agent_id: str, events: list[dict],
                            listings: list) -> dict:
    listing_by_pid = {l.product_id: l for l in listings}
    counts = dict(_NEW_SUPPLY_RISK_ZERO)
    for event in events:
        if not is_event_visible_to_agent(event, agent_id):
            continue
        product_id = _event_product_id(event)
        listing = listing_by_pid.get(product_id)
        if listing is None:
            continue
        if event["event_type"] == "supplier_delist":
            counts["supplier_delist"] += 1
        elif event["event_type"] == "price_change":
            counts["price_change"] += 1
        elif event["event_type"] == "supplier_timeout":
            product = env.products.get(product_id)
            if _has_timeout_risk(env, listing, product):
                counts["timeout_risk"] += 1
    return counts


def _current_supply_risks(env: Environment, listings: list) -> dict:
    counts = dict(_SUPPLY_RISK_ZERO)
    for listing in listings:
        product = env.products.get(listing.product_id)
        if product is None:
            continue
        if not product.is_listed_by_supplier:
            counts["supplier_delisted"] += 1
        if float(product.price) > float(listing.sale_price):
            counts["price_loss_risk"] += 1
        if _has_timeout_risk(env, listing, product):
            counts["timeout_risk"] += 1
    return counts


def _order_changes(env: Environment, agent_id: str,
                   window: Optional[tuple[int, int]]) -> dict:
    counts = _zero_order_status_counts()
    if window is None:
        return counts
    rows = env.conn.execute(
        "SELECT os.status, COUNT(*) AS n"
        " FROM order_status os"
        " JOIN orders o ON o.run_id=os.run_id AND o.order_id=os.order_id"
        " WHERE os.run_id=? AND o.agent_id=? AND os.t>=? AND os.t<=?"
        " GROUP BY os.status",
        (env.run_id, agent_id, int(window[0]), int(window[1])),
    ).fetchall()
    for row in rows:
        status = row["status"]
        if status not in counts:
            continue
        n = int(row["n"] or 0)
        counts[status] = n
        counts["total"] += n
    return counts


def _order_totals(env: Environment, agent_id: str) -> dict:
    rows = env.conn.execute(
        "SELECT current_status, COUNT(*) AS n"
        " FROM orders WHERE run_id=? AND agent_id=?"
        " GROUP BY current_status",
        (env.run_id, agent_id),
    ).fetchall()
    counts = {status: 0 for status in _ORDER_STATUS_KEYS}
    total = 0
    for row in rows:
        status = row["current_status"]
        n = int(row["n"] or 0)
        total += n
        if status in counts:
            counts[status] = n
    return {"total": total, **counts}


def build_store_snapshot(env: Environment, agent_id: str,
                         window: Optional[tuple[int, int]] = None) -> dict:
    if window is None:
        window = current_or_cached_change_window(env, agent_id)
    events = _load_change_events(env, window)
    listings = _current_listings(env, agent_id)
    rules = env.scenario["platform_rules"]
    max_active = int(rules.get("max_active_listings", 100))
    active_count = len(listings)
    rating = _shop_rating_for(env, agent_id)
    return {
        "agent_id": agent_id,
        "tick": _new_tick(env),
        "orders": {
            "changes_since_last_observation": _order_changes(env, agent_id, window),
            "totals": _order_totals(env, agent_id),
        },
        "supply": {
            "listings": {
                "active": active_count,
                "max": max_active,
                "free_slots": max(0, max_active - active_count),
            },
            "events_since_last_observation": _supply_event_counts(
                env, agent_id, events, listings),
            "current_risks": _current_supply_risks(env, listings),
            "new_risks_since_last_observation": _new_supply_risk_counts(
                env, agent_id, events, listings),
        },
        "cash": _new_cash(env, agent_id),
        "shop": ({
            "rating": (
                f"{rating['stars']}★"
                if rating["stars"] is not None else None
            ),
            **rating,
        } if rating else {}),
        "daily_report_available": (
            _tool_available(env, "get_daily_report")
            and t.daily_report_notice_available(env, agent_id)
        ),
    }


def render_observation_text(obs: dict) -> str:
    """Convert a structured observation packet into a human-readable string.

    Used both for injecting a ``role: user`` message into the stored trace
    and by agents that want a pre-rendered text representation.
    """
    tk = obs["tick"]
    cash = obs["cash"]
    orders = obs["orders"]
    supply = obs["supply"]
    head = _format_tick_label(tk)
    changes = orders["changes_since_last_observation"]
    totals = orders["totals"]
    change_status_parts = [f"{status} {changes.get(status, 0)}"
                           for status in _ORDER_STATUS_KEYS]
    total_status_parts = [f"{status} {totals.get(status, 0)}"
                          for status in _ORDER_STATUS_KEYS]
    listing = supply["listings"]
    events = supply["events_since_last_observation"]
    current_risks = supply["current_risks"]
    new_risks = supply["new_risks_since_last_observation"]
    shop = obs.get("shop") or {}
    if shop.get("demand_source") == "public_reviews":
        rating = (
            "Internal service quality (no direct v4 demand effect): "
            f"score {float(shop['service_quality_score']):.2f} / "
            f"stars {int(shop['service_quality_stars'])}★"
        )
    elif shop.get("score") is not None and shop.get("stars") is not None:
        rating = f"score {float(shop['score']):.2f} / stars {int(shop['stars'])}★"
    else:
        rating = "n/a"
    shop_lines = ["Shop:", rating]
    public_reviews = shop.get("public_reviews")
    if public_reviews:
        def _review_value(key: str, *, percent: bool = False,
                          stars: bool = False, signed: bool = False) -> str:
            value = public_reviews.get(key)
            if value is None:
                return "n/a"
            number = float(value)
            if percent:
                return f"{number * 100:.2f}%"
            suffix = "★" if stars else ""
            prefix = "+" if signed and number > 0 else ""
            return f"{prefix}{number:.2f}{suffix}"

        public_label = "Public reviews (drives demand)"
        demand_details = ""
        if public_reviews.get("affects_demand"):
            demand_details = (
                f" / confidence {_review_value('confidence', percent=True)}"
                f" / adjusted_rating_effect {_review_value('quality_multiplier')}×"
                f" / review_volume_trust {_review_value('reputation_multiplier')}×"
                f" / demand {_review_value('demand_multiplier')}×"
            )
        shop_lines.append(
            f"{public_label}: "
            f"rating {_review_value('rating', stars=True)} / "
            f"count {int(public_reviews.get('count', 0))} / "
            f"eligible {int(public_reviews.get('eligible_count', 0))} / "
            f"response_rate {_review_value('response_rate', percent=True)}"
            f"{demand_details} / "
            "full_response_rating "
            f"{_review_value('full_response_rating', stars=True)} / "
            f"selection_gap {_review_value('selection_gap', signed=True)} / "
            f"recent_quality_gap {_review_value('quality_gap', signed=True)}"
        )
    header_lines = [head]
    if obs.get("daily_report_available"):
        header_lines.append(
            "Daily report available: use get_daily_report for today's published market brief "
            "(data through yesterday)."
        )
    sections = [
        "\n".join(header_lines),
        "\n".join([
            "Orders:",
            f"changes since last observation: total {changes['total']} / "
            + " / ".join(change_status_parts),
            f"totals: total {totals['total']} / "
            + " / ".join(total_status_parts),
        ]),
        "\n".join([
            "Supply & listings:",
            f"Shelf utilization: active {listing['active']} / max {listing['max']} / "
            f"free {listing['free_slots']}",
            # "Empty shelf slots reduce product exposure.",
            "",
            "events since last observation: "
            f"price_changes {events['price_changes']} / "
            f"supplier_delists {events['supplier_delists']} / "
            f"timeouts {events['timeouts']} / stockouts {events['stockouts']}",
            "current risks: "
            f"supplier_delisted {current_risks['supplier_delisted']} / "
            f"timeout_risk {current_risks['timeout_risk']} / "
            f"price_loss_risk {current_risks['price_loss_risk']}",
            "new risks since last observation: "
            f"supplier_delist {new_risks['supplier_delist']} / "
            f"price_change {new_risks['price_change']} / "
            f"timeout_risk {new_risks['timeout_risk']}",
        ]),
        "\n".join([
            "Cash:",
            f"balance {cash['balance']:.2f} / "
            f"deposit_pool {cash['deposit_pool']:.2f} / "
            f"in_transit {cash['in_transit']:.2f} / "
            f"receivable {cash['receivable']:.2f} / "
            f"net_assets {cash['net_assets']:.2f} / "
            f"cumulative_fine {cash['cumulative_fine']:.2f}",
        ]),
        "\n".join(shop_lines),
    ]
    sections.append("Continue operating the store. Goal: maximize net_assets.")
    return "\n\n".join(sections)


def _format_tick_label(tk: dict) -> str:
    head = f"Day {tk['day']}, Hour {tk['hour']}"
    if tk.get("datetime"):
        return f"{head} ({tk['datetime']})"
    return head


def compose_observation(env: Environment, agent_id: str,
                        *, include_brief: bool = False,
                        mark_observed: bool = False,
                        prefer_cached: bool = False) -> dict:
    """Slim per-step observation packet. Same shape every time; no per-section
    toggles. Future-state (horizon, run-length) and product-level detail are
    deliberately omitted.

    Fields:
      tick.step                      - raw integer simulator tick
      tick.day, tick.hour            - 1-indexed day, 0-23 hour
      cash                           - full cash and net_assets fields
      orders                         - changes since last observation + totals
      supply                         - listing capacity, supply events, risks
      shop                           - rating fields when enabled
      daily_report_available         - true when today's report exists and this
                                       agent has not successfully read it
      brief (when include_brief=True) - system prompt + platform rules
      text                           - pre-rendered human-readable string
    """
    key = (agent_id, int(env.t))
    cache = _observation_cache(env)
    if (prefer_cached or mark_observed) and key in cache:
        out = copy.deepcopy(cache[key])
    else:
        window = current_or_cached_change_window(env, agent_id)
        out = build_store_snapshot(env, agent_id, window=window)
        out["text"] = render_observation_text(out)
        if mark_observed:
            # Multiple agents may long-poll the same hook concurrently. Keep
            # their cursor/window updates and the shared state-file snapshot
            # in one protocol critical section so a later stale write cannot
            # discard another agent's just-served observation.
            with env.turn_lock:
                cache[key] = copy.deepcopy(out)
                windows = _observation_windows(env)
                windows[key] = window
                steps = _last_observation_steps(env)
                steps[agent_id] = int(env.t)
                agent_log.persist_observation_state(
                    env.runs_root,
                    env.run_id,
                    steps,
                    windows_by_agent_step=windows,
                    daily_report_read_dates_by_agent=(
                        env.daily_report_read_date_by_agent
                    ),
                )
    if include_brief:
        out["brief"] = compose_system_brief(env)
    elif "brief" in out:
        out.pop("brief")
    return out


def get_observation_tool(env: Environment, agent_id: str) -> str:
    """Tool-facing wrapper: returns only the pre-rendered text string."""
    obs = compose_observation(env, agent_id, prefer_cached=True)
    return obs["text"]


def list_tools(env: Environment, agent_id: str) -> list[dict]:
    """Agent self-discovery: return OpenAI tool-calling schemas for every
    tool this agent may call. Honors agent.tool_denylist if set."""
    deny = _denylist(env)
    return registry.openai_schema_dump(deny, env=env)


# Register the aggregated tool in the registry so dashboards / agents that
# poll /tools/schema see it alongside the primitives.
registry.append(registry.ToolSpec(
    name="get_observation",
    description=("Return the current English observation text with order changes "
                  "and current_status totals, Supply & listings, Cash, Shop, "
                  "an unread daily-report availability notice, and the net_assets "
                  "objective. Same text as "
                  "GET /agents/<aid>/observation."),
    parameters={"type": "object", "properties": {}, "required": [],
                "additionalProperties": False},
    examples=[{}],
    method="GET", path_suffix="observation",
    handler=get_observation_tool, mutating=False,
))

registry.append(registry.ToolSpec(
    name="list_tools",
    description=("Self-discovery: returns the OpenAI tool-calling schema list"
                  " for every tool this agent may call."),
    parameters={"type": "object", "properties": {}, "required": [],
                "additionalProperties": False},
    examples=[{}],
    method="GET", path_suffix="list_tools",
    handler=list_tools, mutating=False,
))
