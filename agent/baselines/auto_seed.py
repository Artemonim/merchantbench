"""Legacy auto-seed entry point and shared rule-based baseline implementation.

New runs should use ``rule_based.py`` with either ``daily_report`` or ``random``
selection.  This module keeps the historical ``auto_seed.py`` CLI and
``AutoSeedAgent`` class working as the daily-report compatibility mode.
If cash falls below the safety line, it delists the full shelf.

Run locally:
  cd env && python run.py --port 5050 &
  .venv/bin/python agent/baselines/auto_seed.py --run-id <rid> --base-url http://localhost:5050
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
import uuid
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("auto_seed requires 'requests': "
          ".venv/bin/python -m pip install -r agent/requirements.txt",
          file=sys.stderr)
    raise

_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from sdk.merchantbench_tool_client import MerchantBenchToolClient


VERSION = "3.0"
DEFAULT_MARKUP = 2.00
DEFAULT_CASH_LOW_WATERMARK = 500.0
DEFAULT_PRICE_MAX = 20.0
DEFAULT_SUPPLIER_RATING_MIN = 4.5
DEFAULT_HISTORICAL_RATING_MIN = 4.5
DEFAULT_SUPPLIER_SHIP_HOURS_MAX = 48
DEFAULT_QUANTITY_MIN = 20
MAX_REPORT_QUERIES = 12
SELECTION_MODES = ("daily_report", "random")
RANDOM_SEARCH_PAGES = 12
RANDOM_PAGE_UPPER_BOUND = 2_000
RANDOM_FALLBACK_PAGES = 3


def _table_records(value):
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        columns = value["columns"]
        records = []
        for row in value["rows"]:
            if len(row) != len(columns):
                print(
                    f"[auto_seed] warning: row has {len(row)} values but "
                    f"schema has {len(columns)} columns, padding/truncating",
                    file=sys.stderr,
                )
            records.append(
                dict(itertools.zip_longest(columns, row, fillvalue=None))
            )
        return records
    return value


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AutoSeedAgent:
    def __init__(self, base_url: str, run_id: str, agent_id: str,
                 seed_count: int = 50,
                 markup: float = DEFAULT_MARKUP,
                 cash_low_watermark: float = DEFAULT_CASH_LOW_WATERMARK,
                 timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.run_id = run_id
        self.agent_id = agent_id
        self.seed_count = seed_count
        self.markup = markup
        self.cash_low_watermark = cash_low_watermark
        self.client = MerchantBenchToolClient(base_url, run_id, agent_id,
                                         timeout=timeout)
        self.system_prompt: Optional[str] = None
        self.language: str = "en"
        self._seeded = False
        self._last_refresh_day: Optional[int] = None
        self._listed_product_ids: set[str] = set()
        self.selection_mode = "daily_report"
        self.selection_seed = 0
        self.framework_name = "auto_seed"

    def _t(self, zh: str, en: str) -> str:
        return zh if self.language == "zh" else en

    def _act(self, thought: str, tool_calls_spec: list[tuple[str, dict]]) -> list[dict]:
        """Build assistant message with tool_calls and send to /act.

        Returns list of parsed tool result dicts.
        """
        tc_list = []
        for i, (name, args) in enumerate(tool_calls_spec):
            tc_list.append({
                "id": f"call_{uuid.uuid4().hex[:8]}_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
        assistant_msg = {
            "role": "assistant",
            "content": thought,
            "tool_calls": tc_list,
        }
        input_tokens = _est_tokens(thought)
        output_tokens = sum(_est_tokens(tc["function"]["arguments"]) for tc in tc_list)
        token_usage = {"input": input_tokens, "output": output_tokens,
                       "cache_read": 0, "total": input_tokens + output_tokens}

        resp = self.client.act(assistant_msg, token_usage=token_usage)
        results = []
        for tr in resp.get("tool_results", []):
            try:
                results.append(json.loads(tr["content"]))
            except (json.JSONDecodeError, KeyError):
                results.append(tr.get("content", {}))
        return results

    def register(self) -> None:
        self.client.register(
            framework=getattr(self, "framework_name", "auto_seed"),
            model=(
                f"{getattr(self, 'selection_mode', 'daily_report')}"
                f"_markup_{self.markup:.2f}"
            ),
            version=VERSION,
            extra={
                "seed_count": self.seed_count,
                "markup": self.markup,
                "cash_low_watermark": self.cash_low_watermark,
                "selection_mode": getattr(
                    self, "selection_mode", "daily_report"
                ),
                "selection_seed": getattr(self, "selection_seed", 0),
                # This deterministic baseline has no upstream model API, so
                # zero provider failures/retry exhaustion is complete data.
                "runtime_health_version": 1,
                "runtime_health_capabilities": {
                    "provider_api_failed_attempts": "not_applicable",
                    "retry_exhausted": "not_applicable",
                    "memory_compactions": "not_applicable",
                    "skills_evolutions": "not_applicable",
                },
            },
        )

    def _update_brief_from_obs(self, obs: dict) -> None:
        """Extract brief from observation packet (included on first observation)."""
        brief = obs.get("brief")
        if brief and self.system_prompt is None:
            self.system_prompt = brief.get("system_prompt", "") or ""
            self.language = brief.get("language", "en")

    def _drive_step(self, obs: dict, verbose: bool) -> None:
        tick = obs.get("tick") or {}
        day = int(tick.get("day") or 0)
        selection_mode = getattr(self, "selection_mode", "daily_report")
        balance = (
            None
            if selection_mode == "random"
            else self._query_balance()
        )
        current = self._query_current_listings()
        step_product_ids = {
            str(row["product_id"]) for row in current if row.get("product_id")
        }
        self._listed_product_ids.update(
            step_product_ids
        )

        if current and balance is not None and balance < self.cash_low_watermark:
            self._delist_products(
                [str(row["product_id"]) for row in current if row.get("product_id")],
                reason=self._t("[CASH] 现金低于安全线,下架全部商品。",
                               "[CASH] Balance below safety line — delist all products."),
            )
            self._seeded = False
            self._finish_step(obs, verbose)
            return

        current = self._handle_supply_risks(current)

        daily_refresh = bool(day and self._last_refresh_day != day)
        if (
            daily_refresh
            and current
            and selection_mode != "random"
        ):
            stale_ids = self._stale_listing_ids()
            if stale_ids:
                self._delist_products(
                    stale_ids,
                    reason=self._t("[STALE] 下架 7 天无销量的滞销品。",
                                   "[STALE] Delist listings with zero sales after 7 days."),
                )
                stale_set = set(stale_ids)
                current = [
                    row for row in current
                    if str(row.get("product_id") or "") not in stale_set
                ]

        should_seed = len(current) < self.seed_count
        if should_seed:
            self._seed_listings(
                # Do not immediately re-list a product removed for a current
                # supply risk (or by the daily-report compatibility mode).
                current_product_ids=step_product_ids,
                target_count=max(0, self.seed_count - len(current)),
                selection_day=day,
            )
            self._seeded = True

        if daily_refresh:
            self._last_refresh_day = day

        self._finish_step(obs, verbose)

    def _finish_step(self, obs: dict, verbose: bool) -> None:
        self._act(
            self._t("[EOS] 本步动作结束,调用 end_of_step 释放 hook。",
                     "[EOS] Done — release the per-step hook."),
            [("end_of_step", {})],
        )

        if verbose:
            tk = obs["tick"]
            print(f"[Day {tk['day']} Hour {tk['hour']}] done")

    def _first_result(self, thought: str, tool_name: str, args: dict) -> Any:
        results = self._act(thought, [(tool_name, args)])
        return results[0] if results else {}

    def _query_balance(self) -> Optional[float]:
        result = self._first_result(
            self._t("[Q] 查看现金余额。", "[Q] Check cash balance."),
            "query_balance",
            {},
        )
        try:
            return float(result.get("balance"))
        except (AttributeError, TypeError, ValueError):
            return None

    def _query_current_listings(self) -> list[dict]:
        result = self._first_result(
            self._t("[Q] 查看当前货架。", "[Q] Check current shelf."),
            "query_my_listings",
            {},
        )
        return _table_records(result) if result else []

    def _delist_products(self, product_ids: list[str], reason: str) -> None:
        unique_ids = []
        seen = set()
        for pid in product_ids:
            if not pid or pid in seen:
                continue
            unique_ids.append(pid)
            seen.add(pid)
        if not unique_ids:
            return
        self._act(
            reason,
            [("delist_product", {
                "items": [{"product_id": pid} for pid in unique_ids],
            })],
        )

    def _handle_supply_risks(self, current: list[dict]) -> list[dict]:
        result = self._first_result(
            self._t("[RISK] 检查当前货架的供应链风险。",
                    "[RISK] Check current supply-chain risks for the shelf."),
            "query_supply_chain_anomalies",
            {"mode": "now"},
        )
        risk_rows = (
            _table_records(result.get("listings", []))
            if isinstance(result, dict)
            else []
        )
        if not isinstance(risk_rows, list):
            risk_rows = []
        delist_ids: set[str] = set()
        for row in risk_rows:
            pid = str(row.get("product_id") or "")
            ship_hours = _float_or_none(row.get("supplier_ship_hours"))
            if pid and (
                row.get("supplier_listed") is False
                or (
                    ship_hours is not None
                    and ship_hours > DEFAULT_SUPPLIER_SHIP_HOURS_MAX
                )
            ):
                delist_ids.add(pid)

        current_ids = {
            str(row["product_id"]) for row in current if row.get("product_id")
        }
        delist_ids &= current_ids
        if delist_ids:
            self._delist_products(
                sorted(delist_ids),
                reason=self._t(
                    "[RISK] 供应商已下架或当前发货超时,下架对应商品。",
                    "[RISK] Supplier unavailable or currently timed out — delist.",
                ),
            )

        remaining = [
            row for row in current
            if str(row.get("product_id") or "") not in delist_ids
        ]
        price_updates: list[dict] = []
        for row in remaining:
            pid = str(row.get("product_id") or "")
            supplier_price = _float_or_none(row.get("supplier_price"))
            sale_price = _float_or_none(row.get("sale_price"))
            if not pid or supplier_price is None or supplier_price <= 0:
                continue
            new_price = self._sale_price(supplier_price)
            if sale_price is not None and round(sale_price, 2) == new_price:
                continue
            price_updates.append({
                "product_id": pid,
                "new_price": new_price,
            })
        if price_updates:
            self._act(
                self._t(
                    "[PRICE] 供应商价格变化,售价调整为当前成本的 2 倍。",
                    "[PRICE] Supplier price changed — set sale price to 2x current cost.",
                ),
                [("adjust_price", {"items": price_updates})],
            )
        return remaining

    def _stale_listing_ids(self) -> list[str]:
        result = self._first_result(
            self._t("[REVIEW] 检查 7 天滞销商品。",
                     "[REVIEW] Check stale listings over the last 7 days."),
            "review_my_listings",
            {"sort_by": "days_without_sales", "window_days": 7},
        )
        rows = _table_records(result) if result else []
        stale_ids = []
        for row in rows:
            try:
                days_without_sales = int(row.get("days_without_sales") or 0)
            except (TypeError, ValueError):
                continue
            product_id = row.get("product_id")
            if product_id and days_without_sales >= 7:
                stale_ids.append(str(product_id))
        return stale_ids

    def _seed_listings(self, current_product_ids: Optional[set[str]] = None,
                       target_count: Optional[int] = None,
                       selection_day: int = 0) -> None:
        current_product_ids = current_product_ids or set()
        target = int(target_count if target_count is not None else self.seed_count)
        if target <= 0:
            return
        if getattr(self, "selection_mode", "daily_report") == "random":
            picks = self._random_product_picks(
                current_product_ids,
                target,
                selection_day=selection_day,
            )
            self._list_picks(picks, target, "randomly selected")
            return

        report = self._first_result(
            self._t("[REPORT] 阅读当日商机日报。",
                     "[REPORT] Read today's opportunity report."),
            "get_daily_report",
            {},
        )
        queries = _extract_report_queries(
            report.get("content", "") if isinstance(report, dict) and report.get("ok", True) else ""
        )
        if not queries:
            queries = [""]
        picks: list[dict] = []
        picked_ids = set(current_product_ids)
        for query in queries:
            result = self._first_result(
                self._t(f"[SEARCH] 搜索日报关键词: {query or 'rating'}。",
                         f"[SEARCH] Search report keyword: {query or 'rating'}."),
                "search_products",
                {
                    "query": query,
                    "page": 1,
                    "page_size": 20,
                    "price_max": DEFAULT_PRICE_MAX,
                    "supplier_rating_min": DEFAULT_SUPPLIER_RATING_MIN,
                    "historical_rating_min": DEFAULT_HISTORICAL_RATING_MIN,
                    "supplier_ship_hours_max": DEFAULT_SUPPLIER_SHIP_HOURS_MAX,
                    "quantity_min": DEFAULT_QUANTITY_MIN,
                    "sort_by": "rating" if not query else "relevance",
                },
            )
            if isinstance(result, dict):
                candidates = _table_records(result.get("items", []))
            else:
                candidates = []
            for product in candidates:
                if not _is_viable_product(product):
                    continue
                pid = str(product.get("product_id") or "")
                if not pid or pid in picked_ids or pid in self._listed_product_ids:
                    continue
                picks.append(product)
                picked_ids.add(pid)
                if len(picks) >= target:
                    break
            if len(picks) >= target:
                break

        # If the report terms are too narrow, fall back to highly rated products
        # under the same public-field filters.
        page = 1
        while len(picks) < target and page <= 3:
            result = self._first_result(
                self._t("[SEARCH] 日报关键词候选不足,补充浏览评分商品。",
                         "[SEARCH] Report candidates were insufficient — browse rated products."),
                "search_products",
                {
                    "query": "",
                    "page": page,
                    "page_size": 20,
                    "price_max": DEFAULT_PRICE_MAX,
                    "supplier_rating_min": DEFAULT_SUPPLIER_RATING_MIN,
                    "historical_rating_min": DEFAULT_HISTORICAL_RATING_MIN,
                    "supplier_ship_hours_max": DEFAULT_SUPPLIER_SHIP_HOURS_MAX,
                    "quantity_min": DEFAULT_QUANTITY_MIN,
                    "sort_by": "rating",
                },
            )
            candidates = _table_records(result.get("items", [])) if isinstance(result, dict) else []
            for product in candidates:
                if not _is_viable_product(product):
                    continue
                pid = str(product.get("product_id") or "")
                if not pid or pid in picked_ids or pid in self._listed_product_ids:
                    continue
                picks.append(product)
                picked_ids.add(pid)
                if len(picks) >= target:
                    break
            page += 1
        self._list_picks(picks, target, "report-driven")

    def _random_product_picks(
        self,
        current_product_ids: set[str],
        target: int,
        *,
        selection_day: int,
    ) -> list[dict]:
        """Sample public catalog candidates using a reproducible local RNG.

        Random mode deliberately omits report signals and all business-value
        filters. Any product returned by the public catalog can be sampled.
        """
        # Derive a fresh generator from stable run inputs.  A stale-step retry
        # on the same simulation day therefore selects the same pages/products
        # instead of consuming different state from a mutable RNG.
        seed = int(getattr(self, "selection_seed", 0))
        rng = random.Random(seed * 1_000_003 + int(selection_day))

        page_numbers = rng.sample(
            range(1, RANDOM_PAGE_UPPER_BOUND + 1),
            RANDOM_SEARCH_PAGES,
        )
        candidates_by_id: dict[str, dict] = {}

        def collect_page(page: int) -> None:
            result = self._first_result(
                self._t(
                    f"[RANDOM] 随机浏览商品目录第 {page} 页。",
                    f"[RANDOM] Browse random catalog page {page}.",
                ),
                "search_products",
                {
                    "query": "",
                    "page": page,
                    "page_size": 50,
                    "sort_by": "relevance",
                },
            )
            rows = (
                _table_records(result.get("items", []))
                if isinstance(result, dict)
                else []
            )
            for product in rows:
                pid = str(product.get("product_id") or "")
                if (
                    not pid
                    or pid in current_product_ids
                ):
                    continue
                candidates_by_id.setdefault(pid, product)

        for page in page_numbers:
            collect_page(page)

        # Small synthetic catalogs may not reach the randomly selected pages.
        # Sequential fallback pages keep the mode useful without introducing
        # rating or report-based ranking.
        if len(candidates_by_id) < target:
            for page in range(1, RANDOM_FALLBACK_PAGES + 1):
                collect_page(page)
                if len(candidates_by_id) >= target:
                    break

        candidates = list(candidates_by_id.values())
        rng.shuffle(candidates)
        return candidates[:target]

    def _list_picks(self, picks: list[dict], target: int, label: str) -> None:
        list_items: list[dict] = []
        for product in picks[:target]:
            pid = str(product.get("product_id") or "")
            price = _float_or_none(product.get("price"))
            if not pid or price is None or price <= 0:
                continue
            list_items.append({
                "product_id": pid,
                "sale_price": self._sale_price(price),
            })
        if not list_items:
            return
        self._act(
            self._t(
                f"[SEED] 上架 {len(list_items)} 个规则选品商品。",
                f"[SEED] List {len(list_items)} {label} products.",
            ),
            [("list_product", {"items": list_items})],
        )
        self._listed_product_ids.update(item["product_id"] for item in list_items)

    def _sale_price(self, supplier_price: float) -> float:
        return round(float(supplier_price) * self.markup, 2)

    def run(self, max_steps: int = 2200, verbose: bool = True) -> None:
        self.register()
        while True:
            try:
                obs = self.client.observation()
            except requests.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 410:
                    if verbose:
                        print("[done] run finished", file=sys.stderr)
                    return
                if verbose:
                    print(f"[obs error] {e}", file=sys.stderr)
                continue
            self._update_brief_from_obs(obs)
            tk = obs["tick"]
            t = (tk["day"] - 1) * 24 + tk["hour"]
            if t >= max_steps:
                if verbose:
                    print(f"[done] reached t={t}")
                return
            try:
                self._drive_step(obs, verbose)
            except requests.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 410:
                    if verbose:
                        print("[done] run finished", file=sys.stderr)
                    return
                if verbose:
                    print(f"[act error] {e}", file=sys.stderr)
                continue


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_viable_product(product: dict) -> bool:
    price = _float_or_none(product.get("price"))
    hist = _float_or_none(product.get("historical_avg_rating"))
    shop = _float_or_none(product.get("shop_rating"))
    ship = _float_or_none(product.get("supplier_ship_hours"))
    qty = _float_or_none(product.get("quantity"))
    if price is None or price <= 0 or price > DEFAULT_PRICE_MAX:
        return False
    if hist is not None and hist < DEFAULT_HISTORICAL_RATING_MIN:
        return False
    if shop is not None and shop < DEFAULT_SUPPLIER_RATING_MIN:
        return False
    if ship is not None and ship > DEFAULT_SUPPLIER_SHIP_HOURS_MAX:
        return False
    if qty is not None and qty < DEFAULT_QUANTITY_MIN:
        return False
    return True


class RuleBasedAgent(AutoSeedAgent):
    """Rule-based baseline with selectable sourcing policy."""

    def __init__(
        self,
        base_url: str,
        run_id: str,
        agent_id: str,
        seed_count: int = 50,
        markup: float = DEFAULT_MARKUP,
        cash_low_watermark: float = DEFAULT_CASH_LOW_WATERMARK,
        timeout: float = 600.0,
        selection_mode: str = "daily_report",
        selection_seed: int = 42,
    ):
        if selection_mode not in SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {SELECTION_MODES}, "
                f"got {selection_mode!r}"
            )
        super().__init__(
            base_url,
            run_id,
            agent_id,
            seed_count=seed_count,
            markup=markup,
            cash_low_watermark=cash_low_watermark,
            timeout=timeout,
        )
        self.selection_mode = selection_mode
        self.selection_seed = int(selection_seed)
        self.framework_name = "rule_based"


_REPORT_STOPWORDS = {
    "昨日要闻",
    "整体趋势",
    "品类动态",
    "异动信号词",
    "新闻关联提示",
    "信号解读",
}


def _clean_report_query(value: str) -> Optional[str]:
    text = re.sub(r"<[^>]+>", "", str(value or "")).strip()
    text = re.sub(r"[`*_#|▍•:：,，.。;；!！?？()（）\[\]【】]+", "", text)
    text = re.sub(r"^\d+\s*NEW\s*", "", text, flags=re.I).strip()
    if not text or text in _REPORT_STOPWORDS:
        return None
    if re.search(r"\d+\.?\d*\s*(万|%|点|元|天|日|月)", text):
        return None
    if len(text) < 2 or len(text) > 18:
        return None
    return text


def _extract_report_queries(content: str) -> list[str]:
    candidates: list[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in {"----", "排名"} and cells[1] != "关键词":
                candidates.append(cells[1])
        candidates.extend(re.findall(r"[\"“]([^\"”]{2,18})[\"”]", stripped))
        candidates.extend(re.findall(r"\*\*([^*]{2,18})\*\*", stripped))

    out = []
    seen = set()
    for value in candidates:
        query = _clean_report_query(value)
        if not query or query in seen:
            continue
        out.append(query)
        seen.add(query)
        if len(out) >= MAX_REPORT_QUERIES:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--agent-id", default="agent_0")
    ap.add_argument("--seed-count", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2200)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    AutoSeedAgent(args.base_url, args.run_id, args.agent_id,
                  seed_count=args.seed_count,
                  timeout=args.timeout).run(
        max_steps=args.max_steps, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
