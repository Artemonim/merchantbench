"""Demand-proxy hot-search phrase extraction and ranking.

The ranking intentionally uses only trailing product demand.  It does not
pretend that the simulator has search impressions, clicks, or conversions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import html
import logging
import math
import re
from typing import Iterable

import jieba


jieba.setLogLevel(logging.WARNING)

_TITLE_CHUNK_RE = re.compile(r"[a-z0-9_+\-\u4e00-\u9fff]+", re.IGNORECASE)
_VALID_QUERY_RE = re.compile(r"(?:[a-z][a-z0-9_+\-]*|[\u4e00-\u9fff])+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# These words describe supplier operations rather than shopper intent.  A
# query containing one of them is discarded completely.
_SUPPLY_NOISE = {
    "厂家", "批发", "厂家批发", "厂家直销", "一件代发", "现货", "新款",
    "跨境", "供应", "定制", "logo", "包邮", "促销", "热销", "爆款",
    "工厂", "直销", "货源", "亚马逊",
}

# These are useful attributes inside a complete query, but too broad to rank
# as a shopper query by themselves.
STANDALONE_NOISE_TERMS = frozenset({
    "大容量", "多功能", "高颜值", "便携式", "一次性", "创意", "迷你",
    "ins", "diy", "usb", "pvc", "卫生间", "办公室", "家用", "商用", "专用",
    "二合一", "大风力", "学生宿舍", "充电式",
    "通用型", "全自动", "成人款", "创意个性", "结婚专用", "装饰用品",
    "柔软吸水", "吸水洗脸", "家用客厅", "防虫防潮",
})

_ATTRIBUTE_TOKENS = STANDALONE_NOISE_TERMS | {
    "充电", "手持", "便携", "学生", "宿舍", "小", "大", "强力", "自动",
    "通用", "成人", "款", "个性", "柔软", "吸水", "洗脸", "客厅", "防虫", "防潮",
}


@lru_cache(maxsize=200_000)
def extract_query_phrases(name: str) -> frozenset[str]:
    """Extract deterministic, search-like phrases from one supplier title."""
    text = html.unescape(str(name or "")).lower()
    phrases: set[str] = set()
    for chunk in _TITLE_CHUNK_RE.findall(text):
        tokens = [str(token).strip().lower() for token in jieba.cut(chunk, cut_all=False)]
        tokens = [token for token in tokens if token]
        for start in range(len(tokens)):
            for width in (1, 2, 3):
                parts = tokens[start:start + width]
                if len(parts) != width:
                    continue
                phrase = "".join(parts).strip("_")
                if not 3 <= len(phrase) <= (8 if width == 1 else 10):
                    continue
                if not _VALID_QUERY_RE.fullmatch(phrase) or _NUMBER_RE.fullmatch(phrase):
                    continue
                if any(noise in phrase for noise in _SUPPLY_NOISE):
                    continue
                if phrase in STANDALONE_NOISE_TERMS:
                    continue
                if all(part in _ATTRIBUTE_TOKENS for part in parts):
                    continue
                phrases.add(phrase)
    return frozenset(phrases)


@dataclass(frozen=True)
class HotSearchTrend:
    rank: int
    keyword: str
    category: str
    trend: str
    change_pct: float | None
    rank_change: int | None


@dataclass(frozen=True)
class _Candidate:
    keyword: str
    category: str
    product_indexes: frozenset[int]
    current_score: float
    previous_score: float
    current_demand: float
    previous_demand: float


def _minimum_support(product_count: int) -> int:
    if product_count < 100:
        return 2
    return min(20, max(5, math.ceil(product_count * 0.002)))


def _rank_score(current: float, previous: float, support: int, days: int) -> float:
    if current <= 0.0 or support <= 0:
        return 0.0
    daily_strength = (current / days) / math.sqrt(support)
    confidence = math.sqrt(support / (support + 20.0))
    momentum = math.sqrt((current + 1.0) / (previous + 1.0))
    momentum = min(1.5, max(0.75, momentum))
    return math.log1p(daily_strength) * confidence * momentum


def _trend(current: float, previous: float, days: int) -> tuple[str, float | None]:
    # Compute change percentage first — always useful, even for "new" labels.
    if previous > 0.0:
        change = round((current / previous - 1.0) * 100.0, 1)
    else:
        change = None

    # "new" requires: previous daily rate below 1.0 AND current daily rate >= 1.0.
    # For low-demand emerging keywords (current < days), fall through to the
    # previous <= 0 guard which labels them "rising" with a None change_pct
    # instead of the misleading "stable, 0%".
    if previous / days < 1.0 <= current / days:
        return "new", change

    if previous <= 0.0:
        # Genuinely zero previous demand. If current has any demand at all,
        # label it "new" rather than "stable" so the agent sees the signal.
        if current / days >= 0.1:
            return "new", None
        return "stable", 0.0

    if change >= 50.0:
        label = "surging"
    elif change >= 10.0:
        label = "rising"
    elif change <= -10.0:
        label = "falling"
    else:
        label = "stable"
    return label, change


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0


def _near_duplicate(candidate: _Candidate, selected: _Candidate) -> bool:
    overlap = _jaccard(candidate.product_indexes, selected.product_indexes)
    if overlap >= 0.8:
        return True
    contains = candidate.keyword in selected.keyword or selected.keyword in candidate.keyword
    if contains and overlap >= 0.3:
        return True
    # Detect anagrams: same characters in different order (e.g. "红包结婚" vs
    # "结婚红包"). Only suppress when product overlap > 0 to avoid collapsing
    # semantically distinct keywords that happen to share characters.
    if overlap > 0.0 and sorted(candidate.keyword) == sorted(selected.keyword):
        return True
    return False


def _select_distinct(candidates: list[_Candidate], score_attr: str, limit: int) -> list[_Candidate]:
    ranked = sorted(
        candidates,
        key=lambda item: (-getattr(item, score_attr), -len(item.keyword), item.keyword),
    )
    selected: list[_Candidate] = []
    seen_keywords: set[str] = set()
    for candidate in ranked:
        if getattr(candidate, score_attr) <= 0.0:
            continue
        # Skip if this exact keyword was already selected (prevents same keyword
        # appearing twice with different category labels when category=None).
        if candidate.keyword in seen_keywords:
            continue
        if any(_near_duplicate(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
        seen_keywords.add(candidate.keyword)
        if len(selected) == limit:
            break
    return selected


class HotSearchIndex:
    """A compact, immutable title index with live demand/listing evaluation."""

    def __init__(self, products: Iterable):
        self.products = tuple(products)
        support: Counter[str] = Counter()
        terms_by_product: list[frozenset[str]] = []
        for product in self.products:
            terms = extract_query_phrases(product.name)
            terms_by_product.append(terms)
            support.update(terms)

        retained = {term for term, count in support.items() if count >= 2}
        postings: dict[str, list[int]] = defaultdict(list)
        for product_index, terms in enumerate(terms_by_product):
            for term in terms & retained:
                postings[term].append(product_index)
        self.postings = {term: tuple(indexes) for term, indexes in postings.items()}

    def rank(
        self,
        *,
        category: str | None,
        window_days: int,
        today_idx: int,
        small_share: float = 1.0,
        limit: int = 10,
    ) -> list[HotSearchTrend]:
        # Product objects are retained by reference, so runtime listing changes
        # and curve updates are visible without rebuilding the title index.
        live_products = self.products
        scope_mask = [
            product.is_listed_by_supplier
            and (category is None or product.category == category)
            for product in live_products
        ]
        min_support = _minimum_support(sum(scope_mask))
        current_totals = [0.0] * len(live_products)
        previous_totals = [0.0] * len(live_products)
        pre_previous_totals = [0.0] * len(live_products)
        for index, product in enumerate(live_products):
            if not scope_mask[index]:
                continue
            curve = product.market_curve
            for offset in range(3 * window_days):
                value = float(curve[(today_idx - offset) % 365])
                if offset < window_days:
                    current_totals[index] += value
                elif offset < 2 * window_days:
                    previous_totals[index] += value
                else:
                    pre_previous_totals[index] += value
        # Apply market share scaling so demand reflects the agent's actual
        # market reach, not the full market demand.
        if small_share != 1.0:
            for index in range(len(live_products)):
                if scope_mask[index]:
                    current_totals[index] *= small_share
                    previous_totals[index] *= small_share
                    pre_previous_totals[index] *= small_share
        candidates: list[_Candidate] = []

        for keyword, posting in self.postings.items():
            indexes = frozenset(index for index in posting if scope_mask[index])
            support = len(indexes)
            if support < min_support:
                continue

            current = previous = pre_previous = 0.0
            by_category: dict[str, float] = defaultdict(float)
            for index in indexes:
                product = live_products[index]
                current_part = current_totals[index]
                previous_part = previous_totals[index]
                current += current_part
                previous += previous_part
                pre_previous += pre_previous_totals[index]
                by_category[product.category] += current_part

            if current <= 0.0 and previous <= 0.0:
                continue
            # When current demand is zero (keyword died this window), use
            # previous demand to determine the category rather than picking
            # the lexicographically first one.
            if category:
                dominant_category = category
            elif any(value > 0.0 for value in by_category.values()):
                dominant_category = sorted(
                    by_category, key=lambda value: (-by_category[value], value)
                )[0]
            else:
                # All by_category values are 0 — fall back to previous demand.
                prev_by_cat: dict[str, float] = defaultdict(float)
                for index in indexes:
                    prev_by_cat[live_products[index].category] += previous_totals[index]
                dominant_category = sorted(
                    prev_by_cat, key=lambda value: (-prev_by_cat[value], value)
                )[0]
            candidates.append(_Candidate(
                keyword=keyword,
                category=dominant_category,
                product_indexes=indexes,
                current_score=_rank_score(current, previous, support, window_days),
                previous_score=_rank_score(previous, pre_previous, support, window_days),
                current_demand=current,
                previous_demand=previous,
            ))

        current_top = _select_distinct(candidates, "current_score", limit)
        previous_top = _select_distinct(candidates, "previous_score", limit)
        previous_ranks = {item.keyword: rank for rank, item in enumerate(previous_top, 1)}

        output = []
        for rank, candidate in enumerate(current_top, 1):
            trend, change_pct = _trend(
                candidate.current_demand, candidate.previous_demand, window_days
            )
            previous_rank = previous_ranks.get(candidate.keyword)
            output.append(HotSearchTrend(
                rank=rank,
                keyword=candidate.keyword,
                category=candidate.category,
                trend=trend,
                change_pct=change_pct,
                rank_change=(previous_rank - rank) if previous_rank is not None else None,
            ))
        return output
