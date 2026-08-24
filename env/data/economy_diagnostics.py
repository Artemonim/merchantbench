"""Offline CES listing-day diagnostics for synthetic-economy ablations.

These helpers evaluate catalog demand and gross profit at a chosen sale
price with lifecycle=1 and rating=1. They do not start the simulator.

Run as ``python -m data.economy_diagnostics`` for a catalog report. The
CES helpers do not apply the simulator 1000/hour cap or lifecycle/rating
multipliers; extreme sale prices can overflow to ``0.0``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, TypedDict

import numpy as np
from core.demand import MIN_SALE_PRICE
from core.economy_v6 import EconomyV6, public_return_rate

# * Matches agent.baselines.auto_seed.DEFAULT_MARKUP (rule_based sale = markup * cost).
RULE_BASED_DEFAULT_MARKUP = 2.00
TEN_X_COST_MULTIPLIER = 10.0
APPLIANCES_CATEGORY = "appliances"


class CatalogEconomyAggregates(TypedDict):
    """Mean CES listing-day metrics over a catalog (or a category slice)."""

    n_products: int
    share_eps_lt_1: float
    mean_margin_at_ref: float
    mean_q_day_at_ref: float
    mean_q_day_at_rule_markup: float
    mean_gross_day_at_ref: float
    mean_gross_day_at_rule_markup: float
    mean_gross_day_at_10x_cost: float


def listing_day_demand_at_sale(
    product: Any,
    sale_price: float,
    small_share: float,
) -> float:
    """Return CES listing-day demand at ``sale_price``.

    Hourly weights sum to 1, so a day at lifecycle=1 and rating=1 is
    ``mean(market_curve) * small_share * (sale / ref) ** (-ε)``.

    Args:
        product: Catalog product with ``market_curve``, ``ref_price``,
            and ``elasticity``.
        sale_price: Merchant listing price.
        small_share: Shop share of market demand.

    Returns:
        Expected units per listing-day, or ``0.0`` when the CES inputs
        are invalid (``sale < 0.01``, ``ref <= 0``, ``ε < 0``, or
        non-finite).
    """
    try:
        sale = float(sale_price)
        ref = float(product.ref_price)
        elasticity = float(product.elasticity)
        share = float(small_share)
        curve_mean = float(np.mean(product.market_curve))
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(sale)
        or sale < MIN_SALE_PRICE
        or not math.isfinite(ref)
        or ref <= 0.0
        or not math.isfinite(elasticity)
        or elasticity < 0.0
        or not math.isfinite(share)
        or share < 0.0
        or not math.isfinite(curve_mean)
        or curve_mean <= 0.0
    ):
        return 0.0
    scale = curve_mean * share
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    # * Same CES as core.demand.expected_demand, collapsed over a day.
    try:
        log_demand = math.log(scale) - elasticity * (math.log(sale) - math.log(ref))
    except (OverflowError, ValueError):
        return 0.0
    if not math.isfinite(log_demand):
        return 0.0
    try:
        demand = math.exp(log_demand)
    except OverflowError:
        return 0.0
    if not math.isfinite(demand) or demand <= 0.0:
        return 0.0
    return float(demand)


def listing_day_gross_at_sale(
    product: Any,
    sale_price: float,
    small_share: float,
) -> float:
    """Return expected listing-day gross profit at ``sale_price``.

    Gross is ``q_day * (sale - cost)`` where ``cost`` is ``product.price``.

    Args:
        product: Catalog product with ``price`` plus CES demand fields.
        sale_price: Merchant listing price.
        small_share: Shop share of market demand.

    Returns:
        Expected gross profit per listing-day, or ``0.0`` when demand or
        cost is invalid.
    """
    try:
        sale = float(sale_price)
        cost = float(product.price)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sale) or not math.isfinite(cost):
        return 0.0
    demand = listing_day_demand_at_sale(product, sale, small_share)
    gross = demand * (sale - cost)
    if not math.isfinite(gross):
        return 0.0
    return float(gross)


def catalog_economy_aggregates(
    products: Iterable[Any],
    small_share: float,
    *,
    rule_markup: float = RULE_BASED_DEFAULT_MARKUP,
    category: str | None = None,
) -> CatalogEconomyAggregates:
    """Return catalog-level CES listing-day aggregates.

    Metrics use lifecycle=1 and rating=1. Rule-based markup prices at
    ``rule_markup * cost``. The 10× probe prices at
    ``TEN_X_COST_MULTIPLIER * cost``.

    Args:
        products: Catalog products.
        small_share: Shop share of market demand.
        rule_markup: Multiplier applied to cost for the rule_based probe.
        category: If set, restrict to this ``product.category``.

    Returns:
        Mapping of share / mean demand / mean gross metrics.

    Raises:
        ValueError: If the (filtered) catalog is empty, or ``rule_markup``
            is not finite and positive.
    """
    markup = float(rule_markup)
    if not math.isfinite(markup) or markup <= 0.0:
        raise ValueError(f"rule_markup must be positive and finite, got {rule_markup!r}")
    catalog = _select_products(products, category)
    n = len(catalog)
    share = float(small_share)

    eps_lt_1 = 0.0
    margins: list[float] = []
    q_at_ref: list[float] = []
    q_at_markup: list[float] = []
    gross_at_ref: list[float] = []
    gross_at_markup: list[float] = []
    gross_at_10x: list[float] = []
    for product in catalog:
        elasticity = _finite_float(getattr(product, "elasticity", None))
        if elasticity is not None and elasticity < 1.0:
            eps_lt_1 += 1.0
        ref = _finite_float(getattr(product, "ref_price", None))
        cost = _finite_float(getattr(product, "price", None))
        if ref is not None and ref > 0.0 and cost is not None:
            margins.append((ref - cost) / ref)
        else:
            margins.append(0.0)

        q_ref = listing_day_demand_at_sale(product, ref if ref is not None else 0.0, share)
        q_at_ref.append(q_ref)
        gross_at_ref.append(listing_day_gross_at_sale(product, ref if ref is not None else 0.0, share))

        sale_markup = (cost * markup) if cost is not None else 0.0
        q_at_markup.append(listing_day_demand_at_sale(product, sale_markup, share))
        gross_at_markup.append(listing_day_gross_at_sale(product, sale_markup, share))

        sale_10x = (cost * TEN_X_COST_MULTIPLIER) if cost is not None else 0.0
        gross_at_10x.append(listing_day_gross_at_sale(product, sale_10x, share))

    return {
        "n_products": n,
        "share_eps_lt_1": eps_lt_1 / float(n),
        "mean_margin_at_ref": float(np.mean(margins)),
        "mean_q_day_at_ref": float(np.mean(q_at_ref)),
        "mean_q_day_at_rule_markup": float(np.mean(q_at_markup)),
        "mean_gross_day_at_ref": float(np.mean(gross_at_ref)),
        "mean_gross_day_at_rule_markup": float(np.mean(gross_at_markup)),
        "mean_gross_day_at_10x_cost": float(np.mean(gross_at_10x)),
    }


def expected_contribution_at_sale(
    product: Any,
    sale_price: float,
    economy: EconomyV6,
    *,
    include_refund_expectation: bool = False,
) -> float:
    """Return expected unit contribution at ``sale_price``.

    Uses ``p * (1 - τ) - c - F`` with EconomyV6 category rates. ``τ`` and
    ``F`` are 0 when the matching flag is off. Refund expectation is
    optional and off by default: when on, subtract
    ``return_rate * ((1 - α) * c + reverse_F)``.

    Args:
        product: Catalog product with ``price`` and ``category``.
        sale_price: Merchant listing price ``p``.
        economy: Resolved v6 fee tables and flags.
        include_refund_expectation: If true and refund v6 is on, apply the
            expected refund haircut described above.

    Returns:
        Unit contribution, or ``0.0`` when sale/cost is invalid.
    """
    try:
        sale = float(sale_price)
        cost = float(product.price)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sale) or not math.isfinite(cost):
        return 0.0
    category = str(getattr(product, "category", "") or "")
    take = economy.take_rate(category) if economy.take_rate_enabled else 0.0
    fulfill = economy.fulfillment_fee(category) if economy.fulfillment_enabled else 0.0
    contrib = sale * (1.0 - take) - cost - fulfill
    if include_refund_expectation and economy.refund_enabled:
        refund_rate = getattr(product, "refund_rate", 0.0)
        only_refund_rate = getattr(product, "only_refund_rate", 0.0)
        return_rate = public_return_rate(refund_rate, only_refund_rate)
        recovered = economy.cost_recovery_rate()
        reverse_fee = economy.fulfillment_fee(category) if economy.reverse_fulfillment else 0.0
        contrib -= return_rate * ((1.0 - recovered) * cost + reverse_fee)
    if not math.isfinite(contrib):
        return 0.0
    return float(contrib)


def expected_contribution_at_ref(
    product: Any,
    economy: EconomyV6,
    *,
    include_refund_expectation: bool = False,
) -> float:
    """Return expected unit contribution at ``sale = ref_price``."""
    try:
        ref = float(product.ref_price)
    except (TypeError, ValueError):
        return 0.0
    return expected_contribution_at_sale(
        product,
        ref,
        economy,
        include_refund_expectation=include_refund_expectation,
    )


def share_negative_contribution_at_ref(
    products: Iterable[Any],
    economy: EconomyV6,
    *,
    include_refund_expectation: bool = False,
) -> float:
    """Return the share of catalog products with negative contribution at sale=ref.

    Args:
        products: Catalog products.
        economy: Resolved v6 fee tables and flags.
        include_refund_expectation: Forwarded to ``expected_contribution_at_ref``.

    Returns:
        Fraction in ``[0, 1]``.

    Raises:
        ValueError: If ``products`` is empty.
    """
    catalog = list(products)
    if not catalog:
        raise ValueError("products must be non-empty")
    n_neg = 0
    for product in catalog:
        if (
            expected_contribution_at_ref(
                product,
                economy,
                include_refund_expectation=include_refund_expectation,
            )
            < 0.0
        ):
            n_neg += 1
    return float(n_neg) / float(len(catalog))


def _select_products(products: Iterable[Any], category: str | None) -> list[Any]:
    """Return the catalog, optionally filtered by category."""
    catalog = list(products)
    if category is not None:
        catalog = [product for product in catalog if product.category == category]
    if not catalog:
        if category is None:
            raise ValueError("products must be non-empty")
        raise ValueError(f"no products in category {category!r}")
    return catalog


def _finite_float(value: Any) -> float | None:
    """Parse a finite float, or return ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


_ENV_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _ENV_ROOT.parent
_DEFAULT_SCENARIO = "env/scenarios/default.yaml"
_DEFAULT_SEED = 42
_CLI_SOURCES = ("synthetic", "private_real")
_CLI_FORMATS = ("text", "json")
_LIMITATIONS_NOTE = (
    "CES listing-day helpers do not apply the simulator 1000/hour cap "
    "or lifecycle/rating multipliers; extreme sale prices can overflow "
    "to 0.0 demand/gross."
)


def _positive_int(raw: str) -> int:
    """Parse a positive integer CLI argument."""
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw!r}")
    return value


def _resolve_scenario_path(raw: str) -> Path:
    """Resolve a scenario path against cwd, repo root, and env root.

    Accepts ``env/scenarios/default.yaml`` from either the repo root or
    ``env/`` as the working directory.
    """
    text = str(raw).strip()
    if not text:
        raise argparse.ArgumentTypeError("scenario path must be non-empty")
    given = Path(text)
    probes: list[Path] = [given]
    if not given.is_absolute():
        probes.extend((_REPO_ROOT / given, _ENV_ROOT / given))
        normalized = text.replace("\\", "/")
        if normalized.startswith("env/"):
            probes.append(_ENV_ROOT / Path(normalized[len("env/") :]))
    seen: set[str] = set()
    for probe in probes:
        try:
            resolved = probe.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    raise argparse.ArgumentTypeError(f"scenario file not found: {raw}")


def _build_parser() -> argparse.ArgumentParser:
    """Return the economy-diagnostics CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m data.economy_diagnostics",
        description=("Print CES listing-day catalog diagnostics without starting the simulator."),
        epilog=(
            f"{_LIMITATIONS_NOTE} Examples: "
            "python -m data.economy_diagnostics --source synthetic "
            "--scenario env/scenarios/default.yaml ; "
            "python -m data.economy_diagnostics --source private_real "
            "--scenario env/scenarios/economy_v6.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=_CLI_SOURCES,
        help="Catalog source: synthetic generate() or private_real sqlite.",
    )
    parser.add_argument(
        "--scenario",
        type=_resolve_scenario_path,
        default=_DEFAULT_SCENARIO,
        help=(
            "Scenario YAML path (default: env/scenarios/default.yaml). Resolved relative to cwd, repo root, or env/."
        ),
    )
    parser.add_argument(
        "--num-products",
        type=_positive_int,
        default=None,
        help="Catalog size override (default: scenario data.num_products).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Master seed for generate/subsample (default: 42).",
    )
    parser.add_argument(
        "--format",
        choices=_CLI_FORMATS,
        default="text",
        dest="output_format",
        help="Stdout format (default: text).",
    )
    return parser


def _curve_mean_distribution(products: list[Any]) -> dict[str, float]:
    """Return min/p25/median/p75/max of per-product mean(market_curve)."""
    means: list[float] = []
    for product in products:
        try:
            value = float(np.mean(product.market_curve))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(value):
            means.append(value)
    if not means:
        raise ValueError("catalog has no finite market_curve means")
    arr = np.asarray(means, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _mean_demand_at_10x_cost(products: list[Any], small_share: float) -> float:
    """Return mean listing-day demand at ``TEN_X_COST_MULTIPLIER * cost``."""
    demands: list[float] = []
    for product in products:
        cost = _finite_float(getattr(product, "price", None))
        sale = (cost * TEN_X_COST_MULTIPLIER) if cost is not None else 0.0
        demands.append(listing_day_demand_at_sale(product, sale, small_share))
    return float(np.mean(demands))


def _load_catalog(
    source: str,
    scenario: dict[str, Any],
    num_products: int,
    seed: int,
) -> list[Any]:
    """Return products for ``source`` using generate() or private_real."""
    scenario = copy.deepcopy(scenario)
    scenario.setdefault("run", {})["master_seed"] = int(seed)
    data_cfg = scenario.setdefault("data", {})
    data_cfg["num_products"] = int(num_products)

    if source == "synthetic":
        from data.synth import generate

        products, _hourly = generate(scenario)
        return list(products)

    from data.private_real import (
        PrivateRealDataError,
        load_dataset,
        resolve_dataset_path,
        subsample_catalog,
    )

    pool_path = data_cfg.get("catalog_pool_path") or data_cfg.get("private_real_db_path")
    try:
        products, hourly_dist, _meta = load_dataset(resolve_dataset_path(pool_path))
        products, _hourly = subsample_catalog(
            products,
            hourly_dist,
            int(num_products),
            int(seed),
        )
    except PrivateRealDataError as exc:
        raise FileNotFoundError(str(exc)) from exc
    return list(products)


def _build_report(
    *,
    source: str,
    scenario_path: Path,
    scenario: dict[str, Any],
    products: list[Any],
    seed: int,
) -> dict[str, Any]:
    """Assemble the structured diagnostics payload."""
    data_cfg = scenario.get("data") or {}
    try:
        small_share = float(data_cfg.get("small_share", 1.0))
    except (TypeError, ValueError):
        small_share = 1.0
    aggregates = catalog_economy_aggregates(products, small_share)
    curve = _curve_mean_distribution(products)
    report: dict[str, Any] = {
        "source": source,
        "scenario": str(scenario_path),
        "seed": int(seed),
        "n_products": int(len(products)),
        "small_share": small_share,
        "limitations": {
            "applies_hourly_cap": False,
            "hourly_cap_units": 1000,
            "applies_lifecycle": False,
            "applies_rating": False,
            "extreme_price_overflow_to_zero": True,
            "note": _LIMITATIONS_NOTE,
        },
        "mean_market_curve": curve,
        "aggregates": {
            "at_ref": {
                "mean_listing_day_demand": aggregates["mean_q_day_at_ref"],
                "mean_listing_day_gross": aggregates["mean_gross_day_at_ref"],
            },
            "at_2x_cost": {
                "mean_listing_day_demand": aggregates["mean_q_day_at_rule_markup"],
                "mean_listing_day_gross": aggregates["mean_gross_day_at_rule_markup"],
                "cost_multiplier": RULE_BASED_DEFAULT_MARKUP,
            },
            "at_10x_cost": {
                "mean_listing_day_demand": _mean_demand_at_10x_cost(products, small_share),
                "mean_listing_day_gross": aggregates["mean_gross_day_at_10x_cost"],
                "cost_multiplier": TEN_X_COST_MULTIPLIER,
            },
        },
    }
    economy = EconomyV6.from_scenario(scenario)
    v6_block: dict[str, Any] = {"enabled": bool(economy.enabled)}
    if economy.enabled:
        contribs = [expected_contribution_at_ref(product, economy) for product in products]
        v6_block["mean_expected_contribution_at_ref"] = float(np.mean(contribs))
        v6_block["share_negative_contribution_at_ref"] = share_negative_contribution_at_ref(products, economy)
    report["economy_v6"] = v6_block
    return report


def _fmt_num(value: float) -> str:
    """Format a finite float for text output."""
    return f"{float(value):.6g}"


def _format_text(report: dict[str, Any]) -> str:
    """Render the diagnostics report as human-readable text."""
    curve = report["mean_market_curve"]
    aggregates = report["aggregates"]
    lines = [
        f"Catalog size: {report['n_products']}",
        f"source: {report['source']}",
        f"scenario: {report['scenario']}",
        f"seed: {report['seed']}",
        "",
        "Limitations:",
        f"  {report['limitations']['note']}",
        "",
        "Mean market_curve:",
        f"  min={_fmt_num(curve['min'])}",
        f"  p25={_fmt_num(curve['p25'])}",
        f"  median={_fmt_num(curve['median'])}",
        f"  p75={_fmt_num(curve['p75'])}",
        f"  max={_fmt_num(curve['max'])}",
        "",
        "Aggregates (listing-day demand and gross):",
        "  sale=ref:",
        f"    listing-day demand={_fmt_num(aggregates['at_ref']['mean_listing_day_demand'])}",
        f"    listing-day gross={_fmt_num(aggregates['at_ref']['mean_listing_day_gross'])}",
        "  sale=2x cost:",
        f"    listing-day demand={_fmt_num(aggregates['at_2x_cost']['mean_listing_day_demand'])}",
        f"    listing-day gross={_fmt_num(aggregates['at_2x_cost']['mean_listing_day_gross'])}",
        "  sale=10x cost:",
        f"    listing-day demand={_fmt_num(aggregates['at_10x_cost']['mean_listing_day_demand'])}",
        f"    listing-day gross={_fmt_num(aggregates['at_10x_cost']['mean_listing_day_gross'])}",
    ]
    v6 = report["economy_v6"]
    if v6.get("enabled"):
        lines.extend(
            [
                "",
                "v6 fees:",
                f"  expected contribution at ref (mean)={_fmt_num(v6['mean_expected_contribution_at_ref'])}",
                "  share of products with negative contribution at ref="
                f"{_fmt_num(v6['share_negative_contribution_at_ref'])}",
            ]
        )
    else:
        lines.extend(["", "v6 fees: off"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run listing-day economy diagnostics from the command line.

    Args:
        argv: Argument vector without the program name. ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        from web.runner import load_scenario

        scenario = load_scenario(str(args.scenario))
    except (OSError, ValueError) as exc:
        print(f"error: failed to load scenario {args.scenario}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # * YAML parse errors and extends-cycle failures.
        print(f"error: failed to load scenario {args.scenario}: {exc}", file=sys.stderr)
        return 1

    data_cfg = scenario.get("data") or {}
    if args.num_products is not None:
        num_products = int(args.num_products)
    else:
        try:
            num_products = int(data_cfg["num_products"])
        except (KeyError, TypeError, ValueError):
            print(
                "error: scenario is missing data.num_products; pass --num-products",
                file=sys.stderr,
            )
            return 1
        if num_products <= 0:
            print(
                f"error: data.num_products must be a positive integer, got {num_products!r}",
                file=sys.stderr,
            )
            return 1

    try:
        products = _load_catalog(args.source, scenario, num_products, int(args.seed))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: failed to load catalog: {exc}", file=sys.stderr)
        return 1

    if not products:
        print("error: catalog is empty", file=sys.stderr)
        return 1

    try:
        report = _build_report(
            source=args.source,
            scenario_path=args.scenario,
            scenario=scenario,
            products=products,
            seed=int(args.seed),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
