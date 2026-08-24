"""Resolve economy v6 fee flags and category rates from scenario YAML.

YAML is the source of truth. This module introduces no RNG: fees are
deterministic functions of category plus the enabled flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Fallback rates used only when YAML omits a key.
_DEFAULT_TAKE_RATE = 0.08
_DEFAULT_FULFILLMENT_FEE = 8.0
_DEFAULT_COST_RECOVERY_RATE = 0.85


def _as_float(value: object, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a number") from exc


def _float_map(raw: object, path: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping of category to number")
    out: dict[str, float] = {}
    for key, value in raw.items():
        out[str(key)] = _as_float(value, f"{path}.{key}")
    return out


def _nested_map(block: dict, key: str) -> dict:
    raw = block.get(key)
    return raw if isinstance(raw, dict) else {}


@dataclass(frozen=True)
class EconomyV6:
    """Resolved economy v6 switches and per-category fee tables."""

    enabled: bool = False
    take_rate_enabled: bool = False
    fulfillment_enabled: bool = False
    refund_enabled: bool = False
    reverse_fulfillment: bool = False
    take_rate_default: float = _DEFAULT_TAKE_RATE
    take_rate_by_category: dict[str, float] = field(default_factory=dict)
    fulfillment_default: float = _DEFAULT_FULFILLMENT_FEE
    fulfillment_by_category: dict[str, float] = field(default_factory=dict)
    _refund_cost_recovery_rate: float = _DEFAULT_COST_RECOVERY_RATE

    def take_rate(self, category: str) -> float:
        """Return the platform take-rate for ``category`` (clamped to [0, 1])."""
        raw = self.take_rate_by_category.get(str(category), self.take_rate_default)
        return min(1.0, max(0.0, float(raw)))

    def fulfillment_fee(self, category: str) -> float:
        """Return the outbound/reverse fulfillment fee for ``category`` (RMB)."""
        raw = self.fulfillment_by_category.get(str(category), self.fulfillment_default)
        return max(0.0, float(raw))

    def cost_recovery_rate(self) -> float:
        """Return refund COGS recovery α, or 1.0 when refund v6 is off."""
        if not self.refund_enabled:
            return 1.0
        return min(1.0, max(0.0, float(self._refund_cost_recovery_rate)))

    @classmethod
    def from_scenario(cls, scenario: dict | None) -> EconomyV6:
        """Parse ``scenario["economy_v6"]``. Missing or master-off → v5 cash paths."""
        if not isinstance(scenario, dict):
            return cls()
        block = scenario.get("economy_v6")
        if not isinstance(block, dict):
            return cls()

        master = bool(block.get("enabled", False))
        take_cfg = _nested_map(block, "take_rate")
        fulfill_cfg = _nested_map(block, "fulfillment")
        refund_cfg = _nested_map(block, "refund")

        refund_on = master and bool(refund_cfg.get("enabled", False))
        return cls(
            enabled=master,
            take_rate_enabled=master and bool(take_cfg.get("enabled", False)),
            fulfillment_enabled=master and bool(fulfill_cfg.get("enabled", False)),
            refund_enabled=refund_on,
            reverse_fulfillment=(refund_on and bool(refund_cfg.get("reverse_fulfillment", True))),
            take_rate_default=_as_float(
                take_cfg.get("default", _DEFAULT_TAKE_RATE),
                "economy_v6.take_rate.default",
            ),
            take_rate_by_category=_float_map(
                take_cfg.get("by_category"),
                "economy_v6.take_rate.by_category",
            ),
            fulfillment_default=_as_float(
                fulfill_cfg.get("default_fee", _DEFAULT_FULFILLMENT_FEE),
                "economy_v6.fulfillment.default_fee",
            ),
            fulfillment_by_category=_float_map(
                fulfill_cfg.get("by_category"),
                "economy_v6.fulfillment.by_category",
            ),
            _refund_cost_recovery_rate=_as_float(
                refund_cfg.get("cost_recovery_rate", _DEFAULT_COST_RECOVERY_RATE),
                "economy_v6.refund.cost_recovery_rate",
            ),
        )


def contribution_margin_pct(gmv: float, cogs: float, fee_total: float) -> float:
    """Return (GMV - COGS - fee_total) / GMV as a percent, or 0.0 when GMV is 0.

    Fines are excluded. Distinct from ``net_profit_margin``, which is a 0-1
    fraction of GMV using fee-aware settled net profit (including fines).
    """
    gmv_value = float(gmv)
    if abs(gmv_value) <= 1e-12:
        return 0.0
    return (gmv_value - float(cogs) - float(fee_total)) / gmv_value * 100.0


def public_return_rate(refund_rate: float, only_refund_rate: float) -> float:
    """Public product-card return rate: clamp(refund + only_refund, 0, 1)."""
    combined = float(refund_rate or 0.0) + float(only_refund_rate or 0.0)
    return round(min(1.0, max(0.0, combined)), 4)
