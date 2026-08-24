"""CLI tests for ``python -m data.economy_diagnostics``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data.economy_diagnostics import main

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "env" / "scenarios"
DEFAULT_SCENARIO = SCENARIOS_DIR / "default.yaml"
# * Small catalog keeps this offline CLI path cheap (no simulator).
CLI_N_PRODUCTS = 8
CLI_SEED = 42


def test_main_synthetic_text_prints_key_sections(capsys):
    exit_code = main(
        [
            "--source",
            "synthetic",
            "--scenario",
            str(DEFAULT_SCENARIO),
            "--num-products",
            str(CLI_N_PRODUCTS),
            "--seed",
            str(CLI_SEED),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"Catalog size: {CLI_N_PRODUCTS}" in out
    assert "Mean market_curve:" in out
    assert "min=" in out
    assert "p25=" in out
    assert "median=" in out
    assert "p75=" in out
    assert "max=" in out
    assert "sale=ref:" in out
    assert "sale=2x cost:" in out
    assert "sale=10x cost:" in out
    assert "listing-day demand=" in out
    assert "listing-day gross=" in out
    assert "1000/hour" in out
    assert "overflow" in out


def test_main_synthetic_json_parses(capsys):
    exit_code = main(
        [
            "--source",
            "synthetic",
            "--scenario",
            str(DEFAULT_SCENARIO),
            "--num-products",
            str(CLI_N_PRODUCTS),
            "--seed",
            str(CLI_SEED),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "synthetic"
    assert payload["n_products"] == CLI_N_PRODUCTS
    assert payload["seed"] == CLI_SEED
    curve = payload["mean_market_curve"]
    assert set(curve) >= {"min", "p25", "median", "p75", "max"}
    aggregates = payload["aggregates"]
    for key in ("at_ref", "at_2x_cost", "at_10x_cost"):
        assert "mean_listing_day_demand" in aggregates[key]
        assert "mean_listing_day_gross" in aggregates[key]
    assert payload["limitations"]["applies_hourly_cap"] is False
    assert payload["limitations"]["extreme_price_overflow_to_zero"] is True
    assert payload["economy_v6"]["enabled"] is False


def test_invalid_source_raises_argparse_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["--source", "not-a-source"])
    assert excinfo.value.code not in (0, None)


def test_missing_scenario_raises_argparse_error(tmp_path):
    missing = tmp_path / "absent.yaml"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--source",
                "synthetic",
                "--scenario",
                str(missing),
            ]
        )
    assert excinfo.value.code not in (0, None)
