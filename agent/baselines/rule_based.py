"""Rule-based MerchantBench baseline with daily-report and random sourcing modes."""

from __future__ import annotations

import argparse
import os
import sys

_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from baselines.auto_seed import SELECTION_MODES, RuleBasedAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=(os.environ.get("MERCHANTBENCH_RUN_ID") or os.environ.get("REALSHOP_RUN_ID"))
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MERCHANTBENCH_BASE_URL", os.environ.get("REALSHOP_BASE_URL", "http://localhost:5000")),
    )
    parser.add_argument(
        "--agent-id", default=os.environ.get("MERCHANTBENCH_AGENT_ID", os.environ.get("REALSHOP_AGENT_ID", "agent_0"))
    )
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="daily_report")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.run_id:
        parser.error("--run-id is required (MERCHANTBENCH_RUN_ID or legacy REALSHOP_RUN_ID)")
    RuleBasedAgent(
        args.base_url,
        args.run_id,
        args.agent_id,
        seed_count=args.seed_count,
        timeout=args.timeout,
        selection_mode=args.selection_mode,
        selection_seed=args.selection_seed,
    ).run(max_steps=args.max_steps, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
