#!/usr/bin/env python3
"""Rebuild experiments/run_history from env/runs/*/agent/run_summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "env"
sys.path.insert(0, str(ENV_ROOT))

from storage import agent_log  # noqa: E402


def main() -> int:
    runs_root = ENV_ROOT / "runs"
    payload = agent_log.rebuild_run_history_from_runs(str(runs_root))
    jsonl_path, index_path = agent_log.run_history_paths(str(ROOT))
    print(f"RUN_HISTORY_JSONL={jsonl_path}")
    print(f"RUN_HISTORY_JSON={index_path}")
    print(f"run_count={payload.get('run_count')}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
