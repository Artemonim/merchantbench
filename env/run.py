"""Entry point: python run.py [--host HOST] [--port PORT]."""
from __future__ import annotations

import argparse
import signal
from pathlib import Path

from dotenv import load_dotenv

from web.app import create_app


def _load_local_dotenv() -> str | None:
    """Load an env-local or monorepo-root .env without overriding exports."""
    env_root = Path(__file__).resolve().parent
    for candidate in (env_root / ".env", env_root.parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return str(candidate)
    return None


def _exit_on_sigterm(signum, _frame) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    _load_local_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app = create_app()
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
    finally:
        app.registry.shutdown()


if __name__ == "__main__":
    main()
