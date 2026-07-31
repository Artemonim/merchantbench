"""Flask app factory."""
from __future__ import annotations

import os
from flask import Flask

from web.auth import enforce_optional_auth
from web.routes_agent import make_blueprint as agent_bp
from web.routes_dashboard import make_blueprint as dashboard_bp
from web.runner import RunRegistry, load_default_scenario


def create_app(
    db_path: str | None = None,
    runs_root: str | None = None,
    run_db_filename: str | None = None,
) -> Flask:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_default_scenario()["run"]
    db_path = db_path or os.path.join(here, cfg.get("db_path", "runs/merchantbench.db"))
    runs_root = runs_root or os.path.join(here, cfg.get("runs_root", "runs"))
    run_db_filename = run_db_filename or cfg.get("run_db_filename", "state.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    app = Flask(__name__, template_folder=os.path.join(here, "web", "templates"))
    app.config["JSON_AS_ASCII"] = False
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MERCHANTBENCH_REQUIRE_TOKENS"] = os.environ.get(
        "MERCHANTBENCH_REQUIRE_TOKENS", ""
    ).lower() in ("1", "true", "yes", "on")
    app.config["MERCHANTBENCH_ADMIN_TOKEN"] = os.environ.get("MERCHANTBENCH_ADMIN_TOKEN")
    registry = RunRegistry(
        db_path=db_path,
        runs_root=runs_root,
        run_db_filename=run_db_filename,
    )
    app.registry = registry

    app.before_request(enforce_optional_auth)
    app.register_blueprint(dashboard_bp(registry))
    app.register_blueprint(agent_bp(registry))
    return app
