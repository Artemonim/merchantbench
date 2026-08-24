"""Private daily opportunity report reader.

Reports are private data assets stored outside the open-source surface under
``env/data/private_data/daily_reports``. Each filename is the publication date;
the report summarizes data through the preceding date. Runtime access is
intentionally keyed only by the simulation date so agents cannot request
arbitrary future reports.
"""

from __future__ import annotations

import os
from datetime import date

from compat import env_value

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_DAILY_REPORT_DIR = os.path.join(_HERE, "private_data", "daily_reports")


def resolve_report_dir(path: str | None = None) -> str:
    private_root = env_value("MERCHANTBENCH_PRIVATE_DATA_ROOT", "REALSHOP_PRIVATE_DATA_ROOT")
    if not path:
        return os.path.join(private_root, "daily_reports") if private_root else DEFAULT_DAILY_REPORT_DIR
    candidate = path if os.path.isabs(path) else os.path.join(_ENV_ROOT, path)
    if private_root and not os.path.isabs(path):
        return os.path.join(private_root, os.path.basename(os.path.normpath(path)))
    if os.path.exists(candidate) or not private_root:
        return candidate
    leaf = os.path.basename(os.path.normpath(path))
    return private_root if leaf == os.path.basename(private_root) else os.path.join(private_root, leaf)


def report_path(report_dir: str, report_date: date) -> str:
    filename = report_date.strftime("%Y%m%d") + ".md"
    return os.path.join(report_dir, filename)


def read_report(report_dir: str, report_date: date) -> str | None:
    path = report_path(report_dir, report_date)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
