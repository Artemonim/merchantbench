"""Private daily opportunity report reader.

Reports are private data assets stored outside the open-source surface under
``env/data/private_data/daily_reports``. Each filename is the publication date;
the report summarizes data through the preceding date. Runtime access is
intentionally keyed only by the simulation date so agents cannot request
arbitrary future reports.
"""
from __future__ import annotations

from datetime import date
import os


_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
DEFAULT_DAILY_REPORT_DIR = os.path.join(_HERE, "private_data", "daily_reports")


def resolve_report_dir(path: str | None = None) -> str:
    if not path:
        return DEFAULT_DAILY_REPORT_DIR
    if os.path.isabs(path):
        return path
    return os.path.join(_ENV_ROOT, path)


def report_path(report_dir: str, report_date: date) -> str:
    filename = report_date.strftime("%Y%m%d") + ".md"
    return os.path.join(report_dir, filename)


def read_report(report_dir: str, report_date: date) -> str | None:
    path = report_path(report_dir, report_date)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
