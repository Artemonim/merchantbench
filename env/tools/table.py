"""Helpers for compact agent-facing table payloads."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def compact_table(rows: Iterable[Mapping[str, Any]], columns: list[str] | tuple[str, ...]) -> dict:
    """Return a JSON-serializable column table.

    Multi-row tool outputs use this shape so column names are emitted once
    instead of repeated for every row.
    """
    cols = list(columns)
    values = [[row.get(col) for col in cols] for row in rows]
    return {"columns": cols, "rows": values, "count": len(values)}
