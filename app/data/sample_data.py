"""Reader for the mock Excel tables under ``app/sample_data``.

Each table is its own workbook, keyed by the ``sub`` column:

| table | file | columns |
| --- | --- | --- |
| ``profile`` | ``profile.xlsx`` | cust_name, grade |
| ``manager`` | ``manager.xlsx`` | manager |
| ``team`` | ``team info.xlsx`` | team_name, team_size |

Splitting the data this way lets each mock MCP operation read its own table
independently, so the three reads can run concurrently. Each table is parsed
once and cached, making a later lookup a dict access.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openpyxl

from app.core.exceptions import EmployeeNotFoundError, SampleDataError


SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data"

TABLE_FILES: dict[str, str] = {
    "profile": "profile.xlsx",
    "manager": "manager.xlsx",
    "team": "team info.xlsx",
}

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile": ("sub", "cust_name", "grade"),
    "manager": ("sub", "manager"),
    "team": ("sub", "team_name", "team_size"),
}

INTEGER_COLUMNS = frozenset({"team_size"})

_parse_lock = threading.Lock()
_parsed_tables: dict[str, dict[str, dict[str, Any]]] = {}


def load_table(table: str) -> dict[str, dict[str, Any]]:
    """Return one table keyed by ``sub``, parsing it at most once."""
    parsed = _parsed_tables.get(table)
    if parsed is not None:
        return parsed

    with _parse_lock:
        # Re-check under the lock: another thread may have parsed it meanwhile.
        parsed = _parsed_tables.get(table)
        if parsed is None:
            parsed = _read_table(table)
            _parsed_tables[table] = parsed
        return parsed


def get_row(table: str, user_id: str) -> dict[str, Any]:
    """Return the row of *table* whose ``sub`` is *user_id*."""
    row = load_table(table).get(user_id)
    if row is None:
        raise EmployeeNotFoundError(
            f"No {table!r} row for user id {user_id!r} in {table_path(table).name}"
        )
    return row


def table_path(table: str) -> Path:
    """Return the workbook backing *table*."""
    try:
        filename = TABLE_FILES[table]
    except KeyError:
        raise SampleDataError(f"Unknown mock data table {table!r}") from None
    return SAMPLE_DATA_DIR / filename


def _read_table(table: str) -> dict[str, dict[str, Any]]:
    path = table_path(table)
    required = TABLE_COLUMNS[table]

    parsed: dict[str, dict[str, Any]] = {}
    for row in _read_rows(path, required):
        for column in INTEGER_COLUMNS.intersection(row):
            row[column] = _require_int(row[column], column, path)
        parsed[_require_text(row["sub"], "sub", path)] = row

    if not parsed:
        raise SampleDataError(f"{path} contains no rows")
    return parsed


def _read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, Any]]:
    """Return the first worksheet as a list of column-name keyed rows."""
    if not path.exists():
        raise SampleDataError(f"Mock data file not found: {path}")

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows_iter = sheet.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            raise SampleDataError(f"{path.name} is empty")

        columns = [str(value).strip() if value is not None else "" for value in header]
        missing = [name for name in required if name not in columns]
        if missing:
            raise SampleDataError(
                f"{path.name} is missing required column(s): {', '.join(missing)}"
            )
        indexes = {name: columns.index(name) for name in required}

        rows: list[dict[str, Any]] = []
        for values in rows_iter:
            if all(value is None or not str(value).strip() for value in values):
                continue
            rows.append({name: values[index] for name, index in indexes.items()})
        return rows
    finally:
        workbook.close()


def _require_text(value: Any, column: str, source: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise SampleDataError(f"{source}: column {column!r} has an empty value")
    return text


def _require_int(value: Any, column: str, source: object) -> int:
    if isinstance(value, bool) or value is None:
        raise SampleDataError(f"{source}: column {column!r} must be a number, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SampleDataError(
            f"{source}: column {column!r} must be a number, got {value!r}"
        ) from exc
