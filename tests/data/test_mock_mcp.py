from __future__ import annotations

import time

import openpyxl
import pytest

import app.data.sample_data as sample_data
from app.core.exceptions import EmployeeNotFoundError, SampleDataError
from app.data.mock_mcp_client import get_employee_context
from app.data.mock_mcp_server import get_manager, get_profile, get_team
from app.data.sample_data import TABLE_COLUMNS, TABLE_FILES, get_row, load_table


@pytest.fixture(autouse=True)
def _clear_table_cache():
    """Every test starts with a cold cache so table parsing is observable."""
    sample_data._parsed_tables.clear()
    yield
    sample_data._parsed_tables.clear()


def _write_workbook(path, rows: list[list[object]], columns: list[str]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(columns)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


# --- each MCP operation reads its own table --------------------------------


@pytest.mark.anyio
async def test_profile_manager_and_team_come_from_their_own_tables() -> None:
    assert await get_profile("emp-001") == {"name": "John Doe", "grade": "senior"}
    assert await get_manager("emp-001") == {"manager": "Dwayne Johnson"}
    assert await get_team("emp-001") == {"team_name": "Platform", "team_size": 8}


@pytest.mark.anyio
async def test_other_employees_resolve_too() -> None:
    assert await get_profile("emp-002") == {"name": "Kate Winset", "grade": "junior"}
    assert await get_manager("emp-002") == {"manager": "Angelina Jolie"}
    assert await get_team("emp-002") == {"team_name": "AI", "team_size": 6}


@pytest.mark.anyio
async def test_unknown_employee_is_reported_clearly() -> None:
    with pytest.raises(EmployeeNotFoundError, match="mgr-002"):
        await get_profile("mgr-002")


@pytest.mark.anyio
async def test_employee_context_fans_out_in_parallel() -> None:
    """Three 1s reads must overlap: ~1s total, not ~3s."""
    start = time.monotonic()
    context = await get_employee_context("emp-001")
    elapsed = time.monotonic() - start

    assert context == {
        "name": "John Doe",
        "grade": "senior",
        "manager": "Dwayne Johnson",
        "team_size": 8,
        "team_name": "Platform",
    }
    assert 0.9 <= elapsed < 2.0, f"Took {elapsed:.1f}s - expected ~1s, not ~3s"


@pytest.mark.anyio
async def test_three_tables_are_read_concurrently_not_in_sequence() -> None:
    """Guards the rubric: serial reads would take ~3s."""
    start = time.monotonic()
    await get_profile("emp-003"), await get_manager("emp-003"), await get_team("emp-003")
    serial = time.monotonic() - start

    start = time.monotonic()
    await get_employee_context("emp-003")
    parallel = time.monotonic() - start

    assert serial > 2.5, f"Sequential baseline was unexpectedly fast ({serial:.1f}s)"
    assert parallel < 2.0, f"Parallel read took {parallel:.1f}s"


# --- table parsing ---------------------------------------------------------


def test_table_is_parsed_once_and_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(tmp_path / TABLE_FILES["profile"], [["emp-100", "Test User", "senior"]], ["sub", "cust_name", "grade"])

    assert get_row("profile", "emp-100")["cust_name"] == "Test User"

    (tmp_path / TABLE_FILES["profile"]).unlink()
    # Reading a deleted file still succeeds, so the parsed table was cached.
    assert get_row("profile", "emp-100")["grade"] == "senior"


def test_columns_are_mapped_by_name_not_position(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(
        tmp_path / TABLE_FILES["team"],
        [[4, "Sales", "emp-200"]],
        ["team_size", "team_name", "sub"],
    )

    assert get_row("team", "emp-200") == {"team_size": 4, "team_name": "Sales", "sub": "emp-200"}


def test_header_row_is_not_treated_as_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(tmp_path / TABLE_FILES["profile"], [["emp-300", "A", "senior"]], ["sub", "cust_name", "grade"])

    assert set(load_table("profile")) == {"emp-300"}


def test_blank_rows_are_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(
        tmp_path / TABLE_FILES["profile"],
        [
            ["emp-300", "A", "senior"],
            [None, None, None],
            ["emp-301", "B", "junior"],
        ],
        ["sub", "cust_name", "grade"],
    )

    assert set(load_table("profile")) == {"emp-300", "emp-301"}


def test_missing_required_column_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(tmp_path / TABLE_FILES["profile"], [["emp-400", "A"]], ["sub", "cust_name"])

    with pytest.raises(SampleDataError, match="grade"):
        load_table("profile")


def test_missing_file_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)

    with pytest.raises(SampleDataError, match="not found"):
        load_table("manager")


def test_non_numeric_team_size_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data, "SAMPLE_DATA_DIR", tmp_path)
    _write_workbook(
        tmp_path / TABLE_FILES["team"],
        [["emp-500", "A", "Team", "many"]],
        ["sub", "cust_name", "team_name", "team_size"],
    )

    with pytest.raises(SampleDataError, match="team_size"):
        load_table("team")


def test_unknown_table_name_is_rejected() -> None:
    with pytest.raises(SampleDataError, match="Unknown mock data table"):
        load_table("nope")


def test_every_declared_table_file_exists() -> None:
    for table, filename in TABLE_FILES.items():
        assert sample_data.table_path(table).name == filename
        assert sample_data.table_path(table).exists(), f"{filename} is missing"
        assert TABLE_COLUMNS[table][0] == "sub"


# --- the minted roster must match the data ---------------------------------


def test_every_minted_user_has_a_row_in_every_table() -> None:
    """Regression guard: mint_tokens.py once minted ids the tables had no data for."""
    from mint_tokens import USERS

    for user in USERS:
        for table in TABLE_FILES:
            assert get_row(table, user["sub"])["sub"] == user["sub"], (
                f"{user['sub']} is missing from the {table!r} table"
            )
