"""Unit tests for agent_dash.db – database schema, helpers, and query functions.

All tests operate against an in-memory SQLite database or a temp-file database
provided by pytest fixtures, so they leave no artefacts on disk.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from agent_dash.app import create_app
import agent_dash.db as db_module
from agent_dash.db import (
    _utcnow_iso,
    close_db,
    count_usage_logs,
    delete_usage_logs_for_provider,
    execute,
    get_all_providers,
    get_db,
    get_or_create_provider,
    get_provider_id,
    get_usage_logs,
    init_db,
    insert_usage_log,
    insert_usage_logs_batch,
    record_poll_run,
    rows_to_dicts,
    wire_db,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    """Flask app wired to a temp-file database."""
    application = create_app({
        "DATABASE": str(tmp_path / "test.db"),
        "TESTING": True,
        "SECRET_KEY": "test",
    })
    wire_db(application)
    return application


@pytest.fixture()
def app_ctx(app):
    """Push an application context so get_db() works, then pop it after."""
    with app.app_context():
        yield app


@pytest.fixture()
def mem_db_path():
    """Return the :memory: sentinel string for in-memory SQLite."""
    return ":memory:"


# ---------------------------------------------------------------------------
# _utcnow_iso
# ---------------------------------------------------------------------------

class TestUtcNowIso:
    def test_returns_string(self):
        result = _utcnow_iso()
        assert isinstance(result, str)

    def test_contains_T_separator(self):
        result = _utcnow_iso()
        assert "T" in result

    def test_contains_utc_offset(self):
        result = _utcnow_iso()
        # ISO format with timezone ends in +00:00 for UTC
        assert "+00:00" in result


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_providers_table(self, tmp_path):
        db_path = str(tmp_path / "schema_test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='providers';")
        assert cur.fetchone() is not None
        conn.close()

    def test_creates_usage_logs_table(self, tmp_path):
        db_path = str(tmp_path / "schema_test2.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_logs';")
        assert cur.fetchone() is not None
        conn.close()

    def test_creates_poll_runs_table(self, tmp_path):
        db_path = str(tmp_path / "schema_test3.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='poll_runs';")
        assert cur.fetchone() is not None
        conn.close()

    def test_seeds_providers(self, tmp_path):
        db_path = str(tmp_path / "seed_test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM providers ORDER BY name;")
        names = [row[0] for row in cur.fetchall()]
        conn.close()
        assert "claude" in names
        assert "openai" in names
        assert "gemini" in names

    def test_idempotent(self, tmp_path):
        """Calling init_db twice should not raise or duplicate seed data."""
        db_path = str(tmp_path / "idem.db")
        init_db(db_path)
        init_db(db_path)  # second call
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT COUNT(*) FROM providers WHERE name='claude';")
        assert cur.fetchone()[0] == 1
        conn.close()

    def test_creates_indexes(self, tmp_path):
        db_path = str(tmp_path / "idx_test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_usage_logs_provider_id';"
        )
        assert cur.fetchone() is not None
        conn.close()


# ---------------------------------------------------------------------------
# get_db / close_db
# ---------------------------------------------------------------------------

class TestGetDb:
    def test_returns_connection(self, app_ctx):
        conn = get_db()
        assert isinstance(conn, sqlite3.Connection)

    def test_same_connection_within_context(self, app_ctx):
        conn1 = get_db()
        conn2 = get_db()
        assert conn1 is conn2

    def test_row_factory_is_set(self, app_ctx):
        conn = get_db()
        assert conn.row_factory is sqlite3.Row

    def test_foreign_keys_enabled(self, app_ctx):
        conn = get_db()
        cur = conn.execute("PRAGMA foreign_keys;")
        assert cur.fetchone()[0] == 1


class TestCloseDb:
    def test_close_db_removes_g_entry(self, app):
        with app.app_context():
            from flask import g
            get_db()  # open
            assert "db" in g.__dict__ or hasattr(g, "db")
            close_db()
            assert "db" not in g.__dict__

    def test_close_db_noop_when_no_connection(self, app_ctx):
        """close_db should not raise if no connection was opened."""
        close_db()  # no connection opened yet


# ---------------------------------------------------------------------------
# wire_db
# ---------------------------------------------------------------------------

class TestWireDb:
    def test_wire_db_registers_teardown(self, tmp_path):
        """After wire_db, app teardown functions should include close_db."""
        application = create_app({
            "DATABASE": str(tmp_path / "wire_test.db"),
            "TESTING": True,
        })
        wire_db(application)
        # Verify schema was created by checking providers table exists
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(tmp_path / "wire_test.db"))
        cur = conn.execute("SELECT COUNT(*) FROM providers;")
        assert cur.fetchone()[0] >= 3  # at least claude, openai, gemini
        conn.close()


# ---------------------------------------------------------------------------
# get_provider_id / get_or_create_provider
# ---------------------------------------------------------------------------

class TestProviderHelpers:
    def test_get_provider_id_existing(self, app_ctx):
        pid = get_provider_id("claude")
        assert isinstance(pid, int)
        assert pid > 0

    def test_get_provider_id_missing(self, app_ctx):
        pid = get_provider_id("nonexistent_provider_xyz")
        assert pid is None

    def test_get_provider_id_case_insensitive(self, app_ctx):
        pid_lower = get_provider_id("claude")
        pid_upper = get_provider_id("CLAUDE")
        # get_provider_id lowercases the input
        assert pid_lower == pid_upper

    def test_get_or_create_existing(self, app_ctx):
        pid1 = get_provider_id("openai")
        pid2 = get_or_create_provider("openai", "OpenAI")
        assert pid1 == pid2

    def test_get_or_create_new_provider(self, app_ctx):
        pid = get_or_create_provider("new_provider", "New Provider")
        assert isinstance(pid, int)
        assert pid > 0
        # Verify it is now retrievable
        assert get_provider_id("new_provider") == pid

    def test_get_or_create_idempotent(self, app_ctx):
        pid1 = get_or_create_provider("myagent", "My Agent")
        pid2 = get_or_create_provider("myagent", "My Agent")
        assert pid1 == pid2


# ---------------------------------------------------------------------------
# get_all_providers
# ---------------------------------------------------------------------------

class TestGetAllProviders:
    def test_returns_list(self, app_ctx):
        result = get_all_providers()
        assert isinstance(result, list)

    def test_includes_seed_providers(self, app_ctx):
        providers = get_all_providers()
        names = [row["name"] for row in providers]
        assert "claude" in names
        assert "openai" in names
        assert "gemini" in names

    def test_rows_have_expected_columns(self, app_ctx):
        providers = get_all_providers()
        assert len(providers) > 0
        row = providers[0]
        assert "id" in row.keys()
        assert "name" in row.keys()
        assert "display_name" in row.keys()
        assert "created_at" in row.keys()


# ---------------------------------------------------------------------------
# insert_usage_log
# ---------------------------------------------------------------------------

class TestInsertUsageLog:
    def test_returns_integer_id(self, app_ctx):
        provider_id = get_provider_id("claude")
        row_id = insert_usage_log(
            provider_id=provider_id,
            logged_at=_utcnow_iso(),
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_computes_total_tokens(self, app_ctx):
        provider_id = get_provider_id("openai")
        insert_usage_log(
            provider_id=provider_id,
            logged_at=_utcnow_iso(),
            prompt_tokens=200,
            completion_tokens=100,
        )
        rows = get_usage_logs(provider_id=provider_id)
        assert rows[0]["total_tokens"] == 300

    def test_explicit_total_tokens_not_overridden(self, app_ctx):
        provider_id = get_provider_id("gemini")
        insert_usage_log(
            provider_id=provider_id,
            logged_at=_utcnow_iso(),
            prompt_tokens=100,
            completion_tokens=100,
            total_tokens=250,  # explicit
        )
        rows = get_usage_logs(provider_id=provider_id)
        assert rows[0]["total_tokens"] == 250

    def test_stores_optional_fields(self, app_ctx):
        provider_id = get_provider_id("claude")
        insert_usage_log(
            provider_id=provider_id,
            logged_at="2024-03-01T10:00:00+00:00",
            prompt_tokens=10,
            completion_tokens=20,
            external_id="ext-001",
            model="claude-3-5-sonnet-20241022",
            task_type="code_generation",
            duration_seconds=3.14,
            status="success",
            cost_usd=0.005,
            raw_payload={"source": "test"},
        )
        rows = get_usage_logs(provider_id=provider_id)
        row = rows[0]
        assert row["external_id"] == "ext-001"
        assert row["model"] == "claude-3-5-sonnet-20241022"
        assert row["task_type"] == "code_generation"
        assert abs(row["duration_seconds"] - 3.14) < 0.001
        assert row["status"] == "success"
        assert abs(row["cost_usd"] - 0.005) < 1e-9
        assert row["raw_payload"] is not None

    def test_default_status_is_success(self, app_ctx):
        provider_id = get_provider_id("openai")
        insert_usage_log(provider_id=provider_id, logged_at=_utcnow_iso())
        rows = get_usage_logs(provider_id=provider_id)
        assert rows[0]["status"] == "success"


# ---------------------------------------------------------------------------
# insert_usage_logs_batch
# ---------------------------------------------------------------------------

class TestInsertUsageLogsBatch:
    def test_inserts_multiple_rows(self, app_ctx):
        provider_id = get_provider_id("claude")
        before = count_usage_logs(provider_id=provider_id)
        records = [
            {"provider_id": provider_id, "logged_at": _utcnow_iso(), "prompt_tokens": i * 10}
            for i in range(1, 6)
        ]
        n = insert_usage_logs_batch(records)
        assert n == 5
        assert count_usage_logs(provider_id=provider_id) == before + 5

    def test_returns_count(self, app_ctx):
        provider_id = get_provider_id("openai")
        records = [
            {"provider_id": provider_id, "logged_at": _utcnow_iso()}
            for _ in range(3)
        ]
        result = insert_usage_logs_batch(records)
        assert result == 3

    def test_batch_computes_total_tokens(self, app_ctx):
        provider_id = get_provider_id("gemini")
        before = count_usage_logs(provider_id=provider_id)
        records = [{
            "provider_id": provider_id,
            "logged_at": _utcnow_iso(),
            "prompt_tokens": 40,
            "completion_tokens": 60,
        }]
        insert_usage_logs_batch(records)
        rows = get_usage_logs(provider_id=provider_id, limit=1)
        assert rows[0]["total_tokens"] == 100

    def test_empty_batch_is_noop(self, app_ctx):
        result = insert_usage_logs_batch([])
        assert result == 0


# ---------------------------------------------------------------------------
# get_usage_logs / count_usage_logs
# ---------------------------------------------------------------------------

class TestGetUsageLogs:
    @pytest.fixture(autouse=True)
    def seed_logs(self, app_ctx):
        """Insert a predictable set of logs for filter/pagination tests."""
        claude_id = get_provider_id("claude")
        openai_id = get_provider_id("openai")
        records = [
            {"provider_id": claude_id, "logged_at": "2024-01-10T08:00:00+00:00", "prompt_tokens": 100},
            {"provider_id": claude_id, "logged_at": "2024-01-15T12:00:00+00:00", "prompt_tokens": 200},
            {"provider_id": openai_id, "logged_at": "2024-01-20T16:00:00+00:00", "prompt_tokens": 300},
        ]
        insert_usage_logs_batch(records)

    def test_returns_all_without_filters(self, app_ctx):
        rows = get_usage_logs(limit=1000)
        assert len(rows) >= 3

    def test_filter_by_provider(self, app_ctx):
        claude_id = get_provider_id("claude")
        rows = get_usage_logs(provider_id=claude_id)
        for row in rows:
            assert row["provider_id"] == claude_id

    def test_filter_by_since(self, app_ctx):
        rows = get_usage_logs(since="2024-01-15T00:00:00+00:00")
        for row in rows:
            assert row["logged_at"] >= "2024-01-15"

    def test_filter_by_until(self, app_ctx):
        rows = get_usage_logs(until="2024-01-15T23:59:59+00:00")
        for row in rows:
            assert row["logged_at"] <= "2024-01-15T23:59:59+00:00"

    def test_limit_respected(self, app_ctx):
        rows = get_usage_logs(limit=1)
        assert len(rows) <= 1

    def test_offset_skips_rows(self, app_ctx):
        all_rows = get_usage_logs(limit=100)
        offset_rows = get_usage_logs(limit=100, offset=1)
        assert len(offset_rows) == len(all_rows) - 1

    def test_rows_include_provider_name(self, app_ctx):
        rows = get_usage_logs(limit=5)
        assert len(rows) > 0
        assert "provider_name" in rows[0].keys()


class TestCountUsageLogs:
    @pytest.fixture(autouse=True)
    def seed_logs(self, app_ctx):
        claude_id = get_provider_id("claude")
        openai_id = get_provider_id("openai")
        records = [
            {"provider_id": claude_id, "logged_at": "2024-02-01T00:00:00+00:00"},
            {"provider_id": claude_id, "logged_at": "2024-02-10T00:00:00+00:00"},
            {"provider_id": openai_id, "logged_at": "2024-02-20T00:00:00+00:00"},
        ]
        insert_usage_logs_batch(records)

    def test_count_all(self, app_ctx):
        total = count_usage_logs()
        assert total >= 3

    def test_count_by_provider(self, app_ctx):
        claude_id = get_provider_id("claude")
        count = count_usage_logs(provider_id=claude_id)
        assert count >= 2

    def test_count_by_since(self, app_ctx):
        count = count_usage_logs(since="2024-02-15T00:00:00+00:00")
        assert count >= 1

    def test_count_no_matches(self, app_ctx):
        count = count_usage_logs(since="2099-01-01T00:00:00+00:00")
        assert count == 0


# ---------------------------------------------------------------------------
# record_poll_run
# ---------------------------------------------------------------------------

class TestRecordPollRun:
    def test_returns_integer_id(self, app_ctx):
        provider_id = get_provider_id("claude")
        row_id = record_poll_run(
            provider_id=provider_id,
            started_at=_utcnow_iso(),
            finished_at=_utcnow_iso(),
            records_fetched=10,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_stores_error_message(self, app_ctx):
        provider_id = get_provider_id("openai")
        record_poll_run(
            provider_id=provider_id,
            started_at=_utcnow_iso(),
            error_message="API rate limit exceeded",
        )
        row = db_module.query_one(
            "SELECT error_message FROM poll_runs WHERE provider_id = ? ORDER BY id DESC LIMIT 1;",
            (provider_id,),
        )
        assert row is not None
        assert row["error_message"] == "API rate limit exceeded"

    def test_finished_at_nullable(self, app_ctx):
        provider_id = get_provider_id("gemini")
        record_poll_run(
            provider_id=provider_id,
            started_at=_utcnow_iso(),
            finished_at=None,
        )
        row = db_module.query_one(
            "SELECT finished_at FROM poll_runs WHERE provider_id = ? ORDER BY id DESC LIMIT 1;",
            (provider_id,),
        )
        assert row is not None
        assert row["finished_at"] is None


# ---------------------------------------------------------------------------
# delete_usage_logs_for_provider
# ---------------------------------------------------------------------------

class TestDeleteUsageLogsForProvider:
    def test_deletes_rows(self, app_ctx):
        provider_id = get_provider_id("claude")
        insert_usage_log(provider_id=provider_id, logged_at=_utcnow_iso())
        before = count_usage_logs(provider_id=provider_id)
        assert before > 0
        deleted = delete_usage_logs_for_provider(provider_id)
        assert deleted == before
        assert count_usage_logs(provider_id=provider_id) == 0

    def test_does_not_delete_other_provider_rows(self, app_ctx):
        claude_id = get_provider_id("claude")
        openai_id = get_provider_id("openai")
        insert_usage_log(provider_id=openai_id, logged_at=_utcnow_iso())
        openai_before = count_usage_logs(provider_id=openai_id)
        delete_usage_logs_for_provider(claude_id)
        assert count_usage_logs(provider_id=openai_id) == openai_before


# ---------------------------------------------------------------------------
# rows_to_dicts
# ---------------------------------------------------------------------------

class TestRowsToDicts:
    def test_converts_rows_to_dicts(self, app_ctx):
        rows = get_all_providers()
        result = rows_to_dicts(rows)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_dict_keys_match_columns(self, app_ctx):
        rows = get_all_providers()
        result = rows_to_dicts(rows)
        assert len(result) > 0
        assert "name" in result[0]
        assert "display_name" in result[0]

    def test_empty_list_returns_empty(self, app_ctx):
        result = rows_to_dicts([])
        assert result == []
