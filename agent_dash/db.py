"""SQLite database initialization, schema creation, and query helpers.

This module owns the full database lifecycle for agent_dash:

* :func:`init_db` – creates all tables if they do not exist.
* :func:`get_db` – returns a per-request ``sqlite3.Connection`` stored on
  Flask's ``g`` proxy, creating it on first call.
* :func:`close_db` – teardown function registered with the app to close the
  connection at the end of each request.
* :func:`wire_db` – called from the app factory to register ``close_db`` and
  run ``init_db`` against the configured database path.

Schema overview
---------------
All provider log data is normalised into a single ``usage_logs`` table so
that the metrics layer can query across providers without per-provider
schema knowledge.  A ``providers`` lookup table ensures referential
integrity and supports easy provider enumeration.

Table: ``providers``
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    name        TEXT UNIQUE NOT NULL   -- e.g. 'claude', 'openai', 'gemini'
    display_name TEXT NOT NULL         -- e.g. 'Anthropic Claude'
    created_at  TEXT NOT NULL          -- ISO-8601 timestamp

Table: ``usage_logs``
    id                   INTEGER PRIMARY KEY AUTOINCREMENT
    provider_id          INTEGER NOT NULL REFERENCES providers(id)
    external_id          TEXT            -- provider-assigned request/task id
    logged_at            TEXT NOT NULL   -- ISO-8601 timestamp of the log event
    model                TEXT            -- model variant, e.g. 'gpt-4o'
    task_type            TEXT            -- e.g. 'code_generation', 'review'
    prompt_tokens        INTEGER NOT NULL DEFAULT 0
    completion_tokens    INTEGER NOT NULL DEFAULT 0
    total_tokens         INTEGER NOT NULL DEFAULT 0
    cost_usd             REAL NOT NULL DEFAULT 0.0
    duration_seconds     REAL            -- wall-clock time reported by provider
    status               TEXT NOT NULL DEFAULT 'success'  -- success | error | cancelled
    raw_payload          TEXT            -- JSON blob of original log record
    imported_at          TEXT NOT NULL   -- ISO-8601 timestamp of ingest

Table: ``poll_runs``
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    provider_id  INTEGER NOT NULL REFERENCES providers(id)
    started_at   TEXT NOT NULL
    finished_at  TEXT
    records_fetched INTEGER NOT NULL DEFAULT 0
    error_message   TEXT
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from flask import Flask, g

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL statements
# ---------------------------------------------------------------------------

_DDL_PROVIDERS = """
CREATE TABLE IF NOT EXISTS providers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    display_name TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);
"""

_DDL_USAGE_LOGS = """
CREATE TABLE IF NOT EXISTS usage_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id       INTEGER NOT NULL REFERENCES providers(id),
    external_id       TEXT,
    logged_at         TEXT NOT NULL,
    model             TEXT,
    task_type         TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    duration_seconds  REAL,
    status            TEXT    NOT NULL DEFAULT 'success',
    raw_payload       TEXT,
    imported_at       TEXT    NOT NULL
);
"""

_DDL_POLL_RUNS = """
CREATE TABLE IF NOT EXISTS poll_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id     INTEGER NOT NULL REFERENCES providers(id),
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    records_fetched INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT
);
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_usage_logs_provider_id ON usage_logs (provider_id);",
    "CREATE INDEX IF NOT EXISTS idx_usage_logs_logged_at   ON usage_logs (logged_at);",
    "CREATE INDEX IF NOT EXISTS idx_usage_logs_status      ON usage_logs (status);",
    "CREATE INDEX IF NOT EXISTS idx_poll_runs_provider_id  ON poll_runs  (provider_id);",
]

# Seed data: well-known providers inserted on first init.
_SEED_PROVIDERS: list[tuple[str, str]] = [
    ("claude",  "Anthropic Claude"),
    ("openai",  "OpenAI"),
    ("gemini",  "Google Gemini"),
]


# ---------------------------------------------------------------------------
# Low-level connection factory
# ---------------------------------------------------------------------------

def _open_connection(database_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults.

    * ``detect_types`` enables datetime parsing via column/type affinity.
    * ``row_factory`` is set to :class:`sqlite3.Row` so callers can access
      columns by name as well as index.
    * ``PRAGMA journal_mode=WAL`` reduces write contention for concurrent
      readers (background poller + web requests).
    * ``PRAGMA foreign_keys=ON`` enforces referential integrity.

    Args:
        database_path: Absolute or relative file-system path to the SQLite
            database file. Use ``":memory:"`` for in-memory databases.

    Returns:
        A configured :class:`sqlite3.Connection`.

    Raises:
        sqlite3.OperationalError: If the database file cannot be opened.
    """
    conn = sqlite3.connect(
        database_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(database_path: str) -> None:
    """Create all tables, indexes, and seed data if they do not already exist.

    This function is idempotent: it uses ``CREATE TABLE IF NOT EXISTS`` /
    ``CREATE INDEX IF NOT EXISTS`` so it can be called on every startup
    without data loss.

    Args:
        database_path: Path to the SQLite database file.  Parent directories
            must already exist (the app factory creates them).

    Raises:
        sqlite3.Error: If schema creation fails for any reason.
    """
    logger.info("Initialising database schema at: %s", database_path)
    conn = _open_connection(database_path)
    try:
        with conn:
            conn.execute(_DDL_PROVIDERS)
            conn.execute(_DDL_USAGE_LOGS)
            conn.execute(_DDL_POLL_RUNS)
            for stmt in _DDL_INDEXES:
                conn.execute(stmt)
            _seed_providers(conn)
        logger.info("Database schema ready.")
    except sqlite3.Error as exc:
        logger.error("Failed to initialise database schema: %s", exc)
        raise
    finally:
        conn.close()


def _seed_providers(conn: sqlite3.Connection) -> None:
    """Insert well-known providers if they are not already present.

    Uses ``INSERT OR IGNORE`` to remain idempotent.

    Args:
        conn: An open :class:`sqlite3.Connection` (must be within a
            transaction context).
    """
    now = _utcnow_iso()
    conn.executemany(
        "INSERT OR IGNORE INTO providers (name, display_name, created_at) VALUES (?, ?, ?);",
        [(name, display, now) for name, display in _SEED_PROVIDERS],
    )
    logger.debug("Provider seed data ensured for: %s", [p[0] for p in _SEED_PROVIDERS])


# ---------------------------------------------------------------------------
# Flask per-request connection management
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return the per-request SQLite connection, opening it if necessary.

    The connection is stored on Flask's ``g`` proxy so it is created at most
    once per request and automatically closed by :func:`close_db` at request
    teardown.

    This function **must** be called within an active Flask application
    context so that ``g`` and ``current_app`` are available.

    Returns:
        An open :class:`sqlite3.Connection` with :class:`sqlite3.Row` factory.

    Raises:
        RuntimeError: If called outside a Flask application context.
        sqlite3.OperationalError: If the database file cannot be opened.
    """
    from flask import current_app  # imported here to avoid circular imports

    if "db" not in g:
        database_path: str = current_app.config["DATABASE"]
        logger.debug("Opening database connection to: %s", database_path)
        g.db = _open_connection(database_path)
    return g.db  # type: ignore[return-value]


def close_db(exception: Optional[BaseException] = None) -> None:
    """Close the per-request database connection if one was opened.

    Registered with the Flask application as a teardown function so it is
    called automatically at the end of every request (even on error).

    Args:
        exception: The exception that caused the teardown, or ``None`` for a
            clean teardown.  The argument is required by Flask's teardown API
            but is not used here.
    """
    db: Optional[sqlite3.Connection] = g.pop("db", None)
    if db is not None:
        db.close()
        logger.debug("Database connection closed.")


# ---------------------------------------------------------------------------
# App factory integration
# ---------------------------------------------------------------------------

def wire_db(app: Flask) -> None:
    """Register database lifecycle hooks on the Flask application.

    Call this from the app factory after the ``DATABASE`` config key has been
    set.  This function:

    1. Runs :func:`init_db` to ensure the schema exists.
    2. Registers :func:`close_db` as a teardown function so connections are
       closed after every request.

    Args:
        app: The :class:`flask.Flask` application instance to wire up.

    Raises:
        sqlite3.Error: Propagated from :func:`init_db` if schema creation
            fails.
    """
    database_path: str = app.config["DATABASE"]
    init_db(database_path)
    app.teardown_appcontext(close_db)
    logger.info("Database wired to Flask app (teardown registered).")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query(
    sql: str,
    params: tuple[Any, ...] | list[Any] = (),
) -> list[sqlite3.Row]:
    """Execute a SELECT statement and return all rows.

    Must be called within a Flask application context.

    Args:
        sql: A parameterised SQL SELECT statement.
        params: Positional bind parameters for the statement.

    Returns:
        A list of :class:`sqlite3.Row` objects (may be empty).

    Raises:
        sqlite3.Error: On any database error.
    """
    conn = get_db()
    cursor = conn.execute(sql, params)
    return cursor.fetchall()


def query_one(
    sql: str,
    params: tuple[Any, ...] | list[Any] = (),
) -> Optional[sqlite3.Row]:
    """Execute a SELECT statement and return the first row, or ``None``.

    Must be called within a Flask application context.

    Args:
        sql: A parameterised SQL SELECT statement.
        params: Positional bind parameters for the statement.

    Returns:
        The first :class:`sqlite3.Row`, or ``None`` if no rows matched.

    Raises:
        sqlite3.Error: On any database error.
    """
    conn = get_db()
    cursor = conn.execute(sql, params)
    return cursor.fetchone()


def execute(
    sql: str,
    params: tuple[Any, ...] | list[Any] = (),
) -> sqlite3.Cursor:
    """Execute a single DML or DDL statement within an auto-commit transaction.

    Wraps the execution in a ``with conn:`` context manager so the
    transaction is automatically committed on success or rolled back on
    failure.

    Must be called within a Flask application context.

    Args:
        sql: A parameterised SQL DML/DDL statement.
        params: Positional bind parameters for the statement.

    Returns:
        The :class:`sqlite3.Cursor` produced by the statement (useful for
        ``lastrowid`` on INSERT statements).

    Raises:
        sqlite3.Error: On any database error; the transaction is rolled back.
    """
    conn = get_db()
    with conn:
        cursor = conn.execute(sql, params)
    return cursor


def executemany(
    sql: str,
    param_seq: list[tuple[Any, ...]] | list[list[Any]],
) -> sqlite3.Cursor:
    """Execute a DML statement for each item in *param_seq*.

    All rows are inserted / updated in a single transaction.

    Must be called within a Flask application context.

    Args:
        sql: A parameterised SQL DML statement.
        param_seq: A sequence of parameter tuples/lists.

    Returns:
        The :class:`sqlite3.Cursor` from the final execution.

    Raises:
        sqlite3.Error: On any database error; the whole batch is rolled back.
    """
    conn = get_db()
    with conn:
        cursor = conn.executemany(sql, param_seq)
    return cursor


# ---------------------------------------------------------------------------
# Domain-specific helpers
# ---------------------------------------------------------------------------

def get_provider_id(provider_name: str) -> Optional[int]:
    """Look up the integer primary key for a provider by its canonical name.

    Args:
        provider_name: Lowercase provider name, e.g. ``'claude'``.

    Returns:
        The ``id`` column value, or ``None`` if no matching provider exists.
    """
    row = query_one(
        "SELECT id FROM providers WHERE name = ?;",
        (provider_name.lower(),),
    )
    return int(row["id"]) if row else None


def get_or_create_provider(name: str, display_name: str) -> int:
    """Return the provider id, inserting a new row if the provider is unknown.

    Args:
        name: Lowercase canonical provider name, e.g. ``'claude'``.
        display_name: Human-readable provider name for the UI.

    Returns:
        The integer ``id`` of the provider row.

    Raises:
        sqlite3.Error: On any database error.
    """
    existing = get_provider_id(name)
    if existing is not None:
        return existing

    cursor = execute(
        "INSERT INTO providers (name, display_name, created_at) VALUES (?, ?, ?);",
        (name.lower(), display_name, _utcnow_iso()),
    )
    logger.info("Registered new provider: %s (%s)", name, display_name)
    return cursor.lastrowid  # type: ignore[return-value]


def insert_usage_log(
    provider_id: int,
    logged_at: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    external_id: Optional[str] = None,
    model: Optional[str] = None,
    task_type: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    status: str = "success",
    raw_payload: Optional[dict[str, Any]] = None,
) -> int:
    """Insert a single normalised usage log record.

    Args:
        provider_id: FK reference to ``providers.id``.
        logged_at: ISO-8601 timestamp string of when the event occurred at
            the provider (not the import time).
        prompt_tokens: Number of tokens in the prompt / input.
        completion_tokens: Number of tokens in the completion / output.
        total_tokens: Total token count.  If zero, computed as
            ``prompt_tokens + completion_tokens``.
        cost_usd: Estimated cost in US dollars.
        external_id: Provider-assigned request or task identifier.
        model: Model variant string, e.g. ``'claude-3-5-sonnet-20241022'``.
        task_type: Categorical task type, e.g. ``'code_generation'``.
        duration_seconds: Wall-clock duration reported by the provider.
        status: One of ``'success'``, ``'error'``, or ``'cancelled'``.
        raw_payload: Original log record as a dict; stored as JSON text.

    Returns:
        The ``id`` (``lastrowid``) of the newly inserted row.

    Raises:
        sqlite3.Error: On any database error.
    """
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    raw_json: Optional[str] = json.dumps(raw_payload) if raw_payload is not None else None

    cursor = execute(
        """
        INSERT INTO usage_logs (
            provider_id, external_id, logged_at, model, task_type,
            prompt_tokens, completion_tokens, total_tokens,
            cost_usd, duration_seconds, status, raw_payload, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            provider_id,
            external_id,
            logged_at,
            model,
            task_type,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            duration_seconds,
            status,
            raw_json,
            _utcnow_iso(),
        ),
    )
    return cursor.lastrowid  # type: ignore[return-value]


def insert_usage_logs_batch(
    records: list[dict[str, Any]],
) -> int:
    """Insert multiple normalised usage log records in a single transaction.

    Each dict in *records* must contain at least ``provider_id`` and
    ``logged_at``.  All other fields are optional and fall back to their
    column defaults.

    Args:
        records: List of dicts, each representing one log row.  The keys
            should match the column names of ``usage_logs``.

    Returns:
        The number of rows inserted.

    Raises:
        KeyError: If a required field (``provider_id``, ``logged_at``) is
            missing from any record.
        sqlite3.Error: On any database error; the whole batch is rolled back.
    """
    now = _utcnow_iso()

    def _to_param(rec: dict[str, Any]) -> tuple[Any, ...]:
        prompt = int(rec.get("prompt_tokens", 0))
        completion = int(rec.get("completion_tokens", 0))
        total = int(rec.get("total_tokens", 0)) or (prompt + completion)
        raw = rec.get("raw_payload")
        raw_json = json.dumps(raw) if isinstance(raw, dict) else raw
        return (
            rec["provider_id"],
            rec.get("external_id"),
            rec["logged_at"],
            rec.get("model"),
            rec.get("task_type"),
            prompt,
            completion,
            total,
            float(rec.get("cost_usd", 0.0)),
            rec.get("duration_seconds"),
            rec.get("status", "success"),
            raw_json,
            now,
        )

    params = [_to_param(r) for r in records]
    cursor = executemany(
        """
        INSERT INTO usage_logs (
            provider_id, external_id, logged_at, model, task_type,
            prompt_tokens, completion_tokens, total_tokens,
            cost_usd, duration_seconds, status, raw_payload, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        params,
    )
    logger.info("Batch inserted %d usage log records.", len(records))
    return len(records)


def record_poll_run(
    provider_id: int,
    started_at: str,
    finished_at: Optional[str] = None,
    records_fetched: int = 0,
    error_message: Optional[str] = None,
) -> int:
    """Record the outcome of a background API poll run.

    Args:
        provider_id: FK reference to ``providers.id``.
        started_at: ISO-8601 timestamp when the poll attempt began.
        finished_at: ISO-8601 timestamp when the poll attempt ended, or
            ``None`` if it is still in progress.
        records_fetched: Number of log records successfully retrieved.
        error_message: Error detail string if the poll failed, else ``None``.

    Returns:
        The ``id`` of the newly inserted ``poll_runs`` row.

    Raises:
        sqlite3.Error: On any database error.
    """
    cursor = execute(
        """
        INSERT INTO poll_runs
            (provider_id, started_at, finished_at, records_fetched, error_message)
        VALUES (?, ?, ?, ?, ?);
        """,
        (provider_id, started_at, finished_at, records_fetched, error_message),
    )
    return cursor.lastrowid  # type: ignore[return-value]


def get_all_providers() -> list[sqlite3.Row]:
    """Return all rows from the ``providers`` table.

    Returns:
        A list of :class:`sqlite3.Row` objects with columns
        ``id``, ``name``, ``display_name``, ``created_at``.
    """
    return query("SELECT id, name, display_name, created_at FROM providers ORDER BY name;")


def get_usage_logs(
    provider_id: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[sqlite3.Row]:
    """Retrieve usage log rows with optional filtering.

    Args:
        provider_id: Filter to a specific provider; ``None`` returns all.
        since: ISO-8601 lower bound for ``logged_at`` (inclusive).
        until: ISO-8601 upper bound for ``logged_at`` (inclusive).
        limit: Maximum number of rows to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        A list of :class:`sqlite3.Row` objects ordered by ``logged_at`` DESC.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if provider_id is not None:
        clauses.append("provider_id = ?")
        params.append(provider_id)
    if since is not None:
        clauses.append("logged_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("logged_at <= ?")
        params.append(until)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT ul.*, p.name AS provider_name, p.display_name AS provider_display_name
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        ORDER BY ul.logged_at DESC
        LIMIT ? OFFSET ?;
    """
    params.extend([limit, offset])
    return query(sql, params)


def count_usage_logs(
    provider_id: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> int:
    """Return the total count of usage log rows matching the given filters.

    Args:
        provider_id: Filter to a specific provider; ``None`` counts all.
        since: ISO-8601 lower bound for ``logged_at`` (inclusive).
        until: ISO-8601 upper bound for ``logged_at`` (inclusive).

    Returns:
        Integer count of matching rows.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if provider_id is not None:
        clauses.append("provider_id = ?")
        params.append(provider_id)
    if since is not None:
        clauses.append("logged_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("logged_at <= ?")
        params.append(until)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT COUNT(*) AS cnt FROM usage_logs {where};"
    row = query_one(sql, params)
    return int(row["cnt"]) if row else 0


def delete_usage_logs_for_provider(provider_id: int) -> int:
    """Delete all usage log rows for a specific provider.

    Intended for use during testing or data reset operations.

    Args:
        provider_id: The provider whose logs should be deleted.

    Returns:
        The number of rows deleted.
    """
    conn = get_db()
    with conn:
        cursor = conn.execute(
            "DELETE FROM usage_logs WHERE provider_id = ?;",
            (provider_id,),
        )
    logger.info(
        "Deleted %d usage_logs rows for provider_id=%d.",
        cursor.rowcount,
        provider_id,
    )
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        A string of the form ``'2024-01-15T12:34:56.789012+00:00'``.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Convert a list of :class:`sqlite3.Row` objects to plain dicts.

    Useful when serialising query results to JSON.

    Args:
        rows: Query result rows from any helper function in this module.

    Returns:
        A list of dicts mapping column names to values.
    """
    return [dict(row) for row in rows]
