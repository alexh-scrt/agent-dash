"""Log ingestion and normalization for agent_dash.

This module parses and normalizes CSV and JSON log exports from Claude
(Anthropic), OpenAI, and Gemini into the unified ``usage_logs`` schema
defined in :mod:`agent_dash.db`.

Public API
----------
* :func:`ingest_file` – detect format/provider and dispatch to the correct
  parser, returning the number of records inserted.
* :func:`ingest_csv` – parse a CSV file and normalize into unified records.
* :func:`ingest_json` – parse a JSON file and normalize into unified records.
* :func:`normalize_claude_record` – normalize a single Claude log entry.
* :func:`normalize_openai_record` – normalize a single OpenAI log entry.
* :func:`normalize_gemini_record` – normalize a single Gemini log entry.

Provider detection
------------------
Provider auto-detection uses heuristics based on CSV column names and JSON
field presence.  Callers may also pass the provider name explicitly to
skip detection.

Schema contract
---------------
All normalization functions return a dict with the following keys (matching
``insert_usage_logs_batch`` expectations)::

    provider_id       int   – FK to providers table (required)
    logged_at         str   – ISO-8601 timestamp (required)
    external_id       str | None
    model             str | None
    task_type         str | None
    prompt_tokens     int
    completion_tokens int
    total_tokens      int
    cost_usd          float
    duration_seconds  float | None
    status            str   – 'success' | 'error' | 'cancelled'
    raw_payload       dict  – original record for audit trail
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical provider names (must match providers.name in the DB)
PROVIDER_CLAUDE = "claude"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

KNOWN_PROVIDERS = {PROVIDER_CLAUDE, PROVIDER_OPENAI, PROVIDER_GEMINI}

# Column-name fingerprints used for CSV provider detection.
# If ANY of the listed columns are present the provider matches.
_CLAUDE_CSV_COLUMNS = {"input_tokens", "output_tokens", "stop_reason", "anthropic_model"}
_OPENAI_CSV_COLUMNS = {"prompt_tokens", "completion_tokens", "finish_reason", "object"}
_GEMINI_CSV_COLUMNS = {"candidates_token_count", "prompt_token_count", "finish_reason", "safety_ratings"}

# JSON field fingerprints
_CLAUDE_JSON_FIELDS = {"input_tokens", "output_tokens", "stop_reason", "model"}
_OPENAI_JSON_FIELDS = {"prompt_tokens", "completion_tokens", "finish_reason", "object"}
_GEMINI_JSON_FIELDS = {"candidates_token_count", "prompt_token_count", "usageMetadata"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestError(Exception):
    """Raised when log ingestion fails due to bad input or unknown provider."""


class UnknownProviderError(IngestError):
    """Raised when provider cannot be detected from the file contents."""


class MalformedLogError(IngestError):
    """Raised when a log file cannot be parsed (bad CSV, invalid JSON, etc.)."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_file(
    filepath: str | Path,
    provider: Optional[str] = None,
    file_format: Optional[str] = None,
) -> int:
    """Ingest a CSV or JSON log file and store normalised records in the DB.

    This function reads the file, detects the provider (if not supplied),
    normalises each log record, and calls
    :func:`agent_dash.db.insert_usage_logs_batch` to persist them in a
    single transaction.

    Must be called within an active Flask application context so that the
    database helpers can access ``flask.g``.

    Args:
        filepath: Path to the log file on disk.
        provider: Optional provider override (``'claude'``, ``'openai'``,
            or ``'gemini'``).  If ``None``, the provider is auto-detected
            from the file contents.
        file_format: Optional format override (``'csv'`` or ``'json'``).
            If ``None``, the format is inferred from the file extension.

    Returns:
        The number of records successfully inserted.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        UnknownProviderError: If provider cannot be determined.
        MalformedLogError: If the file cannot be parsed.
        IngestError: For any other ingestion failure.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    # Determine format from extension if not given
    if file_format is None:
        ext = path.suffix.lower().lstrip(".")
        if ext in ("csv",):
            file_format = "csv"
        elif ext in ("json",):
            file_format = "json"
        else:
            raise MalformedLogError(
                f"Cannot determine file format from extension '.{ext}'. "
                "Pass file_format='csv' or 'json' explicitly."
            )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"Cannot read log file {filepath}: {exc}") from exc

    if file_format == "csv":
        return ingest_csv(raw_bytes, provider=provider)
    elif file_format == "json":
        return ingest_json(raw_bytes, provider=provider)
    else:
        raise MalformedLogError(f"Unsupported file format: {file_format!r}")


def ingest_csv(
    data: bytes | str,
    provider: Optional[str] = None,
) -> int:
    """Parse CSV log data and persist normalised records.

    Args:
        data: Raw CSV content as bytes or a string.
        provider: Optional provider override.  If ``None``, provider is
            detected from column names.

    Returns:
        The number of records inserted.

    Raises:
        UnknownProviderError: If provider cannot be detected.
        MalformedLogError: If the CSV cannot be parsed.
    """
    try:
        if isinstance(data, bytes):
            df = pd.read_csv(io.BytesIO(data))
        else:
            df = pd.read_csv(io.StringIO(data))
    except Exception as exc:
        raise MalformedLogError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        logger.warning("CSV file is empty; nothing to ingest.")
        return 0

    columns = {c.lower().strip() for c in df.columns}
    df.columns = [c.lower().strip() for c in df.columns]

    detected_provider = provider or _detect_provider_from_columns(columns)
    if detected_provider is None:
        raise UnknownProviderError(
            f"Cannot detect provider from CSV columns: {sorted(columns)}. "
            "Pass provider='claude'|'openai'|'gemini' explicitly."
        )
    detected_provider = detected_provider.lower()

    records = df.to_dict(orient="records")
    return _normalize_and_insert(records, detected_provider, source_format="csv")


def ingest_json(
    data: bytes | str,
    provider: Optional[str] = None,
) -> int:
    """Parse JSON log data and persist normalised records.

    The JSON may be:
    * A JSON array of log objects: ``[{...}, {...}]``
    * A JSON object with a ``'data'``, ``'items'``, ``'logs'``, or
      ``'usage'`` key containing the array.
    * A single JSON object representing one log record.

    Args:
        data: Raw JSON content as bytes or a string.
        provider: Optional provider override.  If ``None``, provider is
            detected from field names.

    Returns:
        The number of records inserted.

    Raises:
        UnknownProviderError: If provider cannot be detected.
        MalformedLogError: If the JSON cannot be parsed.
    """
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedLogError(f"Failed to parse JSON: {exc}") from exc

    records = _unwrap_json_records(parsed)
    if not records:
        logger.warning("JSON file contains no records; nothing to ingest.")
        return 0

    # Detect provider from first record
    detected_provider = provider or _detect_provider_from_json_record(records[0])
    if detected_provider is None:
        fields = set(records[0].keys()) if records else set()
        raise UnknownProviderError(
            f"Cannot detect provider from JSON fields: {sorted(fields)}. "
            "Pass provider='claude'|'openai'|'gemini' explicitly."
        )
    detected_provider = detected_provider.lower()

    return _normalize_and_insert(records, detected_provider, source_format="json")


def ingest_records(
    records: list[dict[str, Any]],
    provider: str,
) -> int:
    """Normalize and insert pre-parsed records for a known provider.

    Use this when log records have already been parsed into dicts (e.g. from
    an API response) and just need normalization + DB insertion.

    Args:
        records: List of raw log record dicts.
        provider: Canonical provider name (``'claude'``, ``'openai'``,
            ``'gemini'``).

    Returns:
        The number of records inserted.

    Raises:
        IngestError: If the provider name is unknown.
    """
    if provider.lower() not in KNOWN_PROVIDERS:
        raise IngestError(f"Unknown provider: {provider!r}")
    return _normalize_and_insert(records, provider.lower(), source_format="api")


# ---------------------------------------------------------------------------
# Provider-specific normalizers
# ---------------------------------------------------------------------------

def normalize_claude_record(
    record: dict[str, Any],
    provider_id: int,
) -> dict[str, Any]:
    """Normalize a single Claude (Anthropic) log entry to the unified schema.

    Expected source fields (any subset may be present)::

        id / request_id      – external identifier
        created_at / timestamp / date – event timestamp
        model / anthropic_model       – model variant
        task_type / type             – task category
        input_tokens / prompt_tokens  – prompt token count
        output_tokens / completion_tokens – completion token count
        total_tokens                 – total (computed if absent)
        cost / cost_usd / total_cost  – cost in USD
        duration / duration_ms / latency_ms – duration
        stop_reason / status         – completion status

    Args:
        record: Raw log record dict from Claude export.
        provider_id: The integer FK for the Claude provider.

    Returns:
        Normalised record dict ready for :func:`agent_dash.db.insert_usage_logs_batch`.
    """
    r = {k.lower().strip(): v for k, v in record.items()}

    external_id = _coerce_str(
        r.get("id") or r.get("request_id") or r.get("log_id")
    )
    logged_at = _parse_timestamp(
        r.get("created_at") or r.get("timestamp") or r.get("date") or r.get("logged_at")
    )
    model = _coerce_str(
        r.get("model") or r.get("anthropic_model") or r.get("model_id")
    )
    task_type = _coerce_str(
        r.get("task_type") or r.get("type") or r.get("task")
    )
    prompt_tokens = _coerce_int(
        r.get("input_tokens") or r.get("prompt_tokens") or r.get("input_token_count")
    )
    completion_tokens = _coerce_int(
        r.get("output_tokens") or r.get("completion_tokens") or r.get("output_token_count")
    )
    total_tokens = _coerce_int(
        r.get("total_tokens")
    ) or (prompt_tokens + completion_tokens)
    cost_usd = _coerce_float(
        r.get("cost_usd") or r.get("cost") or r.get("total_cost") or r.get("price_usd")
    )
    duration_seconds = _parse_duration(
        r.get("duration_seconds") or r.get("duration") or r.get("duration_ms") or r.get("latency_ms"),
        is_ms=_field_is_ms(r, ["duration_ms", "latency_ms"]),
    )
    status = _normalize_status_claude(
        r.get("stop_reason") or r.get("status") or r.get("finish_reason")
    )

    return {
        "provider_id": provider_id,
        "external_id": external_id,
        "logged_at": logged_at,
        "model": model,
        "task_type": task_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
        "status": status,
        "raw_payload": record,
    }


def normalize_openai_record(
    record: dict[str, Any],
    provider_id: int,
) -> dict[str, Any]:
    """Normalize a single OpenAI log entry to the unified schema.

    Expected source fields (any subset may be present)::

        id                          – external identifier
        created / created_at        – Unix timestamp or ISO string
        model                       – model variant (e.g. 'gpt-4o')
        task_type / type / object   – task category
        usage.prompt_tokens         – nested usage object
        usage.completion_tokens
        usage.total_tokens
        prompt_tokens / completion_tokens / total_tokens – flat variants
        cost / cost_usd / total_cost
        duration / duration_ms
        finish_reason / status

    Args:
        record: Raw log record dict from OpenAI export or API response.
        provider_id: The integer FK for the OpenAI provider.

    Returns:
        Normalised record dict ready for :func:`agent_dash.db.insert_usage_logs_batch`.
    """
    r = {k.lower().strip(): v for k, v in record.items()}

    # OpenAI API responses nest token counts under a 'usage' object
    usage: dict[str, Any] = {}
    if isinstance(r.get("usage"), dict):
        usage = {k.lower(): v for k, v in r["usage"].items()}

    external_id = _coerce_str(r.get("id"))
    # OpenAI 'created' is a Unix timestamp int in API responses
    raw_ts = r.get("created_at") or r.get("created") or r.get("timestamp") or r.get("date")
    logged_at = _parse_timestamp(raw_ts)
    model = _coerce_str(r.get("model"))
    task_type = _coerce_str(
        r.get("task_type") or r.get("type") or r.get("object")
    )
    if task_type and task_type.startswith("chat.completion"):
        task_type = "chat_completion"

    prompt_tokens = _coerce_int(
        usage.get("prompt_tokens") or r.get("prompt_tokens")
    )
    completion_tokens = _coerce_int(
        usage.get("completion_tokens") or r.get("completion_tokens")
    )
    total_tokens = _coerce_int(
        usage.get("total_tokens") or r.get("total_tokens")
    ) or (prompt_tokens + completion_tokens)
    cost_usd = _coerce_float(
        r.get("cost_usd") or r.get("cost") or r.get("total_cost")
    )
    duration_seconds = _parse_duration(
        r.get("duration_seconds") or r.get("duration") or r.get("duration_ms"),
        is_ms=_field_is_ms(r, ["duration_ms"]),
    )
    status = _normalize_status_openai(
        r.get("finish_reason") or r.get("status")
    )

    return {
        "provider_id": provider_id,
        "external_id": external_id,
        "logged_at": logged_at,
        "model": model,
        "task_type": task_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
        "status": status,
        "raw_payload": record,
    }


def normalize_gemini_record(
    record: dict[str, Any],
    provider_id: int,
) -> dict[str, Any]:
    """Normalize a single Gemini (Google) log entry to the unified schema.

    Expected source fields (any subset may be present)::

        name / id / request_id               – external identifier
        create_time / timestamp / date       – event timestamp
        model / model_version / model_id     – model variant
        task_type / type                     – task category
        usageMetadata / usage_metadata       – nested usage object
          .promptTokenCount / prompt_token_count
          .candidatesTokenCount / candidates_token_count
          .totalTokenCount / total_token_count
        prompt_token_count / candidates_token_count (flat)
        cost / cost_usd
        duration / duration_ms
        finish_reason / status

    Args:
        record: Raw log record dict from Gemini export or API response.
        provider_id: The integer FK for the Gemini provider.

    Returns:
        Normalised record dict ready for :func:`agent_dash.db.insert_usage_logs_batch`.
    """
    r = {k.lower().strip(): v for k, v in record.items()}

    # Gemini API responses nest token counts under 'usageMetadata'
    usage: dict[str, Any] = {}
    raw_usage = r.get("usagemetadata") or r.get("usage_metadata") or r.get("usage")
    if isinstance(raw_usage, dict):
        usage = {k.lower(): v for k, v in raw_usage.items()}

    external_id = _coerce_str(
        r.get("name") or r.get("id") or r.get("request_id")
    )
    raw_ts = r.get("create_time") or r.get("created_at") or r.get("timestamp") or r.get("date")
    logged_at = _parse_timestamp(raw_ts)
    model = _coerce_str(
        r.get("model") or r.get("model_version") or r.get("model_id")
    )
    task_type = _coerce_str(r.get("task_type") or r.get("type"))

    prompt_tokens = _coerce_int(
        usage.get("prompttokencount")
        or usage.get("prompt_token_count")
        or r.get("prompt_token_count")
        or r.get("prompt_tokens")
    )
    completion_tokens = _coerce_int(
        usage.get("candidatestokencount")
        or usage.get("candidates_token_count")
        or r.get("candidates_token_count")
        or r.get("completion_tokens")
    )
    total_tokens = _coerce_int(
        usage.get("totaltokencount")
        or usage.get("total_token_count")
        or r.get("total_token_count")
        or r.get("total_tokens")
    ) or (prompt_tokens + completion_tokens)
    cost_usd = _coerce_float(r.get("cost_usd") or r.get("cost"))
    duration_seconds = _parse_duration(
        r.get("duration_seconds") or r.get("duration") or r.get("duration_ms"),
        is_ms=_field_is_ms(r, ["duration_ms"]),
    )
    status = _normalize_status_gemini(
        r.get("finish_reason") or r.get("status")
    )

    return {
        "provider_id": provider_id,
        "external_id": external_id,
        "logged_at": logged_at,
        "model": model,
        "task_type": task_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
        "status": status,
        "raw_payload": record,
    }


# ---------------------------------------------------------------------------
# Internal: dispatch
# ---------------------------------------------------------------------------

def _normalize_and_insert(
    records: list[dict[str, Any]],
    provider: str,
    source_format: str = "unknown",
) -> int:
    """Normalize raw records for *provider* and batch-insert into the DB.

    Args:
        records: List of raw log record dicts.
        provider: Canonical provider name.
        source_format: One of ``'csv'``, ``'json'``, ``'api'`` – used for
            logging only.

    Returns:
        Number of records inserted.

    Raises:
        IngestError: If the provider name is unrecognised.
    """
    from agent_dash.db import get_or_create_provider, insert_usage_logs_batch

    display_names = {
        PROVIDER_CLAUDE: "Anthropic Claude",
        PROVIDER_OPENAI: "OpenAI",
        PROVIDER_GEMINI: "Google Gemini",
    }
    display_name = display_names.get(provider, provider.title())
    provider_id = get_or_create_provider(provider, display_name)

    normalizer = {
        PROVIDER_CLAUDE: normalize_claude_record,
        PROVIDER_OPENAI: normalize_openai_record,
        PROVIDER_GEMINI: normalize_gemini_record,
    }.get(provider)

    if normalizer is None:
        raise IngestError(f"No normalizer registered for provider: {provider!r}")

    normalised: list[dict[str, Any]] = []
    skipped = 0
    for i, raw in enumerate(records):
        try:
            norm = normalizer(raw, provider_id)
            normalised.append(norm)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping record %d from %s (%s): %s",
                i, provider, source_format, exc,
            )
            skipped += 1

    if skipped:
        logger.warning(
            "Skipped %d/%d records during %s ingest (%s).",
            skipped, len(records), provider, source_format,
        )

    if not normalised:
        logger.info("No valid records to insert for provider=%s.", provider)
        return 0

    count = insert_usage_logs_batch(normalised)
    logger.info(
        "Ingested %d records for provider=%s (format=%s).",
        count, provider, source_format,
    )
    return count


# ---------------------------------------------------------------------------
# Internal: provider detection
# ---------------------------------------------------------------------------

def _detect_provider_from_columns(columns: set[str]) -> Optional[str]:
    """Detect provider from a set of CSV column names.

    Uses a simple scoring approach: the provider whose fingerprint columns
    have the most overlap with *columns* wins.

    Args:
        columns: Lowercase CSV column names.

    Returns:
        Canonical provider name, or ``None`` if no match.
    """
    scores: dict[str, int] = {
        PROVIDER_CLAUDE: len(_CLAUDE_CSV_COLUMNS & columns),
        PROVIDER_OPENAI: len(_OPENAI_CSV_COLUMNS & columns),
        PROVIDER_GEMINI: len(_GEMINI_CSV_COLUMNS & columns),
    }
    best_provider = max(scores, key=lambda k: scores[k])
    if scores[best_provider] == 0:
        return None
    logger.debug("Provider detection scores (CSV): %s -> %s", scores, best_provider)
    return best_provider


def _detect_provider_from_json_record(record: dict[str, Any]) -> Optional[str]:
    """Detect provider from a single JSON record's field names.

    Args:
        record: The first record dict from the JSON file.

    Returns:
        Canonical provider name, or ``None`` if no match.
    """
    fields = {k.lower() for k in record.keys()}

    # Gemini-specific nested key
    if "usagemetadata" in fields or "usage_metadata" in fields or "candidatestokencount" in fields:
        return PROVIDER_GEMINI
    if "candidates_token_count" in fields or "prompt_token_count" in fields:
        return PROVIDER_GEMINI

    scores: dict[str, int] = {
        PROVIDER_CLAUDE: len(_CLAUDE_JSON_FIELDS & fields),
        PROVIDER_OPENAI: len(_OPENAI_JSON_FIELDS & fields),
        PROVIDER_GEMINI: len(_GEMINI_JSON_FIELDS & fields),
    }
    best_provider = max(scores, key=lambda k: scores[k])
    if scores[best_provider] == 0:
        return None
    logger.debug("Provider detection scores (JSON): %s -> %s", scores, best_provider)
    return best_provider


# ---------------------------------------------------------------------------
# Internal: JSON unwrapping
# ---------------------------------------------------------------------------

def _unwrap_json_records(parsed: Any) -> list[dict[str, Any]]:
    """Extract the list of log record dicts from a parsed JSON structure.

    Handles:
    * JSON arrays: ``[{...}, ...]``
    * Wrapped objects: ``{"data": [...]}`` / ``{"items": [...]}`` /
      ``{"logs": [...]}`` / ``{"usage": [...]}`` / ``{"records": [...]}`
    * Single objects: ``{...}`` → ``[{...}]``

    Args:
        parsed: The parsed JSON object (list, dict, or other).

    Returns:
        A list of record dicts (may be empty).

    Raises:
        MalformedLogError: If the structure cannot be interpreted.
    """
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]

    if isinstance(parsed, dict):
        for key in ("data", "items", "logs", "usage", "records", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return [r for r in parsed[key] if isinstance(r, dict)]
        # Single record object
        return [parsed]

    raise MalformedLogError(
        f"Unexpected JSON structure: expected list or dict, got {type(parsed).__name__}"
    )


# ---------------------------------------------------------------------------
# Internal: type coercion helpers
# ---------------------------------------------------------------------------

def _coerce_str(value: Any) -> Optional[str]:
    """Convert *value* to a stripped string, or ``None`` if empty/null."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coerce_int(value: Any) -> int:
    """Convert *value* to an int, returning 0 on failure."""
    if value is None:
        return 0
    try:
        # Handle float strings like '100.0'
        return int(float(str(value)))
    except (ValueError, TypeError):
        return 0


def _coerce_float(value: Any) -> float:
    """Convert *value* to a float, returning 0.0 on failure."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _parse_timestamp(value: Any) -> str:
    """Parse a timestamp value into an ISO-8601 UTC string.

    Accepts:
    * ISO-8601 strings (with or without timezone)
    * Unix timestamps (int or float seconds since epoch)
    * ``None`` / empty → returns current UTC time

    Args:
        value: The raw timestamp value.

    Returns:
        ISO-8601 string with UTC timezone.
    """
    if value is None:
        return datetime.now(tz=timezone.utc).isoformat()

    # Unix timestamp (numeric)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            pass

    s = str(value).strip()
    if not s:
        return datetime.now(tz=timezone.utc).isoformat()

    # Try parsing as numeric string (Unix timestamp)
    try:
        unix_val = float(s)
        return datetime.fromtimestamp(unix_val, tz=timezone.utc).isoformat()
    except ValueError:
        pass

    # Try a range of ISO-like formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    # Fallback: log a warning and use current time
    logger.warning("Cannot parse timestamp %r; using current UTC time.", value)
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_duration(
    value: Any,
    is_ms: bool = False,
) -> Optional[float]:
    """Parse a duration value into seconds as a float.

    Args:
        value: Raw duration (in seconds or milliseconds).
        is_ms: If ``True``, the value is in milliseconds and will be
            divided by 1000.

    Returns:
        Duration in seconds, or ``None`` if unparseable.
    """
    if value is None:
        return None
    try:
        secs = float(value)
        return secs / 1000.0 if is_ms else secs
    except (ValueError, TypeError):
        return None


def _field_is_ms(record: dict[str, Any], ms_keys: list[str]) -> bool:
    """Return ``True`` if any of *ms_keys* is present in *record*."""
    return any(k in record for k in ms_keys)


# ---------------------------------------------------------------------------
# Internal: status normalization
# ---------------------------------------------------------------------------

def _normalize_status_claude(raw: Any) -> str:
    """Map Claude stop_reason / status values to the unified status enum."""
    if raw is None:
        return "success"
    s = str(raw).lower().strip()
    if s in ("end_turn", "stop_sequence", "max_tokens", "success", "complete", "completed"):
        return "success"
    if s in ("error", "failed", "failure"):
        return "error"
    if s in ("cancelled", "canceled", "timeout"):
        return "cancelled"
    # Default to success for unknown stop reasons (e.g. 'tool_use')
    return "success"


def _normalize_status_openai(raw: Any) -> str:
    """Map OpenAI finish_reason / status values to the unified status enum."""
    if raw is None:
        return "success"
    s = str(raw).lower().strip()
    if s in ("stop", "length", "tool_calls", "function_call", "success", "complete", "completed"):
        return "success"
    if s in ("content_filter", "error", "failed", "failure"):
        return "error"
    if s in ("cancelled", "canceled", "timeout"):
        return "cancelled"
    return "success"


def _normalize_status_gemini(raw: Any) -> str:
    """Map Gemini finish_reason / status values to the unified status enum."""
    if raw is None:
        return "success"
    s = str(raw).lower().strip()
    if s in ("stop", "max_tokens", "finish_reason_stop", "success", "complete", "completed", "1"):
        return "success"
    if s in (
        "safety", "recitation", "other", "error", "failed",
        "finish_reason_safety", "finish_reason_recitation", "finish_reason_other",
    ):
        return "error"
    if s in ("cancelled", "canceled", "timeout", "unspecified", "finish_reason_unspecified", "0"):
        return "cancelled"
    return "success"
