"""Unit tests for agent_dash.ingest – log parsing and schema normalization.

Covers:
* Provider auto-detection from CSV columns and JSON fields.
* CSV ingestion for Claude, OpenAI, and Gemini.
* JSON ingestion for all three providers.
* Individual normalizer functions.
* Error handling (unknown provider, malformed input, missing file).
* Timestamp and duration coercion helpers.
* Status normalization for all three providers.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from agent_dash.app import create_app
from agent_dash.db import wire_db, get_provider_id, count_usage_logs
from agent_dash.ingest import (
    PROVIDER_CLAUDE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    IngestError,
    MalformedLogError,
    UnknownProviderError,
    _coerce_float,
    _coerce_int,
    _coerce_str,
    _detect_provider_from_columns,
    _detect_provider_from_json_record,
    _normalize_status_claude,
    _normalize_status_gemini,
    _normalize_status_openai,
    _parse_duration,
    _parse_timestamp,
    _unwrap_json_records,
    ingest_csv,
    ingest_file,
    ingest_json,
    ingest_records,
    normalize_claude_record,
    normalize_gemini_record,
    normalize_openai_record,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    """Flask app wired to a temp-file database."""
    application = create_app({
        "DATABASE": str(tmp_path / "ingest_test.db"),
        "TESTING": True,
        "SECRET_KEY": "test",
    })
    wire_db(application)
    return application


@pytest.fixture()
 def app_ctx(app):
    """Active application context."""
    with app.app_context():
        yield app


@pytest.fixture()
def claude_provider_id(app_ctx):
    return get_provider_id(PROVIDER_CLAUDE)


@pytest.fixture()
def openai_provider_id(app_ctx):
    return get_provider_id(PROVIDER_OPENAI)


@pytest.fixture()
def gemini_provider_id(app_ctx):
    return get_provider_id(PROVIDER_GEMINI)


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def make_claude_csv_row(**overrides) -> dict[str, Any]:
    base = {
        "id": "req_abc123",
        "created_at": "2024-03-01T10:00:00+00:00",
        "model": "claude-3-5-sonnet-20241022",
        "task_type": "code_generation",
        "input_tokens": 100,
        "output_tokens": 50,
        "stop_reason": "end_turn",
        "cost_usd": 0.005,
        "duration_seconds": 2.5,
    }
    base.update(overrides)
    return base


def make_openai_csv_row(**overrides) -> dict[str, Any]:
    base = {
        "id": "chatcmpl-xyz",
        "created": "2024-03-01T11:00:00+00:00",
        "model": "gpt-4o",
        "object": "chat.completion",
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "finish_reason": "stop",
        "cost_usd": 0.012,
    }
    base.update(overrides)
    return base


def make_gemini_csv_row(**overrides) -> dict[str, Any]:
    base = {
        "name": "operations/abc",
        "create_time": "2024-03-01T12:00:00+00:00",
        "model": "gemini-1.5-pro",
        "prompt_token_count": 150,
        "candidates_token_count": 60,
        "finish_reason": "STOP",
        "safety_ratings": "[]",
        "cost_usd": 0.003,
    }
    base.update(overrides)
    return base


def csv_from_rows(rows: list[dict]) -> bytes:
    """Serialise a list of dicts to CSV bytes."""
    import pandas as pd
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Provider detection – CSV
# ---------------------------------------------------------------------------

class TestDetectProviderFromColumns:
    def test_detects_claude(self):
        cols = {"id", "input_tokens", "output_tokens", "stop_reason", "created_at"}
        assert _detect_provider_from_columns(cols) == PROVIDER_CLAUDE

    def test_detects_openai(self):
        cols = {"id", "prompt_tokens", "completion_tokens", "finish_reason", "object"}
        assert _detect_provider_from_columns(cols) == PROVIDER_OPENAI

    def test_detects_gemini(self):
        cols = {"candidates_token_count", "prompt_token_count", "finish_reason", "safety_ratings"}
        assert _detect_provider_from_columns(cols) == PROVIDER_GEMINI

    def test_returns_none_for_unknown(self):
        cols = {"col_a", "col_b", "col_c"}
        assert _detect_provider_from_columns(cols) is None

    def test_returns_none_for_empty(self):
        assert _detect_provider_from_columns(set()) is None


# ---------------------------------------------------------------------------
# Provider detection – JSON
# ---------------------------------------------------------------------------

class TestDetectProviderFromJsonRecord:
    def test_detects_claude(self):
        record = {"id": "r1", "input_tokens": 10, "output_tokens": 5, "stop_reason": "end_turn"}
        assert _detect_provider_from_json_record(record) == PROVIDER_CLAUDE

    def test_detects_openai(self):
        record = {"id": "c1", "object": "chat.completion", "prompt_tokens": 10, "finish_reason": "stop"}
        assert _detect_provider_from_json_record(record) == PROVIDER_OPENAI

    def test_detects_gemini_via_usage_metadata(self):
        record = {"usageMetadata": {"promptTokenCount": 10}}
        assert _detect_provider_from_json_record(record) == PROVIDER_GEMINI

    def test_detects_gemini_via_candidates_token_count(self):
        record = {"candidates_token_count": 20, "prompt_token_count": 10}
        assert _detect_provider_from_json_record(record) == PROVIDER_GEMINI

    def test_returns_none_for_unknown(self):
        record = {"foo": 1, "bar": 2}
        assert _detect_provider_from_json_record(record) is None


# ---------------------------------------------------------------------------
# JSON unwrapping
# ---------------------------------------------------------------------------

class TestUnwrapJsonRecords:
    def test_list_passthrough(self):
        data = [{"a": 1}, {"b": 2}]
        assert _unwrap_json_records(data) == data

    def test_filters_non_dicts_from_list(self):
        data = [{"a": 1}, "not_a_dict", 42]
        assert _unwrap_json_records(data) == [{"a": 1}]

    def test_unwraps_data_key(self):
        payload = {"data": [{"a": 1}]}
        assert _unwrap_json_records(payload) == [{"a": 1}]

    def test_unwraps_items_key(self):
        payload = {"items": [{"x": 9}]}
        assert _unwrap_json_records(payload) == [{"x": 9}]

    def test_unwraps_logs_key(self):
        payload = {"logs": [{"y": 7}]}
        assert _unwrap_json_records(payload) == [{"y": 7}]

    def test_single_object_wrapped_in_list(self):
        record = {"id": "r1", "tokens": 100}
        assert _unwrap_json_records(record) == [record]

    def test_raises_on_unsupported_type(self):
        with pytest.raises(MalformedLogError):
            _unwrap_json_records("a plain string")

    def test_empty_list(self):
        assert _unwrap_json_records([]) == []


# ---------------------------------------------------------------------------
# Claude normalizer
# ---------------------------------------------------------------------------

class TestNormalizeClaudeRecord:
    def test_basic_record(self, claude_provider_id):
        raw = make_claude_csv_row()
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["provider_id"] == claude_provider_id
        assert result["external_id"] == "req_abc123"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["task_type"] == "code_generation"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert abs(result["cost_usd"] - 0.005) < 1e-9
        assert abs(result["duration_seconds"] - 2.5) < 1e-9
        assert result["status"] == "success"
        assert result["raw_payload"] is raw

    def test_anthropic_model_field(self, claude_provider_id):
        raw = {"anthropic_model": "claude-3-haiku", "input_tokens": 10, "output_tokens": 5}
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["model"] == "claude-3-haiku"

    def test_output_tokens_alias(self, claude_provider_id):
        raw = {"output_tokens": 75, "input_tokens": 25}
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["completion_tokens"] == 75
        assert result["prompt_tokens"] == 25

    def test_duration_ms_converted(self, claude_provider_id):
        raw = {"duration_ms": 3000, "input_tokens": 0, "output_tokens": 0}
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["duration_seconds"] is not None
        assert abs(result["duration_seconds"] - 3.0) < 1e-6

    def test_stop_reason_error(self, claude_provider_id):
        raw = {"stop_reason": "error", "input_tokens": 0, "output_tokens": 0}
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["status"] == "error"

    def test_missing_timestamp_defaults_to_now(self, claude_provider_id):
        raw = {"input_tokens": 10, "output_tokens": 5}
        result = normalize_claude_record(raw, claude_provider_id)
        assert "T" in result["logged_at"]

    def test_raw_payload_preserved(self, claude_provider_id):
        raw = make_claude_csv_row(extra_field="sentinel_value")
        result = normalize_claude_record(raw, claude_provider_id)
        assert result["raw_payload"]["extra_field"] == "sentinel_value"


# ---------------------------------------------------------------------------
# OpenAI normalizer
# ---------------------------------------------------------------------------

class TestNormalizeOpenAIRecord:
    def test_basic_record(self, openai_provider_id):
        raw = make_openai_csv_row()
        result = normalize_openai_record(raw, openai_provider_id)
        assert result["provider_id"] == openai_provider_id
        assert result["external_id"] == "chatcmpl-xyz"
        assert result["model"] == "gpt-4o"
        assert result["prompt_tokens"] == 200
        assert result["completion_tokens"] == 80
        assert result["total_tokens"] == 280
        assert result["status"] == "success"

    def test_nested_usage_object(self, openai_provider_id):
        raw = {
            "id": "chatcmpl-nested",
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            },
            "finish_reason": "stop",
        }
        result = normalize_openai_record(raw, openai_provider_id)
        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 25
        assert result["total_tokens"] == 75

    def test_chat_completion_object_task_type(self, openai_provider_id):
        raw = {"object": "chat.completion", "prompt_tokens": 10, "completion_tokens": 5}
        result = normalize_openai_record(raw, openai_provider_id)
        assert result["task_type"] == "chat_completion"

    def test_unix_timestamp(self, openai_provider_id):
        raw = {"created": 1709288400, "prompt_tokens": 10, "completion_tokens": 5}
        result = normalize_openai_record(raw, openai_provider_id)
        assert "2024" in result["logged_at"]

    def test_content_filter_status(self, openai_provider_id):
        raw = {"finish_reason": "content_filter", "prompt_tokens": 5, "completion_tokens": 0}
        result = normalize_openai_record(raw, openai_provider_id)
        assert result["status"] == "error"

    def test_cancelled_status(self, openai_provider_id):
        raw = {"finish_reason": "cancelled", "prompt_tokens": 5, "completion_tokens": 0}
        result = normalize_openai_record(raw, openai_provider_id)
        assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Gemini normalizer
# ---------------------------------------------------------------------------

class TestNormalizeGeminiRecord:
    def test_basic_record(self, gemini_provider_id):
        raw = make_gemini_csv_row()
        result = normalize_gemini_record(raw, gemini_provider_id)
        assert result["provider_id"] == gemini_provider_id
        assert result["model"] == "gemini-1.5-pro"
        assert result["prompt_tokens"] == 150
        assert result["completion_tokens"] == 60
        assert result["total_tokens"] == 210
        assert result["status"] == "success"

    def test_nested_usage_metadata(self, gemini_provider_id):
        raw = {
            "model": "gemini-1.5-flash",
            "usageMetadata": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 40,
                "totalTokenCount": 120,
            },
        }
        result = normalize_gemini_record(raw, gemini_provider_id)
        assert result["prompt_tokens"] == 80
        assert result["completion_tokens"] == 40
        assert result["total_tokens"] == 120

    def test_name_field_as_external_id(self, gemini_provider_id):
        raw = {"name": "operations/my-op-123", "prompt_token_count": 10, "candidates_token_count": 5}
        result = normalize_gemini_record(raw, gemini_provider_id)
        assert result["external_id"] == "operations/my-op-123"

    def test_stop_status_success(self, gemini_provider_id):
        raw = {"finish_reason": "STOP", "prompt_token_count": 10, "candidates_token_count": 5}
        result = normalize_gemini_record(raw, gemini_provider_id)
        assert result["status"] == "success"

    def test_safety_status_error(self, gemini_provider_id):
        raw = {"finish_reason": "SAFETY", "prompt_token_count": 10, "candidates_token_count": 0}
        result = normalize_gemini_record(raw, gemini_provider_id)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# CSV ingestion (integration)
# ---------------------------------------------------------------------------

class TestIngestCsv:
    def test_ingest_claude_csv(self, app_ctx):
        rows = [make_claude_csv_row(), make_claude_csv_row(id="req_002", input_tokens=200)]
        csv_bytes = csv_from_rows(rows)
        count = ingest_csv(csv_bytes, provider=PROVIDER_CLAUDE)
        assert count == 2

    def test_ingest_openai_csv(self, app_ctx):
        rows = [make_openai_csv_row()]
        csv_bytes = csv_from_rows(rows)
        count = ingest_csv(csv_bytes, provider=PROVIDER_OPENAI)
        assert count == 1

    def test_ingest_gemini_csv(self, app_ctx):
        rows = [make_gemini_csv_row()]
        csv_bytes = csv_from_rows(rows)
        count = ingest_csv(csv_bytes, provider=PROVIDER_GEMINI)
        assert count == 1

    def test_autodetect_claude_provider(self, app_ctx):
        rows = [make_claude_csv_row()]
        csv_bytes = csv_from_rows(rows)
        # No explicit provider – should detect Claude from columns
        count = ingest_csv(csv_bytes)  # auto-detect
        assert count == 1

    def test_autodetect_openai_provider(self, app_ctx):
        rows = [make_openai_csv_row()]
        csv_bytes = csv_from_rows(rows)
        count = ingest_csv(csv_bytes)
        assert count == 1

    def test_autodetect_gemini_provider(self, app_ctx):
        rows = [make_gemini_csv_row()]
        csv_bytes = csv_from_rows(rows)
        count = ingest_csv(csv_bytes)
        assert count == 1

    def test_unknown_provider_raises(self, app_ctx):
        csv_bytes = b"col_a,col_b\n1,2\n"
        with pytest.raises(UnknownProviderError):
            ingest_csv(csv_bytes)

    def test_malformed_csv_raises(self, app_ctx):
        with pytest.raises(MalformedLogError):
            # Pass something that will fail pd.read_csv entirely
            ingest_csv(b"\x00" * 10)

    def test_empty_csv_returns_zero(self, app_ctx):
        # A CSV with only a header and no data rows
        csv_bytes = b"input_tokens,output_tokens,stop_reason\n"
        count = ingest_csv(csv_bytes, provider=PROVIDER_CLAUDE)
        assert count == 0

    def test_string_input_accepted(self, app_ctx):
        import pandas as pd
        rows = [make_claude_csv_row()]
        df = pd.DataFrame(rows)
        csv_str = df.to_csv(index=False)
        count = ingest_csv(csv_str, provider=PROVIDER_CLAUDE)
        assert count == 1

    def test_records_persisted_to_db(self, app_ctx):
        provider_id = get_provider_id(PROVIDER_CLAUDE)
        before = count_usage_logs(provider_id=provider_id)
        rows = [make_claude_csv_row(), make_claude_csv_row(id="req_persisted")]
        csv_bytes = csv_from_rows(rows)
        ingest_csv(csv_bytes, provider=PROVIDER_CLAUDE)
        after = count_usage_logs(provider_id=provider_id)
        assert after == before + 2


# ---------------------------------------------------------------------------
# JSON ingestion (integration)
# ---------------------------------------------------------------------------

class TestIngestJson:
    def test_ingest_array(self, app_ctx):
        records = [make_claude_csv_row(), make_claude_csv_row(id="r2")]
        json_bytes = json.dumps(records).encode()
        count = ingest_json(json_bytes, provider=PROVIDER_CLAUDE)
        assert count == 2

    def test_ingest_wrapped_data_key(self, app_ctx):
        payload = {"data": [make_openai_csv_row()]}
        json_bytes = json.dumps(payload).encode()
        count = ingest_json(json_bytes, provider=PROVIDER_OPENAI)
        assert count == 1

    def test_ingest_wrapped_items_key(self, app_ctx):
        payload = {"items": [make_gemini_csv_row()]}
        json_bytes = json.dumps(payload).encode()
        count = ingest_json(json_bytes, provider=PROVIDER_GEMINI)
        assert count == 1

    def test_ingest_single_object(self, app_ctx):
        record = make_claude_csv_row()
        json_bytes = json.dumps(record).encode()
        count = ingest_json(json_bytes, provider=PROVIDER_CLAUDE)
        assert count == 1

    def test_autodetect_claude_from_json(self, app_ctx):
        records = [{
            "id": "r1",
            "input_tokens": 100,
            "output_tokens": 50,
            "stop_reason": "end_turn",
            "model": "claude-3-haiku",
        }]
        json_bytes = json.dumps(records).encode()
        count = ingest_json(json_bytes)
        assert count == 1

    def test_autodetect_openai_from_json(self, app_ctx):
        records = [{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish_reason": "stop",
        }]
        json_bytes = json.dumps(records).encode()
        count = ingest_json(json_bytes)
        assert count == 1

    def test_autodetect_gemini_from_json(self, app_ctx):
        records = [{"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10}}]
        json_bytes = json.dumps(records).encode()
        count = ingest_json(json_bytes)
        assert count == 1

    def test_malformed_json_raises(self, app_ctx):
        with pytest.raises(MalformedLogError):
            ingest_json(b"{invalid json", provider=PROVIDER_CLAUDE)

    def test_empty_array_returns_zero(self, app_ctx):
        json_bytes = json.dumps([]).encode()
        count = ingest_json(json_bytes, provider=PROVIDER_CLAUDE)
        assert count == 0

    def test_string_bytes_input(self, app_ctx):
        records = [make_openai_csv_row()]
        json_str = json.dumps(records)
        count = ingest_json(json_str, provider=PROVIDER_OPENAI)
        assert count == 1

    def test_unknown_provider_raises(self, app_ctx):
        records = [{"foo": 1, "bar": 2}]
        json_bytes = json.dumps(records).encode()
        with pytest.raises(UnknownProviderError):
            ingest_json(json_bytes)


# ---------------------------------------------------------------------------
# ingest_file (integration)
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_csv_file(self, app_ctx, tmp_path):
        rows = [make_claude_csv_row()]
        csv_path = tmp_path / "claude_log.csv"
        csv_path.write_bytes(csv_from_rows(rows))
        count = ingest_file(csv_path, provider=PROVIDER_CLAUDE)
        assert count == 1

    def test_json_file(self, app_ctx, tmp_path):
        records = [make_openai_csv_row()]
        json_path = tmp_path / "openai_log.json"
        json_path.write_bytes(json.dumps(records).encode())
        count = ingest_file(json_path, provider=PROVIDER_OPENAI)
        assert count == 1

    def test_missing_file_raises(self, app_ctx, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_file(tmp_path / "nonexistent.csv", provider=PROVIDER_CLAUDE)

    def test_unknown_extension_raises(self, app_ctx, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("hello")
        with pytest.raises(MalformedLogError):
            ingest_file(p, provider=PROVIDER_CLAUDE)

    def test_explicit_format_override(self, app_ctx, tmp_path):
        """file_format parameter overrides extension-based detection."""
        records = [make_claude_csv_row()]
        # Write a JSON file with a .log extension
        p = tmp_path / "data.log"
        p.write_bytes(json.dumps(records).encode())
        count = ingest_file(p, provider=PROVIDER_CLAUDE, file_format="json")
        assert count == 1


# ---------------------------------------------------------------------------
# ingest_records
# ---------------------------------------------------------------------------

class TestIngestRecords:
    def test_inserts_claude_records(self, app_ctx):
        records = [{"input_tokens": 10, "output_tokens": 5, "stop_reason": "end_turn"}]
        count = ingest_records(records, PROVIDER_CLAUDE)
        assert count == 1

    def test_inserts_openai_records(self, app_ctx):
        records = [make_openai_csv_row()]
        count = ingest_records(records, PROVIDER_OPENAI)
        assert count == 1

    def test_inserts_gemini_records(self, app_ctx):
        records = [make_gemini_csv_row()]
        count = ingest_records(records, PROVIDER_GEMINI)
        assert count == 1

    def test_unknown_provider_raises(self, app_ctx):
        with pytest.raises(IngestError):
            ingest_records([{"a": 1}], "unknown_provider_xyz")

    def test_empty_records_returns_zero(self, app_ctx):
        count = ingest_records([], PROVIDER_CLAUDE)
        assert count == 0


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_iso_with_tz(self):
        result = _parse_timestamp("2024-03-01T10:00:00+00:00")
        assert "2024-03-01" in result
        assert "+00:00" in result

    def test_iso_with_z(self):
        result = _parse_timestamp("2024-03-01T10:00:00Z")
        assert "2024" in result

    def test_unix_int(self):
        result = _parse_timestamp(1709288400)
        assert "2024" in result

    def test_unix_float(self):
        result = _parse_timestamp(1709288400.123)
        assert "2024" in result

    def test_unix_string(self):
        result = _parse_timestamp("1709288400")
        assert "2024" in result

    def test_date_only(self):
        result = _parse_timestamp("2024-03-01")
        assert "2024-03-01" in result

    def test_none_returns_current_time(self):
        before = datetime.now(tz=timezone.utc)
        result = _parse_timestamp(None)
        after = datetime.now(tz=timezone.utc)
        parsed = datetime.fromisoformat(result)
        assert before <= parsed <= after

    def test_empty_string_returns_current_time(self):
        result = _parse_timestamp("")
        assert "T" in result

    def test_unparseable_returns_current_time(self):
        result = _parse_timestamp("not-a-date")
        assert "T" in result

    def test_iso_without_tz_gets_utc(self):
        result = _parse_timestamp("2024-03-01T10:00:00")
        assert "+00:00" in result


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

class TestParseDuration:
    def test_seconds_int(self):
        assert _parse_duration(5) == 5.0

    def test_seconds_float(self):
        assert abs(_parse_duration(2.5) - 2.5) < 1e-9

    def test_milliseconds_conversion(self):
        assert abs(_parse_duration(3000, is_ms=True) - 3.0) < 1e-9

    def test_string_value(self):
        assert _parse_duration("10") == 10.0

    def test_none_returns_none(self):
        assert _parse_duration(None) is None

    def test_invalid_string_returns_none(self):
        assert _parse_duration("not_a_number") is None

    def test_zero(self):
        assert _parse_duration(0) == 0.0


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

class TestCoerceHelpers:
    def test_coerce_str_none(self):
        assert _coerce_str(None) is None

    def test_coerce_str_empty(self):
        assert _coerce_str("") is None

    def test_coerce_str_whitespace(self):
        assert _coerce_str("   ") is None

    def test_coerce_str_value(self):
        assert _coerce_str("hello") == "hello"

    def test_coerce_str_strips(self):
        assert _coerce_str("  hi  ") == "hi"

    def test_coerce_str_int(self):
        assert _coerce_str(42) == "42"

    def test_coerce_int_none(self):
        assert _coerce_int(None) == 0

    def test_coerce_int_value(self):
        assert _coerce_int(100) == 100

    def test_coerce_int_float_string(self):
        assert _coerce_int("100.0") == 100

    def test_coerce_int_invalid(self):
        assert _coerce_int("abc") == 0

    def test_coerce_float_none(self):
        assert _coerce_float(None) == 0.0

    def test_coerce_float_value(self):
        assert abs(_coerce_float(3.14) - 3.14) < 1e-9

    def test_coerce_float_string(self):
        assert abs(_coerce_float("2.718") - 2.718) < 1e-6

    def test_coerce_float_invalid(self):
        assert _coerce_float("not_a_float") == 0.0


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------

class TestNormalizeStatusClaude:
    @pytest.mark.parametrize("raw,expected", [
        ("end_turn", "success"),
        ("stop_sequence", "success"),
        ("max_tokens", "success"),
        ("success", "success"),
        ("error", "error"),
        ("failed", "error"),
        ("cancelled", "cancelled"),
        ("timeout", "cancelled"),
        (None, "success"),
        ("tool_use", "success"),  # unknown defaults to success
    ])
    def test_mapping(self, raw, expected):
        assert _normalize_status_claude(raw) == expected


class TestNormalizeStatusOpenAI:
    @pytest.mark.parametrize("raw,expected", [
        ("stop", "success"),
        ("length", "success"),
        ("tool_calls", "success"),
        ("function_call", "success"),
        ("content_filter", "error"),
        ("error", "error"),
        ("cancelled", "cancelled"),
        (None, "success"),
    ])
    def test_mapping(self, raw, expected):
        assert _normalize_status_openai(raw) == expected


class TestNormalizeStatusGemini:
    @pytest.mark.parametrize("raw,expected", [
        ("STOP", "success"),
        ("stop", "success"),
        ("1", "success"),
        ("SAFETY", "error"),
        ("RECITATION", "error"),
        ("OTHER", "error"),
        ("UNSPECIFIED", "cancelled"),
        ("0", "cancelled"),
        (None, "success"),
    ])
    def test_mapping(self, raw, expected):
        assert _normalize_status_gemini(raw) == expected
