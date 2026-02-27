"""Unit tests for agent_dash.poller – background API polling thread.

Tests use mocking to avoid real HTTP calls and focus on:
* Provider poller configuration and is_configured().
* Synthetic record generation.
* PollerThread start/stop lifecycle.
* _poll_one error handling and poll_run recording.
* Module-level start_poller / stop_poller / get_poller_status helpers.
"""

from __future__ import annotations

import time
import threading
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from agent_dash.app import create_app
from agent_dash.db import wire_db, query, get_provider_id
from agent_dash.poller import (
    ClaudePoller,
    GeminiPoller,
    OpenAIPoller,
    PollerThread,
    _build_pollers,
    _synthetic_record,
    _utcnow_iso,
    get_poller_status,
    start_poller,
    stop_poller,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    application = create_app({
        "DATABASE": str(tmp_path / "poller_test.db"),
        "TESTING": True,
        "SECRET_KEY": "test",
        "ENABLE_POLLER": False,
        "ANTHROPIC_API_KEY": "sk-test-anthropic",
        "OPENAI_API_KEY": "sk-test-openai",
        "GEMINI_API_KEY": "gemini-test-key",
        "POLL_INTERVAL_SECONDS": 60,
    })
    wire_db(application)
    return application


@pytest.fixture()
def app_ctx(app):
    with app.app_context():
        yield app


@pytest.fixture(autouse=True)
def reset_poller_state():
    """Ensure the module-level singleton is reset between tests."""
    stop_poller(timeout=2.0)
    yield
    stop_poller(timeout=2.0)


# ---------------------------------------------------------------------------
# _synthetic_record
# ---------------------------------------------------------------------------

class TestSyntheticRecord:
    def test_returns_dict(self):
        rec = _synthetic_record("claude")
        assert isinstance(rec, dict)

    def test_zero_tokens(self):
        rec = _synthetic_record("openai")
        assert rec["input_tokens"] == 0
        assert rec["output_tokens"] == 0
        assert rec["total_tokens"] == 0

    def test_zero_cost(self):
        rec = _synthetic_record("gemini")
        assert rec["cost_usd"] == 0.0

    def test_has_created_at(self):
        rec = _synthetic_record("claude")
        assert "created_at" in rec
        assert "T" in rec["created_at"]

    def test_note_stored(self):
        rec = _synthetic_record("claude", note="api_ping_ok")
        assert rec["_note"] == "api_ping_ok"

    def test_task_type_is_health_check(self):
        rec = _synthetic_record("openai")
        assert rec["task_type"] == "api_health_check"


# ---------------------------------------------------------------------------
# _utcnow_iso
# ---------------------------------------------------------------------------

class TestUtcNowIso:
    def test_returns_string(self):
        assert isinstance(_utcnow_iso(), str)

    def test_contains_utc(self):
        assert "+00:00" in _utcnow_iso()


# ---------------------------------------------------------------------------
# ClaudePoller
# ---------------------------------------------------------------------------

class TestClaudePoller:
    def test_is_configured_true(self):
        poller = ClaudePoller(api_key="sk-test")
        assert poller.is_configured() is True

    def test_is_configured_false(self):
        poller = ClaudePoller(api_key=None)
        assert poller.is_configured() is False

    def test_is_configured_empty_string(self):
        poller = ClaudePoller(api_key="")
        assert poller.is_configured() is False

    def test_fetch_records_no_key_returns_empty(self):
        poller = ClaudePoller(api_key=None)
        assert poller.fetch_records() == []

    def test_fetch_records_success(self):
        poller = ClaudePoller(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        with patch.object(poller.session, "get", return_value=mock_response):
            records = poller.fetch_records()
        assert len(records) == 1
        assert records[0]["_note"] == "api_ping_ok"

    def test_fetch_records_http_error_raises(self):
        poller = ClaudePoller(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = requests.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        with patch.object(poller.session, "get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                poller.fetch_records()

    def test_fetch_records_network_error_raises(self):
        poller = ClaudePoller(api_key="sk-test")
        with patch.object(
            poller.session, "get",
            side_effect=requests.ConnectionError("refused")
        ):
            with pytest.raises(requests.ConnectionError):
                poller.fetch_records()

    def test_session_has_api_key_header(self):
        poller = ClaudePoller(api_key="sk-test-key")
        assert poller.session.headers.get("x-api-key") == "sk-test-key"

    def test_provider_name(self):
        poller = ClaudePoller(api_key=None)
        assert poller.provider_name == "claude"


# ---------------------------------------------------------------------------
# OpenAIPoller
# ---------------------------------------------------------------------------

class TestOpenAIPoller:
    def test_is_configured_true(self):
        poller = OpenAIPoller(api_key="sk-openai")
        assert poller.is_configured() is True

    def test_is_configured_false(self):
        poller = OpenAIPoller(api_key=None)
        assert poller.is_configured() is False

    def test_fetch_records_no_key_returns_empty(self):
        poller = OpenAIPoller(api_key=None)
        assert poller.fetch_records() == []

    def test_fetch_records_usage_200(self):
        poller = OpenAIPoller(api_key="sk-openai")
        usage_payload = {
            "data": [
                {
                    "aggregation_timestamp": 1709288400,
                    "snapshot_id": "gpt-4o",
                    "n_context_tokens_total": 100,
                    "n_generated_tokens_total": 50,
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = usage_payload
        with patch.object(poller.session, "get", return_value=mock_response):
            records = poller.fetch_records()
        assert len(records) == 1
        assert records[0]["prompt_tokens"] == 100
        assert records[0]["completion_tokens"] == 50

    def test_fetch_records_usage_empty_data(self):
        """When /v1/usage returns empty data, a synthetic ping is returned."""
        poller = OpenAIPoller(api_key="sk-openai")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}
        with patch.object(poller.session, "get", return_value=mock_response):
            records = poller.fetch_records()
        assert len(records) == 1
        assert records[0]["_note"] == "usage_api_ok_no_data"

    def test_fetch_records_403_fallback_to_ping(self):
        """HTTP 403 on /v1/usage should fall back to /v1/models ping."""
        poller = OpenAIPoller(api_key="sk-openai")
        usage_response = MagicMock()
        usage_response.status_code = 403

        models_response = MagicMock()
        models_response.status_code = 200
        models_response.raise_for_status = MagicMock()

        call_count = {"n": 0}

        def side_effect(url, **kwargs):
            call_count["n"] += 1
            if "/v1/usage" in url:
                return usage_response
            return models_response

        with patch.object(poller.session, "get", side_effect=side_effect):
            records = poller.fetch_records()
        assert len(records) == 1
        assert records[0]["_note"] == "api_ping_ok"
        assert call_count["n"] == 2

    def test_session_has_auth_header(self):
        poller = OpenAIPoller(api_key="sk-test")
        assert "Bearer sk-test" in poller.session.headers.get("Authorization", "")

    def test_provider_name(self):
        assert OpenAIPoller(api_key=None).provider_name == "openai"


# ---------------------------------------------------------------------------
# GeminiPoller
# ---------------------------------------------------------------------------

class TestGeminiPoller:
    def test_is_configured_true(self):
        poller = GeminiPoller(api_key="gemini-key")
        assert poller.is_configured() is True

    def test_is_configured_false(self):
        poller = GeminiPoller(api_key=None)
        assert poller.is_configured() is False

    def test_fetch_records_no_key_returns_empty(self):
        poller = GeminiPoller(api_key=None)
        assert poller.fetch_records() == []

    def test_fetch_records_success(self):
        poller = GeminiPoller(api_key="gemini-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        with patch.object(poller.session, "get", return_value=mock_response):
            records = poller.fetch_records()
        assert len(records) == 1
        assert records[0]["_note"] == "api_ping_ok"

    def test_fetch_records_http_error_raises(self):
        poller = GeminiPoller(api_key="key")
        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        with patch.object(poller.session, "get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                poller.fetch_records()

    def test_provider_name(self):
        assert GeminiPoller(api_key=None).provider_name == "gemini"


# ---------------------------------------------------------------------------
# _build_pollers
# ---------------------------------------------------------------------------

class TestBuildPollers:
    def test_returns_three_pollers(self, app):
        pollers = _build_pollers(app)
        assert len(pollers) == 3

    def test_provider_names(self, app):
        pollers = _build_pollers(app)
        names = {p.provider_name for p in pollers}
        assert names == {"claude", "openai", "gemini"}

    def test_keys_propagated(self, app):
        pollers = _build_pollers(app)
        by_name = {p.provider_name: p for p in pollers}
        assert by_name["claude"].api_key == "sk-test-anthropic"
        assert by_name["openai"].api_key == "sk-test-openai"
        assert by_name["gemini"].api_key == "gemini-test-key"

    def test_no_keys_all_unconfigured(self, tmp_path):
        app_no_keys = create_app({
            "DATABASE": str(tmp_path / "no_keys.db"),
            "TESTING": True,
        })
        wire_db(app_no_keys)
        pollers = _build_pollers(app_no_keys)
        assert all(not p.is_configured() for p in pollers)


# ---------------------------------------------------------------------------
# PollerThread
# ---------------------------------------------------------------------------

class TestPollerThread:
    def _make_mock_poller(self, name: str, records=None) -> MagicMock:
        mock = MagicMock(spec=ClaudePoller)
        mock.provider_name = name
        mock.is_configured.return_value = True
        mock.fetch_records.return_value = records or []
        return mock

    def test_thread_starts_and_is_alive(self, app):
        mock_poller = self._make_mock_poller("claude")
        # Patch ingest_records so no DB writes required
        with patch("agent_dash.poller.ingest_records", return_value=0):
            thread = PollerThread(app, interval=60, pollers=[mock_poller])
            thread.start()
            time.sleep(0.1)
            assert thread.is_alive()
            thread.stop(timeout=3.0)

    def test_thread_is_daemon(self, app):
        thread = PollerThread(app, interval=60, pollers=[])
        assert thread.daemon is True

    def test_stop_terminates_thread(self, app):
        mock_poller = self._make_mock_poller("claude")
        with patch("agent_dash.poller.ingest_records", return_value=0):
            thread = PollerThread(app, interval=60, pollers=[mock_poller])
            thread.start()
            time.sleep(0.1)
            thread.stop(timeout=5.0)
        assert not thread.is_alive()

    def test_unconfigured_poller_skipped(self, app):
        """Pollers without an API key should not have fetch_records called."""
        mock_poller = self._make_mock_poller("claude")
        mock_poller.is_configured.return_value = False

        with patch("agent_dash.poller.ingest_records", return_value=0):
            thread = PollerThread(app, interval=1, pollers=[mock_poller])
            thread.start()
            time.sleep(0.2)
            thread.stop(timeout=3.0)

        mock_poller.fetch_records.assert_not_called()

    def test_poll_error_does_not_crash_thread(self, app):
        """A failing fetch_records should not kill the poller thread."""
        mock_poller = self._make_mock_poller("openai")
        mock_poller.fetch_records.side_effect = requests.ConnectionError("refused")

        with patch("agent_dash.poller.record_poll_run", return_value=1):
            with patch("agent_dash.poller.get_or_create_provider", return_value=2):
                thread = PollerThread(app, interval=60, pollers=[mock_poller])
                thread.start()
                time.sleep(0.2)
                assert thread.is_alive()
                thread.stop(timeout=3.0)


# ---------------------------------------------------------------------------
# start_poller / stop_poller / get_poller_status
# ---------------------------------------------------------------------------

class TestStartStopPoller:
    def test_start_poller_returns_thread(self, app):
        with patch("agent_dash.poller.ingest_records", return_value=0):
            thread = start_poller(app)
        assert thread is not None
        assert isinstance(thread, PollerThread)
        stop_poller()

    def test_start_poller_idempotent(self, app):
        """Calling start_poller twice should return the same thread."""
        with patch("agent_dash.poller.ingest_records", return_value=0):
            t1 = start_poller(app)
            t2 = start_poller(app)
        assert t1 is t2
        stop_poller()

    def test_start_poller_returns_none_when_no_keys(self, tmp_path):
        app_no_keys = create_app({
            "DATABASE": str(tmp_path / "nk.db"),
            "TESTING": True,
        })
        wire_db(app_no_keys)
        result = start_poller(app_no_keys)
        assert result is None

    def test_get_poller_status_running(self, app):
        with patch("agent_dash.poller.ingest_records", return_value=0):
            start_poller(app)
        status = get_poller_status()
        assert status["running"] is True
        assert status["thread_name"] is not None
        stop_poller()

    def test_get_poller_status_stopped(self):
        stop_poller()  # ensure clean state
        status = get_poller_status()
        assert status["running"] is False
        assert status["thread_name"] is None

    def test_stop_poller_noop_when_not_running(self):
        """stop_poller should not raise if nothing is running."""
        stop_poller()  # first stop
        stop_poller()  # second – should be a no-op
