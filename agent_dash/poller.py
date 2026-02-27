"""Background API poller for agent_dash.

This module provides a background thread that periodically fetches usage
data from AI provider APIs (Claude/Anthropic, OpenAI, Gemini) and persists
the results to the SQLite database via the ingest and DB modules.

Design overview
---------------
* :class:`PollerThread` is a daemon :class:`threading.Thread` subclass that
  runs forever, sleeping for ``POLL_INTERVAL_SECONDS`` between cycles.
* :class:`ProviderPoller` is an abstract base that each provider implements.
  Each implementation knows how to fetch usage data from its provider's API
  and return a list of raw record dicts.
* :func:`start_poller` creates and starts the background thread; it is
  called from the Flask app factory when ``ENABLE_POLLER=true``.
* :func:`stop_poller` signals the thread to stop (for graceful shutdown).

Provider API notes
------------------
Anthropic (Claude)
    No official usage-export API at time of writing.  The poller makes a
    lightweight ``GET /v1/models`` call to verify connectivity, then
    generates a synthetic "ping" record to confirm the API key is valid.
    Real token data must come via CSV/JSON import.

OpenAI
    Uses ``GET /v1/usage`` (dashboard API).  Requires a session key or
    Organisation API key.  Falls back gracefully when the key is absent.

Gemini (Google)
    No public usage-export API.  The poller makes a ``GET`` to the models
    list endpoint to verify the key and generates a synthetic ping record.

All network errors are caught and recorded in the ``poll_runs`` table so
operators can diagnose connectivity problems without crashing the server.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_POLL_INTERVAL = 300  # seconds
_REQUEST_TIMEOUT = 30  # seconds per HTTP request

# Anthropic base URL
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_API_VERSION = "2023-06-01"

# OpenAI base URL
_OPENAI_BASE_URL = "https://api.openai.com"

# Gemini base URL
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


# ---------------------------------------------------------------------------
# Abstract base poller
# ---------------------------------------------------------------------------

class ProviderPoller(ABC):
    """Abstract base class for provider-specific API pollers.

    Subclasses must implement :meth:`fetch_records` to retrieve raw usage
    data from their provider's API.

    Args:
        api_key: The provider API key; may be ``None`` if not configured.
        session: A :class:`requests.Session` to use for HTTP calls.  A new
            session is created if ``None``.
    """

    #: Canonical provider name (must match ``providers.name`` in the DB)
    provider_name: str = ""

    def __init__(
        self,
        api_key: Optional[str],
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self._configure_session()

    def _configure_session(self) -> None:
        """Apply default headers and timeout to the requests session."""
        self.session.headers.update({
            "User-Agent": "agent_dash/0.1.0",
        })

    @abstractmethod
    def fetch_records(self) -> list[dict[str, Any]]:
        """Fetch raw usage log records from the provider API.

        Returns:
            A list of raw record dicts to be normalised and inserted.
            May be empty if no new data is available.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """

    def is_configured(self) -> bool:
        """Return ``True`` if an API key has been supplied for this provider.

        Returns:
            ``True`` if ``api_key`` is non-empty, ``False`` otherwise.
        """
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Claude / Anthropic poller
# ---------------------------------------------------------------------------

class ClaudePoller(ProviderPoller):
    """Poller for the Anthropic Claude API.

    Anthropic does not expose a usage-export endpoint, so this poller
    performs a lightweight models-list call to verify the API key is valid
    and records a synthetic health-check record.  Real token data should
    be imported via the CSV/JSON ingest route.

    Args:
        api_key: Anthropic API key (``ANTHROPIC_API_KEY`` env var).
        session: Optional shared :class:`requests.Session`.
    """

    provider_name = "claude"

    def _configure_session(self) -> None:
        super()._configure_session()
        if self.api_key:
            self.session.headers.update({
                "x-api-key": self.api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
            })

    def fetch_records(self) -> list[dict[str, Any]]:
        """Ping the Anthropic API and return a synthetic health record.

        Returns:
            A list containing a single synthetic record on success, or an
            empty list if the API key is missing or the request fails.
        """
        if not self.is_configured():
            logger.debug("ClaudePoller: no API key configured; skipping.")
            return []

        url = f"{_ANTHROPIC_BASE_URL}/v1/models"
        try:
            response = self.session.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info("ClaudePoller: API ping successful (HTTP %d).", response.status_code)
            # Return a synthetic record to record the successful poll
            return [_synthetic_record(provider="claude", note="api_ping_ok")]
        except requests.HTTPError as exc:
            logger.warning(
                "ClaudePoller: API returned HTTP %d: %s",
                exc.response.status_code if exc.response is not None else -1,
                exc,
            )
            raise
        except requests.RequestException as exc:
            logger.warning("ClaudePoller: network error: %s", exc)
            raise


# ---------------------------------------------------------------------------
# OpenAI poller
# ---------------------------------------------------------------------------

class OpenAIPoller(ProviderPoller):
    """Poller for the OpenAI usage API.

    Attempts to call ``GET /v1/usage`` to retrieve daily usage data.  This
    endpoint requires an *organisation-level* or *admin* API key; standard
    completion keys are insufficient.  Falls back to a models-list ping if
    the usage endpoint is unavailable.

    Args:
        api_key: OpenAI API key (``OPENAI_API_KEY`` env var).
        session: Optional shared :class:`requests.Session`.
    """

    provider_name = "openai"

    def _configure_session(self) -> None:
        super()._configure_session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })

    def fetch_records(self) -> list[dict[str, Any]]:
        """Fetch usage data from the OpenAI API.

        Tries ``/v1/usage`` first; falls back to ``/v1/models`` ping on
        HTTP 403 / 404 (insufficient permissions or endpoint unavailable).

        Returns:
            A list of normalised-ish usage dicts, or empty list on failure.
        """
        if not self.is_configured():
            logger.debug("OpenAIPoller: no API key configured; skipping.")
            return []

        # Try the usage endpoint (date = today in YYYYMMDD format)
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        usage_url = f"{_OPENAI_BASE_URL}/v1/usage"
        params = {"date": today}

        try:
            response = self.session.get(
                usage_url, params=params, timeout=_REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return self._parse_usage_response(response.json())
            elif response.status_code in (403, 404):
                logger.info(
                    "OpenAIPoller: /v1/usage returned %d; falling back to ping.",
                    response.status_code,
                )
                return self._ping_models_endpoint()
            else:
                response.raise_for_status()
                return []
        except requests.HTTPError as exc:
            logger.warning(
                "OpenAIPoller: HTTP error on /v1/usage: %s", exc
            )
            raise
        except requests.RequestException as exc:
            logger.warning("OpenAIPoller: network error: %s", exc)
            raise

    def _parse_usage_response(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract usage records from the /v1/usage API response.

        Args:
            payload: Parsed JSON response body.

        Returns:
            List of raw record dicts with token count fields.
        """
        records: list[dict[str, Any]] = []
        data = payload.get("data") or []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            records.append({
                "id": entry.get("aggregation_timestamp"),
                "created": entry.get("aggregation_timestamp"),
                "model": entry.get("snapshot_id") or entry.get("model"),
                "object": "usage.aggregate",
                "prompt_tokens": entry.get("n_context_tokens_total", 0),
                "completion_tokens": entry.get("n_generated_tokens_total", 0),
                "total_tokens": (
                    entry.get("n_context_tokens_total", 0)
                    + entry.get("n_generated_tokens_total", 0)
                ),
                "finish_reason": "stop",
                "_raw": entry,
            })
        if not records:
            # No data today yet; record a ping
            records.append(_synthetic_record(provider="openai", note="usage_api_ok_no_data"))
        return records

    def _ping_models_endpoint(self) -> list[dict[str, Any]]:
        """Fall back to a /v1/models ping to verify connectivity."""
        url = f"{_OPENAI_BASE_URL}/v1/models"
        try:
            response = self.session.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info("OpenAIPoller: models ping successful.")
            return [_synthetic_record(provider="openai", note="api_ping_ok")]
        except requests.RequestException as exc:
            logger.warning("OpenAIPoller: models ping failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Gemini / Google poller
# ---------------------------------------------------------------------------

class GeminiPoller(ProviderPoller):
    """Poller for the Google Gemini API.

    Google does not expose a usage-export REST API, so this poller verifies
    the API key via the models-list endpoint and records a synthetic ping.
    Real token data should be imported via CSV/JSON ingest.

    Args:
        api_key: Google AI Studio / Gemini API key (``GEMINI_API_KEY``).
        session: Optional shared :class:`requests.Session`.
    """

    provider_name = "gemini"

    def fetch_records(self) -> list[dict[str, Any]]:
        """Ping the Gemini models endpoint and return a synthetic health record.

        Returns:
            A list containing a single synthetic record on success.
        """
        if not self.is_configured():
            logger.debug("GeminiPoller: no API key configured; skipping.")
            return []

        url = f"{_GEMINI_BASE_URL}/v1beta/models"
        params = {"key": self.api_key}
        try:
            response = self.session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info("GeminiPoller: API ping successful (HTTP %d).", response.status_code)
            return [_synthetic_record(provider="gemini", note="api_ping_ok")]
        except requests.HTTPError as exc:
            logger.warning(
                "GeminiPoller: HTTP %d: %s",
                exc.response.status_code if exc.response is not None else -1,
                exc,
            )
            raise
        except requests.RequestException as exc:
            logger.warning("GeminiPoller: network error: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Poller thread
# ---------------------------------------------------------------------------

class PollerThread(threading.Thread):
    """Background daemon thread that polls provider APIs on a fixed schedule.

    The thread runs as a daemon so it does not prevent the process from
    exiting.  Call :meth:`stop` to request a clean shutdown.

    Args:
        app: The Flask application instance.  Used to push an application
            context so DB helpers work correctly inside the thread.
        interval: Seconds to sleep between poll cycles.
        pollers: List of :class:`ProviderPoller` instances to invoke each
            cycle.  Defaults to the three built-in pollers configured from
            the app's config.
    """

    def __init__(
        self,
        app: Any,  # flask.Flask – typed as Any to avoid circular import
        interval: int = _DEFAULT_POLL_INTERVAL,
        pollers: Optional[list[ProviderPoller]] = None,
    ) -> None:
        super().__init__(name="agent_dash-poller", daemon=True)
        self._app = app
        self._interval = interval
        self._pollers = pollers or _build_pollers(app)
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop: poll all providers, sleep, repeat."""
        logger.info(
            "PollerThread started (interval=%ds, providers=%s).",
            self._interval,
            [p.provider_name for p in self._pollers],
        )
        while not self._stop_event.is_set():
            self._run_cycle()
            # Sleep in small increments so stop() is responsive
            elapsed = 0
            while elapsed < self._interval and not self._stop_event.is_set():
                time.sleep(1)
                elapsed += 1
        logger.info("PollerThread stopped.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait up to *timeout* seconds.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        logger.info("PollerThread: stop requested.")
        self._stop_event.set()
        self.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Poll cycle
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Execute one full poll cycle across all configured providers."""
        logger.debug("PollerThread: beginning poll cycle.")
        with self._app.app_context():
            for poller in self._pollers:
                if not poller.is_configured():
                    logger.debug(
                        "Skipping provider %s: API key not configured.",
                        poller.provider_name,
                    )
                    continue
                self._poll_one(poller)
        logger.debug("PollerThread: poll cycle complete.")

    def _poll_one(self, poller: ProviderPoller) -> None:
        """Poll a single provider, persist results, and record the run.

        Args:
            poller: The provider poller to invoke.
        """
        from agent_dash.db import (
            get_provider_id,
            get_or_create_provider,
            record_poll_run,
        )
        from agent_dash.ingest import ingest_records

        provider_name = poller.provider_name
        started_at = _utcnow_iso()
        records_fetched = 0
        error_message: Optional[str] = None

        # Resolve provider_id (creates row if needed)
        display_names = {
            "claude": "Anthropic Claude",
            "openai": "OpenAI",
            "gemini": "Google Gemini",
        }
        display_name = display_names.get(provider_name, provider_name.title())
        provider_id = get_or_create_provider(provider_name, display_name)

        try:
            raw_records = poller.fetch_records()
            records_fetched = len(raw_records)
            if raw_records:
                inserted = ingest_records(raw_records, provider_name)
                logger.info(
                    "Poller: inserted %d records for %s.",
                    inserted, provider_name,
                )
            else:
                logger.info("Poller: no records returned for %s.", provider_name)
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            logger.error(
                "Poller: error fetching %s data: %s", provider_name, exc
            )

        finished_at = _utcnow_iso()
        try:
            record_poll_run(
                provider_id=provider_id,
                started_at=started_at,
                finished_at=finished_at,
                records_fetched=records_fetched,
                error_message=error_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Poller: failed to record poll_run for %s: %s",
                provider_name, exc,
            )


# ---------------------------------------------------------------------------
# Module-level singleton management
# ---------------------------------------------------------------------------

_poller_thread: Optional[PollerThread] = None
_poller_lock = threading.Lock()


def start_poller(app: Any) -> Optional[PollerThread]:
    """Create and start the background poller thread.

    This function is idempotent: calling it when a thread is already running
    returns the existing thread without starting a new one.

    Args:
        app: The Flask application instance.  Must have ``POLL_INTERVAL_SECONDS``
            and provider API key config values set.

    Returns:
        The running :class:`PollerThread`, or ``None`` if no providers are
        configured (i.e. all API keys are absent).
    """
    global _poller_thread  # noqa: PLW0603

    with _poller_lock:
        if _poller_thread is not None and _poller_thread.is_alive():
            logger.info("start_poller: thread already running; skipping.")
            return _poller_thread

        interval = int(app.config.get("POLL_INTERVAL_SECONDS", _DEFAULT_POLL_INTERVAL))
        pollers = _build_pollers(app)
        active_pollers = [p for p in pollers if p.is_configured()]

        if not active_pollers:
            logger.info(
                "start_poller: no provider API keys configured; poller not started."
            )
            return None

        thread = PollerThread(app, interval=interval, pollers=active_pollers)
        thread.start()
        _poller_thread = thread
        logger.info(
            "start_poller: thread started (interval=%ds, providers=%s).",
            interval,
            [p.provider_name for p in active_pollers],
        )
        return thread


def stop_poller(timeout: float = 5.0) -> None:
    """Stop the running background poller thread, if any.

    Args:
        timeout: Maximum seconds to wait for the thread to exit cleanly.
    """
    global _poller_thread  # noqa: PLW0603

    with _poller_lock:
        if _poller_thread is None or not _poller_thread.is_alive():
            logger.info("stop_poller: no running thread to stop.")
            return
        _poller_thread.stop(timeout=timeout)
        _poller_thread = None
        logger.info("stop_poller: thread stopped.")


def get_poller_status() -> dict[str, Any]:
    """Return a dict describing the current poller thread status.

    Returns:
        A dict with keys:
        * ``running`` (bool): Whether the thread is alive.
        * ``thread_name`` (str | None): Thread name if running.
        * ``interval_seconds`` (int | None): Configured poll interval.
    """
    with _poller_lock:
        if _poller_thread is None or not _poller_thread.is_alive():
            return {"running": False, "thread_name": None, "interval_seconds": None}
        return {
            "running": True,
            "thread_name": _poller_thread.name,
            "interval_seconds": _poller_thread._interval,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pollers(app: Any) -> list[ProviderPoller]:
    """Instantiate provider pollers from the Flask app configuration.

    Args:
        app: A Flask application instance with API key config values.

    Returns:
        List of configured :class:`ProviderPoller` instances.
    """
    anthropic_key: Optional[str] = app.config.get("ANTHROPIC_API_KEY")
    openai_key: Optional[str] = app.config.get("OPENAI_API_KEY")
    gemini_key: Optional[str] = app.config.get("GEMINI_API_KEY")

    return [
        ClaudePoller(api_key=anthropic_key),
        OpenAIPoller(api_key=openai_key),
        GeminiPoller(api_key=gemini_key),
    ]


def _synthetic_record(
    provider: str,
    note: str = "poll_ping",
) -> dict[str, Any]:
    """Create a synthetic usage record for API ping events.

    These records let operators see in the dashboard that polling is active
    even for providers that lack a usage-export API.  They contribute zero
    tokens and zero cost.

    Args:
        provider: Canonical provider name.
        note: A short string describing the reason for the synthetic record.

    Returns:
        A raw record dict understood by the provider-specific normalizer.
    """
    now = _utcnow_iso()
    return {
        "id": f"{provider}-ping-{int(time.time())}",
        "created_at": now,
        "model": None,
        "task_type": "api_health_check",
        # Claude field names (also recognized by OpenAI normalizer via
        # prompt_tokens fallback)
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "stop_reason": "end_turn",
        "status": "success",
        "_note": note,
    }


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        A string of the form ``'2024-01-15T12:34:56.789012+00:00'``.
    """
    return datetime.now(tz=timezone.utc).isoformat()
