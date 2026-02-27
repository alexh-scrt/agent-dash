"""Flask application factory, route registration, and server entry point.

This module defines the ``create_app`` factory function which initialises the
Flask application with all configuration, blueprints, and extension hooks.
It also provides a ``main`` entry point consumed by the ``agent-dash`` CLI
command declared in ``pyproject.toml``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from flask import Flask

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env file from project root (if present)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

class DefaultConfig:
    """Default configuration values for the agent_dash Flask application."""

    #: SQLite database file path (can be overridden via AGENT_DASH_DATABASE_URL env var)
    DATABASE: str = str(_PROJECT_ROOT / "agent_dash.db")

    #: Flask secret key for session signing (must be overridden in production)
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

    #: Maximum content length for file uploads (16 MB)
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024

    #: Allowed extensions for log file uploads
    ALLOWED_EXTENSIONS: frozenset = frozenset({"csv", "json"})

    #: Polling interval in seconds for background API fetches
    POLL_INTERVAL_SECONDS: int = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))

    #: Whether to start background poller on app startup
    ENABLE_POLLER: bool = os.environ.get("ENABLE_POLLER", "false").lower() == "true"

    #: Minutes of manual effort assumed per task for time-saved estimates
    MANUAL_EFFORT_MINUTES_PER_TASK: float = float(
        os.environ.get("MANUAL_EFFORT_MINUTES_PER_TASK", "30.0")
    )

    #: Provider API keys (optional; only needed for real-time polling mode)
    ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(config_object: Optional[object] = None) -> Flask:
    """Create and configure the Flask application instance.

    This factory function follows the Flask application-factory pattern,
    allowing multiple instances to be created with different configurations
    (e.g. for testing).

    Args:
        config_object: An optional configuration object or dict that will be
            loaded on top of :class:`DefaultConfig`.  If ``None``, values are
            read purely from environment variables and :class:`DefaultConfig`.

    Returns:
        A fully configured :class:`flask.Flask` application instance.

    Example::

        app = create_app()
        app.run(debug=True)
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    app.config.from_object(DefaultConfig)

    # Allow DATABASE path override via environment variable
    if db_url := os.environ.get("AGENT_DASH_DATABASE_URL"):
        app.config["DATABASE"] = db_url
        logger.info("Using database path from AGENT_DASH_DATABASE_URL: %s", db_url)

    if config_object is not None:
        if isinstance(config_object, dict):
            app.config.update(config_object)
        else:
            app.config.from_object(config_object)
        logger.debug("Loaded custom configuration: %s", config_object)

    # ------------------------------------------------------------------
    # Ensure instance / upload folders exist
    # ------------------------------------------------------------------
    _ensure_directory(Path(app.config["DATABASE"]).parent)

    upload_folder = _PROJECT_ROOT / "uploads"
    _ensure_directory(upload_folder)
    app.config["UPLOAD_FOLDER"] = str(upload_folder)

    # ------------------------------------------------------------------
    # Register core routes (placeholder until phase 5)
    # ------------------------------------------------------------------
    _register_routes(app)

    # ------------------------------------------------------------------
    # Log startup summary
    # ------------------------------------------------------------------
    logger.info(
        "agent_dash application created | database=%s | poller=%s",
        app.config["DATABASE"],
        app.config["ENABLE_POLLER"],
    )

    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_directory(path: Path) -> None:
    """Create *path* and all intermediate parents if they do not already exist.

    Args:
        path: The directory path to create.

    Raises:
        OSError: If the directory cannot be created due to permission issues.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create directory %s: %s", path, exc)
        raise


def _register_routes(app: Flask) -> None:
    """Register all URL routes on the Flask application.

    Routes are defined inline here during the scaffold phase and will be
    extended in later phases (metrics, upload, API endpoints).

    Args:
        app: The :class:`flask.Flask` application to register routes on.
    """
    from flask import jsonify

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health-check endpoint returning application status.

        Returns:
            JSON object with ``status`` and ``version`` fields.
        """
        from agent_dash import __version__
        return jsonify({"status": "ok", "version": __version__}), 200

    @app.route("/", methods=["GET"])
    def index():
        """Temporary landing page; replaced by the full dashboard in phase 5.

        Returns:
            A plain HTML holding page.
        """
        html = (
            "<!doctype html>"
            "<html><head><title>agent_dash</title></head>"
            "<body>"
            "<h1>agent_dash</h1>"
            "<p>Dashboard coming soon. "
            "See <code>/health</code> for status.</p>"
            "</body></html>"
        )
        return html, 200

    logger.debug("Routes registered: %s", [str(rule) for rule in app.url_map.iter_rules()])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point invoked by the ``agent-dash`` console script.

    Reads host, port, and debug settings from environment variables::

        FLASK_HOST  (default: 127.0.0.1)
        FLASK_PORT  (default: 5000)
        FLASK_DEBUG (default: false)

    Example::

        $ agent-dash
        $ FLASK_PORT=8080 FLASK_DEBUG=true agent-dash
    """
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app = create_app()
    logger.info("Starting agent_dash on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
