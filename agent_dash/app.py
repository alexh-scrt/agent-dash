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
    # Wire database
    # ------------------------------------------------------------------
    from agent_dash.db import wire_db
    wire_db(app)

    # ------------------------------------------------------------------
    # Register core routes
    # ------------------------------------------------------------------
    _register_routes(app)

    # ------------------------------------------------------------------
    # Optionally start background poller
    # ------------------------------------------------------------------
    if app.config.get("ENABLE_POLLER", False):
        from agent_dash.poller import start_poller
        start_poller(app)

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


def _allowed_file(filename: str, allowed_extensions: frozenset) -> bool:
    """Return True if the filename has an allowed extension.

    Args:
        filename: The uploaded file's filename.
        allowed_extensions: Set of lowercase allowed extensions.

    Returns:
        True if the extension is allowed, False otherwise.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def _register_routes(app: Flask) -> None:
    """Register all URL routes on the Flask application.

    Args:
        app: The :class:`flask.Flask` application to register routes on.
    """
    import json
    import os
    from pathlib import Path as _Path
    from werkzeug.utils import secure_filename
    from flask import (
        jsonify,
        redirect,
        render_template,
        request,
        url_for,
        flash,
    )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health-check endpoint returning application status.

        Returns:
            JSON object with ``status`` and ``version`` fields.
        """
        from agent_dash import __version__
        return jsonify({"status": "ok", "version": __version__}), 200

    # ------------------------------------------------------------------
    # Main dashboard
    # ------------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def index():
        """Main dashboard page.

        Renders the dashboard template with summary statistics and chart data.
        Accepts optional query parameters for filtering:

        * ``since`` – ISO-8601 date lower bound
        * ``until`` – ISO-8601 date upper bound
        * ``provider`` – provider name filter

        Returns:
            Rendered HTML dashboard.
        """
        from agent_dash.metrics import get_summary_stats, get_providers_list
        from agent_dash.poller import get_poller_status

        since = request.args.get("since") or None
        until = request.args.get("until") or None
        provider = request.args.get("provider") or None

        try:
            stats = get_summary_stats(
                since=since,
                until=until,
                provider=provider,
            )
        except Exception as exc:
            logger.error("Error computing summary stats: %s", exc)
            stats = {
                "generated_at": "",
                "filters": {"since": since, "until": until, "provider": provider},
                "token_spend": {"total_tokens": 0, "total_cost_usd": 0.0, "by_provider": []},
                "completion_rates": {"total_tasks": 0, "success_rate": 0.0, "error_rate": 0.0, "cancellation_rate": 0.0, "by_provider": []},
                "time_saved": {"time_saved_hours": 0.0, "successful_tasks": 0, "total_manual_seconds": 0.0, "total_ai_seconds": 0.0, "time_saved_percent": 0.0, "manual_effort_minutes_per_task": 30.0, "by_provider": []},
                "concentration": {"total_cost_usd": 0.0, "dominant_provider": None, "hhi_cost": 0.0, "by_provider": [], "concentration_warnings": []},
                "task_distribution": {"total_tasks": 0, "task_types": []},
                "daily_trend": {"days": [], "total_days": 0},
            }

        providers = get_providers_list()
        poller_status = get_poller_status()

        # Serialise stats to JSON for JavaScript consumption
        stats_json = json.dumps(stats)

        return render_template(
            "index.html",
            stats=stats,
            stats_json=stats_json,
            providers=providers,
            poller_status=poller_status,
            since=since or "",
            until=until or "",
            selected_provider=provider or "",
        )

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        """File upload page for CSV / JSON log import.

        GET  – renders the upload form.
        POST – processes an uploaded file, runs ingest, redirects to dashboard.

        Returns:
            Rendered upload form (GET) or redirect to dashboard (POST).
        """
        from agent_dash.ingest import (
            ingest_file,
            KNOWN_PROVIDERS,
            IngestError,
            UnknownProviderError,
            MalformedLogError,
        )

        if request.method == "GET":
            return render_template("upload.html")

        # POST – handle file upload
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return render_template("upload.html"), 400

        file = request.files["file"]
        if not file or file.filename == "":
            flash("No file selected.", "error")
            return render_template("upload.html"), 400

        filename = secure_filename(file.filename or "")
        if not _allowed_file(filename, app.config["ALLOWED_EXTENSIONS"]):
            flash(
                f"File type not allowed. Please upload a CSV or JSON file.",
                "error",
            )
            return render_template("upload.html"), 400

        # Provider override from form
        provider_override = request.form.get("provider") or None
        if provider_override and provider_override.lower() not in KNOWN_PROVIDERS:
            provider_override = None

        # Save file to upload folder
        upload_dir = _Path(app.config["UPLOAD_FOLDER"])
        file_path = upload_dir / filename
        try:
            file.save(str(file_path))
        except OSError as exc:
            logger.error("Failed to save uploaded file: %s", exc)
            flash("Failed to save uploaded file. Please try again.", "error")
            return render_template("upload.html"), 500

        # Run ingestion
        try:
            count = ingest_file(
                filepath=file_path,
                provider=provider_override,
            )
            flash(
                f"Successfully imported {count} record(s) from '{filename}'.",
                "success",
            )
            logger.info(
                "Imported %d records from uploaded file '%s'.", count, filename
            )
        except FileNotFoundError:
            flash("Uploaded file could not be found for processing.", "error")
            return render_template("upload.html"), 500
        except UnknownProviderError as exc:
            flash(
                f"Could not determine provider from file. "
                f"Please select a provider manually. ({exc})",
                "error",
            )
            return render_template("upload.html"), 400
        except MalformedLogError as exc:
            flash(f"File could not be parsed: {exc}", "error")
            return render_template("upload.html"), 400
        except IngestError as exc:
            flash(f"Ingestion failed: {exc}", "error")
            return render_template("upload.html"), 400
        except Exception as exc:
            logger.exception("Unexpected error during file ingest: %s", exc)
            flash("An unexpected error occurred during import.", "error")
            return render_template("upload.html"), 500
        finally:
            # Clean up the temporary file
            try:
                os.remove(str(file_path))
            except OSError:
                pass

        return redirect(url_for("index"))

    # ------------------------------------------------------------------
    # API endpoints – JSON
    # ------------------------------------------------------------------

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        """JSON API endpoint returning summary statistics.

        Accepts optional query parameters:
        * ``since`` – ISO-8601 lower bound
        * ``until`` – ISO-8601 upper bound
        * ``provider`` – provider name
        * ``manual_effort_minutes`` – float override for time-saved baseline

        Returns:
            JSON object matching :func:`agent_dash.metrics.get_summary_stats`.
        """
        from agent_dash.metrics import get_summary_stats

        since = request.args.get("since") or None
        until = request.args.get("until") or None
        provider = request.args.get("provider") or None
        manual_effort = request.args.get("manual_effort_minutes", type=float)

        try:
            stats = get_summary_stats(
                since=since,
                until=until,
                provider=provider,
                manual_effort_minutes_per_task=manual_effort,
            )
            return jsonify(stats), 200
        except Exception as exc:
            logger.error("Error in /api/stats: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/providers", methods=["GET"])
    def api_providers():
        """JSON API endpoint returning the list of registered providers.

        Returns:
            JSON array of provider objects.
        """
        from agent_dash.metrics import get_providers_list

        try:
            return jsonify(get_providers_list()), 200
        except Exception as exc:
            logger.error("Error in /api/providers: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/recent", methods=["GET"])
    def api_recent():
        """JSON API endpoint returning recent usage log records.

        Accepts:
        * ``limit`` – integer, default 20
        * ``since`` / ``until`` / ``provider`` filters

        Returns:
            JSON array of recent usage log records.
        """
        from agent_dash.metrics import get_recent_activity

        limit = request.args.get("limit", default=20, type=int)
        since = request.args.get("since") or None
        until = request.args.get("until") or None
        provider = request.args.get("provider") or None

        try:
            records = get_recent_activity(
                limit=limit,
                since=since,
                until=until,
                provider=provider,
            )
            return jsonify(records), 200
        except Exception as exc:
            logger.error("Error in /api/recent: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/poller/status", methods=["GET"])
    def api_poller_status():
        """JSON API endpoint returning background poller status.

        Returns:
            JSON object with poller running state and configuration.
        """
        from agent_dash.poller import get_poller_status

        return jsonify(get_poller_status()), 200

    logger.debug(
        "Routes registered: %s",
        [str(rule) for rule in app.url_map.iter_rules()],
    )


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
