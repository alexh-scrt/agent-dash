"""Tests for the Flask application factory defined in agent_dash.app.

Verifies that the app can be created, configured, and that core routes
respond correctly.
"""

from __future__ import annotations

import pytest

from agent_dash import create_app, __version__


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    """Create a test Flask application pointing at a temporary database."""
    test_config = {
        "DATABASE": str(tmp_path / "test_agent_dash.db"),
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "ENABLE_POLLER": False,
    }
    application = create_app(test_config)
    return application


@pytest.fixture()
def client(app):
    """Return a Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_flask_app(self, app):
        """create_app should return a Flask application instance."""
        from flask import Flask
        assert isinstance(app, Flask)

    def test_app_has_secret_key(self, app):
        """Application must have a SECRET_KEY set."""
        assert app.config["SECRET_KEY"] == "test-secret"

    def test_app_testing_flag(self, app):
        """TESTING flag should be True in the test configuration."""
        assert app.config["TESTING"] is True

    def test_custom_database_path(self, tmp_path):
        """Custom DATABASE path should be reflected in app config."""
        db_path = str(tmp_path / "custom.db")
        app = create_app({"DATABASE": db_path, "TESTING": True})
        assert app.config["DATABASE"] == db_path

    def test_default_config_values(self, tmp_path):
        """Default config values should be set when no override is provided."""
        app = create_app({"DATABASE": str(tmp_path / "d.db"), "TESTING": True})
        assert app.config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024
        assert "csv" in app.config["ALLOWED_EXTENSIONS"]
        assert "json" in app.config["ALLOWED_EXTENSIONS"]

    def test_upload_folder_created(self, app, tmp_path):
        """UPLOAD_FOLDER should be set and the directory should exist."""
        import os
        assert "UPLOAD_FOLDER" in app.config
        assert os.path.isdir(app.config["UPLOAD_FOLDER"])

    def test_create_app_with_dict_config(self, tmp_path):
        """create_app should accept a plain dict as config_object."""
        cfg = {"DATABASE": str(tmp_path / "dict.db"), "TESTING": True, "CUSTOM_KEY": "hello"}
        app = create_app(cfg)
        assert app.config["CUSTOM_KEY"] == "hello"


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestRoutes:
    """Tests for the routes registered by _register_routes."""

    def test_health_check_returns_200(self, client):
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_json(self, client):
        """GET /health should return valid JSON."""
        response = client.get("/health")
        data = response.get_json()
        assert data is not None
        assert data["status"] == "ok"

    def test_health_check_version(self, client):
        """GET /health JSON should include the current package version."""
        response = client.get("/health")
        data = response.get_json()
        assert data["version"] == __version__

    def test_index_returns_200(self, client):
        """GET / should return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_contains_html(self, client):
        """GET / response body should contain HTML markup."""
        response = client.get("/")
        assert b"agent_dash" in response.data

    def test_unknown_route_returns_404(self, client):
        """Unknown routes should return HTTP 404."""
        response = client.get("/nonexistent-route-xyz")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Version / package tests
# ---------------------------------------------------------------------------

class TestPackage:
    """Tests for package-level attributes."""

    def test_version_string(self):
        """__version__ should be a non-empty string."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_create_app_importable(self):
        """create_app should be importable from the top-level package."""
        from agent_dash import create_app as _ca
        assert callable(_ca)
