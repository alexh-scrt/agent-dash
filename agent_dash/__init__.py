"""agent_dash - A lightweight web dashboard for AI coding agent usage analytics.

This package provides a Flask-based web application that aggregates API usage
logs from multiple AI coding agent providers (Claude, OpenAI, Gemini) into a
single unified view, enabling engineering teams to track token spend, task
completion rates, and estimated ROI from their AI agent investments.

Usage::

    from agent_dash import create_app

    app = create_app()
    app.run()

Or via the CLI entry point::

    agent-dash
"""

from agent_dash.app import create_app

__all__ = ["create_app"]
__version__ = "0.1.0"
