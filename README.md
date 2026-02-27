# agent_dash

A lightweight web dashboard that aggregates API usage logs from multiple AI coding agent providers (Claude, OpenAI, Gemini) into a single unified view.

Engineering teams can import CSV/JSON log files or enable real-time API polling to track token spend, task completion rates, estimated time saved versus manual effort, and usage concentration across providers. The goal is to give "centaur-phase" teams clear ROI visibility into their AI agent investments without complex infrastructure.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Dashboard](#running-the-dashboard)
- [Importing Log Files](#importing-log-files)
  - [CSV Format Reference](#csv-format-reference)
  - [JSON Format Reference](#json-format-reference)
- [Real-Time API Polling](#real-time-api-polling)
- [Dashboard Overview](#dashboard-overview)
- [API Endpoints](#api-endpoints)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Unified log ingestion** – Import CSV or JSON exports from Anthropic Claude, OpenAI, and Google Gemini with automatic provider detection and schema normalisation.
- **Real-time polling mode** – Background thread polls provider APIs on a configurable schedule and persists results to SQLite automatically.
- **Interactive dashboard** – Per-provider token spend breakdown, task-type distribution, and completion rate charts powered by Chart.js.
- **Time-saved estimator** – Compares AI task durations against a configurable manual-effort baseline to surface ROI metrics (hours saved, % reduction).
- **Provider concentration analysis** – Spend-share percentages, Herfindahl-Hirschman Index (HHI), and over-reliance warnings when a single vendor exceeds a configurable threshold.
- **Zero heavy infrastructure** – Runs on SQLite; no external database, message queue, or cloud services required.

---

## Requirements

- Python 3.10 or later
- pip

All Python dependencies are declared in `pyproject.toml` and pinned in `requirements.txt`:

| Package | Minimum version |
|---|---|
| Flask | 3.0 |
| pandas | 2.1 |
| requests | 2.31 |
| python-dotenv | 1.0 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/agent_dash.git
cd agent_dash
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the package in editable mode (recommended for development):

```bash
pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest` and `pytest-cov` for running tests.

### 4. Initialise the database

The database is created automatically on first startup. No manual migration steps are required.

---

## Configuration

Configuration is read from environment variables and an optional `.env` file in the project root.

Create a `.env` file by copying the example below:

```dotenv
# .env – local overrides (never commit API keys to version control)

# Flask
SECRET_KEY=change-me-in-production
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=false

# Database
# AGENT_DASH_DATABASE_URL=/absolute/path/to/agent_dash.db

# Background poller
ENABLE_POLLER=false
POLL_INTERVAL_SECONDS=300

# Manual-effort baseline for time-saved calculations (minutes per task)
MANUAL_EFFORT_MINUTES_PER_TASK=30

# Provider API keys (required only for real-time polling)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
```

### Configuration reference

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-change-in-production` | Flask session signing key. **Must** be overridden in production. |
| `FLASK_HOST` | `127.0.0.1` | Interface for the development server to listen on. |
| `FLASK_PORT` | `5000` | TCP port for the development server. |
| `FLASK_DEBUG` | `false` | Enable Flask debug / auto-reload mode. |
| `AGENT_DASH_DATABASE_URL` | `<project_root>/agent_dash.db` | Absolute path to the SQLite database file. |
| `ENABLE_POLLER` | `false` | Set to `true` to start the background API poller on startup. |
| `POLL_INTERVAL_SECONDS` | `300` | How often (in seconds) the poller fetches fresh data from provider APIs. |
| `MANUAL_EFFORT_MINUTES_PER_TASK` | `30` | Assumed minutes a developer would spend on each task manually. Used for time-saved estimates. |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key for Claude polling. |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key for usage polling. |
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio API key for Gemini polling. |

---

## Running the Dashboard

### Using the CLI entry point

```bash
agent-dash
```

This reads `FLASK_HOST`, `FLASK_PORT`, and `FLASK_DEBUG` from the environment (or `.env`).

### Using Flask directly

```bash
flask --app agent_dash.app:create_app run --port 5000
```

### Programmatic usage

```python
from agent_dash import create_app

app = create_app()
app.run(host="0.0.0.0", port=5000, debug=False)
```

Once running, open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Importing Log Files

Navigate to **http://127.0.0.1:5000/upload** or click **Import Logs** in the dashboard header.

1. Drag and drop (or browse to) a `.csv` or `.json` file.
2. Optionally select a **Provider Override** if auto-detection fails.
3. Click **Import File**.

Successfully imported records are immediately reflected in the dashboard.

### CSV Format Reference

CSV files must have a header row. Column names are case-insensitive. Unknown columns are preserved in the `raw_payload` field.

#### Claude (Anthropic)

```
id,created_at,model,task_type,input_tokens,output_tokens,cost_usd,duration_seconds,stop_reason
req_abc,2024-03-01T10:00:00Z,claude-3-5-sonnet-20241022,code_generation,100,50,0.005,2.5,end_turn
```

| Column | Aliases | Description |
|---|---|---|
| `id` | `request_id`, `log_id` | External request identifier |
| `created_at` | `timestamp`, `date`, `logged_at` | Event timestamp (ISO-8601 or Unix epoch) |
| `model` | `anthropic_model`, `model_id` | Model variant |
| `task_type` | `type`, `task` | Task category |
| `input_tokens` | `prompt_tokens`, `input_token_count` | Prompt token count |
| `output_tokens` | `completion_tokens`, `output_token_count` | Completion token count |
| `total_tokens` | — | Auto-computed if absent |
| `cost_usd` | `cost`, `total_cost`, `price_usd` | Estimated cost in USD |
| `duration_seconds` | — | Wall-clock duration |
| `duration_ms` | `latency_ms` | Duration in ms (auto-converted) |
| `stop_reason` | `status`, `finish_reason` | `end_turn` \| `error` \| `cancelled` |

#### OpenAI

```
id,created,model,object,prompt_tokens,completion_tokens,cost_usd,finish_reason
chatcmpl-xyz,1709288400,gpt-4o,chat.completion,200,80,0.012,stop
```

| Column | Aliases | Description |
|---|---|---|
| `id` | — | Request identifier |
| `created` | `created_at`, `timestamp`, `date` | Unix timestamp or ISO-8601 |
| `model` | — | Model variant (e.g. `gpt-4o`) |
| `object` | `type`, `task_type` | e.g. `chat.completion` |
| `prompt_tokens` | — | Flat or nested under `usage{}` |
| `completion_tokens` | — | Flat or nested under `usage{}` |
| `total_tokens` | — | Flat or nested; auto-computed if absent |
| `finish_reason` | `status` | `stop` \| `length` \| `content_filter` \| `cancelled` |
| `cost_usd` | `cost`, `total_cost` | Estimated cost in USD |
| `duration_seconds` | `duration_ms` | Wall-clock duration |

#### Gemini (Google)

```
name,create_time,model,prompt_token_count,candidates_token_count,finish_reason,safety_ratings,cost_usd
operations/abc,2024-03-01T12:00:00Z,gemini-1.5-pro,150,60,STOP,[],0.003
```

| Column | Aliases | Description |
|---|---|---|
| `name` | `id`, `request_id` | Operation name / identifier |
| `create_time` | `created_at`, `timestamp`, `date` | Event timestamp |
| `model` | `model_version`, `model_id` | Model variant |
| `task_type` | `type` | Task category |
| `prompt_token_count` | — | Flat or in `usageMetadata{}` |
| `candidates_token_count` | — | Flat or in `usageMetadata{}` |
| `finish_reason` | `status` | `STOP` \| `SAFETY` \| `RECITATION` \| `UNSPECIFIED` |
| `safety_ratings` | — | Used for provider detection only |
| `cost_usd` | `cost` | Estimated cost in USD |

### JSON Format Reference

Three JSON structures are supported:

```json
// 1. Array of records
[{"input_tokens": 100, "output_tokens": 50, ...}, ...]

// 2. Wrapped object ("data", "items", "logs", "records", "results", or "usage" key)
{"data": [{...}, ...]}

// 3. Single record
{"input_tokens": 100, "output_tokens": 50, "model": "claude-3-haiku"}
```

Field names within each record follow the same conventions as the CSV column reference above. OpenAI JSON API responses with nested `usage` objects and Gemini responses with nested `usageMetadata` objects are supported natively.

---

## Real-Time API Polling

Set `ENABLE_POLLER=true` in your `.env` file and provide at least one provider API key. A background daemon thread will launch on startup and poll each configured provider every `POLL_INTERVAL_SECONDS`.

```dotenv
ENABLE_POLLER=true
POLL_INTERVAL_SECONDS=300
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

### Provider polling notes

| Provider | Behaviour |
|---|---|
| **Claude (Anthropic)** | Anthropic does not expose a usage-export API. The poller performs a lightweight `GET /v1/models` health-check and records a synthetic ping entry. Import actual token data via CSV/JSON. |
| **OpenAI** | Attempts `GET /v1/usage` (requires an organisation-level API key). Falls back to a `GET /v1/models` ping on HTTP 403/404. Usage records are stored when available. |
| **Gemini (Google)** | No public usage-export API. The poller verifies the API key via `GET /v1beta/models` and records a synthetic ping entry. Import actual token data via CSV/JSON. |

Poll outcomes (success or failure) are recorded in the `poll_runs` table and visible via `GET /api/poller/status`.

---

## Dashboard Overview

### Summary cards

| Card | Description |
|---|---|
| **Total Tokens** | Sum of all tokens (prompt + completion) across selected filters |
| **Est. Cost (USD)** | Total estimated cost across all providers |
| **Tasks** | Total log entries with success / error / cancelled breakdown |
| **Success Rate** | Percentage of tasks with `success` status |
| **Time Saved** | Estimated hours saved vs. manual baseline |
| **Time Saved %** | Percentage reduction in effort vs. baseline |
| **Dominant Provider** | Provider with the highest cost share |
| **Active Days** | Days with at least one recorded activity |

### Charts

- **Token Spend by Provider** – Doughnut chart showing each provider's share of total tokens.
- **Task Completion Rates** – Doughnut chart of success / error / cancelled counts.
- **Task Type Distribution** – Horizontal bar chart of top-8 task categories by count.
- **Cost Share by Provider** – Vertical bar chart with per-provider cost percentages.
- **Daily Token Usage Trend** – Multi-axis line chart with token totals and task counts over time.

### Filters

Use the filter bar at the top of the dashboard to scope data:

- **From / To** – ISO date range (e.g. `2024-03-01` to `2024-03-31`)
- **Provider** – Restrict to a single provider

Filters are reflected in all charts, cards, and the recent-activity table. Charts refresh automatically every 60 seconds via AJAX.

### Concentration warnings

If a single provider accounts for ≥ 70% of cost, tokens, or tasks (and at least 5 tasks have been recorded), a yellow warning banner is displayed below the filter bar recommending provider diversification.

---

## API Endpoints

All endpoints return JSON. They accept the same filter query parameters unless noted.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Main dashboard HTML |
| `GET` | `/upload` | Upload form HTML |
| `POST` | `/upload` | Process an uploaded log file |
| `GET` | `/health` | Health check (`{"status": "ok", "version": "..."}`) |
| `GET` | `/api/stats` | Full summary statistics (all metric functions combined) |
| `GET` | `/api/providers` | List of registered providers |
| `GET` | `/api/recent` | Recent usage log records (default limit: 20) |
| `GET` | `/api/poller/status` | Background poller thread status |

### Query parameters

| Parameter | Type | Description |
|---|---|---|
| `since` | ISO-8601 date string | Lower bound for `logged_at` |
| `until` | ISO-8601 date string | Upper bound for `logged_at` |
| `provider` | string | Filter to a single provider (`claude`, `openai`, `gemini`) |
| `limit` | integer | `/api/recent` only – maximum records returned (default `20`) |
| `manual_effort_minutes` | float | `/api/stats` only – override manual-effort baseline |

### Example requests

```bash
# Full statistics for March 2024
curl "http://localhost:5000/api/stats?since=2024-03-01&until=2024-03-31"

# Claude-only recent activity
curl "http://localhost:5000/api/recent?provider=claude&limit=50"

# Poller status
curl "http://localhost:5000/api/poller/status"
```

---

## Development

### Running Tests

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=agent_dash --cov-report=term-missing
```

Run a specific test module:

```bash
pytest tests/test_ingest.py -v
pytest tests/test_metrics.py -v
```

### Project Structure

```
agent_dash/
├── __init__.py          # Package init; exposes create_app
├── app.py               # Flask application factory and route registration
├── db.py                # SQLite schema, lifecycle hooks, query helpers
├── ingest.py            # CSV/JSON log parsing and normalisation
├── metrics.py           # Aggregated metric computation engine
├── poller.py            # Background API polling thread
├── static/
│   └── dashboard.js     # Chart.js initialisation and AJAX refresh
└── templates/
    ├── index.html        # Main dashboard Jinja2 template
    └── upload.html       # File upload form template

tests/
├── __init__.py
├── test_app_factory.py  # Flask app factory and route tests
├── test_db.py           # Database schema and query helper tests
├── test_ingest.py       # Log parsing and normalisation tests
├── test_metrics.py      # Metric computation tests
└── test_poller.py       # Background poller unit tests

pyproject.toml           # Package metadata and dependency declarations
requirements.txt         # Pinned runtime dependencies
README.md                # This file
```

### Database schema

Three tables are created automatically on first startup:

- **`providers`** – Lookup table for Claude, OpenAI, and Gemini (seeded automatically).
- **`usage_logs`** – Unified log records with token counts, costs, durations, and status.
- **`poll_runs`** – Audit trail of background polling attempts (success and failure).

The SQLite file is located at `<project_root>/agent_dash.db` by default. Override with `AGENT_DASH_DATABASE_URL`.

### Adding a new provider

1. Add a normalizer function `normalize_<provider>_record()` in `agent_dash/ingest.py` following the existing pattern.
2. Register the provider name in `KNOWN_PROVIDERS`, `_<PROVIDER>_CSV_COLUMNS`, and `_<PROVIDER>_JSON_FIELDS`.
3. Add the normalizer to the dispatch dict in `_normalize_and_insert()`.
4. Optionally implement a `ProviderPoller` subclass in `agent_dash/poller.py` and add it to `_build_pollers()`.
5. Seed the new provider in `agent_dash/db.py` `_SEED_PROVIDERS`.

---

## Troubleshooting

### "Could not determine provider from file"

Provider auto-detection uses column names (CSV) or field names (JSON) to identify the source. If your export uses non-standard field names:

- Select the provider manually from the **Provider Override** dropdown on the upload page.
- Or pass `provider='claude'|'openai'|'gemini'` explicitly when calling `ingest_file()` programmatically.

### Database is locked

SQLite WAL mode is enabled by default to reduce contention between the background poller and web requests. If you see `database is locked` errors:

- Ensure only one agent_dash process is accessing the database file at a time.
- Check that the database directory has write permissions.

### Charts do not render

Chart.js is loaded from the jsDelivr CDN. If your network blocks CDN access:

- Download Chart.js (`chart.umd.min.js`) and place it in `agent_dash/static/`.
- Update the `<script>` tag in `agent_dash/templates/index.html` to point to the local file.

### Poller thread not starting

- Verify `ENABLE_POLLER=true` is set in your `.env` or environment.
- At least one provider API key must be configured; the poller skips providers without keys.
- Check the application logs for `start_poller: no provider API keys configured` messages.

### Import returns 0 records

- Verify the file contains at least one data row (not just headers).
- Check that the file encoding is UTF-8.
- Review the application logs for per-record skip warnings (e.g. unparseable timestamps).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
