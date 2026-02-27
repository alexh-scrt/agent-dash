# agent_dash 🤖📊

*One dashboard to rule your AI agent spend — Claude, OpenAI, and Gemini unified.*

agent_dash is a lightweight Flask web dashboard that aggregates API usage logs from multiple AI coding agent providers into a single view. Import CSV/JSON exports or enable live API polling to track token spend, task completion rates, and estimated time saved versus manual effort. Built for engineering teams who want clear ROI visibility into their AI agent investments — no complex infrastructure required.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/agent_dash.git
cd agent_dash
pip install -r requirements.txt

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env to add your provider API keys (optional — import-only mode works without them)

# 3. Run the dashboard
agent-dash
# or: python -m agent_dash

# 4. Open your browser
open http://localhost:5000
```

That's it. Navigate to the **Import** page to upload your first log file, or set `ENABLE_POLLER=true` in `.env` to start live polling.

---

## Features

- **Unified log ingestion** — Import CSV and JSON exports from Claude (Anthropic), OpenAI Codex, and Gemini with automatic provider detection and schema normalization.
- **Real-time API polling** — A configurable background thread fetches usage data from provider APIs on a schedule and persists results to a local SQLite database.
- **Interactive charts** — Per-provider token spend breakdowns, task-type distributions, daily trend lines, and completion rate gauges powered by Chart.js.
- **Time-saved estimator** — Compares AI task durations against configurable manual-effort baselines to surface concrete ROI numbers.
- **Provider concentration analysis** — Highlights over-reliance on a single vendor with spend-share percentages, HHI scores, and trend lines.

---

## Usage Examples

### Import a log file via the web UI

Navigate to `http://localhost:5000/upload` and drag-and-drop your CSV or JSON export. Provider is auto-detected from column names and field presence.

### Import via CLI / script

```python
from agent_dash import create_app
from agent_dash.ingest import ingest_file

app = create_app()
with app.app_context():
    count = ingest_file("path/to/openai_usage.csv")
    print(f"Inserted {count} records")
```

### Expected CSV format (OpenAI)

```csv
timestamp,model,prompt_tokens,completion_tokens,total_tokens,task_type,status,duration_seconds
2024-06-01T10:00:00Z,gpt-4o,512,128,640,code_review,success,4.2
2024-06-01T10:05:00Z,gpt-4o,1024,256,1280,refactor,success,8.7
```

### Expected JSON format (Claude)

```json
[
  {
    "created_at": "2024-06-01T10:00:00Z",
    "model": "claude-3-5-sonnet-20241022",
    "input_tokens": 800,
    "output_tokens": 200,
    "task_type": "code_generation",
    "stop_reason": "end_turn",
    "duration_ms": 3200
  }
]
```

### Enable live API polling

```bash
# .env
ENABLE_POLLER=true
POLL_INTERVAL_SECONDS=300

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

### Query metrics programmatically

```python
from agent_dash import create_app
from agent_dash.metrics import get_summary_stats, get_provider_concentration

app = create_app()
with app.app_context():
    stats = get_summary_stats(since="2024-06-01", until="2024-06-30")
    print(stats["total_tokens"], stats["estimated_cost_usd"])

    concentration = get_provider_concentration()
    for p in concentration["providers"]:
        print(f"{p['name']}: {p['spend_share_pct']:.1f}%")
```

---

## Project Structure

```
agent_dash/
├── pyproject.toml              # Project metadata and package config
├── requirements.txt            # Pinned runtime dependencies
├── .env.example                # Environment variable template
├── agent_dash/
│   ├── __init__.py             # Package init; exposes create_app
│   ├── app.py                  # Flask app factory, routes, CLI entry point
│   ├── db.py                   # SQLite schema, init, and query helpers
│   ├── ingest.py               # CSV/JSON log parser and normalizer
│   ├── poller.py               # Background API polling thread
│   ├── metrics.py              # Aggregated metric computation engine
│   ├── templates/
│   │   ├── index.html          # Main dashboard (charts + summary cards)
│   │   └── upload.html         # Log file import form
│   └── static/
│       └── dashboard.js        # Chart.js init and AJAX live-refresh
└── tests/
    ├── test_ingest.py          # Log parsing and normalization tests
    ├── test_metrics.py         # Metric computation tests
    ├── test_db.py              # Database schema and helper tests
    ├── test_poller.py          # Poller thread lifecycle tests
    └── test_app_factory.py     # Flask app factory and route tests
```

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev` | Flask secret key — **change in production** |
| `DATABASE_PATH` | `agent_dash.db` | Path to the SQLite database file |
| `ENABLE_POLLER` | `false` | Set to `true` to enable background API polling |
| `POLL_INTERVAL_SECONDS` | `300` | How often the poller fetches provider data (seconds) |
| `ANTHROPIC_API_KEY` | — | API key for Claude usage polling |
| `OPENAI_API_KEY` | — | API key for OpenAI usage polling |
| `GEMINI_API_KEY` | — | API key for Gemini usage polling |
| `MANUAL_EFFORT_BASELINE_SECONDS` | `1800` | Per-task manual effort baseline for time-saved estimates |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `PORT` | `5000` | Port the server listens on |

> **Note:** Provider API keys are only required when `ENABLE_POLLER=true`. Import-only mode works without any keys.

### Running Tests

```bash
pip install pytest
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
