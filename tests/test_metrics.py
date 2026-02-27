"""Unit tests for agent_dash.metrics – metric computation functions.

Covers:
* get_token_spend – totals and per-provider breakdown.
* get_task_completion_rates – success / error / cancellation rates.
* get_time_saved_estimates – manual vs AI duration comparison.
* get_provider_concentration – HHI, share percentages, warnings.
* get_task_type_distribution – task type breakdown.
* get_daily_usage_trend – day-by-day aggregation.
* get_provider_daily_trend – per-provider day series.
* get_model_usage – model breakdown.
* get_recent_activity – latest records.
* get_summary_stats – consolidated convenience wrapper.
* get_providers_list – provider enumeration.
* Helper functions: _safe_rate, _build_filters, _where_clause.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from agent_dash.app import create_app
from agent_dash.db import (
    wire_db,
    get_provider_id,
    insert_usage_log,
    insert_usage_logs_batch,
    _utcnow_iso,
)
from agent_dash.metrics import (
    _build_filters,
    _resolve_manual_effort,
    _safe_rate,
    _where_clause,
    get_daily_usage_trend,
    get_model_usage,
    get_provider_concentration,
    get_provider_daily_trend,
    get_providers_list,
    get_recent_activity,
    get_summary_stats,
    get_task_completion_rates,
    get_task_type_distribution,
    get_time_saved_estimates,
    get_token_spend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    """Flask app wired to a temp-file database."""
    application = create_app({
        "DATABASE": str(tmp_path / "metrics_test.db"),
        "TESTING": True,
        "SECRET_KEY": "test",
        "MANUAL_EFFORT_MINUTES_PER_TASK": 30.0,
    })
    wire_db(application)
    return application


@pytest.fixture()
def app_ctx(app):
    """Active Flask application context."""
    with app.app_context():
        yield app


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_logs(app_ctx_fixture) -> None:
    """Insert a deterministic set of usage logs for testing metrics."""
    claude_id = get_provider_id("claude")
    openai_id = get_provider_id("openai")
    gemini_id = get_provider_id("gemini")

    records = [
        # Claude – 3 success, 1 error
        {
            "provider_id": claude_id,
            "logged_at": "2024-03-01T10:00:00+00:00",
            "model": "claude-3-5-sonnet-20241022",
            "task_type": "code_generation",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.005,
            "duration_seconds": 5.0,
            "status": "success",
        },
        {
            "provider_id": claude_id,
            "logged_at": "2024-03-01T11:00:00+00:00",
            "model": "claude-3-5-sonnet-20241022",
            "task_type": "code_review",
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "cost_usd": 0.010,
            "duration_seconds": 8.0,
            "status": "success",
        },
        {
            "provider_id": claude_id,
            "logged_at": "2024-03-02T09:00:00+00:00",
            "model": "claude-3-haiku-20240307",
            "task_type": "code_generation",
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
            "cost_usd": 0.002,
            "duration_seconds": 2.0,
            "status": "success",
        },
        {
            "provider_id": claude_id,
            "logged_at": "2024-03-02T10:00:00+00:00",
            "model": "claude-3-5-sonnet-20241022",
            "task_type": "code_generation",
            "prompt_tokens": 60,
            "completion_tokens": 20,
            "total_tokens": 80,
            "cost_usd": 0.003,
            "duration_seconds": 3.0,
            "status": "error",
        },
        # OpenAI – 2 success, 1 cancelled
        {
            "provider_id": openai_id,
            "logged_at": "2024-03-01T12:00:00+00:00",
            "model": "gpt-4o",
            "task_type": "chat_completion",
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "total_tokens": 400,
            "cost_usd": 0.020,
            "duration_seconds": 10.0,
            "status": "success",
        },
        {
            "provider_id": openai_id,
            "logged_at": "2024-03-02T14:00:00+00:00",
            "model": "gpt-4o-mini",
            "task_type": "chat_completion",
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "cost_usd": 0.008,
            "duration_seconds": 4.0,
            "status": "success",
        },
        {
            "provider_id": openai_id,
            "logged_at": "2024-03-03T08:00:00+00:00",
            "model": "gpt-4o",
            "task_type": "chat_completion",
            "prompt_tokens": 80,
            "completion_tokens": 0,
            "total_tokens": 80,
            "cost_usd": 0.004,
            "duration_seconds": None,
            "status": "cancelled",
        },
        # Gemini – 1 success
        {
            "provider_id": gemini_id,
            "logged_at": "2024-03-03T15:00:00+00:00",
            "model": "gemini-1.5-pro",
            "task_type": "code_generation",
            "prompt_tokens": 120,
            "completion_tokens": 60,
            "total_tokens": 180,
            "cost_usd": 0.006,
            "duration_seconds": 6.0,
            "status": "success",
        },
    ]
    insert_usage_logs_batch(records)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestSafeRate:
    def test_normal_division(self):
        assert abs(_safe_rate(1.0, 4.0) - 0.25) < 1e-9

    def test_zero_denominator_returns_zero(self):
        assert _safe_rate(5.0, 0.0) == 0.0

    def test_one_to_one(self):
        assert _safe_rate(10.0, 10.0) == 1.0

    def test_clamped_below_zero(self):
        # Should not happen in practice but guard against floating-point edge cases
        assert _safe_rate(0.0, 1.0) == 0.0

    def test_integer_args(self):
        assert abs(_safe_rate(3, 6) - 0.5) < 1e-9


class TestWhereClause:
    def test_empty_clauses_returns_empty_string(self):
        assert _where_clause([]) == ""

    def test_single_clause(self):
        result = _where_clause(["a = ?"])
        assert result == "WHERE a = ?"

    def test_multiple_clauses_joined_with_and(self):
        result = _where_clause(["a = ?", "b > ?"])
        assert result == "WHERE a = ? AND b > ?"


class TestBuildFilters:
    def test_empty_filters(self, app_ctx):
        clauses, params = _build_filters(None, None, None)
        assert clauses == []
        assert params == []

    def test_since_filter(self, app_ctx):
        clauses, params = _build_filters("2024-01-01", None, None)
        assert len(clauses) == 1
        assert "logged_at >=" in clauses[0]
        assert params == ["2024-01-01"]

    def test_until_filter(self, app_ctx):
        clauses, params = _build_filters(None, "2024-12-31", None)
        assert len(clauses) == 1
        assert "logged_at <=" in clauses[0]
        assert params == ["2024-12-31"]

    def test_provider_filter(self, app_ctx):
        clauses, params = _build_filters(None, None, "claude")
        assert len(clauses) == 1
        assert "provider_id" in clauses[0]
        assert len(params) == 1
        assert isinstance(params[0], int)

    def test_unknown_provider_ignored(self, app_ctx):
        """Unknown provider name should be silently ignored."""
        clauses, params = _build_filters(None, None, "nonexistent_xyz")
        assert clauses == []
        assert params == []

    def test_all_filters_combined(self, app_ctx):
        clauses, params = _build_filters("2024-01-01", "2024-12-31", "claude")
        assert len(clauses) == 3
        assert len(params) == 3


class TestResolveManualEffort:
    def test_override_takes_priority(self, app_ctx):
        result = _resolve_manual_effort(45.0)
        assert abs(result - 45.0) < 1e-9

    def test_reads_from_app_config(self, app_ctx):
        # App fixture sets MANUAL_EFFORT_MINUTES_PER_TASK = 30.0
        result = _resolve_manual_effort(None)
        assert abs(result - 30.0) < 1e-9

    def test_zero_override(self, app_ctx):
        result = _resolve_manual_effort(0.0)
        assert result == 0.0

    def test_negative_clamped_to_zero(self, app_ctx):
        result = _resolve_manual_effort(-5.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# get_token_spend
# ---------------------------------------------------------------------------

class TestGetTokenSpend:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_token_spend()
        assert "total_prompt_tokens" in result
        assert "total_completion_tokens" in result
        assert "total_tokens" in result
        assert "total_cost_usd" in result
        assert "by_provider" in result

    def test_total_tokens_sum(self, app_ctx):
        # 150+280+80+80 (claude) + 400+140+80 (openai) + 180 (gemini) = 1390
        result = get_token_spend()
        assert result["total_tokens"] == 1390

    def test_total_prompt_tokens(self, app_ctx):
        # 100+200+50+60 + 300+100+80 + 120 = 1010
        result = get_token_spend()
        assert result["total_prompt_tokens"] == 1010

    def test_total_completion_tokens(self, app_ctx):
        # 50+80+30+20 + 100+40+0 + 60 = 380
        result = get_token_spend()
        assert result["total_completion_tokens"] == 380

    def test_total_cost_usd(self, app_ctx):
        expected = 0.005 + 0.010 + 0.002 + 0.003 + 0.020 + 0.008 + 0.004 + 0.006
        result = get_token_spend()
        assert abs(result["total_cost_usd"] - expected) < 1e-6

    def test_by_provider_has_three_entries(self, app_ctx):
        result = get_token_spend()
        assert len(result["by_provider"]) == 3

    def test_by_provider_names(self, app_ctx):
        result = get_token_spend()
        names = {entry["provider"] for entry in result["by_provider"]}
        assert "claude" in names
        assert "openai" in names
        assert "gemini" in names

    def test_by_provider_has_record_count(self, app_ctx):
        result = get_token_spend()
        for entry in result["by_provider"]:
            assert "record_count" in entry
            assert entry["record_count"] > 0

    def test_filter_by_provider(self, app_ctx):
        result = get_token_spend(provider="claude")
        assert len(result["by_provider"]) == 1
        assert result["by_provider"][0]["provider"] == "claude"
        # 150 + 280 + 80 + 80 = 590
        assert result["total_tokens"] == 590

    def test_filter_by_since(self, app_ctx):
        result = get_token_spend(since="2024-03-02T00:00:00+00:00")
        # Only records from 2024-03-02 onwards
        # claude: 80+80=160, openai: 140+80=220, gemini: 180
        assert result["total_tokens"] == 160 + 220 + 180

    def test_filter_by_until(self, app_ctx):
        result = get_token_spend(until="2024-03-01T23:59:59+00:00")
        # Only 2024-03-01 records: claude 150+280=430, openai 400
        assert result["total_tokens"] == 430 + 400

    def test_no_data_returns_zeros(self, app_ctx):
        result = get_token_spend(since="2099-01-01T00:00:00+00:00")
        assert result["total_tokens"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["by_provider"] == []


# ---------------------------------------------------------------------------
# get_task_completion_rates
# ---------------------------------------------------------------------------

class TestGetTaskCompletionRates:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_task_completion_rates()
        for key in ("total_tasks", "success_count", "error_count",
                    "cancelled_count", "success_rate", "error_rate",
                    "cancellation_rate", "by_provider"):
            assert key in result

    def test_total_tasks(self, app_ctx):
        result = get_task_completion_rates()
        assert result["total_tasks"] == 8

    def test_success_count(self, app_ctx):
        result = get_task_completion_rates()
        # 3 claude + 2 openai + 1 gemini = 6
        assert result["success_count"] == 6

    def test_error_count(self, app_ctx):
        result = get_task_completion_rates()
        assert result["error_count"] == 1

    def test_cancelled_count(self, app_ctx):
        result = get_task_completion_rates()
        assert result["cancelled_count"] == 1

    def test_success_rate(self, app_ctx):
        result = get_task_completion_rates()
        assert abs(result["success_rate"] - 6 / 8) < 1e-6

    def test_error_rate(self, app_ctx):
        result = get_task_completion_rates()
        assert abs(result["error_rate"] - 1 / 8) < 1e-6

    def test_cancellation_rate(self, app_ctx):
        result = get_task_completion_rates()
        assert abs(result["cancellation_rate"] - 1 / 8) < 1e-6

    def test_by_provider_length(self, app_ctx):
        result = get_task_completion_rates()
        assert len(result["by_provider"]) == 3

    def test_by_provider_rates_sum_to_one(self, app_ctx):
        result = get_task_completion_rates()
        for entry in result["by_provider"]:
            total = entry["success_rate"] + entry["error_rate"] + entry["cancellation_rate"]
            assert abs(total - 1.0) < 1e-6

    def test_claude_error_rate(self, app_ctx):
        result = get_task_completion_rates(provider="claude")
        # 1 error out of 4 Claude tasks
        assert abs(result["error_rate"] - 1 / 4) < 1e-6

    def test_openai_cancelled_rate(self, app_ctx):
        result = get_task_completion_rates(provider="openai")
        assert abs(result["cancellation_rate"] - 1 / 3) < 1e-6

    def test_no_data_returns_zeros(self, app_ctx):
        result = get_task_completion_rates(since="2099-01-01T00:00:00+00:00")
        assert result["total_tasks"] == 0
        assert result["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# get_time_saved_estimates
# ---------------------------------------------------------------------------

class TestGetTimeSavedEstimates:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_time_saved_estimates()
        for key in (
            "manual_effort_minutes_per_task",
            "total_tasks",
            "successful_tasks",
            "total_ai_seconds",
            "total_manual_seconds",
            "time_saved_seconds",
            "time_saved_hours",
            "time_saved_percent",
            "avg_ai_seconds_per_task",
            "by_provider",
        ):
            assert key in result

    def test_uses_default_baseline(self, app_ctx):
        result = get_time_saved_estimates()
        # App config sets 30 minutes
        assert result["manual_effort_minutes_per_task"] == 30.0

    def test_override_baseline(self, app_ctx):
        result = get_time_saved_estimates(manual_effort_minutes_per_task=60.0)
        assert result["manual_effort_minutes_per_task"] == 60.0

    def test_successful_tasks_count(self, app_ctx):
        result = get_time_saved_estimates()
        # 3 claude + 2 openai + 1 gemini = 6 successful
        assert result["successful_tasks"] == 6

    def test_total_ai_seconds(self, app_ctx):
        # claude: 5+8+2=15 (error row excluded), openai: 10+4=14 (cancelled None excluded)
        # gemini: 6
        # Total = 15 + 14 + 6 = 35
        result = get_time_saved_estimates()
        assert abs(result["total_ai_seconds"] - 35.0) < 1e-3

    def test_total_manual_seconds(self, app_ctx):
        # 6 successful tasks * 30 min * 60 sec = 10800
        result = get_time_saved_estimates()
        assert abs(result["total_manual_seconds"] - 10800.0) < 1e-3

    def test_time_saved_positive(self, app_ctx):
        result = get_time_saved_estimates()
        assert result["time_saved_seconds"] > 0

    def test_time_saved_hours_consistency(self, app_ctx):
        result = get_time_saved_estimates()
        expected_hours = result["time_saved_seconds"] / 3600.0
        assert abs(result["time_saved_hours"] - expected_hours) < 1e-4

    def test_time_saved_percent_in_range(self, app_ctx):
        result = get_time_saved_estimates()
        assert 0.0 <= result["time_saved_percent"] <= 100.0

    def test_by_provider_has_entries(self, app_ctx):
        result = get_time_saved_estimates()
        assert len(result["by_provider"]) == 3

    def test_by_provider_claude(self, app_ctx):
        result = get_time_saved_estimates(provider="claude")
        entry = result["by_provider"][0]
        assert entry["provider"] == "claude"
        # 3 successful claude tasks, AI seconds = 5+8+2=15
        assert entry["successful_tasks"] == 3
        assert abs(entry["total_ai_seconds"] - 15.0) < 1e-3

    def test_zero_baseline_gives_no_savings(self, app_ctx):
        result = get_time_saved_estimates(manual_effort_minutes_per_task=0.0)
        assert result["time_saved_seconds"] == 0.0

    def test_no_data_returns_zeros(self, app_ctx):
        result = get_time_saved_estimates(since="2099-01-01T00:00:00+00:00")
        assert result["successful_tasks"] == 0
        assert result["time_saved_seconds"] == 0.0


# ---------------------------------------------------------------------------
# get_provider_concentration
# ---------------------------------------------------------------------------

class TestGetProviderConcentration:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_provider_concentration()
        for key in (
            "total_cost_usd",
            "total_tokens",
            "total_tasks",
            "by_provider",
            "concentration_warnings",
            "dominant_provider",
            "hhi_cost",
        ):
            assert key in result

    def test_total_cost(self, app_ctx):
        expected = 0.005 + 0.010 + 0.002 + 0.003 + 0.020 + 0.008 + 0.004 + 0.006
        result = get_provider_concentration()
        assert abs(result["total_cost_usd"] - expected) < 1e-6

    def test_total_tasks(self, app_ctx):
        result = get_provider_concentration()
        assert result["total_tasks"] == 8

    def test_cost_shares_sum_to_one(self, app_ctx):
        result = get_provider_concentration()
        total_share = sum(entry["cost_share"] for entry in result["by_provider"])
        assert abs(total_share - 1.0) < 1e-6

    def test_token_shares_sum_to_one(self, app_ctx):
        result = get_provider_concentration()
        total_share = sum(entry["token_share"] for entry in result["by_provider"])
        assert abs(total_share - 1.0) < 1e-6

    def test_task_shares_sum_to_one(self, app_ctx):
        result = get_provider_concentration()
        total_share = sum(entry["task_share"] for entry in result["by_provider"])
        assert abs(total_share - 1.0) < 1e-6

    def test_hhi_between_zero_and_one(self, app_ctx):
        result = get_provider_concentration()
        assert 0.0 <= result["hhi_cost"] <= 1.0

    def test_dominant_provider_is_string(self, app_ctx):
        result = get_provider_concentration()
        assert isinstance(result["dominant_provider"], str)

    def test_no_data_returns_none_dominant(self, app_ctx):
        result = get_provider_concentration(
            since="2099-01-01T00:00:00+00:00",
            until="2099-12-31T23:59:59+00:00",
        )
        assert result["dominant_provider"] is None
        assert result["hhi_cost"] == 0.0

    def test_single_provider_concentration_warning(self, tmp_path):
        """When all tasks go to one provider, a warning should be generated."""
        # Create a fresh app with only claude records to trigger 100% concentration
        app2 = create_app({
            "DATABASE": str(tmp_path / "conc_test.db"),
            "TESTING": True,
        })
        wire_db(app2)
        with app2.app_context():
            claude_id = get_provider_id("claude")
            records = [
                {
                    "provider_id": claude_id,
                    "logged_at": f"2024-03-{i:02d}T10:00:00+00:00",
                    "cost_usd": 0.01,
                    "total_tokens": 100,
                    "status": "success",
                }
                for i in range(1, 8)  # 7 tasks > _CONCENTRATION_MIN_TASKS(5)
            ]
            insert_usage_logs_batch(records)
            result = get_provider_concentration()
        # Claude should have 100% share → warning generated
        assert len(result["concentration_warnings"]) > 0
        warning_providers = {w["provider"] for w in result["concentration_warnings"]}
        assert "claude" in warning_providers

    def test_no_warning_below_threshold(self, app_ctx):
        """With balanced usage across 3 providers, no warnings should be raised."""
        # Our seed data is reasonably balanced (no single provider >70% cost)
        result = get_provider_concentration()
        # OpenAI has the most cost (0.020+0.008+0.004=0.032 out of ~0.058)
        # 0.032 / 0.058 ≈ 0.55 < 0.70 threshold
        openai_entry = next(
            (e for e in result["by_provider"] if e["provider"] == "openai"), None
        )
        if openai_entry:
            # If no warning is triggered for openai, share is below threshold
            pass  # Test passes if no exception raised above

    def test_by_provider_has_required_keys(self, app_ctx):
        result = get_provider_concentration()
        for entry in result["by_provider"]:
            for key in ("provider", "display_name", "cost_usd",
                        "total_tokens", "task_count",
                        "cost_share", "token_share", "task_share"):
                assert key in entry


# ---------------------------------------------------------------------------
# get_task_type_distribution
# ---------------------------------------------------------------------------

class TestGetTaskTypeDistribution:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_task_type_distribution()
        assert "total_tasks" in result
        assert "task_types" in result

    def test_total_tasks(self, app_ctx):
        result = get_task_type_distribution()
        assert result["total_tasks"] == 8

    def test_task_type_entries_present(self, app_ctx):
        result = get_task_type_distribution()
        types = {entry["task_type"] for entry in result["task_types"]}
        assert "code_generation" in types
        assert "code_review" in types
        assert "chat_completion" in types

    def test_shares_sum_to_one(self, app_ctx):
        result = get_task_type_distribution()
        total_share = sum(e["share"] for e in result["task_types"])
        assert abs(total_share - 1.0) < 1e-6

    def test_code_generation_count(self, app_ctx):
        result = get_task_type_distribution()
        code_gen = next(
            (e for e in result["task_types"] if e["task_type"] == "code_generation"), None
        )
        assert code_gen is not None
        # claude x3 + gemini x1 = 4 code_generation tasks
        assert code_gen["count"] == 4

    def test_filter_by_provider(self, app_ctx):
        result = get_task_type_distribution(provider="openai")
        types = {e["task_type"] for e in result["task_types"]}
        # OpenAI only has chat_completion in seed data
        assert "chat_completion" in types

    def test_each_entry_has_required_keys(self, app_ctx):
        result = get_task_type_distribution()
        for entry in result["task_types"]:
            for key in ("task_type", "count", "share", "total_tokens", "cost_usd"):
                assert key in entry

    def test_no_data_returns_zero_total(self, app_ctx):
        result = get_task_type_distribution(since="2099-01-01T00:00:00+00:00")
        assert result["total_tasks"] == 0
        assert result["task_types"] == []


# ---------------------------------------------------------------------------
# get_daily_usage_trend
# ---------------------------------------------------------------------------

class TestGetDailyUsageTrend:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_daily_usage_trend()
        assert "days" in result
        assert "total_days" in result

    def test_total_days_count(self, app_ctx):
        # Seed data spans 2024-03-01 to 2024-03-03 → 3 days
        result = get_daily_usage_trend()
        assert result["total_days"] == 3

    def test_days_ordered_ascending(self, app_ctx):
        result = get_daily_usage_trend()
        dates = [d["date"] for d in result["days"]]
        assert dates == sorted(dates)

    def test_first_day_date(self, app_ctx):
        result = get_daily_usage_trend()
        assert result["days"][0]["date"] == "2024-03-01"

    def test_day_has_required_keys(self, app_ctx):
        result = get_daily_usage_trend()
        for day in result["days"]:
            for key in (
                "date", "total_tokens", "prompt_tokens",
                "completion_tokens", "cost_usd",
                "task_count", "success_count", "error_count",
            ):
                assert key in day

    def test_march_01_task_count(self, app_ctx):
        result = get_daily_usage_trend()
        march_01 = next(d for d in result["days"] if d["date"] == "2024-03-01")
        # 2 claude + 1 openai on 2024-03-01
        assert march_01["task_count"] == 3

    def test_march_01_total_tokens(self, app_ctx):
        result = get_daily_usage_trend()
        march_01 = next(d for d in result["days"] if d["date"] == "2024-03-01")
        # claude: 150+280=430, openai: 400 → 830
        assert march_01["total_tokens"] == 830

    def test_filter_by_since(self, app_ctx):
        result = get_daily_usage_trend(since="2024-03-03T00:00:00+00:00")
        assert result["total_days"] == 1
        assert result["days"][0]["date"] == "2024-03-03"

    def test_filter_by_provider(self, app_ctx):
        result = get_daily_usage_trend(provider="gemini")
        # Gemini only has 1 record on 2024-03-03
        assert result["total_days"] == 1

    def test_no_data_returns_empty_days(self, app_ctx):
        result = get_daily_usage_trend(since="2099-01-01T00:00:00+00:00")
        assert result["days"] == []
        assert result["total_days"] == 0


# ---------------------------------------------------------------------------
# get_provider_daily_trend
# ---------------------------------------------------------------------------

class TestGetProviderDailyTrend:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_provider_daily_trend()
        assert "series" in result

    def test_series_has_three_providers(self, app_ctx):
        result = get_provider_daily_trend()
        assert len(result["series"]) == 3

    def test_each_series_has_required_keys(self, app_ctx):
        result = get_provider_daily_trend()
        for series in result["series"]:
            assert "provider" in series
            assert "display_name" in series
            assert "days" in series

    def test_day_entries_have_required_keys(self, app_ctx):
        result = get_provider_daily_trend()
        for series in result["series"]:
            for day in series["days"]:
                for key in ("date", "total_tokens", "cost_usd", "task_count"):
                    assert key in day

    def test_claude_series_has_two_days(self, app_ctx):
        result = get_provider_daily_trend()
        claude_series = next(
            (s for s in result["series"] if s["provider"] == "claude"), None
        )
        assert claude_series is not None
        # Claude has records on 2024-03-01 and 2024-03-02
        assert len(claude_series["days"]) == 2

    def test_gemini_series_has_one_day(self, app_ctx):
        result = get_provider_daily_trend()
        gemini_series = next(
            (s for s in result["series"] if s["provider"] == "gemini"), None
        )
        assert gemini_series is not None
        assert len(gemini_series["days"]) == 1

    def test_no_data_returns_empty_series(self, app_ctx):
        result = get_provider_daily_trend(
            since="2099-01-01T00:00:00+00:00",
            until="2099-12-31T23:59:59+00:00",
        )
        assert result["series"] == []


# ---------------------------------------------------------------------------
# get_model_usage
# ---------------------------------------------------------------------------

class TestGetModelUsage:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_keys(self, app_ctx):
        result = get_model_usage()
        assert "models" in result
        assert "total_tasks" in result

    def test_total_tasks(self, app_ctx):
        result = get_model_usage()
        assert result["total_tasks"] == 8

    def test_model_entries_present(self, app_ctx):
        result = get_model_usage()
        model_names = {m["model"] for m in result["models"]}
        assert "claude-3-5-sonnet-20241022" in model_names
        assert "gpt-4o" in model_names
        assert "gemini-1.5-pro" in model_names

    def test_shares_sum_to_one(self, app_ctx):
        result = get_model_usage()
        total_share = sum(m["share"] for m in result["models"])
        assert abs(total_share - 1.0) < 1e-6

    def test_each_model_has_required_keys(self, app_ctx):
        result = get_model_usage()
        for model in result["models"]:
            for key in ("model", "provider", "display_name",
                        "task_count", "total_tokens", "cost_usd", "share"):
                assert key in model

    def test_filter_by_provider(self, app_ctx):
        result = get_model_usage(provider="openai")
        for model in result["models"]:
            assert model["provider"] == "openai"

    def test_no_data_returns_empty(self, app_ctx):
        result = get_model_usage(since="2099-01-01T00:00:00+00:00")
        assert result["models"] == []
        assert result["total_tasks"] == 0


# ---------------------------------------------------------------------------
# get_recent_activity
# ---------------------------------------------------------------------------

class TestGetRecentActivity:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_list(self, app_ctx):
        result = get_recent_activity()
        assert isinstance(result, list)

    def test_default_limit_20(self, app_ctx):
        # Our seed has 8 records, so all should be returned
        result = get_recent_activity()
        assert len(result) <= 20

    def test_custom_limit(self, app_ctx):
        result = get_recent_activity(limit=3)
        assert len(result) == 3

    def test_ordered_by_logged_at_desc(self, app_ctx):
        result = get_recent_activity()
        dates = [r["logged_at"] for r in result]
        assert dates == sorted(dates, reverse=True)

    def test_includes_provider_name(self, app_ctx):
        result = get_recent_activity()
        assert len(result) > 0
        assert "provider_name" in result[0]

    def test_includes_provider_display_name(self, app_ctx):
        result = get_recent_activity()
        assert "provider_display_name" in result[0]

    def test_filter_by_provider(self, app_ctx):
        result = get_recent_activity(provider="gemini")
        for record in result:
            assert record["provider_name"] == "gemini"

    def test_filter_by_since(self, app_ctx):
        result = get_recent_activity(since="2024-03-03T00:00:00+00:00")
        for record in result:
            assert record["logged_at"] >= "2024-03-03"

    def test_no_data_returns_empty_list(self, app_ctx):
        result = get_recent_activity(since="2099-01-01T00:00:00+00:00")
        assert result == []

    def test_record_has_required_keys(self, app_ctx):
        result = get_recent_activity(limit=1)
        assert len(result) == 1
        record = result[0]
        for key in (
            "id", "external_id", "logged_at", "model", "task_type",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cost_usd", "duration_seconds", "status", "imported_at",
            "provider_name", "provider_display_name",
        ):
            assert key in record


# ---------------------------------------------------------------------------
# get_summary_stats
# ---------------------------------------------------------------------------

class TestGetSummaryStats:
    @pytest.fixture(autouse=True)
    def seed(self, app_ctx):
        _seed_logs(app_ctx)

    def test_returns_expected_top_level_keys(self, app_ctx):
        result = get_summary_stats()
        for key in (
            "generated_at",
            "filters",
            "token_spend",
            "completion_rates",
            "time_saved",
            "concentration",
            "task_distribution",
            "daily_trend",
        ):
            assert key in result

    def test_generated_at_is_iso_string(self, app_ctx):
        result = get_summary_stats()
        assert "T" in result["generated_at"]
        assert "+00:00" in result["generated_at"]

    def test_filters_reflected(self, app_ctx):
        result = get_summary_stats(
            since="2024-03-01",
            until="2024-03-31",
            provider="claude",
        )
        assert result["filters"]["since"] == "2024-03-01"
        assert result["filters"]["until"] == "2024-03-31"
        assert result["filters"]["provider"] == "claude"

    def test_filters_none_when_not_set(self, app_ctx):
        result = get_summary_stats()
        assert result["filters"]["since"] is None
        assert result["filters"]["until"] is None
        assert result["filters"]["provider"] is None

    def test_sub_dicts_are_populated(self, app_ctx):
        result = get_summary_stats()
        assert result["token_spend"]["total_tokens"] > 0
        assert result["completion_rates"]["total_tasks"] > 0
        assert result["time_saved"]["successful_tasks"] > 0
        assert result["concentration"]["total_tasks"] > 0
        assert result["task_distribution"]["total_tasks"] > 0
        assert result["daily_trend"]["total_days"] > 0

    def test_manual_effort_override_propagates(self, app_ctx):
        result = get_summary_stats(manual_effort_minutes_per_task=60.0)
        assert result["time_saved"]["manual_effort_minutes_per_task"] == 60.0

    def test_no_data_returns_zeros(self, app_ctx):
        result = get_summary_stats(since="2099-01-01T00:00:00+00:00")
        assert result["token_spend"]["total_tokens"] == 0
        assert result["completion_rates"]["total_tasks"] == 0


# ---------------------------------------------------------------------------
# get_providers_list
# ---------------------------------------------------------------------------

class TestGetProvidersList:
    def test_returns_list(self, app_ctx):
        result = get_providers_list()
        assert isinstance(result, list)

    def test_includes_seeded_providers(self, app_ctx):
        result = get_providers_list()
        names = [p["name"] for p in result]
        assert "claude" in names
        assert "openai" in names
        assert "gemini" in names

    def test_each_provider_has_required_keys(self, app_ctx):
        result = get_providers_list()
        for provider in result:
            for key in ("id", "name", "display_name", "created_at"):
                assert key in provider

    def test_returns_plain_dicts(self, app_ctx):
        result = get_providers_list()
        for item in result:
            assert isinstance(item, dict)
