"""Metrics computation engine for agent_dash.

This module computes aggregated metrics from stored usage log data:

* :func:`get_token_spend` – per-provider and total token counts and estimated costs.
* :func:`get_task_completion_rates` – success, error, and cancellation rates overall
  and per provider.
* :func:`get_time_saved_estimates` – wall-clock duration vs. configurable manual-effort
  baselines to surface ROI metrics.
* :func:`get_provider_concentration` – spend-share percentages and over-reliance
  indicators.
* :func:`get_task_type_distribution` – breakdown of task types across providers.
* :func:`get_daily_usage_trend` – day-by-day token and cost aggregates for trend lines.
* :func:`get_summary_stats` – single-call convenience wrapper returning all key
  metrics suitable for the dashboard landing page.

All functions accept optional ``since`` / ``until`` ISO-8601 date strings for
time-range filtering and an optional ``provider`` name to scope results to a
single provider.  They must be called within an active Flask application context
because they rely on :func:`agent_dash.db.get_db`.

Return types
------------
All public functions return plain Python dicts or lists of dicts so that the
Flask routes can serialise them directly with ``flask.jsonify``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from agent_dash.db import (
    get_all_providers,
    get_provider_id,
    query,
    query_one,
    rows_to_dicts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default manual-effort minutes assumed per AI task when no app config present.
_DEFAULT_MANUAL_EFFORT_MINUTES = 30.0

#: Concentration threshold above which a provider is flagged as over-relied-upon.
_CONCENTRATION_WARNING_THRESHOLD = 0.70  # 70 %

#: Minimum number of tasks required before a concentration warning is emitted.
_CONCENTRATION_MIN_TASKS = 5


# ---------------------------------------------------------------------------
# Token spend
# ---------------------------------------------------------------------------

def get_token_spend(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Compute token usage and cost aggregates.

    Returns totals across all providers as well as a per-provider breakdown.

    Args:
        since: ISO-8601 lower bound for ``logged_at`` (inclusive). ``None``
            means no lower bound.
        until: ISO-8601 upper bound for ``logged_at`` (inclusive). ``None``
            means no upper bound.
        provider: Restrict results to a single provider canonical name (e.g.
            ``'claude'``). ``None`` returns all providers.

    Returns:
        A dict with the following structure::

            {
                "total_prompt_tokens":     int,
                "total_completion_tokens": int,
                "total_tokens":            int,
                "total_cost_usd":          float,
                "by_provider": [
                    {
                        "provider":            str,
                        "display_name":        str,
                        "prompt_tokens":       int,
                        "completion_tokens":   int,
                        "total_tokens":        int,
                        "cost_usd":            float,
                        "record_count":        int,
                    },
                    ...
                ],
            }
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    # Aggregate per provider
    sql = f"""
        SELECT
            p.name            AS provider,
            p.display_name    AS display_name,
            COALESCE(SUM(ul.prompt_tokens), 0)     AS prompt_tokens,
            COALESCE(SUM(ul.completion_tokens), 0) AS completion_tokens,
            COALESCE(SUM(ul.total_tokens), 0)      AS total_tokens,
            COALESCE(SUM(ul.cost_usd), 0.0)        AS cost_usd,
            COUNT(ul.id)                           AS record_count
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.provider_id
        ORDER BY total_tokens DESC;
    """
    rows = query(sql, params)
    by_provider = [dict(row) for row in rows]

    total_prompt = sum(r["prompt_tokens"] for r in by_provider)
    total_completion = sum(r["completion_tokens"] for r in by_provider)
    total_tokens = sum(r["total_tokens"] for r in by_provider)
    total_cost = sum(r["cost_usd"] for r in by_provider)

    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "by_provider": by_provider,
    }


# ---------------------------------------------------------------------------
# Task completion rates
# ---------------------------------------------------------------------------

def get_task_completion_rates(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Compute task success, error, and cancellation rates.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A dict with the following structure::

            {
                "total_tasks":       int,
                "success_count":     int,
                "error_count":       int,
                "cancelled_count":   int,
                "success_rate":      float,   # 0.0 – 1.0
                "error_rate":        float,
                "cancellation_rate": float,
                "by_provider": [
                    {
                        "provider":          str,
                        "display_name":      str,
                        "total_tasks":       int,
                        "success_count":     int,
                        "error_count":       int,
                        "cancelled_count":   int,
                        "success_rate":      float,
                        "error_rate":        float,
                        "cancellation_rate": float,
                    },
                    ...
                ],
            }
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            p.name                                            AS provider,
            p.display_name                                    AS display_name,
            COUNT(ul.id)                                      AS total_tasks,
            SUM(CASE WHEN ul.status = 'success'   THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN ul.status = 'error'     THEN 1 ELSE 0 END) AS error_count,
            SUM(CASE WHEN ul.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.provider_id
        ORDER BY total_tasks DESC;
    """
    rows = query(sql, params)

    by_provider: list[dict[str, Any]] = []
    for row in rows:
        total = int(row["total_tasks"])
        success = int(row["success_count"])
        error = int(row["error_count"])
        cancelled = int(row["cancelled_count"])
        by_provider.append({
            "provider": row["provider"],
            "display_name": row["display_name"],
            "total_tasks": total,
            "success_count": success,
            "error_count": error,
            "cancelled_count": cancelled,
            "success_rate": _safe_rate(success, total),
            "error_rate": _safe_rate(error, total),
            "cancellation_rate": _safe_rate(cancelled, total),
        })

    total_tasks = sum(r["total_tasks"] for r in by_provider)
    total_success = sum(r["success_count"] for r in by_provider)
    total_error = sum(r["error_count"] for r in by_provider)
    total_cancelled = sum(r["cancelled_count"] for r in by_provider)

    return {
        "total_tasks": total_tasks,
        "success_count": total_success,
        "error_count": total_error,
        "cancelled_count": total_cancelled,
        "success_rate": _safe_rate(total_success, total_tasks),
        "error_rate": _safe_rate(total_error, total_tasks),
        "cancellation_rate": _safe_rate(total_cancelled, total_tasks),
        "by_provider": by_provider,
    }


# ---------------------------------------------------------------------------
# Time-saved estimates
# ---------------------------------------------------------------------------

def get_time_saved_estimates(
    manual_effort_minutes_per_task: Optional[float] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Estimate time saved by AI agents vs. manual effort.

    For each successful task the AI duration (``duration_seconds``) is
    compared against a configurable manual-effort baseline.  When
    ``duration_seconds`` is ``NULL`` an average of known durations is used
    as a proxy, or zero if no durations are recorded.

    The manual-effort baseline is read (in order of preference) from:
    1. The *manual_effort_minutes_per_task* argument.
    2. ``flask.current_app.config['MANUAL_EFFORT_MINUTES_PER_TASK']``.
    3. The module-level default of 30 minutes.

    Args:
        manual_effort_minutes_per_task: Override baseline in minutes per task.
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A dict with the following structure::

            {
                "manual_effort_minutes_per_task": float,
                "total_tasks":                   int,
                "successful_tasks":              int,
                "total_ai_seconds":              float,
                "total_manual_seconds":          float,
                "time_saved_seconds":            float,
                "time_saved_hours":              float,
                "time_saved_percent":            float,   # 0.0 – 100.0
                "avg_ai_seconds_per_task":        float,
                "by_provider": [
                    {
                        "provider":               str,
                        "display_name":           str,
                        "successful_tasks":       int,
                        "total_ai_seconds":       float,
                        "total_manual_seconds":   float,
                        "time_saved_seconds":     float,
                        "time_saved_hours":       float,
                        "avg_ai_seconds_per_task": float,
                    },
                    ...
                ],
            }
    """
    baseline_minutes = _resolve_manual_effort(manual_effort_minutes_per_task)
    baseline_seconds = baseline_minutes * 60.0

    clauses, params = _build_filters(since, until, provider)
    # Only count successful tasks for time-saved computation
    clauses = clauses + ["ul.status = 'success'"]
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            p.name                                       AS provider,
            p.display_name                               AS display_name,
            COUNT(ul.id)                                 AS successful_tasks,
            COALESCE(SUM(ul.duration_seconds), 0.0)      AS total_ai_seconds,
            COALESCE(AVG(ul.duration_seconds), 0.0)      AS avg_ai_seconds
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.provider_id
        ORDER BY successful_tasks DESC;
    """
    rows = query(sql, params)

    by_provider: list[dict[str, Any]] = []
    for row in rows:
        tasks = int(row["successful_tasks"])
        ai_secs = float(row["total_ai_seconds"])
        avg_ai = float(row["avg_ai_seconds"])
        manual_secs = tasks * baseline_seconds
        saved = max(0.0, manual_secs - ai_secs)
        by_provider.append({
            "provider": row["provider"],
            "display_name": row["display_name"],
            "successful_tasks": tasks,
            "total_ai_seconds": round(ai_secs, 3),
            "total_manual_seconds": round(manual_secs, 3),
            "time_saved_seconds": round(saved, 3),
            "time_saved_hours": round(saved / 3600.0, 4),
            "avg_ai_seconds_per_task": round(avg_ai, 3),
        })

    total_tasks = sum(r["successful_tasks"] for r in by_provider)
    total_ai_secs = sum(r["total_ai_seconds"] for r in by_provider)
    total_manual_secs = total_tasks * baseline_seconds
    total_saved = max(0.0, total_manual_secs - total_ai_secs)
    avg_ai = (total_ai_secs / total_tasks) if total_tasks > 0 else 0.0
    saved_pct = _safe_rate(total_saved, total_manual_secs) * 100.0 if total_manual_secs > 0 else 0.0

    # Fetch overall task count (including non-success) for context
    all_clauses, all_params = _build_filters(since, until, provider)
    all_where = _where_clause(all_clauses)
    count_sql = f"SELECT COUNT(id) AS cnt FROM usage_logs ul {all_where};"
    count_row = query_one(count_sql, all_params)
    total_all_tasks = int(count_row["cnt"]) if count_row else 0

    return {
        "manual_effort_minutes_per_task": baseline_minutes,
        "total_tasks": total_all_tasks,
        "successful_tasks": total_tasks,
        "total_ai_seconds": round(total_ai_secs, 3),
        "total_manual_seconds": round(total_manual_secs, 3),
        "time_saved_seconds": round(total_saved, 3),
        "time_saved_hours": round(total_saved / 3600.0, 4),
        "time_saved_percent": round(saved_pct, 2),
        "avg_ai_seconds_per_task": round(avg_ai, 3),
        "by_provider": by_provider,
    }


# ---------------------------------------------------------------------------
# Provider concentration analysis
# ---------------------------------------------------------------------------

def get_provider_concentration(
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> dict[str, Any]:
    """Analyse provider spend concentration and flag over-reliance.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.

    Returns:
        A dict with the following structure::

            {
                "total_cost_usd":   float,
                "total_tokens":     int,
                "total_tasks":      int,
                "by_provider": [
                    {
                        "provider":           str,
                        "display_name":       str,
                        "cost_usd":           float,
                        "total_tokens":       int,
                        "task_count":         int,
                        "cost_share":         float,   # 0.0 – 1.0
                        "token_share":        float,
                        "task_share":         float,
                    },
                    ...
                ],
                "concentration_warnings": [
                    {
                        "provider":     str,
                        "display_name": str,
                        "metric":       str,   # 'cost' | 'tokens' | 'tasks'
                        "share":        float,
                        "threshold":    float,
                        "message":      str,
                    },
                    ...
                ],
                "dominant_provider": str | None,   # provider with highest cost share
                "hhi_cost":         float,          # Herfindahl-Hirschman Index (0–1)
            }
    """
    clauses, params = _build_filters(since, None, None)
    if until:
        clauses.append("ul.logged_at <= ?")
        params.append(until)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            p.name                                      AS provider,
            p.display_name                              AS display_name,
            COALESCE(SUM(ul.cost_usd), 0.0)             AS cost_usd,
            COALESCE(SUM(ul.total_tokens), 0)           AS total_tokens,
            COUNT(ul.id)                                AS task_count
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.provider_id
        ORDER BY cost_usd DESC;
    """
    rows = query(sql, params)

    total_cost = sum(float(r["cost_usd"]) for r in rows)
    total_tokens = sum(int(r["total_tokens"]) for r in rows)
    total_tasks = sum(int(r["task_count"]) for r in rows)

    by_provider: list[dict[str, Any]] = []
    for row in rows:
        cost = float(row["cost_usd"])
        tokens = int(row["total_tokens"])
        tasks = int(row["task_count"])
        by_provider.append({
            "provider": row["provider"],
            "display_name": row["display_name"],
            "cost_usd": round(cost, 6),
            "total_tokens": tokens,
            "task_count": tasks,
            "cost_share": _safe_rate(cost, total_cost),
            "token_share": _safe_rate(tokens, total_tokens),
            "task_share": _safe_rate(tasks, total_tasks),
        })

    # Concentration warnings
    warnings: list[dict[str, Any]] = []
    if total_tasks >= _CONCENTRATION_MIN_TASKS:
        for entry in by_provider:
            for metric, share_key in [
                ("cost", "cost_share"),
                ("tokens", "token_share"),
                ("tasks", "task_share"),
            ]:
                share = entry[share_key]
                if share >= _CONCENTRATION_WARNING_THRESHOLD:
                    warnings.append({
                        "provider": entry["provider"],
                        "display_name": entry["display_name"],
                        "metric": metric,
                        "share": round(share, 4),
                        "threshold": _CONCENTRATION_WARNING_THRESHOLD,
                        "message": (
                            f"{entry['display_name']} accounts for "
                            f"{share * 100:.1f}% of {metric} – consider "
                            "diversifying across providers."
                        ),
                    })

    # Herfindahl-Hirschman Index on cost shares (measure of market concentration)
    hhi = sum(entry["cost_share"] ** 2 for entry in by_provider)

    dominant = by_provider[0]["provider"] if by_provider else None

    return {
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "total_tasks": total_tasks,
        "by_provider": by_provider,
        "concentration_warnings": warnings,
        "dominant_provider": dominant,
        "hhi_cost": round(hhi, 4),
    }


# ---------------------------------------------------------------------------
# Task type distribution
# ---------------------------------------------------------------------------

def get_task_type_distribution(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Return a breakdown of task types and their token / cost footprint.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A dict with the following structure::

            {
                "total_tasks": int,
                "task_types": [
                    {
                        "task_type":   str | None,
                        "count":       int,
                        "share":       float,
                        "total_tokens": int,
                        "cost_usd":    float,
                    },
                    ...
                ],
            }
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            ul.task_type                                AS task_type,
            COUNT(ul.id)                                AS count,
            COALESCE(SUM(ul.total_tokens), 0)           AS total_tokens,
            COALESCE(SUM(ul.cost_usd), 0.0)             AS cost_usd
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.task_type
        ORDER BY count DESC;
    """
    rows = query(sql, params)
    total_tasks = sum(int(r["count"]) for r in rows)

    task_types = [
        {
            "task_type": row["task_type"],
            "count": int(row["count"]),
            "share": _safe_rate(int(row["count"]), total_tasks),
            "total_tokens": int(row["total_tokens"]),
            "cost_usd": round(float(row["cost_usd"]), 6),
        }
        for row in rows
    ]

    return {
        "total_tasks": total_tasks,
        "task_types": task_types,
    }


# ---------------------------------------------------------------------------
# Daily usage trend
# ---------------------------------------------------------------------------

def get_daily_usage_trend(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate token counts and costs by calendar day.

    Returns one entry per UTC day within the requested range.  Days with no
    activity are omitted from the result.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A dict with the following structure::

            {
                "days": [
                    {
                        "date":             str,    # YYYY-MM-DD
                        "total_tokens":     int,
                        "prompt_tokens":    int,
                        "completion_tokens": int,
                        "cost_usd":         float,
                        "task_count":       int,
                        "success_count":    int,
                        "error_count":      int,
                    },
                    ...
                ],
                "total_days": int,
            }
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            SUBSTR(ul.logged_at, 1, 10)                        AS date,
            COALESCE(SUM(ul.total_tokens), 0)                  AS total_tokens,
            COALESCE(SUM(ul.prompt_tokens), 0)                 AS prompt_tokens,
            COALESCE(SUM(ul.completion_tokens), 0)             AS completion_tokens,
            COALESCE(SUM(ul.cost_usd), 0.0)                    AS cost_usd,
            COUNT(ul.id)                                       AS task_count,
            SUM(CASE WHEN ul.status = 'success'   THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN ul.status = 'error'     THEN 1 ELSE 0 END) AS error_count
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY date
        ORDER BY date ASC;
    """
    rows = query(sql, params)

    days = [
        {
            "date": row["date"],
            "total_tokens": int(row["total_tokens"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "cost_usd": round(float(row["cost_usd"]), 6),
            "task_count": int(row["task_count"]),
            "success_count": int(row["success_count"]),
            "error_count": int(row["error_count"]),
        }
        for row in rows
    ]

    return {
        "days": days,
        "total_days": len(days),
    }


# ---------------------------------------------------------------------------
# Per-provider daily trend
# ---------------------------------------------------------------------------

def get_provider_daily_trend(
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate token counts and costs by calendar day AND provider.

    Useful for stacked / multi-line chart rendering on the dashboard.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.

    Returns:
        A dict with the following structure::

            {
                "series": [
                    {
                        "provider":     str,
                        "display_name": str,
                        "days": [
                            {
                                "date":         str,
                                "total_tokens": int,
                                "cost_usd":     float,
                                "task_count":   int,
                            },
                            ...
                        ],
                    },
                    ...
                ],
            }
    """
    clauses, params = _build_filters(since, until, None)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            p.name                                      AS provider,
            p.display_name                              AS display_name,
            SUBSTR(ul.logged_at, 1, 10)                 AS date,
            COALESCE(SUM(ul.total_tokens), 0)           AS total_tokens,
            COALESCE(SUM(ul.cost_usd), 0.0)             AS cost_usd,
            COUNT(ul.id)                                AS task_count
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.provider_id, date
        ORDER BY p.name ASC, date ASC;
    """
    rows = query(sql, params)

    # Group by provider
    series_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        prov = row["provider"]
        if prov not in series_map:
            series_map[prov] = {
                "provider": prov,
                "display_name": row["display_name"],
                "days": [],
            }
        series_map[prov]["days"].append({
            "date": row["date"],
            "total_tokens": int(row["total_tokens"]),
            "cost_usd": round(float(row["cost_usd"]), 6),
            "task_count": int(row["task_count"]),
        })

    return {"series": list(series_map.values())}


# ---------------------------------------------------------------------------
# Model usage breakdown
# ---------------------------------------------------------------------------

def get_model_usage(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Return token and cost aggregates broken down by model variant.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A dict with the following structure::

            {
                "models": [
                    {
                        "model":         str | None,
                        "provider":      str,
                        "display_name":  str,
                        "task_count":    int,
                        "total_tokens":  int,
                        "cost_usd":      float,
                        "share":         float,
                    },
                    ...
                ],
                "total_tasks": int,
            }
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            ul.model                                    AS model,
            p.name                                      AS provider,
            p.display_name                              AS display_name,
            COUNT(ul.id)                                AS task_count,
            COALESCE(SUM(ul.total_tokens), 0)           AS total_tokens,
            COALESCE(SUM(ul.cost_usd), 0.0)             AS cost_usd
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        GROUP BY ul.model, ul.provider_id
        ORDER BY task_count DESC;
    """
    rows = query(sql, params)
    total_tasks = sum(int(r["task_count"]) for r in rows)

    models = [
        {
            "model": row["model"],
            "provider": row["provider"],
            "display_name": row["display_name"],
            "task_count": int(row["task_count"]),
            "total_tokens": int(row["total_tokens"]),
            "cost_usd": round(float(row["cost_usd"]), 6),
            "share": _safe_rate(int(row["task_count"]), total_tasks),
        }
        for row in rows
    ]

    return {
        "models": models,
        "total_tasks": total_tasks,
    }


# ---------------------------------------------------------------------------
# Recent activity
# ---------------------------------------------------------------------------

def get_recent_activity(
    limit: int = 20,
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the most recent usage log records as plain dicts.

    Args:
        limit: Maximum number of records to return (default 20).
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.

    Returns:
        A list of dicts, each representing one usage log row including
        ``provider_name`` and ``provider_display_name`` columns.
    """
    clauses, params = _build_filters(since, until, provider)
    where = _where_clause(clauses)

    sql = f"""
        SELECT
            ul.id,
            ul.external_id,
            ul.logged_at,
            ul.model,
            ul.task_type,
            ul.prompt_tokens,
            ul.completion_tokens,
            ul.total_tokens,
            ul.cost_usd,
            ul.duration_seconds,
            ul.status,
            ul.imported_at,
            p.name         AS provider_name,
            p.display_name AS provider_display_name
        FROM usage_logs ul
        JOIN providers p ON p.id = ul.provider_id
        {where}
        ORDER BY ul.logged_at DESC
        LIMIT ?;
    """
    params.append(limit)
    rows = query(sql, params)
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Summary stats – single convenience wrapper
# ---------------------------------------------------------------------------

def get_summary_stats(
    since: Optional[str] = None,
    until: Optional[str] = None,
    provider: Optional[str] = None,
    manual_effort_minutes_per_task: Optional[float] = None,
) -> dict[str, Any]:
    """Return a consolidated summary of all key metrics for the dashboard.

    This is a convenience wrapper that calls the individual metric functions
    and merges their results into a single dict.  It is designed to power
    the main dashboard page with a single function call.

    Args:
        since: ISO-8601 lower bound for ``logged_at``.
        until: ISO-8601 upper bound for ``logged_at``.
        provider: Restrict results to a single provider.
        manual_effort_minutes_per_task: Manual-effort baseline override.

    Returns:
        A dict containing::

            {
                "generated_at":       str,   # ISO-8601 UTC timestamp
                "filters": {
                    "since":    str | None,
                    "until":    str | None,
                    "provider": str | None,
                },
                "token_spend":         dict,  # from get_token_spend()
                "completion_rates":    dict,  # from get_task_completion_rates()
                "time_saved":          dict,  # from get_time_saved_estimates()
                "concentration":       dict,  # from get_provider_concentration()
                "task_distribution":   dict,  # from get_task_type_distribution()
                "daily_trend":         dict,  # from get_daily_usage_trend()
            }
    """
    logger.debug(
        "Computing summary stats (since=%s, until=%s, provider=%s).",
        since, until, provider,
    )

    token_spend = get_token_spend(since=since, until=until, provider=provider)
    completion_rates = get_task_completion_rates(since=since, until=until, provider=provider)
    time_saved = get_time_saved_estimates(
        manual_effort_minutes_per_task=manual_effort_minutes_per_task,
        since=since,
        until=until,
        provider=provider,
    )
    concentration = get_provider_concentration(since=since, until=until)
    task_distribution = get_task_type_distribution(since=since, until=until, provider=provider)
    daily_trend = get_daily_usage_trend(since=since, until=until, provider=provider)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "filters": {
            "since": since,
            "until": until,
            "provider": provider,
        },
        "token_spend": token_spend,
        "completion_rates": completion_rates,
        "time_saved": time_saved,
        "concentration": concentration,
        "task_distribution": task_distribution,
        "daily_trend": daily_trend,
    }


# ---------------------------------------------------------------------------
# Provider list helper
# ---------------------------------------------------------------------------

def get_providers_list() -> list[dict[str, Any]]:
    """Return all registered providers as a list of dicts.

    Convenience wrapper around :func:`agent_dash.db.get_all_providers` that
    returns plain dicts instead of :class:`sqlite3.Row` objects.

    Returns:
        A list of dicts with keys ``id``, ``name``, ``display_name``,
        ``created_at``.
    """
    return rows_to_dicts(get_all_providers())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_filters(
    since: Optional[str],
    until: Optional[str],
    provider: Optional[str],
) -> tuple[list[str], list[Any]]:
    """Build WHERE clause fragments and bind parameters.

    Args:
        since: ISO-8601 lower bound for ``logged_at``, or ``None``.
        until: ISO-8601 upper bound for ``logged_at``, or ``None``.
        provider: Canonical provider name to filter on, or ``None``.

    Returns:
        A 2-tuple of (list of SQL fragment strings, list of bind params).

    Raises:
        ValueError: If *provider* is specified but not found in the database.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if since:
        clauses.append("ul.logged_at >= ?")
        params.append(since)
    if until:
        clauses.append("ul.logged_at <= ?")
        params.append(until)
    if provider:
        pid = get_provider_id(provider.lower())
        if pid is None:
            logger.warning(
                "get_summary_stats: unknown provider %r; filter ignored.", provider
            )
        else:
            clauses.append("ul.provider_id = ?")
            params.append(pid)

    return clauses, params


def _where_clause(clauses: list[str]) -> str:
    """Build a SQL WHERE clause string from a list of condition fragments.

    Args:
        clauses: List of SQL condition strings (without ``AND`` separators).

    Returns:
        A ``WHERE ...`` string, or an empty string if *clauses* is empty.
    """
    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


def _safe_rate(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or 0.0 if denominator is zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        A float in the range ``[0.0, 1.0]`` (clamped) or 0.0 on
        zero-division.
    """
    if denominator == 0:
        return 0.0
    result = numerator / denominator
    # Clamp to [0, 1] to handle floating-point edge cases
    return max(0.0, min(1.0, result))


def _resolve_manual_effort(override: Optional[float]) -> float:
    """Resolve the manual-effort baseline in minutes.

    Priority order:
    1. *override* argument (if not ``None``).
    2. ``current_app.config['MANUAL_EFFORT_MINUTES_PER_TASK']``.
    3. Module-level default (:data:`_DEFAULT_MANUAL_EFFORT_MINUTES`).

    Args:
        override: Caller-supplied baseline in minutes, or ``None``.

    Returns:
        Resolved baseline as a positive float.
    """
    if override is not None:
        return max(0.0, float(override))

    try:
        from flask import current_app
        value = current_app.config.get(
            "MANUAL_EFFORT_MINUTES_PER_TASK",
            _DEFAULT_MANUAL_EFFORT_MINUTES,
        )
        return max(0.0, float(value))
    except RuntimeError:
        # Outside application context – use module default
        return _DEFAULT_MANUAL_EFFORT_MINUTES
