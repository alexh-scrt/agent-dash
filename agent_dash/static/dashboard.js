/**
 * dashboard.js – Chart.js visualisations and live metric polling for agent_dash.
 *
 * This script reads server-rendered data from window.__AGENT_DASH_STATS__,
 * initialises Chart.js charts, populates the recent-activity table, and
 * periodically polls /api/stats to refresh the dashboard without a full
 * page reload.
 *
 * Dependencies (loaded via CDN in the template):
 *   - Chart.js >= 4.x (window.Chart)
 */

(function () {
  'use strict';

  // ========================================================================
  // Constants
  // ========================================================================

  /** Polling interval for live metric refresh (ms). */
  var REFRESH_INTERVAL_MS = 60000; // 60 seconds

  /** Provider colour palette – matches CSS variables in the template. */
  var PROVIDER_COLORS = {
    claude:  '#d97706',
    openai:  '#10b981',
    gemini:  '#3b82f6',
    _other:  '#6c7ff8',
  };

  /** Status colours. */
  var STATUS_COLORS = {
    success:   '#34d399',
    error:     '#f87171',
    cancelled: '#8890b5',
  };

  /** Shared Chart.js defaults applied to every new chart. */
  var CHART_DEFAULTS = {
    plugins: {
      legend: {
        labels: {
          color: '#8890b5',
          font: { size: 12 },
          boxWidth: 12,
          padding: 14,
        },
      },
      tooltip: {
        backgroundColor: '#1a1d27',
        borderColor: '#2e3149',
        borderWidth: 1,
        titleColor: '#e2e4f0',
        bodyColor: '#8890b5',
        padding: 10,
      },
    },
  };

  // ========================================================================
  // State
  // ========================================================================

  /** Map of Chart.js instance keyed by canvas id. */
  var charts = {};

  /** Reference to the live-refresh timer. */
  var refreshTimer = null;

  // ========================================================================
  // Utility helpers
  // ========================================================================

  /**
   * Return the colour for a given provider name.
   * @param {string} name
   * @returns {string} CSS colour string.
   */
  function providerColor(name) {
    return PROVIDER_COLORS[name] || PROVIDER_COLORS._other;
  }

  /**
   * Format a number with thousand separators.
   * @param {number} n
   * @returns {string}
   */
  function fmtNum(n) {
    if (n == null) return '0';
    return Number(n).toLocaleString();
  }

  /**
   * Format a float as USD.
   * @param {number} n
   * @returns {string}
   */
  function fmtUsd(n) {
    return '$' + Number(n || 0).toFixed(4);
  }

  /**
   * Format a float as a percentage string.
   * @param {number} ratio  0.0 – 1.0
   * @returns {string}
   */
  function fmtPct(ratio) {
    return (Number(ratio || 0) * 100).toFixed(1) + '%';
  }

  /**
   * Format seconds into a human-readable duration string.
   * @param {number|null} secs
   * @returns {string}
   */
  function fmtDuration(secs) {
    if (secs == null) return '—';
    var s = Number(secs);
    if (s < 60) return s.toFixed(1) + 's';
    return (s / 60).toFixed(1) + 'm';
  }

  /**
   * Truncate a string to a maximum length with ellipsis.
   * @param {string|null} str
   * @param {number} max
   * @returns {string}
   */
  function truncate(str, max) {
    if (!str) return '—';
    return str.length > max ? str.slice(0, max) + '\u2026' : str;
  }

  /**
   * Format an ISO-8601 timestamp to a short locale string.
   * @param {string} iso
   * @returns {string}
   */
  function fmtDateTime(iso) {
    if (!iso) return '—';
    try {
      var d = new Date(iso);
      return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch (e) {
      return iso.slice(0, 16).replace('T', ' ');
    }
  }

  /**
   * Safely get a canvas element by id.
   * @param {string} id
   * @returns {HTMLCanvasElement|null}
   */
  function canvas(id) {
    return document.getElementById(id);
  }

  /**
   * Destroy an existing Chart.js instance for a canvas if one exists.
   * @param {string} id  Canvas element id.
   */
  function destroyChart(id) {
    if (charts[id]) {
      charts[id].destroy();
      delete charts[id];
    }
  }

  /**
   * Create a merged options object with shared defaults.
   * @param {object} opts
   * @returns {object}
   */
  function withDefaults(opts) {
    var merged = JSON.parse(JSON.stringify(CHART_DEFAULTS));
    // Deep-merge opts.plugins if present
    if (opts.plugins) {
      for (var k in opts.plugins) {
        merged.plugins[k] = Object.assign(merged.plugins[k] || {}, opts.plugins[k]);
      }
    }
    // Shallow-merge everything else
    for (var key in opts) {
      if (key !== 'plugins') merged[key] = opts[key];
    }
    return merged;
  }

  // ========================================================================
  // Chart initialisers
  // ========================================================================

  /**
   * Render the "Token Spend by Provider" doughnut chart.
   * @param {object} tokenSpend  stats.token_spend from the API.
   */
  function renderTokenSpendChart(tokenSpend) {
    var el = canvas('chartTokenSpend');
    if (!el) return;
    destroyChart('chartTokenSpend');

    var providers = tokenSpend.by_provider || [];
    if (providers.length === 0) return;

    charts['chartTokenSpend'] = new Chart(el, {
      type: 'doughnut',
      data: {
        labels: providers.map(function (p) { return p.display_name; }),
        datasets: [{
          data: providers.map(function (p) { return p.total_tokens; }),
          backgroundColor: providers.map(function (p) { return providerColor(p.provider); }),
          borderColor: '#0f1117',
          borderWidth: 3,
          hoverOffset: 6,
        }],
      },
      options: withDefaults({
        cutout: '65%',
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var val = ctx.raw || 0;
                var total = providers.reduce(function (s, p) { return s + p.total_tokens; }, 0);
                var pct = total > 0 ? ((val / total) * 100).toFixed(1) : '0';
                return ' ' + fmtNum(val) + ' tokens (' + pct + '%)';
              },
            },
          },
        },
      }),
    });
  }

  /**
   * Render the "Task Completion Rates" doughnut chart.
   * @param {object} completionRates  stats.completion_rates from the API.
   */
  function renderCompletionRatesChart(completionRates) {
    var el = canvas('chartCompletionRates');
    if (!el) return;
    destroyChart('chartCompletionRates');

    var success = completionRates.success_count || 0;
    var error = completionRates.error_count || 0;
    var cancelled = completionRates.cancelled_count || 0;

    if (success + error + cancelled === 0) return;

    charts['chartCompletionRates'] = new Chart(el, {
      type: 'doughnut',
      data: {
        labels: ['Success', 'Error', 'Cancelled'],
        datasets: [{
          data: [success, error, cancelled],
          backgroundColor: [
            STATUS_COLORS.success,
            STATUS_COLORS.error,
            STATUS_COLORS.cancelled,
          ],
          borderColor: '#0f1117',
          borderWidth: 3,
          hoverOffset: 6,
        }],
      },
      options: withDefaults({
        cutout: '65%',
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var total = success + error + cancelled;
                var pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : '0';
                return ' ' + fmtNum(ctx.raw) + ' tasks (' + pct + '%)';
              },
            },
          },
        },
      }),
    });
  }

  /**
   * Render the "Task Type Distribution" horizontal bar chart.
   * @param {object} taskDist  stats.task_distribution from the API.
   */
  function renderTaskTypesChart(taskDist) {
    var el = canvas('chartTaskTypes');
    if (!el) return;
    destroyChart('chartTaskTypes');

    var taskTypes = (taskDist.task_types || []).slice(0, 8);
    if (taskTypes.length === 0) return;

    charts['chartTaskTypes'] = new Chart(el, {
      type: 'bar',
      data: {
        labels: taskTypes.map(function (t) { return t.task_type || '(unset)'; }),
        datasets: [{
          label: 'Tasks',
          data: taskTypes.map(function (t) { return t.count; }),
          backgroundColor: PROVIDER_COLORS._other + 'cc',
          borderColor: PROVIDER_COLORS._other,
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: withDefaults({
        indexAxis: 'y',
        scales: {
          x: {
            grid: { color: '#2e3149' },
            ticks: { color: '#8890b5', font: { size: 11 } },
            beginAtZero: true,
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e2e4f0', font: { size: 11 } },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var share = taskTypes[ctx.dataIndex]
                  ? (taskTypes[ctx.dataIndex].share * 100).toFixed(1) + '%'
                  : '';
                return ' ' + fmtNum(ctx.raw) + ' tasks (' + share + ')';
              },
            },
          },
        },
      }),
    });
  }

  /**
   * Render the "Cost Share by Provider" bar chart.
   * @param {object} concentration  stats.concentration from the API.
   */
  function renderCostShareChart(concentration) {
    var el = canvas('chartCostShare');
    if (!el) return;
    destroyChart('chartCostShare');

    var providers = concentration.by_provider || [];
    if (providers.length === 0) return;

    charts['chartCostShare'] = new Chart(el, {
      type: 'bar',
      data: {
        labels: providers.map(function (p) { return p.display_name; }),
        datasets: [{
          label: 'Cost share (%)',
          data: providers.map(function (p) { return (p.cost_share * 100).toFixed(2); }),
          backgroundColor: providers.map(function (p) {
            return providerColor(p.provider) + 'cc';
          }),
          borderColor: providers.map(function (p) { return providerColor(p.provider); }),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: withDefaults({
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#e2e4f0', font: { size: 11 } },
          },
          y: {
            grid: { color: '#2e3149' },
            ticks: {
              color: '#8890b5',
              font: { size: 11 },
              callback: function (val) { return val + '%'; },
            },
            beginAtZero: true,
            max: 100,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ' ' + ctx.raw + '% of total cost';
              },
            },
          },
        },
      }),
    });
  }

  /**
   * Render the "Daily Token Usage Trend" multi-line chart.
   * @param {object} dailyTrend  stats.daily_trend from the API.
   */
  function renderDailyTrendChart(dailyTrend) {
    var el = canvas('chartDailyTrend');
    if (!el) return;
    destroyChart('chartDailyTrend');

    var days = dailyTrend.days || [];
    if (days.length === 0) return;

    var labels = days.map(function (d) { return d.date; });

    charts['chartDailyTrend'] = new Chart(el, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Total Tokens',
            data: days.map(function (d) { return d.total_tokens; }),
            borderColor: PROVIDER_COLORS._other,
            backgroundColor: PROVIDER_COLORS._other + '22',
            fill: true,
            tension: 0.35,
            pointRadius: days.length > 30 ? 0 : 4,
            pointHoverRadius: 6,
            borderWidth: 2,
            yAxisID: 'yTokens',
          },
          {
            label: 'Tasks',
            data: days.map(function (d) { return d.task_count; }),
            borderColor: '#a78bfa',
            backgroundColor: '#a78bfa22',
            fill: false,
            tension: 0.35,
            pointRadius: days.length > 30 ? 0 : 3,
            pointHoverRadius: 5,
            borderWidth: 2,
            yAxisID: 'yTasks',
            borderDash: [4, 4],
          },
        ],
      },
      options: withDefaults({
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            grid: { color: '#2e3149' },
            ticks: {
              color: '#8890b5',
              font: { size: 11 },
              maxTicksLimit: 12,
              maxRotation: 0,
            },
          },
          yTokens: {
            position: 'left',
            grid: { color: '#2e3149' },
            ticks: {
              color: '#8890b5',
              font: { size: 11 },
              callback: function (val) {
                if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
                if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                return val;
              },
            },
            beginAtZero: true,
            title: {
              display: true,
              text: 'Tokens',
              color: '#8890b5',
              font: { size: 11 },
            },
          },
          yTasks: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#8890b5', font: { size: 11 } },
            beginAtZero: true,
            title: {
              display: true,
              text: 'Tasks',
              color: '#8890b5',
              font: { size: 11 },
            },
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.datasetIndex === 0) {
                  return ' Tokens: ' + fmtNum(ctx.raw);
                }
                return ' Tasks: ' + ctx.raw;
              },
            },
          },
        },
      }),
    });
  }

  // ========================================================================
  // Recent activity table
  // ========================================================================

  /**
   * Render the recent activity table by fetching /api/recent.
   */
  function renderRecentActivity() {
    var tbody = document.getElementById('recent-activity-body');
    if (!tbody) return;

    var filters = window.__AGENT_DASH_FILTERS__ || {};
    var params = new URLSearchParams();
    params.set('limit', '20');
    if (filters.since)    params.set('since', filters.since);
    if (filters.until)    params.set('until', filters.until);
    if (filters.provider) params.set('provider', filters.provider);

    fetch('/api/recent?' + params.toString())
      .then(function (res) { return res.json(); })
      .then(function (records) {
        if (!Array.isArray(records) || records.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:24px;">'
            + 'No recent activity to display.</td></tr>';
          return;
        }
        tbody.innerHTML = records.map(function (rec) {
          var statusClass = 'status-' + (rec.status || 'success');
          var provColor = providerColor(rec.provider_name || '');
          return (
            '<tr>' +
            '<td style="font-size:12px;white-space:nowrap;">' + fmtDateTime(rec.logged_at) + '</td>' +
            '<td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
              + 'background:' + provColor + ';margin-right:5px;"></span>'
              + escHtml(rec.provider_display_name || rec.provider_name || '—') + '</td>' +
            '<td style="font-size:12px;">' + escHtml(truncate(rec.model, 28)) + '</td>' +
            '<td style="font-size:12px;">' + escHtml(truncate(rec.task_type, 22)) + '</td>' +
            '<td style="text-align:right;">' + fmtNum(rec.total_tokens) + '</td>' +
            '<td style="text-align:right;">' + fmtUsd(rec.cost_usd) + '</td>' +
            '<td style="text-align:right;">' + fmtDuration(rec.duration_seconds) + '</td>' +
            '<td><span class="status-badge ' + statusClass + '">' + escHtml(rec.status || '—') + '</span></td>' +
            '</tr>'
          );
        }).join('');
      })
      .catch(function (err) {
        console.warn('agent_dash: failed to load recent activity:', err);
        tbody.innerHTML =
          '<tr><td colspan="8" style="text-align:center;color:var(--red);padding:16px;">'
          + 'Failed to load recent activity.</td></tr>';
      });
  }

  /**
   * Escape HTML special characters.
   * @param {string} str
   * @returns {string}
   */
  function escHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ========================================================================
  // Summary card live updates
  // ========================================================================

  /**
   * Update summary card text values from a fresh stats object.
   * @param {object} stats  Full stats object from /api/stats.
   */
  function updateSummaryCards(stats) {
    function setText(id, val) {
      var el = document.getElementById(id);
      if (el) el.textContent = val;
    }

    var ts = stats.token_spend || {};
    var cr = stats.completion_rates || {};
    var sv = stats.time_saved || {};
    var dt = stats.daily_trend || {};

    setText('card-total-tokens', fmtNum(ts.total_tokens));
    setText('card-total-cost',   fmtUsd(ts.total_cost_usd));
    setText('card-total-tasks',  fmtNum(cr.total_tasks));
    setText('card-success-rate', ((cr.success_rate || 0) * 100).toFixed(1) + '%');
    setText('card-time-saved',   ((sv.time_saved_hours || 0)).toFixed(1) + 'h');
    setText('card-time-saved-pct', ((sv.time_saved_percent || 0)).toFixed(1) + '%');
    setText('card-active-days',  dt.total_days || 0);
  }

  // ========================================================================
  // Live refresh
  // ========================================================================

  /**
   * Fetch fresh stats from /api/stats and update charts + cards.
   */
  function refreshStats() {
    var filters = window.__AGENT_DASH_FILTERS__ || {};
    var params = new URLSearchParams();
    if (filters.since)    params.set('since', filters.since);
    if (filters.until)    params.set('until', filters.until);
    if (filters.provider) params.set('provider', filters.provider);

    fetch('/api/stats?' + params.toString())
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (stats) {
        window.__AGENT_DASH_STATS__ = stats;
        initCharts(stats);
        updateSummaryCards(stats);
        renderRecentActivity();
      })
      .catch(function (err) {
        console.warn('agent_dash: live refresh failed:', err);
      });
  }

  /**
   * Schedule periodic live refreshes.
   */
  function scheduleRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(refreshStats, REFRESH_INTERVAL_MS);
  }

  // ========================================================================
  // Chart initialisation
  // ========================================================================

  /**
   * Initialise (or reinitialise) all charts from a stats object.
   * @param {object} stats
   */
  function initCharts(stats) {
    if (!stats) return;
    renderTokenSpendChart(stats.token_spend || {});
    renderCompletionRatesChart(stats.completion_rates || {});
    renderTaskTypesChart(stats.task_distribution || {});
    renderCostShareChart(stats.concentration || {});
    renderDailyTrendChart(stats.daily_trend || {});
  }

  // ========================================================================
  // Bootstrap
  // ========================================================================

  /**
   * Main entry point called on DOMContentLoaded.
   */
  function main() {
    // Apply global Chart.js defaults
    if (window.Chart) {
      Chart.defaults.color = '#8890b5';
      Chart.defaults.borderColor = '#2e3149';
      Chart.defaults.font.family = "'Segoe UI', system-ui, -apple-system, sans-serif";
    } else {
      console.warn('agent_dash: Chart.js not found; charts will not render.');
      return;
    }

    var stats = window.__AGENT_DASH_STATS__;
    if (stats) {
      initCharts(stats);
    }

    renderRecentActivity();
    scheduleRefresh();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }

})();
