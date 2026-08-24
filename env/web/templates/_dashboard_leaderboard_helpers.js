  function batchSummaryCell(row, field, meanHtml, options={}) {
    if (!row?.is_batch_average) return meanHtml;
    const standardDeviation = row?.[`std_${field}`];
    const sampleSize = Number(row?.[`n_${field}`] ?? row?.runs ?? 0);
    const digits = options.digits ?? 2;
    const formatSpread = (value, scale, suffix, formatter) => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
      }
      if (formatter) return formatter(Number(value) * scale);
      return `${fmtExpFixed(Number(value) * scale, digits)}${suffix || ""}`;
    };
    const stdLabel = formatSpread(
      standardDeviation,
      options.stdScale ?? 1,
      options.stdSuffix,
      options.stdFormatter
    );
    return `<span class="batch-stat-cell">${meanHtml} ± ${stdLabel} ` +
      `<span class="batch-stat-n">(n=${esc(sampleSize)})</span></span>`;
  }
  function filteredLeaderboardRows(payload=leaderboardViz.payload) {
    const filters = getVizFilters();
    return (payload?.leaderboard || []).filter(row => {
      if (filters.framework && row.framework !== filters.framework) return false;
      if (filters.model && row.model !== filters.model) return false;
      return true;
    });
  }
  function syncVizShowAll(rows=filteredLeaderboardRows()) {
    const toggle = $("viz-show-all");
    if (!toggle) return;
    const shown = rows.filter(row => !leaderboardViz.hiddenLeaderKeys.has(leaderRowKey(row))).length;
    toggle.disabled = rows.length === 0;
    toggle.checked = rows.length > 0 && shown === rows.length;
    toggle.indeterminate = shown > 0 && shown < rows.length;
  }
  function leaderboardRankingSeriesRow(row, metric) {
    if (!metric.seriesKey) return null;
    const key = leaderRowKey(row);
    return (leaderboardViz.payload?.charts?.[metric.seriesKey] || [])
      .find(seriesRow => leaderRowKey(seriesRow) === key) || null;
  }
  function leaderboardWindowSeriesValue(seriesRow, mode) {
    if (!seriesRow) return null;
    const filters = getVizFilters();
    let baseline = null;
    let last = null;
    const values = [];
    let weightedTotal = 0;
    let totalWeight = 0;
    (seriesRow.data || []).slice().sort((a, b) => Number(a?.[0]) - Number(b?.[0])).forEach(point => {
      const day = pointDay(point?.[0], seriesRow.step_hours);
      if (point?.[1] === null || point?.[1] === undefined) return;
      const value = Number(point?.[1]);
      if (!Number.isFinite(day) || !Number.isFinite(value)) return;
      if (day < filters.dayFrom) baseline = value;
      if (day >= filters.dayFrom && (filters.dayTo === null || day < filters.dayTo)) {
        last = value;
        values.push(value);
        const weight = Number(point?.[2] ?? 1);
        if (Number.isFinite(weight) && weight > 0) {
          weightedTotal += value * weight;
          totalWeight += weight;
        }
      }
    });
    if (last === null) return null;
    if (mode === "mean") return meanExperimentValues(values);
    if (mode === "weightedMean") {
      return totalWeight > 0 ? weightedTotal / totalWeight : null;
    }
    return mode === "delta" ? last - (baseline ?? 0) : last;
  }
  function leaderboardRatioRankingValue(row, numeratorKey, denominatorKey) {
    const numerator = leaderboardWindowSeriesValue(
      leaderboardRankingSeriesRow(row, {seriesKey: numeratorKey}),
      "delta"
    );
    const denominator = leaderboardWindowSeriesValue(
      leaderboardRankingSeriesRow(row, {seriesKey: denominatorKey}),
      "delta"
    );
    if (numerator === null || denominator === null || Math.abs(denominator) < 1e-12) {
      return null;
    }
    return numerator / denominator;
  }
  function leaderboardActivityRankingValue(row, field) {
    const key = leaderRowKey(row);
    const activityRow = (leaderboardViz.payload?.charts?.tool_calls?.runs || [])
      .find(item => leaderRowKey(item) === key);
    if (!activityRow) return null;
    const filters = getVizFilters();
    let availableWindows = 0;
    let effectiveWindows = 0;
    let totalToolCalls = 0;
    let coveredDays = 0;
    (activityRow.activity_by_day || []).forEach(point => {
      const day = pointDay(point?.t, activityRow.step_hours);
      if (
        !Number.isFinite(day)
        || day < filters.dayFrom
        || (filters.dayTo !== null && day >= filters.dayTo)
      ) return;
      coveredDays += 1;
      availableWindows += Number(point?.available_windows || 0);
      effectiveWindows += Number(point?.effective_windows || 0);
      totalToolCalls += Number(point?.total_tool_calls || 0);
    });
    if (field === "total_tool_calls") {
      return coveredDays > 0 ? totalToolCalls : null;
    }
    return availableWindows > 0 ? effectiveWindows / availableWindows : null;
  }
  function leaderboardRawRankingValue(row, metric) {
    const raw = row?.[metric.field];
    if (raw === null || raw === undefined || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }
  function leaderboardUsesFullWindow() {
    const filters = getVizFilters();
    return filters.dayFrom === 0 && filters.dayTo === null;
  }
  function leaderboardRankingValue(row, metric) {
    const rawValue = leaderboardRawRankingValue(row, metric);
    // The summary table is the source of truth for the complete run. Besides
    // making the bar and table identical, this preserves terminal values when
    // a source run ended earlier than the other runs in a batch average.
    if (leaderboardUsesFullWindow() && rawValue !== null) {
      return rawValue;
    }
    if (row?.is_batch_average && (row.source_run_ids || []).length) {
      const rowsByRun = new Map(
        (leaderboardViz.payload?.leaderboard || [])
          .filter(item => !item?.is_batch_average)
          .map(item => [String(item.run_id), item])
      );
      const sourceRows = row.source_run_ids
        .map(runId => rowsByRun.get(String(runId)))
        .filter(Boolean);
      const sourceValues = sourceRows.map(
        sourceRow => leaderboardRankingValue(sourceRow, metric)
      );
      // Never turn a partially available window into a misleading average of
      // only the surviving source curves.
      if (
        !sourceRows.length
        || sourceRows.length !== row.source_run_ids.length
        || sourceValues.some(value => value === null)
      ) return null;
      return meanExperimentValues(sourceValues);
    }
    if (metric.value) {
      const derived = metric.value(row);
      if (derived !== null && Number.isFinite(Number(derived))) {
        return Number(derived);
      }
    }
    if (metric.seriesKey) {
      const seriesValue = leaderboardWindowSeriesValue(
        leaderboardRankingSeriesRow(row, metric),
        metric.windowMode
      );
      if (seriesValue !== null) return seriesValue;
    }
    if (
      (metric.value || metric.seriesKey)
      && (leaderboardViz.payload?.charts?.runs || []).length
    ) return null;
    return rawValue;
  }
