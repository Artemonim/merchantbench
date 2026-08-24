  function formatExperimentDays(value) {
    if (value === null || value === undefined || value === "") return "— days";
    const days = Number(value);
    if (!Number.isFinite(days)) return "— days";
    return `${days.toLocaleString(undefined, {maximumFractionDigits: 2})} days`;
  }
  function formatExperimentNetAssets(value) {
    if (value === null || value === undefined || value === "") return "Net —";
    const amount = Number(value);
    if (!Number.isFinite(amount)) return "Net —";
    return `Net ${amount.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  }
  function formatExperimentRank(value) {
    const rank = Number(value);
    return Number.isFinite(rank) && rank > 0 ? `#${rank}` : "#—";
  }
  function experimentRunOptionLabel(row) {
    const date = String(row?.started_at || "—").slice(0, 10);
    return [
      row?.run_id || "—",
      row?.framework || "None",
      row?.model || "—",
      row?.status || "unknown",
      date || "—",
      formatExperimentDays(row?.horizon_days),
      formatExperimentNetAssets(row?.final_net_assets),
      formatExperimentRank(row?.rank)
    ].join(" · ");
  }
  function experimentRunById(runId) {
    const selected = String(runId || "");
    if (!selected) return null;
    return experimentGroups.runOptions.find(
      option => String(option.run_id) === selected
    ) || null;
  }
  function experimentModelBatchProgress(model, group=selectedExperimentGroup()) {
    const batches = group?.batches || [];
    const modelSlotIds = new Set(
      experimentSlots(group)
        .filter(slot => slot.kind === "model" && slot.model === model)
        .map(slot => slot.id)
    );
    const linked = batches.filter(batch => (
      [...modelSlotIds].some(slotId => String(batch.bindings?.[slotId] || ""))
    )).length;
    return {linked, total: batches.length};
  }
  function experimentRankedBatchSlots(group, batch) {
    const byFramework = new Map();
    const allRows = [];
    experimentSlots(group).forEach((slot, originalIndex) => {
      const runId = String(batch?.bindings?.[slot.id] || "");
      const run = experimentRunById(runId);
      const rawNetAssets = run?.final_net_assets;
      const netAssets = Number(rawNetAssets);
      const comparable = Boolean(runId)
        && rawNetAssets !== null
        && rawNetAssets !== undefined
        && rawNetAssets !== ""
        && Number.isFinite(netAssets);
      const row = {
        ...slot,
        originalIndex,
        netAssets,
        comparable,
        batchRank: null
      };
      allRows.push(row);
      if (!byFramework.has(slot.framework_key)) {
        byFramework.set(slot.framework_key, []);
      }
      byFramework.get(slot.framework_key).push(row);
    });
    allRows
      .filter(row => row.comparable)
      .sort((a, b) => (
        b.netAssets - a.netAssets || a.originalIndex - b.originalIndex
      ))
      .forEach((row, index) => {
        row.batchRank = index + 1;
      });
    return [...byFramework.values()].flatMap(rows => {
      rows.sort((a, b) => {
        if (a.comparable !== b.comparable) return a.comparable ? -1 : 1;
        if (a.comparable && b.netAssets !== a.netAssets) {
          return b.netAssets - a.netAssets;
        }
        return a.originalIndex - b.originalIndex;
      });
      return rows;
    });
  }
  function experimentRunUsageMap(group=selectedExperimentGroup()) {
    const usage = new Map();
    (group?.batches || []).forEach(batch => {
      Object.entries(batch.bindings || {}).forEach(([slotId, runId]) => {
        const selected = String(runId || "");
        if (!selected) return;
        if (!usage.has(selected)) usage.set(selected, []);
        usage.get(selected).push({
          batchId: String(batch.id),
          batchName: String(batch.name || "Batch"),
          slotId: String(slotId)
        });
      });
    });
    return usage;
  }
  function experimentRunPickerValue(runId) {
    const selected = String(runId || "");
    if (!selected) {
      return `<span class="experiment-run-picker-placeholder">Choose a run…</span>`;
    }
    const row = experimentRunById(selected);
    if (!row) {
      return `<span class="experiment-run-picker-value experiment-run-preview-unavailable">` +
        `<code>${esc(selected)}</code><span>run unavailable</span>` +
      `</span>`;
    }
    const status = String(row.status || "unknown");
    return `<span class="experiment-run-picker-value">` +
        frameworkIdentity(row.framework, row.framework) +
        modelIdentity(row.model, experimentModelLabel(row.model)) +
        `<span class="badge badge-sm state-${esc(slugToken(status))}">${esc(status)}</span>` +
        `<code title="${esc(selected)}">${esc(selected)}</code>` +
        `<span class="experiment-run-metric-tag">${esc(formatExperimentNetAssets(row.final_net_assets))}</span>` +
      `</span>`;
  }
  function experimentRunPickerOptions(picker, query="") {
    if (!picker) return "";
    const group = selectedExperimentGroup();
    const batchId = String(picker.dataset.bindingBatch || "");
    const slotId = String(picker.dataset.bindingSlot || "");
    const batch = (group?.batches || []).find(
      row => String(row.id) === batchId
    );
    const currentRunId = String(batch?.bindings?.[slotId] || "");
    const usage = experimentRunUsageMap(group);
    const needle = String(query || "").trim().toLowerCase();
    const rows = experimentGroups.runOptions.filter(row => (
      !needle || experimentRunOptionLabel(row).toLowerCase().includes(needle)
    ));
    const clearOption = currentRunId
      ? `<button type="button" class="experiment-run-option experiment-run-clear" data-experiment-run-option="">Clear this slot</button>`
      : "";
    if (!rows.length) {
      return clearOption +
        `<div class="experiment-run-menu-empty">No matching runs</div>`;
    }
    return clearOption + rows.map(row => {
      const runId = String(row.run_id || "");
      const entries = usage.get(runId) || [];
      const isCurrent = runId === currentRunId;
      const date = String(row.started_at || "—").slice(0, 10);
      const status = String(row.status || "unknown");
      const usageTitle = entries.map(entry => entry.batchName).join(", ");
      const usageTag = entries.length
        ? `<span class="experiment-run-metric-tag ${isCurrent ? "experiment-run-current-tag" : "experiment-run-used-tag"}" title="${esc(usageTitle)}">${isCurrent ? "Current" : `Selected ×${entries.length}`}</span>`
        : "";
      return `<button type="button" role="option" aria-selected="${isCurrent ? "true" : "false"}" ` +
        `class="experiment-run-option${entries.length ? " is-used" : ""}${isCurrent ? " is-current" : ""}" ` +
        `data-experiment-run-option="${esc(runId)}">` +
          `<span class="experiment-run-option-main">` +
            frameworkIdentity(row.framework, row.framework) +
            modelIdentity(row.model, experimentModelLabel(row.model)) +
            `<span class="badge badge-sm state-${esc(slugToken(status))}">${esc(status)}</span>` +
            `<span class="experiment-run-metric-tag">${esc(formatExperimentNetAssets(row.final_net_assets))}</span>` +
            `<span class="experiment-run-metric-tag experiment-run-rank-tag">${esc(formatExperimentRank(row.rank))}</span>` +
            usageTag +
          `</span>` +
          `<span class="experiment-run-option-meta">` +
            `<code title="${esc(runId)}">${esc(runId)}</code>` +
            `<span>${esc(date || "—")}</span>` +
            `<span>${esc(formatExperimentDays(row.horizon_days))}</span>` +
          `</span>` +
        `</button>`;
    }).join("");
  }
  function renderExperimentRunPickerOptions(picker, query="") {
    const options = picker?.querySelector(".experiment-run-options");
    if (options) options.innerHTML = experimentRunPickerOptions(picker, query);
  }
  function closeExperimentRunPickers(except=null) {
    document.querySelectorAll(".experiment-run-picker[open]").forEach(picker => {
      if (picker !== except) picker.open = false;
    });
  }
  function syncExperimentRunOptionMetrics() {
    const rowsByRun = new Map(
      (leaderboardViz.basePayload?.leaderboard || [])
        .filter(row => !row.is_batch_average && row.run_id)
        .map(row => [String(row.run_id), row])
    );
    experimentGroups.runOptions.forEach(row => {
      const leaderboardRow = rowsByRun.get(String(row.run_id || ""));
      row.final_net_assets = leaderboardRow?.avg_final_net_assets ?? null;
      row.rank = leaderboardRow?.rank ?? null;
    });
  }
  function renderExperimentGroupControls() {
    const select = $("experiment-group-select");
    if (!select) return;
    const groups = experimentGroups.groups || [];
    if (
      experimentGroups.loaded
      && experimentGroups.selectedGroupId
      && !groups.some(group => String(group.id) === experimentGroups.selectedGroupId)
    ) {
      experimentGroups.selectedGroupId = "";
      experimentGroups.activeBatchIds.clear();
    }
    select.innerHTML = `<option value="">No group selected</option>` + groups.map(group => (
      `<option value="${esc(group.id)}">${esc(group.name)}</option>`
    )).join("");
    select.value = experimentGroups.selectedGroupId;

    const group = selectedExperimentGroup();
    const editorDisabled = !group || experimentGroups.saving;
    const template = group?.template || {};
    const frameworkCount = (template.frameworks || []).length;
    const modelCount = (template.models || []).length;
    const controlCount = Number(Boolean(template.include_human))
      + Number(Boolean(template.include_rule_based));
    const batchCount = (group?.batches || []).length;
    const summaryTags = $("experiment-manager-summary-tags");
    if (summaryTags) {
      summaryTags.innerHTML = group ? [
        `<span class="experiment-tag">${frameworkCount} framework${frameworkCount === 1 ? "" : "s"}</span>`,
        `<span class="experiment-tag">${modelCount} model${modelCount === 1 ? "" : "s"}</span>`,
        `<span class="experiment-tag">${controlCount} control${controlCount === 1 ? "" : "s"}</span>`,
        `<span class="experiment-tag">${batchCount} batch${batchCount === 1 ? "" : "es"}</span>`
      ].join("") : `<span class="experiment-tag experiment-tag-muted">No group selected</span>`;
    }
    const frameworkCountTag = $("experiment-framework-count");
    if (frameworkCountTag) {
      const selectedCount = frameworkCount + controlCount;
      frameworkCountTag.textContent = `${selectedCount} selected`;
      frameworkCountTag.className = `experiment-tag${selectedCount ? "" : " experiment-tag-muted"}`;
    }
    const modelCountTag = $("experiment-model-count");
    if (modelCountTag) {
      modelCountTag.textContent = `${modelCount} selected`;
      modelCountTag.className = `experiment-tag${modelCount ? "" : " experiment-tag-muted"}`;
    }
    const newButton = $("experiment-group-new");
    if (newButton) {
      newButton.disabled = !experimentGroups.loaded || experimentGroups.saving;
    }
    const exportButton = $("experiment-group-export-md");
    if (exportButton) {
      exportButton.disabled = (
        !group
        || experimentGroups.saving
        || !leaderboardViz.fullPayloadLoaded
      );
      exportButton.title = (
        group && !leaderboardViz.fullPayloadLoaded
          ? "Waiting for complete leaderboard metrics"
          : ""
      );
    }
    if (experimentGroups.loaded) {
      const validBatchIds = new Set((group?.batches || []).map(batch => String(batch.id)));
      experimentGroups.activeBatchIds = new Set(
        [...experimentGroups.activeBatchIds].filter(id => validBatchIds.has(id))
      );
    }
    const toggles = $("experiment-batch-toggles");
    if (toggles) {
      toggles.innerHTML = (group?.batches || []).map(batch => {
        const active = experimentGroups.activeBatchIds.has(String(batch.id));
        return `<button type="button" class="experiment-batch-toggle${active ? " is-active" : ""}" ` +
          `data-experiment-batch="${esc(batch.id)}" aria-pressed="${active ? "true" : "false"}">${esc(batch.name)}</button>`;
      }).join("");
    }

    const nameInput = $("experiment-group-name");
    if (nameInput) {
      nameInput.value = group?.name || "";
      nameInput.disabled = editorDisabled;
    }
    const deleteButton = $("experiment-group-delete");
    if (deleteButton) deleteButton.disabled = editorDisabled;
    const saveButton = $("experiment-group-save");
    if (saveButton) saveButton.disabled = editorDisabled;
    const addBatchButton = $("experiment-batch-add");
    if (addBatchButton) addBatchButton.disabled = editorDisabled;

    const frameworkBox = $("experiment-framework-options");
    if (frameworkBox) {
      const frameworkChecks = experimentGroups.frameworkPresets.map(row => {
        const checked = (group?.template?.frameworks || []).includes(row.key);
        return `<label class="experiment-option-tag${checked ? " is-selected" : ""}"><input type="checkbox" data-template-framework="${esc(row.key)}" ${checked ? "checked" : ""} ${editorDisabled ? "disabled" : ""}>${frameworkIdentity(row.key, row.label)}</label>`;
      });
      frameworkChecks.push(
        `<label class="experiment-option-tag${group?.template?.include_human ? " is-selected" : ""}"><input type="checkbox" data-template-control="include_human" ${group?.template?.include_human ? "checked" : ""} ${editorDisabled ? "disabled" : ""}>${frameworkIdentity("human", "Human")}</label>`,
        `<label class="experiment-option-tag${group?.template?.include_rule_based ? " is-selected" : ""}"><input type="checkbox" data-template-control="include_rule_based" ${group?.template?.include_rule_based ? "checked" : ""} ${editorDisabled ? "disabled" : ""}>${frameworkIdentity("rule_based", "Rule-based")}</label>`
      );
      frameworkBox.innerHTML = frameworkChecks.join("");
    }
    const modelBox = $("experiment-model-options");
    if (modelBox) {
      modelBox.innerHTML = experimentGroups.modelPresets.map(row => {
        const checked = (group?.template?.models || []).includes(row.model);
        const progress = experimentModelBatchProgress(row.model, group);
        return `<label class="experiment-option-tag${checked ? " is-selected" : ""}" title="${esc(row.model)} · ${progress.linked} of ${progress.total} batches linked"><input type="checkbox" data-template-model="${esc(row.model)}" ${checked ? "checked" : ""} ${editorDisabled ? "disabled" : ""}>${modelIdentity(row.model, row.label)}<span class="experiment-option-count">${progress.linked}/${progress.total}</span></label>`;
      }).join("");
    }

    const editors = $("experiment-batch-editors");
    if (editors) {
      const slots = experimentSlots(group);
      editors.innerHTML = (group?.batches || []).map(batch => {
        const boundCount = slots.filter(
          slot => String(batch.bindings?.[slot.id] || "")
        ).length;
        const bindingState = !boundCount
          ? "state-stopped"
          : boundCount === slots.length
            ? "alive-yes"
            : "state-paused";
        return (
          `<div class="experiment-batch-editor" data-batch-editor="${esc(batch.id)}">` +
          `<div class="experiment-batch-editor-head">` +
            `<input type="text" maxlength="120" value="${esc(batch.name)}" data-batch-name="${esc(batch.id)}" ${editorDisabled ? "disabled" : ""}>` +
            `<span class="experiment-tag">${slots.length} slots</span>` +
            `<span class="experiment-tag ${bindingState}">${boundCount} linked</span>` +
            `<button type="button" class="danger" data-batch-remove="${esc(batch.id)}" ${editorDisabled ? "disabled" : ""}>Remove</button>` +
          `</div>` +
          `<div class="experiment-bindings">` +
            (slots.length ? experimentRankedBatchSlots(group, batch).map(slot => {
              const runId = batch.bindings?.[slot.id] || "";
              const slotTags = slot.kind === "model"
                ? frameworkIdentity(slot.framework_key, slot.framework) +
                  modelIdentity(slot.model, experimentModelLabel(slot.model))
                : frameworkIdentity(slot.framework_key, slot.label);
              const rankLabel = slot.batchRank ? `#${slot.batchRank}` : "#—";
              return `<div class="experiment-slot-label" title="${esc(slot.label)} · rank across this batch">` +
                  `<span class="experiment-slot-rank${slot.batchRank ? "" : " is-empty"}">${rankLabel}</span>` +
                  slotTags +
                `</div>` +
                `<div class="experiment-run-binding">` +
                  `<details class="experiment-run-picker${runId ? " is-bound" : ""}" ` +
                    `data-binding-batch="${esc(batch.id)}" data-binding-slot="${esc(slot.id)}">` +
                    `<summary class="experiment-run-picker-trigger" aria-label="Choose run for ${esc(slot.label)}" ` +
                      `${editorDisabled ? `data-picker-disabled="true" aria-disabled="true"` : ""}>` +
                      experimentRunPickerValue(runId) +
                    `</summary>` +
                    `<div class="experiment-run-menu">` +
                      `<input class="experiment-run-search" type="search" placeholder="Search run id, framework, model, status, net assets, or rank…" autocomplete="off">` +
                      `<div class="experiment-run-options" role="listbox"></div>` +
                    `</div>` +
                  `</details>` +
                `</div>`;
            }).join("") : `<div class="meta">Select at least one framework/model or control.</div>`) +
          `</div>` +
        `</div>`
        );
      }).join("");
    }
    updateExperimentAverageNote();
  }
  function applyExperimentBatchVisibility() {
    const baseRows = leaderboardViz.basePayload?.leaderboard || [];
    const batches = selectedExperimentBatches();
    experimentGroups.selectedRunIds = experimentSelectedRunIds();
    if (!batches.length) {
      if (experimentGroups.manualHiddenLeaderKeys !== null) {
        leaderboardViz.hiddenLeaderKeys = new Set(
          experimentGroups.manualHiddenLeaderKeys
        );
        experimentGroups.manualHiddenLeaderKeys = null;
        saveHiddenLeaderKeys();
      }
      return;
    }
    if (experimentGroups.manualHiddenLeaderKeys === null) {
      experimentGroups.manualHiddenLeaderKeys = new Set(
        leaderboardViz.hiddenLeaderKeys
      );
    }
    leaderboardViz.hiddenLeaderKeys = new Set(
      baseRows.map(row => leaderRowKey(row))
    );
    experimentGroups.selectedRunIds.forEach(
      runId => leaderboardViz.hiddenLeaderKeys.delete(runId)
    );
  }
  function finiteExperimentValues(values) {
    return values
      .filter(value => value !== null && value !== undefined && value !== "")
      .map(Number)
      .filter(Number.isFinite);
  }
  function meanExperimentValues(values) {
    const finite = finiteExperimentValues(values);
    if (!finite.length) return null;
    return finite.reduce((sum, value) => sum + value, 0) / finite.length;
  }
  function sampleStdExperimentValues(values) {
    const finite = finiteExperimentValues(values);
    if (finite.length < 2) return null;
    const mean = meanExperimentValues(finite);
    return Math.sqrt(
      finite.reduce((sum, value) => sum + ((value - mean) ** 2), 0)
      / (finite.length - 1)
    );
  }
  function averageExperimentPointArrays(sourceRows, key="data") {
    const byT = new Map();
    sourceRows.forEach(row => {
      (row?.[key] || []).forEach(point => {
        const t = Number(point?.[0]);
        const value = Number(point?.[1]);
        if (!Number.isFinite(t) || !Number.isFinite(value)) return;
        if (!byT.has(t)) byT.set(t, []);
        byT.get(t).push(value);
      });
    });
    return [...byT.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([t, values]) => [t, meanExperimentValues(values)]);
  }
  function averageExperimentCountMaps(sourceRows, key="counts") {
    const names = new Set();
    sourceRows.forEach(row => Object.keys(row?.[key] || {}).forEach(name => names.add(name)));
    return Object.fromEntries([...names].map(name => [
      name,
      meanExperimentValues(
        sourceRows.map(row => Number(row?.[key]?.[name] || 0))
      ) || 0
    ]));
  }
  function averageExperimentToolRuns(toolCalls, averageRows) {
    const rawRuns = toolCalls?.runs || [];
    const rowsByRun = new Map(rawRuns.map(row => [String(row.run_id), row]));
    const averages = averageRows.map(avgRow => {
      const sourceRows = avgRow.source_run_ids.map(runId => rowsByRun.get(String(runId))).filter(Boolean);
      if (!sourceRows.length) return null;
      const stepsByRun = sourceRows.map(row => new Map(
        (row.by_step || []).map(step => [Number(step.t), step])
      ));
      const steps = new Set(
        stepsByRun.flatMap(byStep => [...byStep.keys()])
      );
      const counts = averageExperimentCountMaps(sourceRows);
      return {
        ...experimentChartMeta(avgRow),
        counts,
        total: meanExperimentValues(sourceRows.map(row => row.total)),
        by_step: [...steps].sort((a, b) => a - b).map(t => ({
          t,
          counts: averageExperimentCountMaps(
            stepsByRun.map(byStep => byStep.get(t) || {counts: {}})
          )
        }))
      };
    }).filter(Boolean);
    return {...toolCalls, runs: [...rawRuns, ...averages]};
  }
  const EXPERIMENT_EXPORT_METRICS = [
    {field: "avg_final_net_assets", label: "final-net-assets"},
    {field: "avg_cum_gmv", label: "gmv"},
    {field: "avg_net_profit", label: "net-profit"},
    {field: "avg_net_profit_margin", label: "net-profit-margin-%", scale: 100},
    {field: "avg_cum_fine", label: "total-fines"},
    {field: "avg_orders", label: "orders"},
    {field: "avg_shop_rating_score", label: "average-store-rating"},
    {field: "avg_order_anomaly_rate", label: "order-anomaly-rate-%", scale: 100},
    {field: "avg_active_listings", label: "average-active-listings"},
    {field: "avg_effective_window_rate", label: "effective-window-rate-%", scale: 100},
    {field: "avg_total_tool_calls", label: "total-tool-calls"},
    {field: "avg_tokens", label: "tokens"},
    {field: "avg_usd", label: "usd"},
    {field: "avg_t", label: "t"},
    {field: "horizon", label: "horizon-steps"},
    {field: "elapsed_ms", label: "total-time-ms"}
  ];
  function markdownExperimentText(value) {
    return String(value ?? "—")
      .replace(/\\/g, "\\\\")
      .replace(/\|/g, "\\|")
      .replace(/\r?\n/g, "<br>");
  }
  function markdownExperimentNumber(value, scale=1) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value) * Number(scale || 1);
    if (!Number.isFinite(number)) return "—";
    if (number === 0) return "0";
    const absolute = Math.abs(number);
    if (absolute >= 1e12 || absolute < 1e-6) {
      return number.toExponential(6).replace(/\.?0+e/, "e");
    }
    return number.toFixed(6).replace(/\.?0+$/, "");
  }
  function markdownExperimentSummary(values, scale=1) {
    const finite = finiteExperimentValues(values);
    const mean = meanExperimentValues(finite);
    const standardDeviation = sampleStdExperimentValues(finite);
    return `${markdownExperimentNumber(mean, scale)} ± ` +
      `${markdownExperimentNumber(standardDeviation, scale)} (n=${finite.length})`;
  }
  function experimentGroupExportRows(group, basePayload=leaderboardViz.basePayload) {
    const rowsByRun = new Map(
      (basePayload?.leaderboard || [])
        .filter(row => !row.is_batch_average && row.run_id)
        .map(row => [String(row.run_id), row])
    );
    const optionsByRun = new Map(
      (experimentGroups.runOptions || []).map(row => [String(row.run_id), row])
    );
    const detailRows = [];
    experimentSlots(group).forEach(slot => {
      (group?.batches || []).forEach(batch => {
        const runId = String(batch.bindings?.[slot.id] || "");
        const result = rowsByRun.get(runId) || null;
        const option = optionsByRun.get(runId) || null;
        detailRows.push({
          batch_id: String(batch.id),
          batch: String(batch.name || batch.id),
          slot_id: String(slot.id),
          framework: String(slot.framework || "—"),
          model: String(slot.model || "—"),
          run_id: runId,
          run_framework: String(result?.framework || option?.framework || "—"),
          run_model: String(result?.model || option?.model || "—"),
          status: String(option?.status || result?.terminal_status || "—"),
          result
        });
      });
    });
    return {detailRows, rowsByRun};
  }
  function experimentGroupMarkdown(group, basePayload=leaderboardViz.basePayload) {
    if (!group) return "";
    const slots = experimentSlots(group);
    const batches = group.batches || [];
    const {detailRows, rowsByRun} = experimentGroupExportRows(group, basePayload);
    const metricHeaders = EXPERIMENT_EXPORT_METRICS.map(metric => metric.label);
    const detailHeaders = [
      "batch", "batch-id", "slot", "framework", "model",
      "run-id", "run-framework", "run-model", "status", ...metricHeaders
    ];
    const detailLines = detailRows.map(row => {
      const identity = [
        row.batch, row.batch_id, row.slot_id, row.framework, row.model,
        row.run_id || "—", row.run_framework, row.run_model, row.status
      ].map(markdownExperimentText);
      const metrics = EXPERIMENT_EXPORT_METRICS.map(metric => (
        markdownExperimentNumber(row.result?.[metric.field], metric.scale)
      ));
      return `| ${[...identity, ...metrics].join(" | ")} |`;
    });
    const summaryHeaders = [
      "slot", "framework", "model", "available-runs", ...metricHeaders
    ];
    const summaryLines = slots.map(slot => {
      const sourceRows = batches
        .map(batch => rowsByRun.get(String(batch.bindings?.[slot.id] || "")))
        .filter(Boolean);
      const identity = [
        slot.id, slot.framework, slot.model,
        `${sourceRows.length}/${batches.length}`
      ].map(markdownExperimentText);
      const metrics = EXPERIMENT_EXPORT_METRICS.map(metric => (
        markdownExperimentSummary(
          sourceRows.map(row => row?.[metric.field]),
          metric.scale
        )
      ));
      return `| ${[...identity, ...metrics].join(" | ")} |`;
    });
    const table = (headers, rows) => [
      `| ${headers.join(" | ")} |`,
      `| ${headers.map(() => "---").join(" | ")} |`,
      ...(rows.length ? rows : [`| ${headers.map((_, index) => index ? "" : "—").join(" | ")} |`])
    ].join("\n");
    return [
      `# Experiment group: ${markdownExperimentText(group.name || group.id)}`,
      "",
      `- group-id: \`${markdownExperimentText(group.id)}\``,
      `- batches: ${batches.length}`,
      `- slots: ${slots.length}`,
      "",
      "`model` is the configured slot model; `run-model` is the model recorded by the bound run.",
      "Percent columns are exported in percentage-point units. Summaries use mean ± sample SD (n); sample SD uses n−1.",
      "",
      "## Batch runs",
      "",
      table(detailHeaders, detailLines),
      "",
      "## Final slot summaries (mean ± sample SD)",
      "",
      table(summaryHeaders, summaryLines),
      ""
    ].join("\n");
  }
  function downloadSelectedExperimentGroupMarkdown() {
    const group = selectedExperimentGroup();
    if (!group) {
      setExperimentStatus("Select an experiment group before exporting", true);
      return;
    }
    if (!leaderboardViz.fullPayloadLoaded) {
      setExperimentStatus(
        "Complete leaderboard metrics are still loading; export will be available when they finish.",
        true
      );
      return;
    }
    const markdown = experimentGroupMarkdown(group, leaderboardViz.basePayload);
    const blob = new Blob([markdown], {type: "text/markdown;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const safeName = String(group.name || group.id || "experiment-group")
      .trim()
      .replace(/[^\w.-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      || "experiment-group";
    link.href = url;
    link.download = `${safeName}-batch-analysis.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setExperimentStatus(`Exported ${group.name || group.id}`);
  }
