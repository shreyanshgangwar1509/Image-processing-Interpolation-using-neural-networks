// Single-Page Application Frontend Logic for Image Super-Resolution Benchmark

document.addEventListener("DOMContentLoaded", () => {
  // App State
  let currentFile = null;
  let currentRefFile = null;
  let discoveredModels = [];
  let currentRunResults = null;
  let activeViewMode = "cards"; // 'cards' | 'slider' | 'diff'

  // DOM Elements
  const deviceChip = document.getElementById("device-chip");
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const previewBox = document.getElementById("preview-box");
  const imgPreview = document.getElementById("img-preview");
  const metaFilename = document.getElementById("meta-filename");
  const metaDims = document.getElementById("meta-dims");
  const dropPrompt = dropZone.querySelector(".drop-prompt");

  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const degradationOpts = document.getElementById("degradation-opts");
  const sepRefOpts = document.getElementById("sep-ref-opts");
  const refFileInput = document.getElementById("ref-file-input");

  const selectDevice = document.getElementById("select-device");
  const modelsList = document.getElementById("models-list");
  const btnSelectAll = document.getElementById("btn-select-all");
  const btnRun = document.getElementById("btn-run");

  const emptyState = document.getElementById("empty-state");
  const progressState = document.getElementById("progress-state");
  const progressStatus = document.getElementById("progress-status");
  const resultsDashboard = document.getElementById("results-dashboard");

  const sumDims = document.getElementById("sum-dims");
  const sumCount = document.getElementById("sum-count");
  const sumTime = document.getElementById("sum-time");
  const sumDevice = document.getElementById("sum-device");
  const btnDownloadZip = document.getElementById("btn-download-zip");
  const btnReset = document.getElementById("btn-reset");

  const btnViewCards = document.getElementById("view-cards");
  const btnViewSlider = document.getElementById("view-slider");
  const btnViewDiff = document.getElementById("view-diff");

  const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
  const mainContent = document.querySelector(".main-content");

  const btnLayoutAuto = document.getElementById("btn-layout-auto");
  const btnLayout2col = document.getElementById("btn-layout-2col");
  const btnLayout1col = document.getElementById("btn-layout-1col");

  const modalInspector = document.getElementById("modal-inspector");
  const inspectorModelTitle = document.getElementById("inspector-model-title");
  const inspectorScaleBadge = document.getElementById("inspector-scale-badge");
  const inspectorDimsBadge = document.getElementById("inspector-dims-badge");
  const inspectorModelSwitcher = document.getElementById("inspector-model-switcher");
  const btnInspectorDownload = document.getElementById("btn-inspector-download");
  const btnCloseInspector = document.getElementById("btn-close-inspector");
  const inspectorViewport = document.getElementById("inspector-viewport");
  const inspectorCanvas = document.getElementById("inspector-canvas");
  const inspectorImg = document.getElementById("inspector-img");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const hudZoomLevel = document.getElementById("hud-zoom-level");
  const btnZoom100 = document.getElementById("btn-zoom-100");
  const btnZoomFit = document.getElementById("btn-zoom-fit");
  const btnTogglePixelated = document.getElementById("btn-toggle-pixelated");

  const viewCardsContainer = document.getElementById("view-container-cards");
  const viewSliderContainer = document.getElementById("view-container-slider");
  const cocoSlider = document.getElementById("coco-slider");
  const sliderDivider = document.getElementById("slider-divider");
  const sliderHandle = document.getElementById("slider-handle");
  const sliderBeforeSelect = document.getElementById("slider-before-select");
  const sliderAfterSelect = document.getElementById("slider-after-select");
  const sliderImgBefore = document.getElementById("slider-img-before");
  const sliderImgAfter = document.getElementById("slider-img-after");
  const sliderBeforeWrapper = document.getElementById("slider-before-wrapper");
  const sliderRange = document.getElementById("slider-range");
  const sliderPctLabel = document.getElementById("slider-pct-label");
  const sliderBadgeBeforeText = document.getElementById("slider-badge-before-text");
  const sliderBadgeAfterText = document.getElementById("slider-badge-after-text");
  const sliderPresetButtons = document.querySelectorAll(".btn-slider-preset");
  const sliderDragHint = document.getElementById("slider-drag-hint");

  const metricsTableBody = document.querySelector("#metrics-table tbody");
  const psnrChart = document.getElementById("psnr-chart");
  const metricsRefBadge = document.getElementById("metrics-ref-badge");
  const metricsNote = document.getElementById("metrics-note");

  const btnAbout = document.getElementById("btn-about");
  const modalAbout = document.getElementById("modal-about");
  const btnCloseModal = document.getElementById("btn-close-modal");

  // 1. Fetch Discovered Models
  async function fetchModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      discoveredModels = data.models || [];

      // Update Device Chip
      const gpuStr = data.cuda_available ? `GPU: ${data.device_name}` : "CPU";
      deviceChip.textContent = `Hardware: ${gpuStr}`;
      deviceChip.className = data.cuda_available ? "badge badge-info" : "badge";

      renderModelsList();
    } catch (err) {
      modelsList.innerHTML = `<p class="text-muted" style="color: var(--danger)">Failed to load models: ${err.message}</p>`;
    }
  }

  function renderModelsList() {
    if (discoveredModels.length === 0) {
      modelsList.innerHTML = `<p class="text-muted">No eligible neural checkpoints found in <code>results/</code>.<br>Run an experiment first to produce <code>best.pth</code>.</p>`;
      btnRun.disabled = true;
      return;
    }

    modelsList.innerHTML = "";
    const interpolationModels = discoveredModels.filter(
      (model) => model.type === "interpolation"
    );
    const neuralModels = discoveredModels.filter(
      (model) => model.type !== "interpolation"
    );

    function renderGroup(title, models) {
      if (models.length === 0) return;

      const group = document.createElement("div");
      group.className = "model-group";
      const heading = document.createElement("h3");
      heading.className = "model-group-title";
      heading.textContent = title;
      group.appendChild(heading);

      models.forEach((m) => {
        const chip = document.createElement("label");
        chip.className = "model-chip";
        if (m.type === "interpolation") {
          chip.dataset.interpolation = "true";
        }
        const isAnalytical = m.type === "interpolation" || !m.param_count;
        const paramLabel = isAnalytical ? "Analytical" : `${(m.param_count / 1000).toFixed(1)}k`;
        chip.innerHTML = `
          <input type="checkbox" value="${m.id}" checked>
          <div class="chip-info">
            <span class="chip-title">${m.label}</span>
            <span class="chip-sub">Scale: x${m.scale} | Degradation: ${m.degradation} | Params: ${paramLabel}</span>
          </div>
        `;
        group.appendChild(chip);
      });

      modelsList.appendChild(group);
    }

    renderGroup("Standard interpolation", interpolationModels);
    renderGroup("Neural checkpoints", neuralModels);
    if (neuralModels.length === 0) {
      const hint = document.createElement("p");
      hint.className = "text-muted";
      hint.textContent =
        "No neural checkpoints found. Standard interpolation methods are ready to use.";
      modelsList.appendChild(hint);
    }

    updateInterpolationAvailability();

    updateRunButtonState();
  }

  function updateInterpolationAvailability() {
    const isHrReference = document.querySelector(
      'input[name="mode"]:checked'
    ).value === "hr_reference";

    modelsList.querySelectorAll("[data-interpolation='true']").forEach((chip) => {
      const checkbox = chip.querySelector("input[type='checkbox']");
      chip.hidden = isHrReference;
      checkbox.disabled = isHrReference;
      if (isHrReference) checkbox.checked = false;
    });
  }

  function updateRunButtonState() {
    const selected = modelsList.querySelectorAll(
      "input[type='checkbox']:checked:not(:disabled)"
    );
    btnRun.disabled = !(currentFile && selected.length > 0);
  }

  btnSelectAll.addEventListener("click", (e) => {
    e.preventDefault();
    const checkboxes = modelsList.querySelectorAll(
      "input[type='checkbox']:not(:disabled)"
    );
    const allChecked = Array.from(checkboxes).every((cb) => cb.checked);
    checkboxes.forEach((cb) => (cb.checked = !allChecked));
    updateRunButtonState();
  });

  modelsList.addEventListener("change", updateRunButtonState);

  // 2. File Upload & Drag/Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
  });

  function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file (PNG, JPEG, WebP, BMP).");
      return;
    }

    currentFile = file;
    metaFilename.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      imgPreview.src = e.target.result;
      imgPreview.onload = () => {
        metaDims.textContent = `${imgPreview.naturalWidth} x ${imgPreview.naturalHeight} px`;
      };
      dropPrompt.classList.add("hidden");
      previewBox.classList.remove("hidden");
      updateRunButtonState();
    };
    reader.readAsDataURL(file);
  }

  // 3. Mode Selection Logic
  modeRadios.forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const mode = e.target.value;
      if (mode === "hr_reference") {
        degradationOpts.classList.remove("hidden");
        sepRefOpts.classList.add("hidden");
      } else {
        degradationOpts.classList.add("hidden");
        sepRefOpts.classList.remove("hidden");
      }
      updateInterpolationAvailability();
      updateRunButtonState();
    });
  });

  refFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) currentRefFile = e.target.files[0];
    else currentRefFile = null;
  });

  // 4. Run Benchmark Inference
  btnRun.addEventListener("click", async () => {
    if (!currentFile) return;

    const selectedCheckboxes = modelsList.querySelectorAll(
      "input[type='checkbox']:checked:not(:disabled)"
    );
    const selectedModelIds = Array.from(selectedCheckboxes).map((cb) => cb.value);

    if (selectedModelIds.length === 0) {
      alert("Please select at least one model.");
      return;
    }

    const selectedMode = document.querySelector('input[name="mode"]:checked').value;
    const selectedDevice = selectDevice.value;
    const selectedDegradation = document.getElementById("select-degradation").value;

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("mode", selectedMode);
    formData.append("selected_models", selectedModelIds.join(","));
    formData.append("device_request", selectedDevice);
    formData.append("degradation_type", selectedDegradation);

    if (selectedMode === "lr_input" && currentRefFile) {
      formData.append("ref_file", currentRefFile);
    }

    // UI state transition
    emptyState.classList.add("hidden");
    resultsDashboard.classList.add("hidden");
    progressState.classList.remove("hidden");
    btnRun.disabled = true;

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Inference execution failed.");
      }

      currentRunResults = await response.json();
      displayResults(currentRunResults);
    } catch (err) {
      alert(`Error during inference: ${err.message}`);
      emptyState.classList.remove("hidden");
    } finally {
      progressState.classList.add("hidden");
      btnRun.disabled = false;
    }
  });

  // 5. Display Results
  function displayResults(data) {
    resultsDashboard.classList.remove("hidden");

    // Summary Bar
    sumDims.textContent = `${data.input_width} x ${data.input_height} px`;
    sumCount.textContent = data.results.length;
    sumTime.textContent = `${data.total_elapsed_s.toFixed(2)}s`;
    sumDevice.textContent = data.device_used;

    // Configure Zip Download Button
    btnDownloadZip.onclick = () => {
      window.location.href = `/api/download/${data.session_id}`;
    };

    // Render Views
    renderCardsView(data);
    renderSliderView(data);
    renderMetricsTableAndChart(data);
  }

  function renderCardsView(data) {
    viewCardsContainer.innerHTML = "";

    // First card: Input / Reference
    const inputCard = document.createElement("div");
    inputCard.className = "result-card";
    inputCard.innerHTML = `
      <div class="result-card-header">
        <span>${data.mode === "hr_reference" ? "Degraded LR Input" : "Uploaded LR Input"}</span>
        <span class="badge">${data.input_width}x${data.input_height}</span>
      </div>
      <div class="result-img-box" title="Click to open interactive zoom & pan inspector">
        <img src="${data.lr_b64}" alt="Input image">
      </div>
      <div class="result-card-footer">
        <span>Input Resolution: ${data.input_width}x${data.input_height} px</span>
      </div>
    `;
    const inputImgBox = inputCard.querySelector(".result-img-box");
    if (inputImgBox) {
      inputImgBox.addEventListener("click", () => openInspector("input"));
    }
    viewCardsContainer.appendChild(inputCard);

    // Model Result Cards
    data.results.forEach((res) => {
      const card = document.createElement("div");
      card.className = "result-card";

      if (res.error) {
        card.innerHTML = `
          <div class="result-card-header">
            <span>${res.model_label}</span>
            <span class="badge badge-danger">Failed</span>
          </div>
          <div class="result-img-box" style="color: var(--danger); font-size: 0.85rem; padding: 1rem;">
            Error: ${res.error}
          </div>
        `;
      } else {
        const imgSrc = activeViewMode === "diff" && res.diff_b64 ? res.diff_b64 : res.sr_b64;
        const m = res.metrics || {};

        card.innerHTML = `
          <div class="result-card-header">
            <span>${res.model_label} (x${res.scale})</span>
            <span class="badge badge-info">${res.width}x${res.height}</span>
          </div>
          <div class="result-img-box" title="Click to open interactive zoom & pan inspector">
            <img src="${imgSrc}" alt="${res.model_label}">
          </div>
          <div class="result-card-footer">
            <div class="metric-chips">
              <span class="m-chip">Time: ${(res.inference_time_s * 1000).toFixed(0)}ms</span>
              ${m.psnr ? `<span class="m-chip">PSNR: ${m.psnr.toFixed(2)}dB</span>` : ""}
              ${m.ssim ? `<span class="m-chip">SSIM: ${m.ssim.toFixed(4)}</span>` : ""}
            </div>
            <a href="${res.sr_b64}" download="${res.model_label}_sr.png" class="btn-text btn-download-card" title="Download Full Resolution PNG">📥 Download PNG</a>
          </div>
        `;
        const resultImgBox = card.querySelector(".result-img-box");
        if (resultImgBox) {
          resultImgBox.addEventListener("click", () => openInspector(res.model_name));
        }
      }

      viewCardsContainer.appendChild(card);
    });
  }

  function renderSliderView(data) {
    if (!sliderBeforeSelect || !sliderAfterSelect) return;

    sliderBeforeSelect.innerHTML = "";
    sliderAfterSelect.innerHTML = "";

    const compareOptions = [];

    if (data.mode === "hr_reference") {
      if (data.hr_b64) {
        compareOptions.push({
          id: "orig_hr",
          label: "Original",
          src: data.hr_b64,
        });
      }
      if (data.lr_b64) {
        compareOptions.push({
          id: "orig_lr",
          label: "Original LR",
          src: data.lr_b64,
        });
      }
    } else {
      // lr_input mode
      if (data.hr_b64) {
        // Optional ground truth HR reference provided
        compareOptions.push({
          id: "orig_hr",
          label: "Original",
          src: data.hr_b64,
        });
        if (data.lr_b64) {
          compareOptions.push({
            id: "orig_lr",
            label: "Original LR",
            src: data.lr_b64,
          });
        }
      } else if (data.lr_b64) {
        // Only LR input provided
        compareOptions.push({
          id: "orig_lr",
          label: "Original",
          src: data.lr_b64,
        });
      }
    }

    // Add evaluated models that ran without error
    const validResults = (data.results || []).filter((r) => !r.error && r.sr_b64);
    validResults.forEach((r, idx) => {
      compareOptions.push({
        id: `model_${idx}`,
        label: `${r.model_label} (x${r.scale})`,
        src: r.sr_b64,
      });
    });

    if (compareOptions.length === 0) return;

    compareOptions.forEach((opt, idx) => {
      const elBefore = document.createElement("option");
      elBefore.value = idx;
      elBefore.textContent = opt.label;
      sliderBeforeSelect.appendChild(elBefore);

      const elAfter = document.createElement("option");
      elAfter.value = idx;
      elAfter.textContent = opt.label;
      sliderAfterSelect.appendChild(elAfter);
    });

    // Default left to "Original" (index 0)
    const defaultBeforeIdx = 0;
    // Default right to the first evaluated model, or index 1
    let defaultAfterIdx = compareOptions.findIndex((opt) => opt.id.startsWith("model_"));
    if (defaultAfterIdx === -1) {
      defaultAfterIdx = compareOptions.length > 1 ? 1 : 0;
    }

    sliderBeforeSelect.value = String(defaultBeforeIdx);
    sliderAfterSelect.value = String(defaultAfterIdx);

    function updateSliderImages() {
      const beforeIdx = parseInt(sliderBeforeSelect.value, 10) || 0;
      const afterIdx = parseInt(sliderAfterSelect.value, 10) || 0;

      const beforeOpt = compareOptions[beforeIdx] || compareOptions[0];
      const afterOpt = compareOptions[afterIdx] || compareOptions[compareOptions.length > 1 ? 1 : 0];

      if (beforeOpt && sliderImgBefore) {
        sliderImgBefore.src = beforeOpt.src;
        if (sliderBadgeBeforeText) {
          sliderBadgeBeforeText.textContent = beforeOpt.label;
        }
      }

      if (afterOpt && sliderImgAfter) {
        sliderImgAfter.src = afterOpt.src;
        if (sliderBadgeAfterText) {
          sliderBadgeAfterText.textContent = afterOpt.label;
        }
      }
    }

    sliderBeforeSelect.onchange = updateSliderImages;
    sliderAfterSelect.onchange = updateSliderImages;
    updateSliderImages();

    let currentPct = 50;

    function updateSliderPosition(pct, syncRangeInput = true) {
      currentPct = Math.max(0, Math.min(100, pct));
      if (cocoSlider) {
        cocoSlider.style.setProperty("--slider-pos", `${currentPct}%`);
        cocoSlider.setAttribute("aria-valuenow", Math.round(currentPct));
      }
      if (sliderBeforeWrapper) {
        sliderBeforeWrapper.style.clipPath = `inset(0 ${100 - currentPct}% 0 0)`;
      }
      if (sliderDivider) {
        sliderDivider.style.left = `${currentPct}%`;
      }
      if (syncRangeInput && sliderRange) {
        sliderRange.value = Math.round(currentPct);
      }
      if (sliderPctLabel) {
        sliderPctLabel.textContent = `${Math.round(currentPct)}%`;
      }
      const rounded = Math.round(currentPct);
      sliderPresetButtons.forEach((btn) => {
        const targetPos = parseInt(btn.dataset.pos, 10);
        btn.classList.toggle("active", targetPos === rounded);
      });
    }

    // Connect Range Input
    if (sliderRange) {
      sliderRange.oninput = (e) => {
        updateSliderPosition(parseFloat(e.target.value), false);
      };
    }

    // Connect Presets
    sliderPresetButtons.forEach((btn) => {
      btn.onclick = () => {
        const targetPos = parseInt(btn.dataset.pos, 10);
        updateSliderPosition(targetPos, true);
      };
    });

    // Connect Dragging directly on image and handle
    if (cocoSlider) {
      let isDragging = false;

      function getPosFromPointer(e) {
        const rect = cocoSlider.getBoundingClientRect();
        if (rect.width <= 0) return 50;
        const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
        const offsetX = clientX - rect.left;
        return Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
      }

      function onPointerDown(e) {
        if (e.button !== undefined && e.button !== 0) return;
        isDragging = true;
        cocoSlider.classList.add("is-dragging");
        cocoSlider.classList.add("hint-hidden");
        try {
          cocoSlider.setPointerCapture(e.pointerId);
        } catch (_) {}
        const pct = getPosFromPointer(e);
        updateSliderPosition(pct, true);
      }

      function onPointerMove(e) {
        if (!isDragging) return;
        const pct = getPosFromPointer(e);
        updateSliderPosition(pct, true);
      }

      function onPointerUp(e) {
        if (!isDragging) return;
        isDragging = false;
        cocoSlider.classList.remove("is-dragging");
        try {
          cocoSlider.releasePointerCapture(e.pointerId);
        } catch (_) {}
      }

      cocoSlider.onpointerdown = onPointerDown;
      cocoSlider.onpointermove = onPointerMove;
      cocoSlider.onpointerup = onPointerUp;
      cocoSlider.onpointercancel = onPointerUp;

      // Accessible keyboard control on slider
      cocoSlider.onkeydown = (e) => {
        const step = e.shiftKey ? 10 : 2;
        if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
          e.preventDefault();
          updateSliderPosition(currentPct - step, true);
        } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
          e.preventDefault();
          updateSliderPosition(currentPct + step, true);
        } else if (e.key === "Home") {
          e.preventDefault();
          updateSliderPosition(0, true);
        } else if (e.key === "End") {
          e.preventDefault();
          updateSliderPosition(100, true);
        }
      };
    }

    // Initialize to 50%
    updateSliderPosition(50, true);
  }

  function renderMetricsTableAndChart(data) {
    metricsTableBody.innerHTML = "";
    psnrChart.innerHTML = "";

    const hasRef = data.has_reference;
    metricsRefBadge.textContent = hasRef ? "Ground Truth Reference Available" : "Reference Required for PSNR/SSIM";
    metricsRefBadge.className = hasRef ? "badge badge-success" : "badge badge-warning";

    metricsNote.innerHTML = hasRef
      ? "PSNR and SSIM evaluated on Y-channel (BT.601) with border cropping matching paper conventions."
      : "<strong>LR Direct Upscaling:</strong> No High-Resolution reference image was supplied. Quantitative full-reference metrics (PSNR, SSIM) require an HR ground-truth image. Attach an optional HR reference above or switch to <strong>Image is HR Reference</strong> mode to compute fidelity.";

    const validResults = data.results.filter((r) => !r.error);

    // Find best metric values
    let bestPsnr = -1, bestSsim = -1;
    validResults.forEach((r) => {
      const m = r.metrics || {};
      if (m.psnr && m.psnr > bestPsnr) bestPsnr = m.psnr;
      if (m.ssim && m.ssim > bestSsim) bestSsim = m.ssim;
    });

    validResults.forEach((r) => {
      const m = r.metrics || {};
      const row = document.createElement("tr");

      const psnrCls = m.psnr === bestPsnr ? "best-val" : "";
      const ssimCls = m.ssim === bestSsim ? "best-val" : "";

      const isAnalytical = r.model_type === "interpolation" || r.type === "interpolation" || !r.param_count;
      const paramStr = isAnalytical
        ? `<span class="param-analytical" title="Non-parametric mathematical algorithm">— (Analytical)</span>`
        : (r.param_count ? `${(r.param_count / 1000).toFixed(1)}k` : "—");

      const psnrText = (m.psnr !== undefined && m.psnr !== null)
        ? `${m.psnr.toFixed(2)} dB`
        : `<span class="metric-empty" title="${hasRef ? 'Metric calculation unavailable' : 'Ground truth HR reference required for PSNR'}">${hasRef ? 'N/A' : '— (No HR Ref)'}</span>`;

      const ssimText = (m.ssim !== undefined && m.ssim !== null)
        ? m.ssim.toFixed(4)
        : `<span class="metric-empty" title="${hasRef ? 'Metric calculation unavailable' : 'Ground truth HR reference required for SSIM'}">${hasRef ? 'N/A' : '— (No HR Ref)'}</span>`;

      row.innerHTML = `
        <td><strong>${r.model_label}</strong></td>
        <td>x${r.scale}</td>
        <td>${paramStr}</td>
        <td>${(r.inference_time_s * 1000).toFixed(1)} ms</td>
        <td class="${psnrCls}">${psnrText}</td>
        <td class="${ssimCls}">${ssimText}</td>
      `;
      metricsTableBody.appendChild(row);
    });

    // Render chart bars or placeholder if no reference
    let renderedBars = 0;
    validResults.forEach((r) => {
      const m = r.metrics || {};
      if (m.psnr !== undefined && m.psnr !== null) {
        const barRow = document.createElement("div");
        barRow.className = "chart-bar-row";
        const widthPct = Math.max(10, Math.min(100, (m.psnr / 40.0) * 100));

        barRow.innerHTML = `
          <span class="chart-label">${r.model_label}</span>
          <div class="chart-bar-bg">
            <div class="chart-bar-fill" style="width: ${widthPct}%"></div>
          </div>
          <span class="chart-val">${m.psnr.toFixed(2)} dB</span>
        `;
        psnrChart.appendChild(barRow);
        renderedBars++;
      }
    });

    if (renderedBars === 0) {
      psnrChart.innerHTML = `
        <div class="chart-empty-placeholder">
          <span class="placeholder-icon">📊</span>
          <p class="placeholder-title">Fidelity Chart Unavailable Without Ground Truth</p>
          <p class="placeholder-desc">PSNR and SSIM require an uncorrupted High-Resolution reference image. Attach an optional HR reference in LR mode or select <em>Image is HR Reference</em> mode in the sidebar to generate quantitative comparison bars.</p>
        </div>
      `;
    }
  }

  // 6. View Mode Toggles & Column Density
  btnViewCards.addEventListener("click", () => setViewMode("cards"));
  btnViewSlider.addEventListener("click", () => setViewMode("slider"));
  btnViewDiff.addEventListener("click", () => setViewMode("diff"));

  function setViewMode(mode) {
    activeViewMode = mode;
    [btnViewCards, btnViewSlider, btnViewDiff].forEach((btn) => btn.classList.remove("active"));

    if (mode === "cards") {
      btnViewCards.classList.add("active");
      viewCardsContainer.classList.remove("hidden");
      viewSliderContainer.classList.add("hidden");
      if (currentRunResults) renderCardsView(currentRunResults);
    } else if (mode === "slider") {
      btnViewSlider.classList.add("active");
      viewCardsContainer.classList.add("hidden");
      viewSliderContainer.classList.remove("hidden");
      if (currentRunResults) renderSliderView(currentRunResults);
    } else if (mode === "diff") {
      btnViewDiff.classList.add("active");
      viewCardsContainer.classList.remove("hidden");
      viewSliderContainer.classList.add("hidden");
      if (currentRunResults) renderCardsView(currentRunResults);
    }
  }

  // Sidebar Collapse & Expand Logic
  const sidebar = document.getElementById("sidebar");
  const btnCollapseSidebarInner = document.getElementById("btn-collapse-sidebar-inner");

  function setSidebarCollapsed(collapse) {
    if (!mainContent || !sidebar) return;

    if (collapse) {
      mainContent.classList.add("sidebar-collapsed");
      if (btnToggleSidebar) {
        btnToggleSidebar.classList.add("collapsed-state");
        btnToggleSidebar.title = "Show Controls Sidebar";
        const iconSpan = btnToggleSidebar.querySelector(".toggle-icon");
        const textSpan = btnToggleSidebar.querySelector(".toggle-text");
        if (iconSpan) iconSpan.textContent = "▸";
        if (textSpan) textSpan.textContent = "Show Controls";
      }
      // Guarantee complete hiding at end of animation
      setTimeout(() => {
        if (mainContent.classList.contains("sidebar-collapsed")) {
          sidebar.classList.add("is-completely-hidden");
        }
      }, 260);
    } else {
      sidebar.classList.remove("is-completely-hidden");
      void sidebar.offsetWidth; // Force layout reflow
      mainContent.classList.remove("sidebar-collapsed");
      if (btnToggleSidebar) {
        btnToggleSidebar.classList.remove("collapsed-state");
        btnToggleSidebar.title = "Hide Controls Sidebar";
        const iconSpan = btnToggleSidebar.querySelector(".toggle-icon");
        const textSpan = btnToggleSidebar.querySelector(".toggle-text");
        if (iconSpan) iconSpan.textContent = "☰";
        if (textSpan) textSpan.textContent = "Controls";
      }
    }
  }

  if (sidebar) {
    sidebar.addEventListener("transitionend", (e) => {
      if (e.target === sidebar && e.propertyName === "width") {
        if (mainContent.classList.contains("sidebar-collapsed")) {
          sidebar.classList.add("is-completely-hidden");
        }
      }
    });
  }

  if (btnToggleSidebar && mainContent) {
    btnToggleSidebar.addEventListener("click", () => {
      const isCurrentlyCollapsed = mainContent.classList.contains("sidebar-collapsed");
      setSidebarCollapsed(!isCurrentlyCollapsed);
    });
  }

  if (btnCollapseSidebarInner) {
    btnCollapseSidebarInner.addEventListener("click", () => {
      setSidebarCollapsed(true);
    });
  }

  // Column Density Toggles
  function setLayoutMode(mode) {
    [btnLayoutAuto, btnLayout2col, btnLayout1col].forEach((b) => b && b.classList.remove("active"));
    viewCardsContainer.classList.remove("layout-2col", "layout-1col");

    if (mode === "2col") {
      if (btnLayout2col) btnLayout2col.classList.add("active");
      viewCardsContainer.classList.add("layout-2col");
    } else if (mode === "1col") {
      if (btnLayout1col) btnLayout1col.classList.add("active");
      viewCardsContainer.classList.add("layout-1col");
    } else {
      if (btnLayoutAuto) btnLayoutAuto.classList.add("active");
    }
  }

  if (btnLayoutAuto) btnLayoutAuto.addEventListener("click", () => setLayoutMode("auto"));
  if (btnLayout2col) btnLayout2col.addEventListener("click", () => setLayoutMode("2col"));
  if (btnLayout1col) btnLayout1col.addEventListener("click", () => setLayoutMode("1col"));

  // 7. Full-Screen Image Inspector (Zoom & Pan)
  let isInspectorOpen = false;
  let inspectorItems = [];
  let currentInspectorIdx = 0;
  let zoomLevel = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let isPixelated = false;

  function updateInspectorTransform() {
    inspectorCanvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
    hudZoomLevel.textContent = `${Math.round(zoomLevel * 100)}%`;
  }

  function zoomToFit() {
    const vpW = inspectorViewport.clientWidth || 800;
    const vpH = inspectorViewport.clientHeight || 600;
    const imgW = inspectorImg.naturalWidth || inspectorItems[currentInspectorIdx]?.width || 512;
    const imgH = inspectorImg.naturalHeight || inspectorItems[currentInspectorIdx]?.height || 512;

    const pad = 64;
    const scaleX = (vpW - pad) / imgW;
    const scaleY = (vpH - pad) / imgH;
    zoomLevel = Math.max(0.1, Math.min(Math.min(scaleX, scaleY), 2.0));
    panX = Math.round((vpW - imgW * zoomLevel) / 2);
    panY = Math.round((vpH - imgH * zoomLevel) / 2);
    updateInspectorTransform();
  }

  function zoomTo100() {
    const vpW = inspectorViewport.clientWidth || 800;
    const vpH = inspectorViewport.clientHeight || 600;
    const imgW = inspectorImg.naturalWidth || inspectorItems[currentInspectorIdx]?.width || 512;
    const imgH = inspectorImg.naturalHeight || inspectorItems[currentInspectorIdx]?.height || 512;

    zoomLevel = 1.0;
    panX = Math.round((vpW - imgW) / 2);
    panY = Math.round((vpH - imgH) / 2);
    updateInspectorTransform();
  }

  function zoomBy(factor, clientX, clientY) {
    const rect = inspectorViewport.getBoundingClientRect();
    const mouseX = clientX !== undefined ? clientX - rect.left : rect.width / 2;
    const mouseY = clientY !== undefined ? clientY - rect.top : rect.height / 2;

    const newZoom = Math.max(0.1, Math.min(32.0, zoomLevel * factor));
    if (newZoom === zoomLevel) return;

    panX = mouseX - ((mouseX - panX) / zoomLevel) * newZoom;
    panY = mouseY - ((mouseY - panY) / zoomLevel) * newZoom;
    zoomLevel = newZoom;
    updateInspectorTransform();
  }

  function selectInspectorModel(idx, keepTransform = true) {
    if (idx < 0 || idx >= inspectorItems.length) return;
    currentInspectorIdx = idx;
    const item = inspectorItems[idx];

    inspectorModelTitle.textContent = item.label;
    inspectorScaleBadge.textContent = item.scale ? `x${item.scale}` : "Input";
    inspectorScaleBadge.className = item.isInput ? "badge" : "badge badge-info";
    inspectorDimsBadge.textContent = `${item.width} x ${item.height} px`;

    btnInspectorDownload.href = item.src;
    btnInspectorDownload.download = `${item.label.toLowerCase().replace(/[^a-z0-9]/g, "_")}_inspect.png`;

    inspectorModelSwitcher.querySelectorAll(".switcher-pill").forEach((pill, pIdx) => {
      pill.classList.toggle("active", pIdx === idx);
    });

    inspectorImg.src = item.src;

    if (!keepTransform) {
      if (inspectorImg.complete && inspectorImg.naturalWidth) {
        zoomToFit();
      } else {
        inspectorImg.onload = () => {
          zoomToFit();
          inspectorImg.onload = null;
        };
      }
    } else {
      updateInspectorTransform();
    }
  }

  function openInspector(targetKey) {
    if (!currentRunResults) return;

    inspectorItems = [];

    // Input image
    inspectorItems.push({
      id: "input",
      label: currentRunResults.mode === "hr_reference" ? "Degraded LR Input" : "Uploaded LR Input",
      src: currentRunResults.lr_b64,
      width: currentRunResults.input_width,
      height: currentRunResults.input_height,
      scale: 1,
      isInput: true,
    });

    // Ground Truth Reference if available
    if (currentRunResults.hr_b64) {
      inspectorItems.push({
        id: "ground_truth",
        label: "Ground Truth Reference (HR)",
        src: currentRunResults.hr_b64,
        width: currentRunResults.results[0]?.width || currentRunResults.input_width * 4,
        height: currentRunResults.results[0]?.height || currentRunResults.input_height * 4,
        scale: currentRunResults.results[0]?.scale || 4,
        isInput: false,
      });
    }

    // Model result items
    currentRunResults.results.forEach((r) => {
      if (!r.error) {
        const src = (activeViewMode === "diff" && r.diff_b64) ? r.diff_b64 : r.sr_b64;
        inspectorItems.push({
          id: r.model_name,
          label: activeViewMode === "diff" ? `${r.model_label} (Diff)` : r.model_label,
          src: src,
          width: r.width,
          height: r.height,
          scale: r.scale,
          isInput: false,
        });
      }
    });

    if (inspectorItems.length === 0) return;

    inspectorModelSwitcher.innerHTML = "";
    inspectorItems.forEach((item, idx) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "switcher-pill";
      pill.innerHTML = `<span>${item.label}</span>${item.scale > 1 ? `<small style="opacity:0.7">x${item.scale}</small>` : ""}`;
      pill.addEventListener("click", () => selectInspectorModel(idx, true));
      inspectorModelSwitcher.appendChild(pill);
    });

    let startIdx = 0;
    if (typeof targetKey === "number") {
      startIdx = Math.max(0, Math.min(targetKey, inspectorItems.length - 1));
    } else if (typeof targetKey === "string") {
      const found = inspectorItems.findIndex((it) => it.id === targetKey);
      if (found !== -1) startIdx = found;
    }

    modalInspector.classList.remove("hidden");
    isInspectorOpen = true;

    selectInspectorModel(startIdx, false);
  }

  function closeInspector() {
    modalInspector.classList.add("hidden");
    isInspectorOpen = false;
  }

  btnCloseInspector.addEventListener("click", closeInspector);
  modalInspector.addEventListener("click", (e) => {
    if (e.target === modalInspector) closeInspector();
  });

  // Mouse pan handlers
  inspectorViewport.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    isDragging = true;
    dragStartX = e.clientX - panX;
    dragStartY = e.clientY - panY;
    inspectorViewport.classList.add("is-dragging");
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    panX = e.clientX - dragStartX;
    panY = e.clientY - dragStartY;
    updateInspectorTransform();
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      inspectorViewport.classList.remove("is-dragging");
    }
  });

  // Wheel event handler: Handles trackpad pinch-to-zoom, trackpad two-finger panning, and notched mouse wheel
  inspectorViewport.addEventListener("wheel", (e) => {
    e.preventDefault();

    if (e.ctrlKey) {
      // 1. Touchpad / Trackpad pinch-to-zoom gesture (browser fires wheel with ctrlKey)
      // Continuous fractional deltaY scaling
      const factor = Math.exp(-e.deltaY * 0.01);
      zoomBy(factor, e.clientX, e.clientY);
    } else if (e.deltaMode !== 0 || (Math.abs(e.deltaY) >= 40 && Math.abs(e.deltaX) === 0)) {
      // 2. Physical notched desktop mouse wheel
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      zoomBy(factor, e.clientX, e.clientY);
    } else {
      // 3. Touchpad two-finger pan scroll
      panX -= e.deltaX;
      panY -= e.deltaY;
      updateInspectorTransform();
    }
  }, { passive: false });

  // Touch pan & pinch zoom support (for mobile/tablet touch screens)
  let lastTouchDist = 0;
  let lastTouchMidX = 0;
  let lastTouchMidY = 0;

  inspectorViewport.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      dragStartX = e.touches[0].clientX - panX;
      dragStartY = e.touches[0].clientY - panY;
    } else if (e.touches.length === 2) {
      isDragging = false;
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      lastTouchDist = Math.hypot(dx, dy);
      lastTouchMidX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      lastTouchMidY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    }
  }, { passive: true });

  inspectorViewport.addEventListener("touchmove", (e) => {
    if (e.touches.length === 1 && isDragging) {
      panX = e.touches[0].clientX - dragStartX;
      panY = e.touches[0].clientY - dragStartY;
      updateInspectorTransform();
    } else if (e.touches.length === 2 && lastTouchDist > 0) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const currentDist = Math.hypot(dx, dy);
      const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;

      if (lastTouchDist > 0 && currentDist > 0) {
        const factor = currentDist / lastTouchDist;
        zoomBy(factor, midX, midY);
      }

      panX += (midX - lastTouchMidX);
      panY += (midY - lastTouchMidY);
      updateInspectorTransform();

      lastTouchDist = currentDist;
      lastTouchMidX = midX;
      lastTouchMidY = midY;
    }
  }, { passive: true });

  inspectorViewport.addEventListener("touchend", () => {
    isDragging = false;
    lastTouchDist = 0;
  });

  // Inspector HUD Controls
  btnZoomIn.addEventListener("click", () => zoomBy(1.3));
  btnZoomOut.addEventListener("click", () => zoomBy(1 / 1.3));
  btnZoom100.addEventListener("click", zoomTo100);
  btnZoomFit.addEventListener("click", zoomToFit);
  btnTogglePixelated.addEventListener("click", () => {
    isPixelated = !isPixelated;
    inspectorImg.classList.toggle("pixelated", isPixelated);
    btnTogglePixelated.classList.toggle("active", isPixelated);
  });

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (modalAbout && !modalAbout.classList.contains("hidden") && e.key === "Escape") {
      modalAbout.classList.add("hidden");
      return;
    }

    if (!isInspectorOpen) return;

    if (e.key === "Escape") {
      closeInspector();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const nextIdx = (currentInspectorIdx - 1 + inspectorItems.length) % inspectorItems.length;
      selectInspectorModel(nextIdx, true);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      const nextIdx = (currentInspectorIdx + 1) % inspectorItems.length;
      selectInspectorModel(nextIdx, true);
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      zoomBy(1.25);
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault();
      zoomBy(1 / 1.25);
    } else if (e.key === "0") {
      e.preventDefault();
      zoomToFit();
    } else if (e.key === "1") {
      e.preventDefault();
      zoomTo100();
    } else if (e.key.toLowerCase() === "p") {
      e.preventDefault();
      btnTogglePixelated.click();
    }
  });

  // 8. Reset & About Modal
  btnReset.addEventListener("click", () => {
    currentFile = null;
    currentRefFile = null;
    currentRunResults = null;
    fileInput.value = "";
    refFileInput.value = "";

    dropPrompt.classList.remove("hidden");
    previewBox.classList.add("hidden");
    resultsDashboard.classList.add("hidden");
    emptyState.classList.remove("hidden");
    btnRun.disabled = true;
  });

  btnAbout.onclick = () => modalAbout.classList.remove("hidden");
  btnCloseModal.onclick = () => modalAbout.classList.add("hidden");

  // Initial Load
  fetchModels();
});
