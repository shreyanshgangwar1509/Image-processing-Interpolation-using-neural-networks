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
  const toggleZoom = document.getElementById("toggle-zoom");

  const viewCardsContainer = document.getElementById("view-container-cards");
  const viewSliderContainer = document.getElementById("view-container-slider");
  const sliderModelSelect = document.getElementById("slider-model-select");
  const sliderImgBefore = document.getElementById("slider-img-before");
  const sliderImgAfter = document.getElementById("slider-img-after");
  const sliderBeforeWrapper = document.getElementById("slider-before-wrapper");
  const sliderRange = document.getElementById("slider-range");

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
        chip.innerHTML = `
          <input type="checkbox" value="${m.id}" checked>
          <div class="chip-info">
            <span class="chip-title">${m.label}</span>
            <span class="chip-sub">Scale: x${m.scale} | Degradation: ${m.degradation} | Params: ${(m.param_count / 1000).toFixed(1)}k</span>
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
      <div class="result-img-box">
        <img src="${data.lr_b64}" alt="Input image">
      </div>
      <div class="result-card-footer">
        <span>Input Resolution: ${data.input_width}x${data.input_height} px</span>
      </div>
    `;
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
          <div class="result-img-box">
            <img src="${imgSrc}" alt="${res.model_label}">
          </div>
          <div class="result-card-footer">
            <div class="metric-chips">
              <span class="m-chip">Time: ${(res.inference_time_s * 1000).toFixed(0)}ms</span>
              ${m.psnr ? `<span class="m-chip">PSNR: ${m.psnr.toFixed(2)}dB</span>` : ""}
              ${m.ssim ? `<span class="m-chip">SSIM: ${m.ssim.toFixed(4)}</span>` : ""}
            </div>
            <a href="${res.sr_b64}" download="${res.model_label}_sr.png" class="btn-text" style="align-self: flex-end; margin-top: 0.2rem;">📥 Download PNG</a>
          </div>
        `;
      }

      viewCardsContainer.appendChild(card);
    });
  }

  function renderSliderView(data) {
    sliderModelSelect.innerHTML = "";
    const validResults = data.results.filter((r) => !r.error);

    if (validResults.length === 0) return;

    validResults.forEach((r, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${r.model_label} (x${r.scale})`;
      sliderModelSelect.appendChild(opt);
    });

    sliderImgBefore.src = data.hr_b64 || data.lr_b64;

    function updateSliderImages() {
      const selIdx = parseInt(sliderModelSelect.value) || 0;
      const res = validResults[selIdx];
      if (res) {
        sliderImgAfter.src = res.sr_b64;
      }
    }

    sliderModelSelect.onchange = updateSliderImages;
    updateSliderImages();

    sliderRange.oninput = (e) => {
      sliderBeforeWrapper.style.width = `${e.target.value}%`;
    };
  }

  function renderMetricsTableAndChart(data) {
    metricsTableBody.innerHTML = "";
    psnrChart.innerHTML = "";

    const hasRef = data.has_reference;
    metricsRefBadge.textContent = hasRef ? "Ground Truth Reference Available" : "Reference Required for PSNR/SSIM";
    metricsRefBadge.className = hasRef ? "badge badge-success" : "badge";

    metricsNote.textContent = hasRef
      ? "PSNR and SSIM evaluated on Y-channel (BT.601) with border cropping matching paper conventions."
      : "No reference image supplied. Showing model metadata and inference runtime.";

    const validResults = data.results.filter((r) => !r.error);

    // Find best metric values
    let bestPsnr = -1, bestSsim = -1, bestLpips = 999;
    validResults.forEach((r) => {
      const m = r.metrics || {};
      if (m.psnr && m.psnr > bestPsnr) bestPsnr = m.psnr;
      if (m.ssim && m.ssim > bestSsim) bestSsim = m.ssim;
      if (m.lpips && m.lpips < bestLpips) bestLpips = m.lpips;
    });

    validResults.forEach((r) => {
      const m = r.metrics || {};
      const row = document.createElement("tr");

      const psnrCls = m.psnr === bestPsnr ? "best-val" : "";
      const ssimCls = m.ssim === bestSsim ? "best-val" : "";
      const lpipsCls = m.lpips === bestLpips ? "best-val" : "";

      row.innerHTML = `
        <td><strong>${r.model_label}</strong></td>
        <td>x${r.scale}</td>
        <td>${r.param_count ? (r.param_count / 1000).toFixed(1) + "k" : "N/A"}</td>
        <td>${(r.inference_time_s * 1000).toFixed(1)} ms</td>
        <td class="${psnrCls}">${m.psnr ? m.psnr.toFixed(2) + " dB" : "N/A"}</td>
        <td class="${ssimCls}">${m.ssim ? m.ssim.toFixed(4) : "N/A"}</td>
        <td class="${lpipsCls}">${m.lpips ? m.lpips.toFixed(4) : "N/A"}</td>
      `;
      metricsTableBody.appendChild(row);

      // Render chart bar
      if (m.psnr) {
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
      }
    });
  }

  // 6. View Mode Toggles
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
    } else if (mode === "diff") {
      btnViewDiff.classList.add("active");
      viewCardsContainer.classList.remove("hidden");
      viewSliderContainer.classList.add("hidden");
      if (currentRunResults) renderCardsView(currentRunResults);
    }
  }

  toggleZoom.addEventListener("change", (e) => {
    if (e.target.checked) viewCardsContainer.classList.add("zoom-100");
    else viewCardsContainer.classList.remove("zoom-100");
  });

  // 7. Reset & Modal
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
