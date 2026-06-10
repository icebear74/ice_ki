const state = {
  ollamaModels: [],
  checkpoints: [],
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

/**
 * Fill a <select> element with option values.
 * Keeps a leading "manual" placeholder option if no values are available.
 */
function fillSelect(selectId, manualWrapId, values) {
  const select = $(selectId);
  const currentVal = select.value;
  select.innerHTML = "";

  if (values.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "– keine gefunden –";
    select.appendChild(opt);
    $(manualWrapId).classList.remove("hidden");
    return;
  }

  $(manualWrapId).classList.add("hidden");
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  }
  // Restore previous selection if still in list
  if (currentVal && values.includes(currentVal)) {
    select.value = currentVal;
  }
}

/** Return the effective value: select element, or manual input if visible. */
function selectValue(selectId, manualInputId, manualWrapId) {
  const wrap = $(manualWrapId);
  if (wrap && !wrap.classList.contains("hidden")) {
    return $(manualInputId).value.trim();
  }
  return $(selectId).value.trim();
}

async function loadOllamaModels() {
  setStatus("Lade Ollama-Modelle …");
  const data = await api("/api/ollama/models");
  state.ollamaModels = data.models || [];
  fillSelect("ollamaModel", "ollamaModelManualWrap", state.ollamaModels);
  setStatus(`Ollama-Modelle geladen: ${state.ollamaModels.length}`);
}

async function loadCheckpoints() {
  setStatus("Lade ComfyUI-Checkpoints …");
  const data = await api("/api/comfy/checkpoints");
  state.checkpoints = data.checkpoints || [];
  fillSelect("checkpoint", "checkpointManualWrap", state.checkpoints);
  $("checkpointNote").textContent = data.note || "";
  setStatus(`Checkpoints geladen: ${state.checkpoints.length}`);
}

function collectPayload() {
  return {
    prompt_de: $("promptDe").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: selectValue("ollamaModel", "ollamaModelManual", "ollamaModelManualWrap"),
    translated_prompt: $("translatedPrompt").value.trim() || null,
    checkpoint: selectValue("checkpoint", "checkpointManual", "checkpointManualWrap") || null,
    steps: Number($("steps").value),
    cfg: Number($("cfg").value),
    seed: Number($("seed").value),
    width: Number($("width").value),
    height: Number($("height").value),
    sampler: $("sampler").value.trim(),
    scheduler: $("scheduler").value.trim(),
    image_count: Number($("imageCount").value),
  };
}

function showImages(urls) {
  const wrap = $("images");
  wrap.innerHTML = "";
  for (const url of urls) {
    const img = document.createElement("img");
    img.src = `${url}&_=${Date.now()}`;
    img.alt = "Generated image";
    wrap.appendChild(img);
  }
}

async function translateOnly() {
  const payload = collectPayload();
  if (!payload.prompt_de || !payload.ollama_model) {
    throw new Error("Bitte deutschen Prompt und Ollama-Modell eingeben.");
  }

  setStatus("Übersetze Prompt …");
  const data = await api("/api/translate", {
    method: "POST",
    body: JSON.stringify({
      prompt_de: payload.prompt_de,
      model: payload.ollama_model,
    }),
  });
  $("translatedPrompt").value = data.translated_prompt || "";
  setStatus("Übersetzung abgeschlossen.");
}

async function generateImages() {
  const payload = collectPayload();
  if (!payload.prompt_de || !payload.ollama_model) {
    throw new Error("Bitte deutschen Prompt und Ollama-Modell eingeben.");
  }

  setStatus("Generierung läuft … das kann dauern.");
  const data = await api("/api/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  $("translatedPrompt").value = data.translated_prompt || "";
  showImages(data.images || []);
  setStatus(`Fertig. Bilder: ${(data.images || []).length}`);
}

async function init() {
  try {
    await loadOllamaModels();
  } catch (error) {
    setStatus(`Ollama-Modelle konnten nicht geladen werden: ${error.message}`, true);
    fillSelect("ollamaModel", "ollamaModelManualWrap", []);
  }

  try {
    await loadCheckpoints();
  } catch (error) {
    setStatus(`Checkpoints konnten nicht geladen werden: ${error.message}`, true);
    fillSelect("checkpoint", "checkpointManualWrap", []);
  }
}

$("refreshModelsBtn").addEventListener("click", async () => {
  try {
    await loadOllamaModels();
  } catch (error) {
    setStatus(error.message, true);
    fillSelect("ollamaModel", "ollamaModelManualWrap", []);
  }
});

$("refreshCheckpointsBtn").addEventListener("click", async () => {
  try {
    await loadCheckpoints();
  } catch (error) {
    setStatus(error.message, true);
    fillSelect("checkpoint", "checkpointManualWrap", []);
  }
});

$("translateBtn").addEventListener("click", async () => {
  try {
    await translateOnly();
  } catch (error) {
    setStatus(error.message, true);
  }
});

$("generateBtn").addEventListener("click", async () => {
  try {
    await generateImages();
  } catch (error) {
    setStatus(error.message, true);
  }
});

init();
