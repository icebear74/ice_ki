const state = {
  ollamaModels: [],
  checkpoints: [],
  samplers: [],
  schedulers: [],
  lastTranslatedPrompt: "",
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

function setButtons(disabled) {
  $("generateBtn").disabled = disabled;
  $("translateBtn").disabled = disabled;
}

function showProgress(visible) {
  $("progressWrap").classList.toggle("hidden", !visible);
  if (!visible) {
    $("progressBar").style.width = "0%";
    $("progressLabel").textContent = "";
  }
}

function setProgressBar(step, total, eta) {
  const pct = total > 0 ? Math.round((step / total) * 100) : 0;
  $("progressBar").style.width = `${pct}%`;
  const etaStr = eta != null ? ` · ETA ${eta}s` : "";
  $("progressLabel").textContent =
    total > 0 ? `${step} / ${total} Schritte (${pct}%)${etaStr}` : "";
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

function fillSelectSimple(selectId, values, defaultValue) {
  const select = $(selectId);
  const currentVal = select.value || defaultValue;
  select.innerHTML = "";
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  }
  if (currentVal && values.includes(currentVal)) {
    select.value = currentVal;
  } else if (defaultValue && values.includes(defaultValue)) {
    select.value = defaultValue;
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

async function loadSamplers() {
  try {
    const data = await api("/api/comfy/samplers");
    state.samplers = data.samplers || [];
    state.schedulers = data.schedulers || [];
  } catch {
    state.samplers = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "lcm"];
    state.schedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"];
  }
  fillSelectSimple("sampler", state.samplers, "euler");
  fillSelectSimple("scheduler", state.schedulers, "normal");
}

function collectPayload() {
  const isFollowup = $("followupCheck") && $("followupCheck").checked;
  return {
    prompt_de: $("promptDe").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: selectValue("ollamaModel", "ollamaModelManual", "ollamaModelManualWrap"),
    translated_prompt: $("translatedPrompt").value.trim() || null,
    context_prompt: isFollowup && state.lastTranslatedPrompt ? state.lastTranslatedPrompt : null,
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

function showFollowupSection(translatedPrompt) {
  state.lastTranslatedPrompt = translatedPrompt;
  $("followupSection").classList.remove("hidden");
  const hint = $("followupHint");
  const preview =
    translatedPrompt.length > 80
      ? translatedPrompt.slice(0, 80) + "…"
      : translatedPrompt;
  hint.textContent = `Letzter Prompt: „${preview}"`;
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
      context_prompt: payload.context_prompt,
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

  setButtons(true);
  showProgress(false);

  try {
    // Step 1: Translate if no translated prompt present yet
    if (!payload.translated_prompt) {
      setStatus("Übersetze Prompt …");
      const trans = await api("/api/translate", {
        method: "POST",
        body: JSON.stringify({
          prompt_de: payload.prompt_de,
          model: payload.ollama_model,
          context_prompt: payload.context_prompt,
        }),
      });
      payload.translated_prompt = trans.translated_prompt || "";
      $("translatedPrompt").value = payload.translated_prompt;
    }

    // Step 2: Submit job to ComfyUI
    setStatus("Sende an ComfyUI …");
    const submitData = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const { prompt_id, client_id, translated_prompt } = submitData;
    if (translated_prompt) {
      $("translatedPrompt").value = translated_prompt;
    }
    showFollowupSection(translated_prompt || payload.translated_prompt);

    // Step 3: Stream progress via SSE
    showProgress(true);
    setProgressBar(0, 0, null);

    await new Promise((resolve, reject) => {
      const url = `/api/comfy/progress/${encodeURIComponent(prompt_id)}?client_id=${encodeURIComponent(client_id)}`;
      const evtSource = new EventSource(url);

      evtSource.onmessage = (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        if (data.type === "queued") {
          const pos = data.position ? ` (Position ${data.position})` : "";
          setStatus(`In Warteschlange${pos} …`);
        } else if (data.type === "start") {
          setStatus("Generiere …");
        } else if (data.type === "progress") {
          const { step, max, eta } = data;
          const etaStr = eta != null ? ` · ETA ${eta}s` : "";
          setStatus(`Generiere … Schritt ${step}/${max}${etaStr}`);
          setProgressBar(step, max, eta);
        } else if (data.type === "done") {
          evtSource.close();
          showProgress(false);
          showImages(data.images || []);
          setStatus(`Fertig. Bilder: ${(data.images || []).length}`);
          setButtons(false);
          resolve();
        } else if (data.type === "error") {
          evtSource.close();
          showProgress(false);
          setButtons(false);
          reject(new Error(data.message));
        }
      };

      evtSource.onerror = () => {
        evtSource.close();
        showProgress(false);
        setButtons(false);
        reject(new Error("Verbindung zum Fortschritt-Stream unterbrochen."));
      };
    });
  } catch (err) {
    setButtons(false);
    showProgress(false);
    throw err;
  }
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

  await loadSamplers();
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

$("followupCheck").addEventListener("change", () => {
  const hint = $("followupHint");
  if ($("followupCheck").checked) {
    $("promptDe").placeholder =
      "Änderungsanweisung eingeben, z. B. „Mache die Sonne etwas dunkler"";
    // Clear translated prompt so it gets regenerated with context
    $("translatedPrompt").value = "";
  } else {
    $("promptDe").placeholder =
      "z. B. Ein futuristisches Stadtbild bei Sonnenuntergang";
  }
});

init();
