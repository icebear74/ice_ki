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

function fillDatalist(id, values) {
  const datalist = $(id);
  datalist.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    datalist.appendChild(option);
  }
}

async function loadOllamaModels() {
  setStatus("Lade Ollama-Modelle …");
  const data = await api("/api/ollama/models");
  state.ollamaModels = data.models || [];
  fillDatalist("ollamaModels", state.ollamaModels);
  if (!$("ollamaModel").value && state.ollamaModels.length) {
    $("ollamaModel").value = state.ollamaModels[0];
  }
  setStatus(`Ollama-Modelle geladen: ${state.ollamaModels.length}`);
}

async function loadCheckpoints() {
  setStatus("Lade ComfyUI-Checkpoints …");
  const data = await api("/api/comfy/checkpoints");
  state.checkpoints = data.checkpoints || [];
  fillDatalist("checkpoints", state.checkpoints);
  $("checkpointNote").textContent = data.note || "";
  if (!$("checkpoint").value && state.checkpoints.length) {
    $("checkpoint").value = state.checkpoints[0];
  }
  setStatus(`Checkpoints geladen: ${state.checkpoints.length}`);
}

function collectPayload() {
  return {
    prompt_de: $("promptDe").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: $("ollamaModel").value.trim(),
    translated_prompt: $("translatedPrompt").value.trim() || null,
    checkpoint: $("checkpoint").value.trim() || null,
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
  }

  try {
    await loadCheckpoints();
  } catch (error) {
    setStatus(`Checkpoints konnten nicht geladen werden: ${error.message}`, true);
  }
}

$("refreshModelsBtn").addEventListener("click", async () => {
  try {
    await loadOllamaModels();
  } catch (error) {
    setStatus(error.message, true);
  }
});

$("refreshCheckpointsBtn").addEventListener("click", async () => {
  try {
    await loadCheckpoints();
  } catch (error) {
    setStatus(error.message, true);
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
