// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  currentUser: null,   // { username, role }
  ollamaModels: [],
  checkpoints: [],     // all raw models from ComfyUI (including [unet] prefix)
  checkpointNote: "",  // last note from /api/comfy/checkpoints
  modelAliases: {},    // { "technical_name": "alias" }
  samplers: [],
  schedulers: [],
  templates: [],
  lastTranslatedPrompt: "",
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Generic API helper
// ---------------------------------------------------------------------------
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    if (response.status === 401) {
      // Session expired – show login again
      showLogin();
      throw new Error("Sitzung abgelaufen. Bitte erneut anmelden.");
    }
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

// ---------------------------------------------------------------------------
// Auth / login / logout
// ---------------------------------------------------------------------------
function showLogin() {
  $("loginOverlay").classList.remove("hidden");
  $("mainApp").classList.add("hidden");
  $("loginError").classList.add("hidden");
  $("loginError").textContent = "";
}

function showApp(user) {
  state.currentUser = user;
  $("loginOverlay").classList.add("hidden");
  $("mainApp").classList.remove("hidden");
  $("userBadge").textContent = `${user.username} (${user.role})`;
  // Show admin tab only for admins
  if (user.role === "admin") {
    $("tabAdmin").classList.remove("hidden");
  } else {
    $("tabAdmin").classList.add("hidden");
  }
}

async function tryAutoLogin() {
  try {
    const user = await api("/api/auth/me");
    showApp(user);
    await initAppData();
  } catch {
    showLogin();
  }
}

async function doLogin() {
  const username = $("loginUsername").value.trim();
  const password = $("loginPassword").value;
  if (!username || !password) {
    $("loginError").textContent = "Bitte Benutzernamen und Passwort eingeben.";
    $("loginError").classList.remove("hidden");
    return;
  }
  $("loginBtn").disabled = true;
  try {
    const user = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    $("loginPassword").value = "";
    showApp(user);
    await initAppData();
  } catch (err) {
    $("loginError").textContent = err.message;
    $("loginError").classList.remove("hidden");
  } finally {
    $("loginBtn").disabled = false;
  }
}

async function doLogout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch {
    // ignore
  }
  state.currentUser = null;
  showLogin();
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function showTab(name) {
  const tabs = ["Generate", "Admin"];
  for (const t of tabs) {
    $(`panel${t}`).classList.toggle("hidden", t !== name);
    $(`tab${t}`)?.classList.toggle("active", t === name);
  }
  if (name === "Admin") {
    loadAdminTemplates();
    loadAdminModelAliases();
    loadAdminUsers();
  }
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------
function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function setButtons(disabled) {
  $("generateBtn").disabled = disabled;
  $("translateBtn").disabled = disabled;
  $("refinePromptBtn").disabled = disabled;
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

// ---------------------------------------------------------------------------
// Select helpers
// ---------------------------------------------------------------------------
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

function selectValue(selectId, manualInputId, manualWrapId) {
  const wrap = $(manualWrapId);
  if (wrap && !wrap.classList.contains("hidden")) {
    return $(manualInputId).value.trim();
  }
  return $(selectId).value.trim();
}

// ---------------------------------------------------------------------------
// Data loaders
// ---------------------------------------------------------------------------
async function loadOllamaModels() {
  setStatus("Lade Ollama-Modelle …");
  const data = await api("/api/ollama/models");
  state.ollamaModels = data.models || [];
  fillSelect("ollamaModel", "ollamaModelManualWrap", state.ollamaModels);
  const note = $("ollamaModelNote");
  if (note) {
    note.textContent =
      state.ollamaModels.length === 0
        ? "Keine Modelle gefunden – ist Ollama gestartet? (ollama list)"
        : "";
  }
  setStatus(`Ollama-Modelle geladen: ${state.ollamaModels.length}`);
}

async function loadCheckpoints() {
  setStatus("Lade ComfyUI-Modelle …");
  const data = await api("/api/comfy/checkpoints");

  const allModels = data.checkpoints || [];
  state.checkpoints = allModels;
  state.checkpointNote = data.note || "";
  if (data.aliases) {
    state.modelAliases = { ...state.modelAliases, ...data.aliases };
  }

  const unetCount = (data.unet_models || []).length;
  const ckptCount = allModels.length - unetCount;
  setStatus(`Modelle geladen: ${ckptCount} Checkpoints, ${unetCount} UNet`);

  // Rebuild filtered checkpoint select based on the currently selected workflow
  onWorkflowChange();
}

// Return the model_type required by the currently selected workflow template
function getWorkflowModelType() {
  const tplName = $("workflowTemplate").value;
  if (!tplName || tplName === "default") return "checkpoint";
  const tpl = state.templates.find((t) => t.name === tplName);
  return (tpl && tpl.model_type) ? tpl.model_type : "any";
}

// Filter the full checkpoint list by the model_type
function filterCheckpointsByType(models, modelType) {
  if (modelType === "checkpoint") return models.filter((m) => !m.startsWith("[unet] "));
  if (modelType === "unet") return models.filter((m) => m.startsWith("[unet] "));
  return models;
}

// Rebuild the #checkpoint select from a (possibly filtered) list, applying aliases
function rebuildCheckpointSelect(models) {
  const select = $("checkpoint");
  const manualWrap = $("checkpointManualWrap");
  const prevVal = select.value;
  select.innerHTML = "";

  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "– keine passenden Modelle –";
    select.appendChild(opt);
    manualWrap.classList.remove("hidden");
    return;
  }

  manualWrap.classList.add("hidden");
  for (const name of models) {
    const opt = document.createElement("option");
    opt.value = name;
    const technical = name.replace(/^\[unet\] /, "");
    const alias = state.modelAliases[technical] || state.modelAliases[name] || null;
    opt.textContent = alias || technical;
    select.appendChild(opt);
  }
  // Restore previous selection if still in list, otherwise pre-select first
  if (prevVal && models.includes(prevVal)) {
    select.value = prevVal;
  } else if (models.length > 0) {
    select.value = models[0];
  }
  updateCheckpointNote();
}

function updateCheckpointNote() {
  const val = selectValue("checkpoint", "checkpointManual", "checkpointManualWrap");
  const note = $("checkpointNote");
  if (val && val.startsWith("[unet] ")) {
    note.textContent =
      "⚠ UNet-/Diffusion-Modell gewählt: Workflow muss UNETLoader- oder DiffusionModelLoader-Knoten verwenden.";
    note.classList.add("error");
  } else {
    note.textContent = state.checkpointNote;
    note.classList.remove("error");
  }
}

// Called whenever the workflow selection changes
function onWorkflowChange() {
  const modelType = getWorkflowModelType();
  const filtered = filterCheckpointsByType(state.checkpoints, modelType);
  rebuildCheckpointSelect(filtered);
  const note = $("templateNote");
  if (note) {
    const tplName = $("workflowTemplate").value;
    const tpl = state.templates.find((t) => t.name === tplName);
    const mt = (tpl && tpl.model_type) ? tpl.model_type : (tplName === "default" ? "checkpoint" : "any");
    const mtLabel = mt === "checkpoint" ? "Checkpoints" : mt === "unet" ? "UNet/Diffusion" : "alle Modelle";
    note.textContent = `Zeige: ${mtLabel}. ${state.templates.length <= 1 ? "Admins können weitere Templates freigeben." : state.templates.length + " Template(s) verfügbar."}`;
  }
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

async function loadTemplates() {
  try {
    const data = await api("/api/templates");
    state.templates = data.templates || [];
  } catch {
    state.templates = [];
  }
  const select = $("workflowTemplate");
  const currentVal = select.value;
  select.innerHTML = "";

  // Always include built-in default
  const opt0 = document.createElement("option");
  opt0.value = "default";
  opt0.textContent = "Standard (CheckpointLoaderSimple)";
  select.appendChild(opt0);

  for (const tpl of state.templates) {
    if (tpl.name === "default") continue; // already added above
    const opt = document.createElement("option");
    opt.value = tpl.name;
    opt.textContent = tpl.display_name || tpl.name;
    if (tpl.description) opt.title = tpl.description;
    select.appendChild(opt);
  }

  if (currentVal) select.value = currentVal;

  // Update note + re-filter checkpoints for the now-selected workflow
  onWorkflowChange();
}

// ---------------------------------------------------------------------------
// Generate flow
// ---------------------------------------------------------------------------
function collectPayload() {
  const isFollowup = $("followupCheck") && $("followupCheck").checked;
  return {
    prompt_de: $("promptDe").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: selectValue("ollamaModel", "ollamaModelManual", "ollamaModelManualWrap"),
    translated_prompt: $("translatedPrompt").value.trim() || null,
    translated_negative_prompt: $("translatedNegativePrompt").value.trim() || null,
    context_prompt: isFollowup && state.lastTranslatedPrompt ? state.lastTranslatedPrompt : null,
    checkpoint: selectValue("checkpoint", "checkpointManual", "checkpointManualWrap") || null,
    workflow_template: $("workflowTemplate").value || "default",
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

// ---------------------------------------------------------------------------
// Prompt refinement (Änderungs-Feld)
// ---------------------------------------------------------------------------
async function refinePrompt() {
  const base = $("promptDe").value.trim();
  const changes = $("changesPromptDe").value.trim();
  const model = selectValue("ollamaModel", "ollamaModelManual", "ollamaModelManualWrap");
  const statusEl = $("refineStatus");
  statusEl.classList.remove("error");

  if (!base) {
    statusEl.textContent = "Bitte zuerst einen Basis-Prompt eingeben.";
    statusEl.classList.add("error");
    return;
  }
  if (!changes) {
    statusEl.textContent = "Bitte Änderungen eingeben.";
    statusEl.classList.add("error");
    return;
  }
  if (!model) {
    statusEl.textContent = "Bitte zuerst ein Ollama-Modell wählen.";
    statusEl.classList.add("error");
    return;
  }

  $("refinePromptBtn").disabled = true;
  statusEl.textContent = "Verfeinere Prompt …";
  try {
    const data = await api("/api/refine_prompt", {
      method: "POST",
      body: JSON.stringify({ base_prompt_de: base, changes_de: changes, model }),
    });
    $("promptDe").value = data.refined_prompt_de;
    $("changesPromptDe").value = "";
    $("translatedPrompt").value = "";  // clear stale translation
    statusEl.textContent = "✓ Prompt aktualisiert";
    setTimeout(() => { statusEl.textContent = ""; }, 3000);
  } catch (err) {
    statusEl.textContent = `Fehler: ${err.message}`;
    statusEl.classList.add("error");
  } finally {
    $("refinePromptBtn").disabled = false;
  }
}

async function translateOnly() {
  const payload = collectPayload();
  if (!payload.prompt_de || !payload.ollama_model) {
    throw new Error("Bitte deutschen Prompt und Ollama-Modell eingeben.");
  }

  setStatus("Übersetze Prompts …");
  const tasks = [
    api("/api/translate", {
      method: "POST",
      body: JSON.stringify({
        prompt_de: payload.prompt_de,
        model: payload.ollama_model,
        context_prompt: payload.context_prompt,
      }),
    }).then((data) => {
      $("translatedPrompt").value = data.translated_prompt || "";
    }),
  ];
  if (payload.negative_prompt) {
    tasks.push(
      api("/api/translate", {
        method: "POST",
        body: JSON.stringify({
          prompt_de: payload.negative_prompt,
          model: payload.ollama_model,
        }),
      }).then((data) => {
        $("translatedNegativePrompt").value = data.translated_prompt || "";
      })
    );
  }
  await Promise.all(tasks);
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
    const translateTasks = [];
    if (!payload.translated_prompt) {
      translateTasks.push(
        api("/api/translate", {
          method: "POST",
          body: JSON.stringify({
            prompt_de: payload.prompt_de,
            model: payload.ollama_model,
            context_prompt: payload.context_prompt,
          }),
        }).then((data) => {
          payload.translated_prompt = data.translated_prompt || "";
          $("translatedPrompt").value = payload.translated_prompt;
        })
      );
    }
    if (payload.negative_prompt && !payload.translated_negative_prompt) {
      translateTasks.push(
        api("/api/translate", {
          method: "POST",
          body: JSON.stringify({
            prompt_de: payload.negative_prompt,
            model: payload.ollama_model,
          }),
        }).then((data) => {
          payload.translated_negative_prompt = data.translated_prompt || "";
          $("translatedNegativePrompt").value = payload.translated_negative_prompt;
        })
      );
    }
    if (translateTasks.length > 0) {
      setStatus("Übersetze Prompts …");
      await Promise.all(translateTasks);
    }

    setStatus("Sende an ComfyUI …");
    const submitData = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const { prompt_id, client_id, translated_prompt, translated_negative_prompt } = submitData;
    if (translated_prompt) {
      $("translatedPrompt").value = translated_prompt;
    }
    if (translated_negative_prompt) {
      $("translatedNegativePrompt").value = translated_negative_prompt;
    }
    showFollowupSection(translated_prompt || payload.translated_prompt);

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

// ---------------------------------------------------------------------------
// Admin – templates
// ---------------------------------------------------------------------------
async function loadAdminTemplates() {
  const tbody = $("adminTemplateBody");
  tbody.innerHTML = "<tr><td colspan='7' class='hint'>Lade …</td></tr>";
  try {
    const data = await api("/api/admin/templates");
    tbody.innerHTML = "";
    for (const tpl of data.templates) {
      const mt = tpl.model_type || "any";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(tpl.name)}</td>
        <td class="tpl-display-cell">
          <span class="tpl-display-text">${escHtml(tpl.display_name)}</span>
          <button class="btn-sm tpl-edit-name" data-name="${escHtml(tpl.name)}" type="button" title="Anzeigename bearbeiten">&#9998;</button>
          <span class="tpl-edit-wrap hidden" style="display:none">
            <input class="tpl-name-input" value="${escHtml(tpl.display_name)}" style="width:auto;display:inline;margin-right:0.25rem" />
            <button class="btn-sm tpl-save-name" data-name="${escHtml(tpl.name)}" type="button">&#10003;</button>
            <button class="btn-sm btn-secondary tpl-cancel-name" type="button">&#10005;</button>
          </span>
        </td>
        <td>${escHtml(tpl.source || "")}</td>
        <td>
          <select class="tpl-modeltype" data-name="${escHtml(tpl.name)}">
            <option value="checkpoint" ${mt === "checkpoint" ? "selected" : ""}>Checkpoint</option>
            <option value="unet" ${mt === "unet" ? "selected" : ""}>UNet</option>
            <option value="any" ${mt === "any" ? "selected" : ""}>Beliebig</option>
          </select>
        </td>
        <td><input type="checkbox" class="tpl-approved" data-name="${escHtml(tpl.name)}" ${tpl.approved ? "checked" : ""} /></td>
        <td><input type="checkbox" class="tpl-enabled" data-name="${escHtml(tpl.name)}" ${tpl.enabled ? "checked" : ""} /></td>
        <td><button class="btn-sm btn-danger tpl-delete" data-name="${escHtml(tpl.name)}" type="button">L&ouml;schen</button></td>
      `;
      tbody.appendChild(tr);
    }

    // Inline display_name editing
    tbody.querySelectorAll(".tpl-edit-name").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cell = btn.closest(".tpl-display-cell");
        cell.querySelector(".tpl-display-text").style.display = "none";
        btn.style.display = "none";
        const wrap = cell.querySelector(".tpl-edit-wrap");
        wrap.style.display = "";
        wrap.classList.remove("hidden");
        wrap.querySelector(".tpl-name-input").focus();
      });
    });
    tbody.querySelectorAll(".tpl-cancel-name").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cell = btn.closest(".tpl-display-cell");
        cell.querySelector(".tpl-display-text").style.display = "";
        cell.querySelector(".tpl-edit-name").style.display = "";
        cell.querySelector(".tpl-edit-wrap").style.display = "none";
      });
    });
    tbody.querySelectorAll(".tpl-save-name").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const name = btn.dataset.name;
        const cell = btn.closest(".tpl-display-cell");
        const newName = cell.querySelector(".tpl-name-input").value.trim();
        if (!newName) return;
        try {
          await api(`/api/admin/templates/${encodeURIComponent(name)}`, {
            method: "PATCH",
            body: JSON.stringify({ display_name: newName }),
          });
          cell.querySelector(".tpl-display-text").textContent = newName;
          cell.querySelector(".tpl-display-text").style.display = "";
          cell.querySelector(".tpl-edit-name").style.display = "";
          cell.querySelector(".tpl-edit-wrap").style.display = "none";
          await loadTemplates();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });

    // model_type inline select
    tbody.querySelectorAll(".tpl-modeltype").forEach((sel) => {
      sel.addEventListener("change", async () => {
        const name = sel.dataset.name;
        try {
          await api(`/api/admin/templates/${encodeURIComponent(name)}`, {
            method: "PATCH",
            body: JSON.stringify({ model_type: sel.value }),
          });
          // Update state.templates so onWorkflowChange picks up the new type
          const idx = state.templates.findIndex((t) => t.name === name);
          if (idx >= 0) state.templates[idx].model_type = sel.value;
          onWorkflowChange();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });

    // Attach inline-toggle handlers for approved / enabled
    tbody.querySelectorAll(".tpl-approved, .tpl-enabled").forEach((cb) => {
      cb.addEventListener("change", async () => {
        const name = cb.dataset.name;
        const field = cb.classList.contains("tpl-approved") ? "approved" : "enabled";
        try {
          await api(`/api/admin/templates/${encodeURIComponent(name)}`, {
            method: "PATCH",
            body: JSON.stringify({ [field]: cb.checked }),
          });
          await loadTemplates();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
          cb.checked = !cb.checked; // revert
        }
      });
    });
    tbody.querySelectorAll(".tpl-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm(`Template "${btn.dataset.name}" wirklich löschen?`)) return;
        try {
          await api(`/api/admin/templates/${encodeURIComponent(btn.dataset.name)}`, {
            method: "DELETE",
          });
          await loadAdminTemplates();
          await loadTemplates();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='7' class='error'>${escHtml(err.message)}</td></tr>`;
  }
}

async function discoverTemplates() {
  const status = $("adminDiscoverStatus");
  status.classList.remove("error");
  status.textContent = "Suche nach ComfyUI-Templates …";
  try {
    const data = await api("/api/admin/templates/discover", { method: "POST" });
    if (data.error) {
      status.textContent = `Gefunden: ${data.discovered}, Neu hinzugefügt: ${data.added}. Hinweis: ${data.error}`;
      status.classList.add("error");
    } else {
      status.textContent = `Gefunden: ${data.discovered}, Neu hinzugefügt: ${data.added}`;
    }
    await loadAdminTemplates();
    await loadTemplates();
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
    status.classList.add("error");
  }
}

async function discoverLocalTemplates() {
  const status = $("adminDiscoverStatus");
  status.classList.remove("error");
  status.textContent = "Suche lokale Templates in data/templates/ …";
  try {
    const data = await api("/api/admin/templates/discover_local", { method: "POST" });
    if (data.found === 0) {
      status.textContent =
        "Keine lokalen Templates gefunden. Lege Workflow-JSON-Dateien in comfyui_webui/data/templates/ ab und klicke erneut.";
    } else {
      status.textContent = `${data.found} lokales Template(s) geladen: ${data.templates.join(", ")}`;
    }
    await loadAdminTemplates();
    await loadTemplates();
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
    status.classList.add("error");
  }
}

async function uploadTemplate(file) {
  const status = $("adminDiscoverStatus");
  status.classList.remove("error");
  status.textContent = `Lade hoch: ${file.name} …`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("/api/admin/templates/upload", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      if (response.status === 401) { showLogin(); return; }
      let detail = `${response.status} ${response.statusText}`;
      try { const p = await response.json(); detail = p.detail || JSON.stringify(p); } catch { /* ignore */ }
      throw new Error(detail);
    }
    const record = await response.json();
    status.textContent = `Template "${record.display_name}" erfolgreich hochgeladen und registriert.`;
    await loadAdminTemplates();
    await loadTemplates();
  } catch (err) {
    status.textContent = `Fehler beim Hochladen: ${err.message}`;
    status.classList.add("error");
  }
}

async function addTemplate() {
  const name = $("newTplName").value.trim();
  const display_name = $("newTplDisplay").value.trim();
  const source = $("newTplSource").value.trim() || "local";
  const description = $("newTplDesc").value.trim();
  const approved = $("newTplApproved").checked;

  if (!name || !display_name) {
    alert("Bitte Name und Anzeigename angeben.");
    return;
  }
  try {
    await api("/api/admin/templates", {
      method: "POST",
      body: JSON.stringify({ name, display_name, source, description, approved, enabled: true }),
    });
    $("addTemplateForm").classList.add("hidden");
    await loadAdminTemplates();
    await loadTemplates();
  } catch (err) {
    alert(`Fehler: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Admin – model aliases
// ---------------------------------------------------------------------------
async function loadAdminModelAliases() {
  const tbody = $("modelAliasBody");
  tbody.innerHTML = "<tr><td colspan='3' class='hint'>Lade …</td></tr>";
  try {
    const data = await api("/api/admin/model_aliases");
    state.modelAliases = data.aliases || {};
    tbody.innerHTML = "";
    const entries = Object.entries(data.aliases || {});
    if (entries.length === 0) {
      tbody.innerHTML = "<tr><td colspan='3' class='hint'>Noch keine Aliase definiert.</td></tr>";
      return;
    }
    for (const [techName, alias] of entries) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(techName)}</td>
        <td>${escHtml(alias)}</td>
        <td>
          <button class="btn-sm" data-tech="${escHtml(techName)}" data-alias="${escHtml(alias)}" type="button" onclick="editAliasInForm(this)">&#9998;</button>
          <button class="btn-sm btn-danger alias-delete" data-tech="${escHtml(techName)}" type="button">&#10005;</button>
        </td>
      `;
      tbody.appendChild(tr);
    }
    tbody.querySelectorAll(".alias-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const tech = btn.dataset.tech;
        if (!confirm(`Alias für „${tech}" löschen?`)) return;
        try {
          await api(`/api/admin/model_aliases/${encodeURIComponent(tech)}`, { method: "DELETE" });
          await loadAdminModelAliases();
          await loadCheckpoints(); // refresh display names in model select
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='3' class='error'>${escHtml(err.message)}</td></tr>`;
  }
}

function editAliasInForm(btn) {
  $("aliasNameInput").value = btn.dataset.tech;
  $("aliasDisplayInput").value = btn.dataset.alias;
  $("aliasNameInput").focus();
}

async function saveModelAlias() {
  const techName = $("aliasNameInput").value.trim();
  const alias = $("aliasDisplayInput").value.trim();
  const statusEl = $("modelAliasStatus");
  statusEl.classList.remove("error");

  if (!techName || !alias) {
    statusEl.textContent = "Modellname und Alias sind Pflichtfelder.";
    statusEl.classList.add("error");
    return;
  }

  try {
    await api("/api/admin/model_aliases", {
      method: "PUT",
      body: JSON.stringify({ technical_name: techName, alias }),
    });
    $("aliasNameInput").value = "";
    $("aliasDisplayInput").value = "";
    statusEl.textContent = "✓ Alias gespeichert";
    setTimeout(() => { statusEl.textContent = ""; }, 3000);
    await loadAdminModelAliases();
    await loadCheckpoints(); // refresh display names in model select
  } catch (err) {
    statusEl.textContent = `Fehler: ${err.message}`;
    statusEl.classList.add("error");
  }
}

// ---------------------------------------------------------------------------
// Admin – users
// ---------------------------------------------------------------------------
async function loadAdminUsers() {
  const tbody = $("adminUserBody");
  tbody.innerHTML = "<tr><td colspan='5' class='hint'>Lade …</td></tr>";
  try {
    const data = await api("/api/admin/users");
    tbody.innerHTML = "";
    for (const user of data.users) {
      const tr = document.createElement("tr");
      const isSelf = state.currentUser && user.username === state.currentUser.username;
      tr.innerHTML = `
        <td>${escHtml(user.username)}</td>
        <td>${escHtml(user.role)}</td>
        <td>${user.disabled ? "Ja" : "Nein"}</td>
        <td>${escHtml((user.created_at || "").slice(0, 10))}</td>
        <td>
          ${!isSelf
            ? `<button class="btn-sm ${user.disabled ? "" : "btn-danger"} user-toggle" data-name="${escHtml(user.username)}" data-disabled="${user.disabled}" type="button">${user.disabled ? "Aktivieren" : "Deaktivieren"}</button>`
            : "(eigenes Konto)"
          }
        </td>
      `;
      tbody.appendChild(tr);
    }
    tbody.querySelectorAll(".user-toggle").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const newDisabled = btn.dataset.disabled !== "true";
        try {
          await api(`/api/admin/users/${encodeURIComponent(btn.dataset.name)}`, {
            method: "PATCH",
            body: JSON.stringify({ disabled: newDisabled }),
          });
          await loadAdminUsers();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='5' class='error'>${escHtml(err.message)}</td></tr>`;
  }
}

async function addUser() {
  const username = $("newUserName").value.trim();
  const password = $("newUserPass").value;
  const role = $("newUserRole").value;
  if (!username || !password) {
    alert("Bitte Benutzername und Passwort angeben.");
    return;
  }
  try {
    await api("/api/admin/users", {
      method: "POST",
      body: JSON.stringify({ username, password, role }),
    });
    $("newUserName").value = "";
    $("newUserPass").value = "";
    $("addUserForm").classList.add("hidden");
    await loadAdminUsers();
  } catch (err) {
    alert(`Fehler: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Change password
// ---------------------------------------------------------------------------
function openChangePw() {
  $("cpCurrentPw").value = "";
  $("cpNewPw").value = "";
  $("cpNewPw2").value = "";
  $("cpError").classList.add("hidden");
  $("cpError").textContent = "";
  $("cpSuccess").classList.add("hidden");
  $("cpSuccess").textContent = "";
  $("changePwOverlay").classList.remove("hidden");
}

function closeChangePw() {
  $("changePwOverlay").classList.add("hidden");
}

async function submitChangePw() {
  const current = $("cpCurrentPw").value;
  const newPw = $("cpNewPw").value;
  const newPw2 = $("cpNewPw2").value;
  $("cpError").classList.add("hidden");
  $("cpSuccess").classList.add("hidden");

  if (!current || !newPw || !newPw2) {
    $("cpError").textContent = "Bitte alle Felder ausfüllen.";
    $("cpError").classList.remove("hidden");
    return;
  }
  if (newPw !== newPw2) {
    $("cpError").textContent = "Die neuen Passwörter stimmen nicht überein.";
    $("cpError").classList.remove("hidden");
    return;
  }
  if (newPw.length < 8) {
    $("cpError").textContent = "Das neue Passwort muss mindestens 8 Zeichen lang sein.";
    $("cpError").classList.remove("hidden");
    return;
  }

  $("cpSubmitBtn").disabled = true;
  try {
    await api("/api/auth/change_password", {
      method: "POST",
      body: JSON.stringify({ current_password: current, new_password: newPw }),
    });
    $("cpSuccess").textContent = "Passwort erfolgreich geändert.";
    $("cpSuccess").classList.remove("hidden");
    $("cpCurrentPw").value = "";
    $("cpNewPw").value = "";
    $("cpNewPw2").value = "";
    setTimeout(closeChangePw, 1500);
  } catch (err) {
    $("cpError").textContent = err.message;
    $("cpError").classList.remove("hidden");
  } finally {
    $("cpSubmitBtn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Init – after login, load all dynamic data
// ---------------------------------------------------------------------------
async function initAppData() {
  const errors = [];

  await loadTemplates().catch((e) => errors.push(`Templates: ${e.message}`));

  try {
    await loadOllamaModels();
  } catch (error) {
    errors.push(`Ollama: ${error.message}`);
    fillSelect("ollamaModel", "ollamaModelManualWrap", []);
    const note = $("ollamaModelNote");
    if (note) note.textContent = `Ollama nicht erreichbar: ${error.message}`;
  }

  try {
    await loadCheckpoints();
  } catch (error) {
    errors.push(`Checkpoints: ${error.message}`);
    fillSelect("checkpoint", "checkpointManualWrap", []);
    $("checkpointNote").textContent = `ComfyUI nicht erreichbar: ${error.message}`;
  }

  await loadSamplers();

  if (errors.length > 0) {
    setStatus(errors.join(" | "), true);
  }
}

// ---------------------------------------------------------------------------
// Event listeners – login
// ---------------------------------------------------------------------------
$("loginBtn").addEventListener("click", doLogin);
$("loginPassword").addEventListener("keydown", (e) => {
  if (e.key === "Enter") doLogin();
});

// ---------------------------------------------------------------------------
// Event listeners – app shell
// ---------------------------------------------------------------------------
$("logoutBtn").addEventListener("click", doLogout);
$("changePwBtn").addEventListener("click", openChangePw);
$("cpSubmitBtn").addEventListener("click", submitChangePw);
$("cpCancelBtn").addEventListener("click", closeChangePw);
$("cpNewPw2").addEventListener("keydown", (e) => { if (e.key === "Enter") submitChangePw(); });
$("tabGenerate").addEventListener("click", () => showTab("Generate"));
$("tabAdmin").addEventListener("click", () => showTab("Admin"));

// ---------------------------------------------------------------------------
// Event listeners – generate tab
// ---------------------------------------------------------------------------
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

$("refreshTemplatesBtn").addEventListener("click", async () => {
  try {
    await loadTemplates();
    setStatus("Templates aktualisiert.");
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

$("refinePromptBtn").addEventListener("click", refinePrompt);
$("workflowTemplate").addEventListener("change", onWorkflowChange);
$("checkpoint").addEventListener("change", updateCheckpointNote);

$("generateBtn").addEventListener("click", async () => {
  try {
    await generateImages();
  } catch (error) {
    setStatus(error.message, true);
  }
});

$("followupCheck").addEventListener("change", () => {
  if ($("followupCheck").checked) {
    $("promptDe").placeholder =
      "Änderungsanweisung eingeben, z. B. \u201EMache die Sonne etwas dunkler\u201C";
    $("translatedPrompt").value = "";
  } else {
    $("promptDe").placeholder =
      "z. B. Ein futuristisches Stadtbild bei Sonnenuntergang";
  }
});

// ---------------------------------------------------------------------------
// Event listeners – admin tab
// ---------------------------------------------------------------------------
$("adminDiscoverBtn").addEventListener("click", discoverTemplates);
$("adminDiscoverLocalBtn").addEventListener("click", discoverLocalTemplates);

$("adminUploadTemplateBtn").addEventListener("click", () => {
  $("adminUploadTemplateInput").value = "";
  $("adminUploadTemplateInput").click();
});
$("adminUploadTemplateInput").addEventListener("change", () => {
  const file = $("adminUploadTemplateInput").files[0];
  if (file) uploadTemplate(file);
});

$("adminAddTemplateBtn").addEventListener("click", () => {
  $("addTemplateForm").classList.toggle("hidden");
});
$("addTemplateSubmitBtn").addEventListener("click", addTemplate);
$("addTemplateCancelBtn").addEventListener("click", () => {
  $("addTemplateForm").classList.add("hidden");
});

$("adminAddUserBtn").addEventListener("click", () => {
  $("addUserForm").classList.toggle("hidden");
});
$("addUserSubmitBtn").addEventListener("click", addUser);
$("addUserCancelBtn").addEventListener("click", () => {
  $("addUserForm").classList.add("hidden");
});
$("aliasSaveBtn").addEventListener("click", saveModelAlias);

// ---------------------------------------------------------------------------
// Bootstrap: check session, show login or app
// ---------------------------------------------------------------------------
tryAutoLogin();
