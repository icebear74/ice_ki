// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  currentUser: null,   // { username, role, can_advanced }
  ollamaModels: [],
  checkpoints: [],     // all raw models from ComfyUI (including [unet] prefix)
  samplers: [],
  schedulers: [],
  templates: [],
  mappings: [],        // loaded from /api/mappings
  editingMappingName: null,  // null = create, string = editing existing
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
  // Admin tab only for admins
  if (user.role === "admin") {
    $("tabAdmin").classList.remove("hidden");
  } else {
    $("tabAdmin").classList.add("hidden");
  }
  // Erweitert tab only for users with can_advanced (admins always have it)
  const canAdv = user.can_advanced || user.role === "admin";
  if (canAdv) {
    $("tabAdvanced").classList.remove("hidden");
  } else {
    $("tabAdvanced").classList.add("hidden");
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
  const tabs = ["Generate", "Advanced", "Admin"];
  for (const t of tabs) {
    const panel = $(`panel${t}`);
    if (panel) panel.classList.toggle("hidden", t !== name);
    $(`tab${t}`)?.classList.toggle("active", t === name);
  }
  if (name === "Admin") {
    loadAdminMappings();
    loadAdminTemplates();
    loadAdminModelAliases();
    loadAdminUsers();
    populateMappingFormSelects();
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
function fillSelectSimple(selectId, values, defaultValue) {
  const select = $(selectId);
  if (!select) return;
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

// ---------------------------------------------------------------------------
// Data loaders
// ---------------------------------------------------------------------------
async function loadOllamaModels() {
  try {
    const data = await api("/api/ollama/models");
    state.ollamaModels = data.models || [];
  } catch {
    state.ollamaModels = [];
  }
}

async function loadCheckpoints() {
  try {
    const data = await api("/api/comfy/checkpoints");
    state.checkpoints = data.checkpoints || [];
    if (data.aliases) {
      // store model aliases for display
      state.modelAliases = { ...data.aliases };
    }
  } catch {
    state.checkpoints = [];
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
}

async function loadMappings() {
  try {
    const data = await api("/api/mappings");
    state.mappings = data.mappings || [];
  } catch {
    state.mappings = [];
  }
  const select = $("mappingSelect");
  select.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = "– Bitte auswählen –";
  select.appendChild(opt0);
  for (const m of state.mappings) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.display_name || m.name;
    select.appendChild(opt);
  }
  const note = $("mappingNote");
  if (state.mappings.length === 0) {
    note.textContent = "Keine Mappings verfügbar. Ein Admin muss zuerst Mappings anlegen.";
  } else {
    note.textContent = `${state.mappings.length} Mapping(s) verfügbar. Wähle eines, um die Vorgaben zu laden.`;
  }
}

// Called when the user changes the mapping selector
function onMappingChange() {
  const name = $("mappingSelect").value;
  const mapping = state.mappings.find((m) => m.name === name);
  if (!mapping) return;
  // Populate Erweitert fields from mapping defaults
  $("steps").value = mapping.steps ?? 30;
  $("cfg").value = mapping.cfg ?? 7;
  $("seed").value = mapping.seed ?? -1;
  $("imageCount").value = mapping.image_count ?? 1;
  $("width").value = mapping.width ?? 1024;
  $("height").value = mapping.height ?? 1024;
  if (state.samplers.includes(mapping.sampler)) {
    $("sampler").value = mapping.sampler;
  }
  if (state.schedulers.includes(mapping.scheduler)) {
    $("scheduler").value = mapping.scheduler;
  }
  const note = $("mappingNote");
  note.textContent = `Mapping geladen: ${mapping.display_name}`;
}

// ---------------------------------------------------------------------------
// Generate flow
// ---------------------------------------------------------------------------
function getActiveMapping() {
  const name = $("mappingSelect").value;
  return state.mappings.find((m) => m.name === name) || null;
}

function collectPayload() {
  const mapping = getActiveMapping();
  const canAdv = state.currentUser && (state.currentUser.can_advanced || state.currentUser.role === "admin");

  // Generation parameters: use Erweitert overrides if user has permission,
  // otherwise fall back to mapping values
  const steps    = canAdv ? Number($("steps").value)     : (mapping ? (mapping.steps ?? 30)        : 30);
  const cfg      = canAdv ? Number($("cfg").value)       : (mapping ? (mapping.cfg ?? 7)           : 7);
  const seed     = canAdv ? Number($("seed").value)      : (mapping ? (mapping.seed ?? -1)         : -1);
  const width    = canAdv ? Number($("width").value)     : (mapping ? (mapping.width ?? 1024)      : 1024);
  const height   = canAdv ? Number($("height").value)    : (mapping ? (mapping.height ?? 1024)     : 1024);
  const imgCount = canAdv ? Number($("imageCount").value): (mapping ? (mapping.image_count ?? 1)   : 1);
  const sampler  = canAdv ? $("sampler").value.trim()    : (mapping ? (mapping.sampler || "euler") : "euler");
  const sched    = canAdv ? $("scheduler").value.trim()  : (mapping ? (mapping.scheduler || "normal") : "normal");

  return {
    prompt_de: $("promptDe").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: (mapping && mapping.ollama_model) || "",
    translated_prompt: $("translatedPrompt").value.trim() || null,
    translated_negative_prompt: $("translatedNegativePrompt").value.trim() || null,
    context_prompt: null,
    checkpoint: (mapping && mapping.checkpoint) || null,
    workflow_template: (mapping && mapping.template_name) || "default",
    steps,
    cfg,
    seed,
    width,
    height,
    sampler,
    scheduler: sched,
    image_count: imgCount,
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

// Auto-refine prompt if changes are present; resolves to (possibly updated) prompt_de
async function autoRefineIfNeeded(promptDe, changesDe, ollamaModel) {
  if (!changesDe || !changesDe.trim()) return promptDe;
  if (!ollamaModel) return promptDe;
  try {
    const data = await api("/api/refine_prompt", {
      method: "POST",
      body: JSON.stringify({ base_prompt_de: promptDe, changes_de: changesDe.trim(), model: ollamaModel }),
    });
    $("promptDe").value = data.refined_prompt_de;
    $("changesPromptDe").value = "";
    $("translatedPrompt").value = "";
    return data.refined_prompt_de;
  } catch {
    // If refinement fails, proceed with original prompt
    return promptDe;
  }
}

async function translateOnly() {
  const mapping = getActiveMapping();
  if (!mapping) throw new Error("Bitte zuerst ein Mapping auswählen.");

  let promptDe = $("promptDe").value.trim();
  const changesDe = $("changesPromptDe").value.trim();
  const ollamaModel = mapping.ollama_model || "";

  if (!promptDe) throw new Error("Bitte einen Prompt eingeben.");
  if (!ollamaModel) throw new Error("Das gewählte Mapping hat kein Ollama-Modell konfiguriert.");

  setStatus("Übersetze Prompts …");

  if (changesDe) {
    setStatus("Verfeinere Prompt …");
    promptDe = await autoRefineIfNeeded(promptDe, changesDe, ollamaModel);
  }

  const tasks = [
    api("/api/translate", {
      method: "POST",
      body: JSON.stringify({ prompt_de: promptDe, model: ollamaModel }),
    }).then((data) => {
      $("translatedPrompt").value = data.translated_prompt || "";
    }),
  ];
  const negPrompt = $("negativePrompt").value.trim();
  if (negPrompt) {
    tasks.push(
      api("/api/translate", {
        method: "POST",
        body: JSON.stringify({ prompt_de: negPrompt, model: ollamaModel }),
      }).then((data) => {
        $("translatedNegativePrompt").value = data.translated_prompt || "";
      })
    );
  }
  await Promise.all(tasks);
  setStatus("Übersetzung abgeschlossen.");
}

async function generateImages() {
  const mapping = getActiveMapping();
  if (!mapping) throw new Error("Bitte zuerst ein Mapping auswählen.");

  let promptDe = $("promptDe").value.trim();
  const changesDe = $("changesPromptDe").value.trim();
  const ollamaModel = mapping.ollama_model || "";

  if (!promptDe) throw new Error("Bitte einen Prompt eingeben.");
  if (!ollamaModel) throw new Error("Das gewählte Mapping hat kein Ollama-Modell konfiguriert.");

  setButtons(true);
  showProgress(false);

  try {
    if (changesDe) {
      setStatus("Verfeinere Prompt …");
      promptDe = await autoRefineIfNeeded(promptDe, changesDe, ollamaModel);
    }

    const payload = collectPayload();
    payload.prompt_de = promptDe;
    // Always re-translate so stale translations are never sent
    payload.translated_prompt = null;
    payload.translated_negative_prompt = null;

    const translateTasks = [];
    if (!payload.translated_prompt) {
      translateTasks.push(
        api("/api/translate", {
          method: "POST",
          body: JSON.stringify({
            prompt_de: payload.prompt_de,
            model: ollamaModel,
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
          body: JSON.stringify({ prompt_de: payload.negative_prompt, model: ollamaModel }),
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
    if (translated_prompt) $("translatedPrompt").value = translated_prompt;
    if (translated_negative_prompt) $("translatedNegativePrompt").value = translated_negative_prompt;

    showProgress(true);
    setProgressBar(0, 0, null);

    await new Promise((resolve, reject) => {
      const url = `/api/comfy/progress/${encodeURIComponent(prompt_id)}?client_id=${encodeURIComponent(client_id)}`;
      const evtSource = new EventSource(url);

      evtSource.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

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
// Admin – mappings
// ---------------------------------------------------------------------------
async function loadAdminMappings() {
  const tbody = $("adminMappingBody");
  tbody.innerHTML = "<tr><td colspan='7' class='hint'>Lade …</td></tr>";
  try {
    const data = await api("/api/admin/mappings");
    tbody.innerHTML = "";
    if (data.mappings.length === 0) {
      tbody.innerHTML = "<tr><td colspan='7' class='hint'>Noch keine Mappings angelegt.</td></tr>";
      return;
    }
    for (const m of data.mappings) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(m.name)}</td>
        <td>${escHtml(m.display_name)}</td>
        <td>${escHtml(m.template_name || "default")}</td>
        <td><small>${escHtml(m.checkpoint || "–")}</small></td>
        <td><small>${escHtml(m.ollama_model || "–")}</small></td>
        <td><input type="checkbox" class="map-enabled" data-name="${escHtml(m.name)}" ${m.enabled ? "checked" : ""} /></td>
        <td>
          <button class="btn-sm map-edit" data-name="${escHtml(m.name)}" type="button">&#9998; Bearbeiten</button>
          <button class="btn-sm btn-danger map-delete" data-name="${escHtml(m.name)}" type="button">L&ouml;schen</button>
        </td>
      `;
      tbody.appendChild(tr);
    }
    tbody.querySelectorAll(".map-enabled").forEach((cb) => {
      cb.addEventListener("change", async () => {
        try {
          await api(`/api/admin/mappings/${encodeURIComponent(cb.dataset.name)}`, {
            method: "PATCH",
            body: JSON.stringify({ enabled: cb.checked }),
          });
          await loadMappings();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
          cb.checked = !cb.checked;
        }
      });
    });
    tbody.querySelectorAll(".map-edit").forEach((btn) => {
      btn.addEventListener("click", () => openMappingForm(btn.dataset.name));
    });
    tbody.querySelectorAll(".map-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm(`Mapping „${btn.dataset.name}" wirklich löschen?`)) return;
        try {
          await api(`/api/admin/mappings/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE" });
          await loadAdminMappings();
          await loadMappings();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='7' class='error'>${escHtml(err.message)}</td></tr>`;
  }
}

function _getTemplateModelType(templateName) {
  if (templateName === "default") return "checkpoint";
  const tpl = state.templates.find((t) => t.name === templateName);
  return (tpl && tpl.model_type) ? tpl.model_type : "any";
}

function populateCheckpointSelect(currentValue) {
  const sel = $("newMapCheckpoint");
  if (!sel || sel.tagName !== "SELECT") return;
  const prevValue = currentValue !== undefined ? currentValue : sel.value;
  const templateName = $("newMapTemplate") ? $("newMapTemplate").value : "default";
  const modelType = _getTemplateModelType(templateName);

  sel.innerHTML = "";
  const emptyOpt = document.createElement("option");
  emptyOpt.value = "";
  emptyOpt.textContent = "– Kein Modell –";
  sel.appendChild(emptyOpt);

  const models = (state.checkpoints || []).filter((m) => {
    if (modelType === "checkpoint") return !m.startsWith("[unet]");
    if (modelType === "unet") return m.startsWith("[unet]");
    return true;
  });

  let found = false;
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    const alias = state.modelAliases && state.modelAliases[m];
    opt.textContent = alias ? `${alias} (${m})` : m;
    sel.appendChild(opt);
    if (m === prevValue) found = true;
  }

  if (prevValue && !found) {
    const opt = document.createElement("option");
    opt.value = prevValue;
    opt.textContent = `${prevValue} (nicht verfügbar)`;
    sel.insertBefore(opt, sel.children[1] || null);
  }
  sel.value = prevValue || "";
}

function populateOllamaModelSelect(currentValue) {
  const sel = $("newMapOllamaModel");
  if (!sel || sel.tagName !== "SELECT") return;
  const prevValue = currentValue !== undefined ? currentValue : sel.value;

  sel.innerHTML = "";
  const emptyOpt = document.createElement("option");
  emptyOpt.value = "";
  emptyOpt.textContent = "– Kein Ollama-Modell –";
  sel.appendChild(emptyOpt);

  let found = false;
  for (const m of (state.ollamaModels || [])) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
    if (m === prevValue) found = true;
  }

  if (prevValue && !found) {
    const opt = document.createElement("option");
    opt.value = prevValue;
    opt.textContent = `${prevValue} (nicht verfügbar)`;
    sel.insertBefore(opt, sel.children[1] || null);
  }
  sel.value = prevValue || "";
}

function populateMappingFormSelects() {
  // Populate template dropdown
  const tplSel = $("newMapTemplate");
  if (!tplSel) return;
  const prevTpl = tplSel.value;
  tplSel.innerHTML = "";
  const defOpt = document.createElement("option");
  defOpt.value = "default";
  defOpt.textContent = "Standard (CheckpointLoaderSimple)";
  tplSel.appendChild(defOpt);
  for (const tpl of state.templates) {
    if (tpl.name === "default") continue;
    const opt = document.createElement("option");
    opt.value = tpl.name;
    opt.textContent = tpl.display_name || tpl.name;
    tplSel.appendChild(opt);
  }
  if (prevTpl) tplSel.value = prevTpl;

  // Re-filter checkpoint dropdown when template changes
  tplSel.onchange = () => populateCheckpointSelect();

  // Populate checkpoint and ollama model dropdowns
  populateCheckpointSelect();
  populateOllamaModelSelect();

  // Populate sampler/scheduler dropdowns in form
  fillSelectSimple("newMapSampler", state.samplers.length ? state.samplers : ["euler", "dpmpp_2m", "ddim"], "euler");
  fillSelectSimple("newMapScheduler", state.schedulers.length ? state.schedulers : ["normal", "karras", "simple"], "normal");
}

function openMappingForm(editName) {
  const form = $("addMappingForm");
  form.classList.remove("hidden");
  populateMappingFormSelects();

  if (editName) {
    // Find the mapping from admin data (need to refetch)
    state.editingMappingName = editName;
    $("mappingFormTitle").textContent = `Mapping bearbeiten: ${editName}`;
    $("newMapName").disabled = true;
    api(`/api/admin/mappings`).then((data) => {
      const m = (data.mappings || []).find((x) => x.name === editName);
      if (!m) return;
      $("newMapName").value = m.name;
      $("newMapDisplay").value = m.display_name;
      $("newMapTemplate").value = m.template_name || "default";
      populateCheckpointSelect(m.checkpoint || "");
      populateOllamaModelSelect(m.ollama_model || "");
      $("newMapSteps").value = m.steps ?? 30;
      $("newMapCfg").value = m.cfg ?? 7;
      $("newMapSeed").value = m.seed ?? -1;
      $("newMapWidth").value = m.width ?? 1024;
      $("newMapHeight").value = m.height ?? 1024;
      $("newMapImageCount").value = m.image_count ?? 1;
      if (state.samplers.includes(m.sampler)) $("newMapSampler").value = m.sampler;
      if (state.schedulers.includes(m.scheduler)) $("newMapScheduler").value = m.scheduler;
      $("newMapEnabled").checked = m.enabled !== false;
    }).catch(() => {});
  } else {
    state.editingMappingName = null;
    $("mappingFormTitle").textContent = "Neues Mapping";
    $("newMapName").disabled = false;
    $("newMapName").value = "";
    $("newMapDisplay").value = "";
    $("newMapTemplate").value = "default";
    populateCheckpointSelect("");
    populateOllamaModelSelect("");
    $("newMapSteps").value = 30;
    $("newMapCfg").value = 7;
    $("newMapSeed").value = -1;
    $("newMapWidth").value = 1024;
    $("newMapHeight").value = 1024;
    $("newMapImageCount").value = 1;
    $("newMapEnabled").checked = true;
  }
}

async function saveMapping() {
  const display = $("newMapDisplay").value.trim();
  const name = $("newMapName").value.trim();
  if (!display || !name) {
    alert("Bitte Name und Anzeigename angeben.");
    return;
  }
  const body = {
    name,
    display_name: display,
    template_name: $("newMapTemplate").value || "default",
    checkpoint: $("newMapCheckpoint").value.trim(),
    ollama_model: $("newMapOllamaModel").value.trim(),
    steps: Number($("newMapSteps").value),
    cfg: Number($("newMapCfg").value),
    seed: Number($("newMapSeed").value),
    width: Number($("newMapWidth").value),
    height: Number($("newMapHeight").value),
    sampler: $("newMapSampler").value,
    scheduler: $("newMapScheduler").value,
    image_count: Number($("newMapImageCount").value),
    enabled: $("newMapEnabled").checked,
  };
  try {
    if (state.editingMappingName) {
      await api(`/api/admin/mappings/${encodeURIComponent(state.editingMappingName)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      });
    } else {
      await api("/api/admin/mappings", {
        method: "POST",
        body: JSON.stringify(body),
      });
    }
    $("addMappingForm").classList.add("hidden");
    state.editingMappingName = null;
    $("newMapName").disabled = false;
    await loadAdminMappings();
    await loadMappings();
  } catch (err) {
    alert(`Fehler: ${err.message}`);
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
          populateMappingFormSelects();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
    tbody.querySelectorAll(".tpl-modeltype").forEach((sel) => {
      sel.addEventListener("change", async () => {
        const name = sel.dataset.name;
        try {
          await api(`/api/admin/templates/${encodeURIComponent(name)}`, {
            method: "PATCH",
            body: JSON.stringify({ model_type: sel.value }),
          });
          const idx = state.templates.findIndex((t) => t.name === name);
          if (idx >= 0) state.templates[idx].model_type = sel.value;
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
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
          cb.checked = !cb.checked;
        }
      });
    });
    tbody.querySelectorAll(".tpl-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm(`Template "${btn.dataset.name}" wirklich löschen?`)) return;
        try {
          await api(`/api/admin/templates/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE" });
          await loadAdminTemplates();
          await loadTemplates();
          populateMappingFormSelects();
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
    populateMappingFormSelects();
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
      status.textContent = "Keine lokalen Templates gefunden. Lege Workflow-JSON-Dateien in comfyui_webui/data/templates/ ab und klicke erneut.";
    } else {
      status.textContent = `${data.found} lokales Template(s) geladen: ${data.templates.join(", ")}`;
    }
    await loadAdminTemplates();
    await loadTemplates();
    populateMappingFormSelects();
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
    populateMappingFormSelects();
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
    populateMappingFormSelects();
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
      body: JSON.stringify({ name: techName, alias }),
    });
    $("aliasNameInput").value = "";
    $("aliasDisplayInput").value = "";
    statusEl.textContent = "✓ Alias gespeichert";
    setTimeout(() => { statusEl.textContent = ""; }, 3000);
    await loadAdminModelAliases();
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
  tbody.innerHTML = "<tr><td colspan='6' class='hint'>Lade …</td></tr>";
  try {
    const data = await api("/api/admin/users");
    tbody.innerHTML = "";
    for (const user of data.users) {
      const tr = document.createElement("tr");
      const isSelf = state.currentUser && user.username === state.currentUser.username;
      tr.innerHTML = `
        <td>${escHtml(user.username)}</td>
        <td>${escHtml(user.role)}</td>
        <td>
          <input type="checkbox" class="user-advanced" data-name="${escHtml(user.username)}"
            ${user.can_advanced || user.role === "admin" ? "checked" : ""}
            ${user.role === "admin" ? "disabled title='Admins haben immer Zugriff'" : ""}
          />
        </td>
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
    tbody.querySelectorAll(".user-advanced").forEach((cb) => {
      if (cb.disabled) return;
      cb.addEventListener("change", async () => {
        try {
          await api(`/api/admin/users/${encodeURIComponent(cb.dataset.name)}`, {
            method: "PATCH",
            body: JSON.stringify({ can_advanced: cb.checked }),
          });
        } catch (err) {
          alert(`Fehler: ${err.message}`);
          cb.checked = !cb.checked;
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='6' class='error'>${escHtml(err.message)}</td></tr>`;
  }
}

async function addUser() {
  const username = $("newUserName").value.trim();
  const password = $("newUserPass").value;
  const role = $("newUserRole").value;
  const can_advanced = $("newUserCanAdvanced").checked;
  if (!username || !password) {
    alert("Bitte Benutzername und Passwort angeben.");
    return;
  }
  try {
    const user = await api("/api/admin/users", {
      method: "POST",
      body: JSON.stringify({ username, password, role }),
    });
    // Set can_advanced if checked and role is not admin (admins always have it)
    if (can_advanced && role !== "admin") {
      await api(`/api/admin/users/${encodeURIComponent(user.username)}`, {
        method: "PATCH",
        body: JSON.stringify({ can_advanced: true }),
      });
    }
    $("newUserName").value = "";
    $("newUserPass").value = "";
    $("newUserCanAdvanced").checked = false;
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
  await loadMappings().catch((e) => errors.push(`Mappings: ${e.message}`));
  await loadOllamaModels().catch((e) => errors.push(`Ollama: ${e.message}`));
  await loadCheckpoints().catch((e) => errors.push(`Checkpoints: ${e.message}`));
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
$("tabAdvanced").addEventListener("click", () => showTab("Advanced"));
$("tabAdmin").addEventListener("click", () => showTab("Admin"));

// ---------------------------------------------------------------------------
// Event listeners – generate tab
// ---------------------------------------------------------------------------
$("mappingSelect").addEventListener("change", onMappingChange);
$("refreshMappingsBtn").addEventListener("click", async () => {
  try {
    await loadMappings();
    setStatus("Mappings aktualisiert.");
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

// ---------------------------------------------------------------------------
// Event listeners – admin tab
// ---------------------------------------------------------------------------
$("adminAddMappingBtn").addEventListener("click", () => {
  const form = $("addMappingForm");
  if (form.classList.contains("hidden")) {
    openMappingForm(null);
  } else {
    form.classList.add("hidden");
  }
});
$("saveMappingBtn").addEventListener("click", saveMapping);
$("cancelMappingBtn").addEventListener("click", () => {
  $("addMappingForm").classList.add("hidden");
  state.editingMappingName = null;
  $("newMapName").disabled = false;
});

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

