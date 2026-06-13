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
  gallery: [],         // current gallery items
  galleryUsername: "", // whose gallery is displayed
  galleryMeta: null,   // currently open metadata item
  testRunId: null,     // active test run ID
  testRunPollTimer: null, // setInterval handle for test run polling
  activeTemplateDefaults: null, // workflow_defaults from the selected template (or null)
  activeTemplateName: null,     // display name of the active imported template
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
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail != null) {
        detail = JSON.stringify(payload.detail);
      } else {
        detail = JSON.stringify(payload);
      }
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
    $("galleryAdminBar").classList.remove("hidden");
    $("testModeBtn").classList.remove("hidden");
  } else {
    $("tabAdmin").classList.add("hidden");
    $("galleryAdminBar").classList.add("hidden");
    $("testModeBtn").classList.add("hidden");
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
  if (state.currentUser) {
    try { sessionStorage.removeItem(`pendingJob_${state.currentUser.username}`); } catch {}
    try { sessionStorage.removeItem(`lastImages_${state.currentUser.username}`); } catch {}
  }
  state.currentUser = null;
  showLogin();
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function showTab(name) {
  const tabs = ["Generate", "Gallery", "Advanced", "Admin"];
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
  if (name === "Gallery") {
    loadGallery();
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
    // Admins fetch all templates (including inactive) via the same endpoint;
    // the server tags each record with _inactive=true when not approved+enabled.
    const data = await api("/api/templates");
    state.templates = data.templates || [];
  } catch {
    state.templates = [];
  }
}

async function loadMappings() {
  const isAdmin = state.currentUser && state.currentUser.role === "admin";
  try {
    const endpoint = isAdmin ? "/api/admin/mappings" : "/api/mappings";
    const data = await api(endpoint);
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
    const inactive = isAdmin && m.enabled === false;
    opt.textContent = (m.display_name || m.name) + (inactive ? " (x)" : "");
    if (inactive) opt.style.color = "var(--muted, #888)";
    select.appendChild(opt);
  }
  const note = $("mappingNote");
  if (state.mappings.length === 0) {
    note.textContent = "Keine Mappings verfügbar. Ein Admin muss zuerst Mappings anlegen.";
  } else if (isAdmin) {
    const active = state.mappings.filter((m) => m.enabled !== false).length;
    const inactive = state.mappings.length - active;
    note.textContent = `${state.mappings.length} Mapping(s) gesamt – ${active} aktiv, ${inactive} inaktiv (x).`;
  } else {
    note.textContent = `${state.mappings.length} Mapping(s) verfügbar. Wähle eines, um die Vorgaben zu laden.`;
  }
}

// Called when the user changes the mapping selector
function onMappingChange() {
  const name = $("mappingSelect").value;
  const mapping = state.mappings.find((m) => m.name === name);
  if (!mapping) {
    // No mapping selected – clear template defaults
    state.activeTemplateDefaults = null;
    state.activeTemplateName = null;
    updateAdvancedDefaultHints();
    return;
  }
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

  // Load workflow defaults for the template, if available
  _loadTemplateDefaults(mapping.template_name);
}

// ---------------------------------------------------------------------------
// Workflow template-defaults helpers
// ---------------------------------------------------------------------------

/**
 * Look up workflow_defaults for the given template name and store in state.
 * Clears defaults for the built-in "default" template (no specific defaults).
 */
function _loadTemplateDefaults(templateName) {
  if (!templateName || templateName === "default") {
    state.activeTemplateDefaults = null;
    state.activeTemplateName = null;
    updateAdvancedDefaultHints();
    return;
  }
  const tpl = state.templates.find((t) => t.name === templateName);
  const defaults = tpl && tpl.analysis && tpl.analysis.workflow_defaults
    ? tpl.analysis.workflow_defaults
    : null;

  // Only surface defaults when at least some useful values are available
  const hasAny = defaults && (
    defaults.steps != null ||
    defaults.cfg != null ||
    defaults.sampler_name != null ||
    defaults.scheduler != null ||
    defaults.width != null ||
    defaults.height != null
  );

  state.activeTemplateDefaults = hasAny ? defaults : null;
  state.activeTemplateName = tpl ? (tpl.display_name || tpl.name) : templateName;
  updateAdvancedDefaultHints();
}

/**
 * Update the template-defaults info panel and per-field hint labels in the
 * Erweitert tab.  Call this whenever the active template or any Erweitert
 * field value changes.
 */
function updateAdvancedDefaultHints() {
  const panel = $("templateDefaultsPanel");
  const panelText = $("templateDefaultsText");
  const warningEl = $("templateDeviationWarning");
  const deviationDetails = $("templateDeviationDetails");

  if (!panel) return; // Advanced tab not rendered yet

  const defs = state.activeTemplateDefaults;

  if (!defs) {
    // No imported-template defaults → hide everything
    panel.classList.add("hidden");
    if (warningEl) warningEl.classList.add("hidden");
    _clearAllFieldHints();
    return;
  }

  // Build summary text for the info panel
  const summaryParts = [];
  if (defs.sampler_name != null) summaryParts.push(`Sampler: <strong>${escHtml(String(defs.sampler_name))}</strong>`);
  if (defs.scheduler != null)    summaryParts.push(`Scheduler: <strong>${escHtml(String(defs.scheduler))}</strong>`);
  if (defs.steps != null)        summaryParts.push(`Steps: <strong>${defs.steps}</strong>`);
  if (defs.cfg != null)          summaryParts.push(`CFG: <strong>${defs.cfg}</strong>`);
  if (defs.width != null && defs.height != null) summaryParts.push(`Auflösung: <strong>${defs.width}&times;${defs.height}</strong>`);
  if (defs.batch_size != null)   summaryParts.push(`Bilder: <strong>${defs.batch_size}</strong>`);

  const tplLabel = state.activeTemplateName ? escHtml(state.activeTemplateName) : "Template";
  panelText.innerHTML = `Template-Standard (${tplLabel}): ${summaryParts.join(", ")}`;
  panel.classList.remove("hidden");

  // Per-field hint update
  const fieldMap = [
    { hintId: "tdh_steps",      fieldId: "steps",      defVal: defs.steps,        toNum: true },
    { hintId: "tdh_cfg",        fieldId: "cfg",         defVal: defs.cfg,          toNum: true },
    { hintId: "tdh_imageCount", fieldId: "imageCount",  defVal: defs.batch_size,   toNum: true },
    { hintId: "tdh_width",      fieldId: "width",       defVal: defs.width,        toNum: true },
    { hintId: "tdh_height",     fieldId: "height",      defVal: defs.height,       toNum: true },
    { hintId: "tdh_sampler",    fieldId: "sampler",     defVal: defs.sampler_name, toNum: false },
    { hintId: "tdh_scheduler",  fieldId: "scheduler",   defVal: defs.scheduler,    toNum: false },
  ];

  const deviations = [];

  for (const { hintId, fieldId, defVal, toNum } of fieldMap) {
    const hintEl = $(hintId);
    const fieldEl = $(fieldId);
    if (!hintEl || defVal == null) {
      if (hintEl) { hintEl.classList.add("hidden"); hintEl.className = "template-default-hint hidden"; }
      continue;
    }
    const currentVal = toNum ? Number(fieldEl.value) : fieldEl.value;
    const defValCmp  = toNum ? Number(defVal) : String(defVal);
    const matches = toNum
      ? Math.abs(currentVal - defValCmp) < 0.001
      : currentVal === defValCmp;

    hintEl.classList.remove("hidden");
    if (matches) {
      hintEl.className = "template-default-hint tdh-match";
      hintEl.textContent = `✓ Template-Standard: ${defVal}`;
    } else {
      hintEl.className = "template-default-hint tdh-warn";
      hintEl.textContent = `⚠ Template-Standard: ${defVal} (aktuell: ${currentVal})`;
      deviations.push(`${fieldId}=${currentVal} (Standard: ${defVal})`);
    }
  }

  // Show/hide deviation warning banner
  if (deviations.length > 0) {
    deviationDetails.textContent = " " + deviations.join("; ");
    warningEl.classList.remove("hidden");
  } else {
    warningEl.classList.add("hidden");
  }
}

/** Remove all per-field template-default hints. */
function _clearAllFieldHints() {
  for (const id of ["tdh_steps", "tdh_cfg", "tdh_imageCount", "tdh_width", "tdh_height", "tdh_sampler", "tdh_scheduler"]) {
    const el = $(id);
    if (el) { el.className = "template-default-hint hidden"; el.textContent = ""; }
  }
}

/** Attach change listeners on Erweitert inputs/selects to update hints live. */
function initAdvancedDefaultListeners() {
  for (const id of ["steps", "cfg", "imageCount", "width", "height", "sampler", "scheduler"]) {
    const el = $(id);
    if (el) el.addEventListener("input", updateAdvancedDefaultHints);
    if (el) el.addEventListener("change", updateAdvancedDefaultHints);
  }
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

  // Seed is always taken from the dedicated genSeed field (visible to all users).
  // The value -1 means "random" and is resolved server-side.
  const seed = Number($("genSeed").value);

  // Generation parameters: use Erweitert overrides if user has permission,
  // otherwise fall back to mapping values
  const steps    = canAdv ? Number($("steps").value)     : (mapping ? (mapping.steps ?? 30)        : 30);
  const cfg      = canAdv ? Number($("cfg").value)       : (mapping ? (mapping.cfg ?? 7)           : 7);
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
    const container = document.createElement("div");
    container.className = "result-image-wrap";

    const img = document.createElement("img");
    const sep = url.includes("?") ? "&" : "?";
    const cacheBust = `${url}${sep}_=${Date.now()}`;
    img.src = cacheBust;
    img.alt = "Generiertes Bild";

    const actions = document.createElement("div");
    actions.className = "result-image-actions";

    const dlJpg = document.createElement("button");
    dlJpg.className = "btn-sm btn-secondary";
    dlJpg.type = "button";
    dlJpg.textContent = "⬇ JPG";
    dlJpg.title = "Als JPEG herunterladen";
    dlJpg.addEventListener("click", () => downloadImageFromUrl(cacheBust, "jpg"));

    const dlPng = document.createElement("button");
    dlPng.className = "btn-sm btn-secondary";
    dlPng.type = "button";
    dlPng.textContent = "⬇ PNG";
    dlPng.title = "Als PNG herunterladen";
    dlPng.addEventListener("click", () => downloadImageFromUrl(cacheBust, "png"));

    actions.appendChild(dlJpg);
    actions.appendChild(dlPng);
    container.appendChild(img);
    container.appendChild(actions);
    wrap.appendChild(container);
  }
}

// ---------------------------------------------------------------------------
// Pending-job helpers (sessionStorage, survives page reload)
// ---------------------------------------------------------------------------
function savePendingJob(promptId, clientId) {
  if (!state.currentUser) return;
  try {
    sessionStorage.setItem(
      `pendingJob_${state.currentUser.username}`,
      JSON.stringify({ promptId, clientId }),
    );
  } catch { /* quota – ignore */ }
}

function clearPendingJob() {
  if (!state.currentUser) return;
  try { sessionStorage.removeItem(`pendingJob_${state.currentUser.username}`); } catch {}
}

// ---------------------------------------------------------------------------
// SSE progress connection (reusable for new jobs and reconnect after reload)
// ---------------------------------------------------------------------------
function connectToProgress(promptId, clientId) {
  return new Promise((resolve, reject) => {
    let resolved = false;
    let reconnects = 0;
    const MAX_RECONNECTS = 60; // ~3 min with 3s delays

    showProgress(true);
    setProgressBar(0, 0, null);
    setButtons(true);

    function connect() {
      const url = `/api/comfy/progress/${encodeURIComponent(promptId)}?client_id=${encodeURIComponent(clientId)}`;
      const evtSource = new EventSource(url);

      evtSource.onmessage = (event) => {
        reconnects = 0;
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
          resolved = true;
          showProgress(false);
          const doneImages = data.images || [];
          showImages(doneImages);
          clearPendingJob();
          if (doneImages.length > 0 && state.currentUser) {
            try {
              sessionStorage.setItem(
                `lastImages_${state.currentUser.username}`,
                JSON.stringify(doneImages),
              );
            } catch { /* quota exceeded – ignore */ }
          }
          setStatus(`Fertig. ${doneImages.length} Bild(er) gespeichert.`);
          setButtons(false);
          resolve(doneImages);
        } else if (data.type === "error") {
          evtSource.close();
          resolved = true;
          showProgress(false);
          setButtons(false);
          clearPendingJob();
          reject(new Error(data.message));
        }
      };

      evtSource.onerror = () => {
        evtSource.close();
        if (resolved) return;
        if (reconnects >= MAX_RECONNECTS) {
          showProgress(false);
          setButtons(false);
          clearPendingJob();
          reject(new Error("Verbindung getrennt. Bitte Galerie auf fertige Bilder prüfen."));
          return;
        }
        reconnects++;
        setStatus(`Verbindung unterbrochen – Wiederverbindung ${reconnects} …`);
        setTimeout(connect, 3000);
      };
    }

    connect();
  });
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

    const { prompt_id, client_id, translated_prompt, translated_negative_prompt, actual_seed } = submitData;
    if (translated_prompt) $("translatedPrompt").value = translated_prompt;
    if (translated_negative_prompt) $("translatedNegativePrompt").value = translated_negative_prompt;
    // Show the actual seed that was used (important when seed was -1)
    if (actual_seed != null) $("genSeed").value = actual_seed;

    // Persist job IDs so the page can reconnect to this job after a reload
    savePendingJob(prompt_id, client_id);

    await connectToProgress(prompt_id, client_id);
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
    // Admins see all templates; inactive ones get an (x) prefix
    const label = tpl.display_name || tpl.name;
    opt.textContent = tpl._inactive ? `(x) ${label}` : label;
    if (tpl._inactive) opt.style.color = "var(--muted)";
    tplSel.appendChild(opt);
  }
  if (prevTpl) tplSel.value = prevTpl;

  // Re-filter checkpoint dropdown when template changes
  tplSel.onchange = () => {
    populateCheckpointSelect();
    _updateMappingFormTemplateHint(tplSel.value);
  };

  // Populate checkpoint and ollama model dropdowns
  populateCheckpointSelect();
  populateOllamaModelSelect();

  // Populate sampler/scheduler dropdowns in form
  fillSelectSimple("newMapSampler", state.samplers.length ? state.samplers : ["euler", "dpmpp_2m", "ddim"], "euler");
  fillSelectSimple("newMapScheduler", state.schedulers.length ? state.schedulers : ["normal", "karras", "simple"], "normal");

  // Show template defaults for current selection
  _updateMappingFormTemplateHint(tplSel.value);
}

/**
 * Update the template-defaults hint in the admin mapping form.
 * Creates or updates a small info element below the template selector.
 */
function _updateMappingFormTemplateHint(templateName) {
  const container = $("newMapTemplateHint");
  if (!container) return;

  if (!templateName || templateName === "default") {
    container.className = "hidden";
    container.textContent = "";
    return;
  }
  const tpl = state.templates.find((t) => t.name === templateName);
  const defs = tpl && tpl.analysis && tpl.analysis.workflow_defaults
    ? tpl.analysis.workflow_defaults
    : null;
  if (!defs) {
    container.className = "hint";
    container.textContent = "Keine analysierten Workflow-Standards verfügbar.";
    return;
  }
  const parts = [];
  if (defs.sampler_name != null) parts.push(`Sampler: ${defs.sampler_name}`);
  if (defs.scheduler != null)    parts.push(`Scheduler: ${defs.scheduler}`);
  if (defs.steps != null)        parts.push(`Steps: ${defs.steps}`);
  if (defs.cfg != null)          parts.push(`CFG: ${defs.cfg}`);
  if (defs.width != null && defs.height != null) parts.push(`${defs.width}×${defs.height}`);
  if (defs.batch_size != null)   parts.push(`Bilder: ${defs.batch_size}`);
  if (defs.model_name)           parts.push(`Modell: ${defs.model_name}`);
  container.className = "hint template-form-defaults";
  container.textContent = parts.length
    ? `⚙ Workflow-Standards: ${parts.join(", ")}`
    : "Workflow-Standards konnten nicht extrahiert werden.";
}
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

function _analysisStatusBadge(tpl) {
  const ana = tpl.analysis;
  if (!ana) return '<span title="Noch nicht analysiert" style="opacity:0.5">–</span>';
  const warns = (ana.warnings || []).length;
  const errs = (ana.errors || []).length;
  const parseErr = ana.parse_error ? `Parse-Fehler: ${ana.parse_error}\n` : "";
  const warnText = warns ? `Warnungen:\n• ${(ana.warnings || []).join("\n• ")}\n` : "";
  const errText = errs ? `Fehler:\n• ${(ana.errors || []).join("\n• ")}\n` : "";
  const extras = [
    `Sampler: ${ana.sampler_count ?? "?"}`,
    `Loader: ${ana.model_loader_count ?? "?"}`,
    ana.negative_is_zero_out ? "Negativ: ZeroOut (fest)" : "",
    ana.is_potentially_img2img ? "⚠ Möglicher img2img-Pfad" : "",
  ].filter(Boolean).join(" | ");
  const title = `${parseErr}${errText}${warnText}${extras}`.trim();
  if (!ana.is_usable) {
    return `<span class="badge badge-error" title="${escHtml(title)}">✗ Nicht verwendbar</span>`;
  }
  if (warns > 0) {
    return `<span class="badge badge-warn" title="${escHtml(title)}">⚠ ${warns} Warnung${warns > 1 ? "en" : ""}</span>`;
  }
  return `<span class="badge badge-ok" title="${escHtml(title)}">✓ OK</span>`;
}

async function loadAdminTemplates() {
  const tbody = $("adminTemplateBody");
  tbody.innerHTML = "<tr><td colspan='8' class='hint'>Lade …</td></tr>";
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
        <td class="tpl-analysis-cell">
          ${_analysisStatusBadge(tpl)}
          ${tpl.filename ? `<button class="btn-sm tpl-reanalyze" data-name="${escHtml(tpl.name)}" type="button" title="Analyse neu ausführen" style="margin-left:0.25rem">&#8635;</button>` : ""}
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
    tbody.querySelectorAll(".tpl-reanalyze").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const name = btn.dataset.name;
        const cell = btn.closest(".tpl-analysis-cell");
        cell.innerHTML = '<small class="hint">Analysiere…</small>';
        try {
          const result = await api(`/api/admin/templates/${encodeURIComponent(name)}/analysis`, { method: "GET" });
          // Reload table to show updated analysis
          await loadAdminTemplates();
        } catch (err) {
          cell.innerHTML = `<span class="badge badge-error" title="${escHtml(err.message)}">✗ Fehler</span>`;
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
    tbody.innerHTML = `<tr><td colspan='8' class='error'>${escHtml(err.message)}</td></tr>`;
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
  tbody.innerHTML = "<tr><td colspan='7' class='hint'>Lade …</td></tr>";
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
        <td>
          ${!isSelf
            ? `<button class="btn-sm btn-danger user-delete" data-name="${escHtml(user.username)}" type="button">L&ouml;schen</button>`
            : ""
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
    tbody.querySelectorAll(".user-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm(`Benutzer "${btn.dataset.name}" und seine gesamte Galerie unwiderruflich löschen?`)) return;
        try {
          await api(`/api/admin/users/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE" });
          await loadAdminUsers();
        } catch (err) {
          alert(`Fehler: ${err.message}`);
        }
      });
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan='7' class='error'>${escHtml(err.message)}</td></tr>`;
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
// Gallery
// ---------------------------------------------------------------------------
function updateGalleryDeleteSelectedBtn() {
  const btn = $("galleryDeleteSelectedBtn");
  if (!btn) return;
  const selected = document.querySelectorAll("#galleryGrid .gallery-item-checkbox:checked");
  if (selected.length > 0) {
    btn.classList.remove("hidden");
    btn.textContent = `🗑 Ausgewählte löschen (${selected.length})`;
  } else {
    btn.classList.add("hidden");
  }
}

async function loadGallery(username) {
  const status = $("galleryStatus");
  const grid = $("galleryGrid");
  const title = $("galleryTitle");
  status.textContent = "Lade …";
  grid.innerHTML = "";

  try {
    let data;
    if (username) {
      data = await api(`/api/admin/gallery/${encodeURIComponent(username)}`);
    } else {
      data = await api("/api/gallery");
    }
    state.gallery = data.items || [];
    state.galleryUsername = data.username || (state.currentUser ? state.currentUser.username : "");
    title.textContent = username
      ? `Galerie: ${escHtml(data.username)}`
      : "Meine Galerie";
    renderGallery(state.gallery, state.galleryUsername);
    status.textContent = state.gallery.length === 0
      ? "Noch keine Bilder in der Galerie."
      : `${state.gallery.length} Bild(er)`;
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
    status.classList.add("error");
  }
}

function renderGallery(items, username) {
  const grid = $("galleryGrid");
  grid.innerHTML = "";
  const isAdmin = state.currentUser && state.currentUser.role === "admin";

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "gallery-item";
    card.dataset.id = item.id;

    const dateStr = item.created_at
      ? new Date(item.created_at).toLocaleString("de-DE", { dateStyle: "short", timeStyle: "short" })
      : "";
    const seedStr = item.actual_seed != null ? `Seed: ${item.actual_seed}` : "";
    const modelStr = item.checkpoint ? item.checkpoint.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "") : "";
    const cfgStepsStr = (item.steps || item.cfg)
      ? [item.steps ? `Steps: ${item.steps}` : null, item.cfg ? `CFG: ${item.cfg}` : null]
          .filter(Boolean).join(" · ")
      : "";

    const imgUrl = `/api/gallery/image/${encodeURIComponent(item.id)}?_=${Date.now()}`;

    card.innerHTML = `
      <input type="checkbox" class="gallery-item-checkbox" title="Bild ausw&auml;hlen" />
      <img src="${imgUrl}" alt="Galeriebild" loading="lazy" />
      <div class="gallery-item-meta">
        <span class="gallery-date">${escHtml(dateStr)}</span>
        <span class="gallery-seed hint">${escHtml(seedStr)}</span>
        ${modelStr ? `<span class="gallery-model hint" title="${escHtml(item.checkpoint || "")}">${escHtml(modelStr)}</span>` : ""}
        ${cfgStepsStr ? `<span class="gallery-cfg-steps hint">${escHtml(cfgStepsStr)}</span>` : ""}
      </div>
      <div class="gallery-item-actions">
        <button class="btn-sm gallery-info-btn" data-id="${escHtml(item.id)}" type="button" title="Details anzeigen">&#9432;</button>
        <button class="btn-sm btn-secondary gallery-dl-jpg" data-id="${escHtml(item.id)}" type="button" title="Als JPEG herunterladen">&#11015;JPG</button>
        <button class="btn-sm btn-secondary gallery-dl-png" data-id="${escHtml(item.id)}" type="button" title="Als PNG herunterladen">&#11015;PNG</button>
        <button class="btn-sm btn-danger gallery-del-btn" data-id="${escHtml(item.id)}" data-user="${escHtml(username)}" type="button" title="Bild l&ouml;schen">&#128465;</button>
      </div>
    `;
    grid.appendChild(card);
  }

  // Checkbox: toggle selected class and update toolbar button
  grid.querySelectorAll(".gallery-item-checkbox").forEach((cb) => {
    cb.addEventListener("change", () => {
      cb.closest(".gallery-item").classList.toggle("selected", cb.checked);
      updateGalleryDeleteSelectedBtn();
    });
  });

  grid.querySelectorAll(".gallery-info-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = state.gallery.find((x) => x.id === btn.dataset.id);
      if (item) openGalleryMeta(item);
    });
  });

  grid.querySelectorAll(".gallery-dl-jpg, .gallery-dl-png").forEach((btn) => {
    btn.addEventListener("click", () => {
      const format = btn.classList.contains("gallery-dl-jpg") ? "jpg" : "png";
      const imgUrl = `/api/gallery/image/${encodeURIComponent(btn.dataset.id)}`;
      downloadImageFromUrl(imgUrl, format);
    });
  });

  grid.querySelectorAll(".gallery-del-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm("Dieses Bild wirklich aus der Galerie löschen?")) return;
      try {
        const isAdmin = state.currentUser && state.currentUser.role === "admin";
        const targetUser = btn.dataset.user;
        const ownUser = state.currentUser ? state.currentUser.username : "";
        if (isAdmin && targetUser && targetUser !== ownUser) {
          await api(`/api/admin/gallery/${encodeURIComponent(targetUser)}/${encodeURIComponent(btn.dataset.id)}`, { method: "DELETE" });
        } else {
          await api(`/api/gallery/${encodeURIComponent(btn.dataset.id)}`, { method: "DELETE" });
        }
        await loadGallery(targetUser !== ownUser ? targetUser : null);
      } catch (err) {
        alert(`Fehler: ${err.message}`);
      }
    });
  });

  updateGalleryDeleteSelectedBtn();
}

function openGalleryMeta(item) {
  state.galleryMeta = item;
  $("galleryMetaTitle").textContent = item.created_at
    ? `Bild vom ${new Date(item.created_at).toLocaleString("de-DE")}`
    : "Bilddetails";

  const imgUrl = `/api/gallery/image/${encodeURIComponent(item.id)}?_=${Date.now()}`;

  const rows = [
    ["Prompt (Deutsch)", item.prompt_de],
    ["Neg. Prompt (Deutsch)", item.negative_prompt_de],
    ["Übersetzter Prompt (Englisch)", item.translated_prompt],
    ["Übersetzter Neg. Prompt (Englisch)", item.translated_negative_prompt],
    ["Template", item.workflow_template],
    ["Modell / Checkpoint", item.checkpoint],
    ["Ollama-Modell", item.ollama_model],
    ["Seed (tatsächlich)", item.actual_seed],
    ["Steps", item.steps],
    ["CFG-Scale", item.cfg],
    ["Sampler", item.sampler],
    ["Scheduler", item.scheduler],
    ["Größe", item.width && item.height ? `${item.width} × ${item.height} px` : null],
    ["Anzahl Bilder", item.image_count],
    ["Benutzer", item.username],
  ];

  let html = `<img src="${imgUrl}" alt="Galeriebild" class="gallery-meta-img" />`;
  html += `<table class="gallery-meta-table">`;
  for (const [label, val] of rows) {
    if (val == null || val === "" || val === 0 && label !== "Seed (tatsächlich)") continue;
    html += `<tr><th>${escHtml(label)}</th><td>${escHtml(String(val))}</td></tr>`;
  }
  html += `</table>`;

  $("galleryMetaContent").innerHTML = html;
  $("galleryMetaOverlay").classList.remove("hidden");

  // Wire up download buttons in modal
  $("galleryMetaDlJpgBtn").onclick = () => downloadImageFromUrl(imgUrl, "jpg");
  $("galleryMetaDlPngBtn").onclick = () => downloadImageFromUrl(imgUrl, "png");
}

function closeGalleryMeta() {
  $("galleryMetaOverlay").classList.add("hidden");
  state.galleryMeta = null;
}

function applyGallerySettings() {
  const item = state.galleryMeta;
  if (!item) return;
  closeGalleryMeta();
  showTab("Generate");
  if (item.prompt_de) $("promptDe").value = item.prompt_de;
  if (item.negative_prompt_de) $("negativePrompt").value = item.negative_prompt_de;
  if (item.actual_seed != null) $("genSeed").value = item.actual_seed;
  if (item.translated_prompt) $("translatedPrompt").value = item.translated_prompt;
  if (item.translated_negative_prompt) $("translatedNegativePrompt").value = item.translated_negative_prompt;
  // Advanced fields (applied even if tab is hidden – stored and used on generate)
  if (item.steps) $("steps").value = item.steps;
  if (item.cfg) $("cfg").value = item.cfg;
  if (item.width) $("width").value = item.width;
  if (item.height) $("height").value = item.height;
  if (item.sampler && state.samplers.includes(item.sampler)) $("sampler").value = item.sampler;
  if (item.scheduler && state.schedulers.includes(item.scheduler)) $("scheduler").value = item.scheduler;
  if (item.image_count) $("imageCount").value = item.image_count;
  setStatus("Einstellungen aus Galerie übernommen.");
}

async function loadAdminGalleryUsers() {
  const sel = $("galleryUserSelect");
  sel.innerHTML = '<option value="">– Eigene Galerie –</option>';
  try {
    const data = await api("/api/admin/gallery");
    for (const u of data.users || []) {
      const opt = document.createElement("option");
      opt.value = u.username;
      opt.textContent = `${u.username} (${u.count} Bild${u.count !== 1 ? "er" : ""})`;
      sel.appendChild(opt);
    }
  } catch {
    // ignore – admin panel will show an error if it fails
  }
}

// ---------------------------------------------------------------------------
// Admin – Test mode (parametric sweep over steps / cfg)
// ---------------------------------------------------------------------------

function openTestMode() {
  const panel = $("testModePanel");
  if (!panel) return;
  const isHidden = panel.classList.contains("hidden");
  panel.classList.toggle("hidden", !isHidden);
  if (isHidden) {
    // Pre-fill from Erweitert tab values
    const stepsVal = $("steps") ? $("steps").value : "20";
    const cfgVal   = $("cfg")   ? $("cfg").value   : "7";
    $("tmStepsFrom").value = stepsVal;
    $("tmStepsTo").value   = stepsVal;
    $("tmCfgFrom").value   = cfgVal;
    $("tmCfgTo").value     = cfgVal;
    $("tmCfgStep").value   = "0.5";
    updateTestRunCombinationCount();
    // If there's an active run, reconnect to it
    if (state.testRunId) {
      _startTestRunPolling(state.testRunId);
    }
  }
}

function updateTestRunCombinationCount() {
  const stepsFrom = parseInt($("tmStepsFrom").value) || 0;
  const stepsTo   = parseInt($("tmStepsTo").value)   || 0;
  const cfgFrom   = parseFloat($("tmCfgFrom").value) || 0;
  const cfgTo     = parseFloat($("tmCfgTo").value)   || 0;
  const cfgStep   = parseFloat($("tmCfgStep").value) || 0.5;

  const nSteps = Math.max(0, stepsTo - stepsFrom + 1);
  let nCfg = 0;
  if (cfgStep > 0) {
    let v = cfgFrom;
    while (v <= cfgTo + 1e-9) { nCfg++; v = Math.round((v + cfgStep) * 10000) / 10000; }
  } else {
    nCfg = 1;
  }
  const total = nSteps * nCfg;
  const info = $("tmCombinationInfo");
  if (info) info.textContent = `= ${total} Kombinationen`;
}

async function startTestRun() {
  const mapping = getActiveMapping();
  if (!mapping) {
    alert("Bitte zuerst ein Mapping auswählen.");
    return;
  }
  const promptDe = $("promptDe").value.trim();
  if (!promptDe) {
    alert("Bitte einen deutschen Prompt eingeben.");
    return;
  }

  const stepsFrom = parseInt($("tmStepsFrom").value);
  const stepsTo   = parseInt($("tmStepsTo").value);
  const cfgFrom   = parseFloat($("tmCfgFrom").value);
  const cfgTo     = parseFloat($("tmCfgTo").value);
  const cfgStep   = parseFloat($("tmCfgStep").value);

  if (isNaN(stepsFrom) || isNaN(stepsTo) || stepsFrom < 1 || stepsTo < stepsFrom) {
    alert("Steps: Bitte gültige Werte eingeben (von ≤ bis, min. 1).");
    return;
  }
  if (isNaN(cfgFrom) || isNaN(cfgTo) || cfgFrom < 0 || cfgTo < cfgFrom) {
    alert("CFG: Bitte gültige Werte eingeben (von ≤ bis, min. 0).");
    return;
  }
  if (isNaN(cfgStep) || cfgStep <= 0) {
    alert("CFG-Schritt muss größer als 0 sein.");
    return;
  }

  const canAdv = state.currentUser && (state.currentUser.can_advanced || state.currentUser.role === "admin");
  const seed = Number($("genSeed").value);
  const steps    = canAdv ? Number($("steps").value)      : (mapping.steps ?? 30);
  const cfg      = canAdv ? Number($("cfg").value)        : (mapping.cfg ?? 7);
  const width    = canAdv ? Number($("width").value)      : (mapping.width ?? 1024);
  const height   = canAdv ? Number($("height").value)     : (mapping.height ?? 1024);
  const sampler  = canAdv ? $("sampler").value.trim()     : (mapping.sampler || "euler");
  const sched    = canAdv ? $("scheduler").value.trim()   : (mapping.scheduler || "normal");
  const imgCount = canAdv ? Number($("imageCount").value) : (mapping.image_count ?? 1);

  const translatedPrompt = $("translatedPrompt").value.trim() || null;
  const translatedNeg    = $("translatedNegativePrompt").value.trim() || null;

  const body = {
    prompt_de: promptDe,
    negative_prompt: $("negativePrompt").value.trim(),
    ollama_model: mapping.ollama_model || "",
    translated_prompt: translatedPrompt,
    translated_negative_prompt: translatedNeg,
    checkpoint: mapping.checkpoint || null,
    workflow_template: mapping.template_name || "default",
    seed,
    width,
    height,
    sampler,
    scheduler: sched,
    image_count: imgCount,
    steps_from: stepsFrom,
    steps_to: stepsTo,
    cfg_from: cfgFrom,
    cfg_to: cfgTo,
    cfg_step: cfgStep,
  };

  $("tmStartBtn").disabled = true;
  _setTestRunStatus("Starte Testlauf …", false);

  try {
    const data = await api("/api/admin/testrun", { method: "POST", body: JSON.stringify(body) });
    state.testRunId = data.run_id;
    // Persist so reconnect after reload works
    try { sessionStorage.setItem("pendingTestRun", data.run_id); } catch {}
    _startTestRunPolling(data.run_id);
  } catch (err) {
    $("tmStartBtn").disabled = false;
    _setTestRunStatus(`Fehler: ${err.message}`, true);
  }
}

function _setTestRunStatus(msg, isError) {
  const el = $("tmStatus");
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
}

function _setTestRunProgress(current, total) {
  const bar = $("tmProgressBar");
  const label = $("tmProgressLabel");
  const wrap = $("tmProgressWrap");
  if (!bar || !label || !wrap) return;
  wrap.classList.remove("hidden");
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  bar.style.width = `${pct}%`;
  label.textContent = `${current} / ${total} (${pct}%)`;
}

function _startTestRunPolling(runId) {
  _stopTestRunPolling();
  _pollTestRun(runId); // immediate first poll
  state.testRunPollTimer = setInterval(() => _pollTestRun(runId), 2000);
}

function _stopTestRunPolling() {
  if (state.testRunPollTimer) {
    clearInterval(state.testRunPollTimer);
    state.testRunPollTimer = null;
  }
}

async function _pollTestRun(runId) {
  try {
    const data = await api(`/api/admin/testrun/${encodeURIComponent(runId)}`);
    _setTestRunProgress(data.current, data.total);

    const seedInfo = data.seed_used != null ? ` · Seed: ${data.seed_used}` : "";
    if (data.status === "running" || data.status === "starting") {
      _setTestRunStatus(
        `Läuft … ${data.current}/${data.total} Kombinationen${seedInfo}`,
        false,
      );
      $("tmStartBtn").disabled = true;
      $("tmCancelBtn").classList.remove("hidden");
    } else if (data.status === "done") {
      _stopTestRunPolling();
      state.testRunId = null;
      try { sessionStorage.removeItem("pendingTestRun"); } catch {}
      const errInfo = data.errors.length > 0 ? ` · ${data.errors.length} Fehler` : "";
      _setTestRunStatus(
        `✓ Fertig – ${data.gallery_ids.length} Bild(er) gespeichert${seedInfo}${errInfo}`,
        data.errors.length > 0,
      );
      $("tmStartBtn").disabled = false;
      $("tmCancelBtn").classList.add("hidden");
      $("tmGalleryBtn").classList.remove("hidden");
      if (data.errors.length > 0) {
        const errEl = $("tmErrors");
        if (errEl) errEl.textContent = data.errors.join("\n");
      }
    } else if (data.status === "cancelled") {
      _stopTestRunPolling();
      state.testRunId = null;
      try { sessionStorage.removeItem("pendingTestRun"); } catch {}
      _setTestRunStatus(`Abgebrochen (${data.current}/${data.total} erledigt${seedInfo})`, false);
      $("tmStartBtn").disabled = false;
      $("tmCancelBtn").classList.add("hidden");
    } else if (data.status === "error") {
      _stopTestRunPolling();
      state.testRunId = null;
      try { sessionStorage.removeItem("pendingTestRun"); } catch {}
      _setTestRunStatus(`Fehler: ${data.error || "Unbekannter Fehler"}`, true);
      $("tmStartBtn").disabled = false;
      $("tmCancelBtn").classList.add("hidden");
    }
  } catch {
    // Transient error – keep polling
  }
}

async function cancelTestRun() {
  if (!state.testRunId) return;
  try {
    await api(`/api/admin/testrun/${encodeURIComponent(state.testRunId)}`, { method: "DELETE" });
    _setTestRunStatus("Abbruch angefordert …", false);
  } catch (err) {
    _setTestRunStatus(`Fehler beim Abbruch: ${err.message}`, true);
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
// Image download helper
// ---------------------------------------------------------------------------
async function downloadImageFromUrl(url, format) {
  try {
    const resp = await fetch(url, { credentials: "same-origin" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const sourceBlob = await resp.blob();

    const img = await createImageBitmap(sourceBlob);
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (format === "jpg") {
      // Fill white background so transparent PNGs convert cleanly to JPEG
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(img, 0, 0);
    img.close && img.close();

    const mimeType = format === "jpg" ? "image/jpeg" : "image/png";
    const quality  = format === "jpg" ? 0.92 : undefined;
    const outBlob  = await new Promise((resolve) => canvas.toBlob(resolve, mimeType, quality));

    // Generate a random filename
    const randPart = Array.from(crypto.getRandomValues(new Uint8Array(8)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    const filename = `img_${randPart}.${format}`;

    const a = document.createElement("a");
    a.href = URL.createObjectURL(outBlob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  } catch (err) {
    alert(`Download fehlgeschlagen: ${err.message}`);
  }
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
  if (state.currentUser && state.currentUser.role === "admin") {
    await loadAdminGalleryUsers().catch(() => {});
  }
  // Restore session state after reload
  if (state.currentUser) {
    // Priority 1: reconnect to an in-progress/queued job
    let reconnected = false;
    try {
      const stored = sessionStorage.getItem(`pendingJob_${state.currentUser.username}`);
      if (stored) {
        const { promptId, clientId } = JSON.parse(stored);
        if (promptId && clientId) {
          setStatus("Wiederverbindung zu laufendem Auftrag …");
          reconnected = true;
          connectToProgress(promptId, clientId).catch((err) => {
            setStatus(`Auftrag beendet: ${err.message}`, true);
          });
        }
      }
    } catch { /* corrupt storage – ignore */ }

    // Priority 2: show last finished images (only when no active job)
    if (!reconnected) {
      try {
        const stored = sessionStorage.getItem(`lastImages_${state.currentUser.username}`);
        if (stored) {
          const urls = JSON.parse(stored);
          if (Array.isArray(urls) && urls.length > 0) {
            showImages(urls);
            setStatus("Letzte Bilder der Sitzung wiederhergestellt. Galerie für ältere Bilder.");
          }
        }
      } catch { /* corrupt storage – ignore */ }
    }
  }
  if (errors.length > 0) {
    setStatus(errors.join(" | "), true);
  }
  // Reconnect to a pending test run (admin only)
  if (state.currentUser && state.currentUser.role === "admin") {
    try {
      const storedRunId = sessionStorage.getItem("pendingTestRun");
      if (storedRunId) {
        state.testRunId = storedRunId;
        _startTestRunPolling(storedRunId);
      }
    } catch { /* ignore */ }
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
$("tabGallery").addEventListener("click", () => showTab("Gallery"));
$("tabAdvanced").addEventListener("click", () => {
  showTab("Advanced");
  updateAdvancedDefaultHints();
});
$("tabAdmin").addEventListener("click", () => showTab("Admin"));

// Attach change listeners on Erweitert fields for live template-default feedback
initAdvancedDefaultListeners();

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
// Event listeners – seed field
// ---------------------------------------------------------------------------
$("genSeedRandomBtn").addEventListener("click", () => {
  // Generate a random uint32 seed (0 to 2^32-1) and put it in the seed field
  const arr = new Uint32Array(1);
  crypto.getRandomValues(arr);
  $("genSeed").value = arr[0];
});

// ---------------------------------------------------------------------------
// Event listeners – gallery
// ---------------------------------------------------------------------------
$("galleryReloadBtn").addEventListener("click", () => {
  const sel = $("galleryUserSelect");
  const user = sel && sel.value ? sel.value : null;
  loadGallery(user);
});

$("galleryDeleteSelectedBtn").addEventListener("click", async () => {
  const checked = [...document.querySelectorAll("#galleryGrid .gallery-item-checkbox:checked")];
  if (checked.length === 0) return;
  if (!confirm(`Wirklich ${checked.length} ausgewählte Bild(er) löschen?`)) return;
  const sel = $("galleryUserSelect");
  const targetUser = sel && sel.value ? sel.value : null;
  const ownUser = state.currentUser ? state.currentUser.username : "";
  const isAdmin = state.currentUser && state.currentUser.role === "admin";
  try {
    for (const cb of checked) {
      const card = cb.closest(".gallery-item");
      const id = card.dataset.id;
      if (isAdmin && targetUser && targetUser !== ownUser) {
        await api(`/api/admin/gallery/${encodeURIComponent(targetUser)}/${encodeURIComponent(id)}`, { method: "DELETE" });
      } else {
        await api(`/api/gallery/${encodeURIComponent(id)}`, { method: "DELETE" });
      }
    }
    await loadGallery(targetUser);
    if (isAdmin) await loadAdminGalleryUsers().catch(() => {});
  } catch (err) {
    alert(`Fehler: ${err.message}`);
  }
});

$("galleryDeleteAllBtn").addEventListener("click", async () => {
  const sel = $("galleryUserSelect");
  const targetUser = sel && sel.value ? sel.value : null;
  const displayName = targetUser || (state.currentUser ? state.currentUser.username : "dich");
  if (!confirm(`Wirklich ALLE Bilder der Galerie von „${displayName}" löschen?`)) return;
  try {
    if (targetUser && state.currentUser && targetUser !== state.currentUser.username) {
      await api(`/api/admin/gallery/${encodeURIComponent(targetUser)}`, { method: "DELETE" });
    } else {
      await api("/api/gallery", { method: "DELETE" });
    }
    await loadGallery(targetUser);
    if (state.currentUser && state.currentUser.role === "admin") {
      await loadAdminGalleryUsers().catch(() => {});
    }
  } catch (err) {
    alert(`Fehler: ${err.message}`);
  }
});

$("galleryLoadUserBtn").addEventListener("click", () => {
  const sel = $("galleryUserSelect");
  loadGallery(sel.value || null);
});

$("galleryMetaCloseBtn").addEventListener("click", closeGalleryMeta);
$("galleryMetaCloseBtn2").addEventListener("click", closeGalleryMeta);
$("galleryMetaApplyBtn").addEventListener("click", applyGallerySettings);
$("galleryMetaOverlay").addEventListener("click", (e) => {
  if (e.target === $("galleryMetaOverlay")) closeGalleryMeta();
});

// ---------------------------------------------------------------------------
// Event listeners – test mode (admin only)
// ---------------------------------------------------------------------------
$("testModeBtn").addEventListener("click", openTestMode);
$("tmStartBtn").addEventListener("click", startTestRun);
$("tmCancelBtn").addEventListener("click", cancelTestRun);
$("tmGalleryBtn").addEventListener("click", () => {
  $("testModePanel").classList.add("hidden");
  showTab("Gallery");
});
["tmStepsFrom", "tmStepsTo", "tmCfgFrom", "tmCfgTo", "tmCfgStep"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("input", updateTestRunCombinationCount);
});

// ---------------------------------------------------------------------------
// Bootstrap: check session, show login or app
// ---------------------------------------------------------------------------
tryAutoLogin();

