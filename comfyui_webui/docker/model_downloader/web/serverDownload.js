import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { collectModels, modelForRow } from "./modelMetadata.js";

let models = [];
let token = "";
let tokenPrompt = null;
let panel;
let progress;
const buttons = new Map();

function button(label, action) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.style.cssText = "padding:5px 8px;margin:3px;border:1px solid #777;border-radius:4px;background:#292929;color:white;cursor:pointer";
    element.addEventListener("click", action);
    return element;
}

function askToken() {
    if (token) return Promise.resolve(token);
    if (tokenPrompt) return tokenPrompt;
    tokenPrompt = new Promise((resolve) => {
        const dialog = document.createElement("dialog");
        const title = document.createElement("p");
        title.textContent = "Admin-Token für serverseitige Modelltransfers";
        const input = document.createElement("input");
        input.type = "password";
        input.autocomplete = "off";
        input.setAttribute("aria-label", "Modelltransfer Admin-Token");
        const finish = (value) => {
            token = value;
            dialog.close();
            dialog.remove();
            tokenPrompt = null;
            resolve(value);
        };
        dialog.append(title, input, button("Freischalten", () => finish(input.value.trim())),
            button("Abbrechen", () => finish("")));
        dialog.addEventListener("cancel", (event) => { event.preventDefault(); finish(""); });
        document.body.append(dialog);
        dialog.showModal();
        input.focus();
    });
    return tokenPrompt;
}

async function request(path, options = {}) {
    const credential = await askToken();
    if (!credential) throw new Error("Abgebrochen");
    const response = await api.fetchApi(`/server_download/${path}`, {
        ...options,
        headers: { ...options.headers, Authorization: ["Bearer", credential].join(" ") },
    });
    if (response.status === 401 || response.status === 403) token = "";
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || `HTTP ${response.status}`);
    return data;
}

async function queueModel(model, element) {
    const aliases = { unet: "diffusion_models", clip: "text_encoders", t2i_adapter: "controlnet" };
    model = { ...model, save_path: aliases[model.save_path] || model.save_path };
    for (const group of buttons.values()) group.delete(element);
    element.disabled = true;
    try {
        await request("start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(model),
        });
        element.textContent = "Auf PVC: wartet";
        const key = `${model.save_path}/${model.filename}`;
        if (!buttons.has(key)) buttons.set(key, new Set());
        buttons.get(key).add(element);
        await updateProgress();
    } catch (error) {
        element.textContent = "Erneut auf PVC laden";
        element.disabled = false;
        showPanel();
        progress.textContent = error.message;
    }
}

function addDownloadButton(row, model) {
    const existing = row.querySelector(".ice-model-download");
    const key = JSON.stringify(model);
    if (existing?.dataset.model === key) return;
    existing?.remove();
    const element = button("Auf PVC laden", () => queueModel(model, element));
    element.className = "ice-model-download";
    element.dataset.model = key;
    row.append(element);
}

function inject() {
    if (!document.body) return;
    for (const original of document.querySelectorAll('[data-testid="missing-model-download"]')) {
        const row = original.parentElement;
        const model = modelForRow(row, models);
        if (model) addDownloadButton(row, model);
    }
    for (const row of document.querySelectorAll(".comfy-missing-models .p-listbox-option")) {
        const model = modelForRow(row, models);
        if (model) addDownloadButton(row, model);
    }
    for (const actions of document.querySelectorAll('[data-testid="missing-model-actions"]')) {
        if (actions.querySelector(".ice-model-download-all")) continue;
        const element = button("Alle auf PVC laden", async () => {
            const container = actions.parentElement;
            for (const transfer of container.querySelectorAll(".ice-model-download")) {
                if (!transfer.disabled) await queueModel(JSON.parse(transfer.dataset.model), transfer);
            }
        });
        element.className = "ice-model-download-all";
        actions.append(element);
    }
}

async function updateProgress() {
    if (!token) return;
    try {
        const data = await request("status");
        const lines = [];
        for (const [id, state] of Object.entries(data.downloads || {})) {
            const percent = Math.round(state.progress || 0);
            const label = state.status === "completed" ? "Fertig auf PVC" :
                state.status === "error" ? "Fehler" : `${state.status} ${percent}%`;
            lines.push(`${state.filename}: ${label}${state.error ? ` — ${state.error}` : ""}`);
            for (const element of buttons.get(id) || []) {
                if (element.textContent !== label) element.textContent = label;
                if (state.status === "error" || (state.status === "completed" && !element.dataset.model)) {
                    element.disabled = false;
                }
            }
        }
        if (progress && lines.length) {
            const text = lines.join("\n");
            if (progress.textContent !== text) progress.textContent = text;
        }
    } catch (error) {
        if (progress) progress.textContent = error.message;
    }
}

function showPanel() {
    if (panel) { panel.showModal(); return; }
    panel = document.createElement("dialog");
    panel.style.cssText = "max-width:650px;width:90%;background:#202020;color:white;border:1px solid #888";
    const title = document.createElement("h3");
    title.textContent = "Modelle direkt auf das ComfyUI-PVC laden";
    const note = document.createElement("p");
    note.textContent = "Nur vertrauenswürdige HTTPS-Modellquellen. Der Token bleibt nur bis zum Neuladen im Arbeitsspeicher. Geschützte Downloads ggf. manuell in der WebUI hochladen.";
    const url = document.createElement("input");
    url.placeholder = "HTTPS-Downloadlink";
    url.setAttribute("aria-label", "Modell Downloadlink");
    url.style.width = "95%";
    const filename = document.createElement("input");
    filename.placeholder = "Dateiname.safetensors";
    filename.setAttribute("aria-label", "Modell Dateiname");
    const directory = document.createElement("select");
    directory.setAttribute("aria-label", "Modell Zielverzeichnis");
    progress = document.createElement("pre");
    progress.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere";
    const loadDirectories = async () => {
        try {
            const data = await request("directories");
            directory.replaceChildren(...data.directories.map((name) => {
                const option = document.createElement("option");
                option.value = name;
                option.textContent = name;
                return option;
            }));
            await updateProgress();
        } catch (error) { progress.textContent = error.message; }
    };
    const transfer = button("Download starten", () => queueModel({
        url: url.value.trim(), filename: filename.value.trim(), save_path: directory.value,
    }, transfer));
    panel.append(title, note, url, filename, directory,
        button("Freischalten / Verzeichnisse laden", loadDirectories), transfer,
        button("Modellliste aktualisieren", async () => {
            try { await app.refreshMissingModels?.({ silent: true, reloadDefs: true }); }
            catch { progress.textContent = "Bitte die Modellliste in ComfyUI aktualisieren."; }
        }),
        button("Token vergessen", () => { token = ""; progress.textContent = "Token vergessen."; }),
        button("Schließen", () => panel.close()), progress);
    document.body.append(panel);
    panel.showModal();
}

app.registerExtension({
    name: "ComfyUI.AutoModelDownloader.PVC",
    beforeConfigureGraph(data) { models = collectModels(data); },
    afterConfigureGraph() { inject(); },
    setup() {
        const open = button("Modell-Downloads (PVC)", showPanel);
        open.id = "ice-model-download-panel";
        open.style.cssText += ";position:fixed;bottom:12px;right:12px;z-index:1000";
        document.body.append(open);
        let scheduled = false;
        new MutationObserver(() => {
            if (scheduled) return;
            scheduled = true;
            requestAnimationFrame(() => { scheduled = false; inject(); });
        }).observe(document.body, { childList: true, subtree: true });
        setInterval(updateProgress, 2000);
        inject();
    },
});
