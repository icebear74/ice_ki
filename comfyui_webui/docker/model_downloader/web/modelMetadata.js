export function collectModels(workflow) {
    const models = [];
    const seen = new Set();
    const add = (model) => {
        if (!model || typeof model.name !== "string" ||
            typeof model.directory !== "string" || typeof model.url !== "string") return;
        const key = `${model.directory}/${model.name}`;
        if (!seen.has(key)) {
            seen.add(key);
            models.push({ filename: model.name, save_path: model.directory, url: model.url });
        }
    };
    const scan = (graph) => {
        if (!graph || typeof graph !== "object") return;
        (Array.isArray(graph.models) ? graph.models : []).forEach(add);
        (Array.isArray(graph.nodes) ? graph.nodes : []).forEach((node) => {
            const attached = node?.properties?.models;
            (Array.isArray(attached) ? attached : []).forEach(add);
        });
        const subgraphs = graph.definitions?.subgraphs;
        (Array.isArray(subgraphs) ? subgraphs : []).forEach(scan);
    };
    scan(workflow);
    return models;
}

export function modelForRow(row, models) {
    const names = [...row.querySelectorAll("[title]")].map((el) => el.getAttribute("title"));
    const matches = models.filter((model) => names.includes(model.filename));
    const text = row.textContent;
    const byDirectory = matches.filter((model) => text.includes(model.save_path));
    if (byDirectory.length === 1) return byDirectory[0];
    if (matches.length === 1) return matches[0];
    // Older ComfyUI dialogs embed "directory / filename" with the URL as title.
    for (const element of row.querySelectorAll("[title]")) {
        const parts = (element.textContent || "").trim().split("/").map((part) => part.trim());
        const url = element.getAttribute("title") || "";
        if (parts.length === 2 && url.startsWith("https://")) {
            return { save_path: parts[0], filename: parts[1], url };
        }
    }
    return null;
}
