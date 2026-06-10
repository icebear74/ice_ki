# comfyui_webui

Lokale Python-Weboberfläche für:

1. Deutscher Prompt → Übersetzung nach Englisch über **Ollama**
2. Übergabe des übersetzten Prompts an **ComfyUI** zur Bildgenerierung

## Voraussetzungen

- Lokales **Ollama** (Standard: `http://127.0.0.1:11434`)
- Lokales **ComfyUI** mit API (Standard: `http://127.0.0.1:8188`)
- Python 3.10+

## Setup (venv)

```bash
cd comfyui_webui
./setup_env.sh
source venv/bin/activate
```

## Anwendung starten

```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

Dann im Browser öffnen:

- `http://127.0.0.1:8080`

## Konfiguration per Umgebungsvariablen

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export COMFYUI_BASE_URL="http://127.0.0.1:8188"
```

## Funktionen

- Eingabefeld für deutschen Prompt
- Eingabefeld für Negative Prompt
- Ollama-Modell auswählbar/eingebbar
- Verfügbare Ollama-Modelle abrufen
- Bildparameter einstellbar: Steps, CFG, Seed, Width, Height, Sampler, Scheduler, Anzahl Bilder
- ComfyUI-Checkpoint auswählbar/eingebbar
- Anzeige des übersetzten Prompts vor der Generierung
- Anzeige der generierten Bilder in der UI

## Hinweise zu ComfyUI-Integrationspunkten

Die App nutzt bewusst einfache, robuste Standard-Endpunkte:

- Prompt senden: `POST /prompt`
- History abrufen: `GET /history/{prompt_id}`
- Bild abrufen: `GET /view`
- Checkpoints bevorzugt über `GET /object_info/CheckpointLoaderSimple`
- Fallback für Checkpoints: `GET /models` (falls verfügbar)

Da ComfyUI-Installationen (inkl. Custom Nodes) unterschiedlich sein können, kann der direkte Checkpoint-Abruf je nach Setup variieren. Falls kein Endpunkt verfügbar ist, bleibt die manuelle Eingabe des Checkpoint-Namens möglich.

## Workflow-Template

Die App verwendet ein einfaches internes Standard-Workflow-Template mit diesen Knoten:

- `CheckpointLoaderSimple`
- `CLIPTextEncode` (positiv/negativ)
- `EmptyLatentImage`
- `KSampler`
- `VAEDecode`
- `SaveImage`

Wenn dein ComfyUI-Setup andere Knoten/Parameter benötigt, passe `workflow_template.json` an.  
Die App lädt dieses JSON automatisch (Fallback ist das interne Default-Template in `main.py`).

---

## Ollama auf eine bestimmte Grafikkarte begrenzen

Ollama wählt standardmäßig die erste verfügbare GPU. Mit der Umgebungsvariable
`CUDA_VISIBLE_DEVICES` kannst du den Prozess auf eine bestimmte Karte einschränken.

### GPU-Index herausfinden

```bash
# nvidia-smi zeigt alle GPUs mit Index 0, 1, 2, …
nvidia-smi -L
```

Beispielausgabe:
```
GPU 0: NVIDIA GeForce RTX 3090 (UUID: …)
GPU 1: NVIDIA Tesla P100 (UUID: …)
```

### Ollama auf GPU 1 begrenzen

```bash
CUDA_VISIBLE_DEVICES=1 ollama serve
```

Oder dauerhaft als systemd-Service-Override:

```bash
sudo systemctl edit ollama
```

Inhalt (anpassen):
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
```

Dann neu starten:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Mehrere GPUs zulassen (z. B. 0 und 2)

```bash
CUDA_VISIBLE_DEVICES=0,2 ollama serve
```

### GPU vollständig deaktivieren (CPU-only)

```bash
CUDA_VISIBLE_DEVICES="" ollama serve
```

> **Hinweis:** Die Umgebungsvariable muss gesetzt sein, **bevor** Ollama startet.
> Wird sie nachträglich geändert, muss Ollama neugestartet werden.
