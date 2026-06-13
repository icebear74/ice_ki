# comfyui_webui

Lokale Python-Weboberfläche für:

1. Deutscher Prompt → Übersetzung nach Englisch über **Ollama**
2. Übergabe des übersetzten Prompts an **ComfyUI** zur Bildgenerierung
3. Benutzerverwaltung mit Rollen (admin / user)
4. Template-Freigabe-System: Admins können Workflow-Templates testen und freigeben

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

---

## Benutzerverwaltung & Authentifizierung

### Erster Start (Bootstrap)

Wenn beim ersten Start noch **keine** `data/users.json` existiert, legt die App
automatisch einen Admin-Account an. Das generierte Passwort wird:

1. im Terminal ausgegeben (in deutlich sichtbarer Box)
2. in `data/bootstrap_credentials.txt` gespeichert

**Beispiel-Ausgabe beim ersten Start:**

```
============================================================
  FIRST START – admin account created
  username : admin
  password : abc123XYZ...
  See comfyui_webui/data/bootstrap_credentials.txt
  Delete that file after first login!
============================================================
```

> **Wichtig:** Lösche `data/bootstrap_credentials.txt` nach dem ersten
> Login. Lege anschließend einen eigenen Admin-Account an oder ändere das
> Passwort (manuell in `data/users.json` oder per Admin-UI im nächsten Release).

### Rollen

| Rolle   | Beschreibung |
|---------|-------------|
| `admin` | Vollzugriff: Benutzerverwaltung, Template-Freigabe, alle generieren |
| `user`  | Kann generieren und nur freigegebene Templates sehen |

### Passwort-Speicherung

Passwörter werden **niemals im Klartext gespeichert**.  
Es wird PBKDF2-HMAC-SHA256 mit 600.000 Iterationen und zufälligem Salt verwendet
(Python-Stdlib `hashlib` – keine extra Abhängigkeiten).

### Datei-Speicherort

```
comfyui_webui/
  data/
    users.json               ← Benutzer (wird beim ersten Start angelegt)
    bootstrap_credentials.txt ← Einmaliges Bootstrap-Passwort (löschen!)
    templates.json           ← Template-Registry
    templates/               ← Optional: JSON-Workflow-Dateien
```

Die Dateien in `data/` sind in `.gitignore` eingetragen – sie werden nicht
ins Repository commited.

### Admin-UI

Admins sehen nach dem Login einen zusätzlichen **⚙ Admin**-Tab mit:
- **Template-Verwaltung:** Templates entdecken, freigeben, deaktivieren, löschen
- **Benutzerverwaltung:** Neue Benutzer anlegen, Benutzer deaktivieren/aktivieren

---

## Template-Freigabe-System

### Konzept

1. Admins entdecken oder registrieren Workflow-Templates
2. Nach Test können Templates als **freigegeben** markiert werden
3. Nur freigegebene + aktive Templates sind für normale Benutzer sichtbar

### Template-Quellen

- **`local`**: Manuell in der Admin-UI eingetragen
- **`comfyui`**: Über „ComfyUI-Templates entdecken" von der ComfyUI-Instanz abgerufen
  (nutzt `/api/workflow_templates` falls von ComfyUI bereitgestellt)

### Template-JSON-Dateien

Workflow-Templates können als JSON-Dateien in `data/templates/` abgelegt werden.
Der `filename`-Eintrag im Template-Datensatz zeigt auf diese Datei (relativ zu
`data/templates/`).

> **Wichtig:** Die App benötigt **keine** feste Node-ID-Konvention mehr.  
> Du kannst jeden von ComfyUI exportierten Workflow direkt als JSON-Datei
> ablegen und hochladen – die Analyse erkennt die Rollen automatisch.

---

## Workflow-Analyse und Validierung

### Was wird analysiert?

Beim Hochladen oder Entdecken einer Template-Datei analysiert die App den
Workflow-Graphen automatisch (Modul `workflow_analyzer.py`) und erkennt:

| Rolle | Erkannte Knotentypen |
|-------|----------------------|
| **Sampler** | `KSampler`, `KSamplerAdvanced` |
| **Checkpoint-Loader** | `CheckpointLoaderSimple`, `CheckpointLoader` |
| **UNet-Loader** | `UNETLoader`, `DiffusionModelLoader` |
| **Positiver Prompt** | `CLIPTextEncode`-Knoten im positiven Conditioning-Pfad |
| **Negativer Prompt** | `CLIPTextEncode` oder `ConditioningZeroOut` im negativen Pfad |
| **Latent-Quelle** | `EmptyLatentImage`, `EmptySD3LatentImage`, u. a. |
| **Decoder** | `VAEDecode`, `VAEDecodeTiled` |
| **Output** | `SaveImage`, `PreviewImage` |
| **img2img-Pfade** | `VAEEncode`, `LoadImage` (für zukünftige Unterstützung) |

### Analyse-Ergebnis in der Admin-UI

In der Template-Tabelle zeigt die Spalte **Analyse**:

- **✓ OK** – Workflow ist vollständig verwendbar, keine Warnungen
- **⚠ N Warnung(en)** – verwendbar, aber z. B. mehrdeutiger Sampler, mehrere CLIP-Knoten
- **✗ Nicht verwendbar** – kein Sampler gefunden, Parse-Fehler, o. ä.

Der Tooltip beim Hover zeigt Details zu Warnungen, Fehlern, Sampler- und Loader-Anzahl.

Mit dem **⟳**-Button wird die Analyse für ein bestehendes Template neu ausgeführt
(`GET /api/admin/templates/{name}/analysis`).

### Unterstützte Workflow-Typen

| Workflow-Typ | Unterstützt | Hinweise |
|---|---|---|
| Standard SD 1.x/2.x/XL | ✓ | CheckpointLoaderSimple + KSampler |
| FLUX / UNet-basiert | ✓ | UNETLoader/DiffusionModelLoader erkannt |
| Dual-CLIP (FLUX) | ✓ | Beide CLIPTextEncode-Knoten werden befüllt |
| `ConditioningZeroOut` negativ | ✓ | Negativ-Prompt wird nicht überschrieben |
| Mehrstufige Sampler-Pipelines | ⚠ | Ausgabe-Sampler (→ VAEDecode) wird bevorzugt |
| img2img / Inpainting | ⚠ | Strukturen erkannt, aber Parameter noch nicht vollständig injizierbar |
| Komplexe Conditioning-Graphen | ⚠ | Warnung wenn kein CLIPTextEncode erreichbar |

### Ausführungsverhalten bei importierten Templates

- Importierte Workflow-JSONs werden jetzt **strukturtreu** ausgeführt:
  - Die Pipeline-Struktur (Loader, Wrapper wie `ModelSamplingAuraFlow`, CLIP-/VAE-Ketten) bleibt unverändert.
  - Prompt-Injektion erfolgt nur auf den vom Graph-Analyzer erkannten positiven/negativen Pfaden.
- Für importierte Templates werden Sampler-/Latent-Defaults aus dem Template standardmäßig beibehalten.
  WebUI-Werte werden nur dann in diese Felder geschrieben, wenn sie explizit vom WebUI-Standard abweichen
  (z. B. geänderte Steps statt Default 30).
- Der Seed wird weiterhin pro Lauf gesetzt (fester Seed oder zufällig bei `-1`).
- Bei nicht eindeutigem positiven Prompt-Ziel wird die Generierung mit einem klaren Fehler abgebrochen
  statt stillschweigend den falschen Knoten zu überschreiben.

### Bekannte Einschränkungen

- **img2img**: Der Analyse-Code erkennt `VAEEncode`- und `LoadImage`-Pfade und meldet
  sie als „möglicher img2img-Workflow". Die tatsächliche Bildübergabe ist noch
  **nicht implementiert** – die Architektur ist jedoch darauf vorbereitet.
- Sehr ungewöhnliche Node-Typen für Sampler oder Loader (custom nodes) werden
  unter Umständen nicht erkannt.
- Bei Workflows mit mehr als einem Sampler wählt die WebUI den Sampler,
  dessen Ausgabe direkt in einen `VAEDecode`-Knoten fließt. Ist das nicht
  eindeutig, wird der erste gefundene Sampler verwendet (mit Warnung).

### Logging

Jede Generierungsanfrage wird in `data/generation.log` protokolliert (rotierend,
max. 10 MB, 5 Backups):

```
2024-01-15 12:00:00.123 | INFO | REQUEST id=a1b2c3d4 template='default' checkpoint='v1-5.safetensors' ...
2024-01-15 12:00:00.456 | INFO | ANALYSIS id=a1b2c3d4 usable=True sampler='5' model_type=checkpoint ...
2024-01-15 12:00:01.789 | DEBUG | SET_POSITIVE id=a1b2c3d4 clip_node='2' text='a futuristic cityscape...'
2024-01-15 12:00:01.791 | INFO | QUEUED id=a1b2c3d4 prompt_id=abc-123-...
```

Protokollierte Ereignisse:
- `REQUEST` – eingehende Generierungsparameter
- `TRANSLATED` – übersetzte Prompts
- `TEMPLATE` – gewähltes Template
- `ANALYSIS` – Graph-Analyse-Ergebnis (Rollen, Warnungen)
- `OVERRIDE_POLICY` – Modus für Mutationen (`full` bei Default-Template, sonst `preserve_imported`)
- `SET_POSITIVE` / `SET_NEGATIVE` / `SET_SAMPLER` / `SET_LATENT` / `SET_MODEL` – Mutationen
- `QUEUED` – erfolgreiche Übergabe an ComfyUI
- `COMFYUI_REJECT` / `COMFYUI_UNREACHABLE` – ComfyUI-Fehler

---

## Vorbereitung für Image2Image

Die Architektur ist für spätere img2img-Unterstützung vorbereitet:

- `workflow_analyzer.py` erkennt bereits `VAEEncode`, `LoadImage` und
  `VAEEncodeTiled`-Knoten und setzt `is_potentially_img2img = True`
- Das Analyse-Ergebnis wird in den Template-Metadaten gespeichert
- Der API-Endpunkt `GET /api/admin/templates/{name}/analysis` liefert vollständige
  Graph-Metadaten, die für eine spätere img2img-Implementierung genutzt werden können
- Zukünftig: `GenerateRequest` um `init_image`-Feld erweitern, `_build_workflow()`
  erkennt `primary_latent_id` bereits als `VAEEncode`-Quelle und kann dort das
  Eingabebild übergeben

---

## HTTPS-Empfehlung

### Warum kein automatisches HTTPS in der App?

Für eine **lokale, selbst gehostete** Anwendung empfehlen wir **keinen
eingebauten HTTPS-Terminator** direkt in der FastAPI-App, da:

- Selbstsignierte Zertifikate im Browser Warnungen erzeugen
- Zertifikatserneuerung (Let's Encrypt) von außen erreichbar sein muss
- Ein Reverse Proxy flexibler und sicherer ist

### Empfohlener Ansatz: Nginx oder Caddy als Reverse Proxy

**Option A – Caddy (einfachstes HTTPS mit automatischem Zertifikat):**

```bash
# Caddyfile
yourdomain.local {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy übernimmt automatisch HTTPS, falls die Domain öffentlich erreichbar ist.

**Option B – nginx mit selbstsigniertem Zertifikat für reines LAN:**

```bash
# Zertifikat erzeugen
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"

# nginx.conf (Auszug)
server {
    listen 443 ssl;
    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**Option C – uvicorn mit SSL direkt (für schnelle Tests):**

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"

uvicorn main:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

> Hinweis: Bei selbstsignierten Zertifikaten zeigt der Browser eine
> Sicherheitswarnung. Diese kannst du für lokale Entwicklung akzeptieren.
> Für Produktionsbetrieb im LAN empfehlen wir Option A oder B.

---

## Funktionen

- Eingabefeld für deutschen Prompt
- Eingabefeld für Negative Prompt
- Ollama-Modell auswählbar/eingebbar
- Verfügbare Ollama-Modelle abrufen
- Bildparameter einstellbar: Steps, CFG, Seed, Width, Height, Sampler, Scheduler, Anzahl Bilder
- ComfyUI-Checkpoint auswählbar/eingebbar
- **Workflow-Template auswählbar** (nur freigegebene Templates für normale Benutzer)
- Anzeige des übersetzten Prompts vor der Generierung
- Text in Anführungszeichen wie `"Hallo Welt"` bleibt bei der Übersetzung als exakter sichtbarer Schriftzug erhalten
- Anzeige der generierten Bilder in der UI
- **Login-/Logout-Funktion**
- **Admin-Panel:** Template-Freigabe + Benutzerverwaltung

## Hinweise zu ComfyUI-Integrationspunkten

Die App nutzt bewusst einfache, robuste Standard-Endpunkte:

- Prompt senden: `POST /prompt`
- History abrufen: `GET /history/{prompt_id}`
- Bild abrufen: `GET /view`
- Checkpoints bevorzugt über `GET /object_info/CheckpointLoaderSimple`
- Fallback für Checkpoints: `GET /models` (falls verfügbar)
- Template-Entdeckung: `GET /api/workflow_templates` (falls von ComfyUI bereitgestellt)

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
