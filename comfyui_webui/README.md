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

**Anforderung an Template-Dateien:**  
Die App erwartet dieselbe Node-ID-Konvention wie beim Standard-Template:

| Node | Zweck |
|------|-------|
| `"1"` | Model-Loader (`CheckpointLoaderSimple`, `UNETLoader`, …) |
| `"2"` | Positiver Prompt (`CLIPTextEncode`) |
| `"3"` | Negativer Prompt (`CLIPTextEncode`) |
| `"4"` | Latent-Bild / Größe (`EmptyLatentImage`) |
| `"5"` | Sampler (`KSampler`) |

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
