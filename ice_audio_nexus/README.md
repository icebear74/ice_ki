# ice_audio_nexus

**KI-basierte Video-Audio-Analyse und Personenidentifikation**

Ein System, das Filme und Serien analysiert, Sprecher (inkl. Synchronsprecher in verschiedenen Rollen) sauber trennt und durch menschliches Feedback lernt.

---

## Kernfunktionalitäten

### 1. Sprecher-Diarization & -Identifikation
- **Two-Pass-Verfahren**: PyAnnote erkennt Sprecherwechsel auf Millisekunden-Ebene, anschließend werden hochpräzise 512-dimensionale Embedding-Vektoren extrahiert.
- **Kontext-bewusste Zuordnung**: Eine Stimme (Synchronsprecher) kann mehrere Charakter-Identitäten in unterschiedlichen Serien haben. Die Zuordnung erfolgt immer im Kontext der aktuellen Serie.
- **Lernzyklus**: Nach dem manuellen Labeling im Webinterface berechnet das System optimierte Master-Vektoren für jede Identität.

### 2. Hardware-Setup
| Aufgabe | Gerät |
|---|---|
| FFmpeg Audio-Extraktion | CUDA (h264_nvenc) |
| Speaker Diarization (PyAnnote) | Tesla P4 (`cuda:0`) |
| Embedding-Extraktion (PyAnnote) | Tesla P4 (`cuda:0`) |
| Transkription (Faster-Whisper) | Tesla P100 (`cuda:1`) |

### 3. Datenbank (MariaDB 11.7)
- `voice_profiles`: VECTOR(512) Float32-Embeddings (~2KB/Eintrag)
- `identities`: Verknüpft Stimme mit Charakter und Serien-Kontext
- `episode_segments`: Timeline der Sprecher pro Episode (Millisekunden-Timestamps)

---

## Schnellstart

### 1. Umgebung einrichten
```bash
cd ice_audio_nexus
./setup_env.sh
```

### 2. Datenbank konfigurieren
```bash
cp .env.example .env
nano .env  # DB-Zugangsdaten eintragen
```

MariaDB-Schema manuell anlegen (optional – wird auch automatisch beim Start angelegt):
```bash
mariadb -u root -p < db/schema.sql
```

### 3. Episode scannen (Tesla P4 + P100)
```bash
source venv/bin/activate
python processor/scanner.py \
    --video /pfad/zur/episode.mkv \
    --source "The Walking Dead" \
    --episode "S01E01 - Days Gone Bye"
```

Optionen:
```
--diarization-device cuda:0   # Tesla P4 (Standard)
--whisper-device cuda:1       # Tesla P100 (Standard)
--whisper-model large-v3      # Whisper-Modellgröße
--language de                 # Sprache (Standard: de)
--similarity-threshold 0.85   # Mindest-Ähnlichkeit für Auto-Zuordnung
--skip-transcription          # Nur Diarization, keine Transkription
--no-cuda-ffmpeg              # FFmpeg ohne CUDA-Beschleunigung
```

### 4. Webinterface starten
```bash
python web_ui/api.py
# Browser: http://localhost:8000
```

---

## Verzeichnisstruktur

```
ice_audio_nexus/
├── setup_env.sh           # Einrichtungs-Skript (Python 3.12 venv)
├── .env.example           # Vorlage für Datenbank-Konfiguration
├── .gitignore
├── README.md
├── db/
│   ├── __init__.py
│   ├── schema.sql         # MariaDB 11.7 Schema (manuell oder automatisch)
│   └── database.py        # Verbindung, auto-init, Vektorfunktionen
├── processor/
│   ├── __init__.py
│   └── scanner.py         # FFmpeg-Extraktion, Diarization, Transkription
└── web_ui/
    ├── __init__.py
    ├── api.py             # FastAPI-Backend (Streaming, REST, WebSocket)
    └── templates/
        └── index.html     # HTML5-Player mit interaktiver Sprecher-Timeline
```

---

## Workflow

```
1. scanner.py analysiert Episode
       │
       ▼
2. Segmente in MariaDB (raw_speaker_id: SPEAKER_00, SPEAKER_01, …)
       │
       ▼
3. Webinterface: Video streamen, Sprecher-Timeline anzeigen
       │
       ▼
4. "Jump-to-Speaker": Klick → Video springt zur ersten Szene des Sprechers
       │
       ▼
5. "Rename/Assign": Eingabe Name + Serien-Kontext → alle Segmente aktualisiert
       │
       ▼
6. "Confirm": Bestätigung → is_confirmed = TRUE
       │
       ▼
7. "Episode finalisieren": Master-Vektoren neu berechnet → bessere Erkennung
```

---

## Synchronsprecher-Handling

Das System trennt **physische Stimme** von **Charakter-Identität**:

```
voice_profiles (Stimm-Vektor)
    └─── identities.character_name = "Daryl Dixon"
    │    identities.series_name    = "The Walking Dead"
    │
    └─── identities.character_name = "Charakter Y"
         identities.series_name    = "Serie B"
```

Wenn eine bekannte Stimme in einem neuen Kontext auftaucht, erstellt das System eine neue Identität (mit neuem Serien-Kontext), verweist aber auf denselben `voice_id`.

---

## Datenbank-Schema

```sql
-- Biometrischer Stimm-Fingerabdruck
voice_profiles (id, voice_vector VECTOR(512), sample_count, is_confirmed)

-- Charakter in Serien-Kontext  
identities (id, voice_id, character_name, series_name, sync_actor_name)

-- Timeline pro Episode
episode_segments (id, series_name, episode_title, video_path,
                  start_ms, end_ms, raw_speaker_id, identity_id,
                  transcript, confidence, is_confirmed)
```

---

## API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| GET | `/` | Webinterface |
| GET | `/api/episodes` | Alle bekannten Episoden |
| GET | `/api/segments?series_name=&episode_title=` | Segmente einer Episode |
| GET | `/api/identities` | Alle Identitäten |
| POST | `/api/assign` | Sprecher benennen/zuweisen |
| POST | `/api/confirm` | Zuordnung bestätigen |
| POST | `/api/finalize` | Episode finalisieren (Master-Vektoren) |
| GET | `/api/stream?video_path=&seek=` | Video-Stream (FFmpeg CUDA) |
| WS | `/ws` | WebSocket für Echtzeit-Updates |

---

## Voraussetzungen

- Python 3.12
- MariaDB 11.7 (mit VECTOR-Support)
- FFmpeg (mit CUDA-Support empfohlen)
- NVIDIA-GPUs (Tesla P4 + P100 empfohlen)
- HuggingFace-Token für PyAnnote-Modelle (`HF_TOKEN` in `.env`)
