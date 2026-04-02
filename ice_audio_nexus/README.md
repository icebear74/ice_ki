# ice_audio_nexus

KI-basiertes System zur **automatischen Sprecheridentifikation** in Serien und Filmen.  
Der Scanner analysiert Videos, die Web-UI ermöglicht die manuelle Zuweisung — und jede
Zuweisung trainiert das System für die nächste Episode.

## Workflow: Vom Video zur Datenbank

```
1.  scanner.py ausführen
       │
       ├─ FFmpeg extrahiert Audio (16 kHz Mono WAV)
       ├─ pyannote.audio erkennt Sprechwechsel  →  Segmente (start_ms / end_ms / SPEAKER_xx)
       ├─ pyannote/embedding erzeugt VECTOR(512) pro Segment
       ├─ faster-whisper transkribiert jeden Abschnitt
       └─ Alle Daten landen in MariaDB  →  episode_segments

2.  Web-UI öffnen  (http://localhost:8765)
       │
       ├─ Bibliothek zeigt automatisch alle gescannten Serien/Folgen
       │    (Quelle: episode_segments.video_path aus der DB)
       ├─ Folge auswählen → Video startet, Segmentliste erscheint rechts
       ├─ Klick auf Segment → Video springt zur Stelle, Zuweisung öffnet sich
       └─ Identität zuweisen (neu anlegen oder bestehende wählen)
              │
              └─ Optionales Häkchen „Vektor als neues Sample speichern"
                   → speichert VECTOR(512) in voice_samples
                   → dieser Vektor wird ab sofort für alle neuen Scans genutzt
```

**Wie kommen Vektoren in die Datenbank?**

| Schritt | Was passiert |
|---------|-------------|
| Scanner läuft | `episode_segments.embedding`-Spalte wird mit dem Sprachvektor gefüllt |
| Nutzer weist Identität zu | `episode_segments.identity_id` wird gesetzt |
| Häkchen „Vektor speichern" | Vektor wird in `voice_samples` übernommen → ab jetzt für Auto-Matching aktiv |
| Nächster Scan | System sucht via `VECTOR_DISTANCE` in allen `voice_samples` → automatische Erkennung |

---

## Architektur

```
ice_audio_nexus/
├── setup_env.sh          # Komplettes Python-Setup (Pascal P100/P4 kompatibel)
├── .env.example          # Vorlage für Zugangsdaten
│
├── db/
│   └── database.py       # MariaDB 11.7 – Schema, VECTOR(512) Suche, Library-Query
│
├── processor/
│   └── scanner.py        # FFmpeg + pyannote.audio + faster-whisper → MariaDB
│
└── web_ui/
    ├── api.py             # FastAPI: /api/library, /stream, /api/segments, ...
    └── templates/
        └── index.html    # Interaktive UI: Bibliothek, Videoplayer, Segment-Zuweisung
```

## Multi-Vektor-Identitätssystem

Eine **Identität** (z. B. „Jean-Luc Picard") kann beliebig viele **Voice Samples** besitzen.
Jedes Sample speichert einen eigenen `VECTOR(512)` mit Kontext-Metadaten
(z. B. `TNG Season 1`, `Picard S3E02`).

```
VECTOR_DISTANCE(neuer_vektor, alle voice_samples)
  → kleinste Distanz ermitteln

  dist < MATCH_THRESHOLD   → ✅ Erkannt  – Identität wird automatisch zugewiesen
  dist < SUGGEST_THRESHOLD → ⚠  Vorschlag – Nutzer muss im Web-UI bestätigen
  dist ≥ SUGGEST_THRESHOLD → ❓ Unbekannt – manuell zuweisen
```

Standardwerte: `MATCH_THRESHOLD=0.25`, `SUGGEST_THRESHOLD=0.45` (via `.env` anpassbar).

## Datenbank-Schema

### `episode_segments`
| Spalte             | Typ          | Beschreibung                                           |
|--------------------|--------------|--------------------------------------------------------|
| series_name        | VARCHAR(255) | Serienname (z. B. „The Big Bang Theory")              |
| episode_title      | VARCHAR(255) | Folgenkürzel (z. B. „S01E01")                         |
| video_path         | TEXT         | Absoluter Pfad zur Videodatei – wird von der UI genutzt |
| start_ms / end_ms  | INT          | Zeitstempel in Millisekunden                           |
| speaker_label      | VARCHAR(100) | Temporäres Diarization-Label (SPEAKER_01)              |
| identity_id        | INT FK       | → identities.id (nach Zuweisung)                      |
| matched_sample_id  | INT FK       | → voice_samples.id (welcher Vektor hat gematcht?)     |
| match_distance     | FLOAT        | Cosinus-Distanz des besten Treffers                    |
| is_suggestion      | BOOL         | True = Vorschlag, Nutzerbestätigung ausstehend         |
| transcript         | TEXT         | Whisper-Transkript des Segments                        |

### `voice_samples`
| Spalte       | Typ          | Beschreibung                                    |
|--------------|--------------|-------------------------------------------------|
| identity_id  | INT FK       | → identities.id                                |
| embedding    | VECTOR(512)  | 512-dim Float32 Vektor (für VECTOR_DISTANCE)   |
| context      | VARCHAR(255) | z. B. „TNG Season 1" – Kontext der Aufnahme    |
| is_confirmed | BOOLEAN      | Durch Nutzer manuell bestätigt                  |

## Schnellstart

```bash
# 1. Umgebung einrichten (Pascal GPU: P100/P4 – CUDA 11.8)
cd ice_audio_nexus
chmod +x setup_env.sh && ./setup_env.sh

# 2. Konfiguration anlegen
cp .env.example .env
# → .env editieren: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, VIDEO_DIR, HF_TOKEN

# 3. Web-UI starten (Tabellen werden automatisch angelegt)
source venv/bin/activate
uvicorn web_ui.api:app --host 0.0.0.0 --port 8765

# 4. Video scannen (einmalig pro Episode)
python -m processor.scanner \
    --video /mnt/data/video/serie/The\ Big\ Bang\ Theory/S01/S01E01_Pilot.mkv \
    --series "The Big Bang Theory"
# Episode wird automatisch aus dem Dateinamen erkannt (S01E01)

# 5. Browser öffnen und Sprecher zuweisen
#    http://server-ip:8765
#    → Serie wählen → Folge wählen → Segment anklicken → Identität zuweisen
```

## Hardware-Setup

| GPU          | Aufgabe                         | Env-Variable           |
|--------------|---------------------------------|------------------------|
| Tesla P4 8GB | Speaker Diarization (pyannote)  | `DIARIZATION_DEVICE=cuda:0`  |
| Tesla P100   | Transkription (faster-whisper)  | `TRANSCRIPTION_DEVICE=cuda:1` |

FFmpeg (CUDA) wird für Audio-Extraktion und Video-Streaming verwendet.  
Der `/stream`-Endpunkt nutzt `h264_nvenc` wenn verfügbar, sonst `libx264` als Fallback.

## Web-UI – Funktionen

- **Bibliothek-Dropdown** – zeigt alle gescannten Serien/Staffeln/Folgen direkt aus der DB
  - ✓ grün markiert = bereits gescannt und bereit zur Bearbeitung
- **▶ Video-Player** mit live Speaker-Overlay (Name + Match-Prozent)
- **Klick auf Segment** → Video springt sofort zur passenden Stelle
- **Farbige Timeline** aller Segmente (klickbar)
- **Segment-Sidebar** (unabhängig scrollbar, verschiebt das Video nicht):
  - ✅ Erkannte Sprecher
  - ⚠ Vorschläge (Distanz etwas höher, Bestätigung empfohlen)
  - ❓ Unbekannte Sprecher
- **Zuweisungs-Panel**: Identität wählen/neu anlegen + optional Vektor in DB speichern

