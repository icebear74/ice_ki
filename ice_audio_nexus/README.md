# ice_audio_nexus

KI-basiertes System zur Sprecheridentifikation in Videos mit **Multi-Vektor-Identitätssystem**.

## Architektur

```
ice_audio_nexus/
├── setup_env.sh          # Python 3.12 venv + Abhängigkeiten
├── .env.example          # Vorlage für Zugangsdaten
├── .gitignore
│
├── db/
│   └── database.py       # MariaDB 11.7 – Auto-Schema, VECTOR(512) Suche
│
├── processor/
│   └── scanner.py        # FFmpeg CUDA + PyAnnote + Whisper → MariaDB
│
└── web_ui/
    ├── api.py             # FastAPI Backend
    └── templates/
        └── index.html    # Interaktives Webinterface
```

## Multi-Vektor-Identitätssystem

Eine **Identität** (z. B. "Jean-Luc Picard") kann beliebig viele **Voice Samples** besitzen.
Jedes Sample speichert einen eigenen `VECTOR(512)` Embedding mit Kontext-Metadaten
(z. B. `TNG Season 1`, `Picard S3E02`).

### Erkennungslogik

```
VECTOR_DISTANCE(new_embedding, alle gespeicherten Samples)
  → kleinste Distanz ermitteln
  
  dist < MATCH_THRESHOLD   → ✅ Erkannt – Identität zugewiesen
  dist < SUGGEST_THRESHOLD → ⚠ Vorschlag – Nutzer muss bestätigen
                              (empfohlen: neues Sample anlegen für diese Ära)
  dist ≥ SUGGEST_THRESHOLD → ❓ Unbekannt – neuer Sprecher
```

Standardwerte: `MATCH_THRESHOLD=0.25`, `SUGGEST_THRESHOLD=0.45` (via `.env` anpassbar).

### Warum mehrere Vektoren?

Jean-Luc Picard in **Star Trek TNG (1990)** klingt messbar anders als in  
**Star Trek: Picard (2022)**. Mit einem einzigen "Master-Vektor" würde das System  
die Alterung nicht korrekt abbilden. Durch mehrere Vektoren pro Identität kann das  
System die Stimme über Jahrzehnte robust erkennen.

## Datenbank-Schema

### `identities`
| Spalte      | Typ          | Beschreibung                          |
|-------------|--------------|---------------------------------------|
| id          | INT PK       | Auto-Increment                        |
| name        | VARCHAR(255) | z. B. "Jean-Luc Picard" (unique)     |
| description | TEXT         | Optionale Beschreibung                |
| created_at  | TIMESTAMP    | Erstellungszeitpunkt                  |

### `voice_samples`
| Spalte       | Typ          | Beschreibung                           |
|--------------|--------------|----------------------------------------|
| id           | INT PK       | Auto-Increment                         |
| identity_id  | INT FK       | → identities.id                       |
| embedding    | VECTOR(512)  | 512-dim Float32 Vektor                 |
| context      | VARCHAR(255) | z. B. "TNG Season 1", "Picard S3E02"  |
| is_confirmed | BOOLEAN      | Durch Nutzer bestätigt?                |
| created_at   | TIMESTAMP    | Zeitpunkt der Speicherung              |

### `episode_segments`
| Spalte             | Typ    | Beschreibung                                           |
|--------------------|--------|--------------------------------------------------------|
| id                 | INT PK | Auto-Increment                                         |
| series_name        | TEXT   | Serienname                                             |
| episode_title      | TEXT   | Folgenname                                             |
| video_path         | TEXT   | Pfad zur Videodatei                                    |
| start_ms / end_ms  | INT    | Zeitstempel in Millisekunden                           |
| speaker_label      | TEXT   | Temporäres Diarization-Label (SPEAKER_01)              |
| identity_id        | INT FK | → identities.id (nach Zuordnung)                      |
| matched_sample_id  | INT FK | → voice_samples.id (welcher Vektor hat gematcht?)     |
| match_distance     | FLOAT  | Cosinus-Distanz des besten Treffers                    |
| is_suggestion      | BOOL   | True = Vorschlag, Nutzerbestätigung ausstehend         |
| transcript         | TEXT   | Whisper-Transkript des Segments                        |

## Schnellstart

```bash
# 1. Python-Umgebung einrichten
chmod +x setup_env.sh && ./setup_env.sh

# 2. Konfiguration anlegen
cp .env.example .env
# → .env editieren: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, VIDEO_DIR

# 3. Web-Interface starten (Tabellen werden automatisch angelegt)
source venv/bin/activate
uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8765

# 4. Browser öffnen: http://localhost:8765

# 5. Video scannen (im Hintergrund, auf Tesla P4/P100)
python -m processor.scanner \
    --video /pfad/zur/episode.mkv \
    --series "Star Trek TNG" \
    --episode "The Inner Light"
```

## Hardware-Setup

| GPU          | Aufgabe                         | Variable               |
|--------------|---------------------------------|------------------------|
| Tesla P4 8GB | Speaker Diarization (PyAnnote)  | `DIARIZATION_DEVICE`   |
| Tesla P100   | Transkription (Faster-Whisper)  | `TRANSCRIPTION_DEVICE` |

FFmpeg v8 (selbst kompiliert mit CUDA) wird für Audio-Extraktion und Video-Streaming genutzt.

## Webinterface – Funktionen

- **▶ Video-Player** mit HLS-kompatiblem Stream via FFmpeg CUDA
- **Speaker-Overlay** im Video: zeigt live den Sprecher + "Erkannt via TNG-Sample, 95% Match"
- **Farbige Timeline** aller Segmente (klickbar zum Springen)
- **Segment-Sidebar**:
  - ✅ Erkannte Sprecher (grün)
  - ⚠ Vorschläge – Distanz war etwas höher (orange, Alterungsschutz)
  - ❓ Unbekannte Sprecher (rot)
- **Zuweisungs-Panel**: Klick auf Segment → Identität wählen/neu anlegen + optional als neuen Vektor speichern
- **Identitäten-Tab**: Übersicht aller Personen mit Vektor-Anzahl
