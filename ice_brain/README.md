# ice_brain

Lokaler AI-Assistent mit RAG, Memory, Wikipedia-Cache und Multi-Modell-Routing.

## Architektur

```
Frontends (WebUI, curl, Telegram, …)
    │
    ▼  OpenAI-kompatible API  POST /v1/chat/completions
    │
┌───────────────────────────────────────────────────┐
│  FastAPI Orchestrator  (server.py)                 │
│  1. User-Memory laden         (MariaDB)            │
│  2. Intent-Klassifikation     (Router, P4)         │
│  3. Wikipedia-Tool            (Cache + Live-API)   │
│  4. RAG-Context aufbauen      (vorbereitet)        │
│  5. Haupt-LLM antworten       (P100)               │
│  6. Memory-Extraktion         (async, still)       │
│  7. Enrichment-Loop           (async, alle 30 Min) │
└───────────────────────────────────────────────────┘
       │                │               │
       ▼                ▼               ▼
    P4 GPU          P100 GPU        MariaDB 11.8 LTS
    Router-LLM      Main-LLM       VECTOR + Memory
    Qwen3-4B        Qwen3-8B
```

### Feature-Status

| Feature | Status |
|---|---|
| OpenAI-kompatibler Endpunkt | ✅ aktiv |
| Haupt-LLM (P100) – Qwen3-8B | ✅ aktiv |
| Intent-Router (P4) – Qwen3-4B | ✅ aktiv |
| Auto-Download von GGUF-Modellen | ✅ aktiv |
| MariaDB 11.8 LTS + VECTOR-Typ | ✅ aktiv |
| User-Memory (Extraktion + Recall) | ✅ aktiv |
| Fuzzy-Deduplizierung (difflib) | ✅ aktiv |
| Wikipedia-Tool mit Cache | ✅ aktiv |
| Enrichment-Loop (Background) | ✅ aktiv |
| Conversation-Log (DB) | ✅ aktiv |
| Test-WebUI | ✅ aktiv |
| RAG / Vektorsuche | 🔲 Phase 2 |
| Tool-Calls (Wetter, Kalender) | 🔲 Phase 5 |

---

## Setup

### Voraussetzungen

- **MariaDB 11.8 LTS** (Mindestvoraussetzung – nativer VECTOR-Datentyp + HNSW-Index)
- Python 3.11+
- NVIDIA GPU(s) mit CUDA-Support

### 1 – MariaDB User und Datenbank vorbereiten

```sql
-- Als MariaDB-Root ausführen:
CREATE USER 'ice_brain'@'localhost' IDENTIFIED BY 'DEIN_PASSWORT';
GRANT ALL PRIVILEGES ON ice_brain.* TO 'ice_brain'@'localhost';
FLUSH PRIVILEGES;
```

Der Server legt die Datenbank und alle Tabellen **automatisch** beim ersten Start an.

### 2 – config.py erstellen

```bash
cd ice_brain/
cp config.py.example config.py
# Modell-Pfade, MariaDB-Passwort und ggf. Port anpassen
```

### 3 – GGUF-Modelle

| Rolle | Modell | GPU |
|---|---|---|
| Router | Qwen3-4B Q4_K_M | P4 (GPU 1) |
| Main-LLM | Qwen3-8B Q4_K_M | P100 (GPU 0) |

**Auto-Download:** Wenn `hf_repo` und `hf_file` in der `config.py` gesetzt sind und die GGUF-Datei unter `path` nicht existiert, lädt der Server die Modelle beim ersten Start **automatisch** von HuggingFace herunter:

```python
# Auszug aus config.py.example:
MODELS = {
    'router': {
        'path': '/models/qwen3-4b-q4_k_m.gguf',
        'hf_repo': 'Qwen/Qwen3-4B-GGUF',
        'hf_file': 'qwen3-4b-q4_k_m.gguf',
        ...
    },
    ...
}
```

Manuell mit `huggingface-cli`:
```bash
huggingface-cli download Qwen/Qwen3-4B-GGUF qwen3-4b-q4_k_m.gguf --local-dir /models
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir /models
```

### 4 – Requirements installieren

```bash
pip install -r ice_brain/requirements.txt
```

`llama-cpp-python` mit CUDA-Support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-binary llama-cpp-python --no-cache-dir
```

### 5 – Server starten

```bash
cd ice_brain/
python server.py
# oder:
uvicorn server:app --host 0.0.0.0 --port 8000
```

Browser öffnen: **http://localhost:8000**

---

## Features

### Auto-Download von Modellen

Wenn die GGUF-Datei unter `path` nicht vorhanden ist und `hf_repo` + `hf_file` in der Config gesetzt sind, wird die Datei automatisch von HuggingFace heruntergeladen. Benötigt `huggingface-hub` (ist in `requirements.txt`).

### Wikipedia-Tool mit Cache

Das Modul `tools/wikipedia.py` fragt die deutsche Wikipedia REST API ab und speichert Ergebnisse in der MariaDB-Tabelle `wiki_cache`.

- `wiki_search(query, limit=3)` – Suche + Zusammenfassungen
- `wiki_summary(title)` – Einzelartikel
- `wiki_refresh(title)` – Cache-Eintrag invalidieren und neu laden

Die `keywords`-Spalte in `wiki_cache` enthält lesbare Stichpunkte (z. B. "Espresso, Kaffee, Zubereitung, Crema"), damit Einträge manuell in der DB gefunden und bei Bedarf gelöscht werden können.

### Enrichment-Loop (Background)

Der Worker `workers/enrichment.py` läuft alle 30 Minuten als asyncio-Background-Task und reichert unveranreicherte User-Memory-Einträge (Kategorien: preference, hobby, personal, experience) mit Wikipedia-Wissen an:

1. Main-LLM generiert passende Suchbegriffe
2. `wiki_search()` wird pro Begriff aufgerufen (nutzt den Cache)
3. Treffer werden in `memory_knowledge_link` verknüpft
4. Keywords werden vom Main-LLM extrahiert und in `wiki_cache.keywords` gespeichert

Der Loop überspringt den Lauf wenn das Main-LLM gerade für Nutzergespräche belegt ist.

### Fuzzy-Deduplizierung

`_find_similar()` in `db/memory.py` verwendet jetzt zweistufiges Matching:
1. Schnelles Wort-Matching (wie bisher)
2. Fuzzy-Vergleich via `difflib.SequenceMatcher` (Schwellwert 0.75)

Das verhindert doppelte Einträge bei Tippfehlern (z. B. "Phantasieland" → wird als "Phantasialand" erkannt und aktualisiert statt neu eingefügt).

---

## API

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "main",
    "messages": [{"role": "user", "content": "Hallo!"}]
  }'
```

### Health

```bash
curl http://localhost:8000/health
```

---

## Verzeichnisstruktur

```
ice_brain/
├── README.md
├── requirements.txt
├── config.py.example        ← Template; als config.py kopieren
├── server.py                ← FastAPI Entry Point
├── llm_manager.py           ← GGUF-Modelle laden + Auto-Download
├── router.py                ← Intent-Klassifikation
├── models.py                ← Pydantic-Schemas
├── db/
│   ├── __init__.py
│   ├── connection.py        ← MariaDB Pool + Auto-Init
│   ├── memory.py            ← User-Memory (Fuzzy-Dedup, Extraktion via Main-LLM)
│   └── schema.sql           ← Tabellen (MariaDB 11.8, VECTOR(768))
├── tools/
│   ├── __init__.py
│   ├── dummy_weather.py     ← Beispiel-Tool (Dummy)
│   └── wikipedia.py         ← Wikipedia REST API + MariaDB-Cache
├── workers/
│   ├── __init__.py
│   └── enrichment.py        ← Background Knowledge Enrichment Loop
└── web/
    └── index.html           ← Test-WebUI (kein Framework)
```
