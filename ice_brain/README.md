# ice_brain

Lokaler AI-Assistent mit RAG, Memory und Multi-Modell-Routing – Phase 1 Grundgerüst.

## Architektur

```
Frontends (WebUI, curl, Telegram, …)
    │
    ▼  OpenAI-kompatible API  POST /v1/chat/completions
    │
┌───────────────────────────────────────────┐
│  FastAPI Orchestrator  (server.py)         │
│  1. User-Memory laden       (MySQL)        │
│  2. Intent-Klassifikation   (Router, P4)   │
│  3. Tool-Calls ausführen    (vorbereitet)  │
│  4. RAG-Context aufbauen    (vorbereitet)  │
│  5. Haupt-LLM antworten     (P100)         │
│  6. Memory-Extraktion       (async, still) │
└───────────────────────────────────────────┘
       │                │               │
       ▼                ▼               ▼
    P4 GPU          P100 GPU        MySQL 8.4
    Router-LLM      Main-LLM       Vektoren + Memory
    (3 B, GGUF)     (14 B, GGUF)
```

### Phase 1 – was aktiv ist

| Feature | Status |
|---|---|
| OpenAI-kompatibler Endpunkt | ✅ aktiv |
| Haupt-LLM (P100) | ✅ aktiv |
| Intent-Router (P4) | ✅ klassifiziert + loggt |
| MySQL-Tabellen Auto-Init | ✅ aktiv |
| Conversation-Log (DB) | ✅ aktiv |
| Test-WebUI | ✅ aktiv |
| RAG / Vektorsuche | 🔲 Phase 2 |
| Memory lesen/schreiben | 🔲 Phase 3 |
| Tool-Calls | 🔲 Phase 5 |
| Embeddings | 🔲 Phase 2 |

---

## Setup

### 1 – MySQL User und Datenbank vorbereiten

```sql
-- Als MySQL-Root ausführen:
CREATE USER 'ice_brain'@'localhost' IDENTIFIED BY 'DEIN_PASSWORT';
GRANT ALL PRIVILEGES ON ice_brain.* TO 'ice_brain'@'localhost';
FLUSH PRIVILEGES;
```

Der Server legt die Datenbank und alle Tabellen **automatisch** beim ersten Start an.

### 2 – config.py erstellen

```bash
cd ice_brain/
cp config.py.example config.py
# Modell-Pfade, MySQL-Passwort und ggf. Port anpassen
```

### 3 – GGUF-Modelle herunterladen

| Rolle | Empfehlung | Link |
|---|---|---|
| Router (P4, 8 GB) | Qwen2.5-3B-Instruct Q4_K_M | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| Main-LLM (P100, 16 GB) | DeepSeek-R1-Distill-Qwen-14B Q4_K_M | [HuggingFace](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF) |

```bash
# Beispiel mit huggingface-cli:
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    qwen2.5-3b-instruct-q4_k_m.gguf --local-dir /models
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF \
    DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf --local-dir /models
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

Antwort (OpenAI-kompatibel):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "main",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hallo! Wie kann ich helfen?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "router_intent": "general"
}
```

Das Feld `router_intent` ist eine ice_brain-Erweiterung und zeigt den erkannten Intent an.

### Health

```bash
curl http://localhost:8000/health
```

---

## Roadmap

| Phase | Feature |
|---|---|
| **1** | ✅ Grundgerüst: FastAPI, llama-cpp-python, MySQL Auto-Init, WebUI |
| **2** | RAG: Embedding-Modell, Vektorspalten, Wikipedia-Import |
| **3** | User-Memory: Extraktion nach jedem Turn, Recall bei relevanten Fragen |
| **4** | Global-Memory: Kuratierter Wissensspeicher, Promotion aus User-Memory |
| **5** | Tool-Calls: Wetter, Suche, Kalender, Home-Automation |
| **6** | Telegram-Frontend, Multi-User-Support |
| **7** | Fine-Tuning-Pipeline auf eigenen Conversations |

---

## Verzeichnisstruktur

```
ice_brain/
├── README.md
├── requirements.txt
├── config.py.example        ← Template; als config.py kopieren
├── server.py                ← FastAPI Entry Point
├── llm_manager.py           ← GGUF-Modelle laden + Inference
├── router.py                ← Intent-Klassifikation
├── models.py                ← Pydantic-Schemas
├── db/
│   ├── __init__.py
│   ├── connection.py        ← MySQL Pool + Auto-Init
│   └── schema.sql           ← Tabellen-Definitionen
├── tools/
│   ├── __init__.py          ← Tool-Registry
│   └── dummy_weather.py     ← Beispiel-Tool (Dummy)
└── web/
    └── index.html           ← Test-WebUI (kein Framework)
```
