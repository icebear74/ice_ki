# Step 1 Implementation Plan – seed-first visual identity anchors

Stand: 2026-05-17

## Warum diese Datei existiert

`README_FIRST.md` ist verbindlich: Step 1 muss saubere visuelle Identitätsanker liefern (Identitätsreinheit > rohe Detection-Menge), damit Step 2/3 später belastbar bestimmen können: **wer hat was gesagt**.

---

## Fehlumsetzung / Abweichung im bisherigen Stand

- Der erste Scan (`scanner --video`) lief praktisch weiter **track-first** (u. a. Logs mit `active tracks`/`finished tracks`).
- `STEP1_IMPLEMENTATION_PLAN.md` war dafür zu optimistisch und markierte seed-first-nahe Punkte als vollständig umgesetzt.
- Persona-/Cast-/Voice-Bereich war im Alltag noch zu freitextlastig; Sprecherwechsel-Startregel war nicht sauber modelliert.

---

## In dieser PR korrigiert

### Scanner / Workflow
- `scanner --video <file>` arbeitet jetzt seed-first:
  - Frame-Sampling + Detection + Qualitätsfilter
  - konservative Gruppierung per Embedding-Ähnlichkeit (`FACE_SEED_GROUP_SIMILARITY_THRESHOLD`, Default 0.90)
  - keine dominierende Track-Lebenszyklus-Logik mehr als Kern
  - seed-first Logs mit granularen Reject-Gründen: `rejected_small`, `rejected_blurry`, `rejected_pose`, `rejected_occluded`, `rejected_dark`, `rejected_quality_score`, `verifier_rejects`, `duplicate_matches`
  - Stufen-Transparenz: `quality_passed_before_verifier` und `verifier_rejects_after_quality`
  - optionale JSON-Laufstatistik (`FACE_SEED_DEBUG_STATS_ENABLED`, `FACE_SEED_DEBUG_STATS_DIR`)
- `scanner` ohne Dateiname startet jetzt Expansion-Orchestrierung:
  - nur `confirmed` + `expansion_state=ready`
  - nur auf **freigegebenen Episoden** (`expansion_released=true`)
  - `ignored`/`irrelevant` werden weiter geblockt

### Datenmodell / API
- Video-Workflow erweitert um Episode-Freigabe für Expansion (`expansion_released` in `videos.metadata_json`, API: `POST /api/videos/{id}/expansion_release`).
- Expansion Engine akzeptiert jetzt optionale Episode-Gates (`allowed_video_ids`) und expandiert dadurch nur freigegebenes Material.
- Persona-Modell normalisiert erweitert:
  - neue Entität `voice_actors`
  - `persona_catalog.voice_actor_id`
  - neue Kontexttabelle `role_cast_assignments` mit Startregel (`start_season`, `start_episode`, Sprache, Relevanz)
- Scanner/API-Parameter für Step-1A jetzt umfassend per `.env` steuerbar (Sampling, Detection/Quality, Verifier/Grouping, Duplicate-Handling, Debug-Ausgabe).
- Device-Zuordnung je Komponente konfigurierbar (`FACE_DETECTOR_DEVICE`, `FACE_VERIFIER_DEVICE`, `FACE_EMBEDDING_DEVICE`) mit CPU-Fallback je Komponente.

### WebUI
- Episoden können für Expansion explizit freigegeben/geblockt werden.
- Persona-Bereich auf wiederverwendbare Entitäten erweitert (Schauspieler/Rolle/Synchronsprecher per Auswahl; Freitext nur Fallback für Neu-Anlage).
- Stammdaten-Schnellerfassung für Schauspieler/Synchronsprecher/Rollen ergänzt.
- Rollen-/Besetzungszuordnung inkl. Startregel „ab Staffel X Folge Y“ erfassbar gemacht.

---

## Noch offen / spätere PR

- Vollständige semantische Auswertung mehrerer Sprecherwechselregeln (Konflikt-/Prioritätslogik in Inferenzpfaden).
- Weitere UX-Verbesserung der neuen Stammdaten-/Assignment-Ansichten (Filter, Editieren bestehender Einträge).
- Multimodale Face↔Voice-Fusionsauswertung (Step 2/3).
- Reject-Gründe `pose`/`occluded` werden aktuell heuristisch aus Geometrie + Verifier-Metadaten abgeleitet; feinere Landmark-basierte Trennung bleibt offen.

---

## Konkreter Änderungsplan (ehrlich abgehakt)

### Änderungsplan A – Scanner / Workflow
- [x] 1. Seed-Mode (`--video`) wirklich seed-first machen
- [x] 2. Parameterlosen Scanner-Aufruf als Expansion-/Tracking-Orchestrator implementieren oder korrigieren
- [x] 3. Seed-Discovery-Logs auf seed-first umstellen
- [x] 4. Expansion-/Tracking-Logs separat halten
- [x] 5. Status-/Workflowmodell für Seed-Scan, Review, Freigabe, Ignore, Expansion sauber modellieren
- [x] 6. Staffel-/Mehrfachfolgen-Workflow unterstützen (episodische Expansion-Freigabe)

### Änderungsplan B – WebUI / API
- [x] 1. Seed-Review und Expansion-Status klar trennen
- [x] 2. Episoden oder Gruppen für Tracking/Expansion freigebbar machen
- [x] 3. Persona-Bereich weg von primär Freitext, hin zu wiederverwendbaren Entitäten
- [x] 4. UI-Strukturen für Schauspieler, Synchronsprecher, Produktionen und Rollen bereitstellen
- [x] 5. Rollen-/Besetzungszuordnung in der UI erfassbar machen
- [x] 6. Ignore / Irrelevant sauber in UI und API durchziehen

### Änderungsplan C – Datenmodell
- [x] 1. Prüfen, welche bestehenden Tabellen wiederverwendet werden können
- [x] 2. Notwendige Tabellen/Felder für Actor / Voice Actor / Role / Production / Assignment ergänzen oder normalisieren
- [x] 3. Startregel für Sprecherwechsel im Datenmodell vorbereiten oder implementieren
- [x] 4. Relationen so modellieren, dass reale Person, Rolle und Synchronsprecher sauber getrennt sind

### Änderungsplan D – Dokumentation
- [x] 1. `STEP1_IMPLEMENTATION_PLAN.md` ehrlich korrigieren
- [x] 2. Klar markieren, was bisher falsch oder nur teilweise umgesetzt war
- [x] 3. Dokumentieren, was in dieser PR korrigiert wurde
- [x] 4. Offen dokumentieren, was noch offen bleibt
- [x] 5. README/Doku ergänzen, falls nötig
