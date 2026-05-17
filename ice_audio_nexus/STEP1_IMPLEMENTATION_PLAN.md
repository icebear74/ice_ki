# Step 1 Implementation Plan – seed-first visual identity anchors

Stand: 2026-05-17

## Warum diese Datei existiert

`README_FIRST.md` definiert das eigentliche Projektziel: ein multimodales Persona-Extraktionssystem, das später belastbar bestimmen soll, **wer was gesagt hat**.  
Step 1 ist dafür nicht „möglichst viel Face Detection“, sondern die Erzeugung **sauberer visueller Identitätsanker** mit maximaler Identitätsreinheit.

Diese Datei ist absichtlich gleichzeitig:

- Architekturanker für spätere Sessions/PRs
- Gap-Analyse des aktuellen `ice_audio_nexus`-Stands
- Fortschrittsprotokoll
- Arbeitscheckliste

---

## Zielbild für Step 1

### Neue Leitidee: seed-first statt track-first

Step 1 soll in Zukunft so funktionieren:

1. **Step 1A – High-quality face seed discovery**
   - zuerst hochwertige, eindeutige Einzelbilder/Gesichts-Crops finden
   - gute Bilder dürfen nicht verloren gehen, nur weil ein Track später schwach ist
   - konservative Gruppierung zu anonymen visuellen Gruppen wie `visual_person_001`
   - lieber zu streng clustern als Personen zu vermischen

2. **Step 1B – Review / Zuordnung**
   - Seeds/Gruppen in der WebUI reviewen
   - Bilder aus Gruppen entfernbar
   - Gruppen/Personen als irrelevant oder ignoriert markierbar
   - vorbereitete Rollen/Personas später direkt auswählbar

3. **Step 1C – Expansion / Tracking danach**
   - erst nach Review/Bestätigung weiteres Material automatisch nachziehen
   - Tracking dient danach als Erweiterung, nicht als primäre Wahrheit

---

## Wichtige Modellierungsentscheidungen

### Reale Person vs. Rolle

- interne visuelle Identität basiert auf **realen Personen**
- Rolle bleibt produktions-/kontextabhängig
- UI darf weiterhin rollenzentriert wirken
- Datenmodell muss intern sauber trennen zwischen:
  - realer Person / Face-Identität
  - Rolle / Persona
  - späterer Voice-/Sprecheridentität

### Vorbereitung für spätere Fusion

Diese PR implementiert die Audio-/Fusionsebene noch nicht, aber Step 1 muss bereits so vorbereitet werden, dass später möglich ist:

- Face Match → reale Person
- Voice Match → Sprecheridentität
- kontextbezogene Face/Voice-Verknüpfung pro Produktion/Staffel/Episode/Sprache
- spätere Sprecherwechsel / andere Synchronsprecher

---

## Gap-Analyse des aktuellen `ice_audio_nexus`-Stands

### Scanner / Orchestrierung

- [x] Bestehender Scanner erkennt Gesichter, bildet lokale Tracks und speichert Detections/Tracks
- [x] Bestehende Logik ist aktuell **track-first**
- [x] „clear track“ entscheidet aktuell stark über Candidate-vs-Background
- [ ] Es gibt noch **keine eigenständigen Seed-Objekte**, die unabhängig von Tracks leben
- [ ] Es gibt noch **keine konservative Auto-Gruppierung** zu `visual_person_###`
- [ ] Expansion/Tracking nach bestätigten Seeds existiert noch nicht als eigener Workflow

### Datenmodell

- [x] `actors`, `roles`, `actor_roles`, `face_tracks`, `face_detections`, `face_samples` existieren bereits
- [x] Reale Person und Rolle sind bereits grundsätzlich getrennt modellierbar
- [x] `metadata_json` auf Tracks erlaubt kleine vorbereitende Workflow-Felder ohne Großumbau
- [ ] Es fehlt noch ein echtes Seed-/Visual-Group-Datenmodell
- [ ] Es fehlt noch ein Persona-Katalog mit zusätzlichem Sprecher-/Relevanz-Kontext
- [ ] Es fehlt noch ein explizites Modell für spätere Face-/Voice-Fusion

### WebUI / Review

- [x] Tracks lassen sich bereits ansehen, zuordnen, ignorieren und ausdünnen
- [x] Bestehende UI kann schon Actors und optionale Rollen anlegen
- [x] Diese PR ergänzt erste Seed-first-Workflow-Hooks (siehe „In dieser PR umgesetzt“)
- [ ] Es gibt noch keine echte Gruppenansicht für anonyme visuelle Personen
- [ ] Es gibt noch keine Persona-Katalog-Verwaltung mit Sprecher/Relevanz
- [ ] Expansion nach bestätigten Seeds ist noch nicht in der UI verdrahtet

---

## In dieser PR umgesetzt

### Phase 1 – Grundlage / Planung (erste Commits)
- [x] Diese Fortschritts-/Planungsdatei angelegt und mit Zielbild + Gap-Analyse gefüllt
- [x] `README_FIRST.md` explizit als Projektanker in die Step-1-Planung eingebunden
- [x] Scanner-Metadaten vorbereitet: `seed_workflow`-Block pro Track
- [x] API vorbereitet: Track-Responses mit normalisiertem `seed_workflow`, erster Workflow-Endpunkt
- [x] WebUI vorbereitet: Seed-first-Review-Felder, Rollenliste direkt auswählbar
- [x] README von `ice_audio_nexus` auf seed-first-Richtung aktualisiert

### Phase 2 – WP1–WP5 vollständig umgesetzt (diese Commits)
- [x] **WP1**: Tabellen `visual_groups` + `visual_seeds` + alle DB-Funktionen + API-Endpunkte
- [x] **WP2**: Konservatives Clustering (`cluster_tracks_into_groups`, threshold 0.92, `visual_person_###`)
- [x] **WP3**: Groups-Tab in der WebUI: Gruppenansicht, Seeds entfernbar, States verwaltbar
- [x] **WP4**: Tabelle `persona_catalog` + CRUD-Funktionen + Personas-Tab in der WebUI (Vorabpflege mit Rolle, realer Person, Synchronsprecher, Sprache, Relevanz)
- [x] **WP5**: Expansion-Trigger (nur `confirmed`-Gruppen; `irrelevant`/`ignored` bleiben geblockt)

### Bewusst noch nicht umgesetzt (WP6 / spätere PRs)
- [ ] echte Expansion-Engine (automatisches Tracking nach bestätigten Seeds)
- [ ] Voice-Identitäten als eigene Tabelle
- [ ] Face↔Voice-Fusion-Tabellen / Sprachkontext
- [ ] Sprecherwechsel über Staffeln hinweg

---

## Arbeitspakete

### WP1 – Seed-Objekt / Visual-Group-Modell ✅ UMGESETZT
- [x] Eigenständige Seed-Entität definieren → Tabelle `visual_seeds`
- [x] Eigenständige Visual-Group-Entität definieren → Tabelle `visual_groups`
- [x] Beziehung Track ↔ Seed ↔ Visual Group sauber modelliert
- [x] DB-Funktionen: `create_visual_group`, `list_visual_groups`, `get_visual_group`, `update_visual_group`, `_create_visual_seed`, `list_visual_seeds`, `remove_visual_seed`
- [x] API-Endpunkte: `GET/POST /api/visual_groups`, `GET/PUT /api/visual_groups/{id}`, `DELETE /api/visual_seeds/{id}`

### WP2 – Conservative clustering ✅ UMGESETZT
- [x] `cluster_tracks_into_groups()` – greedy Centroid-Clustering, threshold 0.92
- [x] lieber Split als Fehlmerge (nur sehr sichere Ähnlichkeiten)
- [x] Gruppenlabels `visual_person_001`, `visual_person_002`, ...
- [x] bereits gruppierte Tracks werden beim Re-Run übersprungen
- [x] API-Endpunkt: `POST /api/productions/{id}/cluster`
- [x] UI-Button: „Cluster tracks…" im Groups-Tab

### WP3 – Review-Workflow ✅ UMGESETZT
- [x] Gruppenansicht als eigener Tab im mittleren Panel (Tracks / Groups)
- [x] Seed-Bilder aus Gruppe entfernbar (soft-delete via `remove_visual_seed`)
- [x] Gruppen als `irrelevant`/`ignored`/`confirmed`/`needs_split`/`pending` markierbar
- [x] Group-Détailansicht im rechten Review-Panel (eigener Tab)
- [x] Actor/Role-Zuweisung direkt per Gruppen-Review

### WP4 – Persona-Katalog ✅ UMGESETZT
- [x] Tabelle `persona_catalog` (Produktion, Rolle, reale Person, Synchronsprecher, Sprache, Relevanz)
- [x] Vorabpflege für Rollen: `upsert_persona_catalog` legt Rolle und Actor on-the-fly an
- [x] Verknüpfung zu realer Person über `actor_id`
- [x] Sprecher / Synchronsprecher über `voice_actor_name` + `language`
- [x] Relevanzpriorisierung (0=niedrig, 1=mittel, 2=hoch, 3=lead)
- [x] API-Endpunkte: `GET/POST /api/persona_catalog`, `DELETE /api/persona_catalog/{id}`
- [x] UI-Tab „Personas" im rechten Panel: Liste, Vorabpflege-Formular, Production-Filter

### WP5 – Expansion nach Review ✅ UMGESETZT
- [x] `trigger_group_expansion()` – setzt `expansion_state = 'ready'` nur für `confirmed`-Gruppen
- [x] `block_group_expansion()` – explizit blocken
- [x] `irrelevant` / `ignored`-Gruppen werden nicht expanded, Fehlermeldung mit Begründung
- [x] API-Endpunkte: `POST /api/visual_groups/{id}/expand`, `POST /api/visual_groups/{id}/block_expansion`
- [x] UI-Buttons: „Expand" und „Block" pro Gruppe

### WP6 – Multimodale Vorbereitung
- [ ] Voice-Identitäten modellieren (eigene Tabelle)
- [ ] Produktions-/Sprachkontext für Face↔Voice vorbereiten
- [ ] spätere Sprecherwechsel sauber abbilden (z. B. anderer Sync-Sprecher ab Staffel X)

---

## Empfohlene nächste PRs

1. **PR C – echte Expansion-Engine**
   - Matching gegen bestätigte Seeds um das Videomaterial zu erweitern
   - Tracking als Folgeprozess nach Review
2. **PR D – Voice-Identitäten / Schritt 2 vorbereiten**
   - eigene Tabelle für Speaker-Identitäten
   - Vorbereitung für automatischen Stimmproben-Seed aus bestätigten visuellen Seeds
3. **PR E – Face↔Voice Fusion**
   - Kontextbezogene Verknüpfung zwischen Face und Voice
   - Sprecherwechsel über Staffeln / Sprachversionen

---

## Kurzfazit

WP1–WP5 sind vollständig umgesetzt. Das Step-1-Fundament ist belastbar:

- `visual_groups` + `visual_seeds` + `persona_catalog` als eigene Tabellen
- Konservatives Clustering (0.92 Threshold, lieber Split als Merge)
- Vorabpflege des Persona-Katalogs direkt in der WebUI
- Groups-Tab für Group-zentrierten Review-Workflow (Seeds entfernbar, States, Actor/Role-Zuweisung)
- Expansion nur für bestätigte Gruppen, Schutz vor `irrelevant`/`ignored`-Expansion
- Nächster Schritt: echte Expansion-Engine + Step 2 (Voice) vorbereiten
