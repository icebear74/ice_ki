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

- [x] Diese Fortschritts-/Planungsdatei angelegt und mit Zielbild + Gap-Analyse gefüllt
- [x] `README_FIRST.md` explizit als Projektanker in die Step-1-Planung eingebunden
- [x] Scanner-Metadaten vorbereitet:
  - `seed_workflow`-Block pro Track
  - Top-Seed-Kandidaten im Track-Metadatenblock
  - Tracking wird explizit als Support-Container statt Wahrheitsquelle gedacht
- [x] API vorbereitet:
  - Track-Responses liefern normalisierte `seed_workflow`-Daten
  - neuer Workflow-Endpunkt zum Speichern von Review-/Stage-/Group-Infos
- [x] WebUI vorbereitet:
  - Seed-first-Review-Felder (`group_label`, `stage`, `review_state`, `expansion_state`, `notes`)
  - bestehende Rollenliste nun direkt auswählbar
  - UI-Text macht klar: intern reale Person, optional Rolle/Persona-Kontext
- [x] README von `ice_audio_nexus` auf seed-first-Richtung aktualisiert

### Wichtig: Was diese PR **bewusst noch nicht** macht

- [x] **Keine** neue massive Seed-/Cluster-Tabelle einführen
- [x] **Kein** vollständiges `visual_person_###`-Auto-Clustering implementieren
- [x] **Keine** fertige Expansion-/Tracking-Nachziehlogik bauen
- [x] **Keinen** vollständigen Persona-Katalog inklusive Sprecher/Relevanz fertigstellen
- [x] **Keine** Voice-/Fusion-Logik implementieren

---

## Arbeitspakete ab jetzt

### WP1 – Seed-Objekt / Visual-Group-Modell
- [ ] Eigenständige Seed-Entität definieren
- [ ] Eigenständige Visual-Group-Entität definieren
- [ ] Beziehung Track ↔ Seed ↔ Visual Group sauber modellieren

### WP2 – Conservative clustering
- [ ] nur sehr sichere Ähnlichkeiten automatisch mergen
- [ ] lieber Split als Fehlmerge
- [ ] Gruppenlabels `visual_person_001`, `visual_person_002`, ...

### WP3 – Review-Workflow
- [ ] Gruppenansicht statt reinem Track-Fokus
- [ ] Seed-Bilder aus Gruppe entfernbar
- [ ] Gruppen als irrelevant/ignored/fertig markierbar

### WP4 – Persona-Katalog
- [ ] Vorabpflege für Rolle
- [ ] Verknüpfung zu realer Person
- [ ] Sprecher / Synchronsprecher
- [ ] Relevanzpriorisierung

### WP5 – Expansion nach Review
- [ ] bestätigte Seeds als Suchanker benutzen
- [ ] Tracking/Matching nur als Folgeprozess
- [ ] aggressives Nachziehen für irrelevante Seeds verhindern

### WP6 – Multimodale Vorbereitung
- [ ] Voice-Identitäten modellieren
- [ ] Produktions-/Sprachkontext für Face↔Voice vorbereiten
- [ ] spätere Sprecherwechsel sauber abbilden

---

## Empfohlene nächste PRs

1. **PR A – echtes Seed-/Visual-Group-Datenmodell**
   - neue Tabellen/Objekte statt reinem Track-Metadaten-Hook
2. **PR B – konservatives Auto-Clustering**
   - Seed-Erzeugung + `visual_person_###`
3. **PR C – Persona-Katalog**
   - Rolle / reale Person / Sprecher / Relevanz
4. **PR D – Expansion nach bestätigten Seeds**
   - Tracking als Folgeprozess

---

## Kurzfazit

Diese PR richtet Step 1 noch nicht vollständig neu aus, schafft aber eine belastbare Grundlage:

- Projektziel aus `README_FIRST.md` ist explizit verankert
- track-first-Schwächen sind dokumentiert
- seed-first-Workflow ist beschrieben
- erste UI/API/Metadaten-Hooks für Review, Group-Label und spätere Expansion sind vorhanden

Damit kann die nächste PR den eigentlichen Datenmodell- und Workflow-Umbau deutlich gezielter durchführen.
