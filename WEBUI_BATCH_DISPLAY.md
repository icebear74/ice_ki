# WebUI Batch File Display Feature

## Übersicht (Overview)

Die WebUI zeigt jetzt in Echtzeit welche Dateien im aktuellen Trainings-Batch verarbeitet werden.

## Funktionen (Features)

### 1. Files Used Counter
- Zeigt wie viele Dateien in der aktuellen Epoche bereits verarbeitet wurden
- Format: `128 / 1000` (verarbeitete / gesamt)
- Aktualisiert sich bei jedem Batch

### 2. Batch Size Display
- Zeigt die aktuelle Auflösung des Batches
- Werte: `720`, `540`, oder `720_169`
- Zeigt welche Größe gerade trainiert wird

### 3. Batch Files List
- Scrollbares Textfeld mit allen Dateien im aktuellen Batch
- Format: `size_key/filename.png` (z.B. `720/image_001.png`)
- Nur GT (Ground Truth) Dateinamen
- Zeigt den relativen Pfad (Ende des Pfades)

## WebUI Location

Die neue "Current Batch" Karte befindet sich in der WebUI:
- URL: `http://localhost:5050`
- Position: Nach der "Dataset Files" Karte
- Aktualisierung: Automatisch alle 5 Sekunden

## Beispiel Display

```
📦 Current Batch
┌────────────────────────────────────┐
│ Files Used:  128 / 1000            │
│ Batch Size:  720                   │
│                                    │
│ Files in Batch:                    │
│ ┌────────────────────────────────┐ │
│ │ 720/scene_001_frame_0042.png  │ │
│ │ 720/scene_001_frame_0043.png  │ │
│ │ 720/scene_002_frame_0128.png  │ │
│ │ 720/scene_002_frame_0129.png  │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘
```

## Technische Details

### Datenfluss
1. **MultiSizeDataLoader** liefert `filenames` im Batch
2. **Trainer** erfasst und formatiert die Dateinamen
3. **web_ui DataStore** speichert die Info thread-sicher
4. **WebUI HTML** zeigt die Daten im Browser

### Performance
- Keine zusätzliche Disk I/O (Daten bereits vorhanden)
- Thread-sicher über `web_ui.data_store`
- Minimaler Overhead (~1ms pro Batch)
- Kein Impact auf Training Performance

### Geänderte Dateien
- `vsr_plusplus_NEU/systems/web_ui.py` - DataStore Erweiterung
- `vsr_plusplus_NEU/training/trainer.py` - Batch Tracking
- `vsr_plusplus_NEU/web/templates/monitor.html` - UI Display

## Verwendung (Usage)

1. Starten Sie das Training:
   ```bash
   cd vsr_plusplus_NEU
   python train.py
   ```

2. Öffnen Sie die WebUI:
   ```
   http://localhost:5050
   ```

3. Sehen Sie die "Current Batch" Karte:
   - Files Used Counter zeigt Fortschritt
   - Files in Batch zeigt aktuelle Dateien
   - Batch Size zeigt aktuelle Auflösung

## Vorteile

### Debugging
- Sofort sehen welche Dateien verarbeitet werden
- Prüfen ob bestimmte Dateien im Training sind
- Verifizieren dass alle Größen trainiert werden

### Monitoring
- Überwachen des Trainingsfortschritts in Echtzeit
- Sehen wie viele Dateien noch zu verarbeiten sind
- Erkennen von Problemen mit bestimmten Dateien

### Transparenz
- Volle Einsicht in den Trainingsprozess
- Verstehen welche Daten gerade verwendet werden
- Nachvollziehbarkeit des Trainings

## Hinweise

- Die Anzeige aktualisiert sich automatisch alle 5 Sekunden
- Bei sehr großen Batches (>100 Dateien) kann das Textfeld scrollen
- Die Counter werden bei jeder Epoche zurückgesetzt
- Funktioniert mit Multi-Size Training (720, 540, 720_169)

## Fehlerbehandlung

Falls keine Dateien angezeigt werden:
1. Prüfen Sie ob das Training läuft
2. Warten Sie auf den nächsten Batch (max. 5 Sekunden)
3. Prüfen Sie ob `MultiSizeDataLoader` verwendet wird
4. Prüfen Sie die Browser Console auf JavaScript Fehler

## Zukünftige Erweiterungen

Mögliche Erweiterungen:
- Farbcodierung für verschiedene Größen
- Klickbare Dateinamen zum Öffnen
- Export der Batch-Liste
- Historische Batch-Übersicht
- Statistiken pro Datei (wie oft trainiert)
