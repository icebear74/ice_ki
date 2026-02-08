# Debugging Guide - Generator stoppt nach 1 Video

## Status

✅ **Debug-Logging ist implementiert und aktiv**
✅ **Scene Detection ist entfernt** (verifiziert)
✅ **Priority System funktioniert** (getestet)

## Das Problem

Der Generator stoppt nach der Verarbeitung des ersten Videos, obwohl das Video erfolgreich completed wurde.

## Warum wir Logs brauchen

Ohne Logs können wir nur spekulieren. Die Logs werden uns **exakt** zeigen:
- Wo der Code stoppt
- Welche Exception geworfen wird (falls vorhanden)
- Warum die Loop endet
- Status von self.running
- Welche Videos verarbeitet werden

## Wie man die Logs bekommt

### 1. Generator normal starten

```bash
cd /home/runner/work/ice_ki/ice_ki/dataset_generator_v2
python3 make_dataset_multi.py
```

Der Generator läuft normal, schreibt aber zusätzlich Debug-Logs.

### 2. Warten bis er stoppt

Wenn er nach Video 0 stoppt (wie beschrieben), weitermachen mit Schritt 3.

### 3. Logs analysieren

**Option A - Automatische Analyse:**
```bash
cd /home/runner/work/ice_ki/ice_ki
python3 analyze_generator_log.py
```

Das Script zeigt:
- Wie viele Videos gestartet wurden
- Wie viele Videos completed wurden
- Alle Exceptions
- Alle Warnings
- Warum die Loop endete

**Option B - Manuelle Log-Prüfung:**
```bash
# Letzte 50 Zeilen anzeigen
tail -50 /mnt/data/training/dataset/generator_debug.log

# Alle Exceptions finden
grep -i "exception" /mnt/data/training/dataset/generator_debug.log

# Nach "MAIN LOOP ENDED" suchen
grep "MAIN LOOP ENDED" /mnt/data/training/dataset/generator_debug.log

# Alle Warnings finden
grep -i "warning" /mnt/data/training/dataset/generator_debug.log
```

## Was die Logs zeigen werden

### Szenario 1: Exception in Video 1
```
2024-02-08 12:00:00 - INFO - Processing video 0: SerieUHD - S01E06
2024-02-08 12:05:00 - INFO - Video 0 COMPLETED: 2500/3000 successful
2024-02-08 12:05:01 - DEBUG - --- Loop iteration 1 / 467 ---
2024-02-08 12:05:01 - INFO - Processing video 1: Forrest Gump
2024-02-08 12:05:01 - ERROR - EXCEPTION in video 1: FileNotFoundError: [Errno 2] No such file or directory
```
➡️ **Problem:** Video 1 Datei nicht gefunden oder anderer Fehler

### Szenario 2: self.running = False
```
2024-02-08 12:00:00 - INFO - Processing video 0: SerieUHD - S01E06
2024-02-08 12:05:00 - INFO - Video 0 COMPLETED: 2500/3000 successful
2024-02-08 12:05:01 - DEBUG - --- Loop iteration 1 / 467 ---
2024-02-08 12:05:01 - WARNING - Generator stopped by self.running=False at video 1
```
➡️ **Problem:** self.running wurde auf False gesetzt (Signal, Keyboard 'q', etc.)

### Szenario 3: Videos als completed markiert
```
2024-02-08 12:00:00 - INFO - Processing video 0: SerieUHD - S01E06
2024-02-08 12:05:00 - INFO - Video 0 COMPLETED: 2500/3000 successful
2024-02-08 12:05:01 - DEBUG - --- Loop iteration 1 / 467 ---
2024-02-08 12:05:01 - INFO - Processing video 1: Forrest Gump
2024-02-08 12:05:01 - INFO - Video 1 already completed - SKIPPING
2024-02-08 12:05:01 - DEBUG - --- Loop iteration 2 / 467 ---
2024-02-08 12:05:01 - INFO - Processing video 2: Zurück In Die Zukunft
2024-02-08 12:05:01 - INFO - Video 2 already completed - SKIPPING
...
```
➡️ **Problem:** Alle Videos außer dem ersten sind als completed markiert

### Szenario 4: Fatale Exception
```
2024-02-08 12:00:00 - INFO - Processing video 0: SerieUHD - S01E06
2024-02-08 12:05:00 - INFO - Video 0 COMPLETED: 2500/3000 successful
2024-02-08 12:05:01 - CRITICAL - FATAL EXCEPTION in main loop: MemoryError: ...
```
➡️ **Problem:** Kritischer Fehler (Memory, Disk Space, etc.)

## Nach der Log-Analyse

Sobald wir die Logs haben, können wir:
1. Den **exakten** Fehler identifizieren
2. Den **gezielten** Fix implementieren
3. Testen ob es funktioniert

## Log-Datei Speicherort

```
/mnt/data/training/dataset/generator_debug.log
```

## Tools

- **`analyze_generator_log.py`** - Automatische Log-Analyse
- **`test_priority_system.py`** - Priority System Test
- **`verify_logging.py`** - Logging Implementation Verifikation

## Zusammenfassung

🔧 **Alle Debug-Tools sind bereit**
📝 **Logging ist aktiv**
⏳ **Wir brauchen nur einen Test-Run um die Logs zu bekommen**
🎯 **Die Logs werden den Fehler zeigen**

Dann können wir den Bug fixen! 🚀
