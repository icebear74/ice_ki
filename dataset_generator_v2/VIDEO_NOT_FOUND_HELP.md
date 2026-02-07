# ⚠️ WICHTIG: Video-Dateien nicht gefunden!

## Problem

Der Generator hat in 2 Sekunden "fertig" gemeldet, aber keine Bilder generiert.

**Grund:** Die `generator_config.json` enthält 530 Video-Einträge mit **erfundenen Pfaden**, die nicht existieren!

## Lösung

Du hast **3 Optionen**:

### Option 1: Video-Scanner verwenden (EMPFOHLEN)

Der Video-Scanner scannt automatisch deine echten Video-Dateien und erstellt eine korrekte Config:

```bash
cd dataset_generator_v2

# Scanne deine Videos (ersetze den Pfad!)
python scan_videos.py /pfad/zu/deinen/videos

# Beispiel:
# python scan_videos.py /media/videos
# python scan_videos.py ~/Videos

# Das erstellt: generator_config_REAL.json

# Prüfe die generierte Config
cat generator_config_REAL.json | head -50

# Wenn OK, ersetze die alte Config
mv generator_config.json generator_config.json.backup
mv generator_config_REAL.json generator_config.json

# Jetzt starten
python make_dataset_multi.py
```

### Option 2: Manuelle Config-Erstellung

Erstelle eine neue `generator_config.json` mit deinen echten Videos:

```json
{
  "base_settings": {
    "base_frame_limit": 3000,
    "max_workers": 12,
    "val_percent": 0.0,
    "output_base_dir": "/mnt/data/training/dataset",
    "temp_dir": "/mnt/data/training/dataset/temp",
    "status_file": "/mnt/data/training/dataset/.generator_status.json",
    "min_file_size": 10000,
    "scene_diff_threshold": 45,
    "max_retry_attempts": 10,
    "retry_skip_seconds": 60
  },
  
  "videos": [
    {
      "name": "Mein Video 1",
      "path": "/echter/pfad/zu/video1.mkv",
      "categories": {
        "general": 1.0
      }
    },
    {
      "name": "Mein Video 2",
      "path": "/echter/pfad/zu/video2.mp4",
      "categories": {
        "space": 1.0
      }
    }
  ]
}
```

**Wichtig:**
- Verwende ABSOLUTE Pfade!
- Die Video-Dateien müssen existieren (.mkv, .mp4, .avi)
- Kategorien: `general`, `space`, `toon`
- Gewichte müssen zusammen <= 1.0 sein

### Option 3: Test mit einzelnem Video

Für einen schnellen Test mit nur einem Video:

```bash
# Finde ein echtes Video
find ~ -name "*.mkv" -o -name "*.mp4" | head -5

# Erstelle minimale Test-Config
cat > generator_config_test.json << 'EOF'
{
  "base_settings": {
    "base_frame_limit": 100,
    "max_workers": 4,
    "val_percent": 0.0,
    "output_base_dir": "/tmp/test_dataset",
    "temp_dir": "/tmp/test_dataset/temp",
    "status_file": "/tmp/test_dataset/.status.json",
    "min_file_size": 10000,
    "scene_diff_threshold": 45,
    "max_retry_attempts": 10,
    "retry_skip_seconds": 60
  },
  "videos": [
    {
      "name": "Test Video",
      "path": "/pfad/zu/deinem/test/video.mkv",
      "categories": {
        "general": 1.0
      }
    }
  ]
}
EOF

# Pfad anpassen und testen
python make_dataset_multi.py
```

## Was der Generator jetzt macht

Mit den neuen Verbesserungen:

1. ✅ **Validiert alle Videos vor Start**
   - Zeigt wie viele Videos gefunden/fehlen
   
2. ✅ **Stoppt mit klarer Fehlermeldung** wenn keine Videos existieren
   
3. ✅ **Warnt bei vielen fehlenden Videos**
   - Gibt 5 Sekunden Zeit zum Abbrechen
   
4. ✅ **Loggt jedes übersprungene Video**
   - Zeigt Name und Pfad
   
5. ✅ **Zeigt Anleitung** zur Verwendung von scan_videos.py

## Nächste Schritte

1. Entscheide dich für eine Option oben
2. Erstelle korrekte Config mit echten Video-Pfaden
3. Starte Generator erneut:
   ```bash
   cd dataset_generator_v2
   python make_dataset_multi.py
   ```

4. Der Generator sollte jetzt:
   - Videos validieren ✓
   - Extraktion starten ✓
   - Bilder generieren ✓
   - Fortschritt anzeigen ✓

## Hilfe

Falls Probleme:

```bash
# Teste Video-Scanner
python scan_videos.py --help

# Teste Config
python -c "import json; print(json.load(open('generator_config.json'))['videos'][0])"

# Prüfe Video-Pfad
ls -lh /pfad/zu/video.mkv
```
