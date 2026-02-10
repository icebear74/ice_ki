# Runtime Config Behavior - Important Information

## TL;DR: Dataset Configuration Requires Restart

**⚠️ WICHTIG:** Änderungen an Dataset-Konfiguration (welche Sizes verwendet werden) erfordern einen **NEUSTART** des Trainings!

## Was funktioniert LIVE (während Training läuft)?

Diese Parameter können in `runtime_config.json` geändert werden und werden **alle 10 Steps automatisch neu geladen**:

✅ **Training Parameters** (werden live aktualisiert):
- Learning Rate (`max_lr`, `min_lr`)
- Loss Weights (`l1_weight_target`, `ms_weight_target`, etc.)
- Gradient Clipping (`initial_grad_clip`)
- Plateau Detection (`plateau_safety_threshold`, `plateau_patience`)
- Cooldown (`cooldown_duration`)
- Logging Intervals (`log_tboard_every`, `val_step_every`, `save_step_every`)

## Was erfordert RESTART?

Diese Parameter werden **NUR beim Start** gelesen und können nicht live geändert werden:

❌ **Dataset Configuration** (erfordert Neustart):
- `size_distribution` - Welche Dataset-Sizes verwendet werden
- `data.root` - Pfad zu den Datasets
- `data.dataset_name` - Name des Datasets
- `model` - Modell-Architektur (n_frames, n_feats, n_blocks)
- `training.adaptive_batch` - Batch-Sizes pro Size

## Warum ist das so?

### Technischer Hintergrund

1. **Beim Trainingsstart** (einmalig):
   ```python
   # train.py liest runtime_config.json
   size_dist = {"720": 0.4, "540": 0.4, "720_169": 0.2}
   
   # Erstellt Data Loader basierend auf Konfiguration
   # Lädt Datasets von Disk (kann mehrere Minuten dauern)
   train_loader = create_train_loader(config)
   # → 720 Dataset: 1234 files geladen
   # → 540 Dataset: 1456 files geladen
   # → 720_169 Dataset: 987 files geladen
   ```

2. **Während Training** (alle 10 Steps):
   ```python
   # Prüft ob runtime_config.json geändert wurde
   if runtime_config.check_for_updates():
       # Lädt neue Werte für Training-Parameter
       new_lr = config.get('max_lr')
       # Wendet sie sofort an
       optimizer.set_lr(new_lr)
       
       # ABER: Datasets werden NICHT neu geladen!
       # train_loader bleibt unverändert
   ```

### Warum nicht live ändern?

Das Neu-Laden von Datasets würde:
- Mehrere Minuten dauern (alle PNG-Files neu laden)
- Training unterbrechen
- GPU-Speicher neu allokieren
- Inkonsistenzen im Training erzeugen

Daher ist Dataset-Konfiguration bewusst **startup-only**.

## Beispiel-Szenario

### Problem
```
1. Training gestartet mit:
   size_distribution: {"720": 0.0, "540": 0.5, "720_169": 0.5}
   → 720 wurde NICHT geladen (distribution = 0)

2. Während Training läuft, runtime_config.json geändert zu:
   size_distribution: {"720": 0.4, "540": 0.3, "720_169": 0.3}
   → 720 wird IMMER NOCH NICHT geladen!

3. Web UI zeigt:
   Training Dataset: 540+720_169 (2443 files)
   → 720 fehlt, weil es nie geladen wurde
```

### Lösung
```
1. Training STOPPEN (Ctrl+C oder Pause Button)

2. runtime_config.json anpassen:
   size_distribution: {"720": 0.4, "540": 0.3, "720_169": 0.3}

3. Training NEU STARTEN:
   python vsr_plusplus_NEU/train.py

4. Beim Start wird angezeigt:
   ✅ Multi-size training samples: 3677
   📊 Dataset Sizes Loaded at Startup:
     • 720: 1234 samples (40.0%)
     • 540: 1456 samples (30.0%)
     • 720_169: 987 samples (30.0%)
   
   → Jetzt ist 720 geladen!
```

## Warnung beim Live-Ändern

Seit dieser Update bekommst du eine Warnung, wenn du versuchst, startup-only Parameter zu ändern:

```
🔄 Runtime config file changed externally, reloading...

⚠️  WARNING: Size distribution (which dataset sizes to use) was changed
   size_distribution: {'720': 0.0, '540': 0.5, ...} → {'720': 0.4, '540': 0.3, ...}
   ⚠️  This change requires TRAINING RESTART to take effect!
   ⚠️  Current training session is using the old configuration.
```

## Checkliste: Was muss ich wann tun?

### Ich will Learning Rate ändern
- [x] Ändere `max_lr` oder `min_lr` in runtime_config.json
- [x] Warte ~10 seconds (wird automatisch neu geladen)
- [x] Fertig! ✅

### Ich will Loss Weights ändern
- [x] Ändere `l1_weight_target`, `ms_weight_target`, etc. in runtime_config.json
- [x] Warte ~10 seconds (wird automatisch neu geladen)
- [x] Fertig! ✅

### Ich will 720 Dataset aktivieren/deaktivieren
- [x] Ändere `size_distribution.720` in runtime_config.json
- [x] Training STOPPEN
- [x] Training NEU STARTEN
- [x] Prüfe Startup-Meldung: "Dataset Sizes Loaded at Startup"
- [x] Fertig! ✅

### Ich will Batch Size ändern
- [x] Ändere `training.adaptive_batch.720.batch` in runtime_config.json
- [x] Training STOPPEN
- [x] Training NEU STARTEN
- [x] Fertig! ✅

## Web UI Anzeige

Die Web UI zeigt korrekt, welche Datasets geladen sind:

```
📂 Dataset Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Dataset
  Size: 540+720_169         2443 files  ✓
  (720 NICHT geladen, weil beim Start nicht konfiguriert)

Validation Datasets
  720×720                     3 files  ✓
  720×405 (16:9)              3 files  ✓
```

Wenn 720 mit 0 files angezeigt wird:
- ✅ Das ist KORREKT - 720 wurde nicht beim Start geladen
- ❌ Live-Änderung der Config hilft NICHT
- ✅ Training neu starten mit korrekter size_distribution

## Zusammenfassung

| Was                          | Live-Änderung | Restart nötig |
|------------------------------|---------------|---------------|
| Learning Rate                | ✅ Ja         | ❌ Nein       |
| Loss Weights                 | ✅ Ja         | ❌ Nein       |
| Plateau Detection            | ✅ Ja         | ❌ Nein       |
| **Size Distribution**        | ❌ Nein       | ✅ **JA**     |
| **Dataset Paths**            | ❌ Nein       | ✅ **JA**     |
| **Batch Sizes**              | ❌ Nein       | ✅ **JA**     |
| **Model Architecture**       | ❌ Nein       | ✅ **JA**     |

**Merksatz:** Alles was mit **Daten oder Modell-Struktur** zu tun hat → Restart nötig!
