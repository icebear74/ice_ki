# Checkpoint Migration: 5-Frame → 7-Frame Model

## ⚠️ WICHTIG: Alte Checkpoints sind NICHT kompatibel!

Nach der Umstellung auf das 7-Frame Modell können alte Checkpoints **NICHT** weiterverwendet werden.

## Warum sind Checkpoints inkompatibel?

### Architektur-Unterschiede

**Altes 5-Frame Modell (`VSRBidirectional_3x`):**
```
- Input Shape: [Batch, 5, 3, H, W]
- Center Frame: Index 2 (3. Frame von 5)
- Fusion Layer: TrackedConv2d
  - Gewichte: backward_fuse.conv.weight (1 Layer)
  - Gewichte: backward_fuse.conv.bias
- Forward Pass: Verwendet Frame-Indizes [0, 1, 2, 3, 4]
```

**Neues 7-Frame Modell (`VSRBidirectional_7frames_3x`):**
```
- Input Shape: [Batch, 7, 3, H, W]
- Center Frame: Index 3 (4. Frame von 7)
- Fusion Layer: FusionBlock
  - Gewichte: backward_fuse.conv3x3.weight (2 Layer)
  - Gewichte: backward_fuse.conv3x3.bias
  - Gewichte: backward_fuse.conv1x1.weight
  - Gewichte: backward_fuse.conv1x1.bias
- Forward Pass: Verwendet Frame-Indizes [0, 1, 2, 3, 4, 5, 6]
```

### Konkrete Inkompatibilitäten

1. **Gewichts-Namen unterscheiden sich:**
   - Alt: `backward_fuse.conv.weight`
   - Neu: `backward_fuse.conv3x3.weight` + `backward_fuse.conv1x1.weight`

2. **Gewichts-Shapes unterscheiden sich:**
   - Alt: TrackedConv2d hat nur 1 Conv-Layer
   - Neu: FusionBlock hat 2 Conv-Layer (3x3 + 1x1)

3. **Input-Shape ist unterschiedlich:**
   - Alt: Model erwartet 5 Frames
   - Neu: Model erwartet 7 Frames

## Was muss ich tun?

### Option 1: Alte Checkpoints löschen (EMPFOHLEN)

```bash
# In deinem Training-Verzeichnis
cd /mnt/data/training  # oder wo auch immer deine Checkpoints sind

# Alte Checkpoints löschen
rm -rf checkpoints/*.pth
rm -rf best_models/*.pth

# ODER: In separates Backup-Verzeichnis verschieben
mkdir -p old_checkpoints_5frame
mv checkpoints/*.pth old_checkpoints_5frame/
mv best_models/*.pth old_checkpoints_5frame/

echo "Alte 5-Frame Checkpoints gesichert/gelöscht!"
```

### Option 2: Parallele Verzeichnisse verwenden

Wenn du die alten Checkpoints behalten willst:

```bash
# Alte Checkpoints umbenennen
mv checkpoints checkpoints_5frame
mv best_models best_models_5frame

# Neue Verzeichnisse für 7-Frame Training
mkdir -p checkpoints
mkdir -p best_models

echo "Neue Verzeichnisse für 7-Frame Training erstellt!"
```

### Option 3: Komplett neues DATA_ROOT

```bash
# In config.py
# Alt:
DATA_ROOT = "/mnt/data/training/datasetNeu/master"

# Neu:
DATA_ROOT = "/mnt/data/training/datasetNeu_7frame/master"
```

## Training neu starten

Nach dem Löschen/Verschieben der alten Checkpoints:

```bash
cd /home/runner/work/ice_ki/ice_ki/vsr_plusplus_NEU
python train.py
```

Das Training startet **von Step 0** mit dem neuen 7-Frame Modell!

## Kann ich alte Checkpoints konvertieren?

**NEIN**, eine automatische Konvertierung ist **NICHT möglich**, weil:

1. Die Fusion-Layer-Architektur komplett unterschiedlich ist
2. Das Modell für eine andere Anzahl von Input-Frames designed ist
3. Die gelernten Features für 5 vs. 7 Frames unterschiedlich sind

Du musst **von vorne trainieren** mit dem 7-Frame Modell!

## Vorteile des 7-Frame Modells

Obwohl du von vorne anfangen musst, hat das 7-Frame Modell Vorteile:

✅ **Mehr temporaler Kontext** (7 statt 5 Frames)
✅ **Bessere Bewegungskompensation** durch mehr Nachbar-Frames
✅ **Verbessertes FusionBlock** mit 3x3 Conv (räumlicher Kontext)
✅ **Passt zum dataset_generator_v2** (erstellt 7-Frame Daten)

## Zusammenfassung

| Aspekt | 5-Frame (alt) | 7-Frame (neu) |
|--------|---------------|---------------|
| Frames | 5 | 7 |
| Center Index | 2 | 3 |
| Fusion Layer | TrackedConv2d (1 conv) | FusionBlock (2 convs) |
| Checkpoints | ❌ Inkompatibel | ✅ Neu trainieren |
| Dataset | Teilweise ungenutzt | ✅ Voll genutzt |

**Fazit: Ja, alte Checkpoints müssen gelöscht werden. Training von vorne starten!** 🚀
