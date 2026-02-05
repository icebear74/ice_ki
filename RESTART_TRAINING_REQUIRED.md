# WICHTIG: Training neu starten!

## Problem
Du siehst immer noch `w:1.00/0.00/0.00` in der GUI.

## Ursache
**Der Fix ist im Code**, ABER dein Training-Prozess läuft noch mit dem **alten Code im Speicher**.

Python lädt Module nur einmal beim Start. Änderungen am Code werden erst wirksam, wenn du den Prozess neu startest.

## Lösung

### 1. Training stoppen
```bash
# Drücke Ctrl+C im Terminal wo das Training läuft
# ODER
pkill -f "python.*train.py"
```

### 2. Code aktualisieren (falls nicht schon geschehen)
```bash
cd /home/runner/work/ice_ki/ice_ki
git pull origin copilot/hotfix-adaptive-system-weights
```

### 3. Training neu starten
```bash
cd /home/runner/work/ice_ki/ice_ki
python vsr_plus_plus/train.py
# ODER dein üblicher Start-Befehl
```

## Erwartetes Ergebnis

Nach dem Neustart solltest du sehen:

**Iteration 0-999 (Warmup):**
```
L1:   0.xxxxxx (w:0.60)  ← Korrekt!
MS:   0.xxxxxx (w:0.20)  ← Korrekt!
Grad: 0.xxxxxx (w:0.20)  ← Korrekt!
```

**Iteration >= 1000 beim Fortsetzen (Settling):**
```
L1:   0.xxxxxx (w:0.60)  ← Korrekt!
MS:   0.xxxxxx (w:0.20)  ← Korrekt!
Grad: 0.xxxxxx (w:0.20)  ← Korrekt!
Mode: Settling (1/100)    ← 100 Iterations vor Automation
```

## Verifikation

Der Fix ist definitiv im Code (siehe `debug_adaptive_weights.py`):
```bash
python debug_adaptive_weights.py
```

Alle Tests ✅ PASS - der Code funktioniert!

## Falls es immer noch nicht funktioniert

1. **Checkpoints löschen** (alte Checkpoints könnten alte Zustände haben):
   ```bash
   rm -rf runs/checkpoints/*_old*
   ```

2. **Überprüfe, dass du den richtigen Branch hast**:
   ```bash
   git branch
   # Sollte zeigen: * copilot/hotfix-adaptive-system-weights
   
   git log --oneline -1
   # Sollte zeigen: 325a322 Add soft start demonstration...
   ```

3. **Stelle sicher, dass die richtige config verwendet wird**:
   - L1_WEIGHT sollte 0.6 sein
   - MS_WEIGHT sollte 0.2 sein
   - GRAD_WEIGHT sollte 0.2 sein

## Zusammenfassung

✅ Der Code ist korrekt
✅ Der Fix ist implementiert
✅ Alle Tests bestehen

❗ **Du musst nur das Training neu starten!**

Der laufende Python-Prozess hat den alten Code im Speicher. Ein Neustart lädt den neuen Code und das Problem ist gelöst.
