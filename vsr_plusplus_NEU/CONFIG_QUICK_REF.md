## QUICK REFERENCE: config_p4_optimized.py

### Schnellübersicht / Quick Overview

#### ✅ Aktualisiert für 7-Frame Training
- **72 Features** (statt 64)
- **26 Blocks** (statt 24)
- **Batch=1** (VRAM-sicher)
- **Pfade passen** zu dataset_generator_v2

---

### Verwendung / Usage

```bash
# Konfiguration anzeigen
python3 vsr_plusplus_NEU/config_p4_optimized.py

# In Python importieren
from vsr_plusplus_NEU import config_p4_optimized as cfg
config = cfg.get_config()
```

---

### Key-Parameter / Key Parameters

| Parameter | Wert | Info |
|-----------|------|------|
| N_FEATS | 72 | 7-Frame optimiert |
| N_BLOCKS | 26 | Höhere Qualität |
| BATCH_SIZE | 1 | VRAM-sicher |
| ACCUMULATION_STEPS | 6 | Eff. Batch=6 |
| VRAM | ~3.77 GB | < 6.5 GB ✅ |

---

### Dataset-Pfade / Paths

```python
DATA_ROOT = "/mnt/data/training/datasetNeu/master"
DATASET_ROOT = "/mnt/data/training/datasetNeu"
```

**Struktur:**
```
datasetNeu/master/train/7frames/
├── small_540/     # 540×540
├── medium_169/    # 720×405
└── large_720/     # 720×720
```

---

### Kompatibel mit / Compatible with

✅ dataset_generator_v2  
✅ 7-Frame VSR System  
✅ Adaptive Batch Management  
✅ Runtime Configuration  
✅ Size Tracking  

---

### Dokumentation / Documentation

📖 **CONFIG_UPDATE_7FRAME.md** - Vollständige Dokumentation
📖 **README_7FRAME.md** - 7-Frame System Guide

---

### Verifikation / Verification

```python
assert cfg.N_FEATS == 72
assert cfg.N_BLOCKS == 26
assert cfg.BATCH_SIZE == 1
assert cfg.DATA_ROOT == "/mnt/data/training/datasetNeu/master"
```

---

**Status:** ✅ READY TO USE
