# Schnellreferenz - Welche Config verwenden?

## 🚀 Schnellstart

### Für PRODUKTION (Empfohlen):
```bash
cd dataset_generator_v2
python make_dataset_v2_uhd.py generator_config.json
```
**→ 467 Videos, 4 Kategorien, Prioritäten, GUI**

### Für NEUE PROJEKTE:
```bash
cd dataset_generator_v2
python make_dataset_v2_clean.py generator_config_v2.json
```
**→ Auto-Scan, 2 Kategorien, Vereinfacht**

---

## 📁 Config-Dateien

| Datei | Script | Features |
|-------|--------|----------|
| `generator_config.json` | make_dataset_v2_uhd.py | 467 Videos, 4 Kategorien ⭐ |
| `generator_config_v2.json` | make_dataset_v2_clean.py | Auto-Scan, 2 Kategorien |

---

## ❌ Häufige Fehler

**Fehler:** `AttributeError: ... 'base_dir'`  
**Lösung:** ✅ BEHOBEN! Update auf neueste Version.

**Fehler:** `KeyError: 'videos'`  
**Lösung:** Falsche Config! Verwende `generator_config.json` mit `make_dataset_v2_uhd.py`

**Fehler:** `KeyError: 'base_settings'`  
**Lösung:** Falsche Config! Verwende `generator_config_v2.json` mit `make_dataset_v2_clean.py`

---

## 📖 Mehr Info

Siehe: `dataset_generator_v2/README_CONFIGS.md`
