# Config Loading - Wichtige Information

## Problem gelöst: config.py wird jetzt korrekt geladen

### Das Problem
- `train.py` importierte von `vsr_plus_plus.config` (alter Pfad)
- Die lokale `config.py` in `vsr_plusplus_NEU/` wurde NICHT geladen
- Die `config.py` ist absichtlich in `.gitignore` (lokale Konfiguration)

### Die Lösung
1. Import geändert von `import vsr_plus_plus.config as cfg` zu `import config as cfg`
2. Aktuelles Verzeichnis zu sys.path hinzugefügt: `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`

### Verwendung
1. Erstellen Sie Ihre lokale `config.py` in `vsr_plusplus_NEU/`:
   ```bash
   cp config_p4_optimized.py config.py
   # oder
   cp config.py.active config.py
   ```

2. Bearbeiten Sie die config.py nach Ihren Wünschen

3. Die config.py wird NICHT ins Repo gepusht (ist in .gitignore)

4. `train.py` lädt automatisch Ihre lokale config.py

### Überprüfung
```bash
cd vsr_plusplus_NEU
python3 -c "import config as cfg; print(f'N_FEATS={cfg.N_FEATS}, N_BLOCKS={cfg.N_BLOCKS}')"
```

### Wichtig
- Die `config.py` bleibt lokal (nicht im Repo)
- Jeder Entwickler kann seine eigene config.py haben
- `config_p4_optimized.py` ist die empfohlene Vorlage (im Repo)
