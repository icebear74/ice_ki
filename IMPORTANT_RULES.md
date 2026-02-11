# WICHTIGE REGELN FÜR ÄNDERUNGEN

## ⚠️ IMMUTABLE DIRECTORIES

### vsr_plus_plus/
**NIEMALS ÄNDERN!** Dieses Verzeichnis ist IMMUTABLE (unveränderlich).

- ❌ KEINE Änderungen an Code
- ❌ KEINE Änderungen an Templates
- ❌ KEINE Löschungen
- ❌ KEINE neuen Features

### vsr_plusplus_NEU/
**NUR HIER ÄNDERUNGEN!** Alle Entwicklung findet hier statt.

- ✅ Code-Änderungen erlaubt
- ✅ Template-Änderungen erlaubt
- ✅ Neue Features hinzufügen
- ✅ Bugfixes implementieren

## Warum?

`vsr_plus_plus` ist die stabile Basis-Version, die nicht verändert werden soll.
`vsr_plusplus_NEU` ist die aktive Entwicklungsversion für neue Features.

## Bei versehentlichen Änderungen:

```bash
# Änderungen an vsr_plus_plus rückgängig machen:
git checkout HEAD~1 -- vsr_plus_plus/
```
