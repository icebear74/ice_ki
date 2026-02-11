# Video Category Manager

Ein interaktives Tool zum einfachen Verwalten von Video-Kategorie-Zuordnungen für den Dataset Generator.

## Problem

Die `generator_config.json` hat 3913 Zeilen mit 466 Videos. Jedes Video muss manuell Kategorien zugewiesen werden. Das ist:
- Unübersichtlich
- Fehleranfällig
- Schwer zu editieren

## Lösung

Ein interaktives CLI-Tool mit benutzerfreundlichem Menü zum Verwalten der Zuordnungen.

## Features

✅ **Liste alle Videos** - Mit/ohne Kategorien anzeigen  
✅ **Suche** - Videos nach Namen filtern (Regex)  
✅ **Einzelzuweisung** - Einzelne Videos zu Kategorien zuweisen  
✅ **Multi-Select** - Ganze Serien auf einmal zuweisen  
✅ **Reset** - Alle Zuordnungen zurücksetzen  
✅ **Statistiken** - Überblick über aktuelle Zuordnungen  
✅ **Kategorie-Targets** - Extraktionsziele bearbeiten  
✅ **Backup** - Automatisches Backup vor dem Speichern  

## Verwendung

```bash
cd dataset_generator_v2
python3 video_manager.py
```

## Menü-Optionen

```
1. List all videos          - Alle Videos anzeigen
2. List videos by category  - Videos nach Kategorie filtern
3. List unassigned videos   - Nur Videos ohne Kategorien
4. Search videos by name    - Nach Namen suchen (Regex)
5. Assign video(s)          - Einzelne Videos zuweisen
6. Multi-assign by pattern  - Mehrere Videos auf einmal
7. Remove from category     - Aus Kategorie entfernen
8. Reset all assignments    - ALLES zurücksetzen
9. Show statistics          - Statistiken anzeigen
10. Edit category targets   - Extraktionsziele ändern
s. Save changes             - Speichern
q. Quit                     - Beenden
```

## Beispiel-Workflows

### 1. Alle Videos zurücksetzen

```
Choice: 8
⚠️  Reset ALL video assignments? This cannot be undone! (yes/no): yes
✓ Reset 466 videos

Choice: s
✓ Backup saved to generator_config.json.backup
✓ Saved to generator_config.json
```

### 2. Eine Serie zuweisen (z.B. Star Trek)

```
Choice: 6
Search pattern (regex, e.g., 'Star Trek.*'): Star Trek

ID     Name                                               Categories
----------------------------------------------------------------------------------------------------
12     Star Trek - Der Film                               <unassigned>
13     Star Trek 2 - Der Zorn des Khan                    <unassigned>
14     Star Trek 3 - Auf der Suche nach Mr. Spock         <unassigned>
...

Assign all 12 videos? (y/n): y

Available categories: master, space, toon, universal
Enter weights for each category (0 to skip):
  master: 0.2
  space: 0.8
  toon: 0
  universal: 0

Normalized weights:
  master: 0.20
  space: 0.80

✓ Assigned 12 videos to categories: {'master': 0.2, 'space': 0.8}
```

### 3. Einzelnes Video zuweisen

```
Choice: 4
Search pattern (regex): Avatar

ID     Name                                               Categories
----------------------------------------------------------------------------------------------------
42     Avatar                                             <unassigned>

Choice: 5
Video ID(s) (comma-separated): 42

Available categories: master, space, toon, universal
Enter weights for each category (0 to skip):
  master: 0.3
  space: 0.3
  toon: 0.2
  universal: 0.2

Normalized weights:
  master: 0.30
  space: 0.30
  toon: 0.20
  universal: 0.20

✓ Assigned 1 videos to categories: {'master': 0.3, 'space': 0.3, 'toon': 0.2, 'universal': 0.2}
```

### 4. Mehrere Videos auf einmal zuweisen

```
Choice: 5
Video ID(s) (comma-separated): 10, 15, 23, 45

Available categories: master, space, toon, universal
Enter weights for each category (0 to skip):
  master: 0.25
  space: 0
  toon: 0
  universal: 0.75

✓ Assigned 4 videos to categories: {'master': 0.25, 'universal': 0.75}
```

### 5. Statistiken anzeigen

```
Choice: 9

============================================================
STATISTICS
============================================================

Total videos: 466
Unassigned: 120

Category assignments:
  master         :  346 videos (target: 150000)
  space          :   84 videos (target: 60000)
  toon           :   34 videos (target: 50000)
  universal      :  280 videos (target: 50000)
```

### 6. Nach Muster filtern und anzeigen

```
Choice: 4
Search pattern (regex): ^Shrek

ID     Name                                               Categories
----------------------------------------------------------------------------------------------------
156    Shrek                                              master:0.20, toon:0.80
157    Shrek 2                                            master:0.20, toon:0.80
158    Shrek 3                                            master:0.20, toon:0.80
159    Shrek 4                                            master:0.20, toon:0.80
```

## Kategorien

Aktuell verfügbare Kategorien:
- **master** - Hochwertige Master-Videos
- **universal** - Allgemeine Videos
- **space** - Weltraum/Sci-Fi
- **toon** - Animationsfilme

## Gewichtung

Jedes Video kann zu mehreren Kategorien gehören. Die Gewichte werden automatisch normalisiert:

```
Input:  master: 1.0, universal: 3.0
Result: master: 0.25, universal: 0.75
```

Das bedeutet: 25% der Extracts gehen nach "master", 75% nach "universal".

## Backup

Vor jedem Speichern wird automatisch ein Backup erstellt:
- `generator_config.json.backup`

## Regex-Patterns

Beispiele für Suchpatterns:

- `^Star Trek` - Beginnt mit "Star Trek"
- `Star Trek.*` - Enthält "Star Trek"
- `(?i)shrek` - Case-insensitive "Shrek"
- `(Shrek|Madagascar)` - Shrek ODER Madagascar
- `Season \d+` - Alle Seasons

## Tipps

1. **Immer erst suchen**: Verwenden Sie Option 4, um Videos zu finden
2. **Multi-Select für Serien**: Option 6 ist perfekt für ganze Serien
3. **Statistiken prüfen**: Option 9 zeigt, wie gut die Verteilung ist
4. **Regelmäßig speichern**: Option 's' speichert Ihre Änderungen
5. **Backup nutzen**: Falls was schiefgeht, `.backup` Datei umbenennen

## Integration

Das Tool modifiziert die `generator_config.json` direkt. Alle anderen Tools (dataset generator, etc.) funktionieren weiterhin normal.

## Erweiterte Features (Zukunft)

Mögliche Erweiterungen:
- Auto-Kategorisierung basierend auf Keywords
- Import/Export von Zuordnungen
- Bulk-Operations
- Undo/Redo
- WebUI statt CLI
