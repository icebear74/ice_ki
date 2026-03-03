# VAL Datenstruktur - Komplette Anleitung

## Aktuelle Dataset-Struktur (bestätigt)

```
/mnt/data/training/datasetNeu/
├── master/
│   └── patches/
│       ├── 540/
│       │   ├── GT/           (612M - Ground Truth 540×540)
│       │   └── LR_7frames/   (656M - 7 frames vertikal 1260×180)
│       ├── 720/
│       │   ├── GT/           (543M - Ground Truth 720×720)
│       │   └── LR_7frames/   (581M - 7 frames vertikal 1680×240)
│       └── 720_169/
│           ├── GT/           (608M - Ground Truth 405×720)
│           └── LR_7frames/   (653M - 7 frames vertikal 945×240)
│
├── universal/
│   └── patches/
│       ├── 540/              (2.6G)
│       ├── 720/              (2.4G)
│       └── 720_169/          (2.6G)
│
├── space/
│   └── patches/
│       ├── 540/              (315M)
│       ├── 720/              (281M)
│       └── 720_169/          (318M)
│
└── toon/
    └── patches/
        ├── 540/              (124M)
        └── 720_169/          (124M)
```

## Validation-Verzeichnisse erstellen

### Schritt 1: Verzeichnisse anlegen

```bash
# Für master category
mkdir -p /mnt/data/training/datasetNeu/master/val/540/GT
mkdir -p /mnt/data/training/datasetNeu/master/val/720/GT
mkdir -p /mnt/data/training/datasetNeu/master/val/720_169/GT

# Für universal category
mkdir -p /mnt/data/training/datasetNeu/universal/val/540/GT
mkdir -p /mnt/data/training/datasetNeu/universal/val/720/GT
mkdir -p /mnt/data/training/datasetNeu/universal/val/720_169/GT

# Für space category
mkdir -p /mnt/data/training/datasetNeu/space/val/540/GT
mkdir -p /mnt/data/training/datasetNeu/space/val/720/GT
mkdir -p /mnt/data/training/datasetNeu/space/val/720_169/GT

# Für toon category
mkdir -p /mnt/data/training/datasetNeu/toon/val/540/GT
mkdir -p /mnt/data/training/datasetNeu/toon/val/720_169/GT
```

### Schritt 2: GT-Bilder auswählen und kopieren

Wählen Sie gute, repräsentative Bilder aus den Training-Patches:

```bash
# Beispiel: 10 Validation-Bilder für master/540
cd /mnt/data/training/datasetNeu/master/patches/540/GT/

# Zufällig 10 Bilder auswählen
ls *.png | shuf -n 10 > /tmp/val_images_540.txt

# Diese kopieren nach val/540/GT/
while read img; do
    cp "$img" /mnt/data/training/datasetNeu/master/val/540/GT/
done < /tmp/val_images_540.txt
```

### Schritt 3: Verifikation

```bash
# Prüfen, dass GT-Bilder vorhanden sind
ls -lh /mnt/data/training/datasetNeu/master/val/540/GT/

# Prüfen, dass entsprechende LR-Bilder in patches existieren
cd /mnt/data/training/datasetNeu/master/val/540/GT/
for img in *.png; do
    if [ ! -f "../../../patches/540/LR_7frames/$img" ]; then
        echo "WARNUNG: Kein LR für $img gefunden!"
    fi
done
```

## Wie das Training funktioniert

### Für Training-Daten (mode='train'):
```python
dataset = VSRDataset(
    root="/mnt/data/training/datasetNeu",
    dataset_name="master",
    size_key="540",
    mode="train"
)
```
**Lädt:**
- GT von: `/mnt/data/training/datasetNeu/master/patches/540/GT/`
- LR von: `/mnt/data/training/datasetNeu/master/patches/540/LR_7frames/`

### Für Validation-Daten (mode='val'):
```python
val_dataset = VSRDataset(
    root="/mnt/data/training/datasetNeu",
    dataset_name="master",
    size_key="540",
    mode="val"
)
```
**Lädt:**
- GT von: `/mnt/data/training/datasetNeu/master/val/540/GT/`
- LR von: `/mnt/data/training/datasetNeu/master/patches/540/LR_7frames/` ← automatisch!

**Wichtig**: Sie müssen nur GT-Bilder kopieren. Die LR-Bilder werden automatisch gefunden!

## Finale Struktur nach Setup

```
/mnt/data/training/datasetNeu/master/
├── patches/
│   ├── 540/
│   │   ├── GT/           ← Training GT (viele Bilder)
│   │   └── LR_7frames/   ← Training + Validation LR (viele Bilder)
│   ├── 720/
│   │   ├── GT/
│   │   └── LR_7frames/
│   └── 720_169/
│       ├── GT/
│       └── LR_7frames/
│
└── val/                  ← NEU: Manuell erstellt (lowercase)
    └── GT/               ← Validation GT Verzeichnis
        ├── 540/          ← ~10-50 ausgewählte GT-Bilder für 540
        ├── 720/          ← ~10-50 ausgewählte GT-Bilder für 720
        └── 720_169/      ← ~10-50 ausgewählte GT-Bilder für 720_169
```

## Vollständiges Beispiel

### 1. Verzeichnisse erstellen
```bash
mkdir -p /mnt/data/training/datasetNeu/master/val/540/GT val/720/GT val/720_169/GT
```

### 2. Je 20 Validation-Bilder pro Format auswählen
```bash
# Für 540×540
cd /mnt/data/training/datasetNeu/master/patches/540/GT
ls *.png | shuf -n 20 | xargs -I {} cp {} ../../val/540/GT/

# Für 720×720
cd /mnt/data/training/datasetNeu/master/patches/720/GT
ls *.png | shuf -n 20 | xargs -I {} cp {} ../../val/720/GT/

# Für 720_169 (16:9)
cd /mnt/data/training/datasetNeu/master/patches/720_169/GT
ls *.png | shuf -n 20 | xargs -I {} cp {} ../../val/720_169/GT/
```

### 3. Überprüfung
```bash
# Anzahl der Validation-Bilder
echo "540: $(ls /mnt/data/training/datasetNeu/master/val/540/GT/*.png 2>/dev/null | wc -l)"
echo "720: $(ls /mnt/data/training/datasetNeu/master/val/720/GT/*.png 2>/dev/null | wc -l)"
echo "720_169: $(ls /mnt/data/training/datasetNeu/master/val/720_169/GT/*.png 2>/dev/null | wc -l)"

# Gesamtstruktur anzeigen
du -h -d 4 /mnt/data/training/datasetNeu/master/
```

## Training starten

Nach dem Setup können Sie das Training starten:

```bash
cd /home/runner/work/ice_ki/ice_ki/vsr_plusplus_NEU
python3 train.py
```

Das Training wird automatisch:
1. Training-Daten aus `patches/{size_key}/GT` und `patches/{size_key}/LR_7frames` laden
2. Validation-Daten aus `val/{size_key}/GT` laden
3. Entsprechende LR automatisch in `patches/{size_key}/LR_7frames` finden

## Fehlerbehebung

### "No PNG files found in .../val/540/GT"
```bash
# Verzeichnis existiert nicht oder ist leer
ls -la /mnt/data/training/datasetNeu/master/val/540/GT/

# Lösung: GT-Bilder kopieren (siehe oben)
```

### "No valid GT-LR pairs found"
```bash
# Die GT-Dateinamen in val/540/GT/ stimmen nicht mit denen in patches/540/LR_7frames/ überein
cd /mnt/data/training/datasetNeu/master/val/540/GT/
for img in *.png; do
    if [ ! -f "../../../patches/540/LR_7frames/$img" ]; then
        echo "Kein LR für: $img"
    fi
done

# Lösung: Nur Bilder kopieren, die auch LR-Versionen haben (siehe Schritt 2)
```

### Validation-Verzeichnis nicht gefunden
```bash
# Prüfen, ob lowercase 'val' verwendet wird (nicht 'Val')
ls -la /mnt/data/training/datasetNeu/master/ | grep val

# Sollte zeigen: drwxr-xr-x ... val
# NICHT:        drwxr-xr-x ... Val
```

## Empfohlene Anzahl Validation-Bilder

- **Minimum**: 10 Bilder pro size_key
- **Empfohlen**: 20-50 Bilder pro size_key
- **Maximum**: Nicht mehr als 5% der Training-Daten

Beispiel für master (3.6G Training-Daten):
- 540: 20-30 Bilder
- 720: 20-30 Bilder
- 720_169: 20-30 Bilder
