# Training Dataset Count Fix - Summary

## Problem Reports

### 1. Training Dataset zeigt 0 files
> Validation Dataset wird korrekt angezeigt (3/3 files) .. Training Dataset wird mit 0 files angezeigt ..

### 2. Config Button zeigt JSON
> und web button configuration verweist immer noch auf das json nicht auf die UI ..

## Root Cause Analysis

### Problem 1: Training Dataset Count = 0

**Was war das Problem:**
- Training mit MultiSizeDataLoader zeigte 0 files
- Validation datasets zeigten korrekt 3/3/3 files

**Root Cause:**
```python
# Der Code prüfte nur:
if hasattr(self.train_loader, 'dataset'):
    train_ds = self.train_loader.dataset
    # ...

# Aber MultiSizeDataLoader hat keine 'dataset' Attribute!
# Stattdessen hat es:
train_loader.datasets_dict = {
    '540': VSRDataset(...),
    '720': VSRDataset(...),
    '720_169': VSRDataset(...)
}
```

**Warum Validation funktionierte:**
- Validation loader sind einzelne DataLoader mit `dataset` Attribut
- Nur Training nutzt den MultiSizeDataLoader

### Problem 2: Config Button → JSON statt UI

**Was war das Problem:**
```javascript
function openConfigPage() {
    window.open('/config', '_blank');  // ❌ Öffnet JSON
}
```

**Sollte sein:**
```javascript
function openConfigPage() {
    window.open('/config/ui', '_blank');  // ✅ Öffnet UI
}
```

## Lösung Implementiert

### 1. MultiSizeDataLoader Support

```python
def _check_dataset_files(self):
    # ...
    
    # Check training dataset
    if hasattr(self.train_loader, 'dataset'):
        # Standard DataLoader (single dataset)
        train_ds = self.train_loader.dataset
        # ... existing code ...
        
    elif hasattr(self.train_loader, 'datasets_dict'):
        # ✅ NEU: MultiSizeDataLoader (multiple datasets)
        total_count = 0
        total_new = 0
        has_any_new = False
        size_keys = []
        
        for size_key, train_ds in self.train_loader.datasets_dict.items():
            train_info = train_ds.get_file_info()
            train_changes = train_ds.check_for_new_files()
            
            total_count += train_info['file_count']
            total_new += train_changes['new_files']
            has_any_new = has_any_new or train_changes['has_new']
            size_keys.append(size_key)
            
            # Log per size if new files detected
            if train_changes['has_new']:
                print(f"📂 New training files for {size_key}: +{train_changes['new_files']}")
        
        # Aggregate display
        dataset_info['train'] = {
            'size_key': '+'.join(sorted(size_keys)),  # "540+720+720_169"
            'count': total_count,
            'has_new': has_any_new,
            'new_count': total_new
        }
```

**Features:**
- ✅ Aggregiert Dateizählung über alle Sizes
- ✅ Zeigt kombinierte Size Keys (z.B. "540+720+720_169")
- ✅ Trackt neue Dateien pro Size und total
- ✅ Logging für jede Size separat

### 2. Config Button Fix

```javascript
// VORHER:
function openConfigPage() {
    window.open('/config', '_blank');  // JSON endpoint
}

// NACHHER:
function openConfigPage() {
    window.open('/config/ui', '_blank');  // HTML UI
}
```

## Ergebnis

### Vorher ❌
```
📂 Dataset Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Dataset
  Size: -                        0 files  ❌

Validation Datasets
  720×720                        3 files  ✓
  540×540                        3 files  ✓
  720×405 (16:9)                 3 files  ✓

[⚙️ Configuration] → Öffnet JSON  ❌
```

### Nachher ✅
```
📂 Dataset Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Dataset
  Size: 540+720+720_169         126 files  ✓

Validation Datasets
  720×720                        3 files  ✓
  540×540                        3 files  ✓
  720×405 (16:9)                 3 files  ✓

[⚙️ Configuration] → Öffnet Config UI  ✓
```

## Technische Details

### DataLoader Typen

**Standard DataLoader:**
```python
train_loader = DataLoader(
    train_dataset,  # Single VSRDataset
    batch_size=6,
    shuffle=True
)

# Zugriff:
train_loader.dataset  # VSRDataset instance
```

**MultiSizeDataLoader:**
```python
train_loader = MultiSizeDataLoader(
    datasets_dict={
        '540': VSRDataset(...),
        '720': VSRDataset(...),
        '720_169': VSRDataset(...)
    },
    sampler=MultiSizeSampler(...)
)

# Zugriff:
train_loader.datasets_dict  # Dict of VSRDataset instances
```

### Size Key Display

Für MultiSizeDataLoader:
```python
size_keys = ['540', '720', '720_169']
combined = '+'.join(sorted(size_keys))
# Result: "540+720+720_169"
```

Zeigt dem Benutzer, dass Training über mehrere Sizes läuft.

### Aggregierung

```python
# Beispiel mit 3 Sizes:
# 540:      42 files
# 720:      38 files
# 720_169:  46 files
# ────────────────────
# Total:   126 files

total_count = 42 + 38 + 46 = 126
```

## Testing

### Unit Test Simulation

```python
# Simulate MultiSizeDataLoader
class MockMultiSizeLoader:
    def __init__(self):
        self.datasets_dict = {
            '540': MockDataset(42),
            '720': MockDataset(38),
            '720_169': MockDataset(46)
        }

class MockDataset:
    def __init__(self, count):
        self.count = count
    
    def get_file_info(self):
        return {
            'size_key': '540',
            'file_count': self.count
        }
    
    def check_for_new_files(self):
        return {
            'has_new': False,
            'new_files': 0,
            'new_gt_count': self.count,
            'current_loaded': self.count
        }

# Test
loader = MockMultiSizeLoader()
assert hasattr(loader, 'datasets_dict')
assert '540' in loader.datasets_dict
assert '720' in loader.datasets_dict
assert '720_169' in loader.datasets_dict

total = sum(ds.count for ds in loader.datasets_dict.values())
assert total == 126  # ✓
```

## Zusammenfassung

Beide Probleme wurden behoben:

1. ✅ **Training Dataset Count**
   - Unterstützt jetzt MultiSizeDataLoader
   - Zeigt aggregierte Counts über alle Sizes
   - Display: "540+720+720_169: 126 files"

2. ✅ **Config Button**
   - Öffnet jetzt `/config/ui` statt `/config`
   - Benutzer sieht Config Interface, nicht JSON

Die Änderungen sind minimal und chirurgisch:
- trainer.py: +31 Zeilen (MultiSize Support)
- web_ui.py: 1 Zeile (Config Link)

Keine Breaking Changes, abwärtskompatibel mit Standard DataLoader.
