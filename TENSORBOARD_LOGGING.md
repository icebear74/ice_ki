# TensorBoard Comprehensive Logging - Dokumentation

## Übersicht

Alle wichtigen Trainingsdaten werden jetzt in TensorBoard geloggt, einschließlich:
- Runtime-Konfigurationsänderungen
- Plateau-Detection-Details
- Loss-Weight-Statistiken
- Validation-Events
- Training-Phasen-Übergänge

## Neue TensorBoard-Kategorien

### 1. Config (Konfiguration)

#### Config/Changes
**Typ:** Text  
**Inhalt:** Alle Runtime-Konfigurationsänderungen mit alten → neuen Werten  
**Beispiel:**
```
**plateau_patience**: 250 → 350
**max_lr**: 1.50e-04 → 2.00e-04
```

#### Config/Parameters/*
**Typ:** Scalar  
**Inhalt:** Aktuelle Werte aller konfigurierbaren Parameter  
**Verfügbare Scalars:**
- `Config/Parameters/plateau_patience`
- `Config/Parameters/plateau_safety_threshold`
- `Config/Parameters/max_lr`
- `Config/Parameters/min_lr`
- `Config/Parameters/initial_grad_clip`
- `Config/Parameters/cooldown_duration`

#### Config/Initial_Configuration
**Typ:** Text (Markdown)  
**Inhalt:** Vollständige Konfiguration beim Training-Start  
**Enthält:**
- Model Architecture (n_feats, n_blocks, batch_size)
- Training Parameters (max_lr, warmup, validation intervals)
- Adaptive System (plateau thresholds, cooldown)
- Loss Weights (L1, MS, Gradient, Perceptual)
- Data Settings

### 2. Plateau (Plateau-Detection)

#### Plateau/Counter
**Typ:** Scalar  
**Inhalt:** Aktueller Plateau-Zähler  
**Range:** 0 bis plateau_patience

#### Plateau/Patience
**Typ:** Scalar  
**Inhalt:** Konfigurierter Patience-Wert  
**Info:** Threshold für Plateau-Reset

#### Plateau/Progress_Percent
**Typ:** Scalar  
**Inhalt:** Fortschritt zum Plateau-Reset in %  
**Berechnung:** (counter / patience) * 100

#### Plateau/EMA_Loss
**Typ:** Scalar  
**Inhalt:** Exponential Moving Average des Loss  
**Info:** Geglättet mit alpha=0.1

#### Plateau/Best_Loss
**Typ:** Scalar  
**Inhalt:** Bester Loss-Wert bisher erreicht

#### Plateau/Best_Quality
**Typ:** Scalar  
**Inhalt:** Beste KI-Qualität bisher erreicht

#### Plateau/EMA_Quality
**Typ:** Scalar  
**Inhalt:** EMA der Qualitätsmetrik

#### Plateau/Is_Plateau
**Typ:** Scalar (0 oder 1)  
**Inhalt:** Plateau-Status (0=Nein, 1=Ja)

#### Plateau/Steps_Until_Reset
**Typ:** Scalar  
**Inhalt:** Verbleibende Steps bis potentiellem Reset

### 3. Weights (Loss-Gewichte)

#### Weights/Distribution
**Typ:** Multi-Scalar  
**Inhalt:** Loss-Weight-Verteilung in Prozent  
**Komponenten:**
- `L1_Percent`
- `MS_Percent`
- `Grad_Percent`
- `Perceptual_Percent`

#### Weights/Sum
**Typ:** Scalar  
**Inhalt:** Summe aller Gewichte  
**Sollwert:** ~1.0 (0.95-1.05 gültig)

#### Weights/Distribution_Histogram
**Typ:** Histogram  
**Inhalt:** Verteilung der Gewichte als Histogram

### 4. Events (Ereignisse)

#### Events/Timeline
**Typ:** Text  
**Inhalt:** Chronologische Event-Liste  
**Event-Typen:**
- LR_Boost
- Aggressive_Mode
- Cooldown_Start
- Config_Change
- Validation_Run

#### Events/Config_Change
**Typ:** Scalar (Marker)  
**Inhalt:** Spike (100) bei jeder Config-Änderung

#### Events/Validation_Run
**Typ:** Scalar (Marker)  
**Inhalt:** Spike (50) bei jeder Validation

#### Events/Validation
**Typ:** Text  
**Inhalt:** Detaillierte Validation-Metriken  
**Format:**
```
**Validation Run**
- KI Quality: 78.50%
- Improvement: 12.30%
- PSNR: 28.45dB
- SSIM: 0.8923
```

#### Events/Phase_Change
**Typ:** Scalar (Marker)  
**Inhalt:** Spike (75) bei Phasen-Übergängen

#### Events/Phase_Transitions
**Typ:** Text  
**Inhalt:** Phase-Wechsel-Beschreibungen

### 5. Training (bereits vorhanden, erweitert)

#### Training/Phase
**Typ:** Scalar  
**Inhalt:** Aktuelle Training-Phase als Zahl  
**Mapping:**
- 0 = Stable
- 1 = Aggressive
- 2 = Cooldown
- 3 = Plateau Reducing

## Verwendung

### TensorBoard Starten

```bash
# Standard
tensorboard --logdir=/path/to/Learn/active_run

# Mit bestimmtem Port
tensorboard --logdir=/path/to/Learn/active_run --port 6006

# Mehrere Runs vergleichen
tensorboard --logdir=/path/to/Learn
```

### Nützliche Views

**1. Config-Änderungen nachverfolgen:**
- Öffne `Config/Changes` (Text)
- Siehe alle Parameter-Änderungen chronologisch
- Vergleiche mit `Config/Parameters/*` (Scalars) für Werte-Verlauf

**2. Plateau-Detection analysieren:**
- `Plateau/Counter` vs `Plateau/Progress_Percent`
- Vergleiche `Plateau/EMA_Loss` mit `Loss/Total`
- Siehe `Plateau/Is_Plateau` für Reset-Punkte

**3. Weight-Balance überwachen:**
- `Weights/Distribution` zeigt prozentuale Aufteilung
- `Weights/Sum` sollte nahe 1.0 sein
- Histogram zeigt Verteilungs-Entwicklung

**4. Event-Timeline ansehen:**
- `Events/Timeline` für chronologische Übersicht
- `Events/*` Marker als visuelle Spikes im Graph
- Korreliere Events mit Loss/Quality-Änderungen

**5. Training-Phasen tracken:**
- `Training/Phase` zeigt aktuelle Phase
- Vergleiche mit `Adaptive/AggressiveMode`
- Korreliere mit Loss-Änderungen

## Dashboard-Vorschläge

### Dashboard 1: Configuration Tracking
```
- Config/Parameters/plateau_patience
- Config/Parameters/max_lr
- Config/Parameters/min_lr
- Events/Config_Change (Marker)
```

### Dashboard 2: Plateau Analysis
```
- Plateau/Counter
- Plateau/Progress_Percent
- Plateau/EMA_Loss
- Loss/Total
- Plateau/Is_Plateau (Marker)
```

### Dashboard 3: Weight Balance
```
- Weights/Distribution (Multi-line)
- Weights/Sum
- Loss/L1, Loss/MS, Loss/Grad, Loss/Perceptual
```

### Dashboard 4: Training Flow
```
- Training/Phase
- Events/Timeline
- Adaptive/AggressiveMode
- Adaptive/PlateauCounter
```

### Dashboard 5: Quality & Validation
```
- Quality/KI_Quality
- Quality/Improvement
- Events/Validation_Run (Marker)
- Validation/Loss_Total
```

## Best Practices

### 1. Regelmäßig Config-Änderungen prüfen
Vor dem Analysieren von Metriken, prüfe `Config/Changes` um zu sehen, ob Parameter geändert wurden.

### 2. Plateau-Counter überwachen
Wenn `Plateau/Progress_Percent` über 80% steigt, könnte bald ein Reset erfolgen.

### 3. Weight-Sum validieren
`Weights/Sum` sollte immer zwischen 0.95 und 1.05 sein. Abweichungen deuten auf Fehler hin.

### 4. Events mit Metriken korrelieren
Nutze die Event-Marker um zu sehen, wie Config-Änderungen die Performance beeinflussen.

### 5. Phasen-Übergänge analysieren
Vergleiche Loss-Verhalten in verschiedenen Training-Phasen (Stable vs Aggressive).

## Troubleshooting

### Keine Daten in TensorBoard?
1. Prüfe ob `active_run` Ordner existiert
2. Prüfe ob Training läuft
3. Refresh TensorBoard (F5)

### Config/Changes leer?
- Erst nach der ersten Config-Änderung gefüllt
- Ändere einen Parameter zur Aktivierung

### Plateau-Metriken fehlen?
- Benötigt `get_plateau_info()` in adaptive_system.py
- Bei älteren Versionen nicht verfügbar

### Weights/Sum != 1.0?
- Normal zu Beginn (Initialisierung)
- Sollte sich nach ~100 Steps stabilisieren
- Prüfe adaptive_system Konfiguration

## API-Referenz

### TensorBoardLogger Methoden

```python
# Config Logging
tb_logger.log_config_snapshot(config)
tb_logger.log_config_change(step, param_name, old_value, new_value)

# Plateau Logging
tb_logger.log_plateau_state(step, plateau_info)

# Weight Logging
tb_logger.log_weight_statistics(step, weights)

# Event Logging
tb_logger.log_event(step, event_type, message)
tb_logger.log_validation_event(step, metrics)

# Phase Logging
tb_logger.log_training_phase(step, phase_info)

# Hyperparameter Logging
tb_logger.log_hyperparameters(hparam_dict, metric_dict)
```

### Erforderliche Daten-Formate

**plateau_info:**
```python
{
    'plateau_counter': int,
    'plateau_patience': int,
    'best_loss': float,
    'ema_loss': float,
    'best_quality': float,
    'ema_quality': float,
    'is_plateau': bool,
    'steps_until_reset': int
}
```

**weights:**
```python
{
    'l1': float,
    'ms': float,
    'grad': float,
    'perceptual': float
}
```

**phase_info:**
```python
{
    'phase': str,  # 'stable', 'aggressive', 'cooldown', 'plateau_reducing'
    'phase_changed': bool
}
```

## Siehe auch

- [RUNTIME_CONFIG.md](RUNTIME_CONFIG.md) - Runtime-Konfiguration
- [WEB_UI_VISUALIZATIONS.md](WEB_UI_VISUALIZATIONS.md) - Web-UI Features
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementierungs-Details
