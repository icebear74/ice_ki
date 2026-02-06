# Web UI Improvements - Adaptive System Fields & JSON Download

## Overview
Added missing adaptive system status fields to the web monitoring interface and implemented a JSON data export feature.

---

## New Adaptive System Fields

### 1. Plateau Counter
**Display:** Shows current plateau count with color-coded warnings

**Color Coding:**
- 🟢 **Green** (< 150 steps): "Normal" - Training is progressing
- 🟡 **Yellow** (150-300 steps): "Erhöht" - Slight plateau detected
- 🚨 **Red** (> 300 steps): "WARNUNG" - Critical plateau, triggers LR boost

**HTML:**
```html
<div class="info-card">
    <div class="card-title">Plateau Counter</div>
    <div class="card-value" id="plateauCounter">0</div>
    <div class="card-subtitle" id="plateauWarning"></div>
</div>
```

**JavaScript:**
```javascript
const plateauCounter = data.adaptive_plateau_counter || 0;
if (plateauCounter > 300) {
    plateauEl.style.color = 'var(--accent-red)';
    plateauWarning.textContent = '🚨 WARNUNG';
}
```

---

### 2. LR Boost Status
**Display:** Shows whether the learning rate boost is available

**States:**
- ⚡ **Bereit** (Green): LR boost is available and can be triggered
- ⏳ **Cooldown** (Orange): Boost recently used, waiting 1000 steps

**HTML:**
```html
<div class="info-card">
    <div class="card-title">LR Boost</div>
    <div class="card-value" id="lrBoostStatus">Bereit</div>
</div>
```

**JavaScript:**
```javascript
if (data.adaptive_lr_boost_available) {
    lrBoostStatus.textContent = '⚡ Bereit';
    lrBoostStatus.style.color = 'var(--accent-green)';
} else {
    lrBoostStatus.textContent = '⏳ Cooldown';
    lrBoostStatus.style.color = 'var(--accent-orange)';
}
```

---

### 3. Perceptual Weight with Trend
**Display:** Shows current perceptual weight percentage with trend indicator

**Trend Indicators:**
- ⬆️ **Steigend** (Green): Perceptual weight increasing (trend > 0.001)
- ➡️ **Stabil** (Gray): Perceptual weight stable (-0.001 to 0.001)
- ⬇️ **Fallend** (Orange): Perceptual weight decreasing (trend < -0.001)

**HTML:**
```html
<div class="info-card">
    <div class="card-title">Perceptual Weight</div>
    <div class="card-value" id="perceptualWeightDisplay">5.0%</div>
    <div class="card-subtitle" id="perceptualTrend"></div>
</div>
```

**JavaScript:**
```javascript
const percWeight = (data.perceptual_weight_current * 100).toFixed(1);
percWeightDisplay.textContent = percWeight + '%';

const trend = data.adaptive_perceptual_trend || 0;
if (trend > 0.001) {
    percTrend.textContent = '⬆️ Steigend';
    percTrend.style.color = 'var(--accent-green)';
}
```

---

## JSON Download Feature

### New Control Buttons Section
Added to the header section for easy access:

```html
<div class="control-buttons">
    <button class="btn btn-primary" onclick="downloadDataAsJSON()">
        📥 Download Data (JSON)
    </button>
    <button class="btn btn-success" onclick="requestValidation()">
        🔍 Run Validation
    </button>
</div>
```

### Download Function
**Function:** `downloadDataAsJSON()`

**Features:**
- Fetches complete training data from `/monitoring/data` endpoint
- Creates timestamped filename: `vsr_training_data_YYYY-MM-DD-HHMMSS.json`
- Pretty-formats JSON with 2-space indentation
- Triggers browser download using Blob URL
- Automatic cleanup of temporary URLs

**Implementation:**
```javascript
function downloadDataAsJSON() {
    fetch('/monitoring/data')
        .then(response => response.json())
        .then(data => {
            const now = new Date();
            const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
            const filename = `vsr_training_data_${timestamp}.json`;
            
            const jsonStr = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
}
```

---

## CSS Enhancements

### Button Styles
Added modern button styles with hover effects:

```css
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    font-size: 0.95em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary {
    background: var(--accent-blue);
    color: #000;
}

.btn-primary:hover {
    background: #79c0ff;
    transform: translateY(-2px);
}
```

---

## Data Fields Included in JSON Export

The downloaded JSON includes all training data:

### Basic Metrics
- `step_current`, `step_max`
- `epoch_num`, `epoch_step_current`, `epoch_step_total`
- `learning_rate_value`, `lr_phase_name`

### Loss Values
- `total_loss_value`, `l1_loss_value`
- `ms_loss_value`, `gradient_loss_value`
- `perceptual_loss_value`

### Adaptive Weights
- `l1_weight_current`, `ms_weight_current`
- `gradient_weight_current`, `perceptual_weight_current`
- `gradient_clip_val`

### Adaptive Status (NEW!)
- `adaptive_mode` (Stable/Aggressive)
- `adaptive_is_cooldown` (boolean)
- `adaptive_cooldown_remaining` (steps)
- `adaptive_plateau_counter` (steps) ✨ NEW
- `adaptive_lr_boost_available` (boolean) ✨ NEW
- `adaptive_perceptual_trend` (float) ✨ NEW

### Quality Metrics
- `quality_lr_value`, `quality_ki_value`
- `quality_improvement_value`
- `quality_ki_to_gt_value`, `quality_lr_to_gt_value`
- `validation_loss_value`, `best_quality_ever`

### Performance
- `iteration_duration` (seconds)
- `vram_usage_gb`
- `adam_momentum_avg`

### Layer Activity
- `layer_activity_map` (dict of layer_name: percentage)

### Status
- `training_active`, `validation_running`, `training_paused`
- `last_update_time` (Unix timestamp)

---

## Usage Example

### Viewing Adaptive System Status
1. Open web monitor: `http://localhost:5050/monitoring`
2. Scroll to "🎚️ Adaptive System Status" section
3. Observe:
   - **Modus**: Current mode (Stable/Aggressive)
   - **Cooldown**: Whether adjustments are paused
   - **Plateau Counter**: How long training has been stuck
   - **LR Boost**: Whether recovery boost is ready
   - **Perceptual Weight**: Current value and trend

### Downloading Training Data
1. Click "📥 Download Data (JSON)" button in header
2. File is automatically saved with timestamp
3. Open JSON file to analyze training state
4. Use for debugging, analysis, or archiving

### Example Downloaded Filename
```
vsr_training_data_2026-02-06T082630.json
```

### Example JSON Content (excerpt)
```json
{
  "step_current": 7542,
  "epoch_num": 8,
  "learning_rate_value": 0.00003456,
  "adaptive_mode": "Stable",
  "adaptive_is_cooldown": false,
  "adaptive_plateau_counter": 234,
  "adaptive_lr_boost_available": true,
  "adaptive_perceptual_trend": 0.0015,
  "perceptual_weight_current": 0.087,
  "quality_ki_value": 0.621,
  "quality_improvement_value": 0.189,
  ...
}
```

---

## Benefits

### For Users
1. **Better Visibility**: All adaptive system metrics now visible in web UI
2. **Real-time Monitoring**: See plateau warnings and boost status live
3. **Data Export**: Download complete training state for analysis
4. **Trend Tracking**: Perceptual weight trend helps understand behavior

### For Debugging
1. **Plateau Detection**: Immediately see when training gets stuck
2. **Boost Verification**: Confirm LR boost is working correctly
3. **State Snapshots**: Export data at any point for offline analysis
4. **Correlation Analysis**: Compare metrics across different runs

### For Documentation
1. **Training Records**: Keep JSON snapshots of training milestones
2. **Issue Reporting**: Include full state when reporting problems
3. **Performance Comparison**: Compare different training configurations
4. **Research**: Archive data for papers or analysis

---

## Testing Checklist

- [x] Plateau Counter displays correctly
- [x] Color changes at 150 and 300 steps
- [x] LR Boost status updates
- [x] Perceptual weight shows percentage
- [x] Trend indicator changes based on value
- [x] Download button appears in header
- [x] Download creates properly named file
- [x] JSON is properly formatted
- [x] All fields included in download
- [x] No JavaScript errors

---

## Summary

Successfully implemented:
1. ✅ **3 New Adaptive System Cards** in web UI
2. ✅ **Color-coded Status Indicators** for plateau/boost/trend
3. ✅ **JSON Download Button** with timestamped files
4. ✅ **Complete Data Export** including all new fields
5. ✅ **Modern UI Controls** with hover effects

The web monitoring interface now provides complete visibility into the adaptive training system and allows easy data export for analysis and debugging.
