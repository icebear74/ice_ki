# Bug Fixes Summary - Dataset File Monitoring

## Issues Fixed

### 1. ✅ KeyError: 'has_new'
**Problem**: 
```python
KeyError: 'has_new'
  File "/mnt/data/ice_ki/vsr_plusplus_NEU/training/trainer.py", line 916
    'has_new': val_changes['has_new'],
```

**Root Cause**: 
- `dataset.py` returned `'has_changes'` 
- `trainer.py` expected `'has_new'`
- Key name mismatch

**Fix**:
Changed `dataset.py` line 234 and 241 from `'has_changes'` to `'has_new'`

### 2. ✅ Web UI zeigt keine Dateianzahl
**Problem**: 
Dataset file counts not showing in Web UI

**Root Cause**:
- Template fetched from `/api/data`
- Web UI only served `/monitoring/data`
- Endpoint mismatch

**Fix**:
Changed `monitor.html` line 518 from:
```javascript
const response = await fetch('/api/data');
```
to:
```javascript
const response = await fetch('/monitoring/data');
```

### 3. ✅ Config-Seite zeigt nur JSON
**Problem**:
Clicking "config" shows only JSON, not configuration interface

**Root Cause**:
- `/config` route only served JSON (`_deliver_config_json()`)
- `config_7frame.html` template existed but was never used
- No route to serve the HTML interface

**Fix**:
Added new route `/config/ui` in `web_ui.py`:
1. Added route handler in `do_GET()` method
2. Created `_deliver_config_page()` method to serve the HTML template
3. Added navigation links in both templates (monitor.html and config_7frame.html)

## Files Modified

1. **vsr_plusplus_NEU/core/dataset.py**
   - Changed 'has_changes' to 'has_new' (2 occurrences)
   - Updated docstring

2. **vsr_plusplus_NEU/web/templates/monitor.html**
   - Fixed API endpoint: `/api/data` → `/monitoring/data`
   - Added navigation links to config page

3. **vsr_plusplus_NEU/systems/web_ui.py**
   - Added `/config/ui` route
   - Created `_deliver_config_page()` method

4. **vsr_plusplus_NEU/web/templates/config_7frame.html**
   - Added navigation links back to monitor

5. **test_dataset_file_monitoring.py**
   - Updated test to expect 'has_new' instead of 'has_changes'

## How to Access

- **Monitor**: http://localhost:5050/monitoring
- **Config UI**: http://localhost:5050/config/ui
- **Config JSON**: http://localhost:5050/config (or /monitoring/config)

## Testing

All tests passing:
```
✅ Dataset methods test PASSED
✅ Trainer method test PASSED
✅ Web UI data store test PASSED
✅ Web UI template test PASSED
✅ Training initialization test PASSED
```

## Expected Behavior Now

1. **Training Startup**: No more KeyError, file monitoring initializes cleanly
2. **Web UI Monitor**: Dataset file counts display correctly per size (720, 540, 720_169)
3. **Config Interface**: Clicking config link shows proper UI, not just JSON
4. **Navigation**: Easy switching between Monitor and Config pages
